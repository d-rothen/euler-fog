# GPU Render Pipeline Migration Plan

## Goal

Make CUDA mode keep the fog rendering and camera/capture pipeline on torch tensors for as much of the path as is practical. The CPU implementation remains the reference path and fallback. The migration should reduce device-to-host transfers during large dataset generation without weakening the physical camera model.

## Current State

- Fog rendering already has a torch path for CUDA: RGB normalization, radial depth conversion, atmospheric light estimation, heterogeneous `k`, heterogeneous `L_s`, `L_s` gradients, and fog compositing run on torch tensors.
- The capture stack is the main CPU bottleneck. `CaptureArtifactPipeline.apply_torch` currently converts the rendered torch image to NumPy, runs every capture stage through NumPy/PIL, then converts the final RGB image back to torch.
- Final dataset writes are necessarily CPU-backed today because the existing output backends write NumPy/PIL-compatible images and `.npy` maps.
- Exact JPEG roundtrip is PIL-based. This can be preserved as a CPU fallback or later replaced with a torch approximation, but exact codec behavior is not realistically CUDA-native in this codebase.

## Target Architecture

The capture stack should become stage-wise torch capable:

1. `CaptureArtifactPipeline.apply_torch` calls `stage.apply_torch` in sequence instead of converting the whole stack up front.
2. `ConfiguredCaptureStage.apply_torch` samples configuration/probability with the same NumPy RNG semantics as CPU, then calls a stage-specific `_apply_torch`.
3. Stages without a torch implementation fall back to the existing `_apply_np` implementation for correctness.
4. Torch-capable stages keep HWC float RGB tensors on the source device and preserve dtype/device on return.
5. The GPU path only performs an unavoidable CPU transfer at final output writing, auxiliary map writing, or explicit CPU-only artifacts such as exact JPEG.

## CPU Calls To Replace

High priority torch substitutions:

- Gaussian blur and low-frequency blur: `torch.nn.functional.pad` plus separable/grouped `conv2d`.
- Crop/resize: tensor slicing and `torch.nn.functional.interpolate`.
- Quantization, tone mapping, gamma, saturation, color matrices: elementwise torch ops.
- Vignetting, lens distortion, chromatic aberration: normalized coordinate grids and `torch.nn.functional.grid_sample`.
- Bayer mosaic and demosaic: tensor masks plus grouped convolution.
- Sensor noise: `torch.poisson`, `torch.randn`, torch-generated row/column/fixed pattern maps.
- Auto exposure metrics: torch luminance, quantiles, center weighting, fog-opacity metrics.

Lower priority or partial substitutions:

- Perlin windshield haze fields and droplet masks can be ported, but they are complex and currently less central than sensor/ISP/transport.
- Exact JPEG remains a CPU fallback initially. A later torch approximation can model block quantization/chroma subsampling if device residency matters more than codec exactness.
- Final image encoding and output backend writes remain CPU.

## Implementation Phases

### Phase 1: Stage-wise Torch Dispatch

- Add `ConfiguredCaptureStage.apply_torch` and `_apply_torch` hooks.
- Change `CaptureArtifactPipeline.apply_torch` and batch mode to run stages one at a time.
- Add torch image utility helpers for HWC validation, clipping, color matrices, gamma, blur, resize, crop, and quantization.
- Implement torch-native `ExposureStage`, `ISPStage`, and non-JPEG `TransportStage`.
- Fall back only when a stage uses unsupported CPU-only behavior, especially exact JPEG.

### Phase 2: Optics Stage

- Port lens distortion, chromatic aberration, depth/fog-weighted fringing, gaussian blur, bloom, vignetting, and simple veiling glare to torch.
- Keep droplets and windshield low-frequency haze as fallback until their random fields are ported.
- Use intrinsics directly in torch geometry helpers.

### Phase 3: Sensor Stage

- Port auto exposure metrics to torch.
- Port Bayer masks/mosaic/demosaic and sensor levels.
- Port shot/read/fixed/row/column noise with torch RNG.
- Port noise modulation, black suppression, and shadow-recovery chroma noise.
- Preserve the current qualitative noise behavior while improving CUDA throughput.

### Phase 4: Batch-Aware Capture

- Where configs are identical enough, add batched torch execution for deterministic stages.
- Keep per-sample dispatch for stages that sample independently per image.
- Avoid prematurely batching code that would make RNG order or profile selection hard to reason about.

### Phase 5: Optional Torch JPEG Approximation

- Add an opt-in `jpeg.mode: "approx_torch"` or similar setting.
- Approximate block artifacts, chroma subsampling, and quality-dependent quantization on torch.
- Keep exact PIL JPEG as the default when byte-level codec behavior matters.

## Acceptance Criteria

- CPU output remains behaviorally unchanged for existing tests.
- Torch capture path returns tensors on the original device except for documented CPU fallback stages.
- Non-JPEG exposure, ISP, crop/resize, and quantization can run without NumPy/PIL transfers.
- CUDA tests are skipped when CUDA is unavailable, but CPU-device torch tests validate shapes, ranges, and close parity for deterministic operations.
- The pipeline logs or tests make CPU fallback boundaries clear enough to avoid assuming full CUDA residency when JPEG or unsupported optics/sensor features are enabled.

## Initial Implementation Slice

Start with Phase 1. This is the lowest-risk change because it only changes torch dispatch and adds torch-native implementations for deterministic, post-sensor stages. Optics and sensor remain correct through fallback while subsequent phases replace them incrementally.

## Implementation Progress

- Phase 1 is implemented: capture dispatch is stage-wise for torch inputs, with torch-native exposure, ISP, non-JPEG transport, and reusable tensor utilities.
- Phase 2 is partially implemented: lens distortion, chromatic aberration, depth/fog-weighted fringing, gaussian blur, motion blur, bloom, veiling glare, vignetting, and windshield haze have torch paths. Droplet rendering remains a local CPU fallback when enabled.
- Phase 3 is partially implemented: auto exposure metrics, Bayer mosaic/demosaic, raw quantization, shot/read/fixed/row/column noise, bad pixels, noise modulation, black suppression, and shadow-recovery chroma noise have torch paths.
- Remaining CPU boundaries are exact JPEG roundtrip, final output encoding/writes, auxiliary `.npy` writes, and enabled droplet rendering.
- Remaining optimization work is batch-aware capture execution and an optional torch JPEG approximation.
