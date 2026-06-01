# euler-preprocess

Physics-based preprocessing transforms for multi-modal RGB+depth datasets. Built on top of [euler-loading](https://github.com/d-rothen/euler-loading) and [ds-crawler](https://github.com/d-rothen/ds-crawler).

Available transforms:

| Command | Description |
|---|---|
| `euler-preprocess fog` | Synthetic fog via the Koschmieder atmospheric scattering model |
| `euler-preprocess sky-depth` | Override depth values in sky regions with a constant |
| `euler-preprocess radial` | Convert planar (z-buffer) depth to radial (Euclidean) depth |

## Installation

```bash
uv pip install "euler-preprocess[gpu,progress] @ git+https://github.com/d-rothen/euler-fog"
```

## Usage

```bash
euler-preprocess fog       -c configs/example_dataset_config.json
euler-preprocess sky-depth -c configs/sky_depth_dataset_config.json
euler-preprocess radial    -c configs/radial_dataset_config.json
```

## Configuration

Every subcommand takes a **dataset config** JSON that points to the input data and a **transform config**. Each modality path must be a directory indexed by [ds-crawler](https://github.com/d-rothen/ds-crawler) with an `euler_loading` property that specifies the loader and function. This allows euler-loading to auto-select the correct dataset-specific loader.

### Dataset Config

```json
{
  "transform_config_path": "configs/run1.json",
  "output_path": "/path/to/output",
  "output_slot": "rgb",
  "sample": 42,
  "modalities": {
    "rgb": {"path": "/path/to/rgb", "split": "train"},
    "depth": "/path/to/depth",
    "semantic_segmentation": "/path/to/classSegmentation"
  },
  "hierarchical_modalities": {
    "intrinsics": {"path": "/path/to/intrinsics"}
  },
  "pipeline": {
    "output_root": "/pipeline/output",
    "outputs_manifest_path": "/pipeline/output/.euler_pipeline/pipeline_outputs.json",
    "output_targets": [
      {
        "slot": "rgb",
        "datasetType": "rgb",
        "relativePath": "foggy_rgb",
        "path": "/pipeline/output/foggy_rgb",
        "storage": "directory"
      }
    ]
  }
}
```

| Field | Description |
|---|---|
| `transform_config_path` | Path to the transform-specific config (see below). `fog_config_path` is also accepted for backward compatibility. |
| `output_path` | Output root used when no pipeline target overrides it. Optional if `pipeline.output_root` or `pipeline.output_targets[].path` supplies the destination. |
| `output_slot` | Optional slot selector when `pipeline.output_targets` contains multiple entries. Defaults to `rgb` for `fog`, `depth` for `sky-depth`, and `depth` for `radial`. |
| `sample` | Optional 0-based euler-loading dataset index. When set, only `dataset[sample]` is transformed, which is useful for small augmented benchmark slices from large datasets. |
| `samples` | Optional multi-sample selector. Use a list of 0-based indices (`[0, 10, 20]`) or a slice object such as `{"start": 0, "stop": 1000, "step": 2, "count": 100}`. `stop` is exclusive; `count` caps the selected indices after slicing. Do not set both `sample` and `samples`. |
| `modalities` | Regular modalities that participate in sample-ID intersection. Each value is either a plain path string or an object with a `path` key and an optional `split` key (see below). Which modalities are required depends on the transform (see table below). |
| `hierarchical_modalities` | Per-scene data (e.g. intrinsics). Same format as `modalities`. Loaded once per scene and cached. |
| `pipeline` | Optional runtime routing block compatible with `euler-inference` (`output_root`, `outputs_manifest_path`, `output_targets`). |

#### Inline splits

When a modality directory contains [ds-crawler](https://github.com/d-rothen/ds-crawler) split files (`.ds_crawler/split_<name>.json`), you can select a subset of the data by setting the `split` key on that modality. Sample IDs are matched by intersection across all modalities, so specifying a split on a single modality is sufficient to restrict the entire dataset.

For quick slices after euler-loading has matched modalities, set `samples`.
For example, `{"samples": {"step": 2}}` processes every second matched sample,
and `{"samples": {"start": 10, "step": 5, "count": 20}}` processes 20 samples
starting at index 10 with stride 5.

**Required modalities per transform:**

| Transform | `modalities` | `hierarchical_modalities` |
|---|---|---|
| `fog` | `rgb`, `depth`, `semantic_segmentation` | `intrinsics` when available; used for radial depth conversion and camera-profile optics |
| `sky-depth` | `depth`, `semantic_segmentation` | — |
| `radial` | `depth` | `intrinsics` |

#### Pipeline Runtime Block

`pipeline` follows the same shape as `euler-inference`:

```json
{
  "pipeline": {
    "output_root": "/pipeline/output",
    "outputs_manifest_path": "/pipeline/output/.euler_pipeline/pipeline_outputs.json",
    "output_targets": [
      {
        "slot": "depth",
        "datasetType": "depth",
        "relativePath": "radial_depth.zip",
        "path": "/pipeline/output/radial_depth.zip",
        "storage": "zip"
      }
    ]
  }
}
```

Notes:

- `output_root` is only a fallback when `output_path` is omitted.
- A matching `output_targets[].slot` overrides the write root for that run.
- `output_targets[].modelModalityId` is optional. When provided it is copied into the pipeline manifest; when omitted it is left out there as well.
- `storage: "directory"` writes a dataset directory and `storage: "zip"` writes a zip dataset.
- `storage: "file"` is parsed but rejected at runtime.
- When `outputs_manifest_path` is set and a pipeline target is matched, finalization writes `.euler_pipeline/pipeline_outputs.json` with the same manifest shape used by `euler-inference`.

---

## Fog Transform

### Fog Config

Controls the fog simulation.

```json
{
  "airlight": "from_sky",
  "seed": 1337,
  "depth_scale": 1.0,
  "resize_depth": true,
  "contrast_threshold": 0.05,
  "mode": "sample",
  "device": "cpu",
  "gpu_batch_size": 4,
  "capture": { "preset": "camera" },
  "camera_profile": "dashcam",
  "augmentations": { ... },
  "selection": { ... },
  "models": { ... }
}
```

| Field | Description |
|---|---|
| `airlight` | **Required.** Airlight estimation method: `"from_sky"` (mean sky colour), `"dcp"` (dark channel prior), or `"dcp_heuristic"` (robust DCP with sky-guided colouring when sky pixels exist). |
| `seed` | Random seed for reproducibility. `null` for non-deterministic. |
| `depth_scale` | Multiplier applied to depth values after loading. |
| `resize_depth` | Resize the depth map to match the RGB resolution (bilinear). |
| `contrast_threshold` | Threshold *C_t* used in the visibility-to-attenuation conversion (default `0.05`). |
| `mode` | Optional scenario mode. Omit it or use `"sample"` for current one-scenario-per-image behavior; use `"progressive"` to render every scenario step for every image. |
| `device` | `"cpu"`, `"cuda"`, `"mps"`, or `"gpu"` (alias for cuda). |
| `gpu_batch_size` | Batch size when running on GPU. Uniform-model samples are batched; heterogeneous samples are processed individually. |
| `capture` / `capture_artifacts` | Optional post-fog camera artifact pipeline. Omit it or set `{"stages": []}` for the legacy no-op path. Set `true`, `{"preset": "camera"}`, or a custom `stages` list to enable optics, raw sensor, ISP, and compression artifacts. |
| `camera_profile` | Optional named or inline camera profile whose stage defaults are merged into the capture stack before per-stage overrides. Built-ins are `"default"`, `"generic"`, `"dashcam"`, and `"low_light_fog"`. |
| `camera_profiles` | Optional map of project-specific named profiles. Use this for calibrated lens, sensor, ISP, and transport settings. |
| `scenario_profiles` | Optional top-level correlated condition sampler. Each sampled scenario can choose the fog model, override model parameters, select the airlight method, switch camera profile/settings, and force named capture-stage condition profiles. |
| `augmentations` | Optional stepped augmentation set. When present, every input sample produces every configured augmentation and uses the file-id hierarchy output layout described below. |

### Processing Pipeline

Fog generation is split into two phases:

1. Ideal scene rendering: physics-based fog and auxiliary `scattering_coefficient`
   / `atmospheric_light` maps are computed.
2. Capture artifacts: camera-specific effects are applied to the rendered RGB
   only. Physical fog maps stay stable while the RGB output can receive
   exposure shifts, lens blur, vignetting, raw sensor noise, Bayer/demosaicing
   artifacts, ISP tone/gamma/sharpening, and JPEG/resize/quantization effects.

This keeps physical fog maps stable while making the RGB output extensible for
real-camera simulation.

### Capture Artifact Stack

Enable the recommended camera stack with:

```json
"capture": { "preset": "camera" }
```

or equivalently:

```json
"capture": true
```

For tighter control, provide explicit stages in camera order:

```json
"camera_profiles": {
  "real_drive_front": {
    "optics": {
      "lens_distortion": -0.012,
      "vignetting_strength": 0.18,
      "windshield_haze": {"enabled": true, "probability": 0.55}
    },
    "sensor": {
      "bayer_pattern": "RGGB",
      "iso": {"dist": "choice", "values": [200, 400, 800]},
      "base_iso": 100,
      "auto_exposure": {
        "enabled": true,
        "metering": "center_weighted",
        "target_luminance": {"dist": "uniform", "min": 0.16, "max": 0.24},
        "highlight_protection": 0.7,
        "resolve_iso": true,
        "max_iso": 1600
      },
      "full_well_electrons": [14000, 12000, 13000],
      "read_noise_electrons": {"dist": "uniform", "min": 2.0, "max": 6.0},
      "shadow_recovery_noise": {
        "enabled": true,
        "luminance_threshold": {"dist": "uniform", "min": 0.16, "max": 0.24},
        "luma_sigma": {"dist": "uniform", "min": 0.0, "max": 0.001},
        "chroma_sigma": {"dist": "uniform", "min": 0.012, "max": 0.028},
        "chroma_mode": "balanced",
        "red_chroma_gain": {"dist": "uniform", "min": 0.7, "max": 1.0},
        "blue_chroma_gain": {"dist": "uniform", "min": 1.7, "max": 2.7},
        "chroma_axis_correlation": {"dist": "uniform", "min": 0.05, "max": 0.25},
        "chroma_spatial_sigma": 0.0,
        "chroma_fine_fraction": 1.0,
        "chroma_luminance_preservation": 1.0,
        "black_noise_floor": 0.25,
        "black_suppression_luminance": 0.03,
        "black_suppression_softness": 0.08
      },
      "black_level": [0.003, 0.0035, 0.003],
      "white_level": [1.0, 0.995, 1.0],
      "adc_bit_depth": 12,
      "post_demosaic_bit_depth": 12
    },
    "isp": {
      "tone_map": "reinhard",
      "gamma": "srgb",
      "denoise_sigma": 0.2,
      "sharpen_amount": 0.2,
      "saturation": 0.9
    },
    "transport": {
      "jpeg": {"enabled": true, "quality": {"dist": "uniform", "min": 65, "max": 92}},
      "bit_depth": 8
    }
  }
},
"camera_profile": "real_drive_front",
"capture": {
  "stages": [
    {
      "type": "optics",
      "blur_sigma": {"dist": "uniform", "min": 0.2, "max": 0.8},
      "vignetting_strength": 0.15,
      "windshield_haze": {"enabled": true, "probability": 0.4},
      "droplets": {"enabled": false}
    },
    {
      "type": "sensor",
      "input_space": "srgb",
      "exposure_gain": {"dist": "uniform", "min": 0.85, "max": 1.2},
      "row_noise_sigma": 0.003
    },
    {
      "type": "isp",
      "tone_map": "reinhard",
      "gamma": "srgb",
      "denoise_sigma": 0.2,
      "sharpen_amount": 0.2,
      "saturation": 0.9
    },
    {
      "type": "transport",
      "jpeg": {"enabled": true, "quality": {"dist": "uniform", "min": 65, "max": 92}},
      "bit_depth": 8
    }
  ]
}
```

Supported stage types:

| Stage | Main effects |
|---|---|
| `optics` | Defocus/MTF blur, motion blur, bloom, veiling glare, vignetting, chromatic aberration, lens distortion, windshield haze, optional droplets. |
| `sensor` | Image-driven or sampled exposure, white balance, camera matrix, Bayer mosaic, shot/read noise, fixed-pattern noise, row/column banding, shadow-local recovery noise, hot/dead pixels, bilinear demosaic. |
| `isp` | Denoising, color correction, tone mapping, sRGB/gamma, local contrast, sharpening halos, saturation shifts. |
| `transport` | Crop/resize, bit-depth quantization, JPEG round-trip compression. |
| `exposure` | Lightweight standalone exposure and white-balance stage for simple custom chains. |

Set `sensor.auto_exposure.enabled` to meter the rendered image before raw sensor
sampling. `target_luminance`, `metering`, `highlight_*`, and gain bounds choose
the exposure; `resolve_iso` can raise ISO from the metering pressure, dark pixel
fraction, and fog opacity. When auto exposure is enabled, `exposure_gain` still
applies as scenario-specific exposure compensation.

Set `sensor.shadow_recovery_noise.enabled` to add extra post-demosaic luma and
chroma corruption only where the pre-exposure rendered luminance was low. This
is useful for reducing broad global grain while keeping lifted shadows visibly
noisy. For dark high-ISO scenes, keep `luma_sigma` much lower than
`chroma_sigma`, use `chroma_mode: "balanced"`, and leave
`chroma_luminance_preservation` near `1.0`; this makes the local corruption read
as color noise instead of black luma speckles. `red_chroma_gain`,
`blue_chroma_gain`, and `chroma_axis_correlation` can match camera-specific
color noise, including blue/purple-biased high-ISO noise. Keep
`chroma_spatial_sigma` near `0` and avoid chroma subsampling when the target
noise is fine-grained rather than blocky. `black_noise_floor` with
`black_suppression_luminance`/`black_suppression_softness` reduces the extra
noise in near-clipped black regions, so the strongest visible noise sits in
dim-but-readable shadows.

Set `sensor.noise_adjustment` for relative, scenario-level noise controls on
top of the selected camera/condition profile. `level: 1.0` leaves the authored
profile unchanged; lower values suppress read/static/chroma noise and higher
values amplify it. `static_chroma_bias` ranges from `-1.0` for more fixed
pattern, row/column, banding, and bad-pixel noise to `1.0` for more
chromatic/high-ISO-looking shadow noise:

```json
{
  "capture_overrides": {
    "sensor": {
      "condition_profile": "nominal_gloom",
      "noise_adjustment": {
        "enabled": true,
        "level": 1.25,
        "static_chroma_bias": 0.35
      }
    }
  }
}
```

Any stage can define `condition_profiles` to sample coherent per-image settings
before the stage runs. This is useful for exposure states where ISO, exposure
gain, read noise, banding, and dark/fog noise modulation should move together:

```json
{
  "type": "sensor",
  "condition_profiles": [
    {"name": "clean_daylight", "weight": 0.25, "exposure_gain": 1.0, "iso": 100},
    {"name": "underexposed_noisy", "weight": 0.25, "exposure_gain": 0.65, "iso": 1600}
  ]
}
```

Top-level `scenario_profiles` sample one latent scene/camera condition before
rendering. The selected scenario is merged over the root config, so it can drive
fog density, atmospheric light, camera profile, capture-stage overrides, ISP, and
compression together:

```json
"scenario_profiles": [
  {
    "name": "clean_low_noise_haze",
    "weight": 0.22,
    "model": "heterogeneous_k_ls",
    "airlight_method": "dcp_heuristic",
    "models": {
      "heterogeneous_k_ls": {
        "visibility_m": {"dist": "uniform", "min": 60.0, "max": 130.0}
      }
    },
    "capture_overrides": {
      "sensor": {"condition_profile": "clean_daylight"},
      "isp": {"denoise_sigma": {"dist": "uniform", "min": 0.08, "max": 0.32}}
    }
  },
  {
    "name": "underexposed_dense_gloom",
    "weight": 0.25,
    "model": "heterogeneous_k_ls",
    "airlight_method": "dcp_heuristic",
    "models": {
      "heterogeneous_k_ls": {
        "visibility_m": {"dist": "uniform", "min": 18.0, "max": 55.0}
      }
    },
    "capture_overrides": {
      "sensor": {"condition_profile": "underexposed_noisy"},
      "transport": {"jpeg": {"quality": {"dist": "uniform", "min": 54, "max": 78}}}
    }
  }
]
```

`capture_overrides` is merged after camera-profile and stage settings. Use
`condition_profile` to force one named profile from a stage's
`condition_profiles`; if omitted, the stage continues sampling its own profile
weights locally.

Set top-level `"mode": "progressive"` to emit every configured scenario for
every input image instead of sampling one scenario. Each scenario accepts
`"steps"` and `"progressive_weight"` (or `"max_weight"` / `"weight"` as
aliases); the transform writes steps from weight `0` through the scenario's
configured weight, and weight `1` matches the original scenario. Fog density is
progressed in scattering-coefficient space, while numeric camera/config values
blend from the base config toward the scenario config. Progressive blends clamp
probability-like values and non-negative physical factors back into valid
mathematical domains so extrapolated weights above `1` do not create invalid
render parameters.
Source-backed outputs are written as `fog_progression` variants under each
source file id.

### Fog Model

The core equation is the **Koschmieder model** (atmospheric scattering):

```
I_fog(x) = I(x) * t(x)  +  L_s * (1 - t(x))
```

where:

- **I(x)** is the original RGB colour at pixel *x*
- **t(x) = exp(-k * d(x))** is the transmittance, which falls exponentially with depth *d* and attenuation coefficient *k*
- **L_s** is the atmospheric light (airlight), i.e. the colour of the fog/sky light scattered towards the camera
- **k** is derived from a meteorological visibility distance *V*: `k = -ln(C_t) / V`

Distant objects are attenuated more (`t` approaches 0) and replaced by airlight, just as in real fog.

### How Each Modality is Used

**RGB** — The clean scene image. Normalised to float32 in [0, 1]. This is the *I(x)* term in the fog equation -- it gets blended with the airlight according to transmittance.

**Depth** — A per-pixel depth map in **metres**. Provides the *d(x)* term in the transmittance calculation `t(x) = exp(-k * d(x))`. Pixels with greater depth receive more fog. Invalid values (NaN, inf, negative) are clamped to zero (treated as infinitely close, receiving no fog).

**Semantic Segmentation** — A per-pixel semantic segmentation map from which a boolean sky mask is derived, loaded via euler-loading's dataset-specific `semantic_segmentation` loader. The sky mask is used for airlight estimation when the `airlight` method is `"from_sky"`: the mean RGB of all sky pixels in the clean image is used as the airlight colour *L_s*.

**Intrinsics** *(optional)* — When present, planar (z-buffer) depth is converted to radial (Euclidean) depth before fog is applied.

### Airlight Estimation

The `airlight` config key selects how the atmospheric light *L_s* is estimated:

| Method | Description |
|---|---|
| `from_sky` | Mean RGB of sky pixels in the clean image. Falls back to white `[1, 1, 1]` when no sky pixels exist. |
| `dcp` | Dark Channel Prior — selects the brightest pixel (by channel sum) among the top 0.1% darkest-channel pixels. |
| `dcp_heuristic` | Robust DCP heuristic — pools the brighter half of the top 0.1% darkest-channel pixels, and when sky pixels exist it uses the brightest sky colours as the chromaticity prior while preserving DCP-derived luminance. Optional bias controls can nudge the result toward white or a cool fog tint. |

GPU-native implementations (`DCPAirlightTorch`, `DCPHeuristicAirlightTorch`) are used automatically when running on GPU.

When `airlight` is `"dcp_heuristic"`, you can optionally add:

```json
"dcp_heuristic": {
  "patch_size": 15,
  "top_percent": 0.001,
  "white_bias": 0.1,
  "cool_bias": 0.15,
  "cool_target": [0.93, 0.97, 1.0]
}
```

- `white_bias` mixes the final airlight toward neutral white.
- `cool_bias` mixes the final airlight toward a sky-relative cool target.
- `cool_target` is the cool-white anchor used to derive that sky-relative target. When sky pixels exist, the effective cool target is a blend of the estimated sky colour and `cool_target`; without sky pixels, it falls back to the airlight estimate and `cool_target`.
- `white_bias + cool_bias` must be `<= 1`.
- The tint bias preserves the estimated airlight luminance, so it shifts colour without silently changing fog density.

### Airlight Intensity Dampening

Estimated airlight is dampened by default as fog density increases. This keeps
strong fog closer to the low, grey lighting seen in real in-car fog footage
instead of letting DCP-style estimates wash dense fog toward white.

Each fog model can override the dampening curve:

```json
"airlight_dampening": {
  "enabled": true,
  "apply_to": "estimated",
  "reference_visibility_m": 80.0,
  "min_factor": 0.45,
  "max_factor": 1.0,
  "strength": 1.0
}
```

The factor is:
`min_factor + (max_factor - min_factor) / (1 + strength * beta / reference_beta)`.
`min_factor` and `max_factor` must be finite and non-negative. Values above
`1.0` are allowed when you intentionally want to brighten estimated airlight;
the final RGB output is still clamped to the valid image range. `strength`
must remain finite and non-negative.
`reference_beta` is either `reference_scattering_coefficient` /
`reference_beta`, or it is derived from `reference_visibility_m` using the
model's contrast threshold. The default applies only when `atmospheric_light`
uses an estimated airlight method (`"from_sky"`, `"dcp"`, or
`"dcp_heuristic"`); literal RGB `atmospheric_light` values stay exact unless
`apply_to` is set to `"all"`. Set `"enabled": false` or `apply_to: "none"` to
preserve the previous undampened behavior.

For `heterogeneous_ls` and `heterogeneous_k_ls`, the Perlin atmospheric-light
field is sampled around the dampened base airlight, so the spatial variation
does not reintroduce the old over-illuminated look.

### Model Selection

Each image is assigned a fog model via the `selection` block:

```json
"selection": {
  "mode": "weighted",
  "weights": {
    "uniform": 0.25,
    "heterogeneous_k": 0.35,
    "heterogeneous_ls": 0.25,
    "heterogeneous_k_ls": 0.15
  }
}
```

- **`fixed`** mode: always use a single named model.
- **`weighted`** mode: randomly select a model per image according to normalised weights.

Four models are available:

| Model | Description |
|---|---|
| `uniform` | Constant *k* and *L_s*. Standard homogeneous fog. |
| `heterogeneous_k` | Spatially-varying *k*, constant *L_s*. Simulates patchy fog / fog banks. |
| `heterogeneous_ls` | Constant *k*, spatially-varying *L_s*. Simulates scattered-light colour variation. |
| `heterogeneous_k_ls` | Both *k* and *L_s* vary spatially. Most expressive model. |

### Visibility Distribution

Each model specifies a `visibility_m` distribution from which a visibility distance (in metres) is sampled per image:

| `dist` | Parameters | Description |
|---|---|---|
| `constant` | `value` | Fixed value. |
| `uniform` | `min`, `max` | Uniform random in range. |
| `normal` | `mean`, `std`, optional `min`/`max` | Gaussian, optionally clamped. |
| `lognormal` | `mean`, `sigma`, optional `min`/`max` | Log-normal. |
| `choice` | `values`, optional `weights` | Discrete weighted choice. |

The sampled visibility *V* is converted to the attenuation coefficient:
**k = -ln(C_t) / V**. This happens once per output image. For
`heterogeneous_k` and `heterogeneous_k_ls`, that sampled value is the base
coefficient that is then spatially modulated by the noise field.

### Stepped Augmentations

For benchmark generation, set `augmentations` in the fog config. This switches
the fog transform from one sampled output per input to one output per configured
variant:

```json
{
  "airlight": "from_sky",
  "seed": 1337,
  "contrast_threshold": 0.05,
  "augmentations": {
    "file_id_hierarchy_name": "file_id",
    "attribute_key": "fog_augmentation",
    "models": ["uniform"],
    "visibility_m": [10, 20, 40, 70, 100],
    "airlight_methods": ["from_sky"]
  }
}
```

The matrix form above expands as the Cartesian product of `models`,
`visibility_m` (MOR in metres), optional `scattering_coefficients` / `beta`, and
airlight choices. `file_id_hierarchy_name` names the inserted hierarchy level
when the underlying ds-crawler writer has a hierarchy separator; the directory
name is the source file id in either case. For tighter control, use explicit
variants:

```json
"augmentations": {
  "variants": [
    {
      "id": "mor_010m_sky",
      "model": "uniform",
      "visibility_m": 10,
      "airlight_method": "from_sky"
    },
    {
      "id": "beta_0.15_white",
      "model": "heterogeneous_k",
      "scattering_coefficient": 0.15,
      "atmospheric_light": [1.0, 1.0, 1.0],
      "k_hetero": {
        "scales": "smooth_auto",
        "correlation_length_fraction": 0.25,
        "octaves": 3,
        "min_factor": 0.65,
        "max_factor": 1.45,
        "contrast": 0.65,
        "normalize_to_mean": true
      }
    }
  ]
}
```

Each output entry receives per-file ds-crawler attributes under
`fog_augmentation`, including the augmentation id, source id, source full id,
model, actual scattering coefficient, actual atmospheric light, and configured
MOR/beta descriptors when available. euler-loading exposes these as
`sample["attributes"]["rgb"]["fog_augmentation"]`.

### Heterogeneous Noise Fields

Both `k_hetero` and `ls_hetero` use Perlin FBM (fractional Brownian
motion) to generate spatially-varying factor fields. For realistic fog,
prefer the smooth mode: it keeps Perlin wavelengths tied to the image size,
then optionally reduces noise contrast and applies a final blur before mapping
the noise to physical factors.

```json
"k_hetero": {
  "scales": "smooth_auto",
  "correlation_length_fraction": 0.25,
  "octaves": 3,
  "max_scale": null,
  "min_factor": 0.65,
  "max_factor": 1.45,
  "contrast": 0.65,
  "smooth_sigma_fraction": 0.0,
  "normalize_to_mean": true
}
```

The noise field (values in [0, 1]) is mapped to a factor field:
`factor(x) = min_factor + (max_factor - min_factor) * noise(x)`.
`contrast < 1` compresses the noise around 0.5 before this mapping, avoiding
extreme local fog density. When `normalize_to_mean` is `true`, the factor field
is rescaled so its spatial mean equals 1.0, preserving the overall fog density
while introducing spatial variation. In other words, with heterogeneous `k`:
`k(x) = k_sampled * factor(x)`. If `visibility_m` / MOR was sampled from a
distribution, `k_sampled` is the coefficient derived from that one sampled MOR.
With `normalize_to_mean: true`, the arithmetic mean of the per-pixel `k` map
equals `k_sampled`; the median is not forced to match. With
`normalize_to_mean: false`, the map mean shifts by the mean of the factor field.

For `heterogeneous_ls` / `heterogeneous_k_ls`, `ls_hetero` can also include a
weak view-direction illumination prior. This modulates the atmospheric-light
field itself, so the rendered effect is still gated by fog transmittance:

```json
"ls_hetero": {
  "ls_gradient": {
    "enabled": true,
    "probability": 0.65,
    "axis": "vertical",
    "top_factor": {"dist": "uniform", "min": 1.03, "max": 1.14},
    "bottom_factor": {"dist": "uniform", "min": 0.88, "max": 0.99},
    "gamma": {"dist": "uniform", "min": 0.85, "max": 1.6},
    "normalize_to_mean": true,
    "fog_opacity_weight": 0.65
  }
}
```

| Parameter | Effect |
|---|---|
| `min_factor` / `max_factor` | Range of the multiplicative factor. |
| `normalize_to_mean` | Rescale factors so the image-wide mean equals the base value. Recommended for `k_hetero`. |
| `scales: "smooth_auto"` | Build low-frequency Perlin scales from the image size. |
| `correlation_length_fraction` | Approximate smallest fog feature size as a fraction of the shorter image side. Larger values create smoother gradients. |
| `octaves` / `lacunarity` / `max_scale` | Control how many increasingly broad Perlin components are mixed. |
| `contrast` | Compress or expand the Perlin range before mapping to factors. Values below 1 are recommended. |
| `smooth_sigma` / `smooth_sigma_fraction` | Optional final Gaussian blur in pixels or as a fraction of the shorter image side. |
| `ls_gradient` | Optional `L_s` top-to-bottom or left-to-right factor field. Keep it weak and probabilistic to avoid a deterministic image-position shortcut. |

### Fog Output

CLI runs write a source-backed RGB dataset. The output keeps the source RGB
dataset's relative paths, basenames, extensions, and ds-crawler metadata so the
result stays loadable by `euler-loading`:

```
<output_path>/
  .ds_crawler/dataset-head.json
  .ds_crawler/ds-crawler.json
  .ds_crawler/index.json
  Scene01/
    Camera_0/
      00000.png
```

When a pipeline target is present, `pipeline.output_targets[].path` replaces
`output_path` entirely. Standalone/direct `FogTransform(...)` usage without the
CLI still uses the legacy per-model layout with `config.json` sidecars.

With `augmentations` enabled, source-backed outputs are written one level below
the source file id instead:

```
<output_path>/
  .ds_crawler/dataset-head.json
  .ds_crawler/ds-crawler.json
  .ds_crawler/index.json
  Scene01/
    Camera_0/
      00000/
        mor_10m_airlight_from_sky.png
        mor_20m_airlight_from_sky.png
```

Auxiliary `scattering_coefficient` and `atmospheric_light` pipeline targets use
the same file-id hierarchy and write matching `.npy` augmentation files.

---

## Sky-Depth Transform

Overrides depth values in sky regions with a configurable constant. Useful for datasets where sky depth is encoded as zero or infinity and needs to be normalised to a large finite value.

### Sky-Depth Config

```json
{
  "sky_depth_value": 1000.0
}
```

| Field | Description |
|---|---|
| `sky_depth_value` | Depth value assigned to all sky pixels. Defaults to `1000.0`. |

### Sky-Depth Output

CLI runs write a source-backed depth dataset mirroring the input depth
modality's paths, filenames, extensions, and metadata. Standalone/direct
`SkyDepthTransform(...)` usage keeps the legacy `.npy` output behavior.

---

## Radial Transform

Converts planar (z-buffer) depth to radial (Euclidean) depth using camera intrinsics. For each pixel *(u, v)*:

```
d_radial(u, v) = d_planar(u, v) * sqrt(((u - cx)/fx)^2 + ((v - cy)/fy)^2 + 1)
```

### Radial Config

```json
{}
```

No special parameters are required. The transform reads intrinsics from the `intrinsics` hierarchical modality.

### Radial Output

CLI runs write a source-backed depth dataset mirroring the input depth
modality's layout and writer metadata. The emitted `index.json` also flips
`meta.radial_depth` to `true`. Standalone/direct `RadialTransform(...)` usage
keeps the legacy `.npy` output behavior.
