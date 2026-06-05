# Realism Implementation Plan for the Fog and Camera-Noise Pipeline

This document turns the realism audit into a concrete implementation plan for `d-rothen/euler-preprocess`. It is intentionally written in English while preserving repository identifiers exactly as they appear in the code: file names, class names, function names, config keys, model names, and stage names are not translated or renamed unless the plan explicitly proposes a new key.

## Executive summary

The current architecture is a strong starting point. `FogProcessingPipeline` first renders the ideal fogged scene and only then applies `CaptureArtifactPipeline`; the physical maps returned by the fog renderer (`beta`, `airlight`, `k_map`, `ls_map`) remain separate from post-render camera artifacts. This separation is exactly what we want for densely labeled synthetic datasets: the RGB output can become more camera-realistic without corrupting depth, semantic labels, or the fog auxiliary maps.

The largest remaining realism gap is not the base atmospheric scattering equation. The current `apply_fog()` model implements the standard efficient approximation

```text
I(x) = J(x) * exp(-beta(x) * d(x)) + L_s(x) * (1 - exp(-beta(x) * d(x)))
```

which is appropriate for fast dataset preprocessing. The main issue is that gloomy weather is currently represented mostly through fog density and `airlight`/`ls_map`. Near pixels have high transmission, so `J(x)` passes through almost unchanged. In dense, gloomy scenarios this leaves nearby surfaces too bright, even though real weather would also reduce scene illumination. The correct fix is not only to tweak auto exposure. We should add a separate scene-radiance reduction pass before fog rendering and then improve the existing auto-exposure metering to be fog-aware.

The most important implementation work is therefore:

1. Preserve radiometric linearity before fog rendering.
2. Add a `scene_illumination` / gloom pass that darkens the pre-fog scene radiance `J`, with a depth-weighted near-field term.
3. Extend the existing `auto_exposure` implementation in `SensorStage` so metering can suppress sky and dense far-field fog when desired.
4. Keep and improve the existing `TransportStage`; do not treat transport/JPEG as missing.
5. Add persistent sensor-pattern realism (`PRNU`, `DSNU`, persistent hot/dead pixels, fixed row/column structure) on top of the already implemented shot/read/banding/noise stack.
6. Add camera-profile-level CRF/tone-map LUTs for cheap but visible realism gains.

## Repository facts that matter for this plan

The plan assumes the following current implementation state:

- `euler_preprocess/fog/pipeline.py`
  - `FogProcessingPipeline.render_scene_np()` estimates `airlight` from the input `rgb` and calls `apply_model()`.
  - `FogProcessingPipeline.process_np()` then calls `apply_capture_np()` with a `CaptureContext` containing `sample_id`, `rng`, `intrinsics`, `depth_m`, and `k_map`.
  - `sky_mask` is available in `process_np()` but is not currently forwarded into `CaptureContext.attributes`.

- `euler_preprocess/fog/models.py`
  - `DEFAULT_MODEL_CONFIGS` defines `uniform`, `heterogeneous_k`, `heterogeneous_ls`, and `heterogeneous_k_ls`.
  - `visibility_to_k()` converts `visibility_m` and `contrast_threshold` into the scattering coefficient.
  - `dampen_airlight()` already supports `airlight_dampening`.
  - `apply_fog()` implements direct transmission plus airlight using `t = exp(-k_field * depth_m)`.
  - `apply_model()` builds `k_map` and `ls_map`, applies optional `k_hetero`, `ls_hetero`, and `ls_gradient`, and then calls `apply_fog()`.

- `euler_preprocess/fog/capture.py`
  - `CaptureContext` already has `depth_m`, `k_map`, `fog_opacity`, and `attributes` fields.
  - `SensorStage` already includes `auto_exposure`, `resolve_iso`, shot noise, read noise, fixed-pattern noise, row/column banding, hot/dead pixels, ADC quantization, Bayer mosaic, bilinear demosaic, and `shadow_recovery_noise`.
  - `_context_fog_opacity()` can derive fog opacity from `depth_m` and `k_map` when `fog_opacity` is not explicitly provided.
  - `noise_modulation` and `shadow_recovery_noise` already have hooks for depth/fog-weighted noise adjustments.
  - `TransportStage` exists and performs crop/resize, bit-depth quantization, and JPEG round-trip compression.
  - The default `camera` preset is `optics -> sensor(input_space="srgb") -> isp -> transport`.

These facts correct two important possible misunderstandings: `TransportStage` is not missing, and auto exposure is not missing. The patches below should extend these existing paths instead of introducing parallel duplicate stages.

## P0: Fix the gloomy near-field brightness problem

### Patch P0.1: Make the fog renderer explicit about input color space

**Problem**

`render_scene_np()` currently passes `rgb` directly to both `AtmosphericLightResolver.estimate_np()` and `apply_model()`. If the dataset RGB is sRGB or otherwise display-encoded, fog is mixed in a nonlinear space. That makes both transmission and airlight mixing less physically meaningful.

There is already `_srgb_to_linear()` and `_linear_to_srgb()` in `capture.py`, but those helpers are currently local to capture processing. `SensorStage` can linearize its input when `input_space == "srgb"`, but that happens after fog has already been rendered. For physical fog realism, the scene radiance entering `apply_model()` should already be linear.

**Where to patch**

- `euler_preprocess/fog/pipeline.py`
  - `FogProcessingPipeline.render_scene_np()`
- optionally shared utility location:
  - move or duplicate `_srgb_to_linear()` / `_linear_to_srgb()` into a small shared module such as `euler_preprocess/common/color.py`

**Implementation**

Add a top-level fog config key, for example:

```json
"render_input_space": "srgb"
```

Supported values:

- `"linear"`: `rgb` is already scene-linear.
- `"srgb"`: convert to scene-linear before airlight estimation and fog rendering.

Then update `render_scene_np()` roughly as follows:

```python
rgb_for_render = rgb
if str(model_cfg.get("render_input_space", root_cfg.get("render_input_space", "srgb"))).lower() == "srgb":
    rgb_for_render = srgb_to_linear(rgb)

estimated_airlight = self.atmospheric_light.estimate_np(
    rgb_for_render,
    sky_mask,
    sample_id=sample_id,
    method=airlight_method,
)

foggy_lin, beta, airlight, k_map, ls_map = apply_model(
    rgb_for_render,
    depth_m,
    model_name,
    model_cfg,
    rng,
    self.contrast_threshold_default,
    estimated_airlight,
)
```

**Important association to preserve**

If `foggy_lin` is passed into `CaptureArtifactPipeline`, `SensorStage.input_space` must be `"linear"`. The current `_CAMERA_PRESET` sets `{"type": "sensor", "input_space": "srgb"}`. After this patch, do one of the following:

1. Preferable: change the preset to `{"type": "sensor", "input_space": "linear"}` when the fog renderer outputs linear RGB.
2. Backward-compatible option: add a pipeline-level flag such as `capture_input_space`, and set it automatically from `render_input_space`.
3. Minimal option: re-encode `foggy_lin` to sRGB before capture, but this is less physically clean because the sensor stage then immediately linearizes again.

**Expected effect**

Fog density, `airlight`, and all later exposure/noise behavior become more stable because they operate on linear radiance instead of display-encoded RGB.

### Patch P0.2: Add a `scene_illumination` pass before `apply_fog()`

**Problem**

The current fog model changes visibility and airlight but does not independently reduce the scene illumination `J`. In `apply_fog()`, near pixels have `t ~= 1`, so they remain close to the original RGB. This is the direct cause of overly bright foreground regions in gloomy fog.

Real foggy/gloomy weather changes both the medium and the illumination. The atmosphere reduces contrast and adds airlight, but overcast, low-sun, wet, or stormy conditions also reduce incoming scene radiance before the camera sees it. This should be represented as a pre-fog modification of `J`, not as a post-fog darkening of the final image.

**Where to patch**

- `euler_preprocess/fog/models.py`
  - add helper functions near `apply_fog()`:
    - `apply_scene_illumination_np()`
    - optionally `apply_scene_illumination_torch()` if a torch fog path is later added
  - call the helper inside `apply_model()` before every call to `apply_fog()`.

**Proposed config**

Add this block to model configs, especially to gloomy scenario profiles:

```json
"scene_illumination": {
  "enabled": true,
  "global_ev": {"dist": "uniform", "min": 0.15, "max": 0.75},
  "near_ev": {"dist": "uniform", "min": 0.20, "max": 1.10},
  "near_decay_depth_m": {"dist": "uniform", "min": 8.0, "max": 25.0},
  "fog_coupled_ev": {"dist": "uniform", "min": 0.0, "max": 0.45},
  "airlight_ev_ratio": 0.20,
  "sky_weight": 0.0,
  "min_radiance_scale": 0.08
}
```

Parameter meanings:

- `global_ev`: global exposure reduction applied to all non-sky scene radiance.
- `near_ev`: extra near-field reduction. This directly addresses the foreground-too-bright artifact.
- `near_decay_depth_m`: depth scale for the near-field term.
- `fog_coupled_ev`: optional extra reduction proportional to local fog opacity.
- `airlight_ev_ratio`: optional weak damping of `ls_base` / `ls_field`, but this should be much smaller than the scene-radiance reduction.
- `sky_weight`: how much the pass affects sky pixels. Default should be low or zero if sky is already used to estimate airlight.
- `min_radiance_scale`: numerical safety floor.

**Suggested formula**

Apply this to the pre-fog scene radiance `J`, not to the final foggy image:

```text
near_weight(x) = exp(-max(depth_m(x), 0) / near_decay_depth_m)
fog_opacity(x) = 1 - exp(-max(k_map(x), 0) * max(depth_m(x), 0))
EV(x) = global_ev + near_ev * near_weight(x) + fog_coupled_ev * fog_opacity(x)
J'(x) = J(x) * 2^(-EV(x))
```

If `sky_mask` is available:

```text
EV(x) = EV(x) * (1 - sky_mask(x) * (1 - sky_weight))
```

**Pseudo-code**

```python
def apply_scene_illumination_np(
    rgb_lin: np.ndarray,
    depth_m: np.ndarray,
    k_field,
    model_cfg: dict,
    rng: np.random.Generator,
    sky_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = _resolve_scene_illumination_config(model_cfg, rng)
    if not cfg["enabled"]:
        return rgb_lin, np.zeros(depth_m.shape, dtype=np.float32)

    global_ev = _sample_float(cfg.get("global_ev", 0.0), rng, "scene_illumination.global_ev")
    near_ev = _sample_float(cfg.get("near_ev", 0.0), rng, "scene_illumination.near_ev")
    d0 = max(_sample_float(cfg.get("near_decay_depth_m", 15.0), rng, "scene_illumination.near_decay_depth_m"), 1e-6)
    fog_ev = _sample_float(cfg.get("fog_coupled_ev", 0.0), rng, "scene_illumination.fog_coupled_ev")

    k_map = broadcast_k_field(k_field, depth_m.shape[0], depth_m.shape[1])
    depth = np.maximum(depth_m.astype(np.float32, copy=False), 0.0)
    near_weight = np.exp(-depth / d0).astype(np.float32, copy=False)
    fog_opacity = 1.0 - np.exp(-np.maximum(k_map, 0.0) * depth)

    ev_map = global_ev + near_ev * near_weight + fog_ev * np.clip(fog_opacity, 0.0, 1.0)
    if sky_mask is not None:
        sky_weight = float(np.clip(cfg.get("sky_weight", 0.0), 0.0, 1.0))
        ev_map *= 1.0 - np.asarray(sky_mask, dtype=np.float32) * (1.0 - sky_weight)

    min_scale = max(float(cfg.get("min_radiance_scale", 0.08)), 0.0)
    scale = np.maximum(np.exp2(-ev_map), min_scale).astype(np.float32, copy=False)
    return (rgb_lin * scale[..., None]).astype(np.float32, copy=False), ev_map.astype(np.float32, copy=False)
```

**Needed signature change**

`apply_model()` currently does not receive `sky_mask`. To make sky exclusion possible without guessing, extend the signature:

```python
def apply_model(..., estimated_airlight: np.ndarray, sky_mask: np.ndarray | None = None) -> ...:
```

Then pass `sky_mask` from `FogProcessingPipeline.render_scene_np()`.

**Expected effect**

Foreground objects in gloomy scenarios become darker before fog is mixed, while far objects still converge toward `ls_map` through the existing physical fog equation. This directly fixes the observed artifact without breaking the semantics of `k_map`, `ls_map`, or `airlight`.

### Patch P0.3: Add fog-aware and sky-aware auto-exposure metering

**Problem**

`SensorStage.auto_exposure` already exists and includes `target_luminance`, `highlight_protection`, gain bounds, `resolve_iso`, and fog-dependent ISO boosting. However, `_auto_exposure_metrics_np()` and `_auto_exposure_metrics_torch()` currently compute mean/percentiles over the finite luminance values and a center-weighted mean. They do not weight down sky pixels or dense far-field fog in the meter itself. They only compute `mean_fog_opacity` for later ISO logic.

This means the exposure meter can still be pulled by a bright sky or bright fog veil, depending on the scene and profile.

**Where to patch**

- `euler_preprocess/fog/pipeline.py`
  - pass `sky_mask` into `CaptureContext.attributes`.
- `euler_preprocess/fog/capture.py`
  - `_auto_exposure_metrics_np()`
  - `_auto_exposure_metrics_torch()`
  - optionally add helper functions:
    - `_auto_exposure_weight_np()`
    - `_auto_exposure_weight_torch()`
    - `_weighted_percentile_np()`
    - `_weighted_quantile_torch()`

**Pipeline change**

In `process_np()`, update the `CaptureContext` construction:

```python
context=CaptureContext(
    sample_id=sample_id,
    rng=rng,
    intrinsics=intrinsics,
    depth_m=depth_m,
    k_map=result.k_map,
    attributes={"sky_mask": sky_mask},
)
```

**Config extension**

Extend `sensor.auto_exposure` with optional keys:

```json
"auto_exposure": {
  "enabled": true,
  "metering": "fog_aware_center_weighted",
  "target_luminance": {"dist": "uniform", "min": 0.14, "max": 0.22},
  "highlight_protection": 0.75,
  "manual_gain_weight": 0.0,
  "sky_suppression": 0.85,
  "fog_meter_suppression": 0.60,
  "depth_meter_decay_m": 35.0,
  "min_meter_weight": 0.05,
  "resolve_iso": true,
  "fog_iso_boost": 0.25
}
```

**Suggested metering weight**

```text
w(x) = w_center(x)
     * max(min_meter_weight, exp(-depth_m(x) / depth_meter_decay_m))
     * (1 - fog_meter_suppression * fog_opacity(x))
     * (1 - sky_suppression * sky_mask(x))
```

Use this weight for `meter_percentile`, `highlight_percentile`, and/or weighted means.

**Implementation notes**

- Keep the existing modes (`mean`, `percentile`, `center_percentile`, `highlight`, default center-weighted) for backward compatibility.
- Add new modes rather than changing existing behavior silently:
  - `"fog_aware"`
  - `"fog_aware_center_weighted"`
  - `"sky_aware_center_weighted"`
- Use downsampled luminance or simple weights if performance becomes an issue. This is cheap compared with fog rendering and JPEG.
- The current `manual_gain_weight` default is `1.0`, which means manual gain remains multiplicative even when AE is enabled. For realistic AE-dominant camera profiles, set `manual_gain_weight` closer to `0.0` and use `exposure_compensation_ev` for scenario-specific bias.

**Expected effect**

Exposure becomes less likely to over-brighten foreground regions because the meter no longer treats bright sky and dense far-field airlight as fully representative of the scene exposure target.

## P1: Make the sensor identity persistent

### Patch P1.1: Add persistent PRNU, DSNU, row/column bias, and bad-pixel maps

**Problem**

`SensorStage` already models many important noise sources, but several are sampled per image. Real sensors have persistent spatial structure: pixel-response non-uniformity (`PRNU`), dark-signal non-uniformity (`DSNU`), fixed row/column structure, and stable hot/dead pixels. Without persistence, the images can have plausible noise magnitude but weak camera identity.

**Where to patch**

- `euler_preprocess/fog/capture.py`
  - `SensorStage`
  - `_apply_np()` after `raw_signal` is built and before Poisson sampling
  - `_apply_torch()` equivalent
  - `_apply_bad_pixels_np()` / `_apply_bad_pixels_torch()` or their call sites

**Implementation strategy**

Add a small cache to `SensorStage`. The cache key should include:

```text
(sensor_id, pattern, height, width, base_seed)
```

Proposed config:

```json
"sensor_identity": {
  "enabled": true,
  "sensor_id": "dashcam_front_01",
  "seed": 12345,
  "prnu_sigma": {"dist": "uniform", "min": 0.001, "max": 0.006},
  "dsnu_sigma": {"dist": "uniform", "min": 0.00002, "max": 0.0003},
  "persistent_hot_pixel_probability": 0.00001,
  "persistent_dead_pixel_probability": 0.000005,
  "persistent_row_sigma": 0.0004,
  "persistent_column_sigma": 0.00025
}
```

Apply the maps as follows:

- `PRNU`: multiplicative on the electron-generating signal before Poisson sampling.
- `DSNU`: additive around black level before ADC clipping.
- persistent row/column: additive raw bias, distinct from the existing temporal `row_noise_sigma` and `column_noise_sigma`.
- persistent hot/dead pixels: stable masks generated once per sensor identity, optionally combined with the existing per-image hot/dead pixel probabilities.

**Expected effect**

Noise becomes sensor-like rather than purely image-like. This is especially important if the dataset is intended to resemble dashcam or surveillance footage where fixed pattern and compression are visible across frames.

### Patch P1.2: Keep ISO handling, but make the analog/digital split explicit

**Problem**

The current `iso` model affects `_resolve_electron_capacity()`: higher ISO reduces effective capacity, which increases relative shot noise and clipping pressure. Auto exposure can also update `iso` when `resolve_iso` is enabled. This is already useful. The missing realism is that real pipelines often separate exposure time, analog gain, and digital gain. These have different effects on read noise, ADC clipping, quantization, and post-ADC amplification.

**Where to patch**

- `euler_preprocess/fog/capture.py`
  - `_resolve_auto_exposure_np()`
  - `_resolve_auto_exposure_torch()`
  - `_resolve_electron_capacity()`
  - `_resolve_electron_capacity_torch()`
  - `SensorStage._apply_np()` / `_apply_torch()` around exposure, ADC, and post-ADC scaling

**Proposed config**

```json
"gain_model": {
  "enabled": true,
  "base_iso": 100,
  "analog_iso_max": 800,
  "digital_iso_max": 3200,
  "prefer_exposure_time_until_gain": 1.4,
  "digital_gain_noise_floor": 0.0005
}
```

**Implementation strategy**

Resolve a gain decomposition:

```text
requested_iso = config["iso"]
analog_gain = min(requested_iso / base_iso, analog_iso_max / base_iso)
digital_gain = max((requested_iso / base_iso) / analog_gain, 1.0)
```

Then:

- apply analog behavior before ADC / clipping;
- keep read-noise behavior tied to the analog side;
- apply digital gain after ADC quantization and before ISP;
- optionally increase visible quantization/read floor under high digital gain.

**Expected effect**

High-ISO gloomy scenes become more realistic: shadows can be noisy and color-corrupted without all noise sources scaling identically.

## P1: Improve camera-profile correlations instead of independent random knobs

### Patch P1.3: Add correlated gloomy scenario profiles

**Problem**

The repo already supports `scenario_profiles`, `camera_profiles`, and per-stage `condition_profiles`. The next realism improvement is not only adding new effects, but making existing effects co-vary. Dense gloomy fog should tend to have lower `scene_illumination`, different `airlight_dampening`, altered AE targets, higher ISO, stronger shadow color noise, more veiling glare/windshield haze, and stronger JPEG artifacts depending on the target camera.

**Where to patch**

- Config files and README examples first.
- No core code change may be needed beyond the new keys proposed above.

**Example scenario direction**

```json
"scenario_profiles": [
  {
    "name": "gloomy_dense_fog_dashcam",
    "weight": 1.0,
    "model": "heterogeneous_k_ls",
    "model_overrides": {
      "visibility_m": {"dist": "uniform", "min": 25.0, "max": 70.0},
      "airlight_dampening": {"enabled": true, "min_factor": 0.35, "max_factor": 0.85},
      "scene_illumination": {
        "enabled": true,
        "global_ev": {"dist": "uniform", "min": 0.25, "max": 0.85},
        "near_ev": {"dist": "uniform", "min": 0.35, "max": 1.20},
        "near_decay_depth_m": {"dist": "uniform", "min": 10.0, "max": 22.0},
        "fog_coupled_ev": {"dist": "uniform", "min": 0.10, "max": 0.45}
      }
    },
    "camera_profile": "low_light_fog",
    "capture_overrides": {
      "sensor": {
        "auto_exposure": {
          "enabled": true,
          "metering": "fog_aware_center_weighted",
          "manual_gain_weight": 0.0,
          "target_luminance": {"dist": "uniform", "min": 0.13, "max": 0.20},
          "sky_suppression": 0.85,
          "fog_meter_suppression": 0.65,
          "resolve_iso": true,
          "fog_iso_boost": 0.25
        },
        "shadow_recovery_noise": {
          "enabled": true,
          "chroma_sigma": {"dist": "uniform", "min": 0.010, "max": 0.030},
          "fog_weight": 0.25
        }
      },
      "transport": {
        "jpeg": {"enabled": true, "quality": {"dist": "uniform", "min": 58, "max": 86}},
        "bit_depth": 8
      }
    }
  }
]
```

**Expected effect**

The dataset stops looking like a Cartesian product of independent augmentations and starts looking like coherent camera/weather conditions.

## P2: Improve perception with cheap ISP and transport patches

### Patch P2.1: Add `tone_map: "lut"` or profile-specific CRF curves

**Problem**

`ISPStage` already has `tone_map`, `tone_map_strength`, `gamma`, denoise, local contrast, sharpening, and saturation. This is good, but a generic `reinhard` curve plus sRGB gamma will not match many real cameras. Camera response and tone mapping strongly affect whether fog looks photographic or synthetic.

**Where to patch**

- `euler_preprocess/fog/capture.py`
  - `ISPStage.DEFAULTS`
  - `_apply_tone_map_np()` / `_apply_tone_map_torch()` if present, or the tone-map call site in `ISPStage._apply_np()` / `_apply_torch()`

**Proposed config**

```json
"isp": {
  "tone_map": "lut",
  "tone_map_lut": [0.0, 0.006, 0.014, 0.028, 0.052, 0.090, 0.145, 0.220, 0.320, 0.450, 0.610, 0.780, 0.900, 0.965, 0.995, 1.0],
  "tone_map_lut_domain": "linear",
  "gamma": "srgb"
}
```

**Implementation strategy**

Use a small 1D LUT with linear interpolation. This is cheap and works for both CPU and torch. Keep the existing `reinhard` mode for backward compatibility.

**Expected effect**

Better highlight roll-off, less synthetic midtone behavior, and more plausible gloomy scenes after AE.

### Patch P2.2: Keep `TransportStage`, but tune it per camera/scenario

**Problem**

`TransportStage` already exists. The patch is not to add it, but to make sure scenario profiles use it deliberately. For dashcam-like data, JPEG quality, chroma subsampling, final bit depth, and resizing are often as visible as sensor noise.

**Where to patch**

- Config profiles first.
- `euler_preprocess/fog/capture.py` only if extra options are needed.

**Potential additions**

- expose named JPEG presets such as `"dashcam_low_bitrate"`;
- add `pre_jpeg_resize_scale` for cheap low-bitrate/video-like degradation;
- optionally separate `quantize_before_jpeg` and `quantize_after_jpeg` if experiments show the order matters.

**Expected effect**

The final RGB becomes closer to practical downstream training data, especially for camera/video datasets.

### Patch P2.3: Couple bloom and veiling glare to fog/airlight

**Problem**

`OpticsStage` already has `bloom`, `veiling_glare_strength`, `windshield_haze`, and droplets. However, glare and bloom are currently mostly sampled independently. In foggy backlit or headlight scenes, veiling glare should correlate with bright airlight, dense fog, and local highlights.

**Where to patch**

- `euler_preprocess/fog/capture.py`
  - `OpticsStage._apply_np()` / `_apply_torch()`
  - `_apply_bloom_np()` / `_apply_bloom_torch()` if needed

**Proposed config**

```json
"optics": {
  "fog_coupled_glare": {
    "enabled": true,
    "base_strength": 0.0,
    "fog_strength": 0.06,
    "highlight_strength": 0.08,
    "airlight_strength": 0.04,
    "smooth_sigma": 16.0
  }
}
```

**Expected effect**

Dense fog gains plausible low-frequency flare/veil without increasing physical fog density or incorrectly modifying labels.

## P3: Optional low-cost heterogeneous path integration

**Problem**

The current heterogeneous models use 2D `k_map` and `ls_map`. This is efficient and visually useful, but it is not a true line integral through a depth-varying volume. Fog texture can appear image-plane-attached rather than volume-consistent.

**Where to patch**

- `euler_preprocess/fog/models.py`
  - add `apply_fog_piecewise_np()` alongside `apply_fog()`.
  - enable through `model_cfg["path_integration"]`.

**Proposed config**

```json
"path_integration": {
  "enabled": false,
  "slices": 4,
  "lowres_factor": 4,
  "jitter_slices": true
}
```

**Approximation**

For each pixel, split the ray into a small number of depth intervals:

```text
T_total = product_i exp(-beta_i * delta_d_i)
I = J * T_total + sum_i L_s_i * (1 - exp(-beta_i * delta_d_i)) * product_{j<i} exp(-beta_j * delta_d_j)
```

Keep this disabled by default because P0/P1/P2 give more realism per millisecond.

## Validation plan

### Unit tests

Add tests before or together with the patches:

- `tests/test_scene_illumination.py`
  - `enabled: false` leaves RGB unchanged.
  - `near_ev > 0` darkens near pixels more than far pixels.
  - `sky_weight: 0` leaves sky pixels unchanged when `sky_mask` is supplied.
  - no NaNs for zero, infinite, or invalid depth after sanitization.

- `tests/test_fog_aware_auto_exposure.py`
  - increasing `sky_suppression` reduces the influence of sky pixels on `meter_luminance`.
  - increasing `fog_meter_suppression` reduces the influence of dense far-field fog on `meter_luminance`.
  - legacy `metering` modes remain numerically unchanged when suppression keys are absent.

- `tests/test_sensor_identity.py`
  - same `sensor_id`, shape, pattern, and seed produce identical persistent maps.
  - different `sensor_id` or seed changes persistent maps.
  - persistent hot/dead pixels are stable across images.

- `tests/test_transport_profiles.py`
  - `TransportStage` still performs JPEG round-trip when enabled.
  - lower JPEG quality produces stronger degradation than higher quality on a structured test image.

### Visual validation set

Use a small fixed validation subset with depth, sky mask, and semantic labels:

1. urban foreground with bright nearby surfaces;
2. no-sky street canyon where `from_sky` is weak or unavailable;
3. dense gloomy fog with far-field buildings;
4. low-light/high-ISO scene;
5. dashcam-like scene with signs, text, lane markings, and sharp edges.

Compare before/after outputs by depth band:

```text
near:   0-10 m
middle: 10-40 m
far:    >40 m
```

Track median linear luminance per band. The P0 patches should reduce near-band luminance in gloomy profiles without destroying far-field fog behavior.

### Performance validation

For each patch, report runtime overhead on representative image sizes:

- P0.1 linearization: one extra full-image pass.
- P0.2 scene illumination: one `exp`, one `exp2`, and a few multiplies per pixel.
- P0.3 fog-aware AE: cheap if using downsampled luminance or weighted histograms.
- P1 persistent sensor maps: near-zero per image after cache creation.
- P2 LUT tone map: near-zero overhead.
- P3 path integration: measurable overhead; keep disabled unless needed.

## Suggested implementation order

### PR 1: Radiometric and gloomy-scene fix

Files:

- `euler_preprocess/fog/pipeline.py`
- `euler_preprocess/fog/models.py`
- optional `euler_preprocess/common/color.py`
- tests for linearization and `scene_illumination`

Deliverables:

- `render_input_space`
- `scene_illumination`
- `sky_mask` passed into `apply_model()`
- no changes to labels or auxiliary map semantics except optional new debug output such as `illumination_ev_map`

### PR 2: Fog-aware auto exposure

Files:

- `euler_preprocess/fog/pipeline.py`
- `euler_preprocess/fog/capture.py`
- tests for weighted AE behavior

Deliverables:

- `sky_mask` in `CaptureContext.attributes`
- new AE metering modes
- weighted mean/percentile helpers
- updated README config example

### PR 3: Sensor identity persistence

Files:

- `euler_preprocess/fog/capture.py`
- tests for deterministic persistent maps

Deliverables:

- `sensor_identity` config block
- PRNU/DSNU maps
- persistent hot/dead pixel masks
- persistent row/column bias option

### PR 4: Profile and ISP polish

Files:

- `euler_preprocess/fog/capture.py`
- README/config examples

Deliverables:

- `tone_map: "lut"`
- gloomy scenario profile examples
- better low-light/dashcam profile correlations
- optional fog-coupled glare

### PR 5: Optional path integration experiment

Files:

- `euler_preprocess/fog/models.py`
- benchmark script or tests

Deliverables:

- `path_integration.enabled`
- small-slice approximation
- runtime comparison against current `apply_fog()`

## Practical recommendation

Implement PR 1 and PR 2 first. They directly address the observed foreground-overbrightness problem and should have small runtime cost. After that, implement PR 3 if the generated data should resemble a consistent camera stream rather than independent still-image augmentations. PR 4 is cheap and likely improves visual realism noticeably. PR 5 is scientifically attractive but should remain optional because it has a worse realism-per-compute tradeoff than the exposure, illumination, and sensor-identity fixes.
