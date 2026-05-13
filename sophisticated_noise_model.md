# Toward a More Sophisticated Fog and Camera Noise Model

This note records the current state of the fog/camera corruption model and the
remaining places where the approach is still intentionally approximate.

## Current State

The pipeline now separates ideal scene fog rendering from capture artifacts:

1. The fog model renders RGB using Koschmieder-style transmittance, spatial
   scattering coefficient maps, and spatial atmospheric-light maps.
2. The capture stack then applies camera effects such as optics, exposure, raw
   sensor noise, demosaicing, ISP, and compression.

This is a useful structure because physical auxiliary outputs
(`scattering_coefficient`, `atmospheric_light`) remain tied to the fog model,
while the RGB image can receive camera-specific degradation.

The current implementation includes:

- heterogeneous `k` and `L_s` fields;
- dampened atmospheric light for dense fog;
- optional top-to-bottom `L_s` illumination gradients;
- configurable camera profiles;
- intrinsics-aware optics where relevant;
- heteroscedastic sensor noise that can depend on darkness, distance, and fog
  opacity;
- smoothed row/column banding instead of pixel-independent striping;
- depth/fog-weighted chromatic fringing;
- per-stage `condition_profiles`, currently used for coherent sensor exposure
  states in the dense gloomy daylight config;
- top-level `scenario_profiles` that sample one latent scene/camera condition
  and use it to correlate fog density, atmospheric light behavior, camera
  capture overrides, sensor condition profiles, ISP, and compression.

## Remaining Naive Assumptions

### 1. Condition Sampling Is Now Present, But Still Needs Calibration

Fog density, atmospheric light, `L_s` gradient, exposure state, ISO/noise, and
ISP behavior can now be correlated through top-level `scenario_profiles`.
However, the profile weights and parameter ranges are still hand-authored. They
should eventually be calibrated against the target dataset distribution.

Without calibration, physically inconsistent combinations are still possible if
new scenarios are authored carelessly, for example:

- dense gloomy fog with unusually clean low-ISO sensor settings;
- clear haze with overly aggressive denoising and compression artifacts;
- strong top-down airlight gradients paired with weak fog opacity;
- high ISO without corresponding exposure under-shoot or metering pressure.

The current dense fog config includes scenarios in this style:

- `clear_day_haze`;
- `moderate_day_fog`;
- `dense_gloomy_daylight`;
- `underexposed_dense_fog`;
- `clean_low_noise_reference`.

Each scenario can drive fog model parameters, atmospheric light behavior,
capture profile overrides, exposure behavior, sensor noise, ISP, and transport
quality together.

### 2. Exposure Should Become Image-Driven

Current exposure sampling is useful for dataset diversity, but it is not yet a
camera metering model. A more realistic camera stack should choose exposure from
the rendered foggy image and camera metering assumptions.

Useful controls would include:

- metering mode: center-weighted, average, sky-suppressed, highlight-protecting;
- target middle-gray luminance;
- exposure compensation;
- highlight clipping tolerance;
- allowed shutter/aperture/ISO range;
- ISO escalation after shutter/aperture constraints are hit;
- optional auto white balance from the rendered image.

This would let clean and noisy cases arise naturally. Clear, bright samples
would often remain low ISO and low noise, while dense gloomy fog would push the
camera toward higher ISO, lower exposure, more read noise, and more denoising.

### 3. Heterogeneous Fog Is Image-Space, Not World-Space

The current heterogeneous `k` and `L_s` fields are smooth image-space fields.
For single images this is often sufficient, especially for qualitative realism.
For temporal data, stereo, or multi-camera rigs, it is still too naive.

A fuller model would sample a scene- or world-space fog density field and project
it into each camera. This would make fog structures consistent across frames and
cameras. A practical intermediate step would be a per-scene latent field keyed by
scene ID, camera pose, and timestamp, without requiring full volumetric ray
marching.

### 4. Camera Profiles Lack Full Calibration

The current `camera_profile` mechanism is useful, but it is mostly authored by
hand. If richer calibration is available, the profile should consume it.

Potential additions:

- lens distortion coefficients beyond the current simplified radial model;
- sensor size and pixel pitch;
- calibrated vignetting maps;
- measured camera response curves;
- per-camera color correction matrices;
- per-camera white balance priors;
- rolling shutter and motion metadata;
- compression/ISP profiles per camera stream.

Intrinsics are now assumed to be available and are already used where they are
useful. Extrinsics, distortion, and response metadata would be the next major
fidelity step.

### 5. Capture Artifacts Are Not Yet Efficient Enough

The fog model can use GPU paths, but the full capture stack still relies heavily
on CPU/numpy image processing. This is acceptable for small qualitative testing,
but it may become a bottleneck for giant datasets.

If throughput matters, the highest-value optimization is to port common capture
stages to batched torch operations:

- exposure and white balance;
- vignetting;
- chromatic aberration/fringing;
- shot/read noise;
- fixed-pattern noise;
- smoothed banding;
- simple ISP operations;
- quantization.

JPEG round trips and some PIL-style image operations may remain CPU-bound unless
there is a clear need to replace them.

## Recommended Next Implementation

The next full refinement should add image-driven auto-exposure on top of the
scenario profiles. Scenario sampling handles global correlation; auto-exposure
would make per-image exposure and ISO respond to actual rendered luminance.

### Proposed Config Shape

```json
{
  "scenario_profiles": [
    {
      "name": "clean_low_noise_reference",
      "weight": 0.2,
      "model": "heterogeneous_k_ls",
      "models": {
        "heterogeneous_k_ls": {
          "visibility_m": {"dist": "uniform", "min": 120.0, "max": 260.0}
        }
      },
      "capture_overrides": {
        "sensor": {
          "condition_profile": "clean_daylight"
        },
        "isp": {
          "denoise_sigma": {"dist": "uniform", "min": 0.05, "max": 0.25}
        }
      }
    },
    {
      "name": "underexposed_dense_fog",
      "weight": 0.25,
      "model": "heterogeneous_k_ls",
      "models": {
        "heterogeneous_k_ls": {
          "visibility_m": {"dist": "uniform", "min": 22.0, "max": 55.0}
        }
      },
      "capture_overrides": {
        "sensor": {
          "condition_profile": "underexposed_noisy"
        }
      }
    }
  ]
}
```

The exact schema should follow existing configuration conventions, but the key
idea is that one sampled scenario controls multiple downstream blocks.

### Auto-Exposure Stage

A camera-like exposure resolver could run between ideal fog rendering and raw
sensor simulation:

1. Render fog in linear or display-normalized RGB.
2. Compute luminance statistics with configurable metering.
3. Choose exposure compensation to hit the target luminance.
4. Constrain exposure by highlight protection.
5. Resolve ISO/noise state from the required gain.
6. Pass the resolved exposure and ISO into the sensor stage.

This should replace purely random exposure gain where possible, while still
allowing sampled exposure compensation and metering style for diversity.

## Bias Controls

Any stronger model should avoid deterministic shortcuts. In particular:

- keep `L_s` gradients probabilistic and weak;
- keep clean/noise-free cases in the sampling distribution;
- ensure dense fog does not always imply the same exposure/noise signature;
- ensure top-of-image brightening is not always present;
- randomize metering behavior and exposure compensation within plausible bounds;
- preserve auxiliary physical maps independently from capture artifacts.

The goal is not to simulate every camera perfectly. The goal is to produce a
diverse, physically plausible distribution that prevents the model from learning
simple image-position or corruption-level shortcuts.
