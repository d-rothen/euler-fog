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
| `fog` | `rgb`, `depth`, `semantic_segmentation` | — (intrinsics optional) |
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
  "device": "cpu",
  "gpu_batch_size": 4,
  "capture": { "stages": [] },
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
| `device` | `"cpu"`, `"cuda"`, `"mps"`, or `"gpu"` (alias for cuda). |
| `gpu_batch_size` | Batch size when running on GPU. Uniform-model samples are batched; heterogeneous samples are processed individually. |
| `capture` / `capture_artifacts` | Optional post-fog camera artifact pipeline. Currently only an empty/no-op stage list is supported; exposure, vignetting, and noise stages are reserved for future additions. |
| `augmentations` | Optional stepped augmentation set. When present, every input sample produces every configured augmentation and uses the file-id hierarchy output layout described below. |

### Processing Pipeline

Fog generation is split into two phases:

1. Ideal scene rendering: physics-based fog and auxiliary `scattering_coefficient`
   / `atmospheric_light` maps are computed.
2. Capture artifacts: camera-specific effects are applied to the rendered RGB
   only. This stage is currently a no-op, but it is the intended location for
   exposure reduction, vignetting, sensor noise, and similar capture effects.

This keeps physical fog maps stable while making the RGB output extensible for
real-camera simulation.

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

| Parameter | Effect |
|---|---|
| `min_factor` / `max_factor` | Range of the multiplicative factor. |
| `normalize_to_mean` | Rescale factors so the image-wide mean equals the base value. Recommended for `k_hetero`. |
| `scales: "smooth_auto"` | Build low-frequency Perlin scales from the image size. |
| `correlation_length_fraction` | Approximate smallest fog feature size as a fraction of the shorter image side. Larger values create smoother gradients. |
| `octaves` / `lacunarity` / `max_scale` | Control how many increasingly broad Perlin components are mixed. |
| `contrast` | Compress or expand the Perlin range before mapping to factors. Values below 1 are recommended. |
| `smooth_sigma` / `smooth_sigma_fraction` | Optional final Gaussian blur in pixels or as a fraction of the shorter image side. |

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
