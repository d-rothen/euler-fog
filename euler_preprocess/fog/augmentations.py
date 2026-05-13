from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from euler_preprocess.common.sampling import deep_merge, format_value


_MODEL_OVERRIDE_KEYS = {
    "airlight_dampening",
    "airlight_damping",
    "airlight_intensity_dampening",
    "airlight_intensity_damping",
    "atmospheric_light",
    "contrast_threshold",
    "k_hetero",
    "ls_hetero",
}


@dataclass(frozen=True)
class FogAugmentationSpec:
    """One deterministic fog augmentation emitted for every input sample."""

    id: str
    model_name: str = "uniform"
    model_overrides: dict[str, Any] = field(default_factory=dict)
    airlight_method: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FogAugmentationConfig:
    """Parsed fog augmentation block."""

    specs: tuple[FogAugmentationSpec, ...] = ()
    file_id_hierarchy_name: str | None = "file_id"
    attribute_key: str = "fog_augmentation"


def parse_fog_augmentations(config: dict[str, Any]) -> FogAugmentationConfig:
    """Parse the optional ``augmentations`` block from a fog config.

    Supported shapes:

    * ``"augmentations": [{...}, {...}]`` for explicit variants.
    * ``"augmentations": {"visibility_m": [10, 20], "airlight_methods": [...]}``
      for a Cartesian product over the configured dimensions.

    When the block is absent, an empty config is returned and the legacy
    one-output-per-sample path stays active.
    """

    raw = config.get("augmentations")
    if raw is None:
        return FogAugmentationConfig()

    if isinstance(raw, list):
        specs = tuple(
            _parse_explicit_variant(entry, index=i)
            for i, entry in enumerate(raw)
        )
        return FogAugmentationConfig(specs=specs)

    if not isinstance(raw, dict):
        raise ValueError("augmentations must be an object or a list")

    if raw.get("enabled", True) is False:
        return FogAugmentationConfig()

    file_id_hierarchy_name = raw.get("file_id_hierarchy_name", "file_id")
    if file_id_hierarchy_name in ("", None):
        file_id_hierarchy_name = None
    elif not isinstance(file_id_hierarchy_name, str):
        raise ValueError("augmentations.file_id_hierarchy_name must be a string")

    attribute_key = raw.get("attribute_key", "fog_augmentation")
    if not isinstance(attribute_key, str) or not attribute_key:
        raise ValueError("augmentations.attribute_key must be a non-empty string")

    if "variants" in raw:
        variants = raw["variants"]
        if not isinstance(variants, list):
            raise ValueError("augmentations.variants must be a list")
        common_model_config = _dict_or_empty(raw.get("model_config"))
        specs = tuple(
            _parse_explicit_variant(
                entry,
                index=i,
                common_model_config=common_model_config,
            )
            for i, entry in enumerate(variants)
        )
    else:
        specs = tuple(_expand_matrix(raw))

    if not specs:
        raise ValueError("augmentations must define at least one variant")

    return FogAugmentationConfig(
        specs=specs,
        file_id_hierarchy_name=file_id_hierarchy_name,
        attribute_key=attribute_key,
    )


def _expand_matrix(raw: dict[str, Any]) -> list[FogAugmentationSpec]:
    models = _as_options(raw.get("models", raw.get("model", "uniform")))
    visibility_values = _as_options(
        raw.get("visibility_m", raw.get("mor_m")),
    )
    beta_values = _as_options(
        raw.get("scattering_coefficients", raw.get("scattering_coefficient", raw.get("beta"))),
    )
    airlight_methods = _as_options(
        raw.get("airlight_methods", raw.get("airlight_method")),
    )
    atmospheric_lights = _as_options(
        raw.get("atmospheric_lights", raw.get("atmospheric_light")),
        rgb_triplet_is_single=True,
    )
    common_model_config = _dict_or_empty(raw.get("model_config"))
    common_attributes = _dict_or_empty(raw.get("attributes"))

    specs: list[FogAugmentationSpec] = []
    for index, (model, visibility, beta, airlight_method, atmospheric_light) in enumerate(
        product(
            models,
            visibility_values,
            beta_values,
            airlight_methods,
            atmospheric_lights,
        )
    ):
        variant: dict[str, Any] = {
            "model": model,
            "attributes": dict(common_attributes),
        }
        if visibility is not None:
            variant["visibility_m"] = visibility
        if beta is not None:
            variant["scattering_coefficient"] = beta
        if airlight_method is not None:
            variant["airlight_method"] = airlight_method
        if atmospheric_light is not None:
            variant["atmospheric_light"] = atmospheric_light
        specs.append(
            _parse_explicit_variant(
                variant,
                index=index,
                common_model_config=common_model_config,
            )
        )
    return specs


def _parse_explicit_variant(
    entry: Any,
    *,
    index: int,
    common_model_config: dict[str, Any] | None = None,
) -> FogAugmentationSpec:
    if not isinstance(entry, dict):
        raise ValueError("Each fog augmentation variant must be an object")

    model_name = str(entry.get("model", entry.get("model_name", "uniform")))
    model_overrides = deep_merge(
        dict(common_model_config or {}),
        _dict_or_empty(entry.get("model_config")),
    )

    if "mor_m" in entry and "visibility_m" in entry:
        raise ValueError(
            "Fog augmentation variants cannot set both mor_m and visibility_m"
        )

    if "mor_m" in entry:
        model_overrides["visibility_m"] = _constant_if_number(entry["mor_m"])
    if "visibility_m" in entry:
        model_overrides["visibility_m"] = _constant_if_number(entry["visibility_m"])
    if "scattering_coefficient" in entry and "beta" in entry:
        raise ValueError(
            "Fog augmentation variants cannot set both scattering_coefficient and beta"
        )
    if "scattering_coefficient" in entry:
        model_overrides["scattering_coefficient"] = _constant_if_number(
            entry["scattering_coefficient"]
        )
    if "beta" in entry:
        model_overrides["scattering_coefficient"] = _constant_if_number(entry["beta"])

    for key in _MODEL_OVERRIDE_KEYS:
        if key in entry:
            model_overrides[key] = entry[key]

    airlight_method = entry.get("airlight_method")
    if airlight_method is not None and not isinstance(airlight_method, str):
        raise ValueError("Fog augmentation airlight_method must be a string")

    attributes = _dict_or_empty(entry.get("attributes"))
    descriptor = _descriptor_from_variant(
        entry,
        model_name=model_name,
        airlight_method=airlight_method,
    )
    attributes = {**descriptor, **attributes}

    raw_id = entry.get("id", entry.get("name"))
    aug_id = _sanitize_identifier(str(raw_id)) if raw_id else _generate_id(
        entry,
        model_name=model_name,
        index=index,
        airlight_method=airlight_method,
    )

    return FogAugmentationSpec(
        id=aug_id,
        model_name=model_name,
        model_overrides=model_overrides,
        airlight_method=airlight_method,
        attributes=attributes,
    )


def _descriptor_from_variant(
    entry: dict[str, Any],
    *,
    model_name: str,
    airlight_method: str | None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"model": model_name}
    if "mor_m" in entry:
        descriptor["meteorological_visibility_m"] = entry["mor_m"]
    elif "visibility_m" in entry:
        descriptor["meteorological_visibility_m"] = entry["visibility_m"]
    if "scattering_coefficient" in entry:
        descriptor["configured_scattering_coefficient"] = entry["scattering_coefficient"]
    elif "beta" in entry:
        descriptor["configured_scattering_coefficient"] = entry["beta"]
    if airlight_method is not None:
        descriptor["airlight_method"] = airlight_method
    if "atmospheric_light" in entry:
        descriptor["configured_atmospheric_light"] = entry["atmospheric_light"]
    return descriptor


def _generate_id(
    entry: dict[str, Any],
    *,
    model_name: str,
    index: int,
    airlight_method: str | None,
) -> str:
    parts: list[str] = []
    if model_name != "uniform":
        parts.append(model_name)
    if "mor_m" in entry:
        parts.append(f"mor_{_format_id_value(entry['mor_m'])}m")
    elif "visibility_m" in entry:
        parts.append(f"mor_{_format_id_value(entry['visibility_m'])}m")
    elif "scattering_coefficient" in entry:
        parts.append(f"beta_{_format_id_value(entry['scattering_coefficient'])}")
    elif "beta" in entry:
        parts.append(f"beta_{_format_id_value(entry['beta'])}")
    if airlight_method:
        parts.append(f"airlight_{airlight_method}")
    elif "atmospheric_light" in entry and isinstance(entry["atmospheric_light"], str):
        parts.append(f"airlight_{entry['atmospheric_light']}")
    if not parts:
        parts.append(f"aug_{index:03d}")
    return _sanitize_identifier("_".join(parts))


def _format_id_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return format_value(float(value))
    if isinstance(value, dict) and value.get("dist") == "constant":
        inner = value.get("value")
        if isinstance(inner, (int, float)):
            return format_value(float(inner))
    return _sanitize_identifier(str(value))


def _sanitize_identifier(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "augmentation"


def _constant_if_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return {"dist": "constant", "value": float(value)}
    return value


def _as_options(value: Any, *, rgb_triplet_is_single: bool = False) -> list[Any]:
    if value is None:
        return [None]
    if rgb_triplet_is_single and _is_rgb_triplet(value):
        return [value]
    if isinstance(value, list):
        return value
    return [value]


def _is_rgb_triplet(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) for item in value)
    )


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return dict(value)
