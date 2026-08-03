from __future__ import annotations

import logging
from typing import Any

import numpy as np

from euler_preprocess.common.sampling import sample_value
from euler_preprocess.fog.airlight_from_sky import AirlightFromSky
from euler_preprocess.fog.dcp_airlight import DCPAirlight
from euler_preprocess.fog.dcp_heuristic_airlight import DCPHeuristicAirlight
from euler_preprocess.fog.models import (
    AIRLIGHT_METHODS,
    dampen_airlight_torch,
    estimate_airlight_torch,
    normalize_atmospheric_light_torch,
    resolve_scattering_coefficient,
    uses_estimated_airlight,
)

try:
    import torch
except ImportError:
    torch = None


class AtmosphericLightResolver:
    """Resolve fog atmospheric light estimates and model-level torch L_s values."""

    def __init__(
        self,
        method: str,
        *,
        dcp_heuristic_config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if method not in AIRLIGHT_METHODS:
            raise ValueError(
                f"Unknown airlight method '{method}'. Supported values: "
                f"{AIRLIGHT_METHODS}"
            )
        self.method = method
        self.logger = logger or logging.getLogger("euler-preprocess.fog")
        dcp_heuristic_config = (
            {} if dcp_heuristic_config is None else dcp_heuristic_config
        )
        self.dcp_heuristic_kwargs = self._parse_dcp_heuristic_kwargs(
            dcp_heuristic_config
        )
        self._estimators: dict[str, Any] = {}
        self._estimators_torch: dict[str, Any] = {}
        self.estimator = self._get_estimator(method)
        self.estimator_torch = self._get_estimator_torch(method)

    @staticmethod
    def _parse_dcp_heuristic_kwargs(config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("Config key 'dcp_heuristic' must be an object")
        return {
            key: config[key]
            for key in (
                "patch_size",
                "top_percent",
                "white_bias",
                "cool_bias",
                "cool_target",
            )
            if key in config
        }

    def estimate_np(
        self,
        rgb: np.ndarray,
        sky_mask: np.ndarray,
        *,
        sample_id: str | None,
        method: str | None = None,
    ) -> np.ndarray:
        resolved_method = method or self.method
        estimator = self._get_estimator(resolved_method)
        return estimator.estimate_airlight(rgb, sky_mask, sample_id=sample_id)

    def estimate_torch(
        self,
        rgb_t: "torch.Tensor",
        sky_mask_t: "torch.Tensor",
        *,
        sample_id: str | None,
        method: str | None = None,
    ) -> "torch.Tensor":
        resolved_method = method or self.method
        if resolved_method == "from_sky":
            return estimate_airlight_torch(rgb_t, sky_mask_t, sample_id=sample_id)
        estimator = self._get_estimator_torch(resolved_method)
        if estimator is None:
            raise RuntimeError(
                f"Torch airlight estimator unavailable for method "
                f"'{resolved_method}'."
            )
        if resolved_method == "dcp":
            return estimator.compute(rgb_t)
        return estimator.estimate_airlight(
            rgb_t,
            sky_mask_t,
            sample_id=sample_id,
        )

    def resolve_model_torch(
        self,
        *,
        model_cfg: dict,
        rng: np.random.Generator,
        contrast_threshold_default: float,
        estimated_airlight_t: "torch.Tensor",
        device: Any,
    ) -> tuple[float, "torch.Tensor"]:
        """Resolve beta and dampened base atmospheric light for one torch sample."""
        k_mean, _visibility, contrast_threshold = resolve_scattering_coefficient(
            model_cfg,
            rng,
            contrast_threshold_default,
        )
        al_spec = model_cfg.get("atmospheric_light", "from_sky")
        airlight_is_estimated = uses_estimated_airlight(al_spec)
        if airlight_is_estimated:
            raw_ls_base = normalize_atmospheric_light_torch(
                estimated_airlight_t
            ).squeeze(0)
        else:
            sampled_al = sample_value(al_spec, rng)
            raw_ls_base = normalize_atmospheric_light_torch(
                torch.tensor(sampled_al, device=device, dtype=torch.float32)
            ).squeeze(0)

        ls_base = dampen_airlight_torch(
            raw_ls_base,
            k_mean,
            model_cfg,
            rng,
            contrast_threshold,
            estimated_airlight=airlight_is_estimated,
        )
        return k_mean, ls_base

    def resolve_uniform_batch_torch(
        self,
        *,
        rgb_batch: "torch.Tensor",
        items: list[dict],
        device: Any,
        contrast_threshold_default: float,
        method: str | None = None,
    ) -> tuple[list[float], "torch.Tensor"]:
        """Resolve beta and dampened base L_s for a uniform-model torch batch."""
        al_spec = items[0]["model_cfg"].get("atmospheric_light", "from_sky")
        airlight_is_estimated = uses_estimated_airlight(al_spec)
        if airlight_is_estimated:
            ls_base = self._estimate_batch_torch(rgb_batch, items, device, method)
            ls_base = normalize_atmospheric_light_torch(ls_base)
        else:
            ls_values = []
            for item in items:
                sampled_al = sample_value(al_spec, item["rng"])
                ls_values.append(
                    normalize_atmospheric_light_torch(
                        torch.tensor(
                            sampled_al,
                            device=device,
                            dtype=torch.float32,
                        )
                    ).squeeze(0)
                )
            ls_base = torch.stack(ls_values, dim=0)

        k_means: list[float] = []
        contrast_thresholds: list[float] = []
        for item in items:
            k_mean, _visibility, contrast_threshold = resolve_scattering_coefficient(
                item["model_cfg"],
                item["rng"],
                contrast_threshold_default,
            )
            k_means.append(k_mean)
            contrast_thresholds.append(contrast_threshold)

        ls_base = torch.stack(
            [
                dampen_airlight_torch(
                    ls_base[idx],
                    k_means[idx],
                    items[idx]["model_cfg"],
                    items[idx]["rng"],
                    contrast_thresholds[idx],
                    estimated_airlight=airlight_is_estimated,
                )
                for idx in range(len(items))
            ],
            dim=0,
        )
        return k_means, ls_base

    def _estimate_batch_torch(
        self,
        rgb_batch: "torch.Tensor",
        items: list[dict],
        device: Any,
        method: str | None = None,
    ) -> "torch.Tensor":
        resolved_method = method or self.method
        if resolved_method == "from_sky":
            return self._estimate_from_sky_batch_torch(rgb_batch, items, device)
        estimator = self._get_estimator_torch(resolved_method)
        if estimator is None:
            raise RuntimeError(
                f"Torch airlight estimator unavailable for method '{resolved_method}'."
            )
        al_list = []
        for idx, item in enumerate(items):
            if resolved_method == "dcp":
                al_list.append(estimator.compute(rgb_batch[idx]))
            else:
                sky_mask_t = torch.from_numpy(item["sky_mask"]).to(
                    device=device,
                    dtype=torch.bool,
                )
                al_list.append(
                    estimator.estimate_airlight(
                        rgb_batch[idx],
                        sky_mask_t,
                        sample_id=item["sample_id"],
                    )
                )
        return torch.stack(al_list, dim=0)

    def _estimate_from_sky_batch_torch(
        self,
        rgb_batch: "torch.Tensor",
        items: list[dict],
        device: Any,
    ) -> "torch.Tensor":
        sky_mask_batch = torch.stack(
            [
                torch.from_numpy(item["sky_mask"]).to(device)
                for item in items
            ],
            dim=0,
        ).to(torch.float32)
        mask_sum = sky_mask_batch.sum(dim=(1, 2))
        no_sky = mask_sum == 0
        safe_sum = mask_sum.clone()
        safe_sum[no_sky] = 1.0
        airlight = (rgb_batch * sky_mask_batch[..., None]).sum(dim=(1, 2)) / safe_sum[
            :, None
        ]
        if no_sky.any():
            for idx_ns in no_sky.nonzero(as_tuple=False):
                i = int(idx_ns.item())
                self.logger.warning(
                    "No sky pixels in segmentation mask (sample %s); using "
                    "default airlight fallback [1.0, 1.0, 1.0]",
                    items[i]["sample_id"],
                )
            airlight[no_sky] = 1.0
        return airlight

    def _get_estimator(self, method: str):
        estimator = self._estimators.get(method)
        if estimator is not None:
            return estimator
        if method == "from_sky":
            estimator = AirlightFromSky(sky_depth_threshold=0.0)
        elif method == "dcp":
            estimator = DCPAirlight()
        elif method == "dcp_heuristic":
            estimator = DCPHeuristicAirlight(**self.dcp_heuristic_kwargs)
        else:
            raise ValueError(
                f"Unknown airlight method '{method}'. Supported values: "
                f"{AIRLIGHT_METHODS}"
            )
        self._estimators[method] = estimator
        return estimator

    def _get_estimator_torch(self, method: str):
        estimator = self._estimators_torch.get(method)
        if estimator is not None:
            return estimator
        if torch is None:
            return None
        if method == "dcp":
            from euler_preprocess.fog.dcp_airlight_torch import DCPAirlightTorch

            estimator = DCPAirlightTorch()
        elif method == "dcp_heuristic":
            from euler_preprocess.fog.dcp_heuristic_airlight_torch import (
                DCPHeuristicAirlightTorch,
            )

            estimator = DCPHeuristicAirlightTorch(**self.dcp_heuristic_kwargs)
        else:
            return None
        self._estimators_torch[method] = estimator
        return estimator
