import json
from pathlib import Path

from euler_preprocess.fog.capture import CaptureArtifactPipeline


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "dense_gloomy_daylight_fog_camera.json"
)

NOISE_GROUPS = {
    "read",
    "static",
    "banding",
    "bad_pixels",
    "chroma",
    "shadow_luma",
    "modulation",
}


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _profile_by_name(profiles: list[dict], name: str) -> dict:
    return next(profile for profile in profiles if profile["name"] == name)


def test_dense_gloomy_daylight_config_activates_realism_stack() -> None:
    config = _load_config()

    assert config["render_input_space"] == "srgb"
    CaptureArtifactPipeline.from_config(config)

    profile = config["camera_profiles"]["dense_gloomy_daylight_dashcam"]
    auto_exposure = profile["sensor"]["auto_exposure"]
    assert auto_exposure["metering"] == "fog_aware_center_weighted"
    for key in (
        "sky_suppression",
        "fog_meter_suppression",
        "depth_meter_decay_m",
        "min_meter_weight",
    ):
        assert key in auto_exposure

    assert profile["sensor"]["sensor_identity"]["enabled"] is True
    assert profile["optics"]["fog_coupled_glare"]["enabled"] is True
    assert profile["optics"]["droplets"]["enabled"] is True
    assert profile["isp"]["tone_map"] == "lut"
    assert profile["isp"]["tone_map_lut_domain"] == "linear"
    assert len(profile["isp"]["tone_map_lut"]) >= 2

    assert len(config["scenario_profiles"]) == 6
    clear = _profile_by_name(
        config["scenario_profiles"],
        "clear_weather_camera_reference",
    )
    assert clear["clear_weather"] is True
    assert clear["weight"] == 0.1
    assert clear["model"] == "uniform"
    high_visibility = _profile_by_name(
        config["scenario_profiles"],
        "clear_high_visibility_daylight_reference",
    )
    assert high_visibility.get("clear_weather", False) is False
    assert high_visibility["weight"] == 0.09
    assert high_visibility["model"] == "heterogeneous_k_ls"
    assert sum(scenario["weight"] for scenario in config["scenario_profiles"]) == 1.0
    underexposed_condition = _profile_by_name(
        profile["sensor"]["condition_profiles"],
        "underexposed_noisy",
    )
    low_light_chroma = underexposed_condition["shadow_recovery_noise"]
    assert low_light_chroma["red_chroma_gain"]["min"] >= 0.9
    assert low_light_chroma["blue_chroma_gain"]["max"] <= 2.0

    for scenario in config["scenario_profiles"]:
        model_name = scenario["model"]
        if not scenario.get("clear_weather", False):
            model_config = scenario["models"][model_name]
            assert model_config["scene_illumination"]["enabled"] is True

        capture = scenario["capture_overrides"]
        scenario_auto_exposure = capture["sensor"]["auto_exposure"]
        assert scenario_auto_exposure["metering"] == "fog_aware_center_weighted"
        for key in (
            "sky_suppression",
            "fog_meter_suppression",
            "depth_meter_decay_m",
            "min_meter_weight",
        ):
            assert key in scenario_auto_exposure

        noise_adjustment = capture["sensor"]["noise_adjustment"]
        assert noise_adjustment["enabled"] is True
        assert set(noise_adjustment["groups"]) == NOISE_GROUPS
        weather_optics_enabled = not scenario.get("clear_weather", False)
        assert capture["optics"]["fog_coupled_glare"]["enabled"] is weather_optics_enabled
        assert capture["optics"]["droplets"]["enabled"] is weather_optics_enabled
        assert "tone_map_strength" in capture["isp"]
        assert "quality" in capture["transport"]["jpeg"]

    for name in ("underexposed_dense_gloom", "severe_low_contrast_sensor_stress"):
        scenario = _profile_by_name(config["scenario_profiles"], name)
        sensor = scenario["capture_overrides"]["sensor"]
        white_balance = sensor["white_balance"]
        assert white_balance[0] > white_balance[2]
        assert sensor["noise_adjustment"]["static_chroma_bias"] <= 0.24
        assert sensor["noise_adjustment"]["groups"]["chroma"]["value"] <= 1.28
