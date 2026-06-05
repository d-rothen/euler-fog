from __future__ import annotations

import numpy as np

from euler_preprocess.fog.capture import CaptureArtifactPipeline, CaptureContext


def test_isp_tone_map_lut_interpolates_1d_curve() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "isp",
                        "denoise_sigma": 0.0,
                        "color_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "tone_map": "lut",
                        "tone_map_strength": 1.0,
                        "tone_map_lut": [0.0, 0.25, 1.0],
                        "gamma": "linear",
                        "local_contrast_strength": 0.0,
                        "sharpen_amount": 0.0,
                        "saturation": 1.0,
                    }
                ]
            }
        }
    )
    image = np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32)

    result = pipeline.apply_np(image, CaptureContext(rng=np.random.default_rng(9)))

    np.testing.assert_allclose(result, [[[0.0, 0.25, 1.0]]], atol=1e-6)
