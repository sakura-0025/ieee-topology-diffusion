import unittest

import numpy as np

from ieee33_diffusion.evaluate import _mmd_rbf


class EvaluationMetricsTest(unittest.TestCase):
    def test_real_reference_bandwidth_detects_scale_explosion(self) -> None:
        rng = np.random.default_rng(2026)
        real = rng.normal(size=(128, 3, 2))
        close = real + rng.normal(scale=0.01, size=real.shape)
        exploded = rng.normal(scale=50.0, size=real.shape)
        close_result = _mmd_rbf(real, close, max_samples=128, seed=2026)
        exploded_result = _mmd_rbf(real, exploded, max_samples=128, seed=2026)
        self.assertEqual(close_result["rbf_bandwidth_source"], "real_reference")
        self.assertEqual(
            close_result["rbf_bandwidth_squared"],
            exploded_result["rbf_bandwidth_squared"],
        )
        self.assertGreater(
            exploded_result["mmd_rbf_squared"],
            close_result["mmd_rbf_squared"] + 0.5,
        )


if __name__ == "__main__":
    unittest.main()
