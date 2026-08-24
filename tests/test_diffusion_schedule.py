import unittest

import torch

from ieee33_diffusion.model import GaussianDiffusion, ModelConfig, TopologyConditionedDenoiser


class DiffusionScheduleTest(unittest.TestCase):
    def _model(self, schedule: str, steps: int) -> GaussianDiffusion:
        config = ModelConfig(
            hidden_channels=8,
            time_channels=8,
            num_layers=1,
            diffusion_steps=steps,
            noise_schedule=schedule,
        )
        return GaussianDiffusion(TopologyConditionedDenoiser(config))

    def test_cosine_schedule_reaches_nearly_gaussian_terminal_state(self) -> None:
        model = self._model("cosine", 100)
        self.assertLess(float(model.alpha_bars[-1]), 1.0e-5)
        self.assertTrue(torch.isfinite(model.betas).all())
        self.assertTrue(torch.all((model.betas > 0.0) & (model.betas < 1.0)))

    def test_short_linear_schedule_retains_substantial_signal(self) -> None:
        model = self._model("linear", 100)
        self.assertGreater(float(model.alpha_bars[-1]), 0.30)


if __name__ == "__main__":
    unittest.main()
