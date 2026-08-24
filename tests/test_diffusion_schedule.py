import unittest

import torch

from ieee33_diffusion.model import (
    GaussianDiffusion,
    ModelConfig,
    TopologyConditionedDenoiser,
    TrainingLossConfig,
)
from ieee33_diffusion.physics import PhysicsConfig


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

    def test_high_alpha_threshold_can_gate_out_physics_loss(self) -> None:
        model = self._model("linear", 10)
        batch = 2
        clean = torch.randn(batch, 33, 4)
        node_static = torch.zeros(33, 4)
        edge_index = torch.zeros(batch, 2, 32, dtype=torch.long)
        edge_attr = torch.zeros(batch, 32, 4)
        losses = model.training_losses(
            clean,
            node_static,
            edge_index,
            edge_attr,
            torch.zeros(4),
            torch.ones(4),
            TrainingLossConfig(
                physics_weight=1.0,
                physics_alpha_bar_min=0.99999,
            ),
            PhysicsConfig(base_mva=10.0),
            torch.zeros(batch, 37),
        )
        self.assertEqual(float(losses["physics"]), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))


if __name__ == "__main__":
    unittest.main()
