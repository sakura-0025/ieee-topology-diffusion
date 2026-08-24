"""图扩散模型与物理损失的集成测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from ieee33_diffusion.model import (
    GaussianDiffusion,
    ModelConfig,
    TopologyConditionedDenoiser,
    TrainingLossConfig,
    VectorConditionedDenoiser,
)
from ieee33_diffusion.physics import PhysicsConfig, dataset_base_mva


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "strict_filter_check.npz"


class ModelPhysicsIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Missing test dataset: {DATASET_PATH}")
        cls.data = np.load(DATASET_PATH, allow_pickle=False)

    def test_joint_loss_is_finite_and_backpropagates(self) -> None:
        torch.manual_seed(2026)
        indices = np.array([0, 20])
        raw = self.data["x0"]
        train = raw[self.data["sample_split"] == 0]
        mean = train.mean(axis=(0, 1)).astype(np.float32)
        std = np.maximum(train.std(axis=(0, 1)), 1.0e-6).astype(np.float32)
        clean = torch.tensor((raw[indices] - mean) / std, dtype=torch.float32)
        topology_ids = self.data["sample_topology_id"][indices]
        active = self.data["topology_active_edge_ids"][topology_ids]
        edge_index = torch.tensor(
            np.stack([self.data["master_edge_index"][:, ids] for ids in active]),
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            np.stack([self.data["master_edge_attr"][ids] for ids in active]),
            dtype=torch.float32,
        )
        node_static = torch.tensor(self.data["node_static"], dtype=torch.float32)

        config = ModelConfig(
            hidden_channels=16,
            time_channels=16,
            num_layers=2,
            diffusion_steps=10,
        )
        model = GaussianDiffusion(TopologyConditionedDenoiser(config))
        components = model.training_losses(
            clean,
            node_static,
            edge_index,
            edge_attr,
            torch.from_numpy(mean),
            torch.from_numpy(std),
            TrainingLossConfig(physics_weight=1.0e-4, voltage_weight=1.0),
            PhysicsConfig(base_mva=dataset_base_mva(self.data)),
            torch.tensor(self.data["topology_edge_mask"][topology_ids], dtype=torch.float32),
        )
        self.assertEqual(set(components), {"total", "noise", "physics", "voltage"})
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        self.assertGreater(float(components["physics"].detach()), 0.0)
        components["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_vector_baseline_uses_topology_mask(self) -> None:
        torch.manual_seed(2026)
        config = ModelConfig(
            hidden_channels=16,
            time_channels=16,
            num_layers=2,
            diffusion_steps=10,
            model_type="vector",
        )
        model = GaussianDiffusion(VectorConditionedDenoiser(config))
        clean = torch.randn(2, 33, 4)
        node_static = torch.tensor(self.data["node_static"], dtype=torch.float32)
        topology_ids = np.array([0, 1])
        active = self.data["topology_active_edge_ids"][topology_ids]
        edge_index = torch.tensor(
            np.stack([self.data["master_edge_index"][:, ids] for ids in active]),
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            np.stack([self.data["master_edge_attr"][ids] for ids in active]),
            dtype=torch.float32,
        )
        masks = torch.tensor(
            self.data["topology_edge_mask"][topology_ids], dtype=torch.float32
        )
        loss = model.training_loss(
            clean, node_static, edge_index, edge_attr, topology_mask=masks
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
