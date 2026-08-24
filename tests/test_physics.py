"""交流潮流物理模块的数值与梯度测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from ieee33_diffusion.physics import (
    ac_power_flow_residual,
    build_ybus,
    calculated_power_injections,
    dataset_base_mva,
    power_flow_residual_loss,
    voltage_limit_loss,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "strict_filter_check.npz"


class SyntheticPhysicsTest(unittest.TestCase):
    """不依赖本地NPZ文件的两节点解析一致性测试。"""

    def test_two_bus_state_reconstructs_exact_injections(self) -> None:
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_attr = torch.tensor([[0.01, 0.05, 0.0, 1.0]], dtype=torch.float64)
        voltage = torch.tensor([1.0, 0.97], dtype=torch.float64)
        theta = torch.tensor([0.0, -0.04], dtype=torch.float64)
        ybus = build_ybus(edge_index, edge_attr, num_nodes=2)[0]
        injections = calculated_power_injections(
            voltage, theta, ybus, base_mva=10.0
        )
        state = torch.cat(
            [injections, voltage.unsqueeze(-1), theta.unsqueeze(-1)], dim=-1
        )
        residual = ac_power_flow_residual(
            state, edge_index, edge_attr, base_mva=10.0
        )
        self.assertLess(float(residual.abs().max()), 1.0e-12)

    def test_two_bus_perturbation_is_detected(self) -> None:
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_attr = torch.tensor([[0.01, 0.05]], dtype=torch.float64)
        voltage = torch.tensor([1.0, 0.97], dtype=torch.float64)
        theta = torch.tensor([0.0, -0.04], dtype=torch.float64)
        ybus = build_ybus(edge_index, edge_attr, num_nodes=2)[0]
        injections = calculated_power_injections(
            voltage, theta, ybus, base_mva=10.0
        )
        state = torch.cat(
            [injections, voltage.unsqueeze(-1), theta.unsqueeze(-1)], dim=-1
        )
        state[1, 2] += 0.01
        residual = ac_power_flow_residual(
            state, edge_index, edge_attr, base_mva=10.0
        )
        self.assertGreater(float(residual.abs().max()), 0.1)


class PhysicsTest(unittest.TestCase):
    """使用真实 case33bw 潮流样本检查物理公式。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Missing test dataset: {DATASET_PATH}")
        cls.data = np.load(DATASET_PATH, allow_pickle=False)
        cls.base_mva = dataset_base_mva(cls.data)

    def _sample(self, sample_id: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topology_id = int(self.data["sample_topology_id"][sample_id])
        edge_ids = self.data["topology_active_edge_ids"][topology_id]
        state = torch.tensor(self.data["x0"][sample_id], dtype=torch.float64)
        edge_index = torch.tensor(
            self.data["master_edge_index"][:, edge_ids], dtype=torch.long
        )
        edge_attr = torch.tensor(
            self.data["master_edge_attr"][edge_ids], dtype=torch.float64
        )
        return state, edge_index, edge_attr

    def test_ybus_is_symmetric_and_rows_sum_to_zero(self) -> None:
        _, edge_index, edge_attr = self._sample()
        ybus = build_ybus(edge_index, edge_attr, num_nodes=33)[0]
        self.assertTrue(torch.allclose(ybus, ybus.transpose(0, 1), atol=1.0e-12))
        self.assertLess(float(ybus.sum(dim=1).abs().max()), 1.0e-12)

    def test_dataset_sample_has_small_ac_residual(self) -> None:
        state, edge_index, edge_attr = self._sample()
        residual = ac_power_flow_residual(
            state, edge_index, edge_attr, self.base_mva
        )
        # x0 以 float32 保存，允许重建导纳后的舍入误差处于 1e-4 MW/Mvar 内。
        self.assertLess(float(residual.abs().max()), 1.0e-4)

    def test_voltage_perturbation_increases_residual(self) -> None:
        state, edge_index, edge_attr = self._sample()
        baseline = ac_power_flow_residual(
            state, edge_index, edge_attr, self.base_mva
        ).square().mean()
        perturbed = state.clone()
        perturbed[10, 2] += 0.02
        changed = ac_power_flow_residual(
            perturbed, edge_index, edge_attr, self.base_mva
        ).square().mean()
        self.assertGreater(float(changed), float(baseline) * 1.0e4)

    def test_residual_loss_backpropagates(self) -> None:
        state, edge_index, edge_attr = self._sample()
        state = state.clone().requires_grad_(True)
        residual = ac_power_flow_residual(
            state, edge_index, edge_attr, self.base_mva
        )
        loss = power_flow_residual_loss(residual)
        loss.backward()
        self.assertIsNotNone(state.grad)
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(state.grad.abs().sum()), 0.0)

    def test_batched_topologies(self) -> None:
        samples = [self._sample(0), self._sample(20)]
        state = torch.stack([item[0] for item in samples])
        edge_index = torch.stack([item[1] for item in samples])
        edge_attr = torch.stack([item[2] for item in samples])
        residual = ac_power_flow_residual(
            state, edge_index, edge_attr, self.base_mva
        )
        self.assertEqual(tuple(residual.shape), (2, 33, 2))
        self.assertLess(float(residual.abs().max()), 1.0e-4)

    def test_voltage_limit_loss(self) -> None:
        feasible = torch.tensor([0.90, 1.00, 1.10])
        violated = torch.tensor([0.88, 1.00, 1.12])
        self.assertEqual(float(voltage_limit_loss(feasible)), 0.0)
        self.assertGreater(float(voltage_limit_loss(violated)), 0.0)


if __name__ == "__main__":
    unittest.main()
