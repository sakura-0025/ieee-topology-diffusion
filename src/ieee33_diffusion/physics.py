"""可微分的变拓扑交流潮流物理约束。

本模块使用数据集中的标幺线路阻抗 ``r_pu, x_pu`` 构造节点导纳矩阵，
再由节点电压幅值和相角计算净注入功率。数据集的功率通道单位为 MW/Mvar，
因此标幺复功率需要乘以系统基准容量 ``base_mva``。

当前实现对应平衡单相、仅含串联线路阻抗的 IEEE33 场景。case33bw 的线路
并联电导和电纳均为零；若后续数据包含变压器、分接头或线路充电，应扩展
Ybus 构造方式，而不是继续使用当前简化模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PhysicsConfig:
    """交流潮流损失使用的物理常数和边界。"""

    base_mva: float = 10.0
    voltage_min_pu: float = 0.90
    voltage_max_pu: float = 1.10
    residual_scale_mw: float = 1.0
    residual_scale_mvar: float = 1.0
    include_slack: bool = True


def dataset_base_mva(data: Mapping[str, np.ndarray]) -> float:
    """读取数据集基准容量，并兼容早期未保存该字段的 case33bw 文件。

    新数据集直接保存标量 ``base_mva``。早期数据只在 ``metadata`` 中记录
    算例名称；对于明确标记为 Baran-Wu case33bw 的文件，返回其 10 MVA
    基准。其他旧数据集不能安全推断，必须重新构建或显式提供基准容量。
    """

    if "base_mva" in data:
        value = float(np.asarray(data["base_mva"]).item())
    elif "metadata" in data:
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        if metadata.get("case") != "Baran-Wu case33bw":
            raise ValueError("Dataset does not contain base_mva and is not case33bw.")
        value = 10.0
    else:
        raise ValueError("Dataset does not contain base_mva or recognizable metadata.")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"base_mva must be finite and positive, got {value!r}.")
    return value


def _batched_graph_inputs(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把单图或批图输入统一为 ``[B,2,E]`` 与 ``[B,E,F]``。"""

    if edge_index.ndim == 2:
        edge_index = edge_index.unsqueeze(0)
    if edge_attr.ndim == 2:
        edge_attr = edge_attr.unsqueeze(0)
    if edge_index.ndim != 3 or edge_index.shape[1] != 2:
        raise ValueError("edge_index must have shape [2,E] or [B,2,E].")
    if edge_attr.ndim != 3 or edge_attr.shape[-1] < 2:
        raise ValueError("edge_attr must have shape [E,F] or [B,E,F] with F>=2.")
    if edge_index.shape[0] != edge_attr.shape[0]:
        if edge_index.shape[0] == 1:
            edge_index = edge_index.expand(edge_attr.shape[0], -1, -1)
        elif edge_attr.shape[0] == 1:
            edge_attr = edge_attr.expand(edge_index.shape[0], -1, -1)
        else:
            raise ValueError("edge_index and edge_attr batch sizes do not match.")
    if edge_index.shape[-1] != edge_attr.shape[1]:
        raise ValueError("edge_index and edge_attr contain different edge counts.")
    return edge_index.long(), edge_attr


def build_ybus(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """由活动支路构造批量复数节点导纳矩阵 ``[B,N,N]``。

    ``edge_attr[...,0:2]`` 依次是标幺电阻和电抗。每条无向线路的串联导纳
    ``y=1/(r+jx)`` 对 Ybus 的贡献为 ``Yuu+=y, Yvv+=y, Yuv-=y, Yvu-=y``。
    """

    edge_index, edge_attr = _batched_graph_inputs(edge_index, edge_attr)
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if edge_index.numel():
        min_index = int(edge_index.min())
        max_index = int(edge_index.max())
        if min_index < 0 or max_index >= num_nodes:
            raise ValueError(
                f"edge_index contains node outside [0,{num_nodes - 1}]: "
                f"min={min_index}, max={max_index}."
            )

    resistance = edge_attr[..., 0]
    reactance = edge_attr[..., 1]
    impedance_sq = resistance.square() + reactance.square()
    if torch.any(impedance_sq <= 0):
        raise ValueError("Every active branch must have non-zero impedance.")

    # 显式写出 1/(r+jx)，避免依赖输入是否已经是复数类型。
    admittance = torch.complex(
        resistance / impedance_sq,
        -reactance / impedance_sq,
    )
    batch_size, num_edges = resistance.shape
    source = edge_index[:, 0]
    target = edge_index[:, 1]
    batch_offset = (
        torch.arange(batch_size, device=edge_index.device).unsqueeze(1)
        * num_nodes
        * num_nodes
    )

    def flat_position(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        return (batch_offset + row * num_nodes + col).reshape(-1)

    flat_ybus = torch.zeros(
        batch_size * num_nodes * num_nodes,
        dtype=admittance.dtype,
        device=edge_attr.device,
    )
    values = admittance.reshape(batch_size * num_edges)
    flat_ybus.index_add_(0, flat_position(source, source), values)
    flat_ybus.index_add_(0, flat_position(target, target), values)
    flat_ybus.index_add_(0, flat_position(source, target), -values)
    flat_ybus.index_add_(0, flat_position(target, source), -values)
    return flat_ybus.view(batch_size, num_nodes, num_nodes)


def calculated_power_injections(
    voltage_pu: torch.Tensor,
    theta_rad: torch.Tensor,
    ybus: torch.Tensor,
    base_mva: float,
) -> torch.Tensor:
    """根据 ``V,theta,Ybus`` 计算节点净注入 ``[P_MW,Q_Mvar]``。"""

    squeeze_batch = voltage_pu.ndim == 1
    if squeeze_batch:
        voltage_pu = voltage_pu.unsqueeze(0)
        theta_rad = theta_rad.unsqueeze(0)
    if voltage_pu.ndim != 2 or theta_rad.shape != voltage_pu.shape:
        raise ValueError("voltage_pu and theta_rad must have matching [B,N] shapes.")
    if ybus.ndim == 2:
        ybus = ybus.unsqueeze(0)
    if ybus.ndim != 3 or ybus.shape[1:] != (
        voltage_pu.shape[1],
        voltage_pu.shape[1],
    ):
        raise ValueError("ybus must have shape [N,N] or [B,N,N].")
    if ybus.shape[0] == 1 and voltage_pu.shape[0] > 1:
        ybus = ybus.expand(voltage_pu.shape[0], -1, -1)
    if ybus.shape[0] != voltage_pu.shape[0]:
        raise ValueError("ybus and voltage batch sizes do not match.")
    if base_mva <= 0.0:
        raise ValueError("base_mva must be positive.")

    voltage_complex = torch.polar(voltage_pu, theta_rad)
    current = torch.bmm(ybus, voltage_complex.unsqueeze(-1)).squeeze(-1)
    power = voltage_complex * current.conj() * float(base_mva)
    result = torch.stack([power.real, power.imag], dim=-1)
    return result.squeeze(0) if squeeze_batch else result


def ac_power_flow_residual(
    state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    base_mva: float,
) -> torch.Tensor:
    """计算状态 ``[P,Q,V,theta]`` 的节点交流潮流残差。

    返回形状与 ``state[...,0:2]`` 相同，定义为“生成净注入减去由 V、theta
    和当前拓扑计算的净注入”。正注入表示发电，负注入表示负荷。
    """

    squeeze_batch = state.ndim == 2
    if squeeze_batch:
        state = state.unsqueeze(0)
    if state.ndim != 3 or state.shape[-1] != 4:
        raise ValueError("state must have shape [N,4] or [B,N,4].")

    edge_index, edge_attr = _batched_graph_inputs(edge_index, edge_attr)
    if edge_index.shape[0] == 1 and state.shape[0] > 1:
        edge_index = edge_index.expand(state.shape[0], -1, -1)
        edge_attr = edge_attr.expand(state.shape[0], -1, -1)
    if edge_index.shape[0] != state.shape[0]:
        raise ValueError("state and graph batch sizes do not match.")

    ybus = build_ybus(edge_index, edge_attr, state.shape[1])
    calculated = calculated_power_injections(
        state[..., 2], state[..., 3], ybus, base_mva
    )
    residual = state[..., :2] - calculated
    return residual.squeeze(0) if squeeze_batch else residual


def power_flow_residual_loss(
    residual: torch.Tensor,
    scale_mw: float = 1.0,
    scale_mvar: float = 1.0,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算归一化的节点功率残差均方损失。"""

    if residual.shape[-1] != 2:
        raise ValueError("residual last dimension must be [P,Q].")
    if scale_mw <= 0.0 or scale_mvar <= 0.0:
        raise ValueError("Residual scales must be positive.")
    scale = residual.new_tensor([scale_mw, scale_mvar])
    squared = (residual / scale).square().mean(dim=-1)
    if node_mask is None:
        return squared.mean()
    mask = node_mask.to(device=residual.device, dtype=residual.dtype)
    while mask.ndim < squared.ndim:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(squared)
    return (squared * mask).sum() / mask.sum().clamp_min(1.0)


def voltage_limit_loss(
    voltage_pu: torch.Tensor,
    minimum_pu: float = 0.90,
    maximum_pu: float = 1.10,
) -> torch.Tensor:
    """对电压上下限之外的距离施加平方惩罚。"""

    if minimum_pu >= maximum_pu:
        raise ValueError("minimum_pu must be smaller than maximum_pu.")
    lower = F.relu(minimum_pu - voltage_pu)
    upper = F.relu(voltage_pu - maximum_pu)
    return (lower.square() + upper.square()).mean()


def denormalize_state(
    normalized_state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """把模型状态从逐通道标准化空间恢复到物理量空间。"""

    if normalized_state.shape[-1] != 4:
        raise ValueError("normalized_state last dimension must contain four channels.")
    mean = mean.to(device=normalized_state.device, dtype=normalized_state.dtype)
    std = std.to(device=normalized_state.device, dtype=normalized_state.dtype)
    if mean.shape != (4,) or std.shape != (4,):
        raise ValueError("mean and std must both have shape [4].")
    if torch.any(std <= 0):
        raise ValueError("Every normalization standard deviation must be positive.")
    return normalized_state * std + mean
