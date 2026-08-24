"""拓扑条件图去噪器与 DDPM 正、反向过程。

张量维度约定：B 为批量大小，N=33 为节点数，E=32 为当前拓扑活动边数，
F=4 为待生成通道数 ``[P, Q, V, theta]``。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .physics import (
    PhysicsConfig,
    ac_power_flow_residual,
    denormalize_state,
)


@dataclass(frozen=True)
class ModelConfig:
    """模型与扩散过程配置。"""

    node_channels: int = 4
    node_static_channels: int = 4
    edge_channels: int = 4
    hidden_channels: int = 128
    time_channels: int = 128
    num_layers: int = 6
    diffusion_steps: int = 200
    noise_schedule: str = "linear"
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    model_type: str = "graph"
    num_nodes: int = 33
    num_candidate_edges: int = 37

    def __post_init__(self) -> None:
        if self.model_type not in {"graph", "vector"}:
            raise ValueError("model_type must be 'graph' or 'vector'.")
        if self.noise_schedule not in {"linear", "cosine"}:
            raise ValueError("noise_schedule must be 'linear' or 'cosine'.")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2.")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1.")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingLossConfig:
    """噪声、潮流残差和运行边界三类训练目标的组合方式。"""

    physics_weight: float = 0.0
    voltage_weight: float = 0.0
    physics_time_weight: str = "alpha_bar"

    def __post_init__(self) -> None:
        if self.physics_weight < 0.0 or self.voltage_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if self.physics_time_weight not in {"none", "alpha_bar"}:
            raise ValueError("physics_time_weight must be 'none' or 'alpha_bar'.")

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class SinusoidalTimeEmbedding(nn.Module):
    """把离散扩散时刻 ``t [B]`` 编码成连续向量 ``[B, time_channels]``。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # 不同频率的正弦/余弦让模型同时感知短期与长期扩散步尺度。
        half = self.channels // 2
        scale = math.log(10_000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timestep.device, dtype=torch.float32)
        )
        angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.channels:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class GraphMessageBlock(nn.Module):
    """支持批内不同辐射拓扑的图消息传递块。

    输入维度：
        node_hidden: ``[B, N, H]``；
        edge_index: ``[B, 2, E]``；
        edge_attr: ``[B, E, Fe]``；
        time_hidden: ``[B, Ft]``。
    """

    def __init__(self, hidden_channels: int, edge_channels: int, time_channels: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + time_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(
        self,
        node_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        time_hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden = node_hidden.shape
        num_edges = edge_index.shape[-1]

        # pandapower 支路只存一个方向；配电网消息传递需沿两个方向传播，故显式双向化。
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        src_bi = torch.cat([src, dst], dim=1)
        dst_bi = torch.cat([dst, src], dim=1)
        edge_bi = torch.cat([edge_attr, edge_attr], dim=1)

        # 按批内每个样本自己的 edge_index 提取边两端节点表示。
        src_index = src_bi.unsqueeze(-1).expand(-1, -1, hidden)
        dst_index = dst_bi.unsqueeze(-1).expand(-1, -1, hidden)
        h_src = torch.gather(node_hidden, 1, src_index)
        h_dst = torch.gather(node_hidden, 1, dst_index)
        messages = self.message_mlp(torch.cat([h_src, h_dst, edge_bi], dim=-1))

        # 把 [B,N,H] 临时展平，并给各样本节点编号添加偏移，防止跨样本聚合。
        offsets = torch.arange(batch_size, device=node_hidden.device).unsqueeze(1) * num_nodes
        flat_dst = (dst_bi + offsets).reshape(-1)
        aggregate = torch.zeros(
            batch_size * num_nodes, hidden, device=node_hidden.device, dtype=node_hidden.dtype
        )
        aggregate.index_add_(0, flat_dst, messages.reshape(-1, hidden))

        # 按节点度数取均值，避免支路较多的节点仅因邻居数量而产生更大特征幅值。
        degree = torch.zeros(batch_size * num_nodes, 1, device=node_hidden.device)
        degree.index_add_(
            0,
            flat_dst,
            torch.ones(batch_size * 2 * num_edges, 1, device=node_hidden.device),
        )
        aggregate = (aggregate / degree.clamp_min(1.0)).view(batch_size, num_nodes, hidden)
        time_per_node = time_hidden.unsqueeze(1).expand(-1, num_nodes, -1)
        # 节点自身、邻域信息和当前扩散时刻共同决定残差更新。
        update = self.update_mlp(torch.cat([node_hidden, aggregate, time_per_node], dim=-1))
        return self.norm(node_hidden + update)


class TopologyConditionedDenoiser(nn.Module):
    """预测加入 ``x_t`` 的高斯噪声 epsilon。

    拓扑条件不是一个离散 topology_id，而是当前的活动 ``edge_index`` 及线路参数。
    因而同一套参数可以作用于训练时未出现、但节点与候选边定义一致的新拓扑。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(config.time_channels),
            nn.Linear(config.time_channels, config.time_channels),
            nn.SiLU(),
            nn.Linear(config.time_channels, config.time_channels),
        )
        # 将带噪潮流变量与不加噪的静态节点类型拼接后映射到隐空间。
        self.input_projection = nn.Linear(
            config.node_channels + config.node_static_channels,
            config.hidden_channels,
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(config.edge_channels, config.edge_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                GraphMessageBlock(
                    config.hidden_channels,
                    config.edge_channels,
                    config.time_channels,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.SiLU(),
            nn.Linear(config.hidden_channels, config.node_channels),
        )

    def forward(
        self,
        noisy_x: torch.Tensor,
        timestep: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        topology_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回与 ``noisy_x [B,N,4]`` 同形状的预测噪声。"""

        del topology_mask
        # 全网静态节点特征通常只存一份 [N,Fs]，运行时扩展到批维。
        if node_static.ndim == 2:
            node_static = node_static.unsqueeze(0).expand(noisy_x.shape[0], -1, -1)
        hidden = self.input_projection(torch.cat([noisy_x, node_static], dim=-1))
        time_hidden = self.time_encoder(timestep)
        edge_hidden = self.edge_projection(edge_attr)
        for block in self.blocks:
            hidden = block(hidden, edge_index, edge_hidden, time_hidden)
        return self.output_projection(hidden)


class VectorResidualBlock(nn.Module):
    """非图基线使用的带时间条件全连接残差块。"""

    def __init__(self, hidden_channels: int, time_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_channels + time_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, hidden: torch.Tensor, time_hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden + self.layers(torch.cat([hidden, time_hidden], dim=-1)))


class VectorConditionedDenoiser(nn.Module):
    """将节点状态和37位拓扑掩码展平的条件DDPM基线。

    该模型接收与图模型完全相同的数据和训练目标，但不进行支路消息传递。
    它用于验证性能改善究竟来自图结构归纳偏置，还是仅来自拓扑条件本身。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(config.time_channels),
            nn.Linear(config.time_channels, config.time_channels),
            nn.SiLU(),
            nn.Linear(config.time_channels, config.time_channels),
        )
        input_channels = (
            config.num_nodes * (config.node_channels + config.node_static_channels)
            + config.num_candidate_edges
        )
        self.input_projection = nn.Sequential(
            nn.Linear(input_channels, config.hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                VectorResidualBlock(config.hidden_channels, config.time_channels)
                for _ in range(config.num_layers)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.SiLU(),
            nn.Linear(
                config.hidden_channels, config.num_nodes * config.node_channels
            ),
        )

    def forward(
        self,
        noisy_x: torch.Tensor,
        timestep: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        topology_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_attr
        batch = noisy_x.shape[0]
        if noisy_x.shape[1:] != (self.config.num_nodes, self.config.node_channels):
            raise ValueError("noisy_x does not match configured vector model dimensions.")
        if node_static.ndim == 2:
            node_static = node_static.unsqueeze(0).expand(batch, -1, -1)
        if topology_mask is None:
            raise ValueError("VectorConditionedDenoiser requires topology_mask.")
        if topology_mask.ndim == 1:
            topology_mask = topology_mask.unsqueeze(0).expand(batch, -1)
        if topology_mask.shape != (batch, self.config.num_candidate_edges):
            raise ValueError("topology_mask does not match configured candidate edges.")
        vector_input = torch.cat(
            [
                noisy_x.reshape(batch, -1),
                node_static.reshape(batch, -1),
                topology_mask.to(noisy_x.dtype),
            ],
            dim=-1,
        )
        hidden = self.input_projection(vector_input)
        time_hidden = self.time_encoder(timestep)
        for block in self.blocks:
            hidden = block(hidden, time_hidden)
        return self.output_projection(hidden).view_as(noisy_x)


class GaussianDiffusion(nn.Module):
    """采用线性 beta 调度的噪声预测型 DDPM。"""

    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser
        config = denoiser.config
        # alpha_t = 1-beta_t；alpha_bar_t 是从第 0 步到第 t 步的累计乘积。
        # 线性日程保留用于复现实验；余弦日程在 100/200 等少步数设置下仍能让
        # q(x_T|x_0) 充分接近标准高斯，避免训练末态与采样起点不匹配。
        if config.noise_schedule == "linear":
            betas = torch.linspace(
                config.beta_start, config.beta_end, config.diffusion_steps
            )
        else:
            offset = 0.008
            time = torch.linspace(0, config.diffusion_steps, config.diffusion_steps + 1)
            cumulative = torch.cos(
                ((time / config.diffusion_steps + offset) / (1.0 + offset))
                * math.pi
                / 2.0
            ).square()
            cumulative = cumulative / cumulative[0]
            betas = 1.0 - cumulative[1:] / cumulative[:-1]
            betas = betas.clamp(min=1.0e-8, max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat(
            [torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance)

    @property
    def steps(self) -> int:
        return int(self.betas.shape[0])

    @staticmethod
    def _extract(values: torch.Tensor, timestep: torch.Tensor, ndim: int) -> torch.Tensor:
        """按批内时刻抽取扩散系数，并扩展为可与 ``[B,N,F]`` 广播的形状。"""

        shape = (timestep.shape[0],) + (1,) * (ndim - 1)
        return values.gather(0, timestep).view(shape)

    def q_sample(
        self, clean_x: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """前向闭式加噪：x_t=sqrt(alpha_bar_t)x_0+sqrt(1-alpha_bar_t)epsilon。"""

        alpha_bar = self._extract(self.alpha_bars, timestep, clean_x.ndim)
        return alpha_bar.sqrt() * clean_x + (1.0 - alpha_bar).sqrt() * noise

    def predict_clean_from_noise(
        self,
        noisy_x: torch.Tensor,
        timestep: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """由 ``x_t`` 与预测噪声恢复标准化空间中的清洁状态估计。"""

        alpha_bar = self._extract(self.alpha_bars, timestep, noisy_x.ndim)
        return (
            noisy_x - (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt().clamp_min(1.0e-8)

    def training_losses(
        self,
        clean_x: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        normalization_mean: torch.Tensor,
        normalization_std: torch.Tensor,
        loss_config: TrainingLossConfig,
        physics_config: PhysicsConfig,
        topology_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """计算标准 DDPM 损失及可选的拓扑相关物理损失。

        物理方程不能直接作用于标准化变量。模型先恢复 ``x0_hat``，再反标准化
        为 MW、Mvar、p.u. 和 rad，最后根据批内每个样本自己的活动拓扑计算
        交流潮流残差。
        """

        batch = clean_x.shape[0]
        timestep = torch.randint(self.steps, (batch,), device=clean_x.device)
        noise = torch.randn_like(clean_x)
        noisy_x = self.q_sample(clean_x, timestep, noise)
        predicted = self.denoiser(
            noisy_x,
            timestep,
            node_static,
            edge_index,
            edge_attr,
            topology_mask,
        )
        noise_loss = torch.mean((predicted - noise) ** 2)

        zero = noise_loss.new_zeros(())
        physics_loss = zero
        voltage_loss = zero
        if loss_config.physics_weight > 0.0 or loss_config.voltage_weight > 0.0:
            predicted_clean = self.predict_clean_from_noise(
                noisy_x, timestep, predicted
            )
            physical_state = denormalize_state(
                predicted_clean, normalization_mean, normalization_std
            )
            if loss_config.physics_time_weight == "alpha_bar":
                time_weight = self.alpha_bars.gather(0, timestep)
            else:
                time_weight = torch.ones(batch, device=clean_x.device)

            if loss_config.physics_weight > 0.0:
                residual = ac_power_flow_residual(
                    physical_state,
                    edge_index,
                    edge_attr,
                    physics_config.base_mva,
                )
                scale = residual.new_tensor(
                    [
                        physics_config.residual_scale_mw,
                        physics_config.residual_scale_mvar,
                    ]
                )
                per_node = (residual / scale).square().mean(dim=-1)
                if physics_config.include_slack:
                    per_sample = per_node.mean(dim=-1)
                else:
                    static = node_static
                    if static.ndim == 2:
                        static = static.unsqueeze(0).expand(batch, -1, -1)
                    non_slack = 1.0 - static[..., 1]
                    per_sample = (per_node * non_slack).sum(dim=-1) / non_slack.sum(
                        dim=-1
                    ).clamp_min(1.0)
                physics_loss = (time_weight * per_sample).mean()

            if loss_config.voltage_weight > 0.0:
                voltage = physical_state[..., 2]
                lower = torch.relu(physics_config.voltage_min_pu - voltage)
                upper = torch.relu(voltage - physics_config.voltage_max_pu)
                per_sample_voltage = (lower.square() + upper.square()).mean(dim=-1)
                voltage_loss = (time_weight * per_sample_voltage).mean()

        total = (
            noise_loss
            + loss_config.physics_weight * physics_loss
            + loss_config.voltage_weight * voltage_loss
        )
        return {
            "total": total,
            "noise": noise_loss,
            "physics": physics_loss,
            "voltage": voltage_loss,
        }

    def training_loss(
        self,
        clean_x: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        topology_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """随机选择扩散时刻，以真实噪声和预测噪声的均方误差训练。"""

        batch = clean_x.shape[0]
        timestep = torch.randint(self.steps, (batch,), device=clean_x.device)
        noise = torch.randn_like(clean_x)
        noisy_x = self.q_sample(clean_x, timestep, noise)
        predicted = self.denoiser(
            noisy_x,
            timestep,
            node_static,
            edge_index,
            edge_attr,
            topology_mask,
        )
        return torch.mean((predicted - noise) ** 2)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        device: torch.device,
        topology_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """从标准高斯噪声出发，逐步反演得到归一化潮流样本。"""

        # x_T ~ N(0,I)，其形状为 [样本数, 33, 4]。
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.steps)):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = self.denoiser(
                x, timestep, node_static, edge_index, edge_attr, topology_mask
            )
            alpha = self._extract(self.alphas, timestep, x.ndim)
            alpha_bar = self._extract(self.alpha_bars, timestep, x.ndim)
            posterior_variance = self._extract(
                self.posterior_variance, timestep, x.ndim
            )
            # DDPM 反向均值；非最后一步额外采样随机项，最后一步直接输出均值。
            mean = (x - (1.0 - alpha) / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            if step > 0:
                x = mean + posterior_variance.clamp_min(1.0e-20).sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
