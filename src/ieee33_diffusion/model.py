"""拓扑条件图去噪器与 DDPM 正、反向过程。

张量维度约定：B 为批量大小，N=33 为节点数，E=32 为当前拓扑活动边数，
F=4 为待生成通道数 ``[P, Q, V, theta]``。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


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
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2

    def to_dict(self) -> dict[str, int | float]:
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
    ) -> torch.Tensor:
        """返回与 ``noisy_x [B,N,4]`` 同形状的预测噪声。"""

        # 全网静态节点特征通常只存一份 [N,Fs]，运行时扩展到批维。
        if node_static.ndim == 2:
            node_static = node_static.unsqueeze(0).expand(noisy_x.shape[0], -1, -1)
        hidden = self.input_projection(torch.cat([noisy_x, node_static], dim=-1))
        time_hidden = self.time_encoder(timestep)
        edge_hidden = self.edge_projection(edge_attr)
        for block in self.blocks:
            hidden = block(hidden, edge_index, edge_hidden, time_hidden)
        return self.output_projection(hidden)


class GaussianDiffusion(nn.Module):
    """采用线性 beta 调度的噪声预测型 DDPM。"""

    def __init__(self, denoiser: TopologyConditionedDenoiser) -> None:
        super().__init__()
        self.denoiser = denoiser
        config = denoiser.config
        # alpha_t = 1-beta_t；alpha_bar_t 是从第 0 步到第 t 步的累计乘积。
        betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

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

    def training_loss(
        self,
        clean_x: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """随机选择扩散时刻，以真实噪声和预测噪声的均方误差训练。"""

        batch = clean_x.shape[0]
        timestep = torch.randint(self.steps, (batch,), device=clean_x.device)
        noise = torch.randn_like(clean_x)
        noisy_x = self.q_sample(clean_x, timestep, noise)
        predicted = self.denoiser(noisy_x, timestep, node_static, edge_index, edge_attr)
        return torch.mean((predicted - noise) ** 2)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """从标准高斯噪声出发，逐步反演得到归一化潮流样本。"""

        # x_T ~ N(0,I)，其形状为 [样本数, 33, 4]。
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.steps)):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = self.denoiser(
                x, timestep, node_static, edge_index, edge_attr
            )
            alpha = self._extract(self.alphas, timestep, x.ndim)
            alpha_bar = self._extract(self.alpha_bars, timestep, x.ndim)
            beta = self._extract(self.betas, timestep, x.ndim)
            # DDPM 反向均值；非最后一步额外采样随机项，最后一步直接输出均值。
            mean = (x - (1.0 - alpha) / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            if step > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
