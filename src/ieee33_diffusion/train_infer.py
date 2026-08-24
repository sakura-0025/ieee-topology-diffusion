"""训练拓扑条件 IEEE33 图 DDPM，并在未见拓扑上生成潮流样本。"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import (
    GaussianDiffusion,
    ModelConfig,
    TopologyConditionedDenoiser,
    TrainingLossConfig,
    VectorConditionedDenoiser,
)
from .physics import PhysicsConfig, dataset_base_mva


class IEEE33Dataset(Dataset):
    """按拓扑切分读取潮流样本，并在取样时组装当前辐射图。"""

    def __init__(self, path: Path, split: int, mean: np.ndarray, std: np.ndarray) -> None:
        data = np.load(path, allow_pickle=False)
        # split: 0=训练，1=验证，2=测试；样本切分已继承所属拓扑的切分。
        indices = np.flatnonzero(data["sample_split"] == split)
        self.x0 = data["x0"][indices].astype(np.float32)
        self.topology_ids = data["sample_topology_id"][indices].astype(np.int64)
        self.active_edge_ids = data["topology_active_edge_ids"].astype(np.int64)
        self.topology_edge_mask = data["topology_edge_mask"].astype(np.float32)
        self.master_edge_index = data["master_edge_index"].astype(np.int64)
        self.master_edge_attr = data["master_edge_attr"].astype(np.float32)
        self.node_static = data["node_static"].astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.x0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        topology_id = int(self.topology_ids[index])
        # 从 37 条候选边中只抽取该拓扑实际连通的 32 条边。
        edge_ids = self.active_edge_ids[topology_id]
        # P、Q、V、theta 分通道标准化；mean/std 的形状都是 [4]。
        x = (self.x0[index] - self.mean) / self.std
        return {
            "x0": torch.from_numpy(x),
            "edge_index": torch.from_numpy(self.master_edge_index[:, edge_ids]),
            "edge_attr": torch.from_numpy(self.master_edge_attr[edge_ids]),
            "topology_mask": torch.from_numpy(self.topology_edge_mask[topology_id]),
            "topology_id": torch.tensor(topology_id, dtype=torch.long),
        }


def _normalization(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """仅使用训练集统计每个生成通道的全局均值和标准差。"""

    data = np.load(data_path, allow_pickle=False)
    train_x = data["x0"][data["sample_split"] == 0].astype(np.float64)
    if not len(train_x):
        raise ValueError("Training split is empty.")
    # 只保留特征维 [F]，确保与单样本 [N,F] 广播后仍是 [N,F]；
    # 如果保留两个单维，DataLoader 会错误地产生 [B,1,N,F]。
    mean = train_x.mean(axis=(0, 1)).astype(np.float32)
    std = train_x.std(axis=(0, 1)).astype(np.float32)
    std = np.maximum(std, 1.0e-6)
    return mean, std


def _device(name: str) -> torch.device:
    """auto 模式优先使用 CUDA，否则使用 CPU。"""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子，便于重复实验。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(config: ModelConfig, device: torch.device) -> GaussianDiffusion:
    """按配置构造图或向量去噪器，并封装为完整扩散模型。"""

    if config.model_type == "graph":
        denoiser = TopologyConditionedDenoiser(config)
    elif config.model_type == "vector":
        denoiser = VectorConditionedDenoiser(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    return GaussianDiffusion(denoiser).to(device)


def _conditioning_topology_ids(
    target_ids: np.ndarray,
    available_ids: np.ndarray,
    mode: str,
    seed: int,
) -> np.ndarray:
    """构造用于条件消融的拓扑ID，同时保留原目标拓扑用于评价。"""

    if mode == "correct":
        return target_ids.copy()
    if mode == "base":
        return np.zeros_like(target_ids)
    if mode != "shuffled":
        raise ValueError(f"Unknown topology condition mode: {mode}")
    if len(available_ids) < 2:
        raise ValueError("Shuffled topology conditioning requires at least two topologies.")

    rng = np.random.default_rng(seed + 77)
    shuffled = available_ids.copy()
    for _ in range(100):
        rng.shuffle(shuffled)
        if np.all(shuffled != available_ids):
            break
    else:
        shuffled = np.roll(available_ids, 1)
    mapping = {int(target): int(condition) for target, condition in zip(available_ids, shuffled)}
    return np.asarray([mapping[int(target)] for target in target_ids], dtype=np.int64)


def _evaluate(
    model: GaussianDiffusion,
    loader: DataLoader,
    node_static: torch.Tensor,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    loss_config: TrainingLossConfig,
    physics_config: PhysicsConfig,
    device: torch.device,
) -> dict[str, float]:
    """计算验证集各项平均损失，不更新模型参数。"""

    model.eval()
    losses: dict[str, list[float]] = {
        "total": [],
        "noise": [],
        "physics": [],
        "voltage": [],
    }
    with torch.no_grad():
        for batch in loader:
            components = model.training_losses(
                batch["x0"].to(device),
                node_static,
                batch["edge_index"].to(device),
                batch["edge_attr"].to(device),
                normalization_mean,
                normalization_std,
                loss_config,
                physics_config,
                batch["topology_mask"].to(device),
            )
            for name, value in components.items():
                losses[name].append(float(value))
    return {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in losses.items()
    }


def _evaluate_with_fixed_noise(
    model: GaussianDiffusion,
    loader: DataLoader,
    node_static: torch.Tensor,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    loss_config: TrainingLossConfig,
    physics_config: PhysicsConfig,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """使用固定验证噪声评价，并在结束后恢复训练随机数状态。"""

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return _evaluate(
            model,
            loader,
            node_static,
            normalization_mean,
            normalization_std,
            loss_config,
            physics_config,
            device,
        )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def train(args: argparse.Namespace) -> None:
    """训练模型并保存 last/best 检查点及损失历史。"""

    _seed_everything(args.seed)
    device = _device(args.device)
    mean, std = _normalization(args.data)
    train_set = IEEE33Dataset(args.data, 0, mean, std)
    val_set = IEEE33Dataset(args.data, 1, mean, std)
    if not len(train_set):
        raise ValueError("No training samples found.")

    # 每个 batch 可包含不同 topology_id；Dataset 会为每个样本返回自己的 32 条边。
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    config = ModelConfig(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        model_type=args.model_type,
    )
    model = _build_model(config, device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        initial_config = ModelConfig(**initial["model_config"])
        if initial_config != config:
            raise ValueError(
                "Initial checkpoint model configuration does not match training arguments."
            )
        if not (
            np.allclose(initial["normalization_mean"], mean)
            and np.allclose(initial["normalization_std"], std)
        ):
            raise ValueError("Initial checkpoint normalization does not match dataset.")
        model.load_state_dict(initial["model"])
        print(f"Initialized model from: {args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 节点类型在所有运行断面间不变，因此只需保留一份 [33,4] 条件张量。
    node_static = torch.from_numpy(train_set.node_static).to(device)
    normalization_mean = torch.from_numpy(mean).to(device)
    normalization_std = torch.from_numpy(std).to(device)
    with np.load(args.data, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        voltage_bounds = metadata.get("voltage_bounds_pu", [0.90, 1.10])
        physics_config = PhysicsConfig(
            base_mva=dataset_base_mva(data),
            voltage_min_pu=float(voltage_bounds[0]),
            voltage_max_pu=float(voltage_bounds[1]),
            residual_scale_mw=args.pf_scale_mw,
            residual_scale_mvar=args.pf_scale_mvar,
            include_slack=not args.exclude_slack_physics,
        )
    loss_config = TrainingLossConfig(
        physics_weight=args.physics_weight,
        voltage_weight=args.voltage_weight,
        physics_time_weight=args.physics_time_weight,
        physics_alpha_bar_min=args.physics_alpha_bar_min,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        epoch_losses: dict[str, list[float]] = {
            "total": [],
            "noise": [],
            "physics": [],
            "voltage": [],
        }
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            components = model.training_losses(
                batch["x0"].to(device),
                node_static,
                batch["edge_index"].to(device),
                batch["edge_attr"].to(device),
                normalization_mean,
                normalization_std,
                loss_config,
                physics_config,
                batch["topology_mask"].to(device),
            )
            loss = components["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Training loss became NaN or Inf; reduce constraint weights or "
                    "inspect the diffusion schedule."
                )
            loss.backward()
            # 限制梯度范数，降低扩散训练初期偶发大梯度造成的不稳定。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in components.items():
                epoch_losses[name].append(float(value.detach()))
            progress.set_postfix(
                total=f"{epoch_losses['total'][-1]:.4f}",
                pf=f"{epoch_losses['physics'][-1]:.4f}",
            )

        train_metrics = {
            name: float(np.mean(values)) for name, values in epoch_losses.items()
        }
        val_metrics = _evaluate_with_fixed_noise(
            model,
            val_loader,
            node_static,
            normalization_mean,
            normalization_std,
            loss_config,
            physics_config,
            device,
            args.seed + 100_000,
        )
        metric = (
            val_metrics["total"]
            if np.isfinite(val_metrics["total"])
            else train_metrics["total"]
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        epoch_record.update({f"train_{k}": v for k, v in train_metrics.items()})
        epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(epoch_record)
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.6f} "
            f"train_noise={train_metrics['noise']:.6f} "
            f"train_pf={train_metrics['physics']:.6f} "
            f"val_total={val_metrics['total']:.6f} "
            f"val_pf={val_metrics['physics']:.6f}"
        )

        # 检查点同时保存配置与归一化统计量，推理时无需重新估计训练集统计。
        checkpoint = {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "training_loss_config": loss_config.to_dict(),
            "physics_config": {
                "base_mva": physics_config.base_mva,
                "voltage_min_pu": physics_config.voltage_min_pu,
                "voltage_max_pu": physics_config.voltage_max_pu,
                "residual_scale_mw": physics_config.residual_scale_mw,
                "residual_scale_mvar": physics_config.residual_scale_mvar,
                "include_slack": physics_config.include_slack,
            },
            "normalization_mean": mean,
            "normalization_std": std,
            "epoch": epoch,
            "validation_loss": val_metrics["total"],
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metric < best_val:
            best_val = metric
            torch.save(checkpoint, args.output_dir / "best.pt")

    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    training_seconds = time.perf_counter() - training_started
    run_config = {
        "command": "train",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "function"},
        "model_config": config.to_dict(),
        "training_loss_config": loss_config.to_dict(),
        "physics_config": checkpoint["physics_config"],
        "training_summary": {
            "training_seconds": training_seconds,
            "seconds_per_epoch": training_seconds / args.epochs,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")


def sample(args: argparse.Namespace) -> None:
    """随机选择测试集拓扑，反向扩散并保存物理量尺度下的样本。"""

    _seed_everything(args.seed)
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = _build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data = np.load(args.data, allow_pickle=False)
    split_code = {"validation": 1, "test": 2}[args.split]
    selected_topologies = np.flatnonzero(data["topology_split"] == split_code)
    if not len(selected_topologies):
        raise ValueError(f"No topology is available in the requested {args.split} split.")
    rng = np.random.default_rng(args.seed)
    if args.samples_per_topology is not None:
        selected = np.repeat(selected_topologies, args.samples_per_topology)
        rng.shuffle(selected)
    else:
        selected = rng.choice(selected_topologies, size=args.num_samples, replace=True)
    conditioning_ids = _conditioning_topology_ids(
        selected,
        selected_topologies,
        args.condition_mode,
        args.seed,
    )
    master_index = data["master_edge_index"]
    master_attr = data["master_edge_attr"]
    node_static = torch.from_numpy(data["node_static"].astype(np.float32)).to(device)

    generated_blocks: list[np.ndarray] = []
    sampling_started = time.perf_counter()
    for start in tqdm(range(0, len(selected), args.batch_size), desc="sampling batches"):
        stop = min(start + args.batch_size, len(selected))
        topology_batch = selected[start:stop]
        condition_batch = conditioning_ids[start:stop]
        active = data["topology_active_edge_ids"][condition_batch]
        topology_mask = data["topology_edge_mask"][condition_batch]
        edge_index = np.stack([master_index[:, ids] for ids in active])
        edge_attr = np.stack([master_attr[ids] for ids in active])
        generated_batch = model.sample(
            (len(topology_batch), int(data["node_static"].shape[0]), config.node_channels),
            node_static,
            torch.from_numpy(edge_index).long().to(device),
            torch.from_numpy(edge_attr).float().to(device),
            device,
            torch.from_numpy(topology_mask).float().to(device),
        )
        generated_blocks.append(generated_batch.cpu().numpy())
    generated_normalized = np.concatenate(generated_blocks, axis=0)
    sampling_seconds = time.perf_counter() - sampling_started
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    # 反标准化后恢复 MW、Mvar、p.u. 和 rad 的物理量尺度。
    generated = generated_normalized * std + mean

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        generated_x0=generated.astype(np.float32),
        topology_id=selected.astype(np.int32),
        conditioning_topology_id=conditioning_ids.astype(np.int32),
        active_edge_ids=data["topology_active_edge_ids"][selected].astype(np.int32),
        conditioning_active_edge_ids=data["topology_active_edge_ids"][conditioning_ids].astype(
            np.int32
        ),
        sampling_metadata=np.asarray(
            json.dumps(
                {
                    "split": args.split,
                    "condition_mode": args.condition_mode,
                    "batch_size": args.batch_size,
                    "sampling_seconds": sampling_seconds,
                    "seconds_per_sample": sampling_seconds / len(selected),
                }
            )
        ),
    )
    print(
        f"Saved {len(selected)} generated samples to {args.output} in "
        f"{sampling_seconds:.3f} seconds"
    )


def build_parser() -> argparse.ArgumentParser:
    """建立 train/sample 两个命令行子命令。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the graph DDPM")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs/default"))
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    train_parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start model weights from a compatible checkpoint.",
    )
    train_parser.add_argument("--hidden-channels", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=6)
    train_parser.add_argument("--diffusion-steps", type=int, default=200)
    train_parser.add_argument(
        "--noise-schedule",
        choices=("linear", "cosine"),
        default="linear",
        help="Forward diffusion noise schedule; cosine is suitable for fewer steps.",
    )
    train_parser.add_argument(
        "--model-type",
        choices=("graph", "vector"),
        default="graph",
        help="Use graph message passing or a flattened topology-conditioned baseline.",
    )
    train_parser.add_argument(
        "--physics-weight",
        type=float,
        default=0.0,
        help="Weight of topology-specific AC power-flow residual loss.",
    )
    train_parser.add_argument(
        "--voltage-weight",
        type=float,
        default=0.0,
        help="Weight of voltage-limit violation loss.",
    )
    train_parser.add_argument(
        "--physics-time-weight",
        choices=("none", "alpha_bar"),
        default="alpha_bar",
        help="Down-weight unreliable physics gradients at noisy diffusion steps.",
    )
    train_parser.add_argument(
        "--physics-alpha-bar-min",
        type=float,
        default=0.0,
        help="Apply AC residual only when alpha_bar is at least this value.",
    )
    train_parser.add_argument("--pf-scale-mw", type=float, default=1.0)
    train_parser.add_argument("--pf-scale-mvar", type=float, default=1.0)
    train_parser.add_argument(
        "--exclude-slack-physics",
        action="store_true",
        help="Exclude the slack bus from AC residual loss.",
    )
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.set_defaults(function=train)

    sample_parser = subparsers.add_parser(
        "sample", help="Sample validation or unseen test topologies"
    )
    sample_parser.add_argument("--data", type=Path, required=True)
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path, default=Path("outputs/generated.npz"))
    sample_parser.add_argument("--num-samples", type=int, default=64)
    sample_parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Topology split to generate; use validation while tuning.",
    )
    sample_parser.add_argument(
        "--condition-mode",
        choices=("correct", "base", "shuffled"),
        default="correct",
        help="Use correct, base-topology, or deranged topology conditioning.",
    )
    sample_parser.add_argument("--batch-size", type=int, default=256)
    sample_parser.add_argument(
        "--samples-per-topology",
        type=int,
        default=None,
        help="Generate an equal number for every unseen test topology.",
    )
    sample_parser.add_argument("--device", default="auto")
    sample_parser.add_argument("--seed", type=int, default=2026)
    sample_parser.set_defaults(function=sample)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
