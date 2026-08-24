"""评价生成潮流的统计保真度、物理一致性、多样性和拓扑泛化。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from .physics import ac_power_flow_residual, dataset_base_mva


CHANNEL_NAMES = ("p_injection_mw", "q_injection_mvar", "voltage_pu", "theta_rad")


def _training_normalization(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    train = data["x0"][data["sample_split"] == 0].astype(np.float64)
    mean = train.mean(axis=(0, 1))
    std = np.maximum(train.std(axis=(0, 1)), 1.0e-6)
    return mean, std


def _matched_reference(
    data: np.lib.npyio.NpzFile,
    topology_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """为每个生成样本从相同拓扑中抽取一个真实参考样本。"""

    rng = np.random.default_rng(seed)
    sample_topologies = data["sample_topology_id"]
    reference = np.empty((len(topology_ids),) + data["x0"].shape[1:], dtype=np.float32)
    for topology_id in np.unique(topology_ids):
        generated_positions = np.flatnonzero(topology_ids == topology_id)
        candidates = np.flatnonzero(sample_topologies == topology_id)
        if not len(candidates):
            raise ValueError(f"No reference sample exists for topology {topology_id}.")
        chosen = rng.choice(
            candidates,
            size=len(generated_positions),
            replace=len(generated_positions) > len(candidates),
        )
        reference[generated_positions] = data["x0"][chosen]
    return reference


def _per_node_wasserstein(real: np.ndarray, generated: np.ndarray) -> list[float]:
    result: list[float] = []
    for channel in range(real.shape[-1]):
        distances = [
            wasserstein_distance(real[:, node, channel], generated[:, node, channel])
            for node in range(real.shape[1])
        ]
        result.append(float(np.mean(distances)))
    return result


def _mmd_rbf(
    real: np.ndarray,
    generated: np.ndarray,
    max_samples: int,
    seed: int,
) -> dict[str, float | int]:
    """在标准化扁平状态上计算带中位数带宽的偏置 RBF-MMD²。"""

    rng = np.random.default_rng(seed)
    count = min(max_samples, len(real), len(generated))
    real = real[rng.choice(len(real), count, replace=False)].reshape(count, -1)
    generated = generated[rng.choice(len(generated), count, replace=False)].reshape(
        count, -1
    )
    # 带宽只由真实参考分布确定。若把病态生成样本混入带宽估计，其尺度爆炸
    # 会同步放大 RBF 带宽，反而掩盖生成失败，并使不同模型的 MMD 不可比较。
    bandwidth_count = min(500, len(real))
    bandwidth_points = real[rng.choice(len(real), bandwidth_count, replace=False)]
    sq_distance = np.sum(
        (bandwidth_points[:, None, :] - bandwidth_points[None, :, :]) ** 2,
        axis=-1,
    )
    positive = sq_distance[sq_distance > 0]
    bandwidth_sq = float(np.median(positive)) if len(positive) else 1.0
    bandwidth_sq = max(bandwidth_sq, 1.0e-12)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
        return np.exp(-distance / (2.0 * bandwidth_sq))

    mmd_sq = float(
        kernel(real, real).mean()
        + kernel(generated, generated).mean()
        - 2.0 * kernel(real, generated).mean()
    )
    return {
        "mmd_rbf_squared": max(mmd_sq, 0.0),
        "mmd_sample_count": count,
        "rbf_bandwidth_squared": bandwidth_sq,
        "rbf_bandwidth_source": "real_reference",
    }


def _nearest_reference_distance(
    real: np.ndarray,
    generated: np.ndarray,
    max_samples: int,
    seed: int,
) -> dict[str, float | int]:
    """计算标准化状态到最近真实样本的均方根欧氏距离。"""

    rng = np.random.default_rng(seed + 1)
    real_count = min(max_samples, len(real))
    generated_count = min(max_samples, len(generated))
    real = real[rng.choice(len(real), real_count, replace=False)].reshape(real_count, -1)
    generated = generated[
        rng.choice(len(generated), generated_count, replace=False)
    ].reshape(generated_count, -1)
    minima: list[np.ndarray] = []
    for start in range(0, generated_count, 128):
        block = generated[start : start + 128]
        sq_distance = np.mean((block[:, None, :] - real[None, :, :]) ** 2, axis=-1)
        minima.append(np.sqrt(sq_distance.min(axis=1)))
    nearest = np.concatenate(minima)
    return {
        "nearest_reference_mean": float(nearest.mean()),
        "nearest_reference_p05": float(np.quantile(nearest, 0.05)),
        "nearest_reference_sample_count": generated_count,
    }


def _physics_metrics(
    data: np.lib.npyio.NpzFile,
    generated: np.ndarray,
    topology_ids: np.ndarray,
    batch_size: int,
    tolerance: float,
) -> dict[str, float]:
    master_index = data["master_edge_index"]
    master_attr = data["master_edge_attr"]
    active_by_topology = data["topology_active_edge_ids"]
    residual_blocks: list[np.ndarray] = []
    sample_max_blocks: list[np.ndarray] = []
    sample_mean_blocks: list[np.ndarray] = []
    base_mva = dataset_base_mva(data)
    for start in range(0, len(generated), batch_size):
        stop = min(start + batch_size, len(generated))
        ids = active_by_topology[topology_ids[start:stop]]
        edge_index = np.stack([master_index[:, edge_ids] for edge_ids in ids])
        edge_attr = np.stack([master_attr[edge_ids] for edge_ids in ids])
        with torch.no_grad():
            residual = ac_power_flow_residual(
                torch.from_numpy(generated[start:stop].astype(np.float64)),
                torch.from_numpy(edge_index.astype(np.int64)),
                torch.from_numpy(edge_attr.astype(np.float64)),
                base_mva,
            ).abs().cpu().numpy()
        residual_blocks.append(residual.reshape(-1))
        sample_max_blocks.append(np.max(residual, axis=(1, 2)))
        sample_mean_blocks.append(np.mean(residual, axis=(1, 2)))
    residual = np.concatenate(residual_blocks)
    sample_max = np.concatenate(sample_max_blocks)
    sample_mean = np.concatenate(sample_mean_blocks)
    tolerances = sorted(
        {tolerance, 1.0e-2, 5.0e-2, 1.0e-1, 5.0e-1, 1.0, 2.0, 5.0, 10.0}
    )
    feasible_rates = {
        f"{threshold:g}": float(np.mean(sample_max <= threshold))
        for threshold in tolerances
    }
    mean_residual_rates = {
        f"{threshold:g}": float(np.mean(sample_mean <= threshold))
        for threshold in tolerances
    }
    return {
        "mean_absolute_residual": float(residual.mean()),
        "median_absolute_residual": float(np.median(residual)),
        "p95_absolute_residual": float(np.quantile(residual, 0.95)),
        "p99_absolute_residual": float(np.quantile(residual, 0.99)),
        "maximum_absolute_residual": float(residual.max()),
        "sample_max_residual_median": float(np.median(sample_max)),
        "sample_max_residual_p95": float(np.quantile(sample_max, 0.95)),
        "sample_max_residual_p99": float(np.quantile(sample_max, 0.99)),
        "sample_mean_residual_median": float(np.median(sample_mean)),
        "sample_mean_residual_p95": float(np.quantile(sample_mean, 0.95)),
        "sample_mean_residual_p99": float(np.quantile(sample_mean, 0.99)),
        "physics_feasible_rate": feasible_rates[f"{tolerance:g}"],
        "physics_feasible_rate_by_tolerance": feasible_rates,
        "sample_mean_residual_rate_by_tolerance": mean_residual_rates,
        "physics_tolerance_mw_mvar": tolerance,
    }


def _distribution_metrics(real: np.ndarray, generated: np.ndarray) -> dict[str, object]:
    real_mean = real.mean(axis=(0, 1))
    generated_mean = generated.mean(axis=(0, 1))
    real_std = real.std(axis=(0, 1))
    generated_std = generated.std(axis=(0, 1))
    return {
        "wasserstein_per_channel": dict(
            zip(CHANNEL_NAMES, _per_node_wasserstein(real, generated), strict=True)
        ),
        "absolute_mean_error_per_channel": dict(
            zip(CHANNEL_NAMES, np.abs(generated_mean - real_mean).tolist(), strict=True)
        ),
        "absolute_std_error_per_channel": dict(
            zip(CHANNEL_NAMES, np.abs(generated_std - real_std).tolist(), strict=True)
        ),
        "generated_to_real_std_ratio": dict(
            zip(
                CHANNEL_NAMES,
                (generated_std / np.maximum(real_std, 1.0e-12)).tolist(),
                strict=True,
            )
        ),
    }


def evaluate(
    data_path: Path,
    generated_path: Path,
    physics_tolerance: float,
    batch_size: int,
    metric_samples: int,
    seed: int,
) -> dict[str, object]:
    if physics_tolerance <= 0.0 or batch_size <= 0 or metric_samples <= 1:
        raise ValueError("Tolerance and batch size must be positive; metric_samples > 1.")
    with np.load(data_path, allow_pickle=False) as data, np.load(
        generated_path, allow_pickle=False
    ) as generated_file:
        generated = generated_file["generated_x0"].astype(np.float64)
        topology_ids = generated_file["topology_id"].astype(np.int64)
        if generated.shape != (len(topology_ids), data["x0"].shape[1], 4):
            raise ValueError("Generated state shape or topology_id length is inconsistent.")
        if not np.isfinite(generated).all():
            raise ValueError("Generated samples contain NaN or Inf.")
        if topology_ids.min() < 0 or topology_ids.max() >= len(data["topology_split"]):
            raise ValueError("Generated topology_id contains an out-of-range value.")

        reference = _matched_reference(data, topology_ids, seed).astype(np.float64)
        mean, std = _training_normalization(data)
        real_normalized = (reference - mean) / std
        generated_normalized = (generated - mean) / std
        distribution = _distribution_metrics(reference, generated)
        physics = _physics_metrics(
            data,
            generated,
            topology_ids,
            batch_size,
            physics_tolerance,
        )
        voltage_bounds = json.loads(str(data["metadata"].item())).get(
            "voltage_bounds_pu", [0.90, 1.10]
        )
        voltage = generated[..., 2]
        voltage_violation = np.any(
            (voltage < voltage_bounds[0]) | (voltage > voltage_bounds[1]), axis=1
        )
        flat_real = real_normalized.reshape(len(real_normalized), -1)
        flat_generated = generated_normalized.reshape(len(generated_normalized), -1)
        # 平衡节点的 V、theta 可能是常数。常数列的相关系数没有定义，应明确
        # 排除，而不是让 NaN 悄悄进入 JSON 指标。
        valid_correlation_features = (flat_real.std(axis=0) > 1.0e-10) & (
            flat_generated.std(axis=0) > 1.0e-10
        )
        if int(valid_correlation_features.sum()) < 2:
            raise ValueError("Fewer than two non-constant features for correlation metric.")
        correlation_real = np.corrcoef(
            flat_real[:, valid_correlation_features], rowvar=False
        )
        correlation_generated = np.corrcoef(
            flat_generated[:, valid_correlation_features], rowvar=False
        )
        correlation_error = float(
            np.linalg.norm(correlation_generated - correlation_real, ord="fro")
            / correlation_real.shape[0]
        )

        per_topology: list[dict[str, object]] = []
        base_mask = data["topology_edge_mask"][0].astype(bool)
        for topology_id in np.unique(topology_ids):
            positions = np.flatnonzero(topology_ids == topology_id)
            topology_distribution = _distribution_metrics(
                reference[positions], generated[positions]
            )
            mask = data["topology_edge_mask"][topology_id].astype(bool)
            switch_distance = int(np.count_nonzero(mask != base_mask) // 2)
            topology_physics = _physics_metrics(
                data,
                generated[positions],
                topology_ids[positions],
                batch_size,
                physics_tolerance,
            )
            topology_voltage = generated[positions, :, 2]
            topology_voltage_violation = np.any(
                (topology_voltage < voltage_bounds[0])
                | (topology_voltage > voltage_bounds[1]),
                axis=1,
            )
            per_topology.append(
                {
                    "topology_id": int(topology_id),
                    "topology_split": int(data["topology_split"][topology_id]),
                    "switch_distance_from_base": switch_distance,
                    "generated_sample_count": len(positions),
                    "physics": topology_physics,
                    "voltage_violation_rate": float(topology_voltage_violation.mean()),
                    **topology_distribution,
                }
            )

        report: dict[str, object] = {
            "data_path": str(data_path.resolve()),
            "generated_path": str(generated_path.resolve()),
            "generated_sample_count": len(generated),
            "unique_generated_topologies": int(len(np.unique(topology_ids))),
            "generated_topology_split_codes": sorted(
                {int(data["topology_split"][topology_id]) for topology_id in np.unique(topology_ids)}
            ),
            "distribution": distribution,
            "physics": physics,
            "voltage_violation_rate": float(voltage_violation.mean()),
            "voltage_bounds_pu": [float(voltage_bounds[0]), float(voltage_bounds[1])],
            "normalized_correlation_frobenius_error": correlation_error,
            "correlation_feature_count": int(valid_correlation_features.sum()),
            **_mmd_rbf(real_normalized, generated_normalized, metric_samples, seed),
            **_nearest_reference_distance(
                real_normalized, generated_normalized, metric_samples, seed
            ),
            "per_topology": per_topology,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--metric-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        args.data,
        args.generated,
        args.physics_tolerance,
        args.batch_size,
        args.metric_samples,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation written to: {args.output}")
    print(json.dumps({key: report[key] for key in ("generated_sample_count", "unique_generated_topologies", "voltage_violation_rate", "mmd_rbf_squared")}, indent=2))


if __name__ == "__main__":
    main()
