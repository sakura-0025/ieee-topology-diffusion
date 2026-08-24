"""审计 IEEE33 多拓扑潮流数据，并生成可追溯的 Markdown 清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from .physics import ac_power_flow_residual, dataset_base_mva


REQUIRED_ARRAYS = {
    "metadata",
    "node_static",
    "master_edge_index",
    "master_edge_attr",
    "topology_edge_mask",
    "topology_active_edge_ids",
    "topology_split",
    "x0",
    "node_raw",
    "sample_topology_id",
    "sample_split",
    "sample_quality",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_statistics(values: np.ndarray) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for channel in range(values.shape[-1]):
        flattened = values[..., channel].astype(np.float64).reshape(-1)
        result.append(
            {
                "min": float(flattened.min()),
                "mean": float(flattened.mean()),
                "std": float(flattened.std()),
                "max": float(flattened.max()),
            }
        )
    return result


def _topology_checks(
    edge_index: np.ndarray,
    masks: np.ndarray,
    active_edge_ids: np.ndarray,
    num_nodes: int,
) -> dict[str, int | bool]:
    radial_count = 0
    connected_count = 0
    active_ids_match_count = 0
    for mask, edge_ids in zip(masks.astype(bool), active_edge_ids, strict=True):
        active = np.flatnonzero(mask)
        active_ids_match_count += int(np.array_equal(active, edge_ids))
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(
            (int(edge_index[0, edge_id]), int(edge_index[1, edge_id]))
            for edge_id in active
        )
        connected_count += int(nx.is_connected(graph))
        radial_count += int(nx.is_tree(graph))
    packed = [np.packbits(mask.astype(np.uint8)).tobytes() for mask in masks]
    return {
        "all_edge_counts_n_minus_one": bool(np.all(masks.sum(axis=1) == num_nodes - 1)),
        "connected_count": connected_count,
        "radial_count": radial_count,
        "active_ids_match_count": active_ids_match_count,
        "unique_topology_count": len(set(packed)),
    }


def _physics_audit(
    data: np.lib.npyio.NpzFile,
    sample_count: int,
    seed: int,
) -> dict[str, float | int]:
    total = len(data["x0"])
    count = min(sample_count, total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=count, replace=False)
    topology_ids = data["sample_topology_id"][indices]
    active_ids = data["topology_active_edge_ids"][topology_ids]
    master_index = data["master_edge_index"]
    master_attr = data["master_edge_attr"]
    edge_index = np.stack([master_index[:, ids] for ids in active_ids])
    edge_attr = np.stack([master_attr[ids] for ids in active_ids])
    with torch.no_grad():
        residual = ac_power_flow_residual(
            torch.from_numpy(data["x0"][indices].astype(np.float64)),
            torch.from_numpy(edge_index.astype(np.int64)),
            torch.from_numpy(edge_attr.astype(np.float64)),
            dataset_base_mva(data),
        ).abs().cpu().numpy()
    flattened = residual.reshape(-1)
    return {
        "sample_count": count,
        "mean_absolute_residual": float(flattened.mean()),
        "p95_absolute_residual": float(np.quantile(flattened, 0.95)),
        "maximum_absolute_residual": float(flattened.max()),
    }


def audit_dataset(path: Path, physics_samples: int, seed: int) -> dict[str, object]:
    """执行结构、拓扑、质量和物理一致性审计；失败时抛出异常。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_ARRAYS.difference(data.files))
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {missing}")
        metadata = json.loads(str(data["metadata"].item()))
        num_nodes = int(metadata["num_nodes"])
        num_topologies = int(metadata["num_topologies"])
        num_samples = int(data["x0"].shape[0])
        num_edges = int(metadata["num_candidate_edges"])

        expected_shapes = {
            "node_static": (num_nodes, 4),
            "master_edge_index": (2, num_edges),
            "master_edge_attr": (num_edges, 4),
            "topology_edge_mask": (num_topologies, num_edges),
            "topology_active_edge_ids": (num_topologies, num_nodes - 1),
            "topology_split": (num_topologies,),
            "x0": (num_samples, num_nodes, 4),
            "node_raw": (num_samples, num_nodes, 6),
            "sample_topology_id": (num_samples,),
            "sample_split": (num_samples,),
            "sample_quality": (num_samples, 4),
        }
        shape_errors = {
            name: {"actual": tuple(data[name].shape), "expected": expected}
            for name, expected in expected_shapes.items()
            if tuple(data[name].shape) != expected
        }
        if shape_errors:
            raise ValueError(f"Array shape mismatch: {shape_errors}")
        finite_arrays = ("node_static", "master_edge_attr", "x0", "node_raw")
        nonfinite = {name: int((~np.isfinite(data[name])).sum()) for name in finite_arrays}
        if any(nonfinite.values()):
            raise ValueError(f"Dataset contains NaN or Inf: {nonfinite}")

        topology_ids = data["sample_topology_id"].astype(np.int64)
        if topology_ids.min() < 0 or topology_ids.max() >= num_topologies:
            raise ValueError("sample_topology_id contains an out-of-range value.")
        inherited_split = data["topology_split"][topology_ids]
        split_consistent = bool(np.array_equal(inherited_split, data["sample_split"]))
        if not split_consistent:
            raise ValueError("sample_split does not inherit topology_split.")

        topology = _topology_checks(
            data["master_edge_index"],
            data["topology_edge_mask"],
            data["topology_active_edge_ids"],
            num_nodes,
        )
        if topology["radial_count"] != num_topologies:
            raise ValueError("At least one topology is not a spanning tree.")
        if topology["unique_topology_count"] != num_topologies:
            raise ValueError("Duplicate topology masks are present.")
        if topology["active_ids_match_count"] != num_topologies:
            raise ValueError("Active edge IDs do not match topology masks.")

        topology_split_counts = {
            name: int(np.sum(data["topology_split"] == code))
            for code, name in enumerate(("train", "validation", "test"))
        }
        sample_split_counts = {
            name: int(np.sum(data["sample_split"] == code))
            for code, name in enumerate(("train", "validation", "test"))
        }
        samples_per_topology = np.bincount(topology_ids, minlength=num_topologies)
        quality_sums = data["sample_quality"].astype(np.int64).sum(axis=0)
        physics = _physics_audit(data, physics_samples, seed)
        base_mva = dataset_base_mva(data)
        x0_stats = _channel_statistics(data["x0"])

    return {
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "metadata": metadata,
        "base_mva": base_mva,
        "topology_split_counts": topology_split_counts,
        "sample_split_counts": sample_split_counts,
        "samples_per_topology_min": int(samples_per_topology.min()),
        "samples_per_topology_max": int(samples_per_topology.max()),
        "split_consistent": split_consistent,
        "topology_checks": topology,
        "nonfinite_counts": nonfinite,
        "quality_sums": quality_sums.tolist(),
        "x0_statistics": x0_stats,
        "physics_audit": physics,
        "status": "PASS",
    }


def _format_report(report: dict[str, object]) -> str:
    metadata = report["metadata"]
    assert isinstance(metadata, dict)
    topology = report["topology_checks"]
    assert isinstance(topology, dict)
    physics = report["physics_audit"]
    assert isinstance(physics, dict)
    x0_stats = report["x0_statistics"]
    assert isinstance(x0_stats, list)
    channels = metadata["x0_channels"]

    lines = [
        "# IEEE33 数据集审计清单",
        "",
        f"- 状态：**{report['status']}**",
        f"- 文件：`{report['path']}`",
        f"- 文件大小：{report['file_size_bytes']:,} bytes",
        f"- SHA256：`{report['sha256']}`",
        f"- 算例：{metadata['case']}",
        f"- 基准容量：{report['base_mva']} MVA",
        f"- 拓扑数：{metadata['num_topologies']}",
        f"- 样本数：{metadata['num_topologies'] * metadata['samples_per_topology']}",
        "",
        "## 划分与拓扑检查",
        "",
        f"- 拓扑划分：{report['topology_split_counts']}",
        f"- 样本划分：{report['sample_split_counts']}",
        f"- 每拓扑样本数范围：{report['samples_per_topology_min']}—{report['samples_per_topology_max']}",
        f"- 样本划分继承拓扑划分：{report['split_consistent']}",
        f"- 连通拓扑数：{topology['connected_count']}",
        f"- 辐射拓扑数：{topology['radial_count']}",
        f"- 唯一拓扑数：{topology['unique_topology_count']}",
        f"- 活动边索引与掩码一致数：{topology['active_ids_match_count']}",
        "",
        "## 潮流通道统计",
        "",
        "| 通道 | 最小值 | 均值 | 标准差 | 最大值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for channel, stats in zip(channels, x0_stats, strict=True):
        lines.append(
            f"| {channel} | {stats['min']:.8g} | {stats['mean']:.8g} | "
            f"{stats['std']:.8g} | {stats['max']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## 交流潮流残差抽查",
            "",
            f"- 抽查样本数：{physics['sample_count']}",
            f"- 平均绝对残差：{physics['mean_absolute_residual']:.8g} MW/Mvar",
            f"- 95%分位绝对残差：{physics['p95_absolute_residual']:.8g} MW/Mvar",
            f"- 最大绝对残差：{physics['maximum_absolute_residual']:.8g} MW/Mvar",
            "",
            "## 质量标记",
            "",
            f"`[潮流收敛, 连通, 辐射, 电压合格]` 的样本计数：`{report['quality_sums']}`。",
            "",
            "> 本文件由 `ieee33-audit` 自动生成。数据文件变化后必须重新生成。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/dataset_manifest.md"))
    parser.add_argument("--physics-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.physics_samples <= 0:
        raise ValueError("--physics-samples must be positive.")
    report = audit_dataset(args.data, args.physics_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_format_report(report), encoding="utf-8")
    print(f"Dataset audit PASS: {args.data}")
    print(f"Manifest written to: {args.output}")


if __name__ == "__main__":
    main()
