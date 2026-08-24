#!/usr/bin/env bash
# 阶段3C：不重新训练，分析参数匹配模型在各 validation 拓扑上的条件性能。

set -euo pipefail

epochs="${MATCHED_EPOCHS:-80}"
graph_metrics="outputs/formal_matched_graph_H128_T1000_E${epochs}_seed2026/metrics_validation.json"
vector_metrics="outputs/formal_matched_vector_H192_T1000_E${epochs}_seed2026/metrics_validation.json"
destination="docs/formal_validation_topology_diagnostics.json"

if [[ ! -f "${graph_metrics}" || ! -f "${vector_metrics}" ]]; then
  echo "Matched validation metrics are missing." >&2
  exit 1
fi

uv run --locked python - <<PY
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

channels = ("p_injection_mw", "q_injection_mvar", "voltage_pu", "theta_rad")


def quantiles(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def model_summary(report):
    rows = report["per_topology"]
    by_distance = defaultdict(list)
    for row in rows:
        by_distance[int(row["switch_distance_from_base"])].append(row)

    distance_summary = {}
    for distance, group in sorted(by_distance.items()):
        distance_summary[str(distance)] = {
            "topology_count": len(group),
            "wasserstein_mean": {
                channel: float(
                    np.mean([item["wasserstein_per_channel"][channel] for item in group])
                )
                for channel in channels
            },
            "absolute_mean_error_mean": {
                channel: float(
                    np.mean(
                        [item["absolute_mean_error_per_channel"][channel] for item in group]
                    )
                )
                for channel in channels
            },
            "physics_mean_absolute_residual": float(
                np.mean([item["physics"]["mean_absolute_residual"] for item in group])
            ),
            "sample_max_residual_p95_mean": float(
                np.mean([item["physics"]["sample_max_residual_p95"] for item in group])
            ),
            "voltage_violation_rate_mean": float(
                np.mean([item["voltage_violation_rate"] for item in group])
            ),
        }

    return {
        "topology_count": len(rows),
        "wasserstein_across_topologies": {
            channel: quantiles(
                [row["wasserstein_per_channel"][channel] for row in rows]
            )
            for channel in channels
        },
        "absolute_mean_error_across_topologies": {
            channel: quantiles(
                [row["absolute_mean_error_per_channel"][channel] for row in rows]
            )
            for channel in channels
        },
        "physics_mean_residual_across_topologies": quantiles(
            [row["physics"]["mean_absolute_residual"] for row in rows]
        ),
        "sample_max_residual_p95_across_topologies": quantiles(
            [row["physics"]["sample_max_residual_p95"] for row in rows]
        ),
        "voltage_violation_across_topologies": quantiles(
            [row["voltage_violation_rate"] for row in rows]
        ),
        "by_switch_distance": distance_summary,
        "worst_voltage_wasserstein_topologies": [
            {
                "topology_id": int(row["topology_id"]),
                "switch_distance": int(row["switch_distance_from_base"]),
                "voltage_wasserstein": row["wasserstein_per_channel"]["voltage_pu"],
                "voltage_violation_rate": row["voltage_violation_rate"],
            }
            for row in sorted(
                rows,
                key=lambda item: item["wasserstein_per_channel"]["voltage_pu"],
                reverse=True,
            )[:5]
        ],
        "worst_physics_topologies": [
            {
                "topology_id": int(row["topology_id"]),
                "switch_distance": int(row["switch_distance_from_base"]),
                "mean_absolute_residual": row["physics"]["mean_absolute_residual"],
                "sample_max_residual_p95": row["physics"]["sample_max_residual_p95"],
            }
            for row in sorted(
                rows,
                key=lambda item: item["physics"]["sample_max_residual_p95"],
                reverse=True,
            )[:5]
        ],
    }


graph = json.loads(Path("${graph_metrics}").read_text(encoding="utf-8"))
vector = json.loads(Path("${vector_metrics}").read_text(encoding="utf-8"))
graph_rows = {int(row["topology_id"]): row for row in graph["per_topology"]}
vector_rows = {int(row["topology_id"]): row for row in vector["per_topology"]}
if graph_rows.keys() != vector_rows.keys():
    raise ValueError("Graph and vector reports do not contain identical topologies.")

comparison = {
    "graph_lower_wasserstein_rate": {
        channel: float(
            np.mean(
                [
                    graph_rows[topology]["wasserstein_per_channel"][channel]
                    < vector_rows[topology]["wasserstein_per_channel"][channel]
                    for topology in graph_rows
                ]
            )
        )
        for channel in channels
    },
    "graph_lower_absolute_mean_error_rate": {
        channel: float(
            np.mean(
                [
                    graph_rows[topology]["absolute_mean_error_per_channel"][channel]
                    < vector_rows[topology]["absolute_mean_error_per_channel"][channel]
                    for topology in graph_rows
                ]
            )
        )
        for channel in channels
    },
    "graph_lower_mean_physics_residual_rate": float(
        np.mean(
            [
                graph_rows[topology]["physics"]["mean_absolute_residual"]
                < vector_rows[topology]["physics"]["mean_absolute_residual"]
                for topology in graph_rows
            ]
        )
    ),
    "graph_lower_sample_max_p95_rate": float(
        np.mean(
            [
                graph_rows[topology]["physics"]["sample_max_residual_p95"]
                < vector_rows[topology]["physics"]["sample_max_residual_p95"]
                for topology in graph_rows
            ]
        )
    ),
    "graph_lower_voltage_violation_rate": float(
        np.mean(
            [
                graph_rows[topology]["voltage_violation_rate"]
                < vector_rows[topology]["voltage_violation_rate"]
                for topology in graph_rows
            ]
        )
    ),
}

result = {
    "graph": model_summary(graph),
    "vector": model_summary(vector),
    "paired_comparison": comparison,
}
Path("${destination}").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("Topology diagnostics written to: ${destination}")
PY
