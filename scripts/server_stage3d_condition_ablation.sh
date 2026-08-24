#!/usr/bin/env bash
# 阶段3D：无需训练，以相同噪声比较正确、基础和错配拓扑条件。

set -euo pipefail

data_path="${1:-data/ieee33_feasible_T300_S500_seed2026.npz}"
epochs="${MATCHED_EPOCHS:-80}"
sample_batch="${CONDITION_SAMPLE_BATCH_SIZE:-256}"
samples_per_topology="${CONDITION_SAMPLES_PER_TOPOLOGY:-100}"

mkdir -p docs logs

run_ablation() {
  local model_type="$1"
  local hidden="$2"
  local device="$3"
  local output_dir="outputs/formal_matched_${model_type}_H${hidden}_T1000_E${epochs}_seed2026"

  for condition_mode in base shuffled; do
    uv run --locked ieee33-run sample \
      --data "${data_path}" \
      --checkpoint "${output_dir}/best.pt" \
      --output "${output_dir}/generated_validation_condition_${condition_mode}.npz" \
      --split validation \
      --condition-mode "${condition_mode}" \
      --samples-per-topology "${samples_per_topology}" \
      --batch-size "${sample_batch}" \
      --device "${device}" \
      --seed 2026

    uv run --locked ieee33-evaluate \
      --data "${data_path}" \
      --generated "${output_dir}/generated_validation_condition_${condition_mode}.npz" \
      --output "${output_dir}/metrics_validation_condition_${condition_mode}.json" \
      --physics-tolerance 1e-2 \
      --metric-samples 2000 \
      --seed 2026
  done
}

run_ablation graph 128 cuda:0 > logs/formal_graph_condition_ablation.log 2>&1 &
graph_pid=$!
run_ablation vector 192 cuda:1 > logs/formal_vector_condition_ablation.log 2>&1 &
vector_pid=$!

status=0
wait "${graph_pid}" || status=1
wait "${vector_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "Condition ablation failed. Inspect model-specific logs." >&2
  exit 1
fi

uv run --locked python - <<PY
import json
from pathlib import Path

import numpy as np

channels = ("p_injection_mw", "q_injection_mvar", "voltage_pu", "theta_rad")


def record(report):
    return {
        "wasserstein_per_channel": report["distribution"]["wasserstein_per_channel"],
        "absolute_mean_error_per_channel": report["distribution"][
            "absolute_mean_error_per_channel"
        ],
        "mmd_rbf_squared": report["mmd_rbf_squared"],
        "correlation_error": report["normalized_correlation_frobenius_error"],
        "mean_absolute_residual": report["physics"]["mean_absolute_residual"],
        "p95_absolute_residual": report["physics"]["p95_absolute_residual"],
        "sample_max_residual_p95": report["physics"]["sample_max_residual_p95"],
        "voltage_violation_rate": report["voltage_violation_rate"],
        "per_topology_wasserstein_mean": {
            channel: float(
                np.mean(
                    [
                        row["wasserstein_per_channel"][channel]
                        for row in report["per_topology"]
                    ]
                )
            )
            for channel in channels
        },
    }


result = {}
for model_type, hidden in (("graph", 128), ("vector", 192)):
    output_dir = Path(
        f"outputs/formal_matched_{model_type}_H{hidden}_T1000_E${epochs}_seed2026"
    )
    reports = {
        "correct": json.loads(
            (output_dir / "metrics_validation.json").read_text(encoding="utf-8")
        ),
        "base": json.loads(
            (output_dir / "metrics_validation_condition_base.json").read_text(
                encoding="utf-8"
            )
        ),
        "shuffled": json.loads(
            (output_dir / "metrics_validation_condition_shuffled.json").read_text(
                encoding="utf-8"
            )
        ),
    }
    records = {mode: record(report) for mode, report in reports.items()}
    correct = records["correct"]
    degradation = {}
    for mode in ("base", "shuffled"):
        current = records[mode]
        degradation[mode] = {
            "wasserstein_ratio_to_correct": {
                channel: current["wasserstein_per_channel"][channel]
                / correct["wasserstein_per_channel"][channel]
                for channel in channels
            },
            "per_topology_wasserstein_ratio_to_correct": {
                channel: current["per_topology_wasserstein_mean"][channel]
                / correct["per_topology_wasserstein_mean"][channel]
                for channel in channels
            },
            "mmd_ratio_to_correct": current["mmd_rbf_squared"]
            / correct["mmd_rbf_squared"],
            "physics_mean_ratio_to_correct": current["mean_absolute_residual"]
            / correct["mean_absolute_residual"],
            "sample_max_p95_ratio_to_correct": current["sample_max_residual_p95"]
            / correct["sample_max_residual_p95"],
        }
    result[model_type] = {
        "conditions": records,
        "degradation_vs_correct": degradation,
    }

destination = Path("docs/formal_validation_condition_ablation.json")
destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"Condition ablation summary written to: {destination}")
PY

echo "Stage 3D complete. Test topologies were not sampled."
