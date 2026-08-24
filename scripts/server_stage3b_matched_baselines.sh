#!/usr/bin/env bash
# 阶段3B：参数量匹配后，从头训练80轮比较 Graph 与 Vector DDPM。

set -euo pipefail

data_path="${1:-data/ieee33_feasible_T300_S500_seed2026.npz}"
epochs="${MATCHED_EPOCHS:-80}"
train_batch="${MATCHED_BATCH_SIZE:-1024}"
sample_batch="${MATCHED_SAMPLE_BATCH_SIZE:-256}"
samples_per_topology="${MATCHED_SAMPLES_PER_TOPOLOGY:-100}"

mkdir -p docs logs outputs

if [[ ! -f "${data_path}" ]]; then
  echo "Formal dataset not found: ${data_path}" >&2
  exit 1
fi

uv run --locked python -m unittest discover -s tests -v

run_experiment() {
  local model_type="$1"
  local hidden_channels="$2"
  local device="$3"
  local output_dir="outputs/formal_matched_${model_type}_H${hidden_channels}_T1000_E${epochs}_seed2026"

  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing experiment: ${output_dir}" >&2
    return 1
  fi

  uv run --locked ieee33-run train \
    --data "${data_path}" \
    --output-dir "${output_dir}" \
    --model-type "${model_type}" \
    --epochs "${epochs}" \
    --batch-size "${train_batch}" \
    --learning-rate 2e-4 \
    --hidden-channels "${hidden_channels}" \
    --num-layers 6 \
    --diffusion-steps 1000 \
    --noise-schedule linear \
    --physics-weight 0 \
    --voltage-weight 0 \
    --device "${device}" \
    --seed 2026

  uv run --locked ieee33-run sample \
    --data "${data_path}" \
    --checkpoint "${output_dir}/best.pt" \
    --output "${output_dir}/generated_validation.npz" \
    --split validation \
    --samples-per-topology "${samples_per_topology}" \
    --batch-size "${sample_batch}" \
    --device "${device}" \
    --seed 2026

  uv run --locked ieee33-evaluate \
    --data "${data_path}" \
    --generated "${output_dir}/generated_validation.npz" \
    --output "${output_dir}/metrics_validation.json" \
    --physics-tolerance 1e-2 \
    --metric-samples 2000 \
    --seed 2026
}

echo "Launching matched Graph(H=128) on cuda:0 and Vector(H=192) on cuda:1."
run_experiment graph 128 cuda:0 > logs/formal_matched_graph.log 2>&1 &
graph_pid=$!
run_experiment vector 192 cuda:1 > logs/formal_matched_vector.log 2>&1 &
vector_pid=$!
echo "graph_pid=${graph_pid} vector_pid=${vector_pid}"

status=0
wait "${graph_pid}" || status=1
wait "${vector_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "At least one matched baseline failed. Inspect model-specific logs." >&2
  exit 1
fi

uv run --locked python - <<PY
import json
from pathlib import Path

import numpy as np

settings = (("graph", 128), ("vector", 192))
summary = []
for model_type, hidden in settings:
    output_dir = Path(
        f"outputs/formal_matched_{model_type}_H{hidden}_T1000_E${epochs}_seed2026"
    )
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (output_dir / "metrics_validation.json").read_text(encoding="utf-8")
    )
    with np.load(output_dir / "generated_validation.npz", allow_pickle=False) as generated:
        sampling = json.loads(str(generated["sampling_metadata"].item()))
    best = min(history, key=lambda row: row["val_total"])
    summary.append(
        {
            "model_type": model_type,
            "hidden_channels": hidden,
            "best_epoch": best["epoch"],
            "best_val_noise": best["val_noise"],
            "training_summary": config["training_summary"],
            "sampling_summary": sampling,
            "wasserstein_per_channel": metrics["distribution"]["wasserstein_per_channel"],
            "generated_to_real_std_ratio": metrics["distribution"]["generated_to_real_std_ratio"],
            "mmd_rbf_squared": metrics["mmd_rbf_squared"],
            "correlation_error": metrics["normalized_correlation_frobenius_error"],
            "mean_absolute_residual": metrics["physics"]["mean_absolute_residual"],
            "p95_absolute_residual": metrics["physics"]["p95_absolute_residual"],
            "sample_max_residual_median": metrics["physics"]["sample_max_residual_median"],
            "sample_max_residual_p95": metrics["physics"]["sample_max_residual_p95"],
            "sample_mean_residual_p95": metrics["physics"]["sample_mean_residual_p95"],
            "max_based_feasible_rate": metrics["physics"]["physics_feasible_rate_by_tolerance"],
            "mean_based_feasible_rate": metrics["physics"]["sample_mean_residual_rate_by_tolerance"],
            "voltage_violation_rate": metrics["voltage_violation_rate"],
        }
    )

destination = Path("docs/formal_validation_matched_baselines_summary.json")
destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Matched baseline summary written to: {destination}")
PY

echo "Stage 3B complete. Test topologies were not sampled."
