#!/usr/bin/env bash
# 阶段2C：在线性1000步稳定基线上，以单因素方式扫描 AC 物理损失权重。

set -euo pipefail

dev_data="${1:-data/ieee33_dev_T30_S100_seed2026.npz}"
epochs="${PHYSICS_EPOCHS:-100}"
batch_size="${PHYSICS_BATCH_SIZE:-64}"
device="${PHYSICS_DEVICE:-auto}"
weights="${PHYSICS_WEIGHTS:-1e-4 1e-3 1e-2}"
baseline_dir="outputs/pilot_schedule_linear_T1000_E100_seed2026"

mkdir -p docs logs outputs

if [[ ! -f "${dev_data}" ]]; then
  echo "Development dataset not found: ${dev_data}" >&2
  exit 1
fi
if [[ ! -f "${baseline_dir}/metrics_best.json" ]]; then
  echo "Stable linear-T1000 baseline metrics not found: ${baseline_dir}/metrics_best.json" >&2
  exit 1
fi

uv run --locked python -m unittest discover -s tests -v

# 使用新版多容差指标重新评价已有基线，不重新训练或采样。
uv run --locked ieee33-evaluate \
  --data "${dev_data}" \
  --generated "${baseline_dir}/generated_best.npz" \
  --output "${baseline_dir}/metrics_best_multitolerance.json" \
  --physics-tolerance 1e-2 \
  --metric-samples 600 \
  --seed 2026

for physics_weight in ${weights}; do
  experiment_id="pilot_linear_T1000_pf${physics_weight}_E${epochs}_seed2026"
  output_dir="outputs/${experiment_id}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing experiment: ${output_dir}" >&2
    exit 1
  fi

  uv run --locked ieee33-run train \
    --data "${dev_data}" \
    --output-dir "${output_dir}" \
    --model-type graph \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --learning-rate 2e-4 \
    --hidden-channels 64 \
    --num-layers 4 \
    --diffusion-steps 1000 \
    --noise-schedule linear \
    --physics-weight "${physics_weight}" \
    --voltage-weight 0 \
    --physics-time-weight alpha_bar \
    --device "${device}" \
    --seed 2026

  uv run --locked ieee33-run sample \
    --data "${dev_data}" \
    --checkpoint "${output_dir}/best.pt" \
    --output "${output_dir}/generated_best.npz" \
    --samples-per-topology 100 \
    --device "${device}" \
    --seed 2026

  uv run --locked ieee33-evaluate \
    --data "${dev_data}" \
    --generated "${output_dir}/generated_best.npz" \
    --output "${output_dir}/metrics_best.json" \
    --physics-tolerance 1e-2 \
    --metric-samples 600 \
    --seed 2026
done

uv run --locked python - <<'PY'
import json
from pathlib import Path


def metric_record(name, weight, metrics, history=None):
    record = {
        "experiment": name,
        "physics_weight": weight,
        "wasserstein_per_channel": metrics["distribution"]["wasserstein_per_channel"],
        "generated_to_real_std_ratio": metrics["distribution"]["generated_to_real_std_ratio"],
        "mmd_rbf_squared": metrics["mmd_rbf_squared"],
        "correlation_error": metrics["normalized_correlation_frobenius_error"],
        "mean_absolute_residual": metrics["physics"]["mean_absolute_residual"],
        "p95_absolute_residual": metrics["physics"]["p95_absolute_residual"],
        "p99_absolute_residual": metrics["physics"]["p99_absolute_residual"],
        "maximum_absolute_residual": metrics["physics"]["maximum_absolute_residual"],
        "sample_max_residual_median": metrics["physics"]["sample_max_residual_median"],
        "sample_max_residual_p95": metrics["physics"]["sample_max_residual_p95"],
        "feasible_rate_by_tolerance": metrics["physics"]["physics_feasible_rate_by_tolerance"],
        "voltage_violation_rate": metrics["voltage_violation_rate"],
    }
    if history is not None:
        best = min(history, key=lambda row: row["val_total"])
        record.update(
            {
                "best_epoch": best["epoch"],
                "best_val_total": best["val_total"],
                "best_val_noise": best["val_noise"],
                "best_val_physics": best["val_physics"],
                "weighted_physics_contribution": weight * best["val_physics"],
            }
        )
    return record


baseline_dir = Path("outputs/pilot_schedule_linear_T1000_E100_seed2026")
baseline_metrics = json.loads(
    (baseline_dir / "metrics_best_multitolerance.json").read_text(encoding="utf-8")
)
baseline_history = json.loads(
    (baseline_dir / "history.json").read_text(encoding="utf-8")
)

summary = [metric_record(baseline_dir.name, 0.0, baseline_metrics, baseline_history)]
for output_dir in sorted(Path("outputs").glob("pilot_linear_T1000_pf*_E*_seed2026")):
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_best.json").read_text(encoding="utf-8"))
    weight = float(config["training_loss_config"]["physics_weight"])
    summary.append(metric_record(output_dir.name, weight, metrics, history))

destination = Path("docs/pilot_physics_linear_T1000_summary.json")
destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Physics summary written to: {destination}")
PY

echo "Stage 2C complete. Return docs/pilot_physics_linear_T1000_summary.json."
