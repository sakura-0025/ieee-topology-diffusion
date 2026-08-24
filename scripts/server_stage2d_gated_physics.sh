#!/usr/bin/env bash
# 阶段2D：从稳定基线微调，仅在低噪声时刻施加标幺 AC 物理损失。

set -euo pipefail

dev_data="${1:-data/ieee33_dev_T30_S100_seed2026.npz}"
epochs="${GATED_EPOCHS:-30}"
batch_size="${GATED_BATCH_SIZE:-64}"
device="${GATED_DEVICE:-auto}"
weights="${GATED_WEIGHTS:-1 3 10}"
alpha_min="${GATED_ALPHA_MIN:-0.5}"
learning_rate="${GATED_LEARNING_RATE:-5e-5}"
baseline_dir="outputs/pilot_schedule_linear_T1000_E100_seed2026"
baseline_checkpoint="${baseline_dir}/best.pt"

mkdir -p docs logs outputs

if [[ ! -f "${dev_data}" || ! -f "${baseline_checkpoint}" ]]; then
  echo "Development data or stable baseline checkpoint is missing." >&2
  exit 1
fi

uv run --locked python -m unittest discover -s tests -v

for physics_weight in ${weights}; do
  experiment_id="pilot_gated_a${alpha_min}_pf${physics_weight}_E${epochs}_seed2026"
  output_dir="outputs/${experiment_id}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing experiment: ${output_dir}" >&2
    exit 1
  fi

  uv run --locked ieee33-run train \
    --data "${dev_data}" \
    --output-dir "${output_dir}" \
    --init-checkpoint "${baseline_checkpoint}" \
    --model-type graph \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --learning-rate "${learning_rate}" \
    --hidden-channels 64 \
    --num-layers 4 \
    --diffusion-steps 1000 \
    --noise-schedule linear \
    --physics-weight "${physics_weight}" \
    --voltage-weight 0 \
    --physics-time-weight alpha_bar \
    --physics-alpha-bar-min "${alpha_min}" \
    --pf-scale-mw 10 \
    --pf-scale-mvar 10 \
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

summary = []
for output_dir in sorted(Path("outputs").glob("pilot_gated_a*_pf*_E*_seed2026")):
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_best.json").read_text(encoding="utf-8"))
    loss_config = config["training_loss_config"]
    weight = float(loss_config["physics_weight"])
    best = min(history, key=lambda row: row["val_total"])
    summary.append(
        {
            "experiment": output_dir.name,
            "physics_weight": weight,
            "physics_alpha_bar_min": loss_config["physics_alpha_bar_min"],
            "best_epoch_in_finetuning": best["epoch"],
            "best_val_noise": best["val_noise"],
            "best_val_physics_per_unit_squared": best["val_physics"],
            "weighted_physics_contribution": weight * best["val_physics"],
            "wasserstein_per_channel": metrics["distribution"]["wasserstein_per_channel"],
            "generated_to_real_std_ratio": metrics["distribution"]["generated_to_real_std_ratio"],
            "mmd_rbf_squared": metrics["mmd_rbf_squared"],
            "correlation_error": metrics["normalized_correlation_frobenius_error"],
            "mean_absolute_residual": metrics["physics"]["mean_absolute_residual"],
            "p95_absolute_residual": metrics["physics"]["p95_absolute_residual"],
            "p99_absolute_residual": metrics["physics"]["p99_absolute_residual"],
            "sample_max_residual_median": metrics["physics"]["sample_max_residual_median"],
            "sample_max_residual_p95": metrics["physics"]["sample_max_residual_p95"],
            "feasible_rate_by_tolerance": metrics["physics"]["physics_feasible_rate_by_tolerance"],
            "voltage_violation_rate": metrics["voltage_violation_rate"],
        }
    )

destination = Path("docs/pilot_gated_physics_summary.json")
destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Gated physics summary written to: {destination}")
PY

echo "Stage 2D complete. Return docs/pilot_gated_physics_summary.json."
