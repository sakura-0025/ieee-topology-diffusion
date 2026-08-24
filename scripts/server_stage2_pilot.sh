#!/usr/bin/env bash
# 阶段2：在30拓扑开发集上搜索物理损失量级，不接触正式数据集。
# 本阶段采用单因素设计：只改变 AC 物理损失权重，电压损失固定为 0。

set -euo pipefail

dev_data="${1:-data/ieee33_dev_T30_S100_seed2026.npz}"
epochs="${PILOT_EPOCHS:-30}"
batch_size="${PILOT_BATCH_SIZE:-64}"
device="${PILOT_DEVICE:-auto}"

mkdir -p data docs logs outputs

if [[ ! -f "${dev_data}" ]]; then
  uv run --locked ieee33-build \
    --output "${dev_data}" \
    --num-topologies 30 \
    --samples-per-topology 100 \
    --load-low 0.80 \
    --load-high 1.20 \
    --voltage-min 0.90 \
    --voltage-max 1.10 \
    --seed 2026
fi

uv run --locked ieee33-audit \
  --data "${dev_data}" \
  --output docs/dataset_manifest_dev_T30_S100_seed2026.md \
  --physics-samples 512 \
  --seed 2026

for physics_weight in 0 1e-5 1e-4 1e-3; do
  experiment_id="pilot_graph_pf${physics_weight}_seed2026"
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
    --diffusion-steps 100 \
    --physics-weight "${physics_weight}" \
    --voltage-weight 0 \
    --physics-time-weight alpha_bar \
    --device "${device}" \
    --seed 2026

  uv run --locked ieee33-run sample \
    --data "${dev_data}" \
    --checkpoint "${output_dir}/best.pt" \
    --output "${output_dir}/generated_equal.npz" \
    --samples-per-topology 100 \
    --device "${device}" \
    --seed 2026

  uv run --locked ieee33-evaluate \
    --data "${dev_data}" \
    --generated "${output_dir}/generated_equal.npz" \
    --output "${output_dir}/metrics.json" \
    --physics-tolerance 1e-2 \
    --metric-samples 600 \
    --seed 2026
done

uv run --locked python - <<'PY'
import json
from pathlib import Path

summary = []
for output_dir in sorted(Path("outputs").glob("pilot_graph_pf*_seed2026")):
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    best = min(history, key=lambda row: row["val_total"])
    summary.append(
        {
            "experiment": output_dir.name,
            "physics_weight": config["training_loss_config"]["physics_weight"],
            "best_epoch": best["epoch"],
            "best_val_total": best["val_total"],
            "best_val_noise": best["val_noise"],
            "best_val_physics": best["val_physics"],
            "wasserstein_per_channel": metrics["distribution"]["wasserstein_per_channel"],
            "generated_to_real_std_ratio": metrics["distribution"]["generated_to_real_std_ratio"],
            "mmd_rbf_squared": metrics["mmd_rbf_squared"],
            "correlation_error": metrics["normalized_correlation_frobenius_error"],
            "mean_absolute_residual": metrics["physics"]["mean_absolute_residual"],
            "p95_absolute_residual": metrics["physics"]["p95_absolute_residual"],
            "p99_absolute_residual": metrics["physics"]["p99_absolute_residual"],
            "maximum_absolute_residual": metrics["physics"]["maximum_absolute_residual"],
            "physics_feasible_rate_at_1e-2": metrics["physics"]["physics_feasible_rate"],
            "voltage_violation_rate": metrics["voltage_violation_rate"],
            "nearest_reference_mean": metrics["nearest_reference_mean"],
        }
    )

destination = Path("docs/pilot_physics_weight_summary.json")
destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Pilot summary written to: {destination}")
PY

echo "Stage 2 complete. Return docs/pilot_physics_weight_summary.json for weight selection."
