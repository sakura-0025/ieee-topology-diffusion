#!/usr/bin/env bash
# 阶段2B：校准扩散日程是否能稳定生成分布，不启用任何物理约束。

set -euo pipefail

dev_data="${1:-data/ieee33_dev_T30_S100_seed2026.npz}"
epochs="${SCHEDULE_EPOCHS:-100}"
batch_size="${SCHEDULE_BATCH_SIZE:-64}"
device="${SCHEDULE_DEVICE:-auto}"
schedule_type="${SCHEDULE_TYPE:-cosine}"
diffusion_steps="${SCHEDULE_STEPS:-100}"
experiment_id="pilot_schedule_${schedule_type}_T${diffusion_steps}_E${epochs}_seed2026"
output_dir="outputs/${experiment_id}"
summary_path="docs/pilot_schedule_${schedule_type}_T${diffusion_steps}_summary.json"

mkdir -p docs logs outputs

if [[ ! -f "${dev_data}" ]]; then
  echo "Development dataset not found: ${dev_data}" >&2
  echo "Run scripts/server_stage2_pilot.sh first." >&2
  exit 1
fi
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing experiment: ${output_dir}" >&2
  exit 1
fi

uv run --locked python -m unittest discover -s tests -v

uv run --locked ieee33-run train \
  --data "${dev_data}" \
  --output-dir "${output_dir}" \
  --model-type graph \
  --epochs "${epochs}" \
  --batch-size "${batch_size}" \
  --learning-rate 2e-4 \
  --hidden-channels 64 \
  --num-layers 4 \
  --diffusion-steps "${diffusion_steps}" \
  --noise-schedule "${schedule_type}" \
  --physics-weight 0 \
  --voltage-weight 0 \
  --device "${device}" \
  --seed 2026

for checkpoint_name in best last; do
  generated="${output_dir}/generated_${checkpoint_name}.npz"
  metrics="${output_dir}/metrics_${checkpoint_name}.json"
  uv run --locked ieee33-run sample \
    --data "${dev_data}" \
    --checkpoint "${output_dir}/${checkpoint_name}.pt" \
    --output "${generated}" \
    --samples-per-topology 100 \
    --device "${device}" \
    --seed 2026
  uv run --locked ieee33-evaluate \
    --data "${dev_data}" \
    --generated "${generated}" \
    --output "${metrics}" \
    --physics-tolerance 1e-2 \
    --metric-samples 600 \
    --seed 2026
done

uv run --locked python - <<PY
import json
from pathlib import Path

from ieee33_diffusion.model import GaussianDiffusion, ModelConfig, TopologyConditionedDenoiser

output_dir = Path("${output_dir}")
config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
model_config = ModelConfig(**config["model_config"])
schedule = GaussianDiffusion(TopologyConditionedDenoiser(model_config))
summary = {
    "experiment": output_dir.name,
    "terminal_alpha_bar": float(schedule.alpha_bars[-1]),
    "best_epoch": min(history, key=lambda row: row["val_total"])["epoch"],
    "first_epoch": history[0],
    "last_epoch": history[-1],
    "checkpoints": {},
}
for checkpoint_name in ("best", "last"):
    metrics = json.loads(
        (output_dir / f"metrics_{checkpoint_name}.json").read_text(encoding="utf-8")
    )
    summary["checkpoints"][checkpoint_name] = {
        "wasserstein_per_channel": metrics["distribution"]["wasserstein_per_channel"],
        "generated_to_real_std_ratio": metrics["distribution"]["generated_to_real_std_ratio"],
        "mmd_rbf_squared": metrics["mmd_rbf_squared"],
        "correlation_error": metrics["normalized_correlation_frobenius_error"],
        "mean_absolute_residual": metrics["physics"]["mean_absolute_residual"],
        "p95_absolute_residual": metrics["physics"]["p95_absolute_residual"],
        "maximum_absolute_residual": metrics["physics"]["maximum_absolute_residual"],
        "physics_feasible_rate_at_1e-2": metrics["physics"]["physics_feasible_rate"],
        "voltage_violation_rate": metrics["voltage_violation_rate"],
    }
Path("${summary_path}").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Schedule summary written to: ${summary_path}")
PY

echo "Stage 2B complete. Return ${summary_path}."
