#!/usr/bin/env bash
# 阶段2：在30拓扑开发集上搜索物理损失量级，不接触正式测试结果。

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

  voltage_weight="0"
  if [[ "${physics_weight}" != "0" ]]; then
    voltage_weight="1.0"
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
    --voltage-weight "${voltage_weight}" \
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

echo "Stage 2 complete. Return the four metrics.json files for weight selection."
