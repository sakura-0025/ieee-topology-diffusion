#!/usr/bin/env bash
# 阶段1：审计正式 IEEE33 多拓扑数据，并保存可引用的数据清单。

set -euo pipefail

dataset_path="${1:-data/ieee33_feasible_T300_S500_seed2026.npz}"
manifest_path="${2:-docs/dataset_manifest_T300_S500_seed2026.md}"

if [[ ! -f "${dataset_path}" ]]; then
  echo "Dataset not found: ${dataset_path}" >&2
  exit 1
fi

mkdir -p docs logs

echo "Git commit: $(git rev-parse HEAD)"
echo "Dataset: ${dataset_path}"
echo "Manifest: ${manifest_path}"

uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked ieee33-audit \
  --data "${dataset_path}" \
  --output "${manifest_path}" \
  --physics-samples 2000 \
  --seed 2026

echo "Stage 1 complete. Inspect ${manifest_path} before starting model training."
