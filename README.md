# IEEE 33 节点变拓扑扩散实验

本项目以 Baran–Wu IEEE 33 节点配电系统为第一阶段算例，使用 `uv` 管理环境。完整候选图包含 33 个节点、37 条候选支路；每个合法辐射拓扑投入 32 条物理支路。

## 文件分工

- `src/ieee33_diffusion/build_dataset.py`：使用支路交换生成辐射拓扑，施加负荷扰动并计算 AC 潮流。
- `src/ieee33_diffusion/model.py`：拓扑条件图去噪器和 DDPM 正、反向过程。
- `src/ieee33_diffusion/physics.py`：按活动拓扑构造导纳矩阵并计算可微交流潮流残差。
- `src/ieee33_diffusion/train_infer.py`：训练、验证，以及在未见测试拓扑上的反向采样。
- `src/ieee33_diffusion/audit_dataset.py`：检查数据结构、拓扑划分和物理一致性，生成论文数据清单。

## 快速开始

```powershell
# 若网络较慢，可先设置：$env:UV_HTTP_TIMEOUT='300'
uv sync

# 小规模严格可行数据集（默认要求 0.90 <= V <= 1.10 p.u.）
uv run ieee33-build --output data/ieee33_smoke.npz --num-topologies 12 --samples-per-topology 20

# 训练
uv run ieee33-run train --data data/ieee33_smoke.npz --output-dir outputs/smoke --epochs 20

# 物理引导图扩散训练（权重需通过验证集选择）
uv run ieee33-run train --data data/ieee33_smoke.npz --output-dir outputs/physics_smoke `
  --epochs 20 --physics-weight 1e-4 --voltage-weight 1.0 --physics-time-weight alpha_bar

# 展平状态与37位拓扑掩码的 Vector-DDPM 对照组
uv run ieee33-run train --data data/ieee33_smoke.npz --output-dir outputs/vector_smoke `
  --model-type vector --epochs 20

# 生成数据审计清单
uv run ieee33-audit --data data/ieee33_smoke.npz --output docs/dataset_manifest.md

# 在测试集未见拓扑上生成样本
uv run ieee33-run sample --data data/ieee33_smoke.npz --checkpoint outputs/smoke/best.pt --output outputs/smoke/generated.npz --num-samples 32

# 每个未见测试拓扑生成相同数量的样本并统一评价
uv run ieee33-run sample --data data/ieee33_smoke.npz --checkpoint outputs/smoke/best.pt `
  --output outputs/smoke/generated_equal.npz --samples-per-topology 100
uv run ieee33-evaluate --data data/ieee33_smoke.npz `
  --generated outputs/smoke/generated_equal.npz --output outputs/smoke/metrics.json
```

正式主实验默认构建 300 个拓扑、每个拓扑 500 个运行断面，共 150,000 条严格电压可行样本。训练、验证和测试按 `topology_id` 以 70%/10%/20% 划分，同一拓扑不会跨集合泄漏。

```powershell
# 论文主实验：严格可行数据集，参数均为默认值
uv run ieee33-build `
  --output data/ieee33_feasible_T300_S500_seed2026.npz `
  --num-topologies 300 `
  --samples-per-topology 500 `
  --load-low 0.80 `
  --load-high 1.20 `
  --voltage-min 0.90 `
  --voltage-max 1.10 `
  --seed 2026

# 压力数据集：保留潮流收敛但电压越限的断面
uv run ieee33-build `
  --output data/ieee33_stress_T100_S500_seed2026.npz `
  --num-topologies 100 `
  --samples-per-topology 500 `
  --load-low 0.75 `
  --load-high 1.25 `
  --allow-voltage-violations `
  --seed 2026
```

严格模式是默认模式：拓扑必须在基准负荷下潮流收敛且电压合格；每个运行断面也必须收敛并满足指定电压上下限，否则重新采样。压力模式只要求潮流收敛，电压是否合格记录在 `sample_quality[:,3]` 中。

## Linux 服务器构建

```bash
git clone ssh://git@ssh.github.com:443/sakura-0025/ieee-topology-diffusion.git
cd ieee-topology-diffusion

export UV_HTTP_TIMEOUT=300
uv sync --locked
mkdir -p data logs

nohup uv run --locked ieee33-build \
  --output data/ieee33_feasible_T300_S500_seed2026.npz \
  --num-topologies 300 \
  --samples-per-topology 500 \
  --load-low 0.80 \
  --load-high 1.20 \
  --voltage-min 0.90 \
  --voltage-max 1.10 \
  --seed 2026 \
  > logs/build_feasible_T300_S500_seed2026.log 2>&1 &

echo $! > logs/build_feasible_T300_S500_seed2026.pid
tail -f logs/build_feasible_T300_S500_seed2026.log
```

正式数据已构建后，按论文路线图依次运行阶段脚本：

```bash
# 阶段1：正式数据审计和物理公式抽查
bash scripts/server_stage1_audit.sh \
  data/ieee33_feasible_T300_S500_seed2026.npz \
  docs/dataset_manifest_T300_S500_seed2026.md \
  2>&1 | tee logs/stage1_audit.log

# 阶段2：30拓扑开发集上的物理权重预实验
nohup bash scripts/server_stage2_pilot.sh \
  data/ieee33_dev_T30_S100_seed2026.npz \
  > logs/stage2_pilot.log 2>&1 &
```

阶段2默认比较 `0, 1e-5, 1e-4, 1e-3` 四个物理损失权重。开发实验仅用于选择量级，不能作为最终论文结果。

## 数据结构

生成的 `.npz` 主要包含：

- `master_edge_index [2, 37]`：完整候选支路；
- `master_edge_attr [37, 4]`：`r_pu, x_pu, is_tie, normally_closed`；
- `topology_active_edge_ids [T, 32]`：每个拓扑投入的全局支路 ID；
- `topology_edge_mask [T, 37]`：完整候选边掩码；
- `topology_split [T]`：0/1/2 分别表示训练、验证和测试；
- `x0 [S, 33, 4]`：净注入 `P,Q`、电压幅值 `V`、相角 `theta`；
- `node_raw [S, 33, 6]`：`P_load,Q_load,P_gen,Q_gen,V,theta`；
- `sample_topology_id [S]`：每个潮流断面所属的拓扑；
- `sample_load_scale [S, 32]`：每个负荷的随机缩放系数；
- `sample_quality [S, 4]`：潮流收敛、连通、辐射、电压合格标记。
- `sample_attempt_count [T]`：每个拓扑获得目标样本数所需的总尝试次数；
- `sample_rejected_power_flow [T]`：每个拓扑因潮流不收敛被拒绝的数量；
- `sample_rejected_voltage [T]`：每个拓扑因电压越限被拒绝的数量。

扩散过程只对 `x0` 加噪；拓扑、支路参数和静态节点类型始终作为条件输入。当前版本是可复现的拓扑条件 DDPM 基线，尚未加入潮流方程残差损失或采样引导。
