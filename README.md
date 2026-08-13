# IEEE 33 节点变拓扑扩散实验

本项目以 Baran–Wu IEEE 33 节点配电系统为第一阶段算例，使用 `uv` 管理环境。完整候选图包含 33 个节点、37 条候选支路；每个合法辐射拓扑投入 32 条物理支路。

## 文件分工

- `src/ieee33_diffusion/build_dataset.py`：使用支路交换生成辐射拓扑，施加负荷扰动并计算 AC 潮流。
- `src/ieee33_diffusion/model.py`：拓扑条件图去噪器和 DDPM 正、反向过程。
- `src/ieee33_diffusion/train_infer.py`：训练、验证，以及在未见测试拓扑上的反向采样。

## 快速开始

```powershell
# 若网络较慢，可先设置：$env:UV_HTTP_TIMEOUT='300'
uv sync

# 小规模数据集
uv run ieee33-build --output data/ieee33_smoke.npz --num-topologies 12 --samples-per-topology 20

# 训练
uv run ieee33-run train --data data/ieee33_smoke.npz --output-dir outputs/smoke --epochs 20

# 在测试集未见拓扑上生成样本
uv run ieee33-run sample --data data/ieee33_smoke.npz --checkpoint outputs/smoke/best.pt --output outputs/smoke/generated.npz --num-samples 32
```

正式实验可从 100–500 个拓扑、每个拓扑 1000 个运行断面开始。训练、验证和测试按 `topology_id` 划分，同一拓扑不会跨集合泄漏。

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

扩散过程只对 `x0` 加噪；拓扑、支路参数和静态节点类型始终作为条件输入。当前版本是可复现的拓扑条件 DDPM 基线，尚未加入潮流方程残差损失或采样引导。
