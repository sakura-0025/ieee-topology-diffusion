"""构建 IEEE 33 节点多拓扑 AC 潮流数据集。

数据组织采用“固定候选图 + 拓扑活动边”的方式：完整候选图始终包含
37 条支路，每个辐射拓扑通过掩码选择其中 32 条活动支路。这样既能保持
不同拓扑间统一的全局边编号，又能让图模型只在当前实际连通的支路上传播消息。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from tqdm import tqdm


@dataclass(frozen=True)
class BuildConfig:
    """数据集构建参数。

    ``load_low`` 和 ``load_high`` 是相对于 case33bw 基准负荷的乘数范围。
    每个负荷的 P、Q 使用同一个乘数，以保持其功率因数不变。
    """

    num_topologies: int = 300
    samples_per_topology: int = 500
    load_low: float = 0.80
    load_high: float = 1.20
    voltage_min: float = 0.90
    voltage_max: float = 1.10
    require_voltage_feasible: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    seed: int = 2026
    max_topology_attempts: int = 100_000
    max_sample_attempt_factor: int = 100


def _validate_config(config: BuildConfig) -> None:
    """在耗时计算开始前检查构建参数。"""

    if config.num_topologies < 1 or config.samples_per_topology < 1:
        raise ValueError("num_topologies and samples_per_topology must be positive.")
    if not 0.0 < config.load_low <= config.load_high:
        raise ValueError("Require 0 < load_low <= load_high.")
    if not 0.0 < config.voltage_min < config.voltage_max:
        raise ValueError("Require 0 < voltage_min < voltage_max.")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must lie in (0, 1).")
    if not 0.0 <= config.val_ratio < 1.0:
        raise ValueError("val_ratio must lie in [0, 1).")
    if config.train_ratio + config.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")
    if config.max_sample_attempt_factor < 1:
        raise ValueError("max_sample_attempt_factor must be positive.")


def _run_power_flow(net: pp.pandapowerNet) -> bool:
    """运行一次 Newton-Raphson AC 潮流并返回是否收敛。"""

    try:
        pp.runpp(
            net,
            algorithm="nr",
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=30,
            numba=False,
        )
    except pp.LoadflowNotConverged:
        return False
    return bool(net.converged)


def _voltage_is_feasible(net: pp.pandapowerNet, config: BuildConfig) -> bool:
    """检查所有节点电压是否位于用户指定的统一标幺范围内。"""

    voltage = net.res_bus.vm_pu.to_numpy(float)
    tolerance = 1.0e-8
    return bool(
        np.isfinite(voltage).all()
        and np.all(voltage >= config.voltage_min - tolerance)
        and np.all(voltage <= config.voltage_max + tolerance)
    )


def _master_graph(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取全局候选图、支路特征和原始运行状态。

    返回：
        edge_index: ``[2, 37]``，每列为一条候选支路的首末节点编号。
        edge_attr: ``[37, 4]``，依次为 ``r_pu, x_pu, is_tie, normally_closed``。
        original_mask: ``[37]``，基准拓扑中支路是否投入运行。
    """
    line = net.line.sort_index()
    edge_index = np.vstack(
        [line.from_bus.to_numpy(np.int64), line.to_bus.to_numpy(np.int64)]
    )

    # 将线路欧姆值换算为标幺值：Z_pu = Z_ohm / (V_base^2 / S_base)。
    base_mva = float(net.sn_mva)
    from_kv = net.bus.vn_kv.loc[line.from_bus].to_numpy(float)
    z_base = np.square(from_kv) / base_mva
    r_pu = line.r_ohm_per_km.to_numpy(float) * line.length_km.to_numpy(float) / z_base
    x_pu = line.x_ohm_per_km.to_numpy(float) * line.length_km.to_numpy(float) / z_base

    # case33bw 中原始断开的 5 条线路就是联络线候选，因此可由状态确定 is_tie。
    original_mask = line.in_service.to_numpy(bool)
    is_tie = (~original_mask).astype(np.float32)
    edge_attr = np.column_stack(
        [r_pu, x_pu, is_tie, original_mask.astype(np.float32)]
    ).astype(np.float32)
    return edge_index, edge_attr, original_mask


def _graph_from_mask(edge_index: np.ndarray, mask: np.ndarray, num_nodes: int) -> nx.Graph:
    """将一个 ``[37]`` 支路掩码转换为 NetworkX 无向图。"""

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    active = np.flatnonzero(mask)
    graph.add_edges_from(
        (int(edge_index[0, edge_id]), int(edge_index[1, edge_id]))
        for edge_id in active
    )
    return graph


def _is_tree(edge_index: np.ndarray, mask: np.ndarray, num_nodes: int) -> bool:
    """检查活动边是否构成覆盖全部节点的辐射树。"""

    # N 个节点的树必须恰好有 N-1 条边；先检查边数可快速排除非法拓扑。
    if int(mask.sum()) != num_nodes - 1:
        return False
    return nx.is_tree(_graph_from_mask(edge_index, mask, num_nodes))


def _branch_exchange(
    mask: np.ndarray,
    edge_index: np.ndarray,
    num_nodes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    """执行一次“合一开一”的支路交换，生成新的辐射拓扑。

    先随机闭合一条当前断开的支路。由于原图是树，这一步必然产生唯一环；
    随后在环上随机断开一条原活动支路，使新拓扑重新成为 32 边的生成树。

    返回新掩码、被断开的全局边 ID、被闭合的全局边 ID。
    """
    graph = _graph_from_mask(edge_index, mask, num_nodes)
    inactive_ids = np.flatnonzero(~mask)
    close_id = int(rng.choice(inactive_ids))
    u, v = map(int, edge_index[:, close_id])
    # 树中 u 到 v 的路径唯一。该路径加上新闭合支路后就是唯一环。
    path_nodes = nx.shortest_path(graph, u, v)

    # 使用无序节点对查找全局边 ID，避免支路方向影响匹配。
    lookup: dict[frozenset[int], int] = {}
    for edge_id in np.flatnonzero(mask):
        a, b = map(int, edge_index[:, edge_id])
        lookup[frozenset((a, b))] = int(edge_id)
    cycle_active_ids = [
        lookup[frozenset((a, b))]
        for a, b in zip(path_nodes[:-1], path_nodes[1:], strict=True)
    ]
    open_id = int(rng.choice(cycle_active_ids))

    new_mask = mask.copy()
    new_mask[close_id] = True
    new_mask[open_id] = False
    if not _is_tree(edge_index, new_mask, num_nodes):
        raise RuntimeError("Branch exchange did not produce a spanning tree.")
    return new_mask, open_id, close_id


def generate_topologies(
    net: pp.pandapowerNet,
    edge_index: np.ndarray,
    base_mask: np.ndarray,
    config: BuildConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """通过随机支路交换游走生成互不重复且潮流可收敛的拓扑。

    返回：
        masks: ``[T, 37]``，全部拓扑的候选边掩码。
        parents: ``[T]``，每个拓扑由哪个已有拓扑变换而来。
        actions: ``[T, 2]``，对应 ``[opened_edge_id, closed_edge_id]``。
        stats: 拓扑候选的尝试次数与各类拒绝次数。
    """
    rng = np.random.default_rng(config.seed)
    num_nodes = len(net.bus)
    masks = [base_mask.copy()]
    parents = [-1]
    actions = [(-1, -1)]  # opened edge, closed edge
    # 将布尔掩码压缩为字节串，便于 O(1) 判断拓扑是否已经出现。
    seen = {np.packbits(base_mask).tobytes()}
    current_id = 0

    # 原始 IEEE33 拓扑也必须满足当前模式要求，否则不能作为训练基准。
    net.line.loc[:, "in_service"] = base_mask
    if not _run_power_flow(net):
        raise RuntimeError("The original case33bw topology did not converge.")
    if config.require_voltage_feasible and not _voltage_is_feasible(net, config):
        raise RuntimeError(
            "The original case33bw topology violates the configured voltage bounds."
        )

    attempts = 0
    rejected_duplicate = 0
    rejected_power_flow = 0
    rejected_voltage = 0
    while len(masks) < config.num_topologies and attempts < config.max_topology_attempts:
        attempts += 1
        # 80% 从最新拓扑继续游走，20% 回到历史拓扑分叉，增加拓扑多样性。
        parent_id = current_id if rng.random() < 0.8 else int(rng.integers(len(masks)))
        candidate, opened, closed = _branch_exchange(
            masks[parent_id], edge_index, num_nodes, rng
        )
        key = np.packbits(candidate).tobytes()
        if key in seen:
            rejected_duplicate += 1
            continue

        # 树结构合法不代表潮流一定有解，因此还需用基准负荷做一次收敛筛选。
        net.line.loc[:, "in_service"] = candidate
        if not _run_power_flow(net):
            rejected_power_flow += 1
            continue
        if config.require_voltage_feasible and not _voltage_is_feasible(net, config):
            rejected_voltage += 1
            continue

        seen.add(key)
        masks.append(candidate)
        parents.append(parent_id)
        actions.append((opened, closed))
        current_id = len(masks) - 1

    if len(masks) < config.num_topologies:
        raise RuntimeError(
            f"Only generated {len(masks)} unique convergent topologies after "
            f"{attempts} attempts; requested {config.num_topologies}."
        )
    stats = {
        "attempts": attempts,
        "accepted": len(masks),
        "rejected_duplicate": rejected_duplicate,
        "rejected_power_flow": rejected_power_flow,
        "rejected_voltage": rejected_voltage,
    }
    return (
        np.stack(masks),
        np.asarray(parents, dtype=np.int32),
        np.asarray(actions, dtype=np.int32),
        stats,
    )


def _topology_splits(num_topologies: int, config: BuildConfig) -> np.ndarray:
    """按拓扑而非按样本划分训练/验证/测试集，防止拓扑信息泄漏。"""

    rng = np.random.default_rng(config.seed + 1)
    ids = np.arange(num_topologies)
    movable = ids[1:].copy()
    rng.shuffle(movable)
    # 拓扑 0 是 IEEE33 原始拓扑，固定保留在训练集中作为基准。
    ordered = np.concatenate(([0], movable))

    n_train = max(1, int(round(num_topologies * config.train_ratio)))
    n_val = int(round(num_topologies * config.val_ratio))
    if num_topologies >= 3:
        n_train = min(n_train, num_topologies - 2)
        n_val = max(1, min(n_val, num_topologies - n_train - 1))
    else:
        n_val = max(0, num_topologies - n_train)

    split = np.full(num_topologies, 2, dtype=np.uint8)
    split[ordered[:n_train]] = 0
    split[ordered[n_train : n_train + n_val]] = 1
    return split


def _node_static(net: pp.pandapowerNet) -> np.ndarray:
    """生成 ``[33, 4]`` 静态节点特征。

    四个通道依次是归一化基准电压、平衡节点标记、负荷节点标记和发电节点标记。
    这些量在扩散过程中不加噪，只作为条件输入。
    """

    num_nodes = len(net.bus)
    slack = np.zeros(num_nodes, dtype=np.float32)
    slack[net.ext_grid.bus.to_numpy(np.int64)] = 1.0
    has_load = np.zeros(num_nodes, dtype=np.float32)
    if len(net.load):
        has_load[np.unique(net.load.bus.to_numpy(np.int64))] = 1.0
    has_gen = np.zeros(num_nodes, dtype=np.float32)
    for table_name in ("gen", "sgen"):
        table = getattr(net, table_name)
        if len(table):
            has_gen[np.unique(table.bus.to_numpy(np.int64))] = 1.0
    base_kv = net.bus.vn_kv.to_numpy(np.float32)
    base_kv = base_kv / max(float(base_kv.max()), 1.0)
    return np.column_stack([base_kv, slack, has_load, has_gen]).astype(np.float32)


def _bus_sums(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    """把元件级潮流结果汇总为节点级特征。

    ``x0 [33, 4]`` 是扩散模型的生成对象，通道为净注入 ``P,Q,V,theta``；
    ``raw [33, 6]`` 保留原始 ``Pd,Qd,Pg,Qg,V,theta``，用于物理校验。
    功率单位为 MW/Mvar，电压为 p.u.，相角由度转换为弧度。
    """
    n = len(net.bus)
    p_load = np.zeros(n)
    q_load = np.zeros(n)
    p_gen = np.zeros(n)
    q_gen = np.zeros(n)

    # 一个节点可能挂接多个元件，np.add.at 可按母线编号安全累加。
    if len(net.load):
        np.add.at(p_load, net.load.bus.to_numpy(np.int64), net.load.p_mw.to_numpy(float))
        np.add.at(q_load, net.load.bus.to_numpy(np.int64), net.load.q_mvar.to_numpy(float))
    if len(net.sgen):
        np.add.at(p_gen, net.sgen.bus.to_numpy(np.int64), net.sgen.p_mw.to_numpy(float))
        np.add.at(q_gen, net.sgen.bus.to_numpy(np.int64), net.sgen.q_mvar.to_numpy(float))
    if len(net.gen):
        np.add.at(p_gen, net.gen.bus.to_numpy(np.int64), net.res_gen.p_mw.to_numpy(float))
        np.add.at(q_gen, net.gen.bus.to_numpy(np.int64), net.res_gen.q_mvar.to_numpy(float))
    if len(net.ext_grid):
        np.add.at(
            p_gen,
            net.ext_grid.bus.to_numpy(np.int64),
            net.res_ext_grid.p_mw.to_numpy(float),
        )
        np.add.at(
            q_gen,
            net.ext_grid.bus.to_numpy(np.int64),
            net.res_ext_grid.q_mvar.to_numpy(float),
        )

    voltage = net.res_bus.vm_pu.to_numpy(float)
    theta = np.deg2rad(net.res_bus.va_degree.to_numpy(float))
    # 采用“发电为正、负荷为负”的净注入符号约定。
    p_inj = p_gen - p_load
    q_inj = q_gen - q_load
    x0 = np.column_stack([p_inj, q_inj, voltage, theta]).astype(np.float32)
    raw = np.column_stack([p_load, q_load, p_gen, q_gen, voltage, theta]).astype(
        np.float32
    )
    return x0, raw


def build_dataset(output: Path, config: BuildConfig) -> None:
    """构建并保存多拓扑、多运行断面的压缩 NPZ 数据集。"""

    _validate_config(config)
    # 各阶段使用错开的随机种子，使拓扑、切分与负荷扰动彼此可复现且不共用序列。
    rng = np.random.default_rng(config.seed + 2)
    net = pn.case33bw()
    net.bus.sort_index(inplace=True)
    net.line.sort_index(inplace=True)
    if not np.array_equal(net.bus.index.to_numpy(), np.arange(len(net.bus))):
        raise ValueError("case33bw bus indices must be contiguous from zero.")

    master_edge_index, master_edge_attr, base_mask = _master_graph(net)
    (
        topology_masks,
        topology_parent,
        topology_actions,
        topology_generation_stats,
    ) = generate_topologies(net, master_edge_index, base_mask, config)
    # 对模型而言每个辐射拓扑只需 32 个活动边 ID；同时仍保存完整 37 位掩码供比较。
    topology_active_edge_ids = np.stack(
        [np.flatnonzero(mask) for mask in topology_masks]
    ).astype(np.int32)
    topology_split = _topology_splits(len(topology_masks), config)

    # 保存基准负荷，后续每次扰动都相对于基准值，避免倍率逐轮累乘。
    base_load_p = net.load.p_mw.to_numpy(float).copy()
    base_load_q = net.load.q_mvar.to_numpy(float).copy()
    samples_x0: list[np.ndarray] = []
    samples_raw: list[np.ndarray] = []
    sample_topology: list[int] = []
    sample_scale: list[np.ndarray] = []
    sample_quality: list[tuple[int, int, int, int]] = []
    sample_attempt_count = np.zeros(config.num_topologies, dtype=np.int32)
    sample_rejected_power_flow = np.zeros(config.num_topologies, dtype=np.int32)
    sample_rejected_voltage = np.zeros(config.num_topologies, dtype=np.int32)

    total = config.num_topologies * config.samples_per_topology
    with tqdm(total=total, desc="AC power-flow samples") as progress:
        for topology_id, mask in enumerate(topology_masks):
            net.line.loc[:, "in_service"] = mask
            accepted = 0
            attempts = 0
            while accepted < config.samples_per_topology:
                attempts += 1
                sample_attempt_count[topology_id] += 1
                if attempts > config.samples_per_topology * config.max_sample_attempt_factor:
                    raise RuntimeError(
                        f"Too many rejected samples for topology {topology_id}; "
                        "consider narrowing the load range or using "
                        "--allow-voltage-violations."
                    )
                # 每个负荷独立采样倍率；同一负荷的 P、Q 共用倍率，保持功率因数。
                scales = rng.uniform(config.load_low, config.load_high, len(net.load))
                net.load.loc[:, "p_mw"] = base_load_p * scales
                net.load.loc[:, "q_mvar"] = base_load_q * scales
                if not _run_power_flow(net):
                    sample_rejected_power_flow[topology_id] += 1
                    continue

                x0, raw = _bus_sums(net)
                voltage_ok = _voltage_is_feasible(net, config)
                # 严格模式拒绝电压越限断面；压力模式保留断面并将质量标记置零。
                if config.require_voltage_feasible and not voltage_ok:
                    sample_rejected_voltage[topology_id] += 1
                    continue
                samples_x0.append(x0)
                samples_raw.append(raw)
                sample_topology.append(topology_id)
                sample_scale.append(scales.astype(np.float32))
                sample_quality.append((1, 1, 1, int(voltage_ok)))
                accepted += 1
                progress.update(1)

    # 样本切分完全继承所属拓扑的切分，保证同一拓扑不会跨数据集。
    sample_topology_array = np.asarray(sample_topology, dtype=np.int32)
    sample_split = topology_split[sample_topology_array]
    metadata = {
        "case": "Baran-Wu case33bw",
        "base_mva": float(net.sn_mva),
        "num_nodes": int(len(net.bus)),
        "num_candidate_edges": int(len(net.line)),
        "num_active_edges_per_topology": int(topology_active_edge_ids.shape[1]),
        "num_topologies": int(config.num_topologies),
        "samples_per_topology": int(config.samples_per_topology),
        "dataset_mode": (
            "strict_voltage_feasible"
            if config.require_voltage_feasible
            else "converged_stress"
        ),
        "load_scale_bounds": [config.load_low, config.load_high],
        "voltage_bounds_pu": [config.voltage_min, config.voltage_max],
        "require_voltage_feasible": config.require_voltage_feasible,
        "topology_generation_stats": topology_generation_stats,
        "x0_channels": ["p_injection_mw", "q_injection_mvar", "voltage_pu", "theta_rad"],
        "raw_channels": ["p_load_mw", "q_load_mvar", "p_gen_mw", "q_gen_mvar", "voltage_pu", "theta_rad"],
        "edge_channels": ["r_pu", "x_pu", "is_tie", "normally_closed"],
        "node_static_channels": ["base_kv_scaled", "is_slack", "has_load", "has_generator"],
        "quality_channels": ["pf_converged", "connected", "radial", "voltage_feasible"],
        "seed": config.seed,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    # 所有数组均使用数值类型并关闭 pickle，方便安全、快速地跨脚本读取。
    np.savez_compressed(
        output,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        base_mva=np.asarray(float(net.sn_mva), dtype=np.float32),
        node_static=_node_static(net),
        master_edge_index=master_edge_index.astype(np.int32),
        master_edge_attr=master_edge_attr,
        topology_edge_mask=topology_masks.astype(np.uint8),
        topology_active_edge_ids=topology_active_edge_ids,
        topology_parent=topology_parent,
        topology_actions=topology_actions,
        topology_split=topology_split,
        x0=np.stack(samples_x0),
        node_raw=np.stack(samples_raw),
        sample_topology_id=sample_topology_array,
        sample_split=sample_split,
        sample_load_scale=np.stack(sample_scale),
        sample_quality=np.asarray(sample_quality, dtype=np.uint8),
        sample_attempt_count=sample_attempt_count,
        sample_rejected_power_flow=sample_rejected_power_flow,
        sample_rejected_voltage=sample_rejected_voltage,
    )
    quality = np.asarray(sample_quality, dtype=np.uint8)
    total_attempts = int(sample_attempt_count.sum())
    print(f"Saved {len(samples_x0):,} samples to {output}")
    print(
        "Topologies by split:",
        {name: int(np.sum(topology_split == code)) for code, name in enumerate(("train", "val", "test"))},
    )
    print(
        "Sampling summary:",
        {
            "attempts": total_attempts,
            "accepted": len(samples_x0),
            "acceptance_rate": round(len(samples_x0) / total_attempts, 6),
            "rejected_power_flow": int(sample_rejected_power_flow.sum()),
            "rejected_voltage": int(sample_rejected_voltage.sum()),
            "saved_voltage_feasible_rate": round(float(quality[:, 3].mean()), 6),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ieee33_feasible_T300_S500_seed2026.npz"),
    )
    parser.add_argument("--num-topologies", type=int, default=300)
    parser.add_argument("--samples-per-topology", type=int, default=500)
    parser.add_argument("--load-low", type=float, default=0.80)
    parser.add_argument("--load-high", type=float, default=1.20)
    parser.add_argument("--voltage-min", type=float, default=0.90)
    parser.add_argument("--voltage-max", type=float, default=1.10)
    parser.add_argument(
        "--allow-voltage-violations",
        action="store_true",
        help="Keep converged voltage-violating samples for a stress dataset.",
    )
    parser.add_argument("--max-sample-attempt-factor", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        num_topologies=args.num_topologies,
        samples_per_topology=args.samples_per_topology,
        load_low=args.load_low,
        load_high=args.load_high,
        voltage_min=args.voltage_min,
        voltage_max=args.voltage_max,
        require_voltage_feasible=not args.allow_voltage_violations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_sample_attempt_factor=args.max_sample_attempt_factor,
    )
    build_dataset(args.output, config)


if __name__ == "__main__":
    main()
