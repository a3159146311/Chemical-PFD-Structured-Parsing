###阶段四代码

import json
import uuid
import networkx as nx
from collections import defaultdict


def generate_uuid():
    return uuid.uuid4().hex


def get_dynamic_port(device_name, is_inflow, index, total):
    """
    根据设备的实际空间排序(从上到下)动态分配工业标准端口名
    """
    dev_type = device_name.split('_')[0]

    if dev_type == 'vertical vessel' or dev_type == 'DistillationColumn':
        if not is_inflow:  # 塔的出料口
            if total == 1:
                return "product_out"
            if index == 0:
                return "top_vapor_out"  # 最上面的出料
            if index == total - 1:
                return "bottom_liquid_out"  # 最下面的出料
            return f"side_draw_{index}"  # 中间的出料
        else:  # 塔的进料口
            return f"mixed_feed_{index}"

    elif dev_type == 'boiler' or dev_type == 'Furnace':
        return f"feed_in_{index}" if is_inflow else f"heated_out_{index}"

    # 通用设备的默认端口映射
    prefix = "inlet" if is_inflow else "outlet"
    return f"{prefix}_{index}"


def fourth_stage_final(topology_graph):
    """
    第四阶段：基于 Stage 3 的精确接触点和有向边，生成 API 数据。
    """
    print("🔄 正在将带属性的物理网络折叠为 JSON...")

    project_id = "ab7236a3"
    process_id = "6a7323ae"
    api_payload = {"/api/node/create": [], "/api/edge/sync": []}

    node_uuid_map = {}
    port_registry = {}  # 记录 (设备名, 管道名) 对应的具体端口名

    # ==========================================
    # 1. 注册设备节点并进行空间端口分配
    # ==========================================
    for node in list(topology_graph.nodes):
        if topology_graph.nodes[node].get('type') == 'device':
            # 生成设备节点 JSON
            dev_type = node.split('_')[0]
            node_id = f"{generate_uuid()[:8]}_{dev_type}"
            node_uuid_map[node] = node_id

            api_payload["/api/node/create"].append({
                "project_id": project_id, "process_id": process_id,
                "node_id": node_id, "node_tag": node, "node_type": dev_type
            })

            # 获取该设备的流入管道 (Pipe -> Device)
            in_edges = [(u, data.get('contact_pt', (0, 0))) for u, v, data in topology_graph.in_edges(node, data=True)]
            # 依据 contact_pt 的 Y 坐标 (即 tuple 的第 0 位) 从上到下排序
            in_edges.sort(key=lambda x: x[1][0])
            for i, (pipe, _) in enumerate(in_edges):
                port_registry[(node, pipe)] = get_dynamic_port(node, is_inflow=True, index=i, total=len(in_edges))

            # 获取该设备的流出管道 (Device -> Pipe)
            out_edges = [(v, data.get('contact_pt', (0, 0))) for u, v, data in
                         topology_graph.out_edges(node, data=True)]
            # 同样按 Y 坐标排序
            out_edges.sort(key=lambda x: x[1][0])
            for i, (pipe, _) in enumerate(out_edges):
                port_registry[(node, pipe)] = get_dynamic_port(node, is_inflow=False, index=i, total=len(out_edges))

    # ==========================================
    # 2. 管道折叠与连线生成 (处理 Source/Sink)
    # ==========================================
    edge_counter = 0
    source_count = 0
    sink_count = 0

    for pipe in list(topology_graph.nodes):
        if topology_graph.nodes[pipe].get('type') == 'pipe':
            preds = list(topology_graph.predecessors(pipe))  # 流入该管道的设备 (源头)
            succs = list(topology_graph.successors(pipe))  # 流出该管道的设备 (目标)

            if len(preds) == 0 and len(succs) == 0:
                continue  # 孤立噪点线

            src_node, tgt_node = None, None
            src_port, tgt_port = "unknown", "unknown"

            # --- 解析起点 ---
            if len(preds) == 1:
                src_node = preds[0]
                src_port = port_registry.get((src_node, pipe), "outlet_0")
            elif len(preds) == 0:  # 悬空源头
                v_name = f"Source_{source_count}"
                source_count += 1
                node_uuid_map[v_name] = f"{generate_uuid()[:8]}_{v_name}"
                api_payload["/api/node/create"].append({
                    "project_id": project_id, "process_id": process_id,
                    "node_id": node_uuid_map[v_name], "node_tag": v_name, "node_type": "Source"
                })
                src_node = v_name
                src_port = "outlet_0"

            # --- 解析终点 ---
            if len(succs) == 1:
                tgt_node = succs[0]
                tgt_port = port_registry.get((tgt_node, pipe), "inlet_0")
            elif len(succs) == 0:  # 悬空汇点
                v_name = f"Sink_{sink_count}"
                sink_count += 1
                node_uuid_map[v_name] = f"{generate_uuid()[:8]}_{v_name}"
                api_payload["/api/node/create"].append({
                    "project_id": project_id, "process_id": process_id,
                    "node_id": node_uuid_map[v_name], "node_tag": v_name, "node_type": "Sink"
                })
                tgt_node = v_name
                tgt_port = "inlet_0"

            # --- 生成最终连线 ---
            if src_node and tgt_node:
                edge_info = {
                    "id": f"{generate_uuid()}_e-{edge_counter}",
                    "name": pipe,
                    "source": node_uuid_map[src_node],
                    "target": node_uuid_map[tgt_node],
                    "sourcePort": src_port,
                    "targetPort": tgt_port
                }
                api_payload["/api/edge/sync"].append({
                    "project_id": project_id, "process_id": process_id, "edge_info": edge_info
                })
                edge_counter += 1

    print("✅ JSON 结构化数据折叠完毕！")
    return api_payload


# ==========================================
# 主代码中调用 Fourth Stage
# ==========================================
try:
    api_payload = fourth_stage_final(graph)

    # 打印 JSON
    json_output = json.dumps(api_payload, ensure_ascii=False, indent=4)
    print("\n" + "═" * 50)
    print("🚀 生成的数据 (JSON):")
    print("═" * 50)
    print(json_output)
    print("═" * 50 + "\n")

except Exception as e:
    print(f"❌ 第四阶段执行出错: {e}")