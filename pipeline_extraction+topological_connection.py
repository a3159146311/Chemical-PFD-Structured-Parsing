import cv2
import numpy as np
import networkx as nx
import json
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


def draw_text_with_bg(img, text, position, font_scale, text_color, bg_color=(0, 0, 0), alpha=0.6, thickness=2):
    """
    辅助函数：绘制带有半透明背景框的文字
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 获取文字的宽高
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position

    # 设置背景框的留白边缘 (padding)
    pad = 5
    top_left = (x - pad, y - text_h - pad)
    bottom_right = (x + text_w + pad, y + pad)

    # 绘制半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 绘制文字本体
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)


def third_stage(image_rgb, masks, yolo_model, boxes=None):
    """
    第三阶段：从图像中提取管道骨架，利用端点箭头位置逻辑构建有向拓扑连接。
    """
    H, W = image_rgb.shape[:2]

    vis = image_rgb.copy()
    vis_text = image_rgb.copy()

    label_draw_info = {}

    # 【新增】初始化评估专用的数据字典与计数器
    pred_eval_data = {
        "nodes": {},
        "edges": []
    }
    eval_node_counter = 0

    # ==========================================
    # 0. 掩膜预处理
    # ==========================================
    device_masks_np = []
    union_device_mask = np.zeros((H, W), dtype=np.uint8)

    if masks is not None and len(masks) > 0:
        for i, m in enumerate(masks):
            m_np = (m[0].cpu().numpy() * 255).astype(np.uint8)
            device_masks_np.append(m_np)
            union_device_mask = cv2.bitwise_or(union_device_mask, m_np)

            cls_name = yolo_model.names[int(boxes.cls[i].item())] if boxes is not None else "Device"
            device_name = f"{cls_name}_{i}"
            M = cv2.moments(m_np)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label_draw_info[device_name] = {'pos': (cx - 30, cy), 'type': 'device'}

    # ==========================================
    # 1. 箭头位置提取
    # ==========================================
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    arrows_centroids = []
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # ==========================================
    # 2. 管道骨架化
    # ==========================================
    print(" 正在提取管道骨架...")
    binary[union_device_mask == 255] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    skeleton_bool = skeletonize(binary_cleaned > 0)
    skeleton_uint8 = (skeleton_bool * 255).astype(np.uint8)

    # ==========================================
    # 3. 管道分析与端点信息收集
    # ==========================================
    print(" 正在分析端点与应用位置流向逻辑...")
    labeled_skeleton = label(skeleton_bool, connectivity=2)
    props = regionprops(labeled_skeleton)

    topology_graph = nx.DiGraph()
    vis[skeleton_bool] = [0, 255, 0]

    for r, c in arrows_centroids:
        cv2.circle(vis, (c, r), 6, (255, 0, 255), -1)

    tolerance = 10
    pipe_id = 0
    arrow_search_radius = 35

    for prop in props:
        coords = prop.coords
        if len(coords) < 10:
            continue

        pipe_name = f"Pipe_{pipe_id}"
        pipe_id += 1
        topology_graph.add_node(pipe_name, type='pipe')

        mid_idx = len(coords) // 2
        mid_r, mid_c = coords[mid_idx]
        label_draw_info[pipe_name] = {'pos': (mid_c, mid_r), 'type': 'pipe'}

        coord_set = set(tuple(c) for c in coords)
        endpoints = []
        for r, c in coords:
            neighbors = sum(1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                            if not (dr == 0 and dc == 0) and (r + dr, c + dc) in coord_set)
            if neighbors == 1:
                endpoints.append((r, c))

        if not endpoints and len(coords) > 0:
            endpoints = [(coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])]

        endpoint_infos = []

        for r, c in endpoints:
            cv2.circle(vis, (c, r), 5, (255, 0, 0), -1)

            has_arrow = False
            for ar, ac in arrows_centroids:
                if np.linalg.norm(np.array([r, c]) - np.array([ar, ac])) < arrow_search_radius:
                    has_arrow = True
                    break

            hit_device = None
            for i, d_mask in enumerate(device_masks_np):
                r_min, r_max = max(0, r - tolerance), min(H, r + tolerance + 1)
                c_min, c_max = max(0, c - tolerance), min(W, c + tolerance + 1)

                if np.any(d_mask[r_min:r_max, c_min:c_max] == 255):
                    cls_name = yolo_model.names[int(boxes.cls[i].item())] if boxes is not None else "Device"
                    hit_device = f"{cls_name}_{i}"
                    break

            endpoint_infos.append({
                'pt': (r, c),
                'device': hit_device,
                'has_arrow': has_arrow
            })

        total_arrows_on_pipe = sum(1 for ep in endpoint_infos if ep['has_arrow'])

        connected_dev_eps = []
        seen_devs = set()
        for ep in endpoint_infos:
            if ep['device'] and ep['device'] not in seen_devs:
                connected_dev_eps.append(ep)
                seen_devs.add(ep['device'])

        if len(connected_dev_eps) >= 2:
            ep1, ep2 = connected_dev_eps[0], connected_dev_eps[1]
            for ep in [ep1, ep2]:
                if not topology_graph.has_node(ep['device']):
                    topology_graph.add_node(ep['device'], type='device')

            if ep1['has_arrow']:
                src, tgt = ep2, ep1
            elif ep2['has_arrow']:
                src, tgt = ep1, ep2
            else:
                if ep1['pt'][1] <= ep2['pt'][1]:
                    src, tgt = ep1, ep2
                else:
                    src, tgt = ep2, ep1

            topology_graph.add_edge(src['device'], pipe_name, contact_pt=src['pt'])
            topology_graph.add_edge(pipe_name, tgt['device'], contact_pt=tgt['pt'])

        else:
            if total_arrows_on_pipe > 0:
                for ep in endpoint_infos:
                    if ep['device']:
                        if not topology_graph.has_node(ep['device']):
                            topology_graph.add_node(ep['device'], type='device')

                        if ep['has_arrow']:
                            topology_graph.add_edge(pipe_name, ep['device'], contact_pt=ep['pt'])
                        else:
                            topology_graph.add_edge(ep['device'], pipe_name, contact_pt=ep['pt'])
            else:
                endpoint_infos.sort(key=lambda x: x['pt'][1])

                for idx, ep in enumerate(endpoint_infos):
                    if ep['device']:
                        if not topology_graph.has_node(ep['device']):
                            topology_graph.add_node(ep['device'], type='device')

                        if idx == 0:
                            topology_graph.add_edge(ep['device'], pipe_name, contact_pt=ep['pt'])
                        else:
                            topology_graph.add_edge(pipe_name, ep['device'], contact_pt=ep['pt'])

        # ==========================================================
        # 评估专用数据拦截：保存物理层端点坐标与属性
        # ==========================================================
        is_valid_topology_pipe = any(ep['device'] is not None for ep in endpoint_infos)

        # 只有存在有效连接的管线（排除了游离的文字和噪声），才计入评估数据
        if len(endpoint_infos) >= 2 and is_valid_topology_pipe:

            # 提取管线的两端
            ep1 = endpoint_infos[0]
            ep2 = endpoint_infos[-1]

            # 严格复用系统底层的流向判定规则
            if ep1['has_arrow']:
                src_ep, tgt_ep = ep2, ep1
            elif ep2['has_arrow']:
                src_ep, tgt_ep = ep1, ep2
            else:
                if ep1['pt'][1] <= ep2['pt'][1]:
                    src_ep, tgt_ep = ep1, ep2
                else:
                    src_ep, tgt_ep = ep2, ep1

            attr_src = "Outflow"
            attr_tgt = "Inflow"

            # 坐标转换：pt 存储的是 (y, x)，需转换为 JSON 需要的 [x, y]
            x_src, y_src = int(src_ep['pt'][1]), int(src_ep['pt'][0])
            x_tgt, y_tgt = int(tgt_ep['pt'][1]), int(tgt_ep['pt'][0])

            node_id_src = f"eval_node_{eval_node_counter}"
            node_id_tgt = f"eval_node_{eval_node_counter + 1}"
            eval_node_counter += 2

            pred_eval_data["nodes"][node_id_src] = {"coord": [x_src, y_src], "attr": attr_src}
            pred_eval_data["nodes"][node_id_tgt] = {"coord": [x_tgt, y_tgt], "attr": attr_tgt}
            pred_eval_data["edges"].append([node_id_src, node_id_tgt])
        # ==========================================================

    print(" 有向拓扑结构构建完成 (已采用设备-管道-设备二分图结构)！")

    # ==========================================
    # 渲染实际构成了连接边 (Edge) 的节点标签
    # ==========================================
    connected_nodes = set()
    for u, v, data in topology_graph.edges(data=True):
        connected_nodes.add(u)
        connected_nodes.add(v)

    for node_name in connected_nodes:
        if node_name in label_draw_info:
            info = label_draw_info[node_name]
            if info['type'] == 'device':
                draw_text_with_bg(vis_text, node_name, info['pos'],
                                  font_scale=0.6, text_color=(255, 50, 50))
            elif info['type'] == 'pipe':
                draw_text_with_bg(vis_text, node_name, info['pos'],
                                  font_scale=0.8, text_color=(50, 255, 50))

    # 【新增】阶段三结束前，将预测数据序列化导出
    with open('prediction_for_eval_image.json', 'w', encoding='utf-8') as f:
        json.dump(pred_eval_data, f, indent=4)
    print(" ✅ 评估专用预测数据 (prediction_for_eval_image.json) 已成功导出！")

    return skeleton_uint8, topology_graph, vis, vis_text


# ==========================================
# 主代码中调用 Third Stage
# ==========================================
try:
    skeleton_img, graph, vis_image, vis_text_image = third_stage(image_rgb, masks, yolo_model, boxes=boxes)
    print("\n" + "=" * 50)
    print(" 提取到的有向连接关系 (拓扑边):")
    if len(graph.edges) == 0:
        print("未检测到任何管道与设备的连接。")
    else:
        #  清理打印逻辑，直接打印拆分后的直连边
        for source, target, edge_data in graph.edges(data=True):
            pt = edge_data.get('contact_pt', '未知坐标')
            print(f" - [起点] {source} ──(触点:{pt})──> [终点] {target}")
    print("=" * 50 + "\n")

    plt.figure(figsize=(24, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(skeleton_img, cmap='gray')
    plt.title("1. Skeletonized Pipes", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(vis_image)
    plt.title("2. Directed Topology Base", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(vis_text_image)
    plt.title("3. Entity IDs Reference (Connected Only)", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except Exception as e:
    import traceback

    print(f"❌ 第三阶段执行出错: {e}")
    traceback.print_exc()