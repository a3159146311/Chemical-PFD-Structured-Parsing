import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# 这里粘贴上一轮我给你的 evaluate_pfd_topology 函数
def evaluate_pfd_topology(gt_data, pred_data, dist_threshold=15):
    """
        参数:
            gt_data: 人工标注的字典结构
            pred_data: YOLOv8m-SAM 级联管道输出的字典结构
            dist_threshold: 坐标允许的欧氏距离偏差阈值
        返回:
            endpoint_acc, topo_precision, attr_acc
        """
    # 提取 GT 数据
    gt_node_ids = list(gt_data["nodes"].keys())
    gt_pts = np.array([gt_data["nodes"][nid]["coord"] for nid in gt_node_ids])
    gt_attrs = [gt_data["nodes"][nid]["attr"] for nid in gt_node_ids]
    gt_edges = set((str(src), str(tgt)) for src, tgt in gt_data["edges"])

    # 提取 Pred 数据
    pred_node_ids = list(pred_data["nodes"].keys())
    if len(pred_node_ids) == 0 or len(gt_node_ids) == 0:
        return 0.0, 0.0, 0.0

    pred_pts = np.array([pred_data["nodes"][nid]["coord"] for nid in pred_node_ids])
    pred_attrs = [pred_data["nodes"][nid]["attr"] for nid in pred_node_ids]
    pred_edges = pred_data.get("edges", [])

    # 1. 匈牙利算法进行端点二分图匹配
    cost_matrix = cdist(pred_pts, gt_pts, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pred2gt = {}
    tp_endpoints = 0
    correct_attrs = 0

    for p_idx, g_idx in zip(row_ind, col_ind):
        if cost_matrix[p_idx, g_idx] <= dist_threshold:
            # 记录有效的节点 ID 映射
            matched_pred2gt[pred_node_ids[p_idx]] = gt_node_ids[g_idx]
            tp_endpoints += 1

            # 同时计算属性正确率
            if pred_attrs[p_idx] == gt_attrs[g_idx]:
                correct_attrs += 1

    # 指标 1: 端点定位精度 (匹配上的端点数 / 总预测端点数)
    endpoint_acc = tp_endpoints / len(pred_node_ids) if len(pred_node_ids) > 0 else 0.0

    # 指标 3: Inflow/Outflow 属性正确占比 (仅在定位正确的点中计算)
    attr_acc = correct_attrs / tp_endpoints if tp_endpoints > 0 else 0.0

    # 2. 拓扑连通正确率计算
    mapped_pred_edges = set()
    for src, tgt in pred_edges:
        if src in matched_pred2gt and tgt in matched_pred2gt:
            mapped_pred_edges.add((matched_pred2gt[src], matched_pred2gt[tgt]))

    correct_edges = mapped_pred_edges.intersection(gt_edges)
    topo_precision = len(correct_edges) / len(pred_edges) if len(pred_edges) > 0 else 0.0

    # 在拿到 gt_node_ids 和 pred_node_ids 后直接打印：
    print("\n======= 核心诊断日志 =======")
    print(f"【数量比对】 GT 人工标注端点数: {len(gt_node_ids)} | 算法预测端点数: {len(pred_node_ids)}")

    # 打印前 3 个点的坐标进行抽样对比，检查是否发生坐标系偏移
    print(f"【坐标抽样】 GT 前3点: {gt_pts[:3].tolist() if len(gt_pts) > 0 else 'None'}")
    print(f"【坐标抽样】 Pred 前3点: {pred_pts[:3].tolist() if len(pred_pts) > 0 else 'None'}")

    # ... 在运行完匈牙利算法后打印：
    print(f"【距离分析】 匈牙利算法最小匹配平均距离: {cost_matrix[row_ind, col_ind].mean():.2f} 像素")
    print(f"【匹配成功】 阈值 {dist_threshold} 内匹配成功的点数: {tp_endpoints}")
    print("============================\n")
    return endpoint_acc, topo_precision, attr_acc


# ==========================================
# 批量导入并计算指标的执行脚本
# ==========================================
def load_and_evaluate(gt_json_path, pred_json_path):
    # 1. 读取人工标注的 GT 数据
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_json = json.load(f)

    # 2. 读取 Pipeline 生成的评估专用预测数据
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_json = json.load(f)

    # 3. 传入评估函数自动比对
    ep_acc, topo_prec, attr_acc = evaluate_pfd_topology(gt_json, pred_json, dist_threshold=5.0)

    return ep_acc, topo_prec, attr_acc


# 执行测试
if __name__ == "__main__":
    gt_file = "C:\happpy_paper\拓扑准确率计算\Simplified-process-flow-diagram-of-chemical-absorption-process-for-postcombustion-CO2_png.rf.2e33d8cff0da801eed0d1a3e08c2753d_gt.json"  # 你人工标注生成的文件
    pred_file = "C:\happpy_paper\拓扑准确率计算\prediction_for_eval_image10.json"  # Pipeline 跑出来的评测文件

    try:
        ep_acc, topo_prec, attr_acc = load_and_evaluate(gt_file, pred_file)
        print("====== 实验评估结果 ======")
        print(f"1. 端点定位精度:       {ep_acc:.2%}")
        print(f"2. 拓扑连通正确率:     {topo_prec:.2%}")
        print(f"3. 端口属性分类准确率: {attr_acc:.2%}")
        print("==========================")
    except Exception as e:
        print(f"比对失败，请检查文件格式: {e}")