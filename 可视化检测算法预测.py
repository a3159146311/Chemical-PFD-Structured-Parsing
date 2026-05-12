import cv2
import json


def visualize_prediction(image_path, json_path):
    # 1. 读取原图和预测 JSON
    img = cv2.imread(image_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    edges = data.get("edges", [])

    # 2. 画边 (先画边，以免挡住节点)
    for src_id, tgt_id in edges:
        if src_id in nodes and tgt_id in nodes:
            pt1 = tuple(nodes[src_id]["coord"])
            pt2 = tuple(nodes[tgt_id]["coord"])
            # 用绿色画出算法预测的流向
            cv2.arrowedLine(img, pt1, pt2, (0, 255, 0), 2, tipLength=0.03)

    # 3. 画节点与属性
    for node_id, info in nodes.items():
        pt = tuple(info["coord"])
        attr = info["attr"]

        # 红色代表 Outflow(流出起点)，蓝色代表 Inflow(流入终点)
        color = (255, 0, 0) if attr == 'Inflow' else (0, 0, 255) if attr == 'Outflow' else (0, 255, 255)
        cv2.circle(img, pt, 6, color, -1)
        cv2.putText(img, attr[0], (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. 显示结果
    cv2.imshow("Algorithm Prediction Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 可选：保存可视化结果用于论文配图
    cv2.imwrite("prediction_visualization.png", img)

# 使用示例
visualize_prediction("C:\happpy_paper\good PFD/3-Figure2-1_png.rf.c579082598f232704413fe339c664c78.jpg", "C:\happpy_paper\拓扑准确率计算\手动标注与算法预测集\image1\prediction_for_eval_image1.json")