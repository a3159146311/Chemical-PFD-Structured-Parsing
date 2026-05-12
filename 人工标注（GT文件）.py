import cv2
import json
import numpy as np
import os


class PFDAnnotator:
    def __init__(self, image_path, output_dir="annotated_results"):
        self.image_path = image_path
        self.output_dir = output_dir  # 指定保存结果的文件夹

        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"无法读取图像，请检查路径是否正确: {image_path}")

        self.clone = self.img.copy()
        self.img_disp = self.clone.copy()  # 将显示图像设为实例属性，方便后续保存

        self.nodes = {}  # {node_id: {"coord": [x, y], "attr": attr}}
        self.edges = []  # [[src_id, tgt_id], ...]
        self.node_counter = 0

        self.selected_node = None
        self.window_name = "PFD Topology Annotator"

    def draw(self):
        self.img_disp = self.clone.copy()

        # 绘制管线边
        for src, tgt in self.edges:
            pt1 = tuple(self.nodes[src]["coord"])
            pt2 = tuple(self.nodes[tgt]["coord"])
            cv2.arrowedLine(self.img_disp, pt1, pt2, (0, 255, 0), 2, tipLength=0.05)

        # 绘制端点与属性
        for nid, data in self.nodes.items():
            pt = tuple(data["coord"])
            color = (255, 0, 0) if data["attr"] == 'Inflow' else (0, 0, 255) if data["attr"] == 'Outflow' else (
                0, 255, 255)
            cv2.circle(self.img_disp, pt, 5, color, -1)
            cv2.putText(self.img_disp, f"{nid}:{data['attr'][0]}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(self.window_name, self.img_disp)

    def mouse_callback(self, event, x, y, flags, param):
        # 左键添加节点
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"在 ({x}, {y}) 添加节点。请按键选择属性: [i]Inflow, [o]Outflow, [n]None")
            key = cv2.waitKey(0) & 0xFF
            attr = 'Inflow' if key == ord('i') else 'Outflow' if key == ord('o') else 'None'

            self.nodes[self.node_counter] = {"coord": [x, y], "attr": attr}
            print(f"节点 {self.node_counter} ({attr}) 已添加。")
            self.node_counter += 1
            self.draw()

        # 右键选择节点建立边
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 寻找最近的节点
            min_dist = float('inf')
            closest_node = None
            for nid, data in self.nodes.items():
                nx, ny = data["coord"]
                dist = np.hypot(nx - x, ny - y)
                if dist < 15 and dist < min_dist:
                    min_dist = dist
                    closest_node = nid

            if closest_node is not None:
                if self.selected_node is None:
                    self.selected_node = closest_node
                    print(f"选定起点节点: {closest_node}。请右键点击终点节点。")
                else:
                    self.edges.append([self.selected_node, closest_node])
                    print(f"建立有向边: {self.selected_node} -> {closest_node}")
                    self.selected_node = None
                    self.draw()

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.draw()

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):  # 保存并退出
                # 确保输出文件夹存在
                os.makedirs(self.output_dir, exist_ok=True)

                # 获取图像的基础文件名（不包含路径和后缀）
                base_name = os.path.basename(self.image_path)
                name_without_ext = os.path.splitext(base_name)[0]

                # 1. 保存 JSON 数据
                out_json = os.path.join(self.output_dir, f"{name_without_ext}_gt.json")
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump({"nodes": self.nodes, "edges": self.edges}, f, indent=4)

                # 2. 保存带有标注的图像
                out_image = os.path.join(self.output_dir, f"{name_without_ext}_annotated.jpg")
                cv2.imwrite(out_image, self.img_disp)

                print(f"标注数据已保存至: {out_json}")
                print(f"标注图像已保存至: {out_image}")
                break
            elif k == 27:  # ESC 退出不保存
                print("已取消标注并退出。")
                break
        cv2.destroyAllWindows()


# 使用示例
# 注意：Windows 路径前面最好加上 r，声明为原始字符串，防止 \x \n 等被当成转义字符
image_file = r"C:\happpy_paper\good PFD\3-Figure2-1_png.rf.c579082598f232704413fe339c664c78.jpg"

# 实例化时可以指定保存图和数据的目标文件夹，默认会建在当前运行路径下的 "annotated_results" 文件夹
annotator = PFDAnnotator(image_path=image_file, output_dir=r"C:\happpy_paper\annotated_results")
annotator.run()