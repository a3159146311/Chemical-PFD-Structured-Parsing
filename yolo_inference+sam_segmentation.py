###阶段二代码

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ==========================================
# 1. 配置路径 (替换路径)
# ==========================================
yolo_weight_path = "/kaggle/input/datasets/hejiayi666/123567899/yolo/yolov8m/train/weights/best.pt"
sam_weight_path = "sam_vit_h_4b8939.pth"
test_image_path = "/kaggle/input/datasets/hejiayi666/123567899/yolo/good PFD/3-Figure2-1_png.rf.c579082598f232704413fe339c664c78.jpg"

# ==========================================
# 2. 加载双模型 (YOLO + SAM) 到 GPU
# ==========================================
print(" 正在加载模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 YOLO
yolo_model = YOLO(yolo_weight_path)

# 加载 SAM
sam = sam_model_registry["vit_h"](checkpoint=sam_weight_path)
sam.to(device=device)
predictor = SamPredictor(sam)

# ==========================================
# 3. 读取图像并进行 YOLO 推理
# ==========================================
image_bgr = cv2.imread(test_image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 告诉 SAM 我们要处理这张图
predictor.set_image(image_rgb)

# YOLO 找设备
print(" YOLO 正在寻找设备...")
results = yolo_model.predict(image_rgb, conf=0.2, iou=0.5,
                             imgsz=1280,  # 把图纸放大再看
                             augment=True)  # 开启测试时增强 (TTA)，模型会把图片翻转、缩放看好几遍综合出结果) # 设定置信度阈值
boxes = results[0].boxes

# ==========================================
# 4. SAM 批量生成精准掩膜 (Mask)
# ==========================================
if len(boxes) == 0:
    print("⚠️ YOLO 没有在这张图上找到任何设备！")
else:
    print(f"✅ YOLO 找到了 {len(boxes)} 个设备，正在交由 SAM 提取像素轮廓...")

    # 将 YOLO 的框提取为 PyTorch Tensor 格式
    yolo_boxes_tensor = boxes.xyxy.to(device)

    # 转换坐标系以适应 SAM 的输入标准
    transformed_boxes = predictor.transform.apply_boxes_torch(yolo_boxes_tensor, image_rgb.shape[:2])

    # SAM 批量推理
    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False  # 只需要最高置信度的一个轮廓
    )

    # masks 的形状大概是 [设备数量, 1, 高度, 宽度]，全是 True/False 的布尔值
    print(" SAM 掩膜提取完成！")
    # ... (在 print(" SAM 掩膜提取完成！") 这一行下面加上)

    # 打印一下 masks 的形状和基本信息
    print(f"Masks tensor shape: {masks.shape}")
    # 检查第一个 mask 里有没有 True 的像素（即是否有被分割出来的区域）
    if len(masks) > 0:
        first_mask_np = masks[0][0].cpu().numpy()
        print(f"First mask has True pixels: {np.any(first_mask_np)}")
        print(f"First mask max value: {first_mask_np.max()}, min value: {first_mask_np.min()}")

# ==========================================
# 5. 可视化验证结果并保存/下载
# ==========================================
import os
from IPython.display import FileLink, display

plt.figure(figsize=(12, 12))

# 1. 先画底图
plt.imshow(image_rgb)
ax = plt.gca()

# 2. 创建一个纯红色的覆盖层图像
overlay = np.zeros_like(image_rgb, dtype=np.uint8)

# 3. 遍历所有 mask，把对应区域涂红
for i in range(len(boxes)):
    # 获取 mask 的 numpy 数组 (True/False)
    mask_np = masks[i][0].cpu().numpy()

    # 将 Mask 区域涂成纯红色 (R=255, G=0, B=0)
    overlay[mask_np] = [255, 0, 0]

    # 画 YOLO 的红框和文字
    box = boxes.xyxy[i].cpu().numpy()
    cls_id = int(boxes.cls[i].item())
    cls_name = yolo_model.names[cls_id]

    # 画框
    rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                         fill=False, edgecolor='yellow', linewidth=2)
    ax.add_patch(rect)
    # 写字
    ax.text(box[0], box[1] - 5, cls_name, color='black', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='yellow', edgecolor='yellow', alpha=0.8))

# 4. 将红色覆盖层以半透明叠加到原图上
plt.imshow(overlay, alpha=0.5)

plt.axis('off')
plt.title("YOLO Boxes (Yellow) + SAM Masks (Red Overlay)", fontsize=14)


# 5. 将生成的图像保存到 Kaggle 的输出目录
output_filename = "final_result_with_mask.jpg"
# dpi=300 保证下载的图片是高清的，bbox_inches='tight' 可以去除多余的白边
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
print(f"\n✅ 图片已成功保存为: {output_filename}")

# 6. 显示图片
plt.show()

# 7. 生成可点击的下载链接
print("👇 点击下方链接下载图片到本地电脑 👇")
display(FileLink(output_filename))