import yaml
from ultralytics import YOLO

# 1. 动态生成适应 Kaggle 路径的 data.yaml

data_config = {
    'train': '/kaggle/input/datasets/hejiayi666/classic/yolo/PFD（class=21）/train/images',
    'val': '/kaggle/input/datasets/hejiayi666/classic/yolo/PFD（class=21）/valid/images',
    'nc': 21, # 类别数量
    'names': ['boiler', 'centrifugal  pump', 'closed gate valve', 'column', 'compressor', 'control valve', 'drum', 'furnace', 'gas blower', 'gate valve', 'globe valve', 'hand operated  gate valve', 'heater', 'jump point', 'packing column', 'screw pump', 'selectable  compressor', 'tank', 'tray column', 'triangle  separator', 'vertical vessel'] # 你的具体类别名，记得修改
}

with open('/kaggle/working/data.yaml', 'w') as f:
    yaml.dump(data_config, f)

# 2. 加载模型并开始训练
model = YOLO("yolov8m.pt")

# 开启训练
results = model.train(
    data='/kaggle/working/data.yaml',
    epochs=100,
    imgsz=1024,
    batch=4,   # 一次喂给显卡 4 张图
    project='/kaggle/working/yolov8m', # 结果保存在工作区，方便下载
    copy_paste=0.5,    # 50%概率触发复制粘贴增强，增加箭头出场率
    degrees=90.0,      # 允许图像旋转90/180/270度
    mixup=0.2,         # 增加图像混合
    mosaic=1.0  # 马赛克数据增强，以进一步提升模型对复杂场景的鲁棒性
)
