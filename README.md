# 化工 PFD 流程图结构化识别系统 (Chemical PFD Structured Parsing)

## 1. 项目简介 (Introduction)
本项目为本科毕业设计的配套代码。项目主要结合了 YOLOv8m 目标检测模型与 SAM (Segment Anything Model) 图像分割模型，旨在实现化工工艺流程图（PFD）中设备与拓扑结构的自动识别，并最终将非结构化的图像转化为标准的 JSON 数据格式。

## 2. 目录结构 (Directory Structure)
为了方便阅读，本项目的核心脚本说明如下：
- `yolo_inference.py`: 使用 YOLOv8m 进行 PFD 图纸中化工设备提取的主程序。
- `sam_segmentation.py`: 调用 SAM 模型进行细粒度图像分割的测试脚本。
- `pipeline_extraction+topological_connection.py`: 基于 OpenCV 编写的图像骨架化提取工具。
- `check_SNR.py`: 用于计算图纸图像信噪比（SNR）的预处理评估脚本。
- `pfd_to_json.py`: 将识别到的设备坐标与拓扑关系转换为标准 JSON 格式的格式化脚本。

## 3. 环境配置 (Requirements)
本项目在以下环境中测试通过：
- Python 3.9+
- PyTorch (建议 GPU 版本)
- ultralytics (YOLOv8)
- opencv-python
