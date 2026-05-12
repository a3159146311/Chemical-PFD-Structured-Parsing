import cv2
import numpy as np
import os
import glob


def calculate_optimized_snr(image_path, block_size=32):
    """
    通过寻找最平坦区域来估算图像的 SNR (单位: dB)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    min_std = float('inf')
    best_mu = 0

    # 遍历图像，寻找局部方差最小的块（背景区域）
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            roi = img[y:y + block_size, x:x + block_size]
            current_std = np.std(roi)

            # 寻找最平坦的区域作为噪声参考
            if current_std < min_std and current_std > 0:
                min_std = current_std
                best_mu = np.mean(roi)

    # 全局信号强度 (均值)
    global_mu = np.mean(img)

    # 防止除以零
    if min_std == 0:
        return float('inf')

    # 计算 SNR = 20 * log10(信号均值 / 噪声标准差)
    snr = 20 * np.log10(global_mu / min_std)
    return snr


def batch_process_snr(folder_path, threshold=30):
    # 支持常见的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    print(f"{'文件名':<30} | {'SNR (dB)':<10} | {'结果'}")
    print("-" * 55)

    pass_count = 0
    for img_path in image_files:
        filename = os.path.basename(img_path)
        snr_val = calculate_optimized_snr(img_path)

        if snr_val is None:
            print(f"{filename:<30} | 读取失败")
            continue

        status = "✅ 达标" if snr_val >= threshold else "❌ 噪声大"
        if snr_val >= threshold:
            pass_count += 1

        print(f"{filename[:30]:<30} | {snr_val:>8.2f} | {status}")

    print("-" * 55)
    print(f"检测完成！达标率: {pass_count}/{len(image_files)}")

# --- 使用方法 ---
# 替换为你存放 PFD 图片的文件夹路径
folder_to_check = "C:\happpy_paper\Part 1_YOLO\dataset_image(train)"
batch_process_snr(folder_to_check)