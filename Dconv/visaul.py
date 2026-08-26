import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt

def adjust_brightness(rgb, brightness_factor=1.0):
    """调整图像亮度，单独调整每个通道的亮度因子"""
    # 假设 img 是一个 H x W x 3 的 RGB 图像
    if rgb.shape[2] == 3:
        # 分别增加每个通道的亮度
        rgb[:, :, 0] = rgb[:, :, 0] * brightness_factor # 红色通道
        rgb[:, :, 1] = rgb[:, :, 1] * brightness_factor  # 绿色通道
        rgb[:, :, 2] = rgb[:, :, 2] * brightness_factor  # 蓝色通道
    else:
        print("Error: The input image should be an RGB image with 3 channels.")
        return rgb

    rgb = np.clip(rgb, 0, 1)  # 确保像素值在 [0, 1] 之间
    return rgb

def normalize(img):
    """归一化到 [0, 1]"""
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)
def stretch(band, low=10, high=90):
    p_low = np.percentile(band, low)
    p_high = np.percentile(band, high)
    return np.clip((band - p_low) / (p_high - p_low + 1e-8), 0, 1)


def create_false_color_image(ms_image, bands=(4,2,1), gamma=0.8):
    """
    将多光谱图像转换为伪彩色图像
    - ms_image: H x W x C 多光谱图像
    - bands: 用于组成 RGB 的三个波段索引 (R, G, B)，Python 从0开始
    - gamma: 伽马矫正因子
    """
    rgb = np.stack([
        normalize(ms_image[:, :, bands[0]]),
        normalize(ms_image[:, :, bands[1]]),
        normalize(ms_image[:, :, bands[2]])
    ], axis=-1)
    rgb = adjust_brightness(rgb,1.5)
    if gamma != 1.0:
        rgb = np.power(rgb, gamma)

    rgb = (rgb * 255).astype(np.uint8)
    return rgb

def load_mat_image(mat_path, key='ms'):
    """加载 .mat 文件，返回 H x W x C 格式的图像"""
    data = sio.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_path}, available keys: {list(data.keys())}")
    img = data[key]
    # 保证是 HWC 格式（有些是 CHW）
    if img.shape[0] < 10:  # 假设通道数 < 10
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    return img

# === 示例用法 ===
mat_path = '/HardDisk/HeZou/zup/result/WV3/9_student_0.15.mat'         # 替换为你自己的路径
key = 'proposed'                          # 替换为mat中的key名称
ms_img = load_mat_image(mat_path, key=key)

false_color = create_false_color_image(ms_img)
false_color = cv2.cvtColor(false_color, cv2.COLOR_RGB2BGR)  # 转换为BGR格式

# 保存为图像
cv2.imwrite('false_color.png', false_color)
