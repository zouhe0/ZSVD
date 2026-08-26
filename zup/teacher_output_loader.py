"""
teacher_output_loader.py

教师模型输出加载器：不再加载/运行教师网络（U2Net），
直接从 FusionMamba_2024 的结果目录导入预计算的教师融合输出。

默认目录: /media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_full
可用环境变量 TEACHER_RESULT_DIR 覆盖（服务器路径不同时使用）。
文件: output_mulExm_{data_id}.mat, 键 'sr', 形状 (H, W, C), 数值范围 0~2047。
"""
import os
import numpy as np
import scipy.io as sio
import torch

DEFAULT_TEACHER_RESULT_DIR = r"/media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_full"


def load_teacher_output(data_id, device, result_dir=None):
    """返回 (teacher_output, mat_path)

    teacher_output: (1, C, H, W) 的归一化张量 (0~1)，与训练数据同一尺度（/2047）。
    """
    result_dir = result_dir or os.environ.get("TEACHER_RESULT_DIR", DEFAULT_TEACHER_RESULT_DIR)
    mat_path = os.path.join(result_dir, f"output_mulExm_{data_id}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"教师模型输出文件不存在: {mat_path}")

    data = sio.loadmat(mat_path)
    if "sr" not in data:
        raise KeyError(f"{mat_path} 中未找到键 'sr'，实际键: {[k for k in data if not k.startswith('__')]}")

    sr = np.asarray(data["sr"], dtype=np.float32) / 2047.0   # (H, W, C)
    sr = torch.from_numpy(sr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return sr.to(device), mat_path
