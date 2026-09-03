"""
teacher_output_loader.py

教师输出加载器：支持读取预计算的 MAT 输出，或按需加载 U2Net。

默认目录: /media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_full
可用环境变量 TEACHER_RESULT_DIR 覆盖（服务器路径不同时使用）。
文件: output_mulExm_{data_id}.mat, 键 'sr', 形状 (H, W, C), 数值范围 0~2047。
"""
import os
import sys
import numpy as np
import scipy.io as sio
import torch

DEFAULT_TEACHER_RESULT_DIR = r"/media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_full"


def load_teacher_output(data_id, device, result_dir=None, expected_shape=None):
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

    sr = np.asarray(data["sr"], dtype=np.float32)
    if sr.ndim != 3:
        raise ValueError(f"教师输出应为 (H, W, C)，实际形状: {sr.shape}")

    sr = sr / 2047.0   # (H, W, C)
    sr = torch.from_numpy(sr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    if expected_shape is not None and tuple(sr.shape) != tuple(expected_shape):
        raise ValueError(
            f"教师输出形状 {tuple(sr.shape)} 与学生输入形状 {tuple(expected_shape)} 不一致: {mat_path}"
        )
    return sr.to(device), mat_path


def load_teacher_model(model_path, device):
    """仅在 model 模式下导入并初始化 U2Net。"""
    fusionmamba_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FusionMamba")
    if fusionmamba_dir not in sys.path:
        sys.path.insert(0, fusionmamba_dir)

    from model.u2net import U2Net

    model_teacher = U2Net(
        dim=32,
        pan_dim=1,
        ms_dim=8,
        H=512,
        W=512
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "state_dict" in checkpoint:
        model_teacher.load_state_dict(checkpoint["state_dict"])
    else:
        model_teacher.load_state_dict(checkpoint)
    model_teacher.eval()
    return model_teacher
