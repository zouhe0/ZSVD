"""
prepare_reduced_data.py

使用 Wald 降采样协议提前构造 reduced 仿真数据，避免每次训练都重新降采样。

输出: <data_path 去扩展名>_reduced.h5，包含三个键:
    lms : reduced 多光谱 (由 full 的 lms 经 MTF 模糊 + ratio 倍下采样得到)  [N, C, H/r, W/r]
    pan : reduced 全色   (由 full 的 pan 经 MTF 模糊 + ratio 倍下采样得到)  [N, 1, H/r, W/r]
    gt  : 原始低分辨率多光谱 ms (reduced 数据的 ground truth)               [N, C, H/r, W/r]

数值尺度与原始 h5 一致 (0~2047, float32)，训练时 Dataset 会除以 2047。
默认处理全部图像，输出文件索引与原始 data_id 一一对应；
若使用 --start_id/--end_id 只构造部分数据，输出文件索引 0 对应 start_id。

用法:
    WALD_DEVICE=cuda:0 python prepare_reduced_data.py \
        --data_path test_wv3_OrigScale_multiExm1.h5 --sensor WV3 --ratio 4
"""
import argparse
import os

import h5py
import numpy as np
import torch

from wald_utilities import wald_protocol_v1, wald_protocol_v2


def main():
    parser = argparse.ArgumentParser(description="用 Wald 协议提前构造 reduced 仿真数据")
    parser.add_argument("--data_path", type=str, required=True, help="原始 h5 数据文件路径")
    parser.add_argument("--output_path", type=str, default=None,
                        help="输出 h5 路径，默认: <data_path 去扩展名>_reduced.h5")
    parser.add_argument("--sensor", type=str, default="WV3", help="传感器类型 (WV3/WV2/QB/GF2 等)")
    parser.add_argument("--ratio", type=int, default=4, help="下采样比例")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    parser.add_argument("--start_id", type=int, default=0, help="起始数据ID (默认0)")
    parser.add_argument("--end_id", type=int, default=None, help="结束数据ID(含)，默认处理全部")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.data_path}")

    output_path = args.output_path or os.path.splitext(args.data_path)[0] + "_reduced.h5"

    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    print(f"设备: {device}, 传感器: {args.sensor}, 降采样比例: {args.ratio}")

    with h5py.File(args.data_path, 'r') as f:
        ms = np.array(f['ms'], dtype=np.float32) / 2047.0
        lms = np.array(f['lms'], dtype=np.float32) / 2047.0
        pan = np.array(f['pan'], dtype=np.float32) / 2047.0

    num_total = ms.shape[0]
    end_id = args.end_id if args.end_id is not None else num_total - 1
    ids = list(range(args.start_id, end_id + 1))
    print(f"共 {num_total} 张图，本次构造 ID {args.start_id} ~ {end_id} 共 {len(ids)} 张")

    lms_rr_list, pan_rr_list = [], []
    for i in ids:
        ms_t = torch.from_numpy(ms[i:i + 1]).to(device)
        lms_t = torch.from_numpy(lms[i:i + 1]).to(device)
        pan_t = torch.from_numpy(pan[i:i + 1]).to(device)

        # 与 pretrain.py 相同的 Wald 降采样约定: 对 full 的 lms/pan 做 MTF 模糊 + ratio 倍下采样
        lms_rr = wald_protocol_v1(lms_t, pan_t, args.ratio, args.sensor)
        pan_rr = wald_protocol_v2(ms_t, pan_t, args.ratio, args.sensor)

        lms_rr_list.append(np.asarray(lms_rr.detach().cpu(), dtype=np.float32))
        pan_rr_list.append(np.asarray(pan_rr.detach().cpu(), dtype=np.float32))
        print(f"ID {i}: lms_rr {tuple(lms_rr.shape)}, pan_rr {tuple(pan_rr.shape)}")

    lms_rr_all = np.concatenate(lms_rr_list, axis=0) * 2047.0
    pan_rr_all = np.concatenate(pan_rr_list, axis=0) * 2047.0
    gt_all = ms[ids] * 2047.0

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('lms', data=lms_rr_all.astype(np.float32))
        f.create_dataset('pan', data=pan_rr_all.astype(np.float32))
        f.create_dataset('gt', data=gt_all.astype(np.float32))
    print(f"reduced 数据已保存至: {output_path}")
    print(f"  lms: {lms_rr_all.shape}, pan: {pan_rr_all.shape}, gt: {gt_all.shape}")


if __name__ == "__main__":
    main()
