# HeZou

融合（Fusion）相关模型代码仓库。

## 目录结构

- `Dconv/`：模型训练与测试代码（FusionMamba、SDE 等）
- `zup/`：模型训练与测试代码（含 loss landscape 可视化等）


## Reduced 数据增强训练（zup/trainba.py）

- 先用 `zup/prepare_reduced_data.py` 用 Wald 降采样协议提前构造 reduced 仿真数据（存为 `*_reduced.h5`，避免每次训练重复构造）。
- `trainba.py` 每 `--reduced_every` 个 epoch 用一次 reduced 数据训练：reduced 数据用 gt（原始低分辨率 ms）构造损失，full 数据保持原有蒸馏+空间+光谱损失。
- 常用参数：`--use_reduced 1 --reduced_every 10 --reduced_loss_weight 1.0`。

> 说明：模型权重、数据集、conda 环境与第三方仓库（DLPan-Toolbox、Vim）未纳入本仓库。
