# HeZou

融合（Fusion）相关模型代码仓库。

## 目录结构

- `Dconv/`：模型训练与测试代码（FusionMamba、SDE 等）
- `zup/`：模型训练与测试代码（含 loss landscape 可视化等）


## Reduced 数据增强训练（zup/trainba.py）

- 先用 `zup/prepare_reduced_data.py` 用 Wald 降采样协议提前构造 reduced 仿真数据（存为 `*_reduced.h5`，避免每次训练重复构造）。
- `trainba.py` 每 `--reduced_every` 个 epoch 用一次 reduced 数据训练：reduced 数据用 gt（原始低分辨率 ms）构造损失，full 数据保持原有蒸馏+空间+光谱损失。
- 常用参数：`--use_reduced 1 --reduced_every 10 --reduced_loss_weight 1.0`。

## 运行方法

> 本机只有一张 GPU 时需要 `WALD_DEVICE=cuda:0`；多卡服务器（原 `cuda:2` 环境）可省略该变量。

### 1. 提前构造 reduced 数据（只需一次）

```bash
cd /media/zouhe/Elements/HeZou/zup
WALD_DEVICE=cuda:0 python prepare_reduced_data.py \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --sensor WV3 --ratio 4
```

生成 `test_wv3_OrigScale_multiExm1_reduced.h5`（与原始 h5 同目录）。

### 2. 单图训练（每 `--reduced_every` 个 epoch 训 1 次 reduced）

```bash
WALD_DEVICE=cuda:0 python trainba.py --data_id 0 \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --epochs 240 --reduced_every 10 --device cuda:0
```

### 3. 通过 runba.py 走完整流程（训练 + 测试）

```bash
WALD_DEVICE=cuda:0 python runba.py --process_model 0 --data_id 0 \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --epochs 240 --reduced_every 10 --device cuda:0
```

> 说明：模型权重、数据集、conda 环境与第三方仓库（DLPan-Toolbox、Vim）未纳入本仓库。
