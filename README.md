# HeZou

融合（Fusion）相关模型代码仓库。

## 目录结构

- `Dconv/`：模型训练与测试代码（FusionMamba、SDE 等）
- `zup/`：模型训练与测试代码（含 loss landscape 可视化等）


## Reduced 数据增强训练（zup/trainba.py / zup/train_SDE.py）

- 先用 `zup/prepare_reduced_data.py` 用 Wald 降采样协议提前构造 reduced 仿真数据（存为 `*_reduced.h5`，避免每次训练重复构造）。
- `trainba.py` 与 `train_SDE.py` 按 `--reduced_ratio`（0~100）比例混合 reduced 数据：reduced 数据用 gt（原始低分辨率 ms）构造损失，full 数据保持原有蒸馏+空间+光谱损失。
- 常用参数：`--use_reduced 1 --reduced_ratio 10 --reduced_loss_weight 1.0`（0=纯full无监督，100=纯reduced有监督）（runba 的 process_model=0/1 均透传）。

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

### 2. 单图训练（`--reduced_ratio` 控制 reduced 比例，0~100）

```bash
WALD_DEVICE=cuda:0 python trainba.py --data_id 0 \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --epochs 240 --reduced_ratio 10 --device cuda:0
```

### 3. 通过 runba.py 走完整流程（训练 + 测试）

```bash
WALD_DEVICE=cuda:0 python runba.py --process_model 1 \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --epochs 240 --reduced_ratio 0 --device cuda:0
```

> 教师模型输出直接来自 FusionMamba_2024 的 `results/WV3_full/output_mulExm_*.mat`（不再加载教师网络），可用环境变量 `TEACHER_RESULT_DIR` 覆盖目录。
> 一阶段训练：`runba.py` 的 `process_model=1` 流程已去掉 `pretrain.py`（reduced 预热）阶段，`train_SDE.py` 直接从随机初始化训练融合网络，不再加载 `model_pretrain` 权重。

> 说明：模型权重、数据集、conda 环境与第三方仓库（DLPan-Toolbox、Vim）未纳入本仓库。
