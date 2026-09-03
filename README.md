# HeZou

融合（Fusion）相关模型代码仓库。

## 目录结构

- `Dconv/`：模型训练与测试代码（FusionMamba、SDE 等）
- `zup/`：模型训练与测试代码（含 loss landscape 可视化等）

> 说明：模型权重、数据集、conda 环境与第三方仓库（DLPan-Toolbox、Vim）未纳入本仓库。

## zup 训练与测试

`zup/runba.py` 支持两种教师输出来源：

| 参数 | 可选值/默认值 | 说明 |
| --- | --- | --- |
| `--data_path` | PanCollection 的 WV3 Original Scale 测试集 | 输入 H5 数据文件；启动时会检查文件是否存在 |
| `--teacher_source` | `mat`（默认）、`model` | 直接读取 MAT 参考图，或运行预训练 U2Net 教师模型 |
| `--teacher_result_dir` | FusionMamba 的 `WV3_full` 结果目录 | MAT 参考图所在目录 |
| `--u2net_path` | `FusionMamba/weights/420.pth` | 仅在 `teacher_source=model` 时使用 |

### 直接读取 MAT 教师输出

默认方式不会加载或运行 U2Net。请从 `zup/` 目录启动：

```bash
cd /media/zouhe/Elements/HeZou/zup

python runba.py \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --teacher_source mat \
  --teacher_result_dir /media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_full
```

不传教师相关参数时，等价于使用上面的 MAT 模式和 FusionMamba `WV3_full` 目录。

MAT 文件需符合以下约定：

- 文件名为 `output_mulExm_{data_id}.mat`。
- 融合结果存储在 `sr` 键中，形状为 `(H, W, C)`。
- 数值范围为 `0~2047`；加载后会除以 `2047` 并转换为 `(1, C, H, W)`。
- 教师输出形状必须与当前样本的 LMS 形状一致，否则程序会直接报错。

使用 reduced 数据时，需要显式指定对应的参考目录；`--mode reduce` 不会自动切换教师输出目录：

```bash
python runba.py \
  --data_id 0 \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_multiExm1.h5 \
  --mode reduce \
  --teacher_source mat \
  --teacher_result_dir /media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_reduced
```

批量处理时继续使用相同的教师结果目录，程序会根据 `data_id` 选择对应 MAT 文件：

```bash
python runba.py \
  --process_all \
  --start_id 0 \
  --end_id 19 \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --teacher_source mat \
  --teacher_result_dir /media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_full
```

### 使用预训练教师模型

如需恢复原来的教师模型推理流程：

```bash
python runba.py \
  --data_id 0 \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --teacher_source model \
  --u2net_path FusionMamba/weights/420.pth
```

`model` 模式需要完整安装 FusionMamba/U2Net 相关依赖；MAT 模式不需要运行教师网络。

## zup 消融实验

`zup/run_ablation.py` 会使用固定随机种子 10 重新训练完整模型和五组单因素消融，并在全部样本完成后调用 `/media/zouhe/Elements/baseline/baseline_test/evaluate.py` 计算 Full-Resolution 指标。

| 实验名 | 关闭内容 |
| --- | --- |
| `full` | 无，完整模型 |
| `wo_stage1` | FusionNet 第一阶段预训练，主训练使用随机初始化 |
| `wo_pretrain_loss` | 教师蒸馏损失 `loss_var` |
| `wo_spatial_loss` | SDE 空间损失，同时跳过 SDE 权重加载 |
| `wo_spectral_loss` | 显式光谱损失；SSAT 内的教师光谱质量权重仍保留 |
| `wo_ssat` | `delta_up` 和 `weight` 自适应加权，教师损失改为普通 MSE |

完整运行 0 到 19 号 WV3 Full-Resolution 样本：

```bash
cd /media/zouhe/Elements/HeZou/zup

/home/zouhe/miniconda3/envs/zspan/bin/python run_ablation.py \
  --run_name wv3_full_seed10 \
  --output_root /media/zouhe/Elements/HeZou/zup/ablation_results \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --teacher_source mat \
  --teacher_result_dir /media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_full \
  --start_id 0 \
  --end_id 19 \
  --device cuda:0
```

目标 `run_name` 目录已存在时程序会拒绝覆盖。每次实验的输出结构如下：

```text
ablation_results/<run_name>/
├── config.json
├── checkpoints/
│   ├── shared/
│   └── <variant>/
├── mat/<variant>/wv3_full/0.mat ... 19.mat
├── metrics/wv3_full.csv
└── logs/image_<data_id>/
```

六组学生结果共保存 120 个 MAT。每个 MAT 包含 `proposed`、`I_MS_LR`、`I_MS` 和 `I_PAN`；`metrics/wv3_full.csv` 保存各组 `D_lambda_K`、`D_s`、`HQNR` 的均值、标准差和有效样本数。评测前会检查配置范围内所有 MAT 的文件、字段、形状和有限值，任何结果缺失都会停止评测。

快速检查时可以缩小图片范围和训练轮数，例如：

```bash
/home/zouhe/miniconda3/envs/zspan/bin/python run_ablation.py \
  --run_name smoke_image0 \
  --data_path /media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5 \
  --teacher_source mat \
  --teacher_result_dir /media/zouhe/Elements/baseline/pansharpening/FusionMamba_2024/results/WV3_full \
  --start_id 0 --end_id 0 \
  --pre_epochs 1 --sde_epochs 1 --epochs 1 \
  --device cuda:0
```
