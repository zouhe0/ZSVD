import argparse
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

from data import Dataset, ReducedDataset  # 数据加载器
from mymodel import FusionNet  # 学生模型
from loss import LossCalculator  # 损失计算器
from teacher_output_loader import load_teacher_output  # 教师输出直接来自FusionMamba_2024结果文件

# ================== 基础设置 =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ============= 超参数设置 ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0028, help="学习率")
parser.add_argument("--epochs", type=int, default=250, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--device", type=str, default='cuda:0', help="训练设备")
parser.add_argument("--data_id", type=int, default=0, help="数据ID (0-19)")
parser.add_argument("--sensor", type=str, default='wv3', help="传感器类型")
parser.add_argument("--ratio", type=int, default=4, help="下采样比例")
parser.add_argument("--temperature", type=float, default=1.0, help="蒸馏温度参数")
parser.add_argument("--alfa", type=float, default=0.15, help="损失权重")
parser.add_argument("--data_path", type=str, default='/HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5', help="数据文件路径")
parser.add_argument("--use_reduced", type=int, default=1, choices=[0, 1],
                    help="是否启用reduced数据训练 (1启用, 0禁用)")
parser.add_argument("--reduced_data_path", type=str, default=None,
                    help="预构造的reduced数据h5路径，默认: <data_path>去掉.h5加_reduced.h5")
parser.add_argument("--reduced_ratio", type=float, default=10.0,
                    help="reduced数据比例(0~100): 0=纯full(无监督), 100=纯reduced(有监督), 默认10")
parser.add_argument("--reduced_loss_weight", type=float, default=1.0,
                    help="reduced数据gt损失的权重")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
data_id = args.data_id
sensor = args.sensor.upper()
ratio = args.ratio

ri = 140
alfa = args.alfa 
# 损失权重设置（与train_or.py保持一致）
w_var = 1850000.0   # 变分损失权重（对应train_or.py中的w_var）
w_spa = alfa*ri      # 空间保真度损失权重
w_spec = (1-alfa)*ri      # 光谱损失权重





# 数据文件路径
data_path = args.data_path

# reduced 数据相关设置
use_reduced = bool(args.use_reduced)
reduced_ratio = max(0.0, min(100.0, float(args.reduced_ratio)))
reduced_loss_weight = args.reduced_loss_weight
if args.reduced_data_path and args.reduced_data_path != 'None':
    reduced_data_path = args.reduced_data_path
else:
    reduced_data_path = os.path.splitext(data_path)[0] + "_reduced.h5"

# U2Net预训练模型路径

# =================== 模型初始化 =================== #
# 学生模型 (FusionNet)
model_student = FusionNet().to(device)
print("学生模型初始化完成")

# 教师模型输出直接来自FusionMamba_2024结果文件（不加载教师网络）

# 损失计算器初始化
loss_calculator = LossCalculator(sensor=sensor, ratio=ratio, N=41, device=device)

# 优化器
optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))

# 模型保存函数
def save_checkpoint(model, identifier):
    os.makedirs("model_FUG", exist_ok=True)
    model_out_path = os.path.join("model_FUG", f"{identifier}.pth")
    torch.save(model.state_dict(), model_out_path)

# ================ 知识蒸馏训练过程 ================ #
def train(training_data_loader, reduced_data_loader, identifier, teacher_output):
    print("开始知识蒸馏训练...")

    start_time = time.time()

    # 教师输出直接来自FusionMamba_2024结果文件（不运行教师网络）
    teacher_outputs = [teacher_output]
    
    min_full_loss = float("inf")
    min_red_loss = float("inf")
    full_log_interval = 10  # full 损失每隔 10 个 epoch 打印一次
    target_count = int(round(epochs * reduced_ratio / 100.0))
    reduced_log_interval = min(full_log_interval, max(1, target_count))  # reduced 每隔 min(10, epochs*reduced比例) 个 epoch 打印一次
    last_full_log = None  # (epoch, var, spa, spec, total)：最近一个 full 轮的损失，供 reduced 轮打印 full 损失
   
    for epoch in range(1, epochs + 1):
        model_student.train()
        epoch_total_loss = []
        epoch_loss_var, epoch_loss_spa, epoch_loss_spec, epoch_loss_red = [], [], [], []

        # reduced 混合比例(0~100): 0 → 纯full(无监督)，100 → 纯reduced(有监督)
        # 将 reduced 轮次按比例均匀分布到整个训练过程
        target_count = int(round(epochs * reduced_ratio / 100.0))
        prev_red = ((epoch - 1) * target_count) // epochs
        cur_red = (epoch * target_count) // epochs
        use_reduced_epoch = (reduced_data_loader is not None) and (cur_red > prev_red)

        if use_reduced_epoch:
            # ---------- reduced 数据: 用 gt 构造损失 ----------
            for i, batch in enumerate(reduced_data_loader):
                lms, pan, gt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer.zero_grad()

                # 确保pan维度正确
                if len(pan.shape) == 3:
                    pan = pan.unsqueeze(1)

                # 学生模型前向传播
                res_student = model_student(lms, pan)
                fusion_out = res_student + lms
                fusion_out = fusion_out.squeeze(0)  # [C, H, W]
                gt = gt.squeeze(0)

                # reduced 数据由 full 构造得到，有 gt，直接用 gt 构造损失
                loss_red = torch.mean((fusion_out - gt) ** 2)
                total_loss = reduced_loss_weight * loss_red

                epoch_loss_red.append(loss_red.item())
                epoch_total_loss.append(total_loss.item())

                total_loss.backward()
                optimizer.step()
        else:
            # ---------- full 数据: 保持原有损失（蒸馏 + 空间 + 光谱） ----------
            for i, batch in enumerate(training_data_loader):
                # 跳过教师模型预计算失败的批次
                if teacher_outputs[i] is None:
                    continue

                ms, lms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer.zero_grad()

                # 确保pan维度正确
                if len(pan.shape) == 3:
                    pan = pan.unsqueeze(1)

                # 学生模型前向传播
                res_student = model_student(lms, pan)
                fusion_out = res_student + lms
                fusion_out = fusion_out.squeeze(0)  # [C, H, W]

                # 使用预计算的教师模型输出
                fusion_out_teacher = teacher_outputs[i].to(device).squeeze(0)

                # 变分损失 (与train_or.py中的loss1对应)
                loss_var = torch.mean((fusion_out - fusion_out_teacher) ** 2)

                # 转换为[H, W, C]格式用于计算其他损失
                fusion_out_hw_c = fusion_out.permute(1, 2, 0)

                # 获取低分辨率图像尺寸并计算block_size
                _, H, _ = ms[0].shape
                block_size = H // ratio

                # 空间保真度损失 (与train_or.py中的loss2对应)
                loss_spa = loss_calculator.compute_spatial_fidelity_loss(
                    fusion_out_hw_c,
                    ms[0].permute(1, 2, 0),
                    pan[0].squeeze(0),
                    block_size
                )

                # 光谱损失 (与train_or.py中的loss3对应)
                loss_spec = loss_calculator.compute_spectral_loss(
                    fusion_out_hw_c,
                    ms[0].permute(1, 2, 0)
                )

                # 总损失 (使用train_or.py中的权重)
                total_loss = w_var * loss_var + w_spa * loss_spa + w_spec * loss_spec

                # 记录各损失值
                epoch_loss_var.append(loss_var.item())
                epoch_loss_spa.append(loss_spa.item())
                epoch_loss_spec.append(loss_spec.item())
                epoch_total_loss.append(total_loss.item())

                # 反向传播与优化
                total_loss.backward()
                optimizer.step()

        # 计算平均损失
        if use_reduced_epoch:
            avg_loss_red = np.mean(epoch_loss_red)
            avg_total_loss = reduced_loss_weight * avg_loss_red
        else:
            avg_loss_var = np.mean(epoch_loss_var)
            avg_loss_spa = np.mean(epoch_loss_spa)
            avg_loss_spec = np.mean(epoch_loss_spec)
            avg_total_loss = w_var * avg_loss_var + w_spa * avg_loss_spa + w_spec * avg_loss_spec

        # 输出损失
        # full 损失每隔 full_log_interval(10) 个 epoch 打印一次；若该轮是 reduced 轮，
        # 则打印最近一个 full 轮的损失；reduced 损失每隔 reduced_log_interval 个 epoch 打印一次
        if use_reduced_epoch:
            if epoch % reduced_log_interval == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}] [REDUCED] - Loss_gt: {avg_loss_red:.6f}, "
                      f"Total Loss: {avg_total_loss:.6f}")
                if last_full_log is not None and (epoch % full_log_interval == 0 or epoch == epochs):
                    le, lv, ls, lsp, lt = last_full_log
                    print(f"Epoch [{epoch}/{epochs}] (full@epoch {le}) - Loss1(var): {lv:.6f}, "
                          f"Loss2(fspta): {ls:.6f}, Loss3(fspec): {lsp:.6f}, "
                          f"Total Loss: {lt:.6f}")
        else:
            last_full_log = (epoch, avg_loss_var, avg_loss_spa, avg_loss_spec, avg_total_loss)
            if epoch % full_log_interval == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}] - Loss1(var): {avg_loss_var:.6f}, "
                      f"Loss2(fspta): {avg_loss_spa:.6f}, Loss3(fspec): {avg_loss_spec:.6f}, "
                      f"Total Loss: {avg_total_loss:.6f}")

        # 保存最佳模型
        # full 与 reduced 的总损失量级相差约 7 个数量级，不能混在一起比较；
        # 否则 reduced 轮会永远刷新 best，存下"刚被 reduced 推离 full 最优方向"的模型。
        # best 只按 full 轮损失选择；纯 reduced 训练（无 full 轮）时按 reduced 损失兜底。
        if use_reduced_epoch:
            if avg_total_loss < min_red_loss:
                min_red_loss = avg_total_loss
                if min_full_loss == float("inf"):
                    save_checkpoint(model_student, f"{identifier}_best")
        else:
            if avg_total_loss < min_full_loss:
                min_full_loss = avg_total_loss
                save_checkpoint(model_student, f"{identifier}_best")

    
    # 训练完成
    total_time = time.time() - start_time
    print(f"训练完成，总耗时 {total_time:.2f} 秒")
    print(f"最终最佳损失(full): {min_full_loss:.6f}")
    print(f"最终最佳损失(reduced): {min_red_loss:.6f}")
    print(f"模型已保存至: {identifier}_best.pth")

# ================ 主函数 ================ #
def main():
    # 数据加载
    train_set = Dataset(data_path, data_id)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # reduced 数据加载（预构造文件，避免每次训练重新降采样）
    reduced_loader = None
    if use_reduced:
        if os.path.exists(reduced_data_path):
            try:
                reduced_set = ReducedDataset(reduced_data_path, data_id)
                reduced_loader = DataLoader(
                    dataset=reduced_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True
                )
                print(f"已加载reduced数据: {reduced_data_path} (reduced比例: {reduced_ratio:.0f}%)")
            except Exception as e:
                print(f"加载reduced数据失败: {str(e)}，本次仅使用full数据训练")
                reduced_loader = None
        else:
            print(f"警告: reduced数据文件不存在: {reduced_data_path}")
            print("请先运行 prepare_reduced_data.py 构造数据，本次仅使用full数据训练")
    
    # 导入教师模型输出（来自FusionMamba_2024 results/WV3_full，不加载教师网络）
    teacher_output, teacher_mat_path = load_teacher_output(data_id, device)
    print(f"成功导入教师模型输出: {teacher_mat_path}")

    # 模型标识
    identifier = f"{sensor}_{data_id}_FusionNet"
    
    # 开始训练
    train(train_loader, reduced_loader, identifier, teacher_output)

# 执行主函数
if __name__ == "__main__":
    main()
