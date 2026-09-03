import argparse # type: ignore
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
from data import Dataset  # 数据加载器
from mymodel import FusionNet  # 学生模型
from SDE import Net_ms2pan_dual,sobel_filter,ms2pan_convNet_dual  # SDE模块
print("Updated sys.path:", sys.path)
from loss import LossCalculator  # 损失计算器
from teacher_output_loader import load_teacher_model, load_teacher_output
from wald_utilities import wald_protocol_v1, wald_protocol_v2  # 学生模型

# ================== 基础设置 =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ============= 超参数设置 ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.005, help="学习率")
parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--device", type=str, default='cuda:0', help="训练设备")
parser.add_argument("--data_id", type=int, default=0, help="数据ID (0-19)")
parser.add_argument("--sensor", type=str, default='wv3', help="传感器类型")
parser.add_argument("--ratio", type=int, default=4, help="下采样比例")
parser.add_argument("--temperature", type=float, default=1.0, help="蒸馏温度参数")
# 添加混合精度训练的参数选项
parser.add_argument("--amp",default=True, action="store_true", help="启用混合精度训练")
parser.add_argument("--alfa", type=float, default=0.15, help="损失权重")
parser.add_argument("--data_path", type=str, default=r"HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5", help="数据文件路径")
parser.add_argument("--teacher_source", type=str, default="mat", choices=["mat", "model"], help="教师输出来源")
parser.add_argument("--teacher_result_dir", type=str, default=None, help="MAT教师输出目录")
parser.add_argument("--u2net_path", type=str, default="FusionMamba/weights/420.pth", help="U2Net预训练模型路径")
parser.add_argument(
    "--ablation",
    type=str,
    default="full",
    choices=[
        "full",
        "wo_stage1",
        "wo_pretrain_loss",
        "wo_spatial_loss",
        "wo_spectral_loss",
        "wo_ssat",
    ],
    help="消融实验类型",
)
parser.add_argument("--pretrain_path", type=str, default=None, help="FusionNet预训练权重路径")
parser.add_argument("--sde_path", type=str, default=None, help="SDE空间网络权重路径")
parser.add_argument("--output_path", type=str, default=None, help="主训练最佳权重输出路径")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
data_id = args.data_id
sensor = args.sensor.upper()
ratio = args.ratio
# 混合精度训练标志
use_amp = args.amp and (device.type == 'cuda')  # 只在CUDA设备上启用混合精度


ri = 140*1
alfa = args.alfa 
# 损失权重设置（与train_or.py保持一致）
#w_var = 1800000.0*0.95   # 变分损失权重（对应train_or.py中的w_var，没有像素级权重模型中最优参数

#自适应亚像素级权重参数
w_var = 4230000    #i.e. 1800000*2.35      #1800000.0*2.4#最优值


w_spa = alfa*ri      # 空间保真度损失权重
w_spec = (1-alfa)*ri    # 光谱损失权重

# 数据文件路径
data_path = args.data_path

use_pretrain_init = args.ablation != "wo_stage1"
use_pretrain_loss = args.ablation != "wo_pretrain_loss"
use_spatial_loss = args.ablation != "wo_spatial_loss"
use_spectral_loss = args.ablation != "wo_spectral_loss"
use_ssat = args.ablation != "wo_ssat"


# U2Net预训练模型路径

# =================== 模型初始化 =================== #
# 学生模型 (FusionNet)
model_student = FusionNet().to(device)
if use_pretrain_init:
    pretrain_path = args.pretrain_path or f'model_pretrain/{sensor}/{data_id}_FusionNet_pretrain.pth'
    model_student.load_state_dict(
        torch.load(pretrain_path, map_location=device, weights_only=True)
    )
    print(f"学生模型已加载第一阶段权重: {pretrain_path}")
else:
    print("wo_stage1: 学生模型使用随机初始化")

# 损失计算器初始化
loss_calculator = LossCalculator(sensor=sensor, ratio=ratio, N=41, device=device)

F_ms2pan = None
if use_spatial_loss:
    sde_path = args.sde_path or f'model_SDE/{sensor}/{data_id}_ms2pan_convNet_dual.pth'
    F_ms2pan = ms2pan_convNet_dual().to(device)
    F_ms2pan.load_state_dict(
        torch.load(sde_path, map_location=device, weights_only=True)
    )
    F_ms2pan.eval()
    print(f"已加载SDE空间网络: {sde_path}")
else:
    print("wo_spatial_loss: 不加载SDE空间网络")


# 优化器
optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))


# 初始化混合精度训练的GradScaler
scaler = torch.amp.GradScaler("cuda") if use_amp else None



if use_amp:
    print("已启用混合精度训练 (AMP)")
elif args.amp and device.type != 'cuda':
    print("警告: 混合精度训练仅支持CUDA设备，已自动禁用")

# 模型保存函数
def save_checkpoint(model, identifier):
    model_out_path = args.output_path or os.path.join("model_FUG", f"{identifier}.pth")
    os.makedirs(os.path.dirname(os.path.abspath(model_out_path)), exist_ok=True)
    torch.save(model.state_dict(), model_out_path)

# 在训练前准备教师模型输出
def prepare_teacher_outputs(training_data_loader):
    if not use_pretrain_loss:
        print("wo_pretrain_loss: 不加载教师输出")
        return None

    if args.teacher_source == "mat":
        expected_shape = next(iter(training_data_loader))[1].shape
        teacher_output, teacher_mat_path = load_teacher_output(
            data_id,
            device,
            args.teacher_result_dir,
            expected_shape
        )
        print(f"成功导入教师输出: {teacher_mat_path}")
        return [teacher_output]

    model_teacher = load_teacher_model(args.u2net_path, device)
    print(f"成功加载U2Net预训练模型: {args.u2net_path}")
    teacher_outputs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(training_data_loader):
            ms, _, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            if len(pan.shape) == 3:
                pan = pan.unsqueeze(1)
            try:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        output = model_teacher(ms, pan)
                else:
                    output = model_teacher(ms, pan)
                if isinstance(output, tuple):
                    output = output[0]
                teacher_outputs.append(output.cpu())
            except Exception as e:
                print(f"批次 {batch_idx} 预计算失败: {str(e)}")
                teacher_outputs.append(None)
    return teacher_outputs


# ================ 知识蒸馏训练过程 ================ #
def train(training_data_loader, identifier, teacher_outputs):
    print("开始知识蒸馏训练...")
    print(
        f"消融配置: {args.ablation} | 第一阶段={use_pretrain_init}, "
        f"教师蒸馏损失={use_pretrain_loss}, 空间损失={use_spatial_loss}, "
        f"光谱损失={use_spectral_loss}, SSAT={use_ssat}"
    )

    min_total_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model_student.train()
        epoch_loss_var, epoch_loss_spa = [], []
        epoch_loss_spec, epoch_total_loss = [], []

        for i, batch in enumerate(training_data_loader):
            if use_pretrain_loss and teacher_outputs[i] is None:
                continue

            ms, lms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            if pan.dim() == 3:
                pan = pan.unsqueeze(1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                res_student = model_student(lms, pan)
                fusion_out_ori = res_student + lms
                fusion_out = fusion_out_ori.squeeze(0)
                fusion_out_hw_c = fusion_out.permute(1, 2, 0)

                loss_var = torch.zeros((), device=device)
                loss_spa = torch.zeros((), device=device)
                loss_spec = torch.zeros((), device=device)
                total_loss = torch.zeros((), device=device)

                if use_pretrain_loss:
                    fusion_out_teacher = teacher_outputs[i].to(device).squeeze(0)
                    if use_ssat:
                        teacher_bchw = fusion_out_teacher.unsqueeze(0)
                        ms_up = F.interpolate(ms, scale_factor=ratio, mode='nearest')
                        delta = (teacher_bchw - ms_up).abs()
                        delta_max = delta.amax(dim=(2, 3), keepdim=True)
                        delta_up = torch.exp(-delta / (delta_max + 1e-8))
                        loss_var = torch.mean(
                            ((fusion_out - fusion_out_teacher) * delta_up) ** 2
                        )
                        loss_spec_teacher = loss_calculator.compute_spectral_loss(
                            fusion_out_teacher.permute(1, 2, 0),
                            ms[0].permute(1, 2, 0),
                        )
                        weight = torch.sigmoid(torch.abs(loss_spec_teacher - 28.65) / 10) * 2.0
                        total_loss = total_loss + (w_var / weight) * loss_var
                    else:
                        loss_var = torch.mean((fusion_out - fusion_out_teacher) ** 2)
                        total_loss = total_loss + w_var * loss_var

                if use_spatial_loss:
                    fusion_gra_x, fusion_gra_y = sobel_filter(fusion_out_ori)
                    sde_out_x, sde_out_y = F_ms2pan(fusion_gra_x, fusion_gra_y)
                    pan_gra_x, pan_gra_y = sobel_filter(pan)
                    loss_x = loss_calculator.SDE_Loss(sde_out_x, pan_gra_x)
                    loss_y = loss_calculator.SDE_Loss(sde_out_y, pan_gra_y)
                    loss_spa = (loss_x + loss_y) / 100
                    total_loss = total_loss + w_spa * loss_spa

                if use_spectral_loss:
                    loss_spec = loss_calculator.compute_spectral_loss(
                        fusion_out_hw_c,
                        ms[0].permute(1, 2, 0),
                    )
                    total_loss = total_loss + w_spec * loss_spec

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            epoch_loss_var.append(loss_var.item())
            epoch_loss_spa.append(loss_spa.item())
            epoch_loss_spec.append(loss_spec.item())
            epoch_total_loss.append(total_loss.item())

        if not epoch_total_loss:
            raise RuntimeError("当前epoch没有有效训练批次")

        avg_loss_var = np.mean(epoch_loss_var)
        avg_loss_spa = np.mean(epoch_loss_spa)
        avg_loss_spec = np.mean(epoch_loss_spec)
        avg_total_loss = np.mean(epoch_total_loss)

        # 与原始runba保持一致：最佳checkpoint按未除以SSAT标量weight的
        # 各活动损失加权和选择，确保full实验可严格复现原始基线。
        checkpoint_loss = 0.0
        if use_pretrain_loss:
            checkpoint_loss += w_var * avg_loss_var
        if use_spatial_loss:
            checkpoint_loss += w_spa * avg_loss_spa
        if use_spectral_loss:
            checkpoint_loss += w_spec * avg_loss_spec

        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss1(var): {avg_loss_var:.6f}, "
                f"Loss2(fspta): {avg_loss_spa:.6f}, Loss3(fspec): {avg_loss_spec:.6f}, "
                f"Total Loss: {avg_total_loss:.6f}, Checkpoint Loss: {checkpoint_loss:.6f}")

        if checkpoint_loss < min_total_loss:
            min_total_loss = checkpoint_loss
            save_checkpoint(model_student, f"{identifier}_best")

    total_time = time.time() - start_time
    model_path = args.output_path or os.path.join("model_FUG", f"{identifier}_best.pth")
    print(f"训练完成，总耗时 {total_time:.2f} 秒")
    print(f"最终最佳损失: {min_total_loss:.6f}")
    print(f"模型已保存至: {model_path}")

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
    
    # 模型标识（如果使用混合精度，则添加amp标记）
    identifier = f"{sensor}_{data_id}_FusionNet_SDE" 
    
    teacher_outputs = prepare_teacher_outputs(train_loader)

    # 开始训练
    train(train_loader, identifier, teacher_outputs)

# 执行主函数
if __name__ == "__main__":
    main()
