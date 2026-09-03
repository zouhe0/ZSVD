import argparse
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

from data import Dataset  # 数据加载器
from mymodel import FusionNet  # 学生模型
from loss import LossCalculator  # 损失计算器
from teacher_output_loader import load_teacher_model, load_teacher_output

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
parser.add_argument("--teacher_source", type=str, default="mat", choices=["mat", "model"], help="教师输出来源")
parser.add_argument("--teacher_result_dir", type=str, default=None, help="MAT教师输出目录")
parser.add_argument("--u2net_path", type=str, default="FusionMamba/weights/420.pth", help="U2Net预训练模型路径")
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

# U2Net预训练模型路径

# =================== 模型初始化 =================== #
# 学生模型 (FusionNet)
model_student = FusionNet().to(device)
print("学生模型初始化完成")

# 损失计算器初始化
loss_calculator = LossCalculator(sensor=sensor, ratio=ratio, N=41, device=device)

# 优化器
optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))

# 模型保存函数
def save_checkpoint(model, identifier):
    os.makedirs("model_FUG", exist_ok=True)
    model_out_path = os.path.join("model_FUG", f"{identifier}.pth")
    torch.save(model.state_dict(), model_out_path)

# 在训练前准备教师模型输出
def prepare_teacher_outputs(training_data_loader):
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

    start_time = time.time()

    min_total_loss = float("inf")
   
    for epoch in range(1, epochs + 1):
        model_student.train()
        epoch_loss_var, epoch_loss_spa, epoch_loss_spec = [], [], []
        
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
            total_loss = w_var * loss_var +w_spa * loss_spa + w_spec * loss_spec
            
            # 记录各损失值
            epoch_loss_var.append(loss_var.item())
            epoch_loss_spa.append(loss_spa.item())
            epoch_loss_spec.append(loss_spec.item())
            
            # 反向传播与优化
            total_loss.backward()
            optimizer.step()
        
        # 计算平均损失
        avg_loss_var = np.mean(epoch_loss_var)
        avg_loss_spa = np.mean(epoch_loss_spa)
        avg_loss_spec = np.mean(epoch_loss_spec)
        avg_total_loss = w_var * avg_loss_var + w_spa * avg_loss_spa + w_spec * avg_loss_spec
        
        # 采用train_or.py的输出频率：每10个epoch输出一次
        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss1(var): {avg_loss_var:.6f}, "
                f"Loss2(fspta): {avg_loss_spa:.6f}, Loss3(fspec): {avg_loss_spec:.6f}, "
                f"Total Loss: {avg_total_loss:.6f}")
        
        # 保存最佳模型
        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            save_checkpoint(model_student, f"{identifier}_best")

    
    # 训练完成
    total_time = time.time() - start_time
    print(f"训练完成，总耗时 {total_time:.2f} 秒")
    print(f"最终最佳损失: {min_total_loss:.6f}")
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
    
    teacher_outputs = prepare_teacher_outputs(train_loader)

    # 模型标识
    identifier = f"{sensor}_{data_id}_FusionNet"
    
    # 开始训练
    train(train_loader, identifier, teacher_outputs)

# 执行主函数
if __name__ == "__main__":
    main()
