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
# 导入混合精度训练所需的库
from torch.cuda.amp import autocast, GradScaler

from data import Dataset  # 数据加载器
from mymodel import FusionNet  # 学生模型
from SDE import Net_ms2pan_dual,sobel_filter,ms2pan_convNet_dual  # SDE模块
print("Updated sys.path:", sys.path)
from loss import LossCalculator  # 损失计算器
from teacher_output_loader import load_teacher_output  # 教师输出直接来自FusionMamba_2024结果文件
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


# U2Net预训练模型路径

# =================== 模型初始化 =================== #
# 学生模型 (FusionNet)
model_student = FusionNet().to(device)
# 一阶段训练：直接从随机初始化训练融合网络，不加载model_pretrain预热权重
print("学生模型初始化完成")

# 损失计算器初始化
loss_calculator = LossCalculator(sensor=sensor, ratio=ratio, N=41, device=device)

F_ms2pan = ms2pan_convNet_dual().to(device)
F_ms2pan.load_state_dict(torch.load(f'model_SDE/{sensor}/{data_id}_ms2pan_convNet_dual.pth'))
F_ms2pan.eval()


# 优化器
optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))


# 教师模型输出直接来自FusionMamba_2024结果文件（不加载教师网络）



# 初始化混合精度训练的GradScaler
scaler = GradScaler() if use_amp else None



if use_amp:
    print("已启用混合精度训练 (AMP)")
elif args.amp and device.type != 'cuda':
    print("警告: 混合精度训练仅支持CUDA设备，已自动禁用")

# 模型保存函数
def save_checkpoint(model, identifier):
    os.makedirs("model_FUG", exist_ok=True)
    model_out_path = os.path.join("model_FUG", f"{identifier}.pth")
    torch.save(model.state_dict(), model_out_path)

# ================ 知识蒸馏训练过程 ================ #
def train(training_data_loader, identifier, teacher_output):
    print("开始知识蒸馏训练...")
    
    # 教师输出直接来自FusionMamba_2024结果文件（不运行教师网络）
    teacher_outputs = [teacher_output]
    
    min_total_loss = float("inf")
    start_time = time.time()

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
            
            # 使用混合精度进行前向传播和损失计算
            if use_amp:
                with autocast():
                    # 学生模型前向传播
                    res_student = model_student(lms, pan)
                    fusion_out_ori = res_student + lms  
                    fusion_out = fusion_out_ori.squeeze(0)  # [C, H, W]
                    
                    # 使用预计算的教师模型输出
                    fusion_out_teacher = teacher_outputs[i].to(device).squeeze(0)
                    fusion_out_teacher_bchw = fusion_out_teacher.unsqueeze(0)  # 确保维度正确b c h w

                    #计算patch-wise变分损失权重
                    # teacher_outputs_lr = F.max_pool2d(fusion_out_teacher_bchw, kernel_size=4, stride=4)  # 使用最大池化代替Wald协议
                    
                    # delta = (teacher_outputs_lr - ms).abs()                     # 逐元素差值的绝对值
                    # # delta: [B, C, H, W]
                    # delta_max = delta.amax(dim=(2, 3), keepdim=True)   # 每个通道独立的最大值
                    # delta_inv = torch.exp(-delta / (delta_max + 1e-8))
                    # delta_up = F.interpolate(delta_inv, scale_factor=4, mode='nearest')

                    #计算pixel-wise变分损失权重
                    ms_up = F.interpolate(ms, scale_factor=ratio, mode='nearest')
                    delta = (fusion_out_teacher_bchw - ms_up).abs()  #
                    delta_max = delta.amax(dim=(2, 3), keepdim=True)  # 每个通道独立的最大值
                    delta_up = torch.exp(-delta / (delta_max + 1e-8))

                    #print(delta_up.mean().item())
                    # 变分损失 (与train_or.py中的loss1对应)
                    loss_var = torch.mean(((fusion_out - fusion_out_teacher)*delta_up) ** 2)
                    
                    # 转换为[H, W, C]格式用于计算其他损失
                    fusion_out_hw_c = fusion_out.permute(1, 2, 0)
                    
                    # 获取低分辨率图像尺寸并计算block_size
                    _, H, _ = ms[0].shape
                    block_size = H // ratio
                    
                    # # 空间保真度损失 (与train_or.py中的loss2对应)
                    # loss_spa = loss_calculator.compute_spatial_fidelity_loss(
                    #     fusion_out_hw_c,
                    #     ms[0].permute(1, 2, 0),
                    #     pan[0].squeeze(0),
                    #     block_size
                    # )
                    # 使用SDE网络将多光谱图像转换为全色图像
                    with autocast() if use_amp else torch.no_grad():
                        fusion_out_gra_x,fusion_out_gra_y = sobel_filter(fusion_out_ori)
                        sde_out_x,sde_out_y = F_ms2pan(fusion_out_gra_x,fusion_out_gra_y) # 假设SDE网络输出维度为 [batch_size, 1, H, W]，因此需要squeeze掉通道维度

                    # 计算MS再SDE网络输出和pan图的L2损失
                    pan_gra_x,pan_gra_y = sobel_filter(pan)
                    loss_x = loss_calculator.SDE_Loss(sde_out_x , pan_gra_x)
                    loss_y = loss_calculator.SDE_Loss(sde_out_y , pan_gra_y)
                    loss_spa = (loss_x + loss_y)/100
                    # loss_spa = loss_x/100

                    # 光谱损失 (与train_or.py中的loss3对应)
                    loss_spec = loss_calculator.compute_spectral_loss(
                        fusion_out_hw_c, 
                        ms[0].permute(1, 2, 0)
                    )
                    loss_spec_teacher = loss_calculator.compute_spectral_loss(
                        fusion_out_teacher.permute(1, 2, 0), 
                        ms[0].permute(1, 2, 0)
                    )

                    weight = torch.sigmoid(1*(torch.abs(loss_spec_teacher - 28.65)/10))*2.0

                    # 总损失 (使用train_or.py中的权重)

                    total_loss = (w_var/weight) * loss_var + (w_spa * loss_spa + w_spec * loss_spec)
                
                # 使用梯度缩放器进行反向传播
                scaler.scale(total_loss).backward()
                
                # 更新梯度并执行优化步骤
                scaler.step(optimizer)
                #scaler.update()

                scaler.update()

            else:
                # 标准精度训练流程
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
                _, H, _ = ms.shape
                block_size = H // ratio
                # 使用SDE网络将多光谱图像转换为全色图像
                with autocast() if use_amp else torch.no_grad():
                    sde_out = F_ms2pan(fusion_out_ori).squeeze(1).squeeze(0)  # 假设SDE网络输出维度为 [batch_size, 1, H, W]，因此需要squeeze掉通道维度

                    # 计算MS再SDE网络输出和pan图的L2损失
                    #print(np.shape(sde_out), np.shape(pan[0].squeeze(0)))
                loss_spa = loss_calculator.SDE_Loss(sde_out , pan.squeeze(0))
                                    

                
                # 光谱损失 (与train_or.py中的loss3对应)
                loss_spec = loss_calculator.compute_spectral_loss(
                    fusion_out_hw_c, 
                    ms.permute(1, 2, 0)
                )
                
                # 总损失 (使用train_or.py中的权重)
                total_loss = w_var * loss_var + w_spa * loss_spa + w_spec * loss_spec
                
                # 标准反向传播和优化步骤
                total_loss.backward()
                optimizer.step()
          
            # 记录各损失值
            epoch_loss_var.append(loss_var.item())
            epoch_loss_spa.append(loss_spa.item())
            epoch_loss_spec.append(loss_spec.item())
            
        # 计算平均损失
        avg_loss_var = np.mean(epoch_loss_var)
        avg_loss_spa = np.mean(epoch_loss_spa)
        avg_loss_spec = np.mean(epoch_loss_spec)
        avg_total_loss = w_var * avg_loss_var + w_spa * avg_loss_spa + w_spec * avg_loss_spec
        
        # 采用train_or.py的输出频率：每50个epoch输出一次
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
    
    # 模型标识（如果使用混合精度，则添加amp标记）
    identifier = f"{sensor}_{data_id}_FusionNet_SDE" 
    
    # 导入教师模型输出（来自FusionMamba_2024 results/WV3_full，不加载教师网络）
    teacher_output, teacher_mat_path = load_teacher_output(data_id, device)
    print(f"成功导入教师模型输出: {teacher_mat_path}")

    # 开始训练
    train(train_loader, identifier, teacher_output)

# 执行主函数
if __name__ == "__main__":
    main()
