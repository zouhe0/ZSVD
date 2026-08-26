import argparse # type: ignore
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
# 导入混合精度训练所需的库
from torch.cuda.amp import autocast, GradScaler

from data import Dataset  # 数据加载器
from mymodel import FusionNet  # 学生模型
from SDE import Net_ms2pan_dual,sobel_filter,ms2pan_convNet_dual  # SDE模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FusionMamba'))
print("Updated sys.path:", sys.path)
# 修改导入：使用U2Net作为教师模型
from model.u2net import U2Net
from loss import LossCalculator  # 损失计算器

# ================== 基础设置 =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ============= 超参数设置 ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.00575, help="学习率")
parser.add_argument("--epochs", type=int, default=240, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--device", type=str, default='cuda:2', help="训练设备")
parser.add_argument("--data_id", type=int, default=2, help="数据ID (0-19)")
parser.add_argument("--sensor", type=str, default='wv3', help="传感器类型")
parser.add_argument("--ratio", type=int, default=4, help="下采样比例")
parser.add_argument("--temperature", type=float, default=1.0, help="蒸馏温度参数")
# 添加混合精度训练的参数选项
parser.add_argument("--amp",default=True, action="store_true", help="启用混合精度训练")
parser.add_argument("--alfa", type=float, default=0.15, help="损失权重")
parser.add_argument("--data_path", type=str, default=r"/HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5", help="数据文件路径")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print("train sde decive: ", device)
data_id = args.data_id
sensor = args.sensor.upper()
ratio = args.ratio
# 混合精度训练标志
use_amp = args.amp and (device.type == 'cuda')  # 只在CUDA设备上启用混合精度


ri = 140*1
alfa = args.alfa 
# 损失权重设置（与train_or.py保持一致）
w_var = 1850000.0   # 变分损失权重（对应train_or.py中的w_var）
w_spa = alfa*ri      # 空间保真度损失权重
w_spec = (1-alfa)*ri    # 光谱损失权重

# 数据文件路径
data_path = args.data_path


# U2Net预训练模型路径
u2net_path = r"/HardDisk/HeZou/zup/FusionMamba/weights/420.pth"

# =================== 模型初始化 =================== #
# 学生模型 (FusionNet)
model_student = FusionNet().to(device)
model_student.load_state_dict(torch.load(f'model_pretrain/{sensor}/{data_id}_FusionNet_pretrain.pth'))
print("学生模型初始化完成")

# 损失计算器初始化
loss_calculator = LossCalculator(sensor=sensor, ratio=ratio, N=41, device=device)
F_ms2pan = ms2pan_convNet_dual().to(device)
F_ms2pan.load_state_dict(torch.load(f'model_SDE/{sensor}/{data_id}_ms2pan_convNet_dual.pth'))
F_ms2pan.eval()


# 优化器
optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))


# 教师模型 (U2Net)
try:
    # 设定U2Net输入参数
    model_teacher = U2Net(
        dim=32,      # 特征维度
        pan_dim=1,   # 全色图像通道数
        ms_dim=8,    # 多光谱图像通道数
        H=512,       # 与pan图像尺寸一致
        W=512        # 与pan图像尺寸一致
    ).to(device)
    
    checkpoint = torch.load(u2net_path, map_location=device)
    if 'state_dict' in checkpoint:
        model_teacher.load_state_dict(checkpoint['state_dict'])
    else:
        model_teacher.load_state_dict(checkpoint)
        
    model_teacher.eval()  # 固定教师模型参数
    print(f"成功加载U2Net预训练模型: {u2net_path}")
except Exception as e:
    print(f"加载U2Net模型失败: {str(e)}")
    print("请确保模型结构和权重文件正确")
    exit(1)



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

# 在训练前预先计算所有批次的教师模型输出
def precompute_teacher_outputs(data_loader, teacher_model):
    print("预计算教师模型输出...")
    teacher_outputs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            ms, lms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            # 确保pan维度正确
            if len(pan.shape) == 3:
                pan = pan.unsqueeze(1)
                
            try:
                # 对教师模型预计算也使用混合精度（如果启用）
                if use_amp:
                    with autocast():
                        # 注意：U2Net的输入顺序是(ms, pan)，与LACNET不同
                        output = teacher_model(ms, pan)
                else:
                    output = teacher_model(ms, pan)
                
                # 如果输出为元组，取第一个元素
                if isinstance(output, tuple):
                    output = output[0]
                    
                teacher_outputs.append(output.cpu())  # 存储到CPU内存以节省GPU内存
            except Exception as e:
                print(f"批次 {batch_idx} 预计算失败: {str(e)}")
                # 如果处理失败，添加None，稍后处理
                teacher_outputs.append(None)
    
    valid_count = sum(1 for out in teacher_outputs if out is not None)
    print(f"预计算完成，共 {valid_count}/{len(teacher_outputs)} 个有效样本")
    return teacher_outputs

# ================ 知识蒸馏训练过程 ================ #
def train(training_data_loader, identifier):
    print("开始知识蒸馏训练...")
    
    # 预计算教师模型输出
    teacher_outputs = precompute_teacher_outputs(training_data_loader, model_teacher)
    
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
                    
                    # 变分损失 (与train_or.py中的loss1对应)
                    loss_var = torch.mean((fusion_out - fusion_out_teacher) ** 2)
                    
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
                    total_loss = w_var * loss_var + weight*(w_spa * loss_spa + w_spec * loss_spec)
                
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
                #loss_spec = loss_calculator.compute_spectral_loss(
                #    fusion_out_hw_c, 
                #    ms.permute(1, 2, 0)
                #)
                loss_spec =  torch.mean((fusion_out - fusion_out_teacher) ** 2)
                # 总损失 (使用train_or.py中的权重)
                total_loss = w_var * loss_var + w_spa * loss_spa + w_spec * loss_spec
                
                # 标准反向传播和优化步骤
                total_loss.backward()
                optimizer.step()
            
            

    
    # 保存最终模型  
            # 记录各损失值
            epoch_loss_var.append(loss_var.item())
            epoch_loss_spa.append(loss_spa.item())
            epoch_loss_spec.append(loss_spec.item())


    
            
        # 计算平均损失
        avg_loss_var = np.mean(epoch_loss_var)
        avg_loss_spa = np.mean(epoch_loss_spa)
        avg_loss_spec = np.mean(epoch_loss_spec)
        avg_total_loss = w_var * avg_loss_var + w_spa * avg_loss_spa + w_spec * avg_loss_spec

        if epoch == 1:
            print(f"teachers_loss: {loss_spec_teacher}")
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
    
    # 开始训练
    train(train_loader, identifier)

# 执行主函数
if __name__ == "__main__":
    main()
