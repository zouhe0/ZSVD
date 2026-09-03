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
from torch.autograd import Variable
from data import Dataset  # 数据加载器
from mymodel import FusionNet
from wald_utilities import wald_protocol_v1, wald_protocol_v2  # 学生模型

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FusionMamba'))
print("Updated sys.path:", sys.path)

from loss import pretrain_Losses

# ================== 基础设置 =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ============= 超参数设置 ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.002, help="学习率")
parser.add_argument("--epochs", type=int, default=8, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--device", type=str, default='cuda:2', help="训练设备")
parser.add_argument("--data_id", type=int, default=0, help="数据ID (0-19)")
parser.add_argument("--sensor", type=str, default=None, help="传感器类型")
parser.add_argument("--ratio", type=int, default=4, help="下采样比例")
parser.add_argument("--temperature", type=float, default=1.0, help="蒸馏温度参数")
# 添加混合精度训练的参数选项
parser.add_argument("--amp",default=True, action="store_true", help="启用混合精度训练")
parser.add_argument("--data_path",type=str, default="HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5",help="数据文件路径")
parser.add_argument("--output_path", type=str, default=None, help="预训练权重输出路径")

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



# 数据文件路径
data_path = args.data_path

# =================== 模型初始化 =================== #
# 学生模型 (FusionNet)
model_student = FusionNet().to(device)
print("学生模型初始化完成")



# 优化器
optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))

criterion = pretrain_Losses(device)
# 初始化混合精度训练的GradScaler
scaler = torch.amp.GradScaler("cuda") if use_amp else None

if use_amp:
    print("已启用混合精度训练 (AMP)")
elif args.amp and device.type != 'cuda':
    print("警告: 混合精度训练仅支持CUDA设备,已自动禁用")

# 模型保存函数
def save_checkpoint(model, name):
    model_out_path = args.output_path or os.path.join(
        "model_pretrain", sensor, str(name) + "_FusionNet_pretrain.pth"
    )
    os.makedirs(os.path.dirname(os.path.abspath(model_out_path)), exist_ok=True)
    torch.save(model.state_dict(), model_out_path)

# ================ 知识蒸馏训练过程 ================ #
def train(training_data_loader, name):
    print("开始初阶段预训练训练...")
    

    min_total_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model_student.train()
        epoch_loss = []
        
        for i, batch in enumerate(training_data_loader):

                
            ms, lms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            lms_rr = wald_protocol_v1(lms, pan, 4, 'WV3')
            pan_rr = wald_protocol_v2(ms, pan, 4, 'WV3')
            # lms_rr = torch.rand(1,8,128,128).half().to(device)
            # pan_rr = torch.rand(1,1,128,128).half().to(device)
            # 确保pan维度正确
            if len(pan_rr.shape) == 3:
                pan_rr = pan_rr.unsqueeze(1)
            
            # 使用混合精度进行前向传播和损失计算
            if use_amp:
                with torch.amp.autocast("cuda"):
                    # 学生模型前向传播
                    res_student = model_student(lms_rr, pan_rr)
                    fusion_out = res_student + lms_rr

                    # 计算损失
                    loss  =  criterion(fusion_out, ms)
 
                # 使用梯度缩放器进行反向传播
                scaler.scale(loss).backward()
                
                # 更新梯度并执行优化步骤
                scaler.step(optimizer)
                scaler.update()
            else:
                res_student = model_student(lms_rr, pan_rr)
                fusion_out = res_student + lms_rr

                    # 计算损失
                loss  =  criterion(fusion_out, ms)
          # 标准反向传播和优化步骤
                loss.backward()
                optimizer.step()
            
            # 记录各损失值
            epoch_loss.append(loss.item())

        # 计算平均损失
        avg_loss = np.mean(epoch_loss)
        # 采用train_or.py的输出频率：每50个epoch输出一次
        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss1(avg_loss): {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < min_total_loss:
            min_total_loss = avg_loss
            save_checkpoint(model_student, f"{name}")


    # 训练完成
    total_time = time.time() - start_time
    print(f"训练完成，总耗时 {total_time:.2f} 秒")
    print(f"最终最佳损失: {min_total_loss:.6f}")


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
    name = data_id
    
    # 开始训练
    train(train_loader, name)

# 执行主函数
if __name__ == "__main__":
    main()
