import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse # type: ignore
import scipy.io as sio
import sys
import time
from mymodel import FusionNet  # 学生模型
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FusionMamba'))


def tensor_to_image(tensor):
    """
    将 [C,H,W] 的张量转换为 [H,W,C] 的 numpy 数组用于显示
    """
    tensor = tensor.cpu().numpy()
    if tensor.shape[0] >= 3:
        img = tensor[:3, :, :]
    else:
        img = np.repeat(tensor, 3, axis=0)
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    return img



def main():
    parser = argparse.ArgumentParser(description="Test FusionNet model trained with U2Net as teacher")
    parser.add_argument("--process_model",type =int, default=1, help="模型选择,trainba:0,SDE_train:1")
    parser.add_argument("--data_id", type=int, default=0 , help="测试样本的索引（0 ~ N-1）")
    parser.add_argument("--data_path", type=str, default=r"/HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5", help="h5 数据文件路径")
    parser.add_argument("--u2net_path", type=str, default=r"/HardDisk/HeZou/zup/FusionMamba/weights/420.pth", help="U2Net模型路径")
    parser.add_argument("--show_results", action="store_true", help="显示融合结果")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "reduce"], help="输出模式,reduce为匹配Matlab评测格式")
    parser.add_argument("--gt_path", type=str, default=None, help="测试集地面真实值的路径(仅reduce模式需要)")
    parser.add_argument("--sensor_type", type=str, default="WV3", help="卫星传感器类型,影响波段选择")
    parser.add_argument("--satellite",type=str, default="WV3/", help="卫星名称")
    parser.add_argument("--alfa", type=float, default=0.15, help="融合权重")
    parser.add_argument("--device", type=str, default="cuda:2", help="运行设备")
    args = parser.parse_args()
    
    device = args.device
    
    # --------------------- 加载数据 ---------------------
    print("正在加载数据...")
    with h5py.File(args.data_path, 'r') as data:
        lms_np = np.array(data['lms'][args.data_id], dtype=np.float32) / 2047.0  # [C,H,W]
        pan_np = np.array(data['pan'][args.data_id], dtype=np.float32) / 2047.0   # [1,H,W] 或 [H,W]
        ms_np = np.array(data['ms'][args.data_id], dtype=np.float32) / 2047.0     # [C,H/r,W/r]
    
    # 转换为 torch.Tensor,并放置于指定设备
    lms = torch.from_numpy(lms_np).to(device)  # [C,H,W]
    pan = torch.from_numpy(pan_np).to(device)  # [1,H,W] 或 [H,W]
    ms = torch.from_numpy(ms_np).to(device)    # [C,H/r,W/r]
    
    # 确保pan有正确的维度 [1,H,W]
    if pan.dim() == 2:
        pan = pan.unsqueeze(0)  # 转换为 [1,H,W]
    
    # 添加 batch 维度
    lms_batch = lms.unsqueeze(0)  # [1,C,H,W]
    pan_batch = pan.unsqueeze(0)  # [1,1,H,W]
    ms_batch = ms.unsqueeze(0)    # [1,C,H/r,W/r]
    
    # 获取实际图像尺寸
    input_height, input_width = pan.shape[1], pan.shape[2]
    print(f"输入图像尺寸: {input_height}x{input_width}")
    
    # --------------------- 加载学生模型 ---------------------
    print("正在加载学生模型...")
    model_student = FusionNet().to(device)
    if args.process_model == 0:
        best_checkpoint = os.path.join("model_FUG", args.sensor_type.upper()+f"_{args.data_id}_FusionNet_best.pth")
    else:
        best_checkpoint = os.path.join("model_FUG", args.sensor_type.upper()+f"_{args.data_id}_FusionNet_SDE_best.pth")
    if os.path.exists(best_checkpoint):
        model_student.load_state_dict(torch.load(best_checkpoint, map_location=device))
        print(f"成功加载学生模型: {best_checkpoint}")
    else:
        print(f"未找到学生模型权重文件: {best_checkpoint}")
        return
    
    model_student.eval()
    
    # --------------------- 加载教师模型 ---------------------
    print("正在加载教师模型...")
    try:
        # 初始化U2Net时使用实际图像尺寸
        model_teacher = U2Net(
            dim=32,      # 特征维度
            pan_dim=1,   # 全色图像通道数
            ms_dim=8,    # 多光谱图像通道数
            H=512,  # 使用实际图像高度
            W=512    # 使用实际图像宽度
        ).to(device)
        
        checkpoint = torch.load(args.u2net_path, map_location=device)
        if 'state_dict' in checkpoint:
            model_teacher.load_state_dict(checkpoint['state_dict'])
        else:
            model_teacher.load_state_dict(checkpoint)
        
        model_teacher.eval()
        print(f"成功加载教师模型: {args.u2net_path}")
    except Exception as e:
        print(f"加载教师模型失败: {str(e)}")
        print("继续使用学生模型进行测试...")
        model_teacher = None
    
    # --------------------- 学生模型推理 ---------------------
    print("正在执行学生模型推理...")
    with torch.no_grad():
        # 学生模型推理
        start_time = time.time()
        res_student = model_student(lms_batch, pan_batch)
        fused_student = res_student + lms_batch  # 融合结果（残差模型）
        end_time = time.time()
        print(f"学生模型推理用时: {end_time - start_time:.3f} s")
        fused_student = fused_student.squeeze(0)  # [C,H,W]
        
        # 教师模型推理（如果可用）
        if model_teacher is not None:
            print("正在执行教师模型推理（保持原始分辨率）...")
            # 直接使用原始尺寸进行推理,无需下采样和上采样
            teacher_out = model_teacher(ms_batch, pan_batch)
            if isinstance(teacher_out, tuple):
                teacher_out = teacher_out[0]
                
            fused_teacher = teacher_out.squeeze(0)  # [C,H,W]
    
    # --------------------- 转换为图片格式 ---------------------
    lms_img = tensor_to_image(lms)  # 原始上采样多光谱图像
    fused_student_img = tensor_to_image(fused_student)  # 学生模型融合结果
    
    # 处理全色图像（转换为 [H,W]）
    pan_img = pan_batch.squeeze(0).cpu().numpy()
    if pan_img.ndim == 3:
        pan_img = pan_img.squeeze(0)
    pan_img = np.clip(pan_img, 0, 1)
    
    # --------------------- 显示结果（可选） ---------------------
    if args.show_results:
        plt.figure(figsize=(16, 4))
        
        plt.subplot(1, 4, 1)
        plt.imshow(lms_img)
        plt.title("LMS (Before Fusion)")
        plt.axis("off")
        
        plt.subplot(1, 4, 2)
        plt.imshow(pan_img, cmap='gray')
        plt.title("PAN")
        plt.axis("off")
        
        plt.subplot(1, 4, 3)
        plt.imshow(fused_student_img)
        plt.title("Student Model Fusion")
        plt.axis("off")
        
        if model_teacher is not None:
            fused_teacher_img = tensor_to_image(fused_teacher)
            plt.subplot(1, 4, 4)
            plt.imshow(fused_teacher_img)
            plt.title("Teacher Model Fusion")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    # --------------------- 保存结果到 .mat 文件 ---------------------
    print("正在保存结果...")
    # 将归一化的数据乘以2047恢复到原始范围
    I_student = torch.squeeze(fused_student).permute(1, 2, 0).cpu().detach().numpy() * 2047  # 学生模型融合结果
    I_MS_LR = torch.squeeze(torch.from_numpy(ms_np)).permute(1, 2, 0).cpu().detach().numpy() * 2047  # 原始低分辨率MS图像
    I_MS = torch.squeeze(lms).permute(1, 2, 0).cpu().detach().numpy() * 2047  # 原始上采样MS图像
    I_PAN = torch.squeeze(pan).cpu().detach().numpy() * 2047  # 全色图像

    # 根据模式选择保存方式
    if args.mode == "normal":
        # 保存学生模型结果
        student_dict = {
            'I_MS_LR': I_MS_LR,
            'I_MS': I_MS,
            'I_PAN': I_PAN,
            'proposed': I_student  # 使用'proposed'作为学生模型输出的键名
        }
        os.makedirs(os.path.join("result", args.satellite), exist_ok=True)
        if args.process_model == 0:
            student_save_path = os.path.join("result", args.satellite, f"{args.data_id}_student_{args.alfa}.mat")
        elif args.process_model == 1:
            student_save_path = os.path.join("result", args.satellite,f"{args.data_id}_student_SDE_{args.alfa}.mat")
        sio.savemat(student_save_path, student_dict)
        print(f"学生模型结果已保存至: {student_save_path}")

        # 如果教师模型可用,也保存其结果
        if model_teacher is not None:
            I_teacher = torch.squeeze(fused_teacher).permute(1, 2, 0).cpu().detach().numpy() * 2047  # 教师模型融合结果
            teacher_dict = {
                'I_MS_LR': I_MS_LR,
                'I_MS': I_MS,
                'I_PAN': I_PAN,
                'proposed': I_teacher  # 使用'proposed'作为教师模型输出的键名
            }
            teacher_save_path = os.path.join("result", args.satellite,f"{args.data_id}_teacher.mat")
            sio.savemat(teacher_save_path, teacher_dict)
            print(f"教师模型结果已保存至: {teacher_save_path}")

    elif args.mode == "reduce":
        # Matlab评测兼容模式
        print("使用reduce模式,准备Matlab评测所需数据...")
        
        # 首先尝试从原始h5文件中加载ground truth数据
        try:
            with h5py.File(args.data_path, 'r') as data:
                if 'gt' in data:
                    print("在h5文件中找到gt数据")
                    # 注意：GT数据不需要乘以2047,因为它已经是原始尺度
                    gt_data = np.array(data['gt'][args.data_id], dtype=np.float32)
                    if gt_data.shape[0] < 10:  # 如果第一维是通道维度 [C,H,W]
                        gt_data = np.transpose(gt_data, (1, 2, 0))
                    print(f"GT数据形状: {gt_data.shape}")
                elif 'GT' in data:
                    print("在h5文件中找到GT数据")
                    # 注意：GT数据不需要乘以2047,因为它已经是原始尺度
                    gt_data = np.array(data['GT'][args.data_id], dtype=np.float32)
                    if gt_data.shape[0] < 10:  # 如果第一维是通道维度 [C,H,W]
                        gt_data = np.transpose(gt_data, (1, 2, 0))
                    print(f"GT数据形状: {gt_data.shape}")
                else:
                    print("h5文件中未找到GT数据,查找外部GT文件...")
                    # GT数据加载失败时,使用LMS作为替代
                    gt_data = I_MS
        except Exception as e:
            print(f"从h5文件加载GT数据失败: {str(e)}")
            print("使用LMS作为GT的替代")
            gt_data = I_MS
        
        # 分别保存学生模型和教师模型的评测数据
        # 学生模型评测数据
        student_reduce_dict = {
            'gt': gt_data,              # 地面真实值
            'proposed': I_student,       # 学生模型融合结果
            'ms': I_MS_LR,              # 低分辨率多光谱图像
            'pan': I_PAN,               # 全色图像
            'ratio': 4,                 # 下采样比例
            'sensor_type': args.sensor_type  # 传感器类型
        }
        
        os.makedirs(os.path.join("reduce_result", args.satellite), exist_ok=True)
        student_reduce_path = os.path.join("reduce_result", args.satellite, f"{args.data_id}_student.mat")
        sio.savemat(student_reduce_path, student_reduce_dict)
        print(f"学生模型评测数据已保存至: {student_reduce_path}")
        
        # 如果教师模型可用,也保存其评测数据
        if model_teacher is not None:
            I_teacher = torch.squeeze(fused_teacher).permute(1, 2, 0).cpu().detach().numpy() * 2047
            teacher_reduce_dict = {
                'gt': gt_data,              # 地面真实值
                'proposed': I_teacher,      # 教师模型融合结果
                'ms': I_MS_LR,              # 低分辨率多光谱图像
                'pan': I_PAN,               # 全色图像
                'ratio': 4,                 # 下采样比例
                'sensor_type': args.sensor_type  # 传感器类型
            }
            
            teacher_reduce_path = os.path.join("reduce_result", args.satellite, f"{args.data_id}_teacher.mat")
            sio.savemat(teacher_reduce_path, teacher_reduce_dict)
            print(f"教师模型评测数据已保存至: {teacher_reduce_path}")

    print("处理完成！")

if __name__ == "__main__":
    main()