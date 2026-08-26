import argparse
import subprocess
import os
import time
import sys

"this is test for mix"

def process_single_image(data_id, args):
    """处理单张图片的训练和测试"""
    image_start_time = time.time()
    
    
    # 1. 训练阶段
    if not args.skip_train:
        if args.process_model == 0:
            print("非SDE训练开始。。。")
            train_cmd = [
                sys.executable,  # 当前Python解释器
                "trainba.py",
                f"--data_id={data_id}",
                f"--lr={args.lr}",
                f"--epochs={args.epochs}",
                f"--batch_size={args.batch_size}",
                f"--device={args.device}",
                f"--sensor={args.sensor}",
                f"--ratio={args.ratio}",
                f"--temperature={args.temperature}",
                f"--alfa={args.alfa}",
                F"--data_path={args.data_path}",
                f"--use_reduced={args.use_reduced}",
                f"--reduced_data_path={args.reduced_data_path}",
                f"--reduced_every={args.reduced_every}",
                f"--reduced_loss_weight={args.reduced_loss_weight}"
            ]
            print(f"执行命令: {' '.join(train_cmd)}")
            try:
                train_result = subprocess.run(train_cmd, check=True)
                
                train_time = time.time() - image_start_time
                print(f"训练完成，耗时: {train_time:.2f}秒")
            except subprocess.CalledProcessError:
                print(f"图片 {data_id} 训练失败，跳过此图片")
                return False
        elif args.process_model == 1:
            print('第一阶段SDE网络开始准备训练')
            SDE_train_cmd = [    
                sys.executable,  # 当前Python解释器
                "main_SDE_amp.py",
                f"--lr={args.SDE_lr}",
                f"--epochs={args.SDE_epochs}",
                f"--batch_size={args.batch_size}",
                f"--device={args.device}",
                f"--satellite={args.sensor}",
                f"--name={data_id}",
                f"--data_path={args.data_path}",

            ]
            print(f"执行命令: {' '.join(SDE_train_cmd)}")
            
            # 一阶段训练：不再在reduced上预热，直接训练融合网络；仅需先训练SDE模块
            SDE_process = subprocess.Popen(SDE_train_cmd)

            # 等待 SDE 训练完成
            SDE_process.wait()

            print("SDE网络训练完成，开始直接训练融合网络")
            #print(f"执行命令: {' '.join(SDE_train_cmd)}")
            #try:
            #    SDE_train_result = subprocess.run(SDE_train_cmd, check=True)
                
            #    SDE_train_time = time.time() - image_start_time
            #    print(f"SDE网络训练完成！耗时: {SDE_train_time:.2f} 秒")
            #except subprocess.CalledProcessError:
            #    print(f"图片 {data_id} SDE网络训练失败，跳过此图片")
            #    return False
            train_cmd = [
                sys.executable,  # 当前Python解释器
                "train_SDE.py",
                f"--data_id={data_id}",
                f"--lr={args.lr}",
                f"--epochs={args.epochs}",
                f"--batch_size={args.batch_size}",
                f"--device={args.device}",
                f"--sensor={args.sensor}",
                f"--ratio={args.ratio}",
                f"--temperature={args.temperature}",
                f"--alfa={args.alfa}",
                f"--data_path={args.data_path}",
                f"--use_reduced={args.use_reduced}",
                f"--reduced_data_path={args.reduced_data_path}",
                f"--reduced_every={args.reduced_every}",
                f"--reduced_loss_weight={args.reduced_loss_weight}"
                ]
        
            print(f"执行命令: {' '.join(train_cmd)}")
            try:
                train_result = subprocess.run(train_cmd, check=True)
                
                train_time = time.time() - image_start_time
                print(f"U2Net蒸馏训练完成！耗时: {train_time:.2f} 秒")
            except subprocess.CalledProcessError:
                print(f"图片 {data_id} U2Net蒸馏训练失败，跳过此图片")
                return False
    else:
        print(f"跳过图片 {data_id} 的U2Net蒸馏训练阶段，直接进行测试...")
   


    print(f"\n{'='*20} 开始使用U2Net蒸馏训练图片 {data_id} {'='*20}")
    print(f"数据ID: {data_id}, 传感器: {args.sensor}, 训练轮数: {args.epochs}")
        
 
    # 2. 测试阶段
    print(f"\n{'='*20} 开始U2Net蒸馏测试图片 {data_id} {'='*20}")
    
    test_cmd = [
        sys.executable,  # 当前Python解释器
        "testba.py",
        f"--data_id={data_id}",
        f"--data_path={args.data_path}",
        f"--satellite={args.satellite}",
        f"--u2net_path={args.u2net_path}",
        f"--process_model={args.process_model}",
        f"--alfa={args.alfa}",
        f"--device={args.device}",
        f"--data_path={args.data_path}",
        f"--sensor_type={args.sensor}",
        f"--mode={args.mode}"
    
    ]
    
    # 如果需要显示结果，添加相应参数
    if args.show_results:
        test_cmd.append("--show_results")
    
    print(f"执行命令: {' '.join(test_cmd)}")
    try:
        test_result = subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"图片 {data_id} U2Net蒸馏测试失败")
        return False
    
    ## 显示结果文件路径
    #result_path = os.path.join("result", args.satellite, f"{data_id}_student_SDE.mat")
    
    #print(f"\n图片 {data_id} 结果文件：")
    #print(f"- U2Net蒸馏模型结果: {result_path}")
    
    image_time = time.time() - image_start_time
    print(f"图片 {data_id} 处理完成！总耗时: {image_time:.2f} 秒")
    return True

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="U2Net蒸馏训练与测试一键处理")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备 (cuda/cpu)")
    parser.add_argument('--process_model',type = int, default=1, help ='选择蒸馏模型0:trainba, 1:train_SDE')
    parser.add_argument('--SDE_lr', type=float, default=0.005, help='SDE网络学习率')
    parser.add_argument('--SDE_epochs', type=int, default=45, help='SDE网络训练轮数')#45
    parser.add_argument("--data_id", type=int, default=None,help="数据ID (0-19), 不指定则处理所有图片")
    parser.add_argument("--process_all", action="store_true", help="处理所有图片 (0-19)")
    parser.add_argument("--lr", type=float, default=0.00575, help="学习率")
    parser.add_argument("--epochs", type=int, default=240, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--sensor", type=str, default="WV3", help="传感器类型")
    parser.add_argument("--ratio", type=int, default=4, help="下采样比例")
    parser.add_argument("--temperature", type=float, default=1.0, help="蒸馏温度参数")
    parser.add_argument("--data_path", type=str, default=r"D:/DeepLearning/zspan/test_wv3_multiExm1.h5",  help="数据文件路径")
    parser.add_argument("--u2net_path", type=str, default=r"FusionMamba/weights/420.pth", help="U2Net预训练模型路径")
    parser.add_argument("--satellite", type=str, default="WV3/", help="卫星类型（结果保存用）")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练阶段")
    parser.add_argument("--show_results", action="store_true", help="显示融合结果")
    parser.add_argument("--start_id", type=int, default=0, help="起始数据ID (用于process_all)")
    parser.add_argument("--end_id", type=int, default=19, help="结束数据ID (用于process_all)")
    parser.add_argument("--alfa",type=float, default=0.15, help="U2Net蒸馏模型融合参数")
    parser.add_argument("--mode",type=str, default="normal", choices=["normal", "reduce"], help="模式")
    parser.add_argument("--use_reduced", type=int, default=1, choices=[0, 1], help="是否启用reduced数据训练(0/1)")
    parser.add_argument("--reduced_data_path", type=str, default=None, help="预构造的reduced数据h5路径(默认<data_path>_reduced.h5)")
    parser.add_argument("--reduced_every", type=int, default=10, help="每N个epoch训练一次reduced数据(默认10, 即10%)")
    parser.add_argument("--reduced_loss_weight", type=float, default=1.0, help="reduced数据gt损失权重")
    args = parser.parse_args()
    
    
    total_start_time = time.time()
    
    # 检查参数，默认处理所有图片
    if args.data_id is None:
        args.process_all = True

    if args.process_all:
        # 处理多个图片
        start_id = args.start_id
        end_id = args.end_id
        total_images = end_id - start_id + 1
        
        print(f"\n{'='*20} 开始批量处理 {'='*20}")
        print(f"将处理图片 {start_id} 到 {end_id}，共 {total_images} 张图片")
        
        success_count = 0
        
        for data_id in range(start_id, end_id + 1):
            print(f"\n{'#'*30}")
            print(f"正在处理图片 {data_id} ({data_id - start_id + 1}/{total_images})")
            print(f"{'#'*30}")
            
            if process_single_image(data_id, args):
                success_count += 1
            
            # 显示进度
            progress = (data_id - start_id + 1) / total_images * 100
            elapsed = time.time() - total_start_time
            if data_id > start_id:  # 避免除零错误
                estimated_total = elapsed / (data_id - start_id + 1) * total_images
                remaining = estimated_total - elapsed
                print(f"\n进度: {progress:.1f}% | 已完成: {data_id - start_id + 1}/{total_images}")
                print(f"已用时间: {elapsed:.2f}秒 | 估计剩余: {remaining:.2f}秒")
        
        # 最终总结
        total_time = time.time() - total_start_time
        print(f"\n{'='*20} 批量处理完成 {'='*20}")
        print(f"成功处理: {success_count}/{total_images} 张图片")
        print(f"总耗时: {total_time:.2f} 秒 (平均每张 {total_time/total_images:.2f} 秒)")
    
    else:
        # 处理单个图片
        data_id = args.data_id
        process_single_image(data_id, args)
        
        # 总结
        total_time = time.time() - total_start_time
        print(f"\n{'='*20} 执行完成 {'='*20}")
        print(f"数据ID: {data_id} 的U2Net蒸馏训练与测试已完成")
        print(f"总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()