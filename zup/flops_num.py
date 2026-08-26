import torch
from thop import profile
from SDE import ms2pan_convNet_dual
from mymodel import FusionNet

# 定义输入张量
input_ms0 = torch.randn(1, 8, 64, 64)  # 多光谱图像
input_ms = torch.randn(1, 8, 256, 256)  # 多光谱图像
input_pan = torch.randn(1, 1, 256, 256)  # 全色图像

# 创建模型实例
model1 = ms2pan_convNet_dual()
model2 = FusionNet()

# 计算 ms2pan_convNet_dual 的 FLOPS
flops1, params1 = profile(model1, inputs=(input_ms0, input_ms0), verbose=False)
print(f"ms2pan_convNet_dual FLOPS: {flops1:.2f}, Parameters: {params1:.2f}")

# 计算 FusionNet_new 的 FLOPS
flops2, params2 = profile(model2, inputs=(input_ms, input_pan), verbose=False)
print(f"FusionNet_new FLOPS: {flops2:.2f}, Parameters: {params2:.2f}")