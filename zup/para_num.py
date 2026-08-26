from mymodel import FusionNet 
from SDE import Net_ms2pan
# 创建模型实例
model = FusionNet()
model_SDE = Net_ms2pan()
# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 打印参数量
print(f"Fusion模型参数量: {count_parameters(model)}")

print(f'SDE模型参数量:{count_parameters(model_SDE)}')