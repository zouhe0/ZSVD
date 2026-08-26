import torch
from torch import nn
import torch.nn.functional as F

class Net_ms2pan(nn.Module):
    def __init__(self):
        super(Net_ms2pan, self).__init__()
        self.net = nn.Sequential(nn.Linear(8, 32),
                                 nn.Sigmoid(),
                                 nn.Linear(32, 32),
                                 nn.Sigmoid(),
                                 nn.Linear(32, 1),)
        self.linear = nn.Linear(9, 1)
        self.sigmod = nn.Sigmoid()
    def forward(self, ms):
        out = self.net(ms.permute(0, 2, 3, 1))
        out = torch.cat((out,ms.permute(0, 2, 3, 1)),dim=3)
        out = self.linear(out)
        out = self.sigmod(out)
        return out.permute(0, 3, 1, 2)
    


def sobel_filter(image):
    """
    对输入多通道图像应用 Sobel 过滤器，分别计算水平和垂直方向的梯度。
    image: Tensor of shape (C, H, W)
    返回: (Gx, Gy) 形状均为 (C, H, W)
    """
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    if image.dim() == 3:
        image = image.unsqueeze(0)  # 添加 batch 维度
    
    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)
    
    grads_x = []
    grads_y = []

    for c in range(image.shape[1]):  # 逐通道应用 Sobel 过滤器
        grad_x = F.conv2d(image[:, c:c+1, :, :], sobel_x, padding=1)
        grad_y = F.conv2d(image[:, c:c+1, :, :], sobel_y, padding=1)
        grads_x.append(grad_x)
        grads_y.append(grad_y)
    
    grads_x = torch.cat(grads_x, dim=1)
    grads_y = torch.cat(grads_y, dim=1)
    
   # sobel_output = torch.cat([grads_x, grads_y], dim=1)  # 组合两个方向的梯度，通道数翻倍
    return grads_x, grads_y


# ============== Modify Net_ms2pan for Dual Branch ============== #
class Net_ms2pan_dual(torch.nn.Module):
    def __init__(self):
        super(Net_ms2pan_dual, self).__init__()
        # 假设原始的 Net_ms2pan 包含一个前向方法 forward(self, x)
        # 这里我们创建两个相同的分支
        self.branch_x = Net_ms2pan()
        self.branch_y = Net_ms2pan()

    def forward(self, ms_gra_x, ms_gra_y):
        out_x = self.branch_x(ms_gra_x)
        out_y = self.branch_y(ms_gra_y)
        return out_x, out_y

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

# -----------------------------------------------------
class ms2pan_convNet(nn.Module):
    def __init__(self):
        super(ms2pan_convNet, self).__init__()
        
        spectral_num = 8
        hidden_channels = 11  # 略微增加通道数
        
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=hidden_channels, 
                              kernel_size=3, padding=1, bias=True)
        
        # 深度可分离卷积降低参数量但保持特征提取能力
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                                  padding=1, groups=hidden_channels, bias=False)
        self.pointwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=1, 
                              kernel_size=3, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化权重
        init_weights(self.conv1, self.depthwise, self.pointwise, self.conv3)
        
    def forward(self, x):        
        rs = self.relu(self.conv1(x))
        res = rs
        rs = self.relu(self.depthwise(rs))
        rs = self.relu(self.pointwise(rs))
        rs = rs + res  # 残差连接
        output = self.conv3(rs)
        
        return output
class ms2pan_convNet_dual(nn.Module):
    def __init__(self):
        super(ms2pan_convNet_dual, self).__init__()
        
        self.branch_x = ms2pan_convNet()
        self.branch_y = ms2pan_convNet()

    def forward(self, x, y):
        out_x = self.branch_x(x)
        out_y = self.branch_y(y)
        return out_x, out_y




# ----------------- End-Main-Part ------------------------------------
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor
