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
class FusionNet_new(nn.Module):
    def __init__(self):
        super(FusionNet_new, self).__init__()
        
        spectral_num = 9
        hidden_channels = 9  # 略微增加通道数
        
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=12, 
                              kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, 
                              kernel_size=3, padding=1, bias=True)
        self.conv_ms = nn.Conv2d(in_channels=8, out_channels=8, 
                              kernel_size=3, padding=1, bias=True)
        self.conv_pan = nn.Conv2d(in_channels=1, out_channels=4, 
                              kernel_size=3, padding=1, bias=True)
        self.fc_ms = nn.Linear(8*32*32, 64)
        self.fc_pan = nn.Linear(4*32*32, 64)
        self.pool = nn.AdaptiveAvgPool2d((32, 32))  # 统一到4x4

        # 深度可分离卷积降低参数量但保持特征提取能力
        self.depthwise = nn.Conv2d(12, 12, kernel_size=3, 
                                  padding=1, groups=12, bias=False)
        self.pointwise = nn.Conv2d(12, 12, kernel_size=1, bias=True)
        
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=8, 
                              kernel_size=3, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化权重
        init_weights(self.conv1, self.depthwise, self.pointwise, self.conv3)
        
    def forward(self, x, y):
        pan_concat = torch.cat([y, y, y, y, y, y, y, y], 1)
        input2 = torch.sub(pan_concat, x)
        
        feature_ms = self.relu(self.conv_ms(x))  # 处理多光谱图像
        feature_pan = self.relu(self.conv_pan(y))  # 处理全色图像
        input1 = torch.cat((feature_ms, feature_pan), dim=1)  # 合并特征图
        feature_ms = self.pool(feature_ms)  # 池化处理
        feature_pan = self.pool(feature_pan)  # 池化处理
        feature_ms = feature_ms.view(feature_ms.size(0), -1)  # 展平
        feature_pan = feature_pan.view(feature_pan.size(0), -1)  # 展平
        # 全连接层处理特征
        feature_ms_final = self.fc_ms(feature_ms)
        feature_pan_final = self.fc_pan(feature_pan)

        rs = self.relu(self.conv1(input1))
        res = self.relu(self.conv2(input2))
        rs = self.relu(self.depthwise(rs))
        rs = self.relu(self.pointwise(rs))
        rs = rs + res  # 残差连接
        output = self.conv3(rs)
        
        return output, feature_ms_final, feature_pan_final

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


def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 64, 64), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model,(x,))


'-----------------------------------------------------'
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        
        spectral_num = 8
        hidden_channels = 11  # 略微增加通道数
        
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=hidden_channels, 
                              kernel_size=3, padding=1, bias=True)
        
        # 深度可分离卷积降低参数量但保持特征提取能力
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                                  padding=1, groups=hidden_channels, bias=False)
        self.pointwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=spectral_num, 
                              kernel_size=3, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化权重
        init_weights(self.conv1, self.depthwise, self.pointwise, self.conv3)
        
    def forward(self, x, y):
        pan_concat = torch.cat([y, y, y, y, y, y, y, y], 1)
        input = torch.sub(pan_concat, x)
        
        rs = self.relu(self.conv1(input))
        res = rs
        rs = self.relu(self.depthwise(rs))
        rs = self.relu(self.pointwise(rs))
        rs = rs + res  # 残差连接
        output = self.conv3(rs)
        
        return output