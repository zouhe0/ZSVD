import torch
import torch.nn.functional as F
import numpy as np
from wald_utilities import genMTF, MTF_PAN, fspecial_gauss
import math
import scipy.ndimage as ndimage
from scipy import signal
import kornia.filters as kf
import torch.nn as nn
class LossCalculator:
    def __init__(self, sensor, ratio, N=41, device='cpu'):
        """
        参数:
            sensor: 传感器类型（例如 'WV3', 'QB', 'WV2' 等），不区分大小写
            ratio: 下采样因子
            N: MTF 核尺寸，默认 41
            device: 设备 'cpu' 或 'cuda'
        """
        self.sensor = sensor.upper()  
        self.ratio = ratio
        self.N = N
        self.device = device
        # 预计算MTF核，避免重复计算
        mtf_kernel_np = genMTF(self.ratio, self.sensor, self.N)
        self.mtf_kernel = torch.from_numpy(mtf_kernel_np).float().to(device)
        
        # 初始化Sobel边缘检测器
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    device=device, dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # 初始化多尺度高斯滤波器
        self.gaussian_kernels = []
        for sigma in [0.5, 1.0, 2.0]:
            self.gaussian_kernels.append(self._create_gaussian_kernel(5, sigma).to(device))
        
        # 初始化拉普拉斯滤波器 - 用于边缘检测的另一种方法
        self.laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                           device=device, dtype=torch.float32).view(1, 1, 3, 3)

    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        """创建高斯卷积核"""
        # 创建网格
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= (kernel_size - 1) / 2
        
        # 创建X和Y坐标
        x = coords.repeat(kernel_size, 1)
        y = x.t()
        
        # 计算高斯核
        kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        kernel /= kernel.sum()  # 归一化
        
        return kernel.view(1, 1, kernel_size, kernel_size)
        
    @staticmethod
    def apply_convolution(image, kernel):
        """
        利用 conv2d 对图像进行二维卷积操作（各波段单独卷积）
        
        参数:
            image: torch.Tensor, 形状 (H, W, bands)
            kernel: torch.Tensor, 形状 (k, k, bands)
        返回:
            filtered: torch.Tensor, 形状 (H, W, bands)
        """
        image = image.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        bands = image.shape[1]
        kH, kW = kernel.shape[0], kernel.shape[1]
        if kernel.shape[2] < bands:
            extra = kernel[:, :, -1:].repeat(1, 1, bands - kernel.shape[2])
            kernel = torch.cat([kernel, extra], dim=2)
        kernel = kernel.permute(2, 0, 1).unsqueeze(1)  # (bands, 1, kH, kW)
        filtered = F.conv2d(image, kernel, padding=kH // 2, groups=bands)
        filtered = filtered.squeeze(0).permute(1, 2, 0)
        return filtered

    def compute_spectral_loss(self, X, Y):
        """
        计算光谱损失 fspec - 使用完整的Wald协议:
        fspec(X,Y) = || wald_downsample(X) - Y ||^2_F

        参数:
        X: 重建的高分辨率多光谱图像, torch.Tensor, 形状 (H, W, S)
        Y: 低分辨率多光谱图像, torch.Tensor, 形状 (H//ratio, W//ratio, S)
        
        返回:
            光谱损失（标量 tensor）
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # 准备输入格式 (B,C,H,W)
        X_t = X.permute(2, 0, 1).unsqueeze(0)  # (1, S, H, W)
        
        # 应用MTF滤波
        mtf_kernel = self.mtf_kernel  # 已经预先计算好的传感器特定MTF核
        
        # 将MTF核转换为适合卷积的格式
        MTF_kern = mtf_kernel.permute(2, 0, 1).unsqueeze(1)  # (bands, 1, k, k)
        
        # 应用深度可分离卷积（逐通道卷积）
        bands = X_t.shape[1]
        depthconv = nn.Conv2d(in_channels=bands, 
                            out_channels=bands,
                            kernel_size=MTF_kern.shape[2:],
                            groups=bands, 
                            padding=mtf_kernel.shape[0]//2,
                            padding_mode='replicate',
                            bias=False).to(self.device)
        
        depthconv.weight.data = MTF_kern
        depthconv.weight.requires_grad = False
        
        # 应用MTF滤波
        X_blurred = depthconv(X_t)
        
        # 使用bicubic下采样
        X_down_bicubic = F.interpolate(X_blurred, scale_factor=1/self.ratio, mode='bicubic', align_corners=False)
        
        # 或者，使用23抽头插值（如果需要更高精度）
        # 这需要一个自定义函数或使用wald_utilities中的函数
        # X_down = custom_23tap_downsample(X_blurred, self.ratio)
        
        # 转回原始格式 (H,W,C)
        X_down = X_down_bicubic.squeeze(0).permute(1, 2, 0)
        
        # 计算Frobenius范数损失
        loss = torch.norm(X_down - Y, p='fro') ** 2
        
        return loss
    
    def extract_multiscale_gradients(self, image):
        """
        提取多尺度梯度特征
        
        参数:
            image: torch.Tensor, 形状 (H, W) 或 (H, W, C) 或 [1, 1, H, W]
        返回:
            gradients_list: 包含多尺度梯度的列表
        """
        # 处理输入维度
        if image.dim() > 4:
            # 如果维度过多，打平到正确的维度
            image = image.view(1, 1, image.shape[-2], image.shape[-1])
        elif image.dim() == 3:
            # 将多通道图像转为灰度，使用更精确的权重
            if image.shape[2] > 1:
                # 使用ITU-R BT.601标准的权重
                weights = torch.tensor([0.299, 0.587, 0.114], device=self.device)
                # 确保权重与通道数匹配
                if image.shape[2] > 3:
                    # 额外通道使用均匀权重
                    extra_weights = torch.ones(image.shape[2] - 3, device=self.device) / (image.shape[2] - 3)
                    weights = torch.cat([weights, extra_weights])
                
                # 加权平均生成灰度图
                image = torch.sum(image * weights.view(1, 1, -1), dim=2)
            else:
                image = image.squeeze(2)
            
            # 转换为适合卷积的格式
            image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif image.dim() == 2:
            # 如果是2D图像，直接添加批次和通道维度
            image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 确保图像形状正确
        assert image.dim() == 4 and image.shape[1] == 1, f"图像形状错误，期望[1,1,H,W]，实际为{image.shape}"
        
        # 多尺度梯度特征列表
        gradients_list = []
        
        # 在不同尺度上提取梯度
        for gaussian_kernel in self.gaussian_kernels:
            # 应用高斯平滑
            smoothed = F.conv2d(image, gaussian_kernel, padding=2)
            
            # 应用Sobel滤波器
            grad_x = F.conv2d(smoothed, self.sobel_x, padding=1)
            grad_y = F.conv2d(smoothed, self.sobel_y, padding=1)
            
            # 计算梯度幅值和方向
            grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            grad_direction = torch.atan2(grad_y, grad_x)
            
            # 将梯度幅值和方向合并为特征
            gradient_features = torch.cat([grad_magnitude, grad_direction], dim=1)
            gradients_list.append(gradient_features)
            
            # 应用拉普拉斯滤波器提取边缘
            laplacian = F.conv2d(smoothed, self.laplacian_kernel, padding=1)
            gradients_list.append(laplacian)
        
        return gradients_list
    
    def structure_tensor_features(self, image):
        """
        计算结构张量特征
        
        参数:
            image: torch.Tensor, 形状 [1, 1, H, W]
        返回:
            结构张量特征
        """
        # 计算梯度
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        
        # 计算结构张量成分
        Ixx = grad_x * grad_x
        Iyy = grad_y * grad_y
        Ixy = grad_x * grad_y
        
        # 使用高斯平滑(sigma=1.0)
        kernel = self.gaussian_kernels[1]  # 索引为1的是sigma=1.0的高斯核
        Jxx = F.conv2d(Ixx, kernel, padding=2)
        Jyy = F.conv2d(Iyy, kernel, padding=2)
        Jxy = F.conv2d(Ixy, kernel, padding=2)
        
        # 计算特征值和特征向量
        # (Jxx + Jyy)/2 是平均值
        # sqrt((Jxx - Jyy)^2 + 4*Jxy^2)/2 是特征值之差的一半
        trace = Jxx + Jyy
        delta = torch.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2 + 1e-10)
        
        # 两个特征值
        lambda1 = (trace + delta) / 2
        lambda2 = (trace - delta) / 2
        
        # 计算各向异性和一致性
        anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
        coherence = (lambda1 - lambda2)**2 / (lambda1 + lambda2 + 1e-10)**2
        
        # 输出特征
        return torch.cat([lambda1, lambda2, anisotropy, coherence], dim=1)
    
    def compute_guided_spatial_loss(self, X, I_PAN, lambda_gradient=1.0, lambda_structure=0.5, lambda_texture=0.5):
        """
        完整的guided-deep-decoder空间损失实现
        
        参数:
            X: 重建的高分辨率多光谱图像, torch.Tensor, 形状 (H, W, S)
            I_PAN: 全色图像, torch.Tensor, 形状 (H, W) 或 (1, H, W)
            lambda_gradient: 梯度损失的权重
            lambda_structure: 结构损失的权重
            lambda_texture: 纹理损失的权重
        
        返回:
            spatial_loss: 空间损失（标量 tensor）
        """
        # 确保X是正确的形状
        if X.dim() == 4:  # 如果包含batch维度
            X = X.squeeze(0)
        
        # 确保I_PAN是正确的形状
        if I_PAN.dim() == 3 and I_PAN.shape[0] == 1:  # (1, H, W)
            I_PAN = I_PAN.squeeze(0)
        elif I_PAN.dim() == 4:  # (B, 1, H, W)
            I_PAN = I_PAN.squeeze(0).squeeze(0)
        
        # 1. 创建多光谱图像的灰度表示
        # 使用更精确的波段权重，考虑不同传感器
        if self.sensor == 'WV3' or self.sensor == 'WV2':
            # WorldView的波段权重
            if X.shape[2] >= 8:
                weights = torch.tensor([0.05, 0.05, 0.1, 0.2, 0.35, 0.15, 0.05, 0.05], device=self.device)
                if X.shape[2] > 8:
                    extra = torch.ones(X.shape[2] - 8, device=self.device) * 0.05
                    weights = torch.cat([weights, extra])
            else:
                # 简单均值
                weights = torch.ones(X.shape[2], device=self.device) / X.shape[2]
        elif self.sensor == 'QB':
            # QuickBird波段权重
            weights = torch.tensor([0.1, 0.3, 0.4, 0.2], device=self.device)
            if X.shape[2] > 4:
                extra = torch.ones(X.shape[2] - 4, device=self.device) * 0.05
                weights = torch.cat([weights, extra])
        else:
            # 默认均值权重
            weights = torch.ones(X.shape[2], device=self.device) / X.shape[2]
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 应用权重生成灰度图像
        X_gray = torch.sum(X * weights.view(1, 1, -1), dim=2)
        
        # 转换为卷积格式
        X_gray = X_gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        I_PAN = I_PAN.unsqueeze(0).unsqueeze(0)    # [1, 1, H, W]
        
        # 2. 提取多尺度梯度特征
        ms_gradients_X = self.extract_multiscale_gradients(X_gray)
        ms_gradients_PAN = self.extract_multiscale_gradients(I_PAN)
        
        # 3. 计算结构张量特征
        st_features_X = self.structure_tensor_features(X_gray)
        st_features_PAN = self.structure_tensor_features(I_PAN)
        
        # 4. 计算损失
        gradient_loss = 0.0
        # 多尺度梯度损失
        for gx, gp in zip(ms_gradients_X, ms_gradients_PAN):
            # 归一化梯度特征
            gx_mean = torch.mean(gx, dim=(2, 3), keepdim=True)
            gx_std = torch.std(gx, dim=(2, 3), keepdim=True) + 1e-6
            gp_mean = torch.mean(gp, dim=(2, 3), keepdim=True)
            gp_std = torch.std(gp, dim=(2, 3), keepdim=True) + 1e-6
            
            gx_norm = (gx - gx_mean) / gx_std
            gp_norm = (gp - gp_mean) / gp_std
            
            # L2损失
            gradient_loss += torch.mean((gx_norm - gp_norm) ** 2)
        
        # 结构张量损失
        # 归一化结构特征
        st_X_mean = torch.mean(st_features_X, dim=(2, 3), keepdim=True)
        st_X_std = torch.std(st_features_X, dim=(2, 3), keepdim=True) + 1e-6
        st_PAN_mean = torch.mean(st_features_PAN, dim=(2, 3), keepdim=True)
        st_PAN_std = torch.std(st_features_PAN, dim=(2, 3), keepdim=True) + 1e-6
        
        st_X_norm = (st_features_X - st_X_mean) / st_X_std
        st_PAN_norm = (st_features_PAN - st_PAN_mean) / st_PAN_std
        
        structure_loss = torch.mean((st_X_norm - st_PAN_norm) ** 2)
        
        # 纹理信息损失（使用GLCM或其近似）
        # 这里使用局部方差作为纹理特征的简化实现
        kernel_size = 5
        padding = kernel_size // 2
        
        # 计算局部方差
        mean_filter = torch.ones((1, 1, kernel_size, kernel_size), device=self.device) / (kernel_size * kernel_size)
        local_mean_X = F.conv2d(X_gray, mean_filter, padding=padding)
        local_mean_PAN = F.conv2d(I_PAN, mean_filter, padding=padding)
        
        local_var_X = F.conv2d((X_gray - local_mean_X)**2, mean_filter, padding=padding)
        local_var_PAN = F.conv2d((I_PAN - local_mean_PAN)**2, mean_filter, padding=padding)
        
        # 归一化局部方差
        var_X_mean = torch.mean(local_var_X)
        var_X_std = torch.std(local_var_X) + 1e-6
        var_PAN_mean = torch.mean(local_var_PAN)
        var_PAN_std = torch.std(local_var_PAN) + 1e-6
        
        var_X_norm = (local_var_X - var_X_mean) / var_X_std
        var_PAN_norm = (local_var_PAN - var_PAN_mean) / var_PAN_std
        
        texture_loss = torch.mean((var_X_norm - var_PAN_norm) ** 2)
        
        # 计算总损失
        total_loss = lambda_gradient * gradient_loss + \
                    lambda_structure * structure_loss + \
                    lambda_texture * texture_loss
                    
        return total_loss
    
    def compute_ergas_loss(self, X, I_PAN):
        """
        计算ERGAS损失 (Erreur Relative Globale Adimensionnelle de Synthèse)
        
        参数:
            X: 重建的高分辨率多光谱图像灰度化后的结果, torch.Tensor, 形状 [1, 1, H, W]
            I_PAN: 全色图像, torch.Tensor, 形状 [1, 1, H, W]
        
        返回:
            ERGAS损失（标量 tensor）
        """
        # 计算波段维度上的均方误差
        a1 = torch.mean((X - I_PAN) ** 2, dim=(-2, -1))
        # 计算参考图像的均值的平方
        a2 = torch.mean(I_PAN, dim=(-2, -1)) ** 2
        # 计算相对误差
        com = a1 / a2
        # 计算ERGAS
        ergas = 100 * (1 / self.ratio) * (com ** 0.5)
        
        return ergas.mean()

    def compute_spatial_fidelity_loss(self, X, I_LRMS, I_PAN, block_size, use_ergas=False, lamda=0.1):
        """
        计算空间损失 - 完全按照FusionMamba实现
        
        参数:
            X: 重建的高分辨率多光谱图像，torch.Tensor, 形状 (H, W, S)
            I_LRMS: 低分辨率多光谱图像（不使用）
            I_PAN: 全色图像，形状 (H, W) 或 (1, H, W)
            block_size: 块大小参数（不使用）
            use_ergas: 是否使用ERGAS损失，默认True
            lamda: ERGAS损失权重，默认0.1
        返回:
            空间损失（标量 tensor）
        """
        # 确保X是正确的形状
        if X.dim() == 4:  # 如果包含batch维度
            X = X.squeeze(0)
        
        # 确保I_PAN是正确的形状
        if I_PAN.dim() == 3 and I_PAN.shape[0] == 1:  # (1, H, W)
            I_PAN = I_PAN.squeeze(0)
        elif I_PAN.dim() == 4:  # (B, 1, H, W)
            I_PAN = I_PAN.squeeze(0).squeeze(0)
        
        # 将多光谱图像转为灰度 - 按照传感器类型使用特定权重
        if self.sensor == 'WV3' or self.sensor == 'WV2':
            # WorldView的波段权重
            if X.shape[2] >= 8:
                weights = torch.tensor([0.05, 0.05, 0.1, 0.2, 0.35, 0.15, 0.05, 0.05], device=self.device)
                if X.shape[2] > 8:
                    extra = torch.ones(X.shape[2] - 8, device=self.device) * 0.05
                    weights = torch.cat([weights, extra])
            else:
                weights = torch.ones(X.shape[2], device=self.device) / X.shape[2]
        elif self.sensor == 'QB':
            # QuickBird波段权重
            weights = torch.tensor([0.1, 0.3, 0.4, 0.2], device=self.device)
            if X.shape[2] > 4:
                extra = torch.ones(X.shape[2] - 4, device=self.device) * 0.05
                weights = torch.cat([weights, extra])
        else:
            # 默认均值权重
            weights = torch.ones(X.shape[2], device=self.device) / X.shape[2]
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 生成灰度图像
        X_gray = torch.sum(X * weights.view(1, 1, -1), dim=2)
        
        # 转换为卷积格式
        X_gray = X_gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        I_PAN = I_PAN.unsqueeze(0).unsqueeze(0)    # [1, 1, H, W]
        
        # 完全按照FusionMamba的实现方式计算损失
        if use_ergas:
            # L1损失
            l1_loss = torch.mean(torch.abs(X_gray - I_PAN))
            
            # ERGAS损失
            b, c, _, _ = X_gray.shape
            a1 = torch.mean((X_gray - I_PAN) ** 2, dim=(-2, -1))
            a2 = torch.mean(I_PAN, dim=(-2, -1)) ** 2
            com = a1 / (a2 + 1e-8)  # 添加小常数避免除零
            ergas = 100 * (1 / self.ratio) * torch.sqrt(com)
            ergas_loss = ergas.mean()
            
            # 组合两种损失
            total_loss = l1_loss + lamda * ergas_loss
        else:
            # 仅使用L1损失
            total_loss = torch.mean(torch.abs(X_gray - I_PAN))
        
        return total_loss
    def SDE_Loss(self, x, y):
            mse_loss = nn.MSELoss()  # 创建MSELoss的实例
            loss = mse_loss(x, y)     # 使用实例的forward方法计算损失
            return loss
    
class SDE_Losses(nn.Module):
    def __init__(self, device):
        super(SDE_Losses, self).__init__()
        self.mse = nn.MSELoss().to(device)

    def forward(self, lms_rr, pan_rr):
        loss = self.mse(lms_rr, pan_rr)

        return loss
class pretrain_Losses(nn.Module):
    def __init__(self, device):
        super(pretrain_Losses, self).__init__()
        self.mse = nn.MSELoss().to(device)

    def forward(self, lms_rr, pan_rr):
        loss = self.mse(lms_rr, pan_rr)

        return loss