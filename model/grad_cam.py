# -*- coding: utf-8 -*-
"""
慧眼 EyeCare - Grad-CAM 可解释性模块
基于 Grad-CAM (Gradient-weighted Class Activation Mapping) 实现
生成真实的模型注意力热力图，展示AI决策依据
"""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM 热力图生成器
    
    用法：
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(image_tensor, target_class)
    """

    def __init__(self, model, target_layer):
        """
        初始化GradCAM

        参数:
            model: PyTorch模型
            target_layer: 目标层（用于计算Grad-CAM的卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册前向钩子（获取激活值）
        target_layer.register_forward_hook(self._save_activation)
        # 注册反向钩子（获取梯度）
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """保存前向传播的激活值"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        生成Grad-CAM热力图

        参数:
            input_tensor: 输入图像张量 (1, C, H, W)
            target_class: 目标类别索引，如果为None则使用预测概率最大的类别

        返回:
            numpy.ndarray: 归一化的热力图 (H, W)，值范围[0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad = True

        # 前向传播
        output = self.model(input_tensor)

        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播（只对目标类别求梯度）
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # 获取梯度和激活值
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # 全局平均池化梯度，得到每个通道的权重
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # 加权求和
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU：只保留正贡献
        cam = F.relu(cam)

        # 归一化到 [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def overlay_heatmap_on_image(heatmap, original_image, alpha=0.5, colormap='jet'):
    """
    将热力图叠加到原图上

    参数:
        heatmap: numpy.ndarray, 热力图 (H, W), 值范围[0, 1]
        original_image: PIL.Image, 原图
        alpha: float, 热力图透明度
        colormap: str, 颜色映射 ('jet' 或 'hot')

    返回:
        PIL.Image: 叠加后的图像
    """
    # 获取原图尺寸
    orig_w, orig_h = original_image.size

    # 将热力图resize到原图尺寸
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR))
    heatmap_resized = heatmap_resized.astype(np.float32) / 255.0

    # 转换为彩色热力图
    if colormap == 'jet':
        # Jet colormap: 蓝 -> 青 -> 黄 -> 红
        heatmap_colored = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for i in range(3):
            if i == 0:  # Red channel
                heatmap_colored[:, :, 0] = np.clip((heatmap_resized - 0.5) * 2, 0, 1) * 255
            elif i == 1:  # Green channel
                green = np.where(heatmap_resized < 0.5,
                                 heatmap_resized * 2,
                                 1 - (heatmap_resized - 0.5) * 2)
                heatmap_colored[:, :, 1] = green * 255
            else:  # Blue channel
                heatmap_colored[:, :, 2] = np.clip((0.5 - heatmap_resized) * 2, 0, 1) * 255
    else:  # hot colormap
        heatmap_colored = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        heatmap_colored[:, :, 0] = np.clip(heatmap_resized * 2, 0, 1) * 255
        heatmap_colored[:, :, 1] = np.clip((heatmap_resized - 0.5) * 2, 0, 1) * 255
        heatmap_colored[:, :, 2] = np.clip((heatmap_resized - 0.75) * 4, 0, 1) * 255

    heatmap_pil = Image.fromarray(heatmap_colored)

    # 叠加
    original_array = np.array(original_image.convert('RGB')).astype(np.float32)
    heatmap_array = np.array(heatmap_pil).astype(np.float32)

    # 只在热力图较强的地方叠加（弱的地方保留原图）
    mask = heatmap_resized[:, :, np.newaxis]
    blended = original_array * (1 - alpha * mask) + heatmap_array * (alpha * mask)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def get_mobilenetv2_target_layer(model):
    """
    获取MobileNetV2模型用于Grad-CAM的目标层
    MobileNetV2的最后一个卷积块在 features[-1] 中
    """
    # features[-1] 是最后一个ConvBNReLU层
    # 里面的 [0] 是Conv2d
    return model.features[-1][0]
