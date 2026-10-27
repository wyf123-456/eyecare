# -*- coding: utf-8 -*-
"""
慧眼 EyeCare - AI模型推理模块
加载训练好的MobileNetV2模型，进行眼底图像分类预测
"""

import os
import json
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# ============================================================
# 配置
# ============================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CLASS_INFO_PATH = os.path.join(MODEL_DIR, "class_info.json")

# 全局模型缓存（避免重复加载）
_model = None
_class_info = None
_device = None
_transform = None
_gradcam = None  # GradCAM实例


# ============================================================
# 模型定义（与训练时一致）
# ============================================================
def get_model(num_classes):
    """获取 MobileNetV2 模型结构（与训练脚本一致）"""
    model = models.mobilenet_v2(weights=None)  # 推理时不下载预训练权重

    # 替换分类头（与训练时结构完全一致）
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model


# ============================================================
# 图像预处理
# ============================================================
def get_transform(image_size=224):
    """获取验证集预处理变换"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ============================================================
# 模型加载
# ============================================================
def load_model():
    """加载训练好的模型（单例模式）"""
    global _model, _class_info, _device, _transform

    if _model is not None:
        return _model, _class_info, _device, _transform

    # 设备
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载类别信息
    with open(CLASS_INFO_PATH, 'r', encoding='utf-8') as f:
        _class_info = json.load(f)

    num_classes = _class_info["num_classes"]
    image_size = _class_info["image_size"]

    # 创建模型并加载权重
    _model = get_model(num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
    # 兼容两种保存格式：直接state_dict 或 checkpoint字典
    if "model_state_dict" in checkpoint:
        _model.load_state_dict(checkpoint["model_state_dict"])
    else:
        _model.load_state_dict(checkpoint)
    _model.to(_device)
    _model.eval()

    # 预处理变换
    _transform = get_transform(image_size)

    print(f"[EyeCare] 模型加载完成，设备: {_device}，类别数: {num_classes}")

    return _model, _class_info, _device, _transform


# ============================================================
# 预测函数
# ============================================================
def predict(image):
    """
    对单张眼底图像进行预测

    参数:
        image: PIL.Image 格式的RGB图像

    返回:
        dict: 预测结果，包含:
            - predicted_class: 预测类别名称
            - confidence: 置信度 (0-1)
            - all_predictions: 所有类别的概率列表 [{class: name, prob: float}]
            - predicted_idx: 预测类别索引
            - inference_time: 推理时间（秒）
    """
    model, class_info, device, transform = load_model()

    start_time = time.time()

    # 预处理
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    # 结果整理
    classes = class_info["classes"]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = classes[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    all_predictions = [
        {"class": classes[i], "prob": float(probabilities[i])}
        for i in range(len(classes))
    ]
    # 按概率降序排列
    all_predictions.sort(key=lambda x: x["prob"], reverse=True)

    inference_time = time.time() - start_time

    return {
        "predicted_class": predicted_class,
        "predicted_idx": predicted_idx,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "inference_time": inference_time,
        "classes": classes,
    }


# ============================================================
# 类别映射到疾病数据库key
# ============================================================
CLASS_TO_DISEASE_KEY = {
    "正常": "normal",
    "轻度糖网": "dr_mild",
    "中度糖网": "dr_moderate",
    "重度糖网": "dr_severe",
    "增殖期糖网": "dr_proliferative",
    # 兼容旧版混合眼病模型
    "青光眼疑似": "glaucoma_suspect",
    "黄斑变性": "amd_dry",
}


def get_disease_key(predicted_class):
    """将预测类别名映射到疾病数据库的key"""
    return CLASS_TO_DISEASE_KEY.get(predicted_class, "normal")


# ============================================================
# 生成完整的诊断结果（兼容原mock_data格式）
# ============================================================
def generate_diagnosis_result(image):
    """
    生成完整的诊断结果，格式与原mock_fundus_analysis兼容

    参数:
        image: PIL.Image 或 图片路径

    返回:
        dict: 完整诊断结果
    """
    from utils.mock_data import DISEASE_DATABASE

    # 模型预测
    pred = predict(image)
    disease_key = get_disease_key(pred["predicted_class"])
    disease = DISEASE_DATABASE.get(disease_key, DISEASE_DATABASE["normal"])

    # 构建热力图中心（根据不同疾病类型）
    heatmap_centers = {
        "normal": (0.5, 0.5),
        "dr_mild": (0.45, 0.55),
        "dr_moderate": (0.5, 0.5),
        "dr_severe": (0.55, 0.45),
        "dr_proliferative": (0.5, 0.5),
        "glaucoma_suspect": (0.35, 0.4),
        "amd_dry": (0.5, 0.45),
    }

    # 构建 all_predictions 为列表格式（兼容visualizer）
    all_pred_list = [
        {
            "name": item["class"],
            "probability": item["prob"],
            "is_primary": (item["class"] == pred["predicted_class"])
        }
        for item in pred["all_predictions"]
    ]

    result = {
        "primary_diagnosis": disease["name"],
        "risk_level": disease["risk_level"],
        "risk_color": disease["risk_color"],
        "severity": disease["severity"],
        "confidence": pred["confidence"],
        "description": disease["description"],
        "features": disease["features"],
        "recommendation": disease["recommendation"],
        "doctor_advice": disease["doctor_advice"],
        "all_predictions": all_pred_list,
        "heatmap_center": heatmap_centers.get(disease_key, (0.5, 0.5)),
        "analysis_time": f"{pred['inference_time']*1000:.0f}ms",
        "patient_info": {
            "age": "-",
            "gender": "-",
        },
        # 额外的模型信息
        "model_info": {
            "model_name": "MobileNetV2",
            "num_classes": len(pred["classes"]),
            "inference_time": pred["inference_time"],
            "device": str(_device) if _device else "cpu",
        }
    }

    return result, disease_key


# ============================================================
# 检查模型是否可用
# ============================================================
def is_model_available():
    """检查模型文件是否存在"""
    return os.path.exists(MODEL_PATH) and os.path.exists(CLASS_INFO_PATH)


# ============================================================
# 真实 Grad-CAM 热力图生成
# ============================================================
def get_gradcam():
    """获取GradCAM实例（单例模式）"""
    global _gradcam, _model

    if _gradcam is not None:
        return _gradcam

    if _model is None:
        load_model()

    from model.grad_cam import GradCAM, get_mobilenetv2_target_layer

    target_layer = get_mobilenetv2_target_layer(_model)
    _gradcam = GradCAM(_model, target_layer)

    print("[EyeCare] Grad-CAM 初始化完成")
    return _gradcam


def predict_with_gradcam(image):
    """
    预测 + 生成真实Grad-CAM热力图

    参数:
        image: PIL.Image 或 图片路径

    返回:
        tuple: (pred_result, heatmap_pil)
            - pred_result: 预测结果字典（与predict()返回格式一致）
            - heatmap_pil: PIL.Image，叠加了热力图的图像
    """
    from model.grad_cam import overlay_heatmap_on_image

    model, class_info, device, transform = load_model()
    gradcam = get_gradcam()

    start_time = time.time()

    # 预处理
    if isinstance(image, str):
        original_image = Image.open(image).convert("RGB")
    else:
        original_image = image.copy()

    img_tensor = transform(original_image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # 前向传播
    model.eval()
    outputs = model(img_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0].cpu().detach().numpy()

    # 预测结果
    classes = class_info["classes"]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = classes[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    all_predictions = [
        {"class": classes[i], "prob": float(probabilities[i])}
        for i in range(len(classes))
    ]
    all_predictions.sort(key=lambda x: x["prob"], reverse=True)

    # 生成Grad-CAM热力图（对预测类别）
    model.zero_grad()
    target_score = outputs[0, predicted_idx]
    target_score.backward(retain_graph=True)

    # 获取梯度和激活值，计算CAM
    # 注意：GradCAM类已经通过钩子保存了梯度和激活值
    # 这里直接重新生成一下，确保用的是当前预测的梯度
    gradcam_obj = gradcam
    gradients = gradcam_obj.gradients
    activations = gradcam_obj.activations

    if gradients is not None and activations is not None:
        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        # 加权求和
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        # ReLU
        cam = torch.nn.functional.relu(cam)
        # 归一化
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
    else:
        # fallback：全零热力图
        cam = np.zeros((7, 7), dtype=np.float32)

    # 叠加热力图到原图
    heatmap_pil = overlay_heatmap_on_image(cam, original_image, alpha=0.6, colormap='jet')

    inference_time = time.time() - start_time

    pred_result = {
        "predicted_class": predicted_class,
        "predicted_idx": predicted_idx,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "inference_time": inference_time,
        "classes": classes,
    }

    return pred_result, heatmap_pil


def generate_diagnosis_result_with_gradcam(image):
    """
    生成完整诊断结果 + 真实Grad-CAM热力图

    参数:
        image: PIL.Image 或 图片路径

    返回:
        tuple: (result_dict, disease_key, heatmap_pil)
    """
    from utils.mock_data import DISEASE_DATABASE

    # 预测 + 热力图
    pred, heatmap_pil = predict_with_gradcam(image)
    disease_key = get_disease_key(pred["predicted_class"])
    disease = DISEASE_DATABASE.get(disease_key, DISEASE_DATABASE["normal"])

    # 构建 all_predictions 列表
    all_pred_list = [
        {
            "name": item["class"],
            "probability": item["prob"],
            "is_primary": (item["class"] == pred["predicted_class"])
        }
        for item in pred["all_predictions"]
    ]

    result = {
        "primary_diagnosis": disease["name"],
        "risk_level": disease["risk_level"],
        "risk_color": disease["risk_color"],
        "severity": disease["severity"],
        "confidence": pred["confidence"],
        "description": disease["description"],
        "features": disease["features"],
        "recommendation": disease["recommendation"],
        "doctor_advice": disease["doctor_advice"],
        "all_predictions": all_pred_list,
        "heatmap_center": (0.5, 0.5),  # 真实Grad-CAM不需要这个了
        "analysis_time": f"{pred['inference_time']*1000:.0f}ms",
        "patient_info": {
            "age": "-",
            "gender": "-",
        },
        "model_info": {
            "model_name": "MobileNetV2",
            "num_classes": len(pred["classes"]),
            "inference_time": pred["inference_time"],
            "device": str(_device) if _device else "cpu",
            "gradcam": True,  # 标记使用了真实Grad-CAM
        }
    }

    return result, disease_key, heatmap_pil


if __name__ == "__main__":
    # 简单测试
    print(f"模型可用: {is_model_available()}")
    if is_model_available():
        model, class_info, device, transform = load_model()
        print(f"类别: {class_info['classes']}")
        print(f"最佳验证准确率: {class_info['best_val_acc']}%")
