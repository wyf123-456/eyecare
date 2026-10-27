# -*- coding: utf-8 -*-
"""
慧眼 EyeCare - 模型评估与可视化脚本
加载已训练好的模型，在验证集上评估，生成混淆矩阵、分类报告、训练曲线
（不需要重新训练，直接用 model/best_model.pth）
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    print("[字体] 已设置中文字体")
except:
    print("[字体] 未找到中文字体，将使用英文标签")

from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 配置
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CLASS_INFO_PATH = os.path.join(MODEL_DIR, "class_info.json")

VAL_CSV = r"D:\valid.csv"
VAL_IMG_DIR = r"D:\val_images\val_images"

# 从训练日志中提取的历史数据（用于重绘训练曲线）
# 格式: (epoch, train_loss, val_loss, train_acc, val_acc)
TRAIN_HISTORY = [
    (1,  1.2682, 1.0779, 57.30, 68.03),
    (2,  1.1503, 1.1370, 62.90, 62.30),
    (3,  1.0812, 1.0005, 65.05, 70.22),
    (4,  1.0821, 1.0266, 65.84, 71.31),
    (5,  1.0634, 0.9892, 66.48, 67.76),
    (6,  1.0336, 0.9953, 66.89, 72.13),
    (7,  1.0238, 0.9942, 67.03, 70.22),
    (8,  1.0035, 0.9815, 67.92, 71.04),
    (9,  0.9849, 0.9727, 69.49, 71.86),
    (10, 0.9827, 0.9758, 70.14, 72.13),
    (11, 0.9758, 1.0108, 69.22, 71.31),
    (12, 0.9793, 1.0136, 69.62, 70.77),
]


# ============================================================
# 数据集
# ============================================================
class APTOSDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        label = int(row['diagnosis'])

        img_path_png = os.path.join(self.img_dir, f"{img_id}.png")
        img_path_jpg = os.path.join(self.img_dir, f"{img_id}.jpg")

        if os.path.exists(img_path_png):
            img = Image.open(img_path_png).convert('RGB')
        elif os.path.exists(img_path_jpg):
            img = Image.open(img_path_jpg).convert('RGB')
        else:
            raise FileNotFoundError(f"找不到图片: {img_id}")

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# 模型（与训练时一致）
# ============================================================
def get_model(num_classes):
    model = models.mobilenet_v2(weights=None)
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
# 推理验证集
# ============================================================
def evaluate_model():
    print("=" * 60)
    print("  模型评估")
    print("=" * 60)

    # 加载类别信息
    with open(CLASS_INFO_PATH, 'r', encoding='utf-8') as f:
        class_info = json.load(f)

    classes = class_info["classes"]
    num_classes = class_info["num_classes"]
    image_size = class_info["image_size"]
    best_val_acc = class_info["best_val_acc"]

    print(f"\n模型: MobileNetV2")
    print(f"类别数: {num_classes}")
    print(f"图片尺寸: {image_size}x{image_size}")
    print(f"最佳验证准确率: {best_val_acc}%")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    print("\n加载模型...")
    model = get_model(num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✅ 模型加载完成")

    # 验证集
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = APTOSDataset(VAL_CSV, VAL_IMG_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"验证集大小: {len(val_dataset)} 张")

    # 推理
    print("\n开始推理...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(val_loader)} batch")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 准确率
    accuracy = 100. * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"\n✅ 验证准确率: {accuracy:.2f}%")

    return all_preds, all_labels, all_probs, classes


# ============================================================
# 绘制训练曲线（英文标签，避免字体问题）
# ============================================================
def plot_training_curves_en():
    """用英文标签绘制训练曲线，避免中文字体问题"""
    epochs = [h[0] for h in TRAIN_HISTORY]
    train_losses = [h[1] for h in TRAIN_HISTORY]
    val_losses = [h[2] for h in TRAIN_HISTORY]
    train_accs = [h[3] for h in TRAIN_HISTORY]
    val_accs = [h[4] for h in TRAIN_HISTORY]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', markersize=4)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 标注最佳点
    best_idx = np.argmax(val_accs)
    ax1.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({epochs[best_idx]})')
    ax1.legend(fontsize=10)

    # Accuracy
    ax2.plot(epochs, train_accs, 'b-o', label='Train Acc', markersize=4)
    ax2.plot(epochs, val_accs, 'r-o', label='Val Acc', markersize=4)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 标注最佳点
    best_acc = max(val_accs)
    ax2.scatter([epochs[best_idx]], [best_acc], c='green', s=100, zorder=5, edgecolors='black')
    ax2.annotate(f'Best: {best_acc:.2f}%',
                 xy=(epochs[best_idx], best_acc),
                 xytext=(epochs[best_idx]+0.5, best_acc+1),
                 fontsize=10, fontweight='bold', color='green')

    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[保存] 训练曲线: {save_path}")


# ============================================================
# 绘制混淆矩阵（英文标签）
# ============================================================
def plot_confusion_matrix_en(y_true, y_pred, classes):
    """用英文标签绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 英文类别名
    class_names_en = [
        'Normal',
        'Mild DR',
        'Moderate DR',
        'Severe DR',
        'Proliferative DR'
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names_en,
           yticklabels=class_names_en,
           title='Confusion Matrix (Normalized)',
           ylabel='True Label',
           xlabel='Predicted Label')

    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] 混淆矩阵: {save_path}")


# ============================================================
# 保存分类报告
# ============================================================
def save_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("\n" + "=" * 60)
    print("  分类报告")
    print("=" * 60)
    print(report)

    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  EyeCare Model Evaluation Report\n")
        f.write("  Model: MobileNetV2 (Transfer Learning on APTOS 2019)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Dataset: APTOS 2019 Blindness Detection\n")
        f.write(f"Validation samples: {len(y_true)}\n")
        f.write(f"Overall Accuracy: {100.*np.sum(y_true==y_pred)/len(y_true):.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n")

    print(f"[保存] 分类报告: {report_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    # 1. 重绘训练曲线
    print("\n" + "=" * 60)
    print("  生成训练曲线图")
    print("=" * 60)
    plot_training_curves_en()

    # 2. 模型评估
    preds, labels, probs, classes = evaluate_model()

    # 3. 混淆矩阵
    print("\n" + "=" * 60)
    print("  生成混淆矩阵")
    print("=" * 60)
    plot_confusion_matrix_en(labels, preds, classes)

    # 4. 分类报告
    save_classification_report(labels, preds, classes)

    # 5. 完成
    print("\n" + "=" * 60)
    print("🎉 评估完成！生成的文件：")
    print(f"  训练曲线: {os.path.join(MODEL_DIR, 'training_curves.png')}")
    print(f"  混淆矩阵: {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")
    print(f"  分类报告: {os.path.join(MODEL_DIR, 'classification_report.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
