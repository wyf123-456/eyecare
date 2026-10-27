# -*- coding: utf-8 -*-
"""
慧眼 EyeCare - 模型训练脚本
基于 MobileNetV2 + 迁移学习，训练糖尿病视网膜病变5级分类模型
数据集格式：APTOS 2019 (id_code, diagnosis)

使用方法：
    1. 确保数据集路径配置正确（见下方 DATA_CONFIG）
    2. 在 PyCharm 中直接运行此脚本，或命令行执行：python train.py
    3. 训练完成后，模型和配置会自动保存到 model/ 目录
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

# sklearn 是可选依赖，没装也能跑（混淆矩阵和分类报告会跳过）
try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# 数据集路径配置
# ============================================================
DATA_CONFIG = {
    "train_csv": r"D:\train_1.csv",
    "train_img_dir": r"D:\train_images\train_images",
    "val_csv": r"D:\valid.csv",
    "val_img_dir": r"D:\val_images\val_images",
}

# 类别定义（APTOS 5级糖尿病视网膜病变分级）
CLASSES = [
    "正常",
    "轻度糖网",
    "中度糖网",
    "重度糖网",
    "增殖期糖网",
]
CLASS_TO_IDX = {
    "normal": 0,
    "dr_mild": 1,
    "dr_moderate": 2,
    "dr_severe": 3,
    "dr_proliferative": 4,
}

# 模型保存目录
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


# ============================================================
# 超参数配置
# ============================================================
TRAIN_CONFIG = {
    "image_size": 224,
    "batch_size": 16,
    "num_epochs": 15,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "lr_scheduler_patience": 3,
    "early_stop_patience": 6,
    "freeze_backbone": True,       # 第一阶段是否冻结backbone（只训练分类头）
    "unfreeze_after_epoch": 5,     # 第几轮开始解冻backbone（设为None则不解冻）
    "num_workers": 0,              # Windows建议设为0
    "seed": 42,
}


# ============================================================
# 自定义数据集
# ============================================================
class APTOSDataset(Dataset):
    """APTOS 糖尿病视网膜病变数据集"""

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

        # 加载图片（支持.png和.jpg）
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
# 数据增强
# ============================================================
def get_train_transform(image_size=224):
    """训练集数据增强"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transform(image_size=224):
    """验证集预处理"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ============================================================
# 模型定义（与 predict.py 保持一致）
# ============================================================
def get_model(num_classes, pretrained=True):
    """获取 MobileNetV2 模型"""
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    else:
        model = models.mobilenet_v2(weights=None)

    # 替换分类头（与 predict.py 结构完全一致）
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model


def set_freeze_backbone(model, freeze=True):
    """冻结或解冻backbone特征提取层"""
    for param in model.features.parameters():
        param.requires_grad = not freeze


# ============================================================
# 训练一个epoch
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 打印进度
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} - "
                  f"Loss: {loss.item():.4f} - "
                  f"Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ============================================================
# 验证
# ============================================================
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# ============================================================
# 绘制训练曲线
# ============================================================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """绘制损失和准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-o', label='训练损失', markersize=4)
    ax1.plot(epochs, val_losses, 'r-o', label='验证损失', markersize=4)
    ax1.set_title('训练与验证损失', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-o', label='训练准确率', markersize=4)
    ax2.plot(epochs, val_accs, 'r-o', label='验证准确率', markersize=4)
    ax2.set_title('训练与验证准确率', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] 训练曲线已保存到: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """绘制混淆矩阵"""
    if not HAS_SKLEARN:
        print("[警告] 未安装scikit-learn，跳过混淆矩阵绘制")
        return
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title='混淆矩阵 (归一化)',
           ylabel='真实标签',
           xlabel='预测标签')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子里写数字
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] 混淆矩阵已保存到: {save_path}")


# ============================================================
# 主训练函数
# ============================================================
def main():
    print("=" * 70)
    print("  慧眼 EyeCare - 模型训练")
    print("  模型: MobileNetV2 + 迁移学习")
    print("  任务: 糖尿病视网膜病变 5 级分类")
    print("=" * 70)

    # --- 0. 设置随机种子 ---
    cfg = TRAIN_CONFIG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["seed"])

    # --- 1. 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备] 使用: {device}")
    if device.type == 'cpu':
        print("  ⚠️  使用CPU训练，速度较慢，建议耐心等待")

    # --- 2. 数据集 ---
    print(f"\n[数据] 加载数据集...")
    print(f"  训练标注: {DATA_CONFIG['train_csv']}")
    print(f"  训练图片: {DATA_CONFIG['train_img_dir']}")
    print(f"  验证标注: {DATA_CONFIG['val_csv']}")
    print(f"  验证图片: {DATA_CONFIG['val_img_dir']}")

    train_dataset = APTOSDataset(
        csv_path=DATA_CONFIG["train_csv"],
        img_dir=DATA_CONFIG["train_img_dir"],
        transform=get_train_transform(cfg["image_size"])
    )
    val_dataset = APTOSDataset(
        csv_path=DATA_CONFIG["val_csv"],
        img_dir=DATA_CONFIG["val_img_dir"],
        transform=get_val_transform(cfg["image_size"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"]
    )

    # 统计类别分布
    train_labels = pd.read_csv(DATA_CONFIG["train_csv"])['diagnosis'].values
    val_labels = pd.read_csv(DATA_CONFIG["val_csv"])['diagnosis'].values
    print(f"\n  训练集: {len(train_dataset)} 张")
    for i in range(5):
        print(f"    类别{i} ({CLASSES[i]}): {np.sum(train_labels == i)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    for i in range(5):
        print(f"    类别{i} ({CLASSES[i]}): {np.sum(val_labels == i)} 张")

    # --- 3. 模型 ---
    print(f"\n[模型] 构建 MobileNetV2 ...")
    num_classes = len(CLASSES)
    model = get_model(num_classes, pretrained=True)

    # 第一阶段：冻结backbone，只训练分类头
    if cfg["freeze_backbone"]:
        set_freeze_backbone(model, freeze=True)
        print("  阶段一: 冻结backbone，只训练分类头")

    model.to(device)

    # 计算可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # --- 4. 损失函数 & 优化器 ---
    # 类别不平衡，使用加权交叉熵
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"\n[损失] 类别权重: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=cfg["lr_scheduler_patience"], verbose=True
    )

    # --- 5. 训练循环 ---
    print(f"\n[训练] 开始训练，共 {cfg['num_epochs']} 个epoch")
    print("-" * 70)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, cfg["num_epochs"] + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{cfg['num_epochs']}")

        # 解冻backbone（第二阶段）
        if cfg["unfreeze_after_epoch"] is not None and epoch == cfg["unfreeze_after_epoch"] + 1:
            print(f"  🔓 阶段二: 解冻backbone，使用小学习率微调整个网络")
            set_freeze_backbone(model, freeze=False)
            # 重新创建优化器，使用更小的学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg["learning_rate"] * 0.1
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  可训练参数: {trainable_params:,}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc, val_preds, val_labels_arr = validate(
            model, val_loader, criterion, device
        )

        # 学习率调整
        scheduler.step(val_acc)

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  结果: 训练Loss={train_loss:.4f}, 训练Acc={train_acc:.2f}% | "
              f"验证Loss={val_loss:.4f}, 验证Acc={val_acc:.2f}%")
        print(f"  学习率: {current_lr:.6f} | 用时: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "train_config": cfg,
            }, best_model_path)
            print(f"  ✅ 新最佳模型! 验证准确率: {best_val_acc:.2f}% (已保存)")
        else:
            patience_counter += 1
            print(f"  最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")

        # 早停
        if patience_counter >= cfg["early_stop_patience"]:
            print(f"\n⏹️  早停触发！连续 {cfg['early_stop_patience']} 个epoch没有提升")
            break

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"训练完成！总用时: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print("=" * 70)

    # --- 6. 保存 class_info.json ---
    class_info = {
        "classes": CLASSES,
        "num_classes": num_classes,
        "image_size": cfg["image_size"],
        "best_val_acc": round(best_val_acc, 2),
        "class_to_idx": CLASS_TO_IDX,
        "train_epochs": best_epoch,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "model_arch": "MobileNetV2",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    class_info_path = os.path.join(MODEL_SAVE_DIR, "class_info.json")
    with open(class_info_path, 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 类别信息已保存到: {class_info_path}")

    # --- 7. 加载最佳模型，生成评估报告 ---
    print(f"\n[评估] 加载最佳模型并生成评估报告...")
    checkpoint = torch.load(
        os.path.join(MODEL_SAVE_DIR, "best_model.pth"),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_loss, val_acc, val_preds, val_labels_arr = validate(
        model, val_loader, criterion, device
    )

    # 分类报告
    if HAS_SKLEARN:
        report = classification_report(
            val_labels_arr, val_preds,
            target_names=CLASSES,
            digits=4
        )
        print("\n分类报告:")
        print(report)

        # 保存分类报告
        report_path = os.path.join(MODEL_SAVE_DIR, "classification_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"EyeCare Model Evaluation Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Model: MobileNetV2\n")
            f.write(f"Best Val Acc: {best_val_acc:.2f}%\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"[保存] 分类报告已保存到: {report_path}")
    else:
        print("\n[提示] 未安装scikit-learn，跳过详细分类报告")

    # --- 8. 绘制曲线和混淆矩阵 ---
    curves_path = os.path.join(MODEL_SAVE_DIR, "training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    cm_path = os.path.join(MODEL_SAVE_DIR, "confusion_matrix.png")
    plot_confusion_matrix(val_labels_arr, val_preds, CLASSES, cm_path)

    # --- 9. 完成 ---
    print("\n" + "=" * 70)
    print("🎉 训练全部完成！")
    print(f"  模型文件: {os.path.join(MODEL_SAVE_DIR, 'best_model.pth')}")
    print(f"  配置文件: {class_info_path}")
    print(f"  训练曲线: {curves_path}")
    print(f"  混淆矩阵: {cm_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
