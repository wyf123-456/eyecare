# -*- coding: utf-8 -*-
"""
慧眼 EyeCare - 可视化模块
绘制眼底图、Grad-CAM热力图、性能图表等
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter
import math


def setup_matplotlib():
    """配置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def generate_sample_fundus_image(case_type: str = "normal", size: int = 400) -> Image.Image:
    """
    生成模拟眼底图像（圆形眼底图）

    Args:
        case_type: 病例类型
        size: 图像大小

    Returns:
        PIL Image
    """
    # 创建黑色背景
    img = Image.new('RGB', (size, size), '#0a0a0a')
    draw = ImageDraw.Draw(img)

    center = size // 2
    radius = int(size * 0.42)

    # 眼底基底颜色（橙红色）
    base_color = {
        "normal": (200, 120, 60),
        "dr_mild": (190, 110, 55),
        "dr_moderate": (170, 95, 45),
        "glaucoma_suspect": (210, 130, 70),
        "amd_dry": (180, 100, 50),
        "hypertensive": (195, 115, 58),
        "myopia_high": (220, 150, 90),
    }.get(case_type, (200, 120, 60))

    # 绘制眼底圆形区域（带渐变）
    for r in range(radius, 0, -2):
        alpha = 1.0 - (r / radius) * 0.3
        color = (
            int(base_color[0] * alpha),
            int(base_color[1] * alpha),
            int(base_color[2] * alpha)
        )
        draw.ellipse([center - r, center - r, center + r, center + r], fill=color)

    # 绘制视盘（视神经乳头）- 偏鼻侧（左侧）
    disc_x = int(center - radius * 0.35)
    disc_y = center
    disc_r = int(radius * 0.18)

    # 视盘颜色
    disc_color = {
        "normal": (255, 200, 160),
        "dr_mild": (250, 195, 155),
        "dr_moderate": (240, 180, 140),
        "glaucoma_suspect": (200, 160, 120),  # 青光眼视盘颜色偏白
        "amd_dry": (250, 195, 155),
        "hypertensive": (230, 180, 140),
        "myopia_high": (255, 210, 170),
    }.get(case_type, (255, 200, 160))

    draw.ellipse([disc_x - disc_r, disc_y - int(disc_r * 1.1),
                  disc_x + disc_r, disc_y + int(disc_r * 1.1)],
                 fill=disc_color, outline=(180, 130, 90))

    # 青光眼：大视杯
    if case_type == "glaucoma_suspect":
        cup_r = int(disc_r * 0.7)  # C/D比约0.7
        draw.ellipse([disc_x - cup_r, disc_y - int(cup_r * 1.1),
                      disc_x + cup_r, disc_y + int(cup_r * 1.1)],
                     fill=(255, 220, 180))

    # 黄斑（颞侧，偏右）
    macula_x = int(center + radius * 0.3)
    macula_y = center
    macula_r = int(radius * 0.12)

    # 黄斑中心凹
    macula_color = (170, 90, 40)
    draw.ellipse([macula_x - macula_r, macula_y - int(macula_r * 0.8),
                  macula_x + macula_r, macula_y + int(macula_r * 0.8)],
                 fill=macula_color)

    # 黄斑中心凹反光点
    foveal_r = int(macula_r * 0.2)
    draw.ellipse([macula_x - foveal_r, macula_y - foveal_r,
                  macula_x + foveal_r, macula_y + foveal_r],
                 fill=(255, 220, 180))

    # 视网膜血管（动脉+静脉）
    vessel_color_artery = (255, 150, 130)  # 动脉：鲜红色
    vessel_color_vein = (180, 60, 50)  # 静脉：暗红色

    # 从视盘发出的主要血管
    for angle in np.linspace(-math.pi * 0.7, math.pi * 0.7, 8):
        # 血管路径
        start_x = disc_x + math.cos(angle) * disc_r * 0.5
        start_y = disc_y + math.sin(angle) * disc_r * 0.5

        end_x = center + math.cos(angle) * radius * 0.9
        end_y = center + math.sin(angle + 0.1) * radius * 0.9

        # 动脉（细）
        draw.line([(start_x, start_y), (end_x, end_y)],
                  fill=vessel_color_artery, width=2)
        # 静脉（粗）
        draw.line([(start_x + 5, start_y + 3), (end_x + 5, end_y + 3)],
                  fill=vessel_color_vein, width=4)

    # ==========================================
    # 不同疾病的特征表现
    # ==========================================

    if case_type == "dr_mild":
        # 微动脉瘤：散在的小红点
        for _ in range(8):
            x = center + int(np.random.normal(0, radius * 0.3))
            y = center + int(np.random.normal(0, radius * 0.25))
            # 检查是否在眼底范围内
            if math.hypot(x - center, y - center) < radius * 0.8:
                dot_r = random.randint(2, 5)
                draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                             fill=(180, 30, 30))

        # 少量点状出血
        for _ in range(3):
            x = center + int(np.random.normal(0, radius * 0.35))
            y = center + int(np.random.normal(0, radius * 0.3))
            if math.hypot(x - center, y - center) < radius * 0.75:
                draw.ellipse([x - 4, y - 6, x + 4, y + 6],
                             fill=(160, 20, 20))

    elif case_type == "dr_moderate":
        # 更多微动脉瘤和出血
        for _ in range(20):
            x = center + int(np.random.normal(0, radius * 0.35))
            y = center + int(np.random.normal(0, radius * 0.3))
            if math.hypot(x - center, y - center) < radius * 0.8:
                dot_r = random.randint(2, 6)
                draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                             fill=(180, 30, 30))

        # 片状出血
        for _ in range(5):
            x = center + int(np.random.normal(0, radius * 0.3))
            y = center + int(np.random.normal(0, radius * 0.25))
            if math.hypot(x - center, y - center) < radius * 0.7:
                for _ in range(8):
                    dx = random.randint(-12, 12)
                    dy = random.randint(-8, 8)
                    draw.ellipse([x + dx - 3, y + dy - 2, x + dx + 3, y + dy + 2],
                                 fill=(140, 15, 15))

        # 硬性渗出（黄色蜡样斑点）
        for _ in range(12):
            x = macula_x + int(np.random.normal(0, radius * 0.15))
            y = macula_y + int(np.random.normal(0, radius * 0.12))
            if math.hypot(x - macula_x, y - macula_y) < radius * 0.25:
                ex_r = random.randint(3, 8)
                draw.ellipse([x - ex_r, y - int(ex_r * 0.6),
                              x + ex_r, y + int(ex_r * 0.6)],
                             fill=(255, 230, 100))

    elif case_type == "amd_dry":
        # 玻璃膜疣（黄斑区周围的黄白色小点）
        for _ in range(25):
            angle = np.random.uniform(0, 2 * math.pi)
            dist = np.random.uniform(radius * 0.05, radius * 0.2)
            x = macula_x + int(math.cos(angle) * dist)
            y = macula_y + int(math.sin(angle) * dist)
            if math.hypot(x - center, y - center) < radius * 0.85:
                drusen_r = random.randint(3, 7)
                draw.ellipse([x - drusen_r, y - int(drusen_r * 0.6),
                              x + drusen_r, y + int(drusen_r * 0.6)],
                             fill=(255, 240, 180))

        # 色素紊乱
        for _ in range(10):
            x = macula_x + int(np.random.normal(0, radius * 0.1))
            y = macula_y + int(np.random.normal(0, radius * 0.08))
            if math.hypot(x - macula_x, y - macula_y) < radius * 0.15:
                draw.ellipse([x - 3, y - 2, x + 3, y + 2],
                             fill=(120, 60, 20))

    elif case_type == "hypertensive":
        # 动脉变细、反光增强（铜丝样）
        for angle in np.linspace(-math.pi * 0.7, math.pi * 0.7, 8):
            start_x = disc_x + math.cos(angle) * disc_r * 0.5
            start_y = disc_y + math.sin(angle) * disc_r * 0.5
            end_x = center + math.cos(angle) * radius * 0.9
            end_y = center + math.sin(angle + 0.1) * radius * 0.9

            # 动脉变细，反光增强
            draw.line([(start_x, start_y), (end_x, end_y)],
                      fill=(255, 200, 180), width=1)  # 中央反光带
            draw.line([(start_x - 1, start_y - 1), (end_x - 1, end_y - 1)],
                      fill=(200, 100, 80), width=2)

    elif case_type == "myopia_high":
        # 豹纹状眼底（脉络膜血管可见）
        for angle in np.linspace(0, 2 * math.pi, 20):
            for r_mult in np.linspace(0.3, 0.9, 5):
                x = center + int(math.cos(angle) * radius * r_mult)
                y = center + int(math.sin(angle + 0.1) * radius * r_mult * 0.95)
                if math.hypot(x - center, y - center) < radius * 0.95:
                    draw.line([(x - 2, y), (x + 2, y)],
                              fill=(160, 85, 40), width=1)

        # 视盘倾斜，近视弧
        draw.ellipse([disc_x - disc_r - 5, disc_y - int(disc_r * 1.3),
                      disc_x + disc_r + 5, disc_y + int(disc_r * 1.3)],
                     fill=(255, 215, 175), outline=(180, 130, 90))

    # 添加高斯模糊让图像更真实
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img


def generate_gradcam_heatmap(fundus_img: Image.Image, center: tuple,
                              intensity: float = 0.8, case_type: str = "normal",
                              seed: int = None) -> Image.Image:
    """
    生成Grad-CAM热力图（叠加在原图上）

    Args:
        fundus_img: 眼底原图
        center: 热力图中心 (x_ratio, y_ratio)
        intensity: 热力强度
        case_type: 病例类型（影响热力分布）
        seed: 随机种子，用于可复现的热力图（同一病例同一种子结果一致）

    Returns:
        叠加热力图后的图像
    """
    width, height = fundus_img.size
    cx = int(center[0] * width)
    cy = int(center[1] * height)

    # 如果没指定种子，用case_type生成确定性种子，保证同病例热力图一致
    if seed is None:
        seed = hash(case_type) % (2**31)
    
    # 设置随机种子，保证结果可复现
    rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)

    # 创建热力图
    heatmap = np.zeros((height, width), dtype=np.float32)

    # 生成多高斯分布叠加，模拟真实Grad-CAM
    n_blobs = random.randint(3, 6)
    for _ in range(n_blobs):
        # 随机偏移
        offset_x = int(np.random.normal(0, width * 0.1))
        offset_y = int(np.random.normal(0, height * 0.1))
        blob_cx = cx + offset_x
        blob_cy = cy + offset_y

        # 随机扩散范围
        sigma = random.randint(int(width * 0.08), int(width * 0.18))

        # 2D高斯
        y, x = np.ogrid[:height, :width]
        dist_sq = (x - blob_cx) ** 2 + (y - blob_cy) ** 2
        blob = np.exp(-dist_sq / (2 * sigma ** 2))
        heatmap += blob * random.uniform(0.5, 1.0)

    # 归一化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # 只保留眼底圆形区域内的热力
    center_pt = (width // 2, height // 2)
    radius = int(width * 0.42)
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center_pt[0]) ** 2 + (y - center_pt[1]) ** 2)
    mask = dist_from_center <= radius
    heatmap = heatmap * mask

    # 恢复随机状态
    random.setstate(rng_state)
    np.random.set_state(np_rng_state)

    # 热力图转彩色（jet色图）
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_uint8).convert('L')

    # 应用彩色映射
    heatmap_colored = Image.new('RGB', (width, height))
    heatmap_pixels = heatmap_pil.load()
    out_pixels = heatmap_colored.load()

    for y in range(height):
        for x in range(width):
            val = heatmap_pixels[x, y]
            # 简化版jet colormap
            if val < 32:
                r, g, b = 0, 0, int(val * 4)
            elif val < 96:
                r, g, b = 0, int((val - 32) * 4), 255
            elif val < 160:
                r, g, b = int((val - 96) * 4), 255, int(255 - (val - 96) * 4)
            elif val < 224:
                r, g, b = 255, int(255 - (val - 160) * 4), 0
            else:
                r, g, b = 255, int((val - 224) * 4), int((val - 224) * 4)
            out_pixels[x, y] = (r, g, b)

    # 与原图混合
    result = Image.blend(fundus_img, heatmap_colored, alpha=intensity * 0.5)

    return result


def draw_risk_gauge(risk_level: str, score: float, color: str,
                    figsize=(5, 3)) -> Image.Image:
    """
    绘制风险仪表盘

    Args:
        risk_level: 风险等级文字
        score: 风险分数 0-1
        color: 颜色
        figsize: 图像大小

    Returns:
        PIL Image
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 半圆仪表盘
    theta = np.linspace(np.pi, 0, 100)
    r = 1.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 背景弧
    ax.plot(x, y, color='#e5e7eb', linewidth=20, solid_capstyle='round')

    # 前景弧（根据分数）
    end_angle = np.pi - score * np.pi
    theta_fill = np.linspace(np.pi, end_angle, 50)
    x_fill = r * np.cos(theta_fill)
    y_fill = r * np.sin(theta_fill)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('risk', ['#10B981', '#F59E0B', '#EF4444'])
    color_val = cmap(score)

    ax.plot(x_fill, y_fill, color=color_val, linewidth=20, solid_capstyle='round')

    # 指针
    pointer_x = r * np.cos(end_angle) * 0.8
    pointer_y = r * np.sin(end_angle) * 0.8
    ax.plot([0, pointer_x], [0, pointer_y], color='#374151', linewidth=3, solid_capstyle='round')
    ax.scatter([0], [0], s=100, color='#374151', zorder=5)

    # 风险等级文字
    ax.text(0, 0.3, risk_level, fontsize=20, fontweight='bold',
            ha='center', color=color_val)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='white', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def draw_prediction_barplot(predictions: list, figsize=(8, 5)) -> Image.Image:
    """
    绘制预测概率柱状图

    Args:
        predictions: 预测结果列表
        figsize: 图像大小

    Returns:
        PIL Image
    """
    setup_matplotlib()

    names = [p["name"] for p in predictions]
    probs = [p["probability"] * 100 for p in predictions]
    is_primary = [p.get("is_primary", False) for p in predictions]

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = ['#EF4444' if p else '#667eea' for p in is_primary]
    bars = ax.barh(names[::-1], probs[::-1], color=colors[::-1],
                   height=0.6, edgecolor='white', linewidth=2)

    for bar, prob in zip(bars, probs[::-1]):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f'{prob:.1f}%', va='center', fontsize=11, fontweight='bold', color='#333')

    ax.set_xlim(0, 105)
    ax.set_xlabel('预测概率 (%)', fontsize=12, color='#555')
    ax.set_title('AI 诊断结果分布', fontsize=14, fontweight='bold', color='#333', pad=15)
    ax.tick_params(axis='y', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#ddd')
    ax.spines['left'].set_color('#ddd')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='white', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def draw_pipeline_flow(steps: list, figsize=(14, 4)) -> Image.Image:
    """绘制技术流程图"""
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    n = len(steps)
    box_width = 1.8
    box_height = 1.2
    spacing = 2.8
    start_x = 1.0
    y_center = 2.0

    colors = ['#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#10b981']

    for i, step in enumerate(steps):
        x = start_x + i * spacing

        rect = mpatches.FancyBboxPatch(
            (x - box_width / 2, y_center - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=colors[i % len(colors)],
            edgecolor='white', linewidth=2, alpha=0.9
        )
        ax.add_patch(rect)

        ax.text(x, y_center + 0.25, step["icon"],
                fontsize=24, ha='center', va='center', color='white')
        ax.text(x, y_center - 0.15, f'步骤{i+1}',
                fontsize=9, ha='center', va='center', color='white', alpha=0.9)
        ax.text(x, y_center - 0.4, step["name"],
                fontsize=10, ha='center', va='center', color='white', fontweight='bold')

        if i < n - 1:
            ax.annotate('', xy=(x + spacing - box_width / 2 - 0.1, y_center),
                        xytext=(x + box_width / 2 + 0.1, y_center),
                        arrowprops=dict(arrowstyle='->', color='#aaa', lw=2))

    ax.set_xlim(0, start_x + n * spacing)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('AI眼底影像分析技术流程', fontsize=16, fontweight='bold', color='#333', pad=20)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='white', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def draw_performance_comparison(models: list, figsize=(10, 5)) -> Image.Image:
    """绘制模型性能对比图"""
    setup_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100)
    fig.patch.set_facecolor('white')

    model_names = [m["name"] for m in models]
    accs = [m["acc"] for m in models]
    aucs = [m["auc"] for m in models]

    colors = ['#a8d8ea', '#aa96da', '#fcbad3', '#ff8b94']

    # 左图：准确率
    bars1 = ax1.bar(model_names, accs, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylim(75, 95)
    ax1.set_ylabel('准确率 (%)', fontsize=11)
    ax1.set_title('分类准确率对比', fontsize=13, fontweight='bold', pad=10)
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val}%', ha='center', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15, labelsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 右图：AUC
    bars2 = ax2.bar(model_names, aucs, color=colors, edgecolor='white', linewidth=2)
    ax2.set_ylim(0.90, 1.0)
    ax2.set_ylabel('AUC', fontsize=11)
    ax2.set_title('AUC对比', fontsize=13, fontweight='bold', pad=10)
    for bar, val in zip(bars2, aucs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15, labelsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='white', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def draw_fundus_comparison(fundus_img: Image.Image, gradcam_img: Image.Image,
                           diagnosis: str, figsize=(10, 5)) -> Image.Image:
    """
    绘制原图与热力图对比

    Args:
        fundus_img: 眼底原图
        gradcam_img: Grad-CAM热力图
        diagnosis: 诊断结果
        figsize: 图像大小

    Returns:
        PIL Image
    """
    setup_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100)
    fig.patch.set_facecolor('white')

    ax1.imshow(fundus_img)
    ax1.set_title('眼底原图', fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')

    ax2.imshow(gradcam_img)
    ax2.set_title(f'Grad-CAM 热力图 → {diagnosis}', fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='white', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img
