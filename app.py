# -*- coding: utf-8 -*-
"""
慧眼 EyeCare - AI眼底影像初筛助手
主应用入口
基于深度学习的眼底图像智能分析系统
"""


import streamlit as st
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.mock_data import (
    SAMPLE_CASES, DISEASE_DATABASE,
    get_pipeline_steps,
    get_performance_data, get_health_education
)
from utils.visualizer import (
    generate_sample_fundus_image, generate_gradcam_heatmap,
    draw_risk_gauge, draw_prediction_barplot,
    draw_pipeline_flow, draw_performance_comparison,
    draw_fundus_comparison
)
from utils.llm_helper import ask_eye_doctor
from model.predict import (
    generate_diagnosis_result, is_model_available, load_model,
    generate_diagnosis_result_with_gradcam
)


# ============================================================
# 工具函数
# ============================================================

SAMPLE_IMAGES_DIR = os.path.dirname(os.path.abspath(__file__))


def load_sample_image(image_file, size=None):
    """
    加载示例真实图片

    参数:
        image_file: 图片文件名
        size: 目标尺寸（等比例缩放裁剪到正方形），None则返回原图

    返回:
        PIL.Image
    """
    img_path = os.path.join(SAMPLE_IMAGES_DIR, image_file)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        if size:
            # 等比例裁剪到正方形
            from PIL import ImageOps
            img = ImageOps.fit(img, (size, size), Image.LANCZOS)
        return img
    else:
        # fallback：生成模拟图
        return generate_sample_fundus_image("normal", size=size if size else 300)


def get_sample_image_by_disease(disease_key, size=None):
    """根据疾病key获取对应的示例图片"""
    key_map = {
        "normal": "正常眼底影像.png",
        "dr_mild": "轻度糖尿病.png",
        "dr_moderate": "中度糖尿病.png",
        "dr_severe": "重度糖尿病.png",
        "dr_proliferative": "增殖期糖网.png",
    }
    image_file = key_map.get(disease_key)
    if image_file:
        return load_sample_image(image_file, size=size)
    else:
        return generate_sample_fundus_image(disease_key, size=size if size else 250)


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="慧眼 EyeCare - AI眼底影像初筛助手",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "page" not in st.session_state:
    st.session_state["page"] = "首页"


# ============================================================
# 自定义 CSS
# ============================================================

st.markdown("""
<style>
    /* 全局背景 */
    .main {
        background: linear-gradient(180deg, #F0F9FF 0%, #F8FAFC 200px, #F8FAFC 100%);
    }
    
    .main .block-container { 
        padding-top: 0;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0C4A6E 0%, #075985 50%, #0369A1 100%);
    }
    [data-testid="stSidebar"] * {
        color: #F0F9FF !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown div {
        color: #F0F9FF !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        transition: all 0.3s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.25) !important;
        border-color: rgba(255,255,255,0.5) !important;
        transform: translateX(4px) !important;
    }
    
    /* 侧边栏眼睛图标动画 */
    .sidebar-eye {
        display: inline-block;
        animation: eyeBlink 4s ease-in-out infinite;
    }
    @keyframes eyeBlink {
        0%, 45%, 55%, 100% {
            transform: scaleY(1);
        }
        50% {
            transform: scaleY(0.1);
        }
    }
    .sidebar-logo {
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-8px);
        }
    }

    /* 渐变标题 */
    .gradient-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0891B2 0%, #0284C7 50%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Hero 区域大标题 */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(135deg, #0C4A6E 0%, #0369A1 50%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: #475569;
        line-height: 1.7;
        margin-top: 1rem;
    }

    /* 普通卡片 */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.03);
        transition: all 0.3s;
    }
    .card:hover {
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.05);
        transform: translateY(-2px);
    }

    /* 功能卡片 */
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 1px solid #E2E8F0;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        box-shadow: 0 20px 25px -5px rgba(8, 145, 178, 0.15), 0 8px 10px -6px rgba(8, 145, 178, 0.1);
        transform: translateY(-4px);
        border-color: #0891B2;
    }
    .feature-icon {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #E0F2FE, #BAE6FD);
        margin-bottom: 1rem;
    }

    /* 亮点卡片 */
    .highlight-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: all 0.3s;
    }
    .highlight-card:hover {
        border-color: #38BDF8;
        box-shadow: 0 8px 16px rgba(56, 189, 248, 0.15);
    }
    .highlight-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }

    /* 标签 */
    .tag {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .tag-primary { background: #E0F2FE; color: #0369a1; }
    .tag-success { background: #D1FAE5; color: #059669; }
    .tag-warning { background: #FEF3C7; color: #D97706; }
    .tag-danger { background: #FEE2E2; color: #DC2626; }
    .tag-purple { background: #EDE9FE; color: #7C3AED; }
    .tag-teal { background: #CCFBF1; color: #0F766E; }

    /* 风险等级 */
    .risk-low { background: linear-gradient(135deg, #10B981, #059669); }
    .risk-mid { background: linear-gradient(135deg, #F59E0B, #D97706); }
    .risk-high { background: linear-gradient(135deg, #EF4444, #DC2626); }

    /* 结果卡片 */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .result-level {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    /* 指标卡片 */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem 1rem;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: all 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
    }
    .metric-value { 
        font-size: 1.75rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #0891B2, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label { 
        font-size: 0.85rem; 
        color: #64748B; 
        margin-top: 0.5rem;
        font-weight: 500;
    }

    .feature-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0891B2, #2563EB);
        margin-right: 8px;
        vertical-align: middle;
    }

    /* 区块标题 */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    .section-subtitle {
        color: #64748B;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* 隐藏默认元素 */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Streamlit按钮美化 */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(8, 145, 178, 0.25) !important;
    }
    
    /* 主按钮样式 */
    .stButton [kind="primary"] > button {
        background: linear-gradient(135deg, #0891B2, #2563EB) !important;
        border: none !important;
    }
    .stButton [kind="primary"] > button:hover {
        background: linear-gradient(135deg, #0E7490, #1D4ED8) !important;
    }
    
    /* 上传区域美化 */
    [data-testid="stFileUploader"] {
        border-radius: 16px !important;
        border: 2px dashed #BAE6FD !important;
        background: #F0F9FF !important;
    }
    
    /* selectbox 美化 */
    .stSelectbox > div > div > div {
        border-radius: 12px !important;
        border-color: #E2E8F0 !important;
    }
    
    /* expander 美化 */
    .streamlit-expanderHeader {
        border-radius: 12px !important;
        background: white !important;
        border: 1px solid #E2E8F0 !important;
    }
    .streamlit-expanderContent {
        background: #FAFBFC !important;
        border-radius: 0 0 12px 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div class="sidebar-logo" style="font-size:3.5rem; margin-bottom:0.5rem;">
            <span class="sidebar-eye">👁️</span>
        </div>
        <div style="font-size:1.35rem; font-weight:800;
                    color: #FFFFFF;
                    text-shadow: 0 2px 8px rgba(0,0,0,0.2);">
            慧眼 EyeCare
        </div>
        <div style="font-size:0.8rem; color:#BAE6FD; margin-top:0.4rem; font-weight:500;">
            AI眼底影像智能筛查助手
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📋 功能导航**")

    pages = [
        "🏠 首页",
        "🔬 AI眼底筛查",
        "📚 疾病知识库",
        "💡 健康科普",
    ]

    for p in pages:
        if st.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state["page"] = p
            st.rerun()


# ============================================================
# 页面：首页
# ============================================================

def render_home():
    # ====== Hero 区域 ======
    st.markdown("""
    <div style="padding: 3rem 0 2rem 0; text-align:center;">
        <div style="display:inline-block; padding:0.5rem 1.25rem; background:#E0F2FE; 
                    border-radius:9999px; color:#0369A1; font-size:0.9rem; font-weight:600;
                    margin-bottom:1.5rem;">
            🔬 AI驱动的眼底影像智能筛查平台
        </div>
        <div class="hero-title">
            让每一双眼睛<br>都能被温柔守护
        </div>
        <p class="hero-subtitle" style="max-width:600px; margin:1rem auto 0 auto;">
            基于深度学习的糖尿病视网膜病变智能筛查助手<br>
            上传眼底照片，10秒获取糖网5级风险分级和专业医学建议
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 居中的开始按钮
    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        if st.button("🚀 立即体验筛查", use_container_width=True, type="primary", key="home_start"):
            st.session_state["page"] = "🔬 AI眼底筛查"
            st.rerun()

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    # ====== 三大核心功能 ======
    st.markdown('<p class="section-title" style="text-align:center;">💡 核心功能</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle" style="text-align:center;">一站式眼底健康管理，从智能筛查到健康科普全覆盖</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3 style="margin:0.5rem 0 0.25rem 0; font-size:1.2rem;">AI智能筛查</h3>
            <p style="color:#64748B; margin:0.75rem 0; font-size:0.9rem; line-height:1.6;">
                上传眼底照片，AI自动分析糖网等级
                支持5级分类，附Grad-CAM可解释热力图
            </p>
            <div style="margin-top:1rem;">
                <span class="tag tag-primary">5级分类</span>
                <span class="tag tag-purple">Grad-CAM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("开始筛查", use_container_width=True, key="home_screen"):
            st.session_state["page"] = "🔬 AI眼底筛查"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <h3 style="margin:0.5rem 0 0.25rem 0; font-size:1.2rem;">疾病知识库</h3>
            <p style="color:#64748B; margin:0.75rem 0; font-size:0.9rem; line-height:1.6;">
                糖尿病视网膜病变5级分类详解
                了解症状、危害、治疗和随访建议
            </p>
            <div style="margin-top:1rem;">
                <span class="tag tag-warning">5级糖网</span>
                <span class="tag tag-primary">医学科普</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("浏览百科", use_container_width=True, key="home_dict"):
            st.session_state["page"] = "📚 疾病知识库"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <h3 style="margin:0.5rem 0 0.25rem 0; font-size:1.2rem;">健康科普</h3>
            <p style="color:#64748B; margin:0.75rem 0; font-size:0.9rem; line-height:1.6;">
                糖网预防、日常护眼、高危人群指南
                科学护眼，守护清晰视界
            </p>
            <div style="margin-top:1rem;">
                <span class="tag tag-success">预防指南</span>
                <span class="tag tag-teal">护眼知识</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("阅读科普", use_container_width=True, key="home_edu"):
            st.session_state["page"] = "💡 健康科普"
            st.rerun()

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    # ====== 产品亮点 ======
    st.markdown('<p class="section-title" style="text-align:center;">⭐ 产品亮点</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle" style="text-align:center;">专业、高效、可解释的AI眼底筛查方案</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    highlights = [
        {"icon": "🎯", "title": "精准分级", "desc": "MobileNetV2深度学习模型，5级糖网精准分类"},
        {"icon": "🔥", "title": "可解释AI", "desc": "Grad-CAM热力图可视化，透明展示AI决策依据"},
        {"icon": "⚡", "title": "快速高效", "desc": "CPU推理仅需数秒，基层场景也能快速出结果"},
        {"icon": "🏥", "title": "临床价值", "desc": "辅助基层初筛，提高糖网早诊率，降低致盲风险"},
    ]

    for i, h in enumerate(highlights):
        with [c1, c2, c3, c4][i]:
            st.markdown(f"""
            <div class="highlight-card">
                <div class="highlight-icon">{h['icon']}</div>
                <h4 style="margin:0.25rem 0; font-size:1.05rem;">{h['title']}</h4>
                <p style="color:#64748B; font-size:0.85rem; margin:0.5rem 0 0 0; line-height:1.6;">
                    {h['desc']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    # ====== 性能数据 ======
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0891B2 0%, #0E7490 50%, #1E40AF 100%);
                border-radius: 24px;
                padding: 2.5rem 2rem;
                margin: 1rem 0;
                color: white;
                text-align: center;">
        <h3 style="color:white; margin:0 0 0.5rem 0; font-size:1.5rem;">📊 模型性能数据</h3>
        <p style="color:rgba(255,255,255,0.8); margin-bottom:2rem;">基于APTOS 2019眼底数据集训练验证</p>
    </div>
    """, unsafe_allow_html=True)

    # 数据指标卡片（叠加在渐变背景下面，白色卡片）
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:1.5rem 1rem; 
                    text-align:center; margin-top:-3rem; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
            <div style="font-size:2rem; font-weight:800; 
                        background:linear-gradient(135deg,#0891B2,#2563EB);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                72.13%
            </div>
            <div style="color:#64748B; font-size:0.9rem; margin-top:0.5rem; font-weight:500;">
                验证集准确率
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:1.5rem 1rem; 
                    text-align:center; margin-top:-3rem; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
            <div style="font-size:2rem; font-weight:800; 
                        background:linear-gradient(135deg,#0891B2,#2563EB);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                2,930
            </div>
            <div style="color:#64748B; font-size:0.9rem; margin-top:0.5rem; font-weight:500;">
                训练样本数
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:1.5rem 1rem; 
                    text-align:center; margin-top:-3rem; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
            <div style="font-size:2rem; font-weight:800; 
                        background:linear-gradient(135deg,#0891B2,#2563EB);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                5级
            </div>
            <div style="color:#64748B; font-size:0.9rem; margin-top:0.5rem; font-weight:500;">
                糖网分级
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:1.5rem 1rem; 
                    text-align:center; margin-top:-3rem; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
            <div style="font-size:1.6rem; font-weight:800; 
                        background:linear-gradient(135deg,#0891B2,#2563EB);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                MobileNetV2
            </div>
            <div style="color:#64748B; font-size:0.9rem; margin-top:0.5rem; font-weight:500;">
                骨干网络
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)


# ============================================================
# 页面：AI眼底筛查
# ============================================================

def render_screening():
    st.markdown("## 🔬 AI眼底筛查")
    st.markdown("上传眼底照片，AI自动分析并给出风险评估")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 📷 选择眼底图像")

        # 方式一：预设示例
        st.markdown("#### 方式一：选择示例图片")
        case_options = {c["name"]: c["id"] for c in SAMPLE_CASES}
        selected_case_label = st.selectbox("选择示例眼底图", list(case_options.keys()))
        case_id = case_options[selected_case_label]

        # 显示示例缩略图
        case = next((c for c in SAMPLE_CASES if c["id"] == case_id), SAMPLE_CASES[0])
        case_key = case["actual_diagnosis"]
        sample_img = load_sample_image(case.get("image_file", ""), size=300)
        st.image(sample_img, caption=case["description"], width=200)

        st.markdown("#### 方式二：上传图片")
        uploaded_file = st.file_uploader("上传眼底照片（支持 JPG/PNG 格式）",
                                          type=["jpg", "jpeg", "png"])

        if st.button("🔍 开始AI分析", use_container_width=True, type="primary"):
            with st.spinner("AI正在分析眼底图像..."):
                if uploaded_file:
                    # 上传的图：读取并调整大小
                    from PIL import ImageOps
                    fundus_img = Image.open(uploaded_file).convert("RGB")
                    fundus_img = ImageOps.fit(fundus_img, (400, 400))
                    use_uploaded = True
                else:
                    fundus_img = load_sample_image(case.get("image_file", ""), size=400)
                    use_uploaded = False

                # 使用真实AI模型进行诊断
                if is_model_available():
                    result, detected_key, gradcam_img = generate_diagnosis_result_with_gradcam(fundus_img)
                else:
                    # 降级：如果模型不可用，使用模拟数据
                    from utils.mock_data import mock_fundus_analysis
                    result = mock_fundus_analysis(case_id if not use_uploaded else None)
                    detected_key = case_key
                    # 生成模拟热力图
                    heatmap_center = result["heatmap_center"]
                    gradcam_img = generate_gradcam_heatmap(fundus_img, heatmap_center,
                                                           intensity=0.7, case_type=detected_key)

                st.session_state["screening_result"] = {
                    "fundus_img": fundus_img,
                    "gradcam_img": gradcam_img,
                    "result": result,
                    "case_key": detected_key,
                    "use_uploaded": use_uploaded,
                    "is_real_model": is_model_available(),
                }

    with col2:
        st.markdown("### 📊 分析结果")

        if "screening_result" not in st.session_state:
            st.info("👈 请选择示例图片或上传图片，然后点击「开始AI分析」")
        else:
            data = st.session_state["screening_result"]
            result = data["result"]
            risk_color = result["risk_color"]
            risk_level = result["risk_level"]
            conf = round(result["confidence"] * 100, 1)

            # 风险等级卡片
            risk_class = "risk-low" if result["severity"] == 0 else \
                         "risk-mid" if result["severity"] == 1 else "risk-high"

            st.markdown(f"""
            <div class="result-card {risk_class}">
                <div style="font-size:0.9rem; opacity:0.9;">AI诊断：{result['primary_diagnosis']}</div>
                <div class="result-level">{risk_level}</div>
                <div style="opacity:0.9;">置信度：{conf}%</div>
            </div>
            """, unsafe_allow_html=True)

            # 仪表盘
            risk_score = 0.1 if result["severity"] == 0 else \
                         0.5 if result["severity"] == 1 else 0.85
            # 仪表盘里用英文，避免Linux服务器中文字体乱码
            gauge_risk_label = "LOW RISK" if result["severity"] == 0 else \
                               "MODERATE RISK" if result["severity"] == 1 else "HIGH RISK"
            gauge_img = draw_risk_gauge(gauge_risk_label, risk_score, risk_color, figsize=(5, 2.5))
            st.image(gauge_img, use_container_width=True)

    # 下方：详细结果
    if "screening_result" in st.session_state:
        st.markdown("---")
        data = st.session_state["screening_result"]
        result = data["result"]

        st.markdown("### 🔬 影像分析")

        # 原图 vs 热力图对比
        # 热力图标题用英文，避免matplotlib中文乱码
        en_diag_map = {
            "正常": "Normal",
            "正常眼底": "Normal",
            "轻度糖尿病视网膜病变": "Mild DR",
            "轻度糖网": "Mild DR",
            "中度糖尿病视网膜病变": "Moderate DR",
            "中度糖网": "Moderate DR",
            "重度糖尿病视网膜病变": "Severe DR",
            "重度糖网": "Severe DR",
            "增殖期糖尿病视网膜病变": "Proliferative DR",
            "增殖期糖网": "Proliferative DR",
        }
        en_diagnosis = en_diag_map.get(result["primary_diagnosis"], result["primary_diagnosis"])
        comp_img = draw_fundus_comparison(
            data["fundus_img"], data["gradcam_img"],
            en_diagnosis, figsize=(10, 4.5)
        )
        st.image(comp_img, use_container_width=True)

        st.caption("左：眼底原图 | 右：Grad-CAM热力图（红色区域为AI重点关注区域）")

        st.markdown("### 📋 异常发现")
        for feat in result["features"]:
            st.markdown(f"""
            <div style="padding:0.5rem 1rem; background:#F8FAFC; border-radius:8px; margin-bottom:0.5rem;
                        border-left:4px solid #06b6d4;">
                <span class="feature-dot"></span>{feat}
            </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 📈 诊断概率分布")
            # 把类别名转成英文，避免matplotlib中文乱码
            en_name_map = {
                "正常": "Normal",
                "轻度糖尿病视网膜病变": "Mild DR",
                "中度糖尿病视网膜病变": "Moderate DR",
                "重度糖尿病视网膜病变": "Severe DR",
                "增殖期糖尿病视网膜病变": "Proliferative DR",
                "正常眼底": "Normal",
                "轻度糖网": "Mild DR",
                "中度糖网": "Moderate DR",
                "重度糖网": "Severe DR",
                "增殖期糖网": "Proliferative DR",
            }
            en_predictions = []
            for p in result["all_predictions"]:
                name = p.get("class", p.get("name", ""))
                en_name = en_name_map.get(name, name)
                en_predictions.append({
                    "name": en_name,
                    "probability": p.get("probability", p.get("prob", 0)),
                    "is_primary": p.get("is_primary", False)
                })
            bar_img = draw_prediction_barplot(en_predictions, figsize=(8, 4.5))
            st.image(bar_img, use_container_width=True)

        with c2:
            st.markdown("### 💊 医生建议")
            st.markdown(f"""
            <div class="card" style="border-left:4px solid {result['risk_color']};">
                <p><strong>结论：</strong>{result['description']}</p>
                <p><strong>建议：</strong>{result['recommendation']}</p>
                <p><strong>专业解读：</strong>{result['doctor_advice']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.warning("⚠️ **免责声明**：本系统仅供初筛参考，不能替代专业医生诊断。如有异常，请及时到正规医院眼科就诊。")


# ============================================================
# 页面：疾病知识库
# ============================================================

def render_disease_library():
    st.markdown("## 📚 疾病知识库")
    st.markdown("常见眼底疾病科普与知识")
    st.markdown("---")

    # 分类筛选
    categories = ["全部", "正常", "糖尿病视网膜病变"]
    cat_filter = st.selectbox("疾病分类", categories)

    # 映射
    cat_map = {
        "正常": ["normal"],
        "糖尿病视网膜病变": ["dr_mild", "dr_moderate", "dr_severe", "dr_proliferative"],
    }

    filtered_keys = list(DISEASE_DATABASE.keys())
    if cat_filter != "全部":
        filtered_keys = cat_map.get(cat_filter, [])

    # 展示
    for key in filtered_keys:
        disease = DISEASE_DATABASE[key]
        risk_bg = disease["risk_color"]

        with st.expander(f"👁️ {disease['chinese_name']} — {disease['name']}", expanded=False):
            c1, c2 = st.columns([3, 1])

            with c1:
                risk_tag = f'<span class="tag tag-danger" style="background:#FEE2E2; color:#DC2626;">{disease["risk_level"]}</span>' \
                    if disease["severity"] >= 2 else \
                    f'<span class="tag tag-warning" style="background:#FEF3C7; color:#D97706;">{disease["risk_level"]}</span>' \
                    if disease["severity"] == 1 else \
                    f'<span class="tag tag-success" style="background:#D1FAE5; color:#059669;">{disease["risk_level"]}</span>'

                st.markdown(f"""
                <div>
                    {risk_tag}
                    <span class="tag tag-primary">严重程度：{'⭐' * (disease['severity'] + 1)}</span>
                </div>
                <p style="margin-top:0.75rem; line-height:1.8;">
                    <strong>疾病概述：</strong>{disease['description']}
                </p>
                """, unsafe_allow_html=True)

                st.markdown("**主要特征：**")
                for f in disease["features"]:
                    st.markdown(f"- {f}")

                st.markdown(f"""
                <div style="margin-top:1rem; padding:1rem; background:#EFF6FF; border-radius:8px;">
                    <strong>💡 建议：</strong>{disease['recommendation']}
                </div>
                """, unsafe_allow_html=True)

            with c2:
                # 显示对应眼底图
                fundus_img = get_sample_image_by_disease(key, size=250)
                st.image(fundus_img, caption="眼底影像图", use_container_width=True)


# ============================================================
# 页面：健康科普
# ============================================================

def render_health_education():
    st.markdown("## 💡 健康科普")
    st.markdown("了解眼底健康知识，守护光明")
    st.markdown("---")

    # ====== AI眼科医生问答 ======
    st.markdown("### 🤖 AI眼科医生在线问答")
    st.markdown("有任何眼部健康问题？问问AI小助手吧～")

    # API Key 输入（本地运行时手动输入，部署到Streamlit Cloud用secrets）
    try:
        api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    except:
        api_key = ""

    if not api_key:
        with st.expander("🔑 点击输入API Key（可选，不输入使用演示模式）", expanded=False):
            api_key_input = st.text_input("DeepSeek API Key", type="password", key="llm_api_key")
            if api_key_input:
                api_key = api_key_input
                st.success("API Key已设置！")
            else:
                st.info("💡 不输入API Key将使用演示模式，回答为预设内容。")

    # 快速提问按钮
    st.markdown("**快速提问：**")
    quick_questions = [
        "糖尿病患者怎么预防糖网？",
        "眼睛经常干涩怎么办？",
        "黄斑变性怎么预防？",
        "眼底检查都查什么？"
    ]
    cols = st.columns(2)
    for i, q in enumerate(quick_questions):
        if cols[i % 2].button(q, key=f"quick_q_{i}", use_container_width=True):
            st.session_state["llm_question"] = q

    # 输入框
    question = st.text_input("输入你的问题：", 
                            value=st.session_state.get("llm_question", ""),
                            placeholder="比如：糖网早期有什么症状？",
                            key="llm_input")

    if st.button("🔍 提问", type="primary", key="llm_submit"):
        if question.strip():
            with st.spinner("AI小助手正在思考..."):
                answer = ask_eye_doctor(question, api_key=api_key)
            
            st.markdown("#### 📝 AI小助手的回答")
            st.markdown(f"""
            <div style="padding:1rem 1.25rem; background:#F0FDF4; border-radius:12px;
                        border-left:4px solid #10B981; line-height:1.8;">
                {answer.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            st.caption("⚠️ 以上内容仅供健康科普参考，不能替代专业医疗诊断。如有不适请及时就医。")
        else:
            st.warning("请输入你的问题～")

    st.markdown("---")

    # ====== 科普文章 ======
    st.markdown("### 📖 健康科普文章")
    articles = get_health_education()

    for i, article in enumerate(articles):
        with st.expander(f"📖 {article['title']}", expanded=(i == 0)):
            st.markdown(f"""
            <span class="tag tag-primary">{article['category']}</span>
            <p style="margin-top:1rem; line-height:1.8;">{article['summary']}</p>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔑 关键要点")
            for point in article["key_points"]:
                st.markdown(f"""
                <div style="padding:0.5rem 1rem; background:#F8FAFC; border-radius:8px; margin-bottom:0.5rem;
                            border-left:4px solid #10B981;">
                    ✓ {point}
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# 主入口
# ============================================================

page = st.session_state.get("page", "🏠 首页")

if "首页" in page:
    render_home()
elif "筛查" in page:
    render_screening()
elif "知识库" in page:
    render_disease_library()
elif "科普" in page:
    render_health_education()
else:
    render_home()
