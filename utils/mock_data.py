# -*- coding: utf-8 -*-
"""
慧眼 EyeCare - 演示数据模块
包含预设眼底病例、模拟诊断结果、疾病知识库等
"""

import random
import numpy as np
import math


# ============================================================
# 眼底疾病知识库
# ============================================================
DISEASE_DATABASE = {
    "normal": {
        "name": "正常眼底",
        "chinese_name": "正常眼底",
        "severity": 0,
        "risk_level": "低风险",
        "risk_color": "#10B981",
        "description": "眼底结构正常，视盘边界清晰，黄斑区形态良好，视网膜血管走形正常，未见明显异常。",
        "features": [
            "视盘边界清晰，色泽正常",
            "黄斑中心凹反光可见",
            "视网膜动脉静脉比例正常（约2:3）",
            "未见出血、渗出、新生血管等异常"
        ],
        "recommendation": "眼底状况良好，建议每年进行一次常规眼底检查，保持健康的生活方式。",
        "doctor_advice": "继续保持良好的用眼习惯，控制血糖、血压、血脂，定期复查即可。"
    },
    "dr_mild": {
        "name": "轻度糖尿病视网膜病变",
        "chinese_name": "轻度糖网",
        "severity": 1,
        "risk_level": "建议复查",
        "risk_color": "#F59E0B",
        "description": "可见微动脉瘤和/或少量点状出血，属于糖尿病视网膜病变早期阶段。",
        "features": [
            "可见微动脉瘤（视网膜内小红点）",
            "少量点状出血",
            "硬性渗出不明显",
            "黄斑区基本正常"
        ],
        "recommendation": "建议3-6个月内复查眼底，控制好血糖、血压和血脂，避免病情进展。",
        "doctor_advice": "这是糖网早期表现，目前不影响视力，但需要重视。严格控制血糖（HbA1c<7%），定期眼科随访，避免病情进展到增殖期。"
    },
    "dr_moderate": {
        "name": "中度糖尿病视网膜病变",
        "chinese_name": "中度糖网",
        "severity": 2,
        "risk_level": "建议就医",
        "risk_color": "#F97316",
        "description": "存在较多出血点、硬性渗出和棉絮斑，视网膜微血管损伤明显。",
        "features": [
            "多处视网膜内出血",
            "硬性渗出（黄色蜡样斑点）",
            "棉絮斑（软性渗出）",
            "视网膜微血管异常(IRMA)"
        ],
        "recommendation": "建议尽快到眼科就诊，可能需要眼底荧光造影检查，评估是否需要激光治疗。",
        "doctor_advice": "中度糖网需要密切关注。建议做眼底荧光血管造影(FFA)和OCT检查，评估黄斑水肿情况。根据病情可能需要抗VEGF治疗或激光光凝。"
    },
    "dr_severe": {
        "name": "重度糖尿病视网膜病变",
        "chinese_name": "重度糖网",
        "severity": 3,
        "risk_level": "需尽快就医",
        "risk_color": "#EF4444",
        "description": "广泛的视网膜内出血、棉絮斑，静脉串珠样改变，属于增殖前期，极易进展为增殖期糖网。",
        "features": [
            "四个象限均有大量视网膜内出血",
            "两个以上象限静脉串珠样改变",
            "一个以上象限视网膜微血管异常(IRMA)",
            "广泛棉絮斑，视网膜缺血明显"
        ],
        "recommendation": "请尽快到眼科专科就诊，建议行全视网膜激光光凝治疗，防止进展为增殖期糖网。",
        "doctor_advice": "重度糖网是视力丧失的高风险阶段。强烈建议尽快做眼底荧光血管造影，评估视网膜缺血范围。根据情况需要行全视网膜激光光凝(PRP)或抗VEGF治疗，严格控制血糖血压，密切随访。"
    },
    "dr_proliferative": {
        "name": "增殖期糖尿病视网膜病变",
        "chinese_name": "增殖期糖网",
        "severity": 4,
        "risk_level": "高危-紧急就医",
        "risk_color": "#DC2626",
        "description": "视网膜新生血管形成，可伴玻璃体出血、纤维增殖，是糖尿病致盲的主要原因。",
        "features": [
            "视盘或视网膜新生血管(NVD/NVE)",
            "玻璃体出血或视网膜前出血",
            "纤维血管增殖膜",
            "牵拉性视网膜脱离风险"
        ],
        "recommendation": "请立即到眼科就诊！增殖期糖网有致盲风险，需尽快行全视网膜激光光凝或手术治疗。",
        "doctor_advice": "增殖期糖网是糖尿病最严重的眼部并发症，有致盲风险。需立即进行全视网膜激光光凝(PRP)联合抗VEGF治疗。如已发生玻璃体出血或视网膜脱离，需行玻璃体切割手术。"
    }
}


# ============================================================
# 预设示例病例（用于演示模式）
# ============================================================
SAMPLE_CASES = [
    {
        "id": "normal_01",
        "name": "正常眼底影像",
        "category": "正常",
        "actual_diagnosis": "normal",
        "image_file": "正常眼底影像.png",
        "description": "健康成年人眼底，视盘边界清晰，黄斑区正常，视网膜血管走形良好，无出血、渗出等异常。",
        "patient_info": {"age": 35, "gender": "男", "symptom": "常规体检", "history": "无特殊病史"}
    },
    {
        "id": "dr_mild_01",
        "name": "轻度糖尿病",
        "category": "糖尿病视网膜病变",
        "actual_diagnosis": "dr_mild",
        "image_file": "轻度糖尿病.png",
        "description": "糖尿病病史，眼底可见微动脉瘤和少量点状出血，属于轻度非增殖期糖尿病视网膜病变。",
        "patient_info": {"age": 52, "gender": "女", "symptom": "糖尿病随访", "history": "2型糖尿病5年"}
    },
    {
        "id": "dr_moderate_01",
        "name": "中度糖尿病",
        "category": "糖尿病视网膜病变",
        "actual_diagnosis": "dr_moderate",
        "image_file": "中度糖尿病.png",
        "description": "糖尿病病史，眼底可见较多出血点和硬性渗出，黄斑区可能受累，属于中度非增殖期。",
        "patient_info": {"age": 58, "gender": "男", "symptom": "视力轻度下降", "history": "2型糖尿病8年"}
    },
    {
        "id": "dr_severe_01",
        "name": "重度糖尿病",
        "category": "糖尿病视网膜病变",
        "actual_diagnosis": "dr_severe",
        "image_file": "重度糖尿病.png",
        "description": "糖尿病病史，眼底广泛出血、棉绒斑，静脉串珠样改变，属于重度非增殖期，需密切随访。",
        "patient_info": {"age": 65, "gender": "女", "symptom": "视物模糊", "history": "2型糖尿病12年"}
    },
    {
        "id": "dr_proliferative_01",
        "name": "增殖期糖网",
        "category": "糖尿病视网膜病变",
        "actual_diagnosis": "dr_proliferative",
        "image_file": "增殖期糖网.png",
        "description": "糖尿病病史，眼底可见新生血管、纤维增殖膜，属于增殖期糖尿病视网膜病变，有失明风险。",
        "patient_info": {"age": 68, "gender": "男", "symptom": "视力严重下降", "history": "2型糖尿病15年"}
    },
]


# ============================================================
# 模拟眼底诊断（演示模式）
# ============================================================
def mock_fundus_analysis(case_id: str = None, uploaded_image=None, seed: int = None) -> dict:
    """
    模拟眼底影像分析结果

    Args:
        case_id: 预设病例ID
        uploaded_image: 上传的图片（演示模式忽略）
        seed: 随机种子，用于结果可复现（同一种子=同一结果）

    Returns:
        诊断结果字典
    """
    # 如果有种子，设置随机状态，保证结果可复现
    if seed is not None:
        rng_state = random.getstate()
        random.seed(seed)
        np_seed_state = np.random.get_state()
        np.random.seed(seed)

    try:
        if case_id:
            # 从预设病例中查找
            case = next((c for c in SAMPLE_CASES if c["id"] == case_id), SAMPLE_CASES[0])
            diagnosis_key = case["actual_diagnosis"]
        else:
            # 根据种子选择结果（上传图片时模拟）
            keys = list(DISEASE_DATABASE.keys())
            diagnosis_key = random.choice(keys)
            case = {
                "patient_info": {"age": random.randint(30, 75), "gender": random.choice(["男", "女"]),
                                 "symptom": "体检", "history": "无特殊"}
            }

        disease = DISEASE_DATABASE[diagnosis_key]

        # 生成置信度（根据严重程度略有差异）
        base_conf = {
            "normal": 0.96,
            "dr_mild": 0.92,
            "dr_moderate": 0.89,
            "dr_severe": 0.85,
            "dr_proliferative": 0.87,
        }
        confidence = base_conf.get(diagnosis_key, 0.85) + random.uniform(-0.02, 0.02)
        confidence = max(0.80, min(0.99, confidence))

        # 生成所有类别的预测概率
        all_probs = _generate_prediction_probabilities(diagnosis_key)

        # 生成热力图中心点（模拟Grad-CAM关注区域）
        heatmap_center = _get_heatmap_center(diagnosis_key)

        # 组装结果
        result = {
            "primary_diagnosis": disease["chinese_name"],
            "diagnosis_key": diagnosis_key,
            "severity": disease["severity"],
            "risk_level": disease["risk_level"],
            "risk_color": disease["risk_color"],
            "confidence": round(confidence, 4),
            "description": disease["description"],
            "features": disease["features"],
            "recommendation": disease["recommendation"],
            "doctor_advice": disease["doctor_advice"],
            "all_predictions": all_probs,
            "heatmap_center": heatmap_center,
            "patient_info": case.get("patient_info", {}),
            "analysis_time": "约2.3秒"
        }

    finally:
        # 恢复原随机状态
        if seed is not None:
            random.setstate(rng_state)
            np.random.set_state(np_seed_state)

    return result


def _generate_prediction_probabilities(primary_key: str) -> list:
    """生成所有类别的预测概率分布"""
    # 基础概率分布
    base_probs = {
        "正常眼底": 0.55,
        "轻度糖网": 0.15,
        "中度糖网": 0.15,
        "重度糖网": 0.08,
        "增殖期糖网": 0.07,
    }

    # 映射
    key_map = {
        "normal": "正常眼底",
        "dr_mild": "轻度糖网",
        "dr_moderate": "中度糖网",
        "dr_severe": "重度糖网",
        "dr_proliferative": "增殖期糖网",
    }

    primary_name = key_map.get(primary_key, "正常眼底")

    # 调整概率，让主诊断概率最高
    primary_prob = 0.75 + random.uniform(0.05, 0.15)
    remaining = 1.0 - primary_prob

    results = []
    other_keys = [k for k in base_probs.keys() if k != primary_name]

    # 给主诊断分配高概率
    results.append({"name": primary_name, "probability": round(primary_prob, 4), "is_primary": True})

    # 其他类别分配剩余概率
    for i, name in enumerate(other_keys):
        if i < len(other_keys) - 1:
            prob = remaining * random.uniform(0.2, 0.4)
            remaining -= prob
        else:
            prob = remaining
        results.append({"name": name, "probability": round(prob, 4), "is_primary": False})

    # 按概率排序
    results.sort(key=lambda x: x["probability"], reverse=True)

    return results


def _get_heatmap_center(diagnosis_key: str) -> tuple:
    """获取不同疾病的热力图中心位置（模拟Grad-CAM关注区域）"""
    # 归一化坐标 (x, y)，以眼底中心为原点
    heatmap_centers = {
        "normal": (0.5, 0.5),  # 整体均匀
        "dr_mild": (0.45, 0.55),  # 视网膜散在
        "dr_moderate": (0.5, 0.5),  # 广泛分布
        "dr_severe": (0.5, 0.5),  # 广泛出血
        "dr_proliferative": (0.48, 0.52),  # 新生血管区域
    }
    center = heatmap_centers.get(diagnosis_key, (0.5, 0.5))
    # 加一点随机扰动
    jitter = (random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03))
    return (center[0] + jitter[0], center[1] + jitter[1])


# ============================================================
# 技术流程数据
# ============================================================
def get_pipeline_steps() -> list:
    """获取眼底影像分析技术流程"""
    return [
        {
            "step": 1,
            "name": "图像采集",
            "description": "眼底相机或手机拍摄眼底图像",
            "icon": "📷",
            "detail": "支持眼底相机、智能手机+适配器等多种采集方式，分辨率≥1000×1000像素"
        },
        {
            "step": 2,
            "name": "图像预处理",
            "description": "归一化、增强、去噪、视盘黄斑定位",
            "icon": "🔧",
            "detail": "CLAHE对比度增强 + 绿色通道提取 + 视盘/黄斑检测定位 + 尺寸归一化"
        },
        {
            "step": 3,
            "name": "特征提取",
            "description": "CNN深度特征提取（ResNet/EfficientNet）",
            "icon": "🧠",
            "detail": "使用在ImageNet上预训练的EfficientNet-B5作为骨干网络，提取高层语义特征"
        },
        {
            "step": 4,
            "name": "分类预测",
            "description": "全连接层+Softmax输出各类别概率",
            "icon": "🎯",
            "detail": "迁移学习微调，多标签分类，支持多种眼底病变同时检测"
        },
        {
            "step": 5,
            "name": "可解释性",
            "description": "Grad-CAM热力图可视化关注区域",
            "icon": "🔥",
            "detail": "基于梯度的类激活映射，展示模型决策依据，提高医疗AI可信度"
        },
        {
            "step": 6,
            "name": "报告生成",
            "description": "生成结构化诊断报告和建议",
            "icon": "📋",
            "detail": "输出风险等级、异常发现、就医建议、随访计划，支持医生复核"
        }
    ]


# ============================================================
# 性能数据
# ============================================================
def get_performance_data() -> dict:
    """获取模型性能数据"""
    return {
        "datasets": [
            {"name": "EyePACS", "samples": 88702, "classes": 5, "type": "糖尿病视网膜病变分级"},
            {"name": "Messidor-2", "samples": 1748, "classes": 4, "type": "糖网分级（公开基准）"},
            {"name": "IDRiD", "samples": 516, "classes": 5, "type": "印度糖尿病视网膜病变数据集"},
            {"name": "OIA-ODIR", "samples": 7000, "classes": 8, "type": "中国眼科多病种数据集"},
        ],
        "metrics": [
            {"name": "AUC（糖网二分类）", "value": "0.978", "unit": ""},
            {"name": "准确率（5分类）", "value": "89.2%", "unit": ""},
            {"name": "敏感性", "value": "94.5%", "unit": ""},
            {"name": "特异性", "value": "91.3%", "unit": ""},
            {"name": "单图推理时间", "value": "85", "unit": "ms"},
            {"name": "支持病种", "value": "10+", "unit": "种"},
        ],
        "model_comparison": [
            {"name": "ResNet-50", "acc": 82.5, "auc": 0.942, "params": "25.6M"},
            {"name": "Inception-v3", "acc": 85.3, "auc": 0.956, "params": "23.8M"},
            {"name": "EfficientNet-B3", "acc": 87.8, "auc": 0.968, "params": "12.2M"},
            {"name": "EfficientNet-B5 (本系统)", "acc": 89.2, "auc": 0.978, "params": "30.4M"},
        ]
    }


# ============================================================
# 健康科普数据
# ============================================================
def get_health_education() -> list:
    """获取健康科普文章列表"""
    return [
        {
            "title": "什么是糖尿病视网膜病变？",
            "category": "糖尿病视网膜病变",
            "summary": "糖尿病视网膜病变是糖尿病最常见的微血管并发症，也是工作年龄人群失明的首要原因。长期高血糖导致视网膜微血管损伤，逐渐出现微动脉瘤、出血、渗出，最终可导致视力丧失。",
            "key_points": [
                "30%的糖尿病患者会出现糖网",
                "病程越长，风险越高（10年以上病程患病率达50%）",
                "早期可无明显症状，发现时可能已到晚期",
                "定期眼底检查是早发现的关键"
            ]
        },
        {
            "title": "如何预防糖尿病视网膜病变？",
            "category": "疾病预防",
            "summary": "糖尿病视网膜病变虽然可怕，但通过积极的预防和管理，可以显著降低发病风险、延缓病情进展。关键在于控制基础疾病、定期筛查和健康生活方式。",
            "key_points": [
                "严格控制血糖：糖化血红蛋白(HbA1c)控制在7%以下",
                "控制血压：目标血压<130/80mmHg",
                "控制血脂：降低低密度脂蛋白胆固醇",
                "戒烟限酒：吸烟会显著加速糖网进展",
                "定期眼底检查：确诊糖尿病后至少每年查一次"
            ]
        },
        {
            "title": "糖网患者日常护眼指南",
            "category": "疾病预防",
            "summary": "已经确诊糖尿病视网膜病变的患者，除了遵医嘱治疗外，日常生活中的护眼措施也非常重要，可以帮助稳定病情、保护视力。",
            "key_points": [
                "避免剧烈运动和重体力劳动（尤其是增殖期患者）",
                "控制用眼时间，每40分钟休息5-10分钟",
                "保持情绪稳定，避免血压剧烈波动",
                "饮食清淡，减少高盐高脂高糖摄入",
                "如出现视力骤降、黑影增多立即就医"
            ]
        },
        {
            "title": "日常如何保护眼底健康？",
            "category": "健康生活",
            "summary": "眼底健康与全身健康密切相关。保持健康的生活方式不仅对身体好，也能有效预防多种眼底疾病，守护清晰视界。",
            "key_points": [
                "均衡饮食：多吃深色蔬菜、鱼类、坚果，补充叶黄素和Omega-3",
                "适度运动：每周3-5次有氧运动，改善微循环",
                "充足睡眠：每天7-8小时，让眼睛得到充分休息",
                "防晒护眼：强光环境下佩戴防紫外线太阳镜",
                "减少蓝光：控制电子屏幕使用时间，睡前少看手机"
            ]
        },
        {
            "title": "哪些人需要特别注意眼底健康？",
            "category": "高危人群",
            "summary": "有些人群患眼底疾病的风险明显高于普通人，需要更加重视眼底检查和日常防护，做到早发现、早干预。",
            "key_points": [
                "糖尿病患者：确诊后立即查眼底，之后每半年至一年复查",
                "高血压患者：每年查眼底，监测视网膜血管变化",
                "高度近视人群（>600度）：每年散瞳查眼底",
                "50岁以上中老年人：每年常规眼底检查",
                "有家族眼病史者：定期筛查，密切关注"
            ]
        },
        {
            "title": "眼底检查：为什么每个人都该做？",
            "category": "健康科普",
            "summary": "眼底是人体唯一能直接观察到血管和神经的部位。一次眼底检查不仅能发现眼病，还能提示糖尿病、高血压、动脉硬化等全身疾病的风险，被称为'全身健康的窗口'。",
            "key_points": [
                "糖尿病患者建议每半年查一次眼底",
                "40岁以上人群建议每年检查",
                "高度近视患者建议每年散瞳查眼底",
                "眼底检查无创、快捷、费用不高",
                "AI辅助筛查让眼底检查更高效、更普及"
            ]
        }
    ]
