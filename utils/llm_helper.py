"""
大模型工具 - 调用DeepSeek API实现AI眼科问答
使用方式：在Streamlit Secrets中配置 DEEPSEEK_API_KEY，或用户手动输入
"""
import requests
import json


def ask_eye_doctor(question, api_key=None, model="deepseek-chat"):
    """
    调用DeepSeek大模型，回答眼科相关问题
    
    Args:
        question: 用户问题
        api_key: DeepSeek API Key，如果为None则返回模拟回答
        model: 模型名称
    
    Returns:
        answer: 回答文本
    """
    # 如果没有API key，返回模拟回答（用于演示）
    if not api_key:
        return _mock_answer(question)
    
    # 系统提示词：限定眼科领域，回答专业且通俗易懂
    system_prompt = """你是一位专业的眼科AI助手，名叫"慧眼小助手"。
你的职责是回答用户关于眼部健康、眼底疾病、糖网筛查、护眼知识等方面的问题。

回答要求：
1. 语言通俗易懂，避免过于专业的术语，必要时加以解释
2. 回答结构清晰，分点说明
3. 每次回答控制在300字以内
4. 如果涉及严重疾病或紧急情况，建议用户及时就医
5. 不要给出具体的药物处方或诊断结论，只提供健康科普和建议
6. 语气亲切、有温度，像一位贴心的家庭医生"""

    try:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"抱歉，AI助手暂时无法回答（{str(e)}）。请稍后再试，或检查API Key是否正确。"


def _mock_answer(question):
    """模拟回答（无API key时使用，用于演示）"""
    q = question.lower()
    
    if "糖网" in q or "糖尿病视网膜" in q:
        return """糖尿病视网膜病变（简称"糖网"）是糖尿病最常见的微血管并发症。

🔍 简单来说：长期高血糖会损伤眼底的微小血管，导致血管渗漏、出血、新生血管增生等问题，严重时可致盲。

💡 预防建议：
1. 控好血糖、血压、血脂是根本
2. 糖尿病患者每年至少做一次眼底检查
3. 发现视力下降、视物变形及时就医
4. 戒烟限酒，适度运动

如果已经确诊糖网，请遵医嘱定期复查，必要时进行激光治疗或抗VEGF药物注射。"""
    
    elif "护眼" in q or "保护眼睛" in q or "疲劳" in q:
        return """日常护眼记住这几点：

👀 20-20-20法则：每用眼20分钟，看20英尺（约6米）外的物体20秒
💧 多眨眼：长时间看屏幕眨眼次数减少，容易干眼
🌞 户外防晒：强光下戴太阳镜，防紫外线损伤
🥗 饮食均衡：多吃深色蔬菜、鱼类，补充叶黄素和Omega-3
😴 充足睡眠：让眼睛得到充分休息
📱 控制屏幕时间：睡前1小时尽量不玩手机

如果眼干、眼涩严重，可以用人工泪液缓解，严重时建议就医。"""
    
    elif "青光眼" in q:
        return """青光眼是一种以视神经损伤为特征的眼病，通常与眼压升高有关，被称为"视力的隐形小偷"。

⚠️ 高危人群：
- 40岁以上人群
- 有青光眼家族史
- 高度近视
- 糖尿病患者
- 长期使用激素药物

💡 建议：
1. 40岁后每年查眼压和眼底
2. 高危人群每半年检查一次
3. 出现眼胀、头痛、视力模糊及时就医

早期发现、早期治疗是保住视力的关键！"""
    
    elif "黄斑" in q:
        return """黄斑是眼底视网膜最中心的区域，负责我们的中心视力和精细视觉，比如看书、看人脸都靠它。

⚠️ 黄斑变性高危因素：
- 年龄（50岁以上风险增加）
- 吸烟
- 高血压
- 长期紫外线照射
- 家族遗传

💡 预防建议：
1. 戒烟
2. 外出戴太阳镜
3. 多吃深色蔬菜、坚果
4. 50岁后定期查眼底

如果发现视物变形、中心暗点，要赶紧去眼科检查！"""
    
    elif "饮食" in q or "吃什么" in q:
        return """对眼睛好的食物：

🥬 深色蔬菜：菠菜、西兰花、玉米（含叶黄素、玉米黄质）
🐟 深海鱼：三文鱼、金枪鱼（含Omega-3）
🥚 鸡蛋：叶黄素、锌
🍊 柑橘类水果：维生素C
🥜 坚果：维生素E
🥕 胡萝卜：β-胡萝卜素

💡 提醒：饮食均衡最重要，不要过度依赖保健品。如果已经有眼部疾病，请遵医嘱治疗，饮食只是辅助。"""
    
    elif "检查" in q or "筛查" in q:
        return """眼底检查常用的几种方式：

🔍 直接眼底镜检查：最基础，医生用仪器直接看眼底
📷 眼底照相：拍一张眼底照片，方便留存对比
🔬 OCT（光学相干断层扫描）：看视网膜各层结构，很精细
💧 荧光素眼底血管造影：看血管渗漏情况（需要打针）

💡 糖网筛查建议：
- 确诊糖尿病后立即查眼底
- 正常的话每年查一次
- 已经有糖网的，根据严重程度每3-6个月查一次

早发现、早治疗，能有效降低致盲风险！"""
    
    else:
        return f"""你问的是"**{question}**"，这是一个很好的问题！

作为AI眼科助手，我可以帮你解答：
- 糖尿病视网膜病变相关知识
- 日常护眼方法
- 眼底疾病科普
- 高危人群注意事项
- 饮食与护眼
- 眼底检查相关问题

你可以试着问我：
• "糖尿病患者怎么预防糖网？"
• "眼睛经常干涩怎么办？"
• "黄斑变性怎么预防？"

如果有具体的眼部不适，建议及时到医院眼科就诊哦～"""
