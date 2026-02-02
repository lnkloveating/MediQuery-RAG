"""
src/tools.py
医学计算工具库
包含: BMI、血压、理想体重、每日热量、心率区间计算
"""
from langchain_core.tools import tool # 建议用 langchain_core
from typing import Optional

# --- 1. 定义工具函数 ---

@tool
def calculate_bmi(height_cm: float, weight_kg: float) -> str:
    """
    计算BMI（身体质量指数）
    参数:
        height_cm: 身高（厘米）
        weight_kg: 体重（公斤）
    返回:
        BMI值和健康状态评估
    """
    try:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        # 判断健康状态
        if bmi < 18.5:
            status = "偏瘦"
            advice = "建议增加营养摄入，适当增重"
        elif 18.5 <= bmi < 24:
            status = "正常"
            advice = "保持良好的生活习惯"
        elif 24 <= bmi < 28:
            status = "超重"
            advice = "建议控制饮食，增加运动"
        else:
            status = "肥胖"
            advice = "建议就医咨询，制定减重计划"
        
        return f"BMI: {bmi:.2f}\n状态: {status}\n建议: {advice}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def calculate_blood_pressure_risk(systolic: int, diastolic: int) -> str:
    """
    评估血压风险等级
    参数:
        systolic: 收缩压（高压）
        diastolic: 舒张压（低压）
    返回:
        血压等级和风险评估
    """
    try:
        if systolic < 120 and diastolic < 80:
            level = "正常血压"
            risk = "低风险"
            advice = "保持健康生活方式"
        elif systolic < 130 and diastolic < 80:
            level = "正常高值"
            risk = "轻度风险"
            advice = "注意饮食，减少盐摄入"
        elif systolic < 140 or diastolic < 90:
            level = "1级高血压"
            risk = "中度风险"
            advice = "建议就医，可能需要药物治疗"
        elif systolic < 160 or diastolic < 100:
            level = "2级高血压"
            risk = "高风险"
            advice = "需要就医，进行规范治疗"
        else:
            level = "3级高血压"
            risk = "极高风险"
            advice = "立即就医！需要紧急干预"
        
        return f"血压: {systolic}/{diastolic} mmHg\n等级: {level}\n风险: {risk}\n建议: {advice}"
    except Exception as e:
        return f"评估错误: {str(e)}"


@tool
def calculate_ideal_weight(height_cm: float, gender: str) -> str:
    """
    计算理想体重范围
    参数:
        height_cm: 身高（厘米）
        gender: 性别（"男" 或 "女"）
    返回:
        理想体重范围
    """
    try:
        height_m = height_cm / 100
        
        if gender in ["男", "male", "m", "男性"]:
            # 男性：理想BMI 22
            ideal_weight = 22 * (height_m ** 2)
            min_weight = 18.5 * (height_m ** 2)
            max_weight = 24 * (height_m ** 2)
        else:
            # 女性：理想BMI 21
            ideal_weight = 21 * (height_m ** 2)
            min_weight = 18.5 * (height_m ** 2)
            max_weight = 24 * (height_m ** 2)
        
        return f"理想体重: {ideal_weight:.1f} kg\n健康范围: {min_weight:.1f} - {max_weight:.1f} kg"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def calculate_daily_calories(weight_kg: float, height_cm: float, age: int, gender: str, activity_level: str) -> str:
    """
    计算每日所需热量
    参数:
        weight_kg: 体重（公斤）
        height_cm: 身高（厘米）
        age: 年龄
        gender: 性别（"男" 或 "女"）
        activity_level: 活动水平（"sedentary"久坐, "light"轻度, "moderate"中度, "active"活跃, "very_active"非常活跃）
    返回:
        每日所需热量和营养建议
    """
    try:
        # 基础代谢率 (BMR) - 使用Mifflin-St Jeor公式
        if gender in ["男", "male", "m", "男性"]:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        
        # 活动系数
        activity_multipliers = {
            "sedentary": 1.2,      # 久坐
            "light": 1.375,        # 轻度活动
            "moderate": 1.55,      # 中度活动
            "active": 1.725,       # 活跃
            "very_active": 1.9     # 非常活跃
        }
        
        multiplier = activity_multipliers.get(activity_level.lower(), 1.2)
        daily_calories = bmr * multiplier
        
        # 营养素分配
        protein_g = weight_kg * 1.6  # 每公斤体重1.6克蛋白质
        fat_g = daily_calories * 0.25 / 9  # 25%热量来自脂肪
        carbs_g = (daily_calories - protein_g * 4 - fat_g * 9) / 4  # 剩余热量来自碳水
        
        return f"""每日热量需求: {daily_calories:.0f} 千卡
基础代谢率: {bmr:.0f} 千卡
活动水平: {activity_level}

营养素建议:
- 蛋白质: {protein_g:.0f}g (约 {protein_g*4:.0f} 千卡)
- 脂肪: {fat_g:.0f}g (约 {fat_g*9:.0f} 千卡)
- 碳水化合物: {carbs_g:.0f}g (约 {carbs_g*4:.0f} 千卡)"""
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def calculate_target_heart_rate(age: int, intensity: str = "moderate") -> str:
    """
    计算目标心率区间（用于运动）
    参数:
        age: 年龄
        intensity: 运动强度（"light"轻度, "moderate"中度, "vigorous"剧烈）
    返回:
        目标心率区间
    """
    try:
        max_hr = 220 - age  # 最大心率
        
        intensity_ranges = {
            "light": (0.5, 0.6),      # 50-60% 最大心率
            "moderate": (0.6, 0.7),   # 60-70% 最大心率
            "vigorous": (0.7, 0.85)   # 70-85% 最大心率
        }
        
        low_pct, high_pct = intensity_ranges.get(intensity.lower(), (0.6, 0.7))
        target_low = max_hr * low_pct
        target_high = max_hr * high_pct
        
        return f"""最大心率: {max_hr} 次/分钟
目标心率区间 ({intensity}): {target_low:.0f} - {target_high:.0f} 次/分钟

建议:
- 轻度运动: {max_hr*0.5:.0f}-{max_hr*0.6:.0f} 次/分钟（热身、恢复）
- 中度运动: {max_hr*0.6:.0f}-{max_hr*0.7:.0f} 次/分钟（有氧耐力）
- 剧烈运动: {max_hr*0.7:.0f}-{max_hr*0.85:.0f} 次/分钟（高强度训练）"""
    except Exception as e:
        return f"计算错误: {str(e)}"


# --- 2. 导出配置 ---

# 工具列表（供LangGraph使用）
medical_tools = [
    calculate_bmi,
    calculate_blood_pressure_risk,
    calculate_ideal_weight,
    calculate_daily_calories,
    calculate_target_heart_rate
]

# 为了兼容之前的 multi_agent.py 代码，加一个别名
medical_tools_list = medical_tools

# 工具描述（供路由器判断）
TOOL_DESCRIPTIONS = {
    "calculate_bmi": "计算BMI指数，需要身高和体重",
    "calculate_blood_pressure_risk": "评估血压风险，需要收缩压和舒张压",
    "calculate_ideal_weight": "计算理想体重，需要身高和性别",
    "calculate_daily_calories": "计算每日热量需求，需要体重、身高、年龄、性别、活动水平",
    "calculate_target_heart_rate": "计算运动目标心率，需要年龄"
}