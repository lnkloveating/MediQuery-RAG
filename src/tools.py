"""
医学计算工具库
"""
from typing import Dict, Union

# ==========================================
# PART 1: 占位符 (防止 main.py 报错)
# ==========================================

# ⚠️ 必须保留这个变量名，因为 main.py 引用了它
# 把它设为空列表，表示不再提供给 Agent 任何自动调用的工具
medical_tools_list = [] 

# 同理保留这个变量
TOOL_DESCRIPTIONS = {}


# ==========================================
# PART 2: 纯计算函数 (新功能核心)
# 供结构化问诊后台静默调用，只返回数值
# ==========================================

def calculate_bmi_pure(height_cm: float, weight_kg: float) -> Dict[str, Union[float, str]]:
    """纯 BMI 计算"""
    try:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return {"value": round(bmi, 2), "unit": "kg/m²", "type": "BMI"}
    except Exception as e:
        return {"error": str(e)}

def calculate_bmr_pure(weight_kg: float, height_cm: float, age: int, gender: str) -> Dict[str, Union[float, str]]:
    """纯 BMR 计算 (Mifflin-St Jeor 公式)"""
    try:
        # 简单归一化性别
        g = str(gender).lower()
        is_male = g in ["男", "male", "m", "男性"]
        
        # BMR公式
        if is_male:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
            
        return {"value": round(bmr, 0), "unit": "kcal/day", "type": "BMR"}
    except Exception as e:
        return {"error": str(e)}

def calculate_ideal_weight_pure(height_cm: float, gender: str) -> Dict[str, Union[float, str]]:
    """纯理想体重计算"""
    try:
        height_m = height_cm / 100
        g = str(gender).lower()
        is_male = g in ["男", "male", "m", "男性"]
        
        factor = 22 if is_male else 21
        ideal = factor * (height_m ** 2)
        
        return {"value": round(ideal, 1), "unit": "kg", "type": "Ideal Weight"}
    except Exception as e:
        return {"error": str(e)}

# 映射表，方便调用
PURE_CALC_TOOLS = {
    "BMI": calculate_bmi_pure,
    "BMR": calculate_bmr_pure,
    "IDEAL_WEIGHT": calculate_ideal_weight_pure
}