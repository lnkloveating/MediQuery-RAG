"""
核心工具模块
负责：模式检测、文档评分、查询重写等辅助功能
"""
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ASSESSMENT_KEYWORDS, SCIENCE_KEYWORDS


def detect_mode(user_input: str) -> str:
    """
    检测用户意图，判断是健康评估还是科普问答
    
    Args:
        user_input: 用户输入
    
    Returns:
        "assessment" 或 "science"
    """
    input_lower = user_input.lower()
    
    # 如果明确标记为科普咨询，直接返回science模式
    # 这是结构化问诊生成的查询，不需要计算BMI等
    if "【咨询需求】" in user_input or "不需要计算" in user_input:
        return "science"
    
    has_numbers = any(char.isdigit() for char in user_input)
    
    assessment_score = sum(1 for kw in ASSESSMENT_KEYWORDS if kw in input_lower)
    science_score = sum(1 for kw in SCIENCE_KEYWORDS if kw in input_lower)
    
    # 只有明确要求计算时才进入assessment模式
    # 比如 "计算BMI"、"帮我算一下"
    calc_keywords = ["计算", "算一下", "帮我算", "多少"]
    has_calc_request = any(kw in input_lower for kw in calc_keywords)
    
    if has_calc_request and (has_numbers or assessment_score > 0):
        return "assessment"
    
    return "science"


def grade_documents(question: str, docs: List[str], llm) -> str:
    """
    评估检索到的文档是否与问题相关
    
    Args:
        question: 用户问题
        docs: 检索到的文档列表
        llm: LLM 实例
    
    Returns:
        "yes" 或 "no"
    """
    if not docs:
        return "no"
    
    context = "\n".join(docs[:2])
    prompt = f"""
    评估文档是否与问题相关。
    文档：{context}
    问题：{question}
    只回答：yes 或 no
    """
    score = llm.invoke(prompt).content.strip().lower()
    return "yes" if "yes" in score else "no"


def rewrite_query(question: str, llm) -> str:
    """
    重写搜索查询以获得更好的检索结果
    
    Args:
        question: 原始问题
        llm: LLM 实例
    
    Returns:
        重写后的查询
    """
    prompt = f"原问题检索失败，请重写一个更好的医学搜索词。原问题：{question}\n只输出新的查询词。"
    return llm.invoke(prompt).content.strip()
