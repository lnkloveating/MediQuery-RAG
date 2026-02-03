"""
健康信息提取模块
负责：从用户消息中识别并提取健康相关信息

扩展指南：
- 修改提取规则：编辑 EXTRACTION_PROMPT 模板
- 添加新的信息类别：在 config/settings.py 的 HEALTH_CATEGORIES 中添加
"""
import json
from typing import List, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.profile_store import profile_store
from config.settings import HEALTH_CATEGORIES


# ============================================================
# 提取 Prompt 模板
# 修改此模板可以调整 LLM 的提取行为
# ============================================================
EXTRACTION_PROMPT = """
分析用户消息，提取健康相关的个人信息。

用户消息："{user_message}"

提取规则：
1. 身体指标：必须包含完整数值，如"身高165cm"、"体重77kg"，不要拆分
2. 过敏信息：如"鸡蛋过敏"、"海鲜过敏"（important设为true）
3. 疾病史：如"有高血压"、"糖尿病"（important设为true）
4. 生活习惯：如"每天吸烟"、"不喝酒"
5. 用药情况：如"正在服用降压药"（important设为true）

【重要规则】
- 身高体重必须带单位：身高xxxcm，体重xxxkg
- 过敏、疾病、用药的 important 必须为 true
- 每条信息独立一个对象，不要合并

返回JSON数组示例：
[
  {{"category": "身体指标", "content": "身高165cm", "important": false}},
  {{"category": "身体指标", "content": "体重77kg", "important": false}},
  {{"category": "过敏信息", "content": "鸡蛋过敏", "important": true}}
]

没有健康信息返回：[]
只返回JSON，不要其他文字。
"""


def extract_health_info(user_message: str, user_id: str, llm) -> List[dict]:
    """
    从用户消息中提取健康信息并存储
    
    Args:
        user_message: 用户输入的消息
        user_id: 用户ID（anonymous 则不存储）
        llm: LLM 实例
    
    Returns:
        提取到的信息列表
    """
    if not user_id or user_id == "anonymous":
        return []
    
    prompt = EXTRACTION_PROMPT.format(user_message=user_message)
    extracted_items = []
    
    try:
        result = llm.invoke(prompt).content.strip()
        
        # 清理 markdown 代码块
        if "```" in result:
            parts = result.split("```")
            for part in parts:
                if "[" in part:
                    result = part.replace("json", "").strip()
                    break
        
        # 解析 JSON
        if result and "[" in result:
            info_list = json.loads(result)
            if not isinstance(info_list, list):
                info_list = [info_list]
            
            for info in info_list:
                if info and isinstance(info, dict) and info.get("content"):
                    # 存入数据库
                    added = profile_store.add_health_record(
                        user_id=user_id,
                        category=info["category"],
                        content=info["content"],
                        important=info.get("important", False)
                    )
                    if added:
                        print(f"  💾 已记录: [{info['category']}] {info['content']}")
                        extracted_items.append(info)
                        
    except json.JSONDecodeError:
        pass
    except Exception as e:
        pass
    
    return extracted_items


def load_health_profile(user_id: str) -> str:
    """
    加载用户健康档案，格式化为文本
    
    Args:
        user_id: 用户ID
    
    Returns:
        格式化的健康档案文本
    """
    if not user_id or user_id == "anonymous":
        return ""
    
    records = profile_store.get_health_records(user_id)
    if not records:
        return ""
    
    # 按类别整理
    profile_dict = {}
    important_items = []
    
    for record in records:
        category = record["category"]
        content = record["content"]
        
        if category not in profile_dict:
            profile_dict[category] = []
        profile_dict[category].append(content)
        
        if record["important"]:
            important_items.append(f"⚠️ {content}")
    
    # 格式化输出
    lines = []
    
    # 重要信息优先显示
    if important_items:
        lines.append("【⚠️ 重要提醒】")
        lines.extend(important_items)
        lines.append("")
    
    # 按类别显示
    for category, contents in profile_dict.items():
        lines.append(f"【{category}】")
        for c in contents:
            lines.append(f"  • {c}")
    
    return "\n".join(lines)