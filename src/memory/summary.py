"""
短期记忆模块 - 对话摘要
负责：当对话过长时，压缩历史消息为摘要

扩展指南：
- 修改摘要策略：编辑 SUMMARY_PROMPT 模板
- 调整触发阈值：修改 config/settings.py 中的 MAX_MESSAGES_BEFORE_SUMMARY
"""
from typing import List, Tuple
from langchain_core.messages import HumanMessage, AIMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MAX_MESSAGES_BEFORE_SUMMARY, KEEP_RECENT_MESSAGES


# ============================================================
# 摘要 Prompt 模板
# ============================================================
SUMMARY_PROMPT = """
请总结以下对话的关键信息，重点提取：

1. 用户提到的身体指标（身高、体重、血压等具体数值）
2. 用户的健康状况（疾病、过敏、症状）
3. 用户的主要问题和关注点
4. 助手给出的重要建议

对话内容：
{conversation}

请用简洁的要点形式总结（不超过300字），保留所有具体数值和重要健康信息：
"""


def should_summarize(messages: list) -> bool:
    """
    判断是否需要进行摘要
    
    Args:
        messages: 消息列表
    
    Returns:
        是否需要摘要
    """
    return len(messages) > MAX_MESSAGES_BEFORE_SUMMARY


def summarize_messages(messages: list, llm) -> Tuple[str, list]:
    """
    将旧消息压缩为摘要
    
    Args:
        messages: 完整的消息列表
        llm: LLM 实例
    
    Returns:
        (摘要文本, 保留的最近消息列表)
    """
    if not should_summarize(messages):
        return "", messages
    
    print(f"  📝 [对话摘要] 消息数 {len(messages)} 超过阈值，正在压缩...")
    
    # 分离旧消息和新消息
    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]
    
    # 构建对话文本
    conversation_text = []
    for msg in old_messages:
        if hasattr(msg, 'content') and msg.content:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            # 截断过长的单条消息
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            conversation_text.append(f"{role}: {content}")
    
    prompt = SUMMARY_PROMPT.format(conversation="\n".join(conversation_text))
    
    try:
        summary = llm.invoke(prompt).content.strip()
        print(f"  ✓ 摘要完成，压缩了 {len(old_messages)} 条消息")
        return summary, recent_messages
    except Exception as e:
        print(f"  ⚠️ 摘要生成失败: {e}")
        return "", recent_messages