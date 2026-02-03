# agents/graph.py
"""
Graph 构建模块
负责：构建 LangGraph 工作流

扩展指南：
- 添加新节点：在 build_graph() 中使用 workflow.add_node()
- 修改流程：调整 add_edge() 和 add_conditional_edges()
"""
import sqlite3
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CHAT_HISTORY_DB


# ============================================================
# State 定义
# ============================================================
class MedicalState(TypedDict):
    """工作流状态定义"""
    messages: Annotated[list, add_messages]
    mode: str  # "assessment" | "science"
    user_id: str
    need_tool: bool
    need_rag: bool
    need_web: bool
    tool_output: str
    rag_output: str
    final_answer: str
    documents: List[str]
    loop_step: int
    used_web_search: bool
    health_profile: str
    summary: str


def build_graph(nodes: dict):
    """
    构建 LangGraph 工作流
    
    Args:
        nodes: 节点函数字典
    
    Returns:
        编译后的 app
    """
    workflow = StateGraph(MedicalState)
    
    # 注册节点
    workflow.add_node("router", nodes["router"])
    workflow.add_node("assessment_tool", nodes["assessment_tool"])
    workflow.add_node("retrieve", nodes["retrieve"])
    workflow.add_node("grade_loop", nodes["grade_loop"])
    workflow.add_node("web_search", nodes["web_search"])
    workflow.add_node("summarizer", nodes["summarizer"])
    
    # 设置入口
    workflow.add_edge(START, "router")
    
    # 路由后的条件边
    def route_after_router(state):
        return "assessment_tool" if state["mode"] == "assessment" else "retrieve"
    
    workflow.add_conditional_edges("router", route_after_router)
    
    # 固定边
    workflow.add_edge("assessment_tool", "retrieve")
    workflow.add_edge("retrieve", "grade_loop")
    
    # Self-RAG 循环的条件边
    def route_self_rag(state):
        decision = state.get("final_answer")
        if decision == "ready":
            return "summarizer"
        elif decision == "go_web":
            return "web_search"
        return "retrieve"
    
    workflow.add_conditional_edges(
        "grade_loop", 
        route_self_rag,
        {"summarizer": "summarizer", "web_search": "web_search", "retrieve": "retrieve"}
    )
    
    workflow.add_edge("web_search", "grade_loop")
    workflow.add_edge("summarizer", END)
    
    # 编译
    conn = sqlite3.connect(CHAT_HISTORY_DB, check_same_thread=False)
    memory = SqliteSaver(conn)
    app = workflow.compile(checkpointer=memory)
    
    return app
