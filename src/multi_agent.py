"""
src/multi_agent.py
多Agent编排层: 能够处理计算、问答、建议的复杂任务
"""
import sys
import os
import uuid
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.messages import AIMessage, HumanMessage

# 导入同级模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools import medical_tools_list
# 从引擎导入资源，避免重复代码
from medical_engine import llm, llm_with_tools, search_knowledge_base

# --- 1. 定义状态 ---
class MultiAgentState(TypedDict):
    # 消息历史
    messages: Annotated[list, add_messages]
    
    # 任务标志 (Router 决定)
    need_tool: bool
    need_rag: bool
    need_lifestyle: bool
    
    # 中间结果
    tool_output: str
    rag_output: str
    lifestyle_output: str
    
    # 最终结果
    final_answer: str

# --- 2. 节点定义 ---

def router_node(state: MultiAgentState):
    """
    路由节点：分析这三件事分别是否需要做
    """
    question = state["messages"][-1].content
    print(f"\n🧭 [路由] 分析任务: {question}")
    
    prompt = f"""
    你是一个任务规划器。请分析用户问题，判断需要执行哪些步骤。
    
    问题："{question}"
    
    请回答以下三个关键词中的一个或多个（用空格隔开）：
    - TOOL (如果涉及身高体重、血压、热量计算)
    - RAG (如果涉及疾病原理、治疗、定义、医学知识)
    - LIFESTYLE (如果涉及具体的饮食、运动、睡眠建议)
    
    示例：
    "算一下BMI" -> TOOL
    "什么是糖尿病" -> RAG
    "我太胖了怎么减肥" -> RAG LIFESTYLE
    "算BMI并给点建议" -> TOOL LIFESTYLE
    """
    
    decision = llm.invoke(prompt).content.upper()
    print(f"  👉 规划结果: {decision}")
    
    return {
        "need_tool": "TOOL" in decision,
        "need_rag": "RAG" in decision,
        "need_lifestyle": "LIFESTYLE" in decision
    }

def tool_node(state: MultiAgentState):
    """工具节点"""
    print("🔧 [工具Agent] 正在计算...")
    question = state["messages"][-1].content
    
    response = llm_with_tools.invoke(question)
    
    output = "无计算结果"
    if response.tool_calls:
        results = []
        for call in response.tool_calls:
            tool = next((t for t in medical_tools_list if t.name == call["name"]), None)
            if tool:
                print(f"  ⚙️ 调用: {tool.name}")
                try:
                    res = tool.invoke(call["args"])
                    results.append(str(res))
                except Exception as e:
                    results.append(f"工具执行错误: {e}")
        output = "\n".join(results)
        print(f"  ✅ 计算完成")
    else:
        print("  ⚠️ 模型未调用工具")
        
    return {"tool_output": output}

def rag_node(state: MultiAgentState):
    """医学知识节点"""
    print("📚 [RAG Agent] 正在查询知识库...")
    question = state["messages"][-1].content
    
    # 如果前面有计算结果（比如算出了肥胖），把计算结果也加进检索上下文
    search_query = question
    if state.get("tool_output"):
        search_query += f" {state['tool_output']}"
    
    context = search_knowledge_base(search_query, k=3)
    
    if not context:
        return {"rag_output": "知识库中未找到直接相关信息。"}
    
    prompt = f"基于资料回答问题：\n资料：{context}\n问题：{question}"
    answer = llm.invoke(prompt).content
    return {"rag_output": answer}

def lifestyle_node(state: MultiAgentState):
    """生活建议节点"""
    print("🏃 [生活方式Agent] 正在生成建议...")
    question = state["messages"][-1].content
    
    # 专门搜运动饮食相关
    query = f"运动 饮食 睡眠 预防 {question}"
    context = search_knowledge_base(query, k=4)
    
    prompt = f"""
    你是健康教练。请基于以下医学资料，为用户提供生活方式建议（运动、饮食、睡眠）。
    如果前面有计算出的健康风险（如肥胖、高血压），请针对性给出建议。
    
    参考资料：
    {context}
    
    用户问题：{question}
    """
    advice = llm.invoke(prompt).content
    return {"lifestyle_output": advice}

def summarizer_node(state: MultiAgentState):
    """总结节点"""
    print("📊 [总结Agent] 正在整合...")
    
    parts = []
    if state.get("tool_output"):
        parts.append(f"📋 【健康评估】\n{state['tool_output']}")
    
    if state.get("rag_output"):
        parts.append(f"📖 【医学知识】\n{state['rag_output']}")
        
    if state.get("lifestyle_output"):
        parts.append(f"💡 【生活建议】\n{state['lifestyle_output']}")
    
    if not parts:
        final_text = "抱歉，我不确定如何回答您的问题，请尝试问得更具体一些。"
    else:
        final_text = "\n\n" + "="*30 + "\n\n".join(parts)
    
    return {
        "final_answer": final_text,
        "messages": [AIMessage(content=final_text)]
    }

# --- 3. 构建图逻辑 (流水线模式) ---
workflow = StateGraph(MultiAgentState)

workflow.add_node("router", router_node)
workflow.add_node("tool_agent", tool_node)
workflow.add_node("rag_agent", rag_node)
workflow.add_node("lifestyle_agent", lifestyle_node)
workflow.add_node("summarizer", summarizer_node)

# 起点
workflow.add_edge(START, "router")

# 条件跳转逻辑
def route_after_router(state):
    if state["need_tool"]: return "tool_agent"
    if state["need_rag"]: return "rag_agent"
    if state["need_lifestyle"]: return "lifestyle_agent"
    return "summarizer"

def route_after_tool(state):
    if state["need_rag"]: return "rag_agent"
    if state["need_lifestyle"]: return "lifestyle_agent"
    return "summarizer"

def route_after_rag(state):
    if state["need_lifestyle"]: return "lifestyle_agent"
    return "summarizer"

# 连接边
workflow.add_conditional_edges("router", route_after_router)
workflow.add_conditional_edges("tool_agent", route_after_tool)
workflow.add_conditional_edges("rag_agent", route_after_rag)
workflow.add_edge("lifestyle_agent", "summarizer")
workflow.add_edge("summarizer", END)

# --- 4. 运行 ---
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn)

app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("="*50)
    print("🚀 多Agent医学专家系统 (持久化记忆版)")
    print("👉 指令说明：")
    print("   q      - 退出")
    print("   /new   - 切换到新随机账号 (保留旧数据)")
    print("   /clear - 清空当前账号记忆 (物理删除)")
    print("="*50)
    
    print("\n💡 提示：输入旧的 ID 可以恢复上次的对话记忆。")
    user_input_id = input("请输入会话 ID (直接回车将自动生成新 ID): ").strip()
    
    if user_input_id:
        thread_id = user_input_id
        print(f"📂 已加载历史会话: {thread_id}")
    else:
        thread_id = str(uuid.uuid4())
        print(f"✨ 已创建新会话 ID: {thread_id}")
    
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 患者: ").strip()
            
            # 1. 退出指令
            if user_input.lower() in ["q", "quit", "exit"]:
                print("👋 再见！")
                break
            
            # 2. 切换新用户指令 (/new)
            if user_input.lower() in ["/new", "new"]:
                thread_id = str(uuid.uuid4())
                print(f"\n✨ 已切换到新随机账号: {thread_id}")
                print("-" * 30)
                continue

            # 3. 🔥 新增：清空当前记忆指令 (/clear)
            if user_input.lower() in ["/clear", "clear", "清空"]:
                print(f"\n🧹 正在清空 ID [{thread_id}] 的所有记忆...")
                
                # 直接操作数据库删除对应 ID 的记录
                cursor = conn.cursor()
                # 删除检查点 (LangGraph 的存储表名为 checkpoints 和 checkpoint_blobs 或 checkpoint_writes)
                # 为了兼容性，我们尝试删除所有相关表中的该 ID 数据
                try:
                    cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                    cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
                    cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
                    conn.commit()
                    print("✅ 记忆已物理清除！您现在就像初次见面一样。")
                except Exception as db_e:
                    print(f"⚠️ 清除部分数据时遇到问题 (可能是表不存在，但不影响使用): {db_e}")
                
                print("-" * 30)
                # 清空后不需要 continue，可以直接让用户开始新一轮对话，或者 continue 让用户重新输入
                continue
            
            # 4. 切换指定用户指令 (/load)
            if user_input.startswith("/load"):
                parts = user_input.split()
                if len(parts) > 1:
                    thread_id = parts[1]
                    print(f"\n📂 已切换到会话: {thread_id}")
                    continue
            
            if not user_input:
                continue

            # 5. 运行图
            config = {"configurable": {"thread_id": thread_id}}
            
            for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
                pass 
            
            final_state = app.get_state(config)
            print(final_state.values.get("final_answer"))
            
        except KeyboardInterrupt:
            print("\n👋 用户强制中断")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")