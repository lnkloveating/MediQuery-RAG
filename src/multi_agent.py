"""
Self-RAG 高级编排层: 包含路由、反思检索、重写查询、记忆持久化
"""
import sys
import os
import uuid
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.messages import AIMessage, HumanMessage

# 导入同级模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools import medical_tools_list
# 从引擎导入资源 (注意这里多导入了一个 vectorstore 用于原生检索)
from medical_engine import llm, llm_with_tools, search_knowledge_base, vectorstore

# --- 1. 定义状态 (新增了 Self-RAG 需要的字段) ---
class MultiAgentState(TypedDict):
    # 消息历史
    messages: Annotated[list, add_messages]
    
    # 任务标志
    need_tool: bool
    need_rag: bool
    need_lifestyle: bool
    
    # 结果存储
    tool_output: str
    rag_output: str
    lifestyle_output: str
    final_answer: str
    
    # 🔥 Self-RAG 专用字段
    documents: List[str]   # 存储检索到的文档内容
    loop_step: int         # 循环计数器(防止死循环)

# --- 2. 辅助函数 (Self-RAG 核心能力) ---

def grade_documents(question: str, docs: List[str]) -> str:
    """阅卷老师：判断文档是否相关"""
    print("  ⚖️ [Self-RAG] 正在评估文档质量...")
    # 简单拼接前两个文档进行评估
    context = "\n".join(docs[:2]) 
    prompt = f"""
    你是一名评分员。请评估检索到的文档是否包含回答用户问题所需的信息。
    
    文档片段：
    {context}
    
    用户问题：
    {question}
    
    如果文档能部分或全部回答问题，或者包含相关关键词，回答 'yes'。
    如果文档完全不相关，回答 'no'。
    只回答一个单词：yes 或 no
    """
    score = llm.invoke(prompt).content.strip().lower()
    print(f"    👉 评分结果: {score}")
    return "yes" if "yes" in score else "no"

def rewrite_query(question: str) -> str:
    """改题专家：重写搜索词"""
    print(f"  🔄 [Self-RAG] 正在优化搜索词...")
    prompt = f"""
    你是一个搜索引擎优化专家。原问题检索效果不佳，请根据语义重写一个更好的搜索查询词。
    
    原问题：{question}
    
    只输出新的查询词，不要有任何解释。
    """
    new_query = llm.invoke(prompt).content.strip()
    print(f"    👉 新搜索词: {new_query}")
    return new_query

# --- 3. 节点定义 ---

def router_node(state: MultiAgentState):
    """路由节点"""
    question = state["messages"][-1].content
    print(f"\n🧭 [路由] 分析任务: {question}")
    
    prompt = f"""
    你是一个任务规划器。分析用户问题，判断需要执行哪些步骤。
    
    问题："{question}"
    
    请回答以下关键词中的一个或多个（用空格隔开）：
    - TOOL (涉及身高体重、血压、热量计算)
    - RAG (涉及疾病原理、治疗、定义、医学知识)
    - LIFESTYLE (涉及饮食、运动、睡眠建议)
    
    只输出关键词。
    """
    decision = llm.invoke(prompt).content.upper()
    print(f"  👉 规划结果: {decision}")
    
    return {
        "need_tool": "TOOL" in decision,
        "need_rag": "RAG" in decision,
        "need_lifestyle": "LIFESTYLE" in decision,
        "loop_step": 0, # 重置循环计数
        "documents": []
    }

def tool_node(state: MultiAgentState):
    """工具节点"""
    print("🔧 [工具Agent] 正在计算...")
    question = state["messages"][-1].content
    response = llm_with_tools.invoke(question)
    
    output = ""
    if response.tool_calls:
        results = []
        for call in response.tool_calls:
            tool = next((t for t in medical_tools_list if t.name == call["name"]), None)
            if tool:
                try:
                    res = tool.invoke(call["args"])
                    results.append(str(res))
                except Exception as e:
                    results.append(f"Error: {e}")
        output = "\n".join(results)
    
    return {"tool_output": output}

def retrieve_node(state: MultiAgentState):
    """Self-RAG: 检索节点"""
    print("📚 [Self-RAG] 执行检索...")
    question = state["messages"][-1].content
    
    # 如果是重写过的问题，它会作为新的一条 HumanMessage 存在 messages 里
    # 我们取最后一条消息作为查询词
    
    # 结合之前的工具计算结果来搜 (增强上下文)
    if state.get("tool_output"):
        question += f" {state['tool_output']}"

    # 使用 vectorstore 原生检索，获取 list
    docs = vectorstore.similarity_search(question, k=4)
    doc_contents = [d.page_content for d in docs]
    
    return {"documents": doc_contents, "loop_step": state["loop_step"] + 1}

def grade_and_generate_node(state: MultiAgentState):
    """Self-RAG: 评分与生成决策节点"""
    question = state["messages"][-1].content
    docs = state["documents"]
    
    # 1. 评分
    score = grade_documents(question, docs)
    
    # 2. 决策逻辑
    if score == "yes" or state["loop_step"] >= 3:
        # 如果相关，或者已经重试了3次，就强制生成
        if score == "no":
            print("  ⚠️ 重试次数已达上限，强制生成回答。")
        
        print("💡 [Self-RAG] 生成最终回答...")
        context = "\n\n".join(docs)
        prompt = f"基于资料回答：\n资料：{context}\n问题：{question}"
        answer = llm.invoke(prompt).content
        return {"rag_output": answer, "final_answer": "ready"} # 标记完成
        
    else:
        # 3. 如果不相关且没超限 -> 重写问题
        new_query = rewrite_query(question)
        # 将新问题加入历史，供下一轮检索使用
        return {"messages": [HumanMessage(content=new_query)]}

def lifestyle_node(state: MultiAgentState):
    """生活建议节点"""
    print("🏃 [生活方式Agent] 生成建议...")
    question = state["messages"][-1].content
    query = f"运动 饮食 睡眠 预防 {question}"
    context = search_knowledge_base(query, k=4)
    
    prompt = f"基于以下资料提供生活建议：\n资料：{context}\n问题：{question}"
    advice = llm.invoke(prompt).content
    return {"lifestyle_output": advice}

def summarizer_node(state: MultiAgentState):
    """总结节点"""
    print("📊 [总结Agent] 整合输出...")
    parts = []
    if state.get("tool_output"): parts.append(f"📋 【健康评估】\n{state['tool_output']}")
    if state.get("rag_output"): parts.append(f"📖 【医学知识】\n{state['rag_output']}")
    if state.get("lifestyle_output"): parts.append(f"💡 【生活建议】\n{state['lifestyle_output']}")
    
    final_text = "\n\n" + "="*30 + "\n\n".join(parts) if parts else "抱歉，我无法回答。"
    return {"final_answer": final_text, "messages": [AIMessage(content=final_text)]}

# --- 4. 构建图逻辑 ---
workflow = StateGraph(MultiAgentState)

workflow.add_node("router", router_node)
workflow.add_node("tool_agent", tool_node)
workflow.add_node("lifestyle_agent", lifestyle_node)
workflow.add_node("summarizer", summarizer_node)

# Self-RAG 子图节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_loop", grade_and_generate_node)

# 起点
workflow.add_edge(START, "router")

# Router 路由逻辑
def route_after_router(state):
    # 并行/串行逻辑：优先处理 Tool，然后 RAG，最后 Lifestyle
    if state["need_tool"]: return "tool_agent"
    if state["need_rag"]: return "retrieve"
    if state["need_lifestyle"]: return "lifestyle_agent"
    return "summarizer"

workflow.add_conditional_edges("router", route_after_router)

# Tool 后的路由
def route_after_tool(state):
    if state["need_rag"]: return "retrieve"
    if state["need_lifestyle"]: return "lifestyle_agent"
    return "summarizer"

workflow.add_conditional_edges("tool_agent", route_after_tool)

# Self-RAG 内部循环逻辑
workflow.add_edge("retrieve", "grade_loop")

def route_self_rag(state):
    # 检查 grade_and_generate_node 的输出
    # 如果生成了 rag_output (即 final_answer == 'ready')，则退出循环
    if state.get("final_answer") == "ready":
        # RAG 结束后，看是否需要生活建议
        if state["need_lifestyle"]: return "lifestyle_agent"
        return "summarizer"
    else:
        # 否则回炉重造（利用重写后的 query 再次检索）
        return "retrieve"

workflow.add_conditional_edges("grade_loop", route_self_rag, 
    {"lifestyle_agent": "lifestyle_agent", "summarizer": "summarizer", "retrieve": "retrieve"}
)

# Lifestyle 后的路由
workflow.add_edge("lifestyle_agent", "summarizer")
workflow.add_edge("summarizer", END)

# --- 5. 运行 (带持久化) ---
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("="*50)
    print("🚀 Self-RAG 医学专家系统 (含反思能力)")
    print("👉 /new: 新对话 | /clear: 清空记忆 | q: 退出")
    print("="*50)
    
    # 初始化 ID
    user_id = input("请输入会话ID (回车新ID): ").strip()
    thread_id = user_id if user_id else str(uuid.uuid4())
    print(f"✨ 当前ID: {thread_id}\n" + "-"*30)
    
    while True:
        try:
            user_input = input("\n👤 患者: ").strip()
            if user_input.lower() in ["q", "quit"]: break
            
            # 记忆管理指令
            if user_input == "/new":
                thread_id = str(uuid.uuid4())
                print(f"🧹 新会话: {thread_id}"); continue
            
            if user_input == "/clear":
                conn.cursor().execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                conn.commit()
                print("✅ 记忆已清除"); continue
            
            if not user_input: continue

            config = {"configurable": {"thread_id": thread_id}}
            
            # 运行并打印最终结果
            # 注意：Self-RAG 中间步骤多，stream_mode="values" 会打印很多过程
            # 这里我们只打印最终结果，中间过程通过 print 调试信息查看
            final_res = None
            for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
                # 实时捕获最终结果
                if "summarizer" in event:
                    final_res = event["summarizer"]["final_answer"]
            
            if final_res:
                print(final_res)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 错误: {e}")