"""
src/multi_agent.py
Self-RAG + Web Search: 本地搜不到 -> 自动联网搜 -> 智能回答
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

# 🔥 关键修改：从引擎导入 web_search_tool
# 如果这里报错，请检查 src/medical_engine.py 是否已经添加了 TavilySearchResults
try:
    from medical_engine import llm, llm_with_tools, search_knowledge_base, vectorstore, web_search_tool
except ImportError:
    print("❌ 错误: 无法从 medical_engine 导入 web_search_tool。请确保你已更新 medical_engine.py 并安装了 tavily-python。")
    sys.exit(1)

# --- 1. 定义状态 ---
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    need_tool: bool
    need_rag: bool
    need_lifestyle: bool
    
    tool_output: str
    rag_output: str
    lifestyle_output: str
    final_answer: str
    
    documents: List[str]
    loop_step: int
    # 🔥 新增标志：是否使用了网络搜索
    used_web_search: bool 

# --- 2. 辅助函数 ---

def grade_documents(question: str, docs: List[str]) -> str:
    """阅卷老师：判断文档是否相关"""
    print("  ⚖️ [评判] 正在评估文档质量...")
    if not docs: return "no"
    
    # 简单拼接前两个文档进行评估
    context = "\n".join(docs[:2]) 
    prompt = f"""
    你是一名评分员。请评估文档是否包含回答问题的信息。
    
    文档片段：
    {context}
    
    用户问题：
    {question}
    
    如果文档能提供哪怕一点点线索，都回答 'yes'。
    只有完全不相关才回答 'no'。
    只回答：yes 或 no
    """
    score = llm.invoke(prompt).content.strip().lower()
    print(f"    👉 评分: {score}")
    return "yes" if "yes" in score else "no"

def rewrite_query(question: str) -> str:
    """改题专家：重写搜索词"""
    print(f"  🔄 [优化] 正在重写搜索词...")
    prompt = f"""
    原问题检索失败，请重写一个更好的搜索查询词。
    原问题：{question}
    只输出新的查询词。
    """
    new_query = llm.invoke(prompt).content.strip()
    print(f"    👉 新词: {new_query}")
    return new_query

# --- 3. 节点定义 ---

def router_node(state: MultiAgentState):
    """路由节点"""
    question = state["messages"][-1].content
    print(f"\n🧭 [路由] 分析任务: {question}")
    
    prompt = f"""
    分析用户问题，选择关键词（空格隔开）：
    - TOOL (计算类：BMI、血压、热量)
    - RAG (知识类：疾病、治疗、原理)
    - LIFESTYLE (建议类：饮食、运动)
    
    问题："{question}"
    """
    decision = llm.invoke(prompt).content.upper()
    print(f"  👉 规划: {decision}")
    
    return {
        "need_tool": "TOOL" in decision,
        "need_rag": "RAG" in decision,
        "need_lifestyle": "LIFESTYLE" in decision,
        "loop_step": 0,
        "documents": [],
        "used_web_search": False # 初始化为 False
    }

def tool_node(state: MultiAgentState):
    """工具节点"""
    print("🔧 [工具] 正在计算...")
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
    """本地检索节点"""
    print("📚 [本地RAG] 检索知识库...")
    question = state["messages"][-1].content
    if state.get("tool_output"): question += f" {state['tool_output']}"

    # 本地检索
    docs = vectorstore.similarity_search(question, k=4)
    doc_contents = [d.page_content for d in docs]
    
    return {"documents": doc_contents, "loop_step": state["loop_step"] + 1}

def web_search_node(state: MultiAgentState):
    """🔥 新增：Web 搜索节点"""
    print("🌐 [Web搜索] 本地无结果，正在联网搜索...")
    question = state["messages"][-1].content
    
    try:
        # 使用 Tavily 搜索
        results = web_search_tool.invoke({"query": question})
        # 提取内容 (Tavily 返回的是 list[dict])
        web_contents = [res['content'] for res in results]
        print(f"    ✅ 联网获取了 {len(web_contents)} 条结果")
        return {"documents": web_contents, "used_web_search": True}
    except Exception as e:
        print(f"    ❌ 联网搜索失败: {e}")
        return {"documents": ["网络搜索失败，请稍后重试。"], "used_web_search": True}

def grade_and_generate_node(state: MultiAgentState):
    """评分与生成决策节点"""
    question = state["messages"][-1].content
    docs = state["documents"]
    
    # 1. 评分
    score = grade_documents(question, docs)
    
    # 2. 决策逻辑
    if score == "yes":
        # A. 资料相关 -> 直接生成
        print("💡 [生成] 资料相关，生成回答...")
        context = "\n\n".join(docs)
        # 标注来源
        source_tag = "(来源: 互联网)" if state["used_web_search"] else "(来源: 本地知识库)"
        prompt = f"基于资料回答({source_tag})：\n资料：{context}\n问题：{question}"
        answer = llm.invoke(prompt).content
        return {"rag_output": answer, "final_answer": "ready"}
        
    elif state["loop_step"] >= 3:
        # B. 重试次数超限
        if not state["used_web_search"]:
            # -> 还没联网过 -> 指示路由去联网
            print("  ⚠️ 本地多次重试失败，转入 Web 搜索...")
            return {"final_answer": "go_web"}
        else:
            # -> 联网了还是不行 -> 强行回答
            print("  ⚠️ 联网也搜不到，强行回答。")
            context = "\n\n".join(docs)
            prompt = f"资料相关性低，请尽力回答：\n资料：{context}\n问题：{question}"
            answer = llm.invoke(prompt).content
            return {"rag_output": answer, "final_answer": "ready"}
            
    else:
        # C. 不相关且没超限 -> 重写问题
        new_query = rewrite_query(question)
        return {"messages": [HumanMessage(content=new_query)]}

def lifestyle_node(state: MultiAgentState):
    """生活建议节点"""
    print("🏃 [生活] 生成建议...")
    question = state["messages"][-1].content
    context = search_knowledge_base(f"建议 {question}", k=4)
    prompt = f"提供生活建议：\n资料：{context}\n问题：{question}"
    advice = llm.invoke(prompt).content
    return {"lifestyle_output": advice}

def summarizer_node(state: MultiAgentState):
    """总结节点"""
    print("📊 [总结] 整合输出...")
    parts = []
    if state.get("tool_output"): parts.append(f"📋 【健康评估】\n{state['tool_output']}")
    if state.get("rag_output"): parts.append(f"📖 【医学知识】\n{state['rag_output']}")
    if state.get("lifestyle_output"): parts.append(f"💡 【生活建议】\n{state['lifestyle_output']}")
    
    final_text = "\n\n" + "="*30 + "\n\n".join(parts) if parts else "抱歉，无法回答。"
    return {"final_answer": final_text, "messages": [AIMessage(content=final_text)]}

# --- 4. 构建图逻辑 ---
workflow = StateGraph(MultiAgentState)

workflow.add_node("router", router_node)
workflow.add_node("tool_agent", tool_node)
workflow.add_node("lifestyle_agent", lifestyle_node)
workflow.add_node("summarizer", summarizer_node)

# Self-RAG + Web 节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_loop", grade_and_generate_node)
workflow.add_node("web_search", web_search_node) # 🔥 新增节点

workflow.add_edge(START, "router")

def route_after_router(state):
    if state["need_tool"]: return "tool_agent"
    if state["need_rag"]: return "retrieve"
    if state["need_lifestyle"]: return "lifestyle_agent"
    return "summarizer"

workflow.add_conditional_edges("router", route_after_router)
workflow.add_conditional_edges("tool_agent", lambda x: "retrieve" if x["need_rag"] else ("lifestyle_agent" if x["need_lifestyle"] else "summarizer"))

# 核心：本地检索 -> 评分/生成
workflow.add_edge("retrieve", "grade_loop")

# 🔥 核心路由逻辑更新：处理 Web 搜索跳转
def route_self_rag(state):
    decision = state.get("final_answer")
    
    if decision == "ready":
        # 完成 RAG，看是否需要生活建议
        return "lifestyle_agent" if state["need_lifestyle"] else "summarizer"
    elif decision == "go_web":
        # 本地搜不到 -> 去联网
        return "web_search"
    else:
        # 继续本地重试 (Rewrite loop)
        return "retrieve"

workflow.add_conditional_edges("grade_loop", route_self_rag, 
    {"lifestyle_agent": "lifestyle_agent", "summarizer": "summarizer", "retrieve": "retrieve", "web_search": "web_search"}
)

# 联网搜索后，再次去评分和生成 (给它一次机会判断网上的内容对不对)
workflow.add_edge("web_search", "grade_loop")

workflow.add_edge("lifestyle_agent", "summarizer")
workflow.add_edge("summarizer", END)

# --- 5. 运行 ---
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("="*50)
    print("🚀 Self-RAG + Web Search (全能医学助手)")
    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️ 警告: 未检测到 TAVILY_API_KEY，联网搜索可能失败！")
    print("="*50)
    
    user_id = input("Session ID (Enter for new): ").strip()
    thread_id = user_id if user_id else str(uuid.uuid4())
    print(f"✨ Session: {thread_id}")
    
    while True:
        try:
            user_input = input("\n👤 患者: ").strip()
            if user_input.lower() in ["q", "quit"]: break
            if user_input == "/new": thread_id = str(uuid.uuid4()); print("✨ New Session"); continue
            if not user_input: continue

            config = {"configurable": {"thread_id": thread_id}}
            
            # 捕获最终结果
            final_res = None
            for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
                if "summarizer" in event:
                    final_res = event["summarizer"]["final_answer"]
            
            if final_res:
                print(final_res)
            
        except KeyboardInterrupt: break
        except Exception as e: print(f"❌ Error: {e}")