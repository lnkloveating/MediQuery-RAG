import sys
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# --- 1. 配置与初始化 ---
DB_PATH = "./medical_db"

# 检查向量库是否存在
if not os.path.exists(DB_PATH):
    print(f"❌ 错误：向量库不存在 {DB_PATH}")
    print("请先运行数据入库脚本！")
    sys.exit(1)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str 

# 模型：建议 temperature=0 保持严谨
llm = ChatOllama(model="qwen2.5:7b", temperature=0)
embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# --- 2. 节点定义 ---

def retrieve_node(state: State):
    user_query = state["messages"][-1].content
    
    # 🔧 修复：直接用原问题检索（小模型关键词提取不稳定）
    search_query = user_query
    print(f"🔍 [系统] 检索词: {search_query}")
    
    # 检索相关文档
    raw_docs = vectorstore.similarity_search(search_query, k=5)  # 直接取5个
    
    # 逻辑去重
    unique_docs = []
    seen_titles = set()
    for doc in raw_docs:
        title = doc.metadata.get('title', '')
        if title and title not in seen_titles:
            unique_docs.append(doc)
            seen_titles.add(title)
            
    if not unique_docs:
        context_text = "未检索到相关资料"
    else:
        context_text = "\n\n".join([
            f"【来源: {d.metadata.get('title', '未知')}】\n{d.page_content}" 
            for d in unique_docs
        ])
    
    print(f"📚 [系统] 检索到 {len(unique_docs)} 条相关资料")
    return {"context": context_text}


def generate_node(state: State):
    """
    节点 B: 拟稿专家。根据检索结果写草稿。
    """
    query = state["messages"][-1].content
    context = state["context"]
    
    # 🔧 修复：简化prompt，适配小模型
    prompt = f"""你是医学助手，根据下方资料回答问题。
资料中没提到的内容说"未提及"，不要编造。

【资料】
{context}

【问题】
{query}

【回答】"""
    
    print("🤖 [AI] 正在基于资料生成回答草稿...")
    response = llm.invoke(prompt)
    return {"messages": [response]}

def human_review_node(state: State):
    """
    节点 C: 审核站台。本身不干活，只作为断点。
    """
    pass

# --- 3. 构建图 (工作流) ---
workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("human_review", human_review_node)

# 连线：开始 -> 检索 -> 生成 -> 审核 -> 结束
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "human_review")
workflow.add_edge("human_review", END)

# 编译：加入记忆和断点
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_review"] # 在进入审核前暂停
)

# --- 4. 运行 Demo ---
if __name__ == "__main__":
    thread_id = "demo_session_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    print("="*60)
    print("🩺 《超越百岁》医学智能助手 (RAG + HITL版) 已启动")
    print("💡 特性：基于专属知识库 + 医生实时审核机制")
    print("="*60)
    
    while True:
        user_input = input("\n👤 患者提问 (输入 q 退出): ")
        if user_input.lower() in ["q", "quit"]: break
        
        # 1. 运行直到断点
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
            pass
            
        # 2. 获取当前状态（AI 的草稿）
        snapshot = app.get_state(config)
        if not snapshot.values: continue
        
        ai_msg = snapshot.values["messages"][-1]
        context_used = snapshot.values.get("context", "")
        
        print("\n" + "-"*30)
        print("👀 [后台] AI 检索到的参考资料片段：")
        print(context_used[:300] + "...") 
        print("-"*30)
        
        print(f"\n📝 [待审核回答]:\n{ai_msg.content}")
        print("\n" + "="*30)
        
        # 3. 医生审核
        feedback = input("👨‍⚕️ 医生操作 [回车=通过 / 输入文字=修改]: ")
        
        if feedback.strip():
            print("✏️  回答已修正。")
            app.update_state(config, {"messages": [AIMessage(content=feedback)]})
        else:
            print("✅ 审核通过。")
            
        # 4. 继续流程
        for event in app.stream(None, config):
            pass
            
        final_state = app.get_state(config)
        print(f"\n📨 [发送给患者]: {final_state.values['messages'][-1].content}")