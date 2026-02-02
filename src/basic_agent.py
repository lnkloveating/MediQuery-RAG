import operator
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver # 引入记忆模块

# --- 1. 定义状态 ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 2. 初始化模型 ---
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# --- 3. 定义节点 ---
def chatbot_node(state: AgentState):
    # 直接调用模型，因为有 MemorySaver，state["messages"] 会自动包含历史记录
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# --- 4. 构建图 ---
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 【改进点】加入记忆
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 5. 运行 ---
if __name__ == "__main__":
    # 必须指定 thread_id，这样才有记忆
    config = {"configurable": {"thread_id": "control_group_001"}}
    
    print("🤖 基础版 AI (无知识库) 已启动...")
    print("💡 用途：用于展示未经过 RAG 增强的通用回答效果")
    
    # 【改进点】初始化人设（只在第一次对话前注入）
    # 检查历史消息，如果为空，则插入 SystemMessage
    initial_state = app.get_state(config)
    if not initial_state.values:
        print("🔧 注入初始人设...")
        sys_msg = SystemMessage(content="你是一个医疗AI助手。你没有外部知识库，请仅基于你的训练数据回答问题。")
        app.update_state(config, {"messages": [sys_msg]})

    while True:
        user_input = input("\n患者(你): ")
        if user_input.lower() in ["q", "quit"]: break
        
        # 只需要传入新消息，历史消息由 MemorySaver 自动管理
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
            for value in event.values():
                print(f"Agent(Qwen): {value['messages'][-1].content}")