import operator
from typing import Annotated, TypedDict, Union
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- 1. 定义状态 (State) ---
# 这是 Agent 的"短期记忆"。它会保存对话历史，供所有节点读取。
class AgentState(TypedDict):
    # add_messages 是 LangGraph 的魔法，它会自动把新消息追加到列表里，而不是覆盖
    messages: Annotated[list[BaseMessage], add_messages]

# --- 2. 初始化模型 (Engine) ---
# 连接你那台 OMEN 8 Pro 上的 Ollama
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,  # 医疗场景设为 0，让它严谨点，别瞎编
)

# --- 3. 定义节点 (Nodes) ---
# 节点就是"干活的人"。目前我们只有一个全能节点：chatbot

def chatbot_node(state: AgentState):
    """
    这是最基础的对话节点。
    它接收当前的状态（state），调用大模型，然后返回生成的回答。
    """
    # 获取历史消息
    messages = state["messages"]
    
    # 调用 Qwen2.5
    response = llm.invoke(messages)
    
    # 返回结果，LangGraph 会自动把它加到 state["messages"] 里
    return {"messages": [response]}

# --- 4. 构建图 (Build the Graph) ---
# 这是 Agent 的"指挥中心"

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("chatbot", chatbot_node)

# 添加边 (Edges)
# 逻辑：开始 -> 聊天 -> 结束
# 以后我们会在这里加：开始 -> 检索(RAG) -> 检查(Safety) -> 聊天
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 编译图 (Compile)
# 这一步会把你的逻辑变成一个可执行的程序
app = workflow.compile()

# --- 5. 本地测试代码 ---
if __name__ == "__main__":
    print("🏥 SafeMed-Agent 核心引擎已启动 (按 'q' 退出)...")
    
    # 给它注入一个“人设”，让它知道自己是医疗助手
    sys_msg = SystemMessage(content="你是一个专业的医疗AI助手。请用中文回答，保持严谨。")
    
    # 模拟简单的终端对话
    while True:
        user_input = input("\n患者(你): ")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("再见！")
            break
            
        # 运行图
        # config={"configurable": {"thread_id": "1"}} 以后用于持久化记忆
        inputs = {"messages": [sys_msg, HumanMessage(content=user_input)]}
        
        # stream 方法让字一个一个蹦出来，看着更爽
        for event in app.stream(inputs):
            for value in event.values():
                print(f"Agent(Qwen): {value['messages'][-1].content}")