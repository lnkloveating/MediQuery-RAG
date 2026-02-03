"""
科普医疗助手 - 双模式版本
模式1: 健康评估（引导式输入）
模式2: 医学科普（自由问答）
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

# 导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools import medical_tools_list

try:
    from medical_engine import llm, llm_with_tools, vectorstore, web_search_tool
except ImportError:
    print("❌ 错误: 无法导入医学引擎")
    sys.exit(1)

# --- 配置 ---
WELCOME_MESSAGE = """
╔════════════════════════════════════════════════════════════╗
║                🏥 科普医疗智能助手                          ║
║                                                            ║
║  我可以帮你：                                               ║
║  1  【健康评估】计算BMI、血压评估、热量需求等                  ║
║  2  【医学科普】疾病预防、症状解读、生活建议等                 ║
║                                                            ║
║  💡 提示：我的知识来自《超越百岁》医学书籍及网络搜索           ║
║  ⚠️  注意：建议仅供参考，不能替代专业医疗诊断！               ║
╚════════════════════════════════════════════════════════════╝
"""

# 健康评估工具说明
ASSESSMENT_TOOLS = """
可用的健康评估工具：

 基础指标：
  1. BMI计算 - 需要：身高(cm)、体重(kg)
  2. 血压评估 - 需要：收缩压、舒张压
  3. 理想体重 - 需要：身高(cm)、性别
"""

# 科普示例问题
SCIENCE_EXAMPLES = """
医学科普示例问题：

🩺 疾病预防：
  • "如何预防糖尿病？"
  • "怎样降低心脏病风险？"
  • "预防阿尔茨海默病的方法？"

🏃 运动健康：
  • "什么是二区训练？"
  • "运动对健康有什么好处？"
  • "如何科学减肥？"

🍎 饮食营养：
  • "糖尿病患者怎么吃？"
  • "高血压要注意什么饮食？"
  • "果糖为什么会引发疾病？"

😴 睡眠与健康：
  • "睡眠不好有什么危害？"
  • "如何改善睡眠质量？"
  • "深度睡眠有什么作用？"
"""

# --- State定义 ---
class GuidedState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str  # "assessment" | "science" | None
    need_tool: bool
    need_rag: bool
    need_web: bool
    
    tool_output: str
    rag_output: str
    final_answer: str
    
    documents: List[str]
    loop_step: int
    used_web_search: bool

# --- 辅助函数 ---

def detect_mode(user_input: str) -> str:
    """智能检测用户意图"""
    keywords_assessment = ["计算", "评估", "BMI", "血压", "体重", "身高", "热量", "心率", "kg", "cm"]
    keywords_science = ["预防", "什么是", "为什么", "怎么", "如何", "有什么", "原因", "作用", "好处"]
    
    input_lower = user_input.lower()
    
    # 检测数字（通常是计算类问题）
    has_numbers = any(char.isdigit() for char in user_input)
    
    # 关键词匹配
    assessment_score = sum(1 for kw in keywords_assessment if kw in input_lower)
    science_score = sum(1 for kw in keywords_science if kw in input_lower)
    
    if has_numbers or assessment_score > 0:
        return "assessment"
    elif science_score > 0:
        return "science"
    else:
        return "science"  # 默认科普模式


def grade_documents(question: str, docs: List[str]) -> str:
    """评估文档相关性"""
    if not docs: return "no"
    
    context = "\n".join(docs[:2])
    prompt = f"""
    评估文档是否与问题相关。
    文档：{context}
    问题：{question}
    
    如果文档能提供线索，回答 'yes'，否则 'no'。
    只回答：yes 或 no
    """
    score = llm.invoke(prompt).content.strip().lower()
    return "yes" if "yes" in score else "no"


def rewrite_query(question: str) -> str:
    """重写搜索词"""
    prompt = f"""
    原问题检索失败，请重写一个更好的医学搜索词。
    原问题：{question}
    只输出新的查询词。
    """
    return llm.invoke(prompt).content.strip()

# --- 节点定义 ---

def router_node(state: GuidedState):
    """路由节点"""
    question = state["messages"][-1].content
    
    # 智能检测模式
    mode = detect_mode(question)
    
    print(f"\n🧭 [智能路由]")
    print(f"  检测到模式: {'🔢 健康评估' if mode == 'assessment' else '📖 医学科普'}")
    
    # 判断需要什么
    if mode == "assessment":
        return {
            "mode": "assessment",
            "need_tool": True,
            "need_rag": True,  # 评估后也给建议
            "need_web": False,
            "loop_step": 0,
            "documents": [],
            "used_web_search": False
        }
    else:
        return {
            "mode": "science",
            "need_tool": False,
            "need_rag": True,
            "need_web": False,
            "loop_step": 0,
            "documents": [],
            "used_web_search": False
        }


def assessment_tool_node(state: GuidedState):
    """健康评估工具节点"""
    print("🔢 [健康评估] 正在计算...")
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
                    results.append(f"📊 {str(res)}")
                    print(f"  ✓ 使用工具: {call['name']}")
                except Exception as e:
                    results.append(f"❌ 计算错误: {e}")
        output = "\n\n".join(results)
    else:
        output = "⚠️ 未能识别出具体的计算请求。\n💡 提示：请提供明确的数据，如 '我170cm，70kg，BMI多少？'"
    
    return {"tool_output": output}


def retrieve_node(state: GuidedState):
    """本地检索节点"""
    print("📚 [知识库检索]")
    question = state["messages"][-1].content
    
    # 如果是评估模式，加上工具结果一起检索
    if state.get("tool_output"):
        search_query = f"{question} 健康建议"
    else:
        search_query = question
    
    docs = vectorstore.similarity_search(search_query, k=4)
    doc_contents = [d.page_content for d in docs]
    
    print(f"  找到 {len(doc_contents)} 条相关资料")
    
    return {"documents": doc_contents, "loop_step": state["loop_step"] + 1}


def web_search_node(state: GuidedState):
    """Web搜索节点"""
    print("🌐 [联网搜索] 本地知识库无答案，正在搜索互联网...")
    question = state["messages"][-1].content
    
    try:
        results = web_search_tool.invoke({"query": question})
        web_contents = [res['content'] for res in results]
        print(f"  ✓ 获取了 {len(web_contents)} 条网络结果")
        return {"documents": web_contents, "used_web_search": True}
    except Exception as e:
        print(f"  ❌ 联网搜索失败: {e}")
        return {"documents": ["⚠️ 网络搜索暂时不可用"], "used_web_search": True}


def grade_and_generate_node(state: GuidedState):
    """评分与生成节点"""
    question = state["messages"][-1].content
    docs = state["documents"]
    mode = state.get("mode", "science")
    
    # 评分
    score = grade_documents(question, docs)
    print(f"  评分: {'✓ 相关' if score == 'yes' else '✗ 不相关'}")
    
    if score == "yes":
        # 生成答案
        print("💡 [生成答案]")
        context = "\n\n".join(docs)
        source_tag = "(来源: 互联网)" if state["used_web_search"] else "(来源: 医学知识库)"
        
        if mode == "assessment":
            # 评估模式：结合计算结果给建议
            tool_result = state.get("tool_output", "")
            prompt = f"""
            你是专业的健康顾问。根据计算结果和医学知识，给出建议。
            
            【健康评估结果】
            {tool_result}
            
            【医学知识参考】{source_tag}
            {context}
            
            【用户问题】
            {question}
            
            请给出：
            1. 结果解读（通俗易懂）
            2. 健康建议（具体可行）
            3. 注意事项
            
            语气要专业但亲切，像医生和朋友的结合。
            """
        else:
            # 科普模式：清晰解释
            prompt = f"""
            你是医学科普专家。用通俗易懂的语言解释医学知识。
            
            【医学知识】{source_tag}
            {context}
            
            【问题】
            {question}
            
            要求：
            1. 先简单回答（2-3句话）
            2. 如有必要，展开详细解释
            3. 给出实用建议
            4. 语言通俗，不要太多专业术语
            """
        
        answer = llm.invoke(prompt).content
        return {"rag_output": answer, "final_answer": "ready"}
    
    elif state["loop_step"] >= 3:
        # 超过重试次数
        if not state["used_web_search"]:
            print("  ⚠️ 本地搜索失败，转入联网搜索")
            return {"final_answer": "go_web"}
        else:
            print("  ⚠️ 联网也无法找到，尝试给出答案")
            context = "\n\n".join(docs)
            prompt = f"根据有限信息尽力回答：\n资料：{context}\n问题：{question}"
            answer = llm.invoke(prompt).content
            return {"rag_output": answer, "final_answer": "ready"}
    else:
        # 重写查询
        print("  🔄 优化搜索词，重新检索...")
        new_query = rewrite_query(question)
        return {"messages": [HumanMessage(content=new_query)]}


def summarizer_node(state: GuidedState):
    """总结节点"""
    mode = state.get("mode", "science")
    tool_output = state.get("tool_output", "")
    rag_output = state.get("rag_output", "")
    
    if mode == "assessment" and tool_output:
        # 评估模式：结构化输出
        final_text = f"""
╔═══════════════════════════════════════════════════════════╗
║                    🔢 健康评估结果                        ║
╚═══════════════════════════════════════════════════════════╝

{tool_output}

{'─' * 60}

📖 【医学建议】
{rag_output if rag_output else '暂无额外建议'}

{'─' * 60}

⚠️  重要提示：
本评估仅供参考，不能替代专业医疗诊断。
如有健康问题，请咨询专业医生。
"""
    else:
        # 科普模式：简洁输出
        final_text = f"""
╔═══════════════════════════════════════════════════════════╗
║                    📖 医学科普解答                         ║
╚═══════════════════════════════════════════════════════════╝

{rag_output if rag_output else '抱歉，暂时无法找到相关信息。'}

{'─' * 60}

💡 温馨提示：
以上信息来自医学知识库和可靠来源，仅供科普学习。
具体治疗方案请遵医嘱。
"""
    
    return {"final_answer": final_text, "messages": [AIMessage(content=final_text)]}

# --- 构建图 ---
workflow = StateGraph(GuidedState)

workflow.add_node("router", router_node)
workflow.add_node("assessment_tool", assessment_tool_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_loop", grade_and_generate_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("summarizer", summarizer_node)

workflow.add_edge(START, "router")

def route_after_router(state):
    if state["mode"] == "assessment":
        return "assessment_tool"
    else:
        return "retrieve"

workflow.add_conditional_edges("router", route_after_router)
workflow.add_edge("assessment_tool", "retrieve")
workflow.add_edge("retrieve", "grade_loop")

def route_self_rag(state):
    decision = state.get("final_answer")
    if decision == "ready":
        return "summarizer"
    elif decision == "go_web":
        return "web_search"
    else:
        return "retrieve"

workflow.add_conditional_edges("grade_loop", route_self_rag,
    {"summarizer": "summarizer", "web_search": "web_search", "retrieve": "retrieve"}
)

workflow.add_edge("web_search", "grade_loop")
workflow.add_edge("summarizer", END)

# --- 编译 ---
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)

# --- 交互式菜单 ---
def show_mode_menu():
    """显示模式选择菜单"""
    print("""
请选择使用模式：

  1️⃣  【健康评估】计算健康指标，获取个性化建议
  2️⃣  【医学科普】学习疾病预防、症状解读等知识
  
  💡 或者直接提问，系统会自动识别！
  
输入 1 或 2 选择模式，或直接输入问题：
""")

def show_assessment_guide():
    """显示评估引导"""
    print(ASSESSMENT_TOOLS)
    print("\n请输入你的问题（或输入 /back 返回）：")

def show_science_guide():
    """显示科普引导"""
    print(SCIENCE_EXAMPLES)
    print("\n请输入你的问题（或输入 /back 返回）：")

# --- 运行 ---
if __name__ == "__main__":
    # 欢迎界面
    print(WELCOME_MESSAGE)
    
    # API密钥检查
    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️  提示: 未配置 TAVILY_API_KEY，联网搜索将不可用")
        print("   如需使用，请访问 https://tavily.com 获取API密钥\n")
    
    # 会话管理
    user_id = input("👤 输入你的名字（或按Enter使用临时会话）: ").strip()
    thread_id = user_id if user_id else str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n✨ 会话已建立: {thread_id}")
    print("━" * 60)
    
    current_mode = None  # None | "assessment" | "science"
    
    while True:
        try:
            # 根据状态显示不同菜单
            if current_mode is None:
                show_mode_menu()
            
            user_input = input("👉 ").strip()
            
            # 特殊命令
            if user_input.lower() in ["q", "quit", "exit"]:
                print("\n👋 再见！祝你健康！")
                break
            
            if user_input == "/new":
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                current_mode = None
                print(f"✨ 新会话: {thread_id}\n")
                continue
            
            if user_input == "/back":
                current_mode = None
                continue
            
            if not user_input:
                continue
            
            # 模式选择
            if user_input == "1":
                current_mode = "assessment"
                show_assessment_guide()
                continue
            elif user_input == "2":
                current_mode = "science"
                show_science_guide()
                continue
            
            # 处理问题
            print("\n" + "━" * 60)
            
            final_res = None
            for event in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config
            ):
                if "summarizer" in event:
                    final_res = event["summarizer"]["final_answer"]
            
            if final_res:
                print(final_res)
            
            print("\n" + "━" * 60)
            
            # 继续提问提示
            print("\n💬 继续提问，或输入 /back 返回主菜单")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            print("请重新输入或输入 /back 返回主菜单\n")