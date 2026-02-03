"""
科普医疗助手 - 简洁优化版
两种模式：
1. 个人健康顾问 - 需要登录，有记忆功能，个性化建议
2. 医学科普问答 - 无需登录，直接问答，Self-RAG + Web Search
"""
import sys
import os
import uuid
import json
import sqlite3
from datetime import datetime
from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import trim_messages, SystemMessage

# 导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools import medical_tools_list

try:
    from medical_engine import llm, llm_with_tools, vectorstore, web_search_tool
except ImportError:
    print("❌ 错误: 无法导入医学引擎")
    sys.exit(1)

# ============================================================
# 数据库配置
# ============================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "user_memory.db")

# ============================================================
# 持久化存储类
# ============================================================
class PersistentHealthStore:
    """SQLite 持久化存储用户健康档案"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                important INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def user_exists(self, user_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def create_user(self, user_id: str, display_name: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
                (user_id, display_name)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def update_last_active(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
            (user_id,)
        )
        conn.commit()
        conn.close()
    
    def get_user_info(self, user_id: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, display_name, created_at, last_active FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "user_id": row[0],
                "display_name": row[1],
                "created_at": row[2],
                "last_active": row[3]
            }
        return None
    
    def add_health_record(self, user_id: str, category: str, content: str, important: bool = False):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM health_records WHERE user_id = ? AND category = ? AND content = ?",
            (user_id, category, content)
        )
        if cursor.fetchone():
            conn.close()
            return False
        
        cursor.execute(
            "INSERT INTO health_records (user_id, category, content, important) VALUES (?, ?, ?, ?)",
            (user_id, category, content, 1 if important else 0)
        )
        conn.commit()
        conn.close()
        return True
    
    def get_health_records(self, user_id: str) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT category, content, important, created_at FROM health_records WHERE user_id = ? ORDER BY important DESC, created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"category": row[0], "content": row[1], "important": bool(row[2]), "created_at": row[3]}
            for row in rows
        ]
    
    def clear_health_records(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_records WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()


# 初始化全局存储
health_store = PersistentHealthStore(DB_PATH)

# ============================================================
# 配置
# ============================================================
MAX_MESSAGES_BEFORE_SUMMARY = 16
KEEP_RECENT_MESSAGES = 6
_current_thread_id = ""

def set_current_thread_id(thread_id: str):
    global _current_thread_id
    _current_thread_id = thread_id

# ============================================================
# State定义
# ============================================================
class GuidedState(TypedDict):
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

# ============================================================
# 健康信息提取与加载
# ============================================================
def extract_health_info(user_message: str, user_id: str) -> List[dict]:
    """从用户消息中提取健康信息"""
    if not user_id or user_id == "anonymous":
        return []
    
    extract_prompt = f"""
分析用户消息，提取健康相关的个人信息。

用户消息："{user_message}"

提取规则：
1. 身体指标：必须包含完整数值，如"身高165cm"、"体重77kg"，不要拆分
2. 过敏信息：如"对鸡蛋过敏"、"海鲜过敏"（important设为true）
3. 疾病史：如"有高血压"、"糖尿病"（important设为true）
4. 生活习惯：如"每天吸烟"、"不喝酒"
5. 用药情况：如"正在服用降压药"

【重要】提取时保持信息完整，例如：
- "身高165" → content应为"身高165cm"
- "体重77" → content应为"体重77kg"  
- "对鸡蛋过敏" → content应为"鸡蛋过敏"

返回JSON数组：
[{{"category": "身体指标", "content": "身高165cm", "important": false}},
 {{"category": "身体指标", "content": "体重77kg", "important": false}},
 {{"category": "过敏信息", "content": "鸡蛋过敏", "important": true}}]

没有健康信息返回：[]
只返回JSON。
"""
    
    extracted_items = []
    
    try:
        result = llm.invoke(extract_prompt).content.strip()
        
        if "```" in result:
            parts = result.split("```")
            for part in parts:
                if "[" in part:
                    result = part.replace("json", "").strip()
                    break
        
        if result and "[" in result:
            info_list = json.loads(result)
            if not isinstance(info_list, list):
                info_list = [info_list]
            
            for info in info_list:
                if info and isinstance(info, dict) and info.get("content"):
                    added = health_store.add_health_record(
                        user_id=user_id,
                        category=info["category"],
                        content=info["content"],
                        important=info.get("important", False)
                    )
                    if added:
                        print(f"  💾 已记录: [{info['category']}] {info['content']}")
                        extracted_items.append(info)
                        
    except (json.JSONDecodeError, Exception):
        pass
    
    return extracted_items


def load_health_profile(user_id: str) -> str:
    """加载用户健康档案"""
    if not user_id or user_id == "anonymous":
        return ""
    
    records = health_store.get_health_records(user_id)
    if not records:
        return ""
    
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
    
    lines = []
    if important_items:
        lines.append("【⚠️ 重要提醒】")
        lines.extend(important_items)
        lines.append("")
    
    for category, contents in profile_dict.items():
        lines.append(f"【{category}】")
        for c in contents:
            lines.append(f"  • {c}")
    
    return "\n".join(lines)


# ============================================================
# 辅助函数
# ============================================================
def detect_mode(user_input: str) -> str:
    """检测用户意图"""
    keywords_assessment = ["计算", "评估", "BMI", "血压", "体重", "身高", "热量", "心率", "kg", "cm"]
    input_lower = user_input.lower()
    has_numbers = any(char.isdigit() for char in user_input)
    assessment_score = sum(1 for kw in keywords_assessment if kw in input_lower)
    
    if has_numbers or assessment_score > 0:
        return "assessment"
    return "science"


def grade_documents(question: str, docs: List[str]) -> str:
    """评估文档相关性"""
    if not docs:
        return "no"
    
    context = "\n".join(docs[:2])
    prompt = f"""
    评估文档是否与问题相关。
    文档：{context}
    问题：{question}
    只回答：yes 或 no
    """
    score = llm.invoke(prompt).content.strip().lower()
    return "yes" if "yes" in score else "no"


def rewrite_query(question: str) -> str:
    """重写搜索词"""
    prompt = f"原问题检索失败，请重写一个更好的医学搜索词。原问题：{question}\n只输出新的查询词。"
    return llm.invoke(prompt).content.strip()


# ============================================================
# 节点定义
# ============================================================
def router_node(state: GuidedState):
    """路由节点"""
    messages = state["messages"]
    user_id = state.get("user_id", "anonymous")
    question = messages[-1].content
    
    print(f"\n🧭 [分析问题中...]")
    
    # 只有登录用户才提取健康信息
    if user_id and user_id != "anonymous":
        extract_health_info(question, user_id)
    
    health_profile = load_health_profile(user_id) if user_id != "anonymous" else ""
    
    mode = detect_mode(question)
    print(f"  → {'健康评估' if mode == 'assessment' else '知识检索'}")
    
    return {
        "mode": mode,
        "need_tool": mode == "assessment",
        "need_rag": True,
        "need_web": False,
        "loop_step": 0,
        "documents": [],
        "used_web_search": False,
        "health_profile": health_profile,
        "summary": ""
    }


def assessment_tool_node(state: GuidedState):
    """健康评估工具节点"""
    print("📊 [计算健康指标...]")
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
                except Exception as e:
                    results.append(f"❌ 计算错误: {e}")
        output = "\n\n".join(results)
    else:
        output = "⚠️ 请提供具体数据，如 '我170cm，70kg，计算BMI'"
    
    return {"tool_output": output}


def retrieve_node(state: GuidedState):
    """本地检索节点"""
    print("📚 [检索知识库...]")
    question = state["messages"][-1].content
    
    search_query = f"{question} 健康建议" if state.get("tool_output") else question
    docs = vectorstore.similarity_search(search_query, k=4)
    doc_contents = [d.page_content for d in docs]
    
    return {"documents": doc_contents, "loop_step": state["loop_step"] + 1}


def web_search_node(state: GuidedState):
    """Web搜索节点"""
    print("🌐 [联网搜索...]")
    question = state["messages"][-1].content
    
    try:
        results = web_search_tool.invoke({"query": question})
        web_contents = [res['content'] for res in results]
        return {"documents": web_contents, "used_web_search": True}
    except Exception as e:
        return {"documents": ["⚠️ 网络搜索暂时不可用"], "used_web_search": True}


def grade_and_generate_node(state: GuidedState):
    """评分与生成节点"""
    question = state["messages"][-1].content
    docs = state["documents"]
    mode = state.get("mode", "science")
    health_profile = state.get("health_profile", "")
    
    score = grade_documents(question, docs)
    
    if score == "yes":
        print("💡 [生成回答...]")
        context = "\n\n".join(docs)
        source_tag = "(来源: 互联网)" if state["used_web_search"] else "(来源: 医学知识库)"
        
        # 构建记忆上下文（仅健康顾问模式）
        memory_context = ""
        if health_profile:
            memory_context = f"""
【用户健康档案】
{health_profile}
---
"""
        
        if mode == "assessment":
            tool_result = state.get("tool_output", "")
            prompt = f"""
你是专业的健康顾问。根据计算结果和医学知识，给出个性化建议。

{memory_context}
【评估结果】
{tool_result}

【参考资料】{source_tag}
{context}

【问题】{question}

请给出：1. 结果解读 2. 健康建议 3. 注意事项（特别注意过敏史和疾病史）
语气专业但亲切。
"""
        else:
            prompt = f"""
你是医学科普专家。用通俗易懂的语言回答。

{memory_context}
【参考资料】{source_tag}
{context}

【问题】{question}

要求：先简要回答，再展开解释，最后给出实用建议。
"""
        
        answer = llm.invoke(prompt).content
        return {"rag_output": answer, "final_answer": "ready"}
    
    elif state["loop_step"] >= 3:
        if not state["used_web_search"]:
            return {"final_answer": "go_web"}
        else:
            context = "\n\n".join(docs)
            prompt = f"根据有限信息尽力回答：\n资料：{context}\n问题：{question}"
            answer = llm.invoke(prompt).content
            return {"rag_output": answer, "final_answer": "ready"}
    else:
        new_query = rewrite_query(question)
        return {"messages": [HumanMessage(content=new_query)]}


def summarizer_node(state: GuidedState):
    """总结节点"""
    mode = state.get("mode", "science")
    tool_output = state.get("tool_output", "")
    rag_output = state.get("rag_output", "")
    health_profile = state.get("health_profile", "")
    
    profile_note = "\n📋 已参考你的健康档案" if health_profile else ""
    
    if mode == "assessment" and tool_output:
        final_text = f"""
{'═' * 50}
📊 健康评估结果
{'═' * 50}

{tool_output}

{'─' * 50}
💡 建议
{'─' * 50}

{rag_output if rag_output else '暂无额外建议'}{profile_note}

⚠️ 以上仅供参考，具体请咨询医生。
"""
    else:
        final_text = f"""
{'═' * 50}
📖 回答
{'═' * 50}

{rag_output if rag_output else '抱歉，暂时无法找到相关信息。'}{profile_note}

💡 以上信息仅供科普学习，具体请遵医嘱。
"""
    
    return {"final_answer": final_text, "messages": [AIMessage(content=final_text)]}


# ============================================================
# 构建图
# ============================================================
workflow = StateGraph(GuidedState)

workflow.add_node("router", router_node)
workflow.add_node("assessment_tool", assessment_tool_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_loop", grade_and_generate_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("summarizer", summarizer_node)

workflow.add_edge(START, "router")

def route_after_router(state):
    return "assessment_tool" if state["mode"] == "assessment" else "retrieve"

workflow.add_conditional_edges("router", route_after_router)
workflow.add_edge("assessment_tool", "retrieve")
workflow.add_edge("retrieve", "grade_loop")

def route_self_rag(state):
    decision = state.get("final_answer")
    if decision == "ready":
        return "summarizer"
    elif decision == "go_web":
        return "web_search"
    return "retrieve"

workflow.add_conditional_edges("grade_loop", route_self_rag,
    {"summarizer": "summarizer", "web_search": "web_search", "retrieve": "retrieve"}
)

workflow.add_edge("web_search", "grade_loop")
workflow.add_edge("summarizer", END)

# ============================================================
# 编译
# ============================================================
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)


# ============================================================
# 用户界面
# ============================================================
def show_welcome():
    print("""
╔══════════════════════════════════════════════════════════╗
║              🏥 科普医疗智能助手                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║   请选择服务模式：                                        ║
║                                                          ║
║   [1] 🩺 个人健康顾问                                    ║
║       • 记住你的身体数据和健康状况                        ║
║       • 提供个性化的健康评估和建议                        ║
║       • 关闭后下次登录可恢复记忆                          ║
║                                                          ║
║   [2] 📚 医学科普问答                                    ║
║       • 无需登录，直接提问                                ║
║       • 基于医学知识库和网络搜索回答                       ║
║       • 适合了解疾病预防、健康知识等                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


def user_login() -> tuple[str, str]:
    """用户登录（健康顾问模式）"""
    print("""
┌──────────────────────────────────────────────────────────┐
│  👤 登录 / 注册                                          │
│                                                          │
│  老用户：输入你的ID                                       │
│  新用户：按 Enter 创建账号                                │
└──────────────────────────────────────────────────────────┘
""")
    
    user_input = input("🔑 用户ID: ").strip()
    
    if user_input:
        if health_store.user_exists(user_input):
            user_info = health_store.get_user_info(user_input)
            health_store.update_last_active(user_input)
            records = health_store.get_health_records(user_input)
            print(f"\n✅ 欢迎回来，{user_info['display_name']}！")
            if records:
                print(f"   已加载 {len(records)} 条健康记录")
            return user_input, user_info['display_name']
        else:
            print(f"\n❌ ID '{user_input}' 不存在")
            retry = input("   创建新账号？(y/n): ").strip().lower()
            if retry != 'y':
                return user_login()
    
    # 新用户
    display_name = input("\n📝 输入你的名字: ").strip() or "用户"
    user_id = f"{display_name}_{uuid.uuid4().hex[:8]}"
    health_store.create_user(user_id, display_name)
    
    print(f"""
┌──────────────────────────────────────────────────────────┐
│  ✅ 账号创建成功！                                        │
│                                                          │
│  👤 {display_name:<52}│
│  🆔 {user_id:<52}│
│                                                          │
│  ⚠️  请记住你的ID，下次登录需要输入                        │
└──────────────────────────────────────────────────────────┘
""")
    return user_id, display_name


def show_health_profile(user_id: str):
    """显示健康档案"""
    profile = load_health_profile(user_id)
    user_info = health_store.get_user_info(user_id)
    
    if profile:
        print(f"""
┌──────────────────────────────────────────────────────────┐
│  📋 健康档案                                              │
├──────────────────────────────────────────────────────────┤
│  👤 {user_info['display_name'] if user_info else user_id:<52}│
│  🆔 {user_id:<52}│
└──────────────────────────────────────────────────────────┘

{profile}
""")
    else:
        print("\n📋 健康档案为空，告诉我你的身高体重、过敏史等信息，我会记住。\n")


def run_health_advisor():
    """运行健康顾问模式"""
    user_id, display_name = user_login()
    thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
    set_current_thread_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"""
{'━' * 58}
  🩺 健康顾问模式 | {display_name}
  
  /p 查看档案 | /c 清空档案 | /id 查看ID
  /q 返回主菜单 | /qq 退出程序
{'━' * 58}
""")
    
    while True:
        try:
            user_input = input("\n👉 ").strip()
            
            if not user_input:
                continue
            
            if user_input == "/qq":
                print(f"\n👋 再见！你的ID: {user_id}")
                return "exit_program"
            
            if user_input in ["/q", "q"]:
                print(f"\n📋 已保存，你的ID: {user_id}")
                return "back_to_menu"
            
            if user_input == "/p":
                show_health_profile(user_id)
                continue
            
            if user_input == "/c":
                if input("⚠️ 确定清空？(y/n): ").strip().lower() == "y":
                    health_store.clear_health_records(user_id)
                    print("  ✓ 已清空")
                continue
            
            if user_input == "/id":
                print(f"\n🆔 {user_id}")
                continue
            
            # 处理问题
            for event in app.stream(
                {"messages": [HumanMessage(content=user_input)], "user_id": user_id},
                config
            ):
                if "summarizer" in event:
                    print(event["summarizer"]["final_answer"])
            
        except KeyboardInterrupt:
            print(f"\n\n📋 已保存，你的ID: {user_id}")
            return "back_to_menu"
        except Exception as e:
            print(f"\n❌ 出错: {e}")


def run_science_qa():
    """运行医学科普模式"""
    thread_id = f"science_{uuid.uuid4().hex[:8]}"
    set_current_thread_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"""
{'━' * 58}
  📚 医学科普问答
  
  直接输入问题即可
  /q 返回主菜单 | /qq 退出程序
  
  示例：什么是二区训练？/ 如何预防糖尿病？
{'━' * 58}
""")
    
    while True:
        try:
            user_input = input("\n👉 ").strip()
            
            if not user_input:
                continue
            
            if user_input == "/qq":
                print("\n👋 再见！")
                return "exit_program"
            
            if user_input in ["/q", "q"]:
                return "back_to_menu"
            
            # 处理问题（无用户ID，即无记忆）
            for event in app.stream(
                {"messages": [HumanMessage(content=user_input)], "user_id": "anonymous"},
                config
            ):
                if "summarizer" in event:
                    print(event["summarizer"]["final_answer"])
            
        except KeyboardInterrupt:
            return "back_to_menu"
        except Exception as e:
            print(f"\n❌ 出错: {e}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️ 提示: 未配置 TAVILY_API_KEY，联网搜索将不可用\n")
    
    while True:
        show_welcome()
        choice = input("请选择 [1/2] (q退出): ").strip()
        
        if choice == "1":
            result = run_health_advisor()
            if result == "exit_program":
                break
            print()  # 返回菜单时换行
        elif choice == "2":
            result = run_science_qa()
            if result == "exit_program":
                break
            print()
        elif choice.lower() in ["q", "quit", "exit"]:
            print("\n👋 再见！")
            break
        else:
            print("\n⚠️ 请输入 1 或 2\n")