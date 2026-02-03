"""
科普医疗助手 - 持久化记忆版本
核心功能：
1. 持久化记忆 - 使用 SQLite 保存用户健康档案，关闭终端后不会丢失
2. 用户ID系统 - 新用户自动生成ID，老用户输入ID直接恢复记忆
3. 对话摘要 - 智能压缩历史对话，保留关键信息
4. 健康信息提取 - 自动识别并存储用户的健康数据

使用方式：
- 新用户：直接按 Enter，输入名字，系统自动生成 ID（如 zhang_a8f3b2c1）
- 老用户：输入之前的 ID，直接恢复所有记忆
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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
║                🏥 科普医疗智能助手 (持久化版)                ║
║                                                            ║
║  我可以帮你：                                               ║
║  1  【健康评估】计算BMI、血压评估、热量需求等                  ║
║  2  【医学科普】疾病预防、症状解读、生活建议等                 ║
║                                                            ║
║  🆕 持久化记忆：关闭终端后，你的健康信息不会丢失！             ║
║     • 新用户：按 Enter，输入名字，获得专属ID                  ║
║     • 老用户：输入ID，立即恢复所有记忆                        ║
║                                                            ║
║  💡 提示：我的知识来自《超越百岁》医学书籍及网络搜索           ║
║  ⚠️  注意：建议仅供参考，不能替代专业医疗诊断！               ║
╚════════════════════════════════════════════════════════════╝
"""

# ============================================================
# 数据库配置
# ============================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "user_memory.db")

# ============================================================
# 持久化存储类
# ============================================================
class PersistentHealthStore:
    """
    使用 SQLite 持久化存储用户健康档案
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 健康档案表
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
        
        # 对话摘要表（可选，用于跨会话保留重要对话上下文）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def user_exists(self, user_id: str) -> bool:
        """检查用户是否存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def create_user(self, user_id: str, display_name: str) -> bool:
        """创建新用户"""
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
        """更新用户最后活跃时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
            (user_id,)
        )
        conn.commit()
        conn.close()
    
    def get_user_info(self, user_id: str) -> Optional[dict]:
        """获取用户信息"""
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
        """添加健康记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查是否已存在相同内容（避免重复）
        cursor.execute(
            "SELECT 1 FROM health_records WHERE user_id = ? AND category = ? AND content = ?",
            (user_id, category, content)
        )
        if cursor.fetchone():
            conn.close()
            return False  # 已存在
        
        cursor.execute(
            "INSERT INTO health_records (user_id, category, content, important) VALUES (?, ?, ?, ?)",
            (user_id, category, content, 1 if important else 0)
        )
        conn.commit()
        conn.close()
        return True
    
    def get_health_records(self, user_id: str) -> List[dict]:
        """获取用户所有健康记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT category, content, important, created_at FROM health_records WHERE user_id = ? ORDER BY important DESC, created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "category": row[0],
                "content": row[1],
                "important": bool(row[2]),
                "created_at": row[3]
            }
            for row in rows
        ]
    
    def clear_health_records(self, user_id: str):
        """清空用户健康记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_records WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
    
    def delete_user(self, user_id: str):
        """删除用户及其所有数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_records WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM conversation_summaries WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
    
    def list_all_users(self) -> List[dict]:
        """列出所有用户（用于调试）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, display_name, last_active FROM users ORDER BY last_active DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"user_id": row[0], "display_name": row[1], "last_active": row[2]}
            for row in rows
        ]
    
    def save_conversation_summary(self, user_id: str, thread_id: str, summary: str):
        """保存对话摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_summaries (user_id, thread_id, summary) VALUES (?, ?, ?)",
            (user_id, thread_id, summary)
        )
        conn.commit()
        conn.close()
    
    def get_recent_summaries(self, user_id: str, limit: int = 3) -> List[str]:
        """获取最近的对话摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT summary FROM conversation_summaries WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]


# 初始化全局存储
health_store = PersistentHealthStore(DB_PATH)

# ============================================================
# 记忆配置
# ============================================================
MAX_MESSAGES_BEFORE_SUMMARY = 16
KEEP_RECENT_MESSAGES = 6
DEBUG_MEMORY = False  # 设置为 True 开启调试日志

# 全局变量用于在节点间传递 thread_id
_current_thread_id = ""

def toggle_debug_mode():
    """切换调试模式"""
    global DEBUG_MEMORY
    DEBUG_MEMORY = not DEBUG_MEMORY
    print(f"  调试模式: {'开启' if DEBUG_MEMORY else '关闭'}")

def set_current_thread_id(thread_id: str):
    """设置当前线程ID"""
    global _current_thread_id
    _current_thread_id = thread_id

# ============================================================
# State定义
# ============================================================
class GuidedState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
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
# 健康信息提取函数
# ============================================================
def extract_health_info(user_message: str, user_id: str) -> List[dict]:
    """
    从用户消息中提取健康相关信息，存入持久化数据库
    """
    extract_prompt = f"""
分析以下用户消息，提取健康/医疗相关的个人信息。

用户消息："{user_message}"

需要提取的信息类型：
1. 身体指标：身高、体重、年龄、性别、血压、血糖等
2. 过敏信息：药物过敏、食物过敏等（非常重要！）
3. 疾病史：糖尿病、高血压、心脏病等慢性病
4. 生活习惯：吸烟、饮酒、运动习惯等
5. 用药情况：正在服用的药物

【重要】请返回 JSON 数组格式，包含所有提取到的信息：
[
  {{"category": "类别1", "content": "具体内容1", "important": true/false}},
  {{"category": "类别2", "content": "具体内容2", "important": true/false}}
]

如果没有健康相关信息，返回空数组：[]

注意：
- 过敏信息的 important 必须设为 true
- 疾病史的 important 设为 true
- 每种信息单独一条记录
- 只返回 JSON，不要其他文字
"""
    
    extracted_items = []
    
    try:
        result = llm.invoke(extract_prompt).content.strip()
        
        if DEBUG_MEMORY:
            print(f"  🔍 [DEBUG] LLM 返回: {result[:200]}...")
        
        # 清理 markdown 代码块
        if "```" in result:
            parts = result.split("```")
            for part in parts:
                if "[" in part:
                    result = part.replace("json", "").strip()
                    break
        
        # 解析 JSON
        if result and result != "null" and "[" in result:
            info_list = json.loads(result)
            
            if not isinstance(info_list, list):
                info_list = [info_list]
            
            for info in info_list:
                if info and isinstance(info, dict) and info.get("content"):
                    # 存入数据库
                    added = health_store.add_health_record(
                        user_id=user_id,
                        category=info["category"],
                        content=info["content"],
                        important=info.get("important", False)
                    )
                    
                    if added:
                        print(f"  💾 [持久化记忆] 已记录: [{info['category']}] {info['content']}")
                        extracted_items.append(info)
                    elif DEBUG_MEMORY:
                        print(f"  ℹ️ [DEBUG] 跳过重复记录: {info['content']}")
                        
    except json.JSONDecodeError as e:
        if DEBUG_MEMORY:
            print(f"  ⚠️ [DEBUG] JSON 解析失败: {e}")
    except Exception as e:
        if DEBUG_MEMORY:
            print(f"  ⚠️ [DEBUG] 健康信息提取失败: {e}")
    
    return extracted_items


def load_health_profile(user_id: str) -> str:
    """
    从数据库加载用户的健康档案
    """
    records = health_store.get_health_records(user_id)
    
    if not records:
        return ""
    
    # 按类别整理
    profile_dict = {}
    important_items = []
    
    for record in records:
        category = record["category"]
        content = record["content"]
        important = record["important"]
        
        if category not in profile_dict:
            profile_dict[category] = []
        profile_dict[category].append(content)
        
        if important:
            important_items.append(f"⚠️ {content}")
    
    # 格式化输出
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
# 对话摘要函数
# ============================================================
def summarize_old_messages(messages: list, user_id: str, thread_id: str) -> tuple[str, list]:
    """
    当对话过长时，将旧消息压缩成摘要并保存到数据库
    """
    if len(messages) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return "", messages
    
    print(f"  📝 [对话摘要] 消息数 {len(messages)} 超过阈值，正在压缩...")
    
    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]
    
    conversation_text = []
    for msg in old_messages:
        if hasattr(msg, 'content') and msg.content:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            conversation_text.append(f"{role}: {content}")
    
    summary_prompt = f"""
请总结以下对话的关键信息，重点提取：
1. 用户提到的身体指标（具体数值）
2. 用户的健康状况（疾病、过敏、症状）
3. 用户的主要问题和关注点
4. 助手给出的重要建议

对话内容：
{chr(10).join(conversation_text)}

请用简洁的要点形式总结（不超过300字）：
"""
    
    try:
        summary = llm.invoke(summary_prompt).content.strip()
        
        # 保存到数据库
        health_store.save_conversation_summary(user_id, thread_id, summary)
        
        print(f"  ✓ 摘要生成完成，压缩了 {len(old_messages)} 条消息")
        return summary, recent_messages
    except Exception as e:
        print(f"  ⚠️ 摘要生成失败: {e}")
        return "", recent_messages


# ============================================================
# 辅助函数
# ============================================================
def detect_mode(user_input: str) -> str:
    """智能检测用户意图"""
    keywords_assessment = ["计算", "评估", "BMI", "血压", "体重", "身高", "热量", "心率", "kg", "cm"]
    keywords_science = ["预防", "什么是", "为什么", "怎么", "如何", "有什么", "原因", "作用", "好处"]
    
    input_lower = user_input.lower()
    has_numbers = any(char.isdigit() for char in user_input)
    
    assessment_score = sum(1 for kw in keywords_assessment if kw in input_lower)
    science_score = sum(1 for kw in keywords_science if kw in input_lower)
    
    if has_numbers or assessment_score > 0:
        return "assessment"
    elif science_score > 0:
        return "science"
    else:
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


# ============================================================
# 节点定义
# ============================================================

# 全局变量用于在节点间传递 thread_id
_current_thread_id = ""

def router_node(state: GuidedState):
    """路由节点"""
    messages = state["messages"]
    user_id = state.get("user_id", "anonymous")
    question = messages[-1].content
    
    print(f"\n🧭 [智能路由]")
    
    # 提取并存储健康信息
    extract_health_info(question, user_id)
    
    # 加载用户健康档案
    health_profile = load_health_profile(user_id)
    if health_profile:
        print(f"  📋 已加载用户健康档案")
    
    # 检查是否需要摘要压缩
    summary = ""
    if len(messages) > MAX_MESSAGES_BEFORE_SUMMARY:
        summary, messages = summarize_old_messages(messages, user_id, _current_thread_id)
    
    # 加载历史摘要
    recent_summaries = health_store.get_recent_summaries(user_id, limit=2)
    if recent_summaries:
        summary = "\n---\n".join([summary] + recent_summaries) if summary else "\n---\n".join(recent_summaries)
    
    mode = detect_mode(question)
    print(f"  检测到模式: {'🔢 健康评估' if mode == 'assessment' else '📖 医学科普'}")
    
    if mode == "assessment":
        return {
            "mode": "assessment",
            "need_tool": True,
            "need_rag": True,
            "need_web": False,
            "loop_step": 0,
            "documents": [],
            "used_web_search": False,
            "health_profile": health_profile,
            "summary": summary
        }
    else:
        return {
            "mode": "science",
            "need_tool": False,
            "need_rag": True,
            "need_web": False,
            "loop_step": 0,
            "documents": [],
            "used_web_search": False,
            "health_profile": health_profile,
            "summary": summary
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
    
    health_profile = state.get("health_profile", "")
    summary = state.get("summary", "")
    
    score = grade_documents(question, docs)
    print(f"  评分: {'✓ 相关' if score == 'yes' else '✗ 不相关'}")
    
    if score == "yes":
        print("💡 [生成答案]")
        context = "\n\n".join(docs)
        source_tag = "(来源: 互联网)" if state["used_web_search"] else "(来源: 医学知识库)"
        
        memory_context = ""
        if health_profile:
            memory_context += f"""
【⚠️ 用户健康档案 - 请务必参考】
{health_profile}
"""
        if summary:
            memory_context += f"""
【历史对话摘要】
{summary}
"""
        
        if mode == "assessment":
            tool_result = state.get("tool_output", "")
            prompt = f"""
你是专业的健康顾问。根据计算结果和医学知识，给出个性化建议。

{memory_context}

【健康评估结果】
{tool_result}

【医学知识参考】{source_tag}
{context}

【用户问题】
{question}

请给出：
1. 结果解读（通俗易懂）
2. 健康建议（具体可行，需考虑用户的健康档案）
3. 注意事项（特别注意用户的过敏史和疾病史！）

语气要专业但亲切。
"""
        else:
            prompt = f"""
你是医学科普专家。用通俗易懂的语言解释医学知识。

{memory_context}

【医学知识】{source_tag}
{context}

【问题】
{question}

要求：
1. 先简单回答（2-3句话）
2. 如有必要，展开详细解释
3. 给出实用建议（需考虑用户的健康档案）
4. 如果用户有特殊情况（过敏、疾病），要特别提醒
5. 语言通俗，不要太多专业术语
"""
        
        answer = llm.invoke(prompt).content
        return {"rag_output": answer, "final_answer": "ready"}
    
    elif state["loop_step"] >= 3:
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
        print("  🔄 优化搜索词，重新检索...")
        new_query = rewrite_query(question)
        return {"messages": [HumanMessage(content=new_query)]}


def summarizer_node(state: GuidedState):
    """总结节点"""
    mode = state.get("mode", "science")
    tool_output = state.get("tool_output", "")
    rag_output = state.get("rag_output", "")
    health_profile = state.get("health_profile", "")
    
    profile_note = ""
    if health_profile:
        profile_note = "\n📋 已参考你的健康档案生成个性化建议"
    
    if mode == "assessment" and tool_output:
        final_text = f"""
╔═══════════════════════════════════════════════════════════╗
║                    🔢 健康评估结果                        ║
╚═══════════════════════════════════════════════════════════╝

{tool_output}

{'─' * 60}

📖 【医学建议】
{rag_output if rag_output else '暂无额外建议'}
{profile_note}

{'─' * 60}

⚠️  重要提示：
本评估仅供参考，不能替代专业医疗诊断。
如有健康问题，请咨询专业医生。
"""
    else:
        final_text = f"""
╔═══════════════════════════════════════════════════════════╗
║                    📖 医学科普解答                         ║
╚═══════════════════════════════════════════════════════════╝

{rag_output if rag_output else '抱歉，暂时无法找到相关信息。'}
{profile_note}

{'─' * 60}

💡 温馨提示：
以上信息来自医学知识库和可靠来源，仅供科普学习。
具体治疗方案请遵医嘱。
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

# ============================================================
# 编译
# ============================================================
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)


# ============================================================
# 用户交互命令
# ============================================================
def show_health_profile(user_id: str):
    """显示用户健康档案"""
    profile = load_health_profile(user_id)
    user_info = health_store.get_user_info(user_id)
    
    if profile:
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║                    📋 你的健康档案                         ║
╚═══════════════════════════════════════════════════════════╝

👤 用户: {user_info['display_name'] if user_info else user_id}
🆔 ID: {user_id}
📅 创建于: {user_info['created_at'] if user_info else '未知'}

{profile}
""")
    else:
        print("\n📋 你的健康档案为空。告诉我你的身高体重、过敏史等信息，我会记住的！\n")


def clear_health_profile(user_id: str):
    """清空用户健康档案"""
    health_store.clear_health_records(user_id)
    print("  ✓ 健康档案已清空（用户账号保留）")


def list_users():
    """列出所有用户"""
    users = health_store.list_all_users()
    if users:
        print("\n📋 已注册用户列表：")
        print("─" * 50)
        for u in users:
            print(f"  🆔 {u['user_id']}")
            print(f"     名字: {u['display_name']}")
            print(f"     最后活跃: {u['last_active']}")
            print()
    else:
        print("\n📋 暂无注册用户\n")


def show_mode_menu():
    print("""
请选择使用模式：

  1️⃣  【健康评估】计算健康指标，获取个性化建议
  2️⃣  【医学科普】学习疾病预防、症状解读等知识
  
  💡 或者直接提问，系统会自动识别！
  
  📌 命令：
     /profile  - 查看健康档案
     /clear    - 清空健康档案
     /id       - 查看你的用户ID
     /users    - 列出所有用户（调试）
     /debug    - 开启/关闭调试模式
     /new      - 开始新会话（保留记忆）
  
输入 1 或 2 选择模式，或直接输入问题：
""")


# ============================================================
# 用户登录/注册
# ============================================================
def user_login() -> tuple[str, str]:
    """
    用户登录流程
    返回: (user_id, display_name)
    """
    print("""
╔════════════════════════════════════════════════════════════╗
║                      👤 用户登录                            ║
╠════════════════════════════════════════════════════════════╣
║  • 老用户：输入你的ID（如 zhang_a8f3b2c1）                   ║
║  • 新用户：直接按 Enter，然后输入名字                         ║
╚════════════════════════════════════════════════════════════╝
""")
    
    user_input = input("🔑 请输入用户ID（新用户按Enter）: ").strip()
    
    if user_input:
        # 尝试登录
        if health_store.user_exists(user_input):
            user_info = health_store.get_user_info(user_input)
            health_store.update_last_active(user_input)
            print(f"\n✅ 欢迎回来，{user_info['display_name']}！")
            
            # 显示已有的健康档案预览
            records = health_store.get_health_records(user_input)
            if records:
                print(f"   📋 已加载 {len(records)} 条健康记录")
            
            return user_input, user_info['display_name']
        else:
            print(f"\n❌ 用户ID '{user_input}' 不存在")
            retry = input("   是否创建新账号？(y/n): ").strip().lower()
            if retry != 'y':
                return user_login()  # 重新登录
    
    # 新用户注册
    print("\n📝 创建新账号")
    display_name = input("   请输入你的名字: ").strip()
    
    if not display_name:
        display_name = "匿名用户"
    
    # 生成唯一ID
    user_id = f"{display_name}_{uuid.uuid4().hex[:8]}"
    
    # 创建用户
    health_store.create_user(user_id, display_name)
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║                    ✨ 账号创建成功！                         ║
╠════════════════════════════════════════════════════════════╣
║  👤 名字: {display_name:<47}║
║  🆔 ID:   {user_id:<47}║
║                                                            ║
║  ⚠️  请牢记你的ID，下次登录时需要输入！                       ║
╚════════════════════════════════════════════════════════════╝
""")
    
    return user_id, display_name


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print(WELCOME_MESSAGE)
    
    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️  提示: 未配置 TAVILY_API_KEY，联网搜索将不可用")
        print("   如需使用，请访问 https://tavily.com 获取API密钥\n")
    
    # 用户登录
    user_id, display_name = user_login()
    
    # 创建会话
    thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
    set_current_thread_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n   会话ID: {thread_id}")
    print("━" * 60)
    
    current_mode = None
    
    while True:
        try:
            if current_mode is None:
                show_mode_menu()
            
            user_input = input("👉 ").strip()
            
            # 退出
            if user_input.lower() in ["q", "quit", "exit"]:
                print(f"\n👋 再见，{display_name}！")
                print(f"   你的健康信息已保存，下次用ID登录即可恢复：{user_id}")
                break
            
            # 命令处理
            if user_input == "/profile":
                show_health_profile(user_id)
                continue
            
            if user_input == "/clear":
                confirm = input("⚠️ 确定要清空健康档案吗？(y/n): ").strip().lower()
                if confirm == "y":
                    clear_health_profile(user_id)
                continue
            
            if user_input == "/id":
                print(f"\n🆔 你的用户ID: {user_id}")
                print(f"   （下次登录时输入此ID即可恢复记忆）\n")
                continue
            
            if user_input == "/users":
                list_users()
                continue
            
            if user_input == "/debug":
                toggle_debug_mode()
                continue
            
            if user_input == "/new":
                thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
                set_current_thread_id(thread_id)
                config = {"configurable": {"thread_id": thread_id}}
                current_mode = None
                print(f"✨ 新会话: {thread_id}")
                print("   📋 健康档案已保留\n")
                continue
            
            if user_input == "/back":
                current_mode = None
                continue
            
            if not user_input:
                continue
            
            if user_input == "1":
                current_mode = "assessment"
                print("\n请输入你的问题（或输入 /back 返回）：")
                continue
            elif user_input == "2":
                current_mode = "science"
                print("\n请输入你的问题（或输入 /back 返回）：")
                continue
            
            # 处理问题
            print("\n" + "━" * 60)
            
            final_res = None
            for event in app.stream(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": user_id
                },
                config
            ):
                if "summarizer" in event:
                    final_res = event["summarizer"]["final_answer"]
            
            if final_res:
                print(final_res)
            
            print("\n" + "━" * 60)
            print("\n💬 继续提问，或输入 /back 返回主菜单")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 再见！你的ID: {user_id}")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()
            print("请重新输入或输入 /back 返回主菜单\n")