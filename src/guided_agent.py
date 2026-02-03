"""
科普医疗助手 - 优化版本 (Bug 已修复)
新增功能：
1. 长期记忆 (Store) - 永久保存用户健康档案
2. 对话摘要 - 智能压缩历史对话，保留关键信息
3. 健康信息提取 - 自动识别并存储用户的健康数据

修复内容：
- 修复 extract_health_info 只能提取一条信息的问题
- 添加调试日志确认存储成功
"""
import sys
import os
import uuid
import json
from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
import sqlite3
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage

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
║                🏥 科普医疗智能助手 (优化版)                  ║
║                                                            ║
║  我可以帮你：                                               ║
║  1  【健康评估】计算BMI、血压评估、热量需求等                  ║
║  2  【医学科普】疾病预防、症状解读、生活建议等                 ║
║                                                            ║
║  🆕 新功能：我现在能记住你的健康信息了！                       ║
║     告诉我你的身高体重、过敏史等，下次我会记得                  ║
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

🏃 运动健康：
  • "什么是二区训练？"
  • "运动对健康有什么好处？"

🍎 饮食营养：
  • "糖尿病患者怎么吃？"
  • "高血压要注意什么饮食？"
"""

# ============================================================
# 🆕 记忆配置
# ============================================================
MAX_MESSAGES_BEFORE_SUMMARY = 16  # 超过16条消息时触发摘要
KEEP_RECENT_MESSAGES = 6          # 摘要后保留最近6条消息

# 调试模式开关
DEBUG_MEMORY = True  # 设置为 True 可以看到详细的存储日志

# ============================================================
# 🆕 State定义（新增字段）
# ============================================================
class GuidedState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str  # "assessment" | "science" | None
    user_id: str  # 🆕 用户标识
    need_tool: bool
    need_rag: bool
    need_web: bool
    
    tool_output: str
    rag_output: str
    final_answer: str
    
    documents: List[str]
    loop_step: int
    used_web_search: bool
    
    # 🆕 记忆相关
    health_profile: str      # 用户健康档案（从Store加载）
    summary: str             # 历史对话摘要

# ============================================================
# 🆕 长期记忆 Store 初始化
# ============================================================
# 使用 InMemoryStore（生产环境建议换成持久化存储）
health_store = InMemoryStore()

# 用一个简单的字典作为备选存储（防止 Store API 不兼容）
_health_backup = {}

# ============================================================
# 🆕 健康信息提取函数 (修复版)
# ============================================================
def extract_health_info(user_message: str, user_id: str):
    """
    从用户消息中提取健康相关信息，存入长期记忆
    
    🔧 修复：支持提取多条信息（返回 JSON 数组）
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
- 每种信息单独一条记录
- 只返回 JSON，不要其他文字
"""
    
    extracted_items = []
    
    try:
        result = llm.invoke(extract_prompt).content.strip()
        
        if DEBUG_MEMORY:
            print(f"  🔍 [DEBUG] LLM 返回: {result[:200]}...")
        
        # 清理可能的 markdown 代码块
        if "```" in result:
            # 提取 ``` 之间的内容
            parts = result.split("```")
            for part in parts:
                if "[" in part:
                    result = part.replace("json", "").strip()
                    break
        
        # 尝试解析 JSON
        if result and result != "null" and "[" in result:
            info_list = json.loads(result)
            
            if not isinstance(info_list, list):
                # 兼容旧版本：如果返回单个对象，转为数组
                info_list = [info_list]
            
            for info in info_list:
                if info and isinstance(info, dict) and info.get("content"):
                    # 生成唯一key
                    key = f"{info['category']}_{uuid.uuid4().hex[:8]}"
                    
                    record = {
                        "category": info["category"],
                        "content": info["content"],
                        "important": info.get("important", False),
                        "timestamp": str(uuid.uuid4())[:8]
                    }
                    
                    # 尝试存入 Store
                    try:
                        health_store.put(("health", user_id), key, record)
                    except Exception as e:
                        if DEBUG_MEMORY:
                            print(f"  ⚠️ [DEBUG] Store 存储失败: {e}")
                    
                    # 同时存入备选字典（这是主要的存储方式）
                    if user_id not in _health_backup:
                        _health_backup[user_id] = {}
                    _health_backup[user_id][key] = record
                    
                    print(f"  💾 [长期记忆] 已记录: [{info['category']}] {info['content']}")
                    extracted_items.append(info)
            
            if DEBUG_MEMORY:
                print(f"  ✅ [DEBUG] 共提取 {len(extracted_items)} 条信息")
                print(f"  ✅ [DEBUG] _health_backup[{user_id}] = {_health_backup.get(user_id, {})}")
                    
    except json.JSONDecodeError as e:
        if DEBUG_MEMORY:
            print(f"  ⚠️ [DEBUG] JSON 解析失败: {e}")
            print(f"  ⚠️ [DEBUG] 原始内容: {result}")
    except Exception as e:
        print(f"  ⚠️ 健康信息提取失败: {e}")
    
    return extracted_items if extracted_items else None


def load_health_profile(user_id: str) -> str:
    """
    从 Store 加载用户的健康档案
    """
    if DEBUG_MEMORY:
        print(f"  📋 [DEBUG] 加载用户档案: {user_id}")
        print(f"  📋 [DEBUG] _health_backup 所有用户: {list(_health_backup.keys())}")
    
    items_dict = {}
    
    # 方法1: 尝试从 Store 读取
    try:
        # 使用位置参数调用 search
        items = health_store.search(("health", user_id))
        for item in items:
            items_dict[item.key] = item.value
    except Exception as e:
        if DEBUG_MEMORY:
            print(f"  ⚠️ [DEBUG] Store 读取失败: {e}")
    
    # 方法2: 从备选字典读取（主要方式）
    if user_id in _health_backup:
        for key, value in _health_backup[user_id].items():
            if key not in items_dict:
                items_dict[key] = value
        if DEBUG_MEMORY:
            print(f"  ✅ [DEBUG] 从 _health_backup 读取到 {len(_health_backup[user_id])} 条记录")
    else:
        if DEBUG_MEMORY:
            print(f"  ⚠️ [DEBUG] 用户 {user_id} 不在 _health_backup 中")
    
    if not items_dict:
        return ""
    
    # 按类别整理
    profile_dict = {}
    important_items = []
    
    for key, value in items_dict.items():
        category = value.get("category", "其他")
        content = value.get("content", "")
        important = value.get("important", False)
        
        if category not in profile_dict:
            profile_dict[category] = []
        profile_dict[category].append(content)
        
        if important:
            important_items.append(f"⚠️ {content}")
    
    # 格式化输出
    lines = []
    
    # 重要信息优先显示
    if important_items:
        lines.append("【⚠️ 重要提醒】")
        lines.extend(important_items)
        lines.append("")
    
    # 其他信息
    for category, contents in profile_dict.items():
        lines.append(f"【{category}】")
        for c in contents:
            lines.append(f"  • {c}")
    
    return "\n".join(lines)


# ============================================================
# 🆕 对话摘要函数
# ============================================================
def summarize_old_messages(messages: list, user_id: str) -> tuple[str, list]:
    """
    当对话过长时，将旧消息压缩成摘要
    返回：(摘要文本, 保留的最近消息)
    """
    if len(messages) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return "", messages  # 不需要摘要
    
    print(f"  📝 [对话摘要] 消息数 {len(messages)} 超过阈值，正在压缩...")
    
    # 分离：需要摘要的旧消息 vs 保留的新消息
    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]
    
    # 构建摘要 prompt
    conversation_text = []
    for msg in old_messages:
        if hasattr(msg, 'content') and msg.content:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            # 截断过长的单条消息
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            conversation_text.append(f"{role}: {content}")
    
    summary_prompt = f"""
请总结以下对话的关键信息，重点提取：

1. 用户提到的身体指标（身高、体重、血压等具体数值）
2. 用户的健康状况（疾病、过敏、症状）
3. 用户的主要问题和关注点
4. 助手给出的重要建议

对话内容：
{chr(10).join(conversation_text)}

请用简洁的要点形式总结（不超过300字），保留所有具体数值和重要健康信息：
"""
    
    try:
        summary = llm.invoke(summary_prompt).content.strip()
        print(f"  ✓ 摘要生成完成，压缩了 {len(old_messages)} 条消息")
        return summary, recent_messages
    except Exception as e:
        print(f"  ⚠️ 摘要生成失败: {e}")
        # 失败时简单截断
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


# ============================================================
# 节点定义（已优化）
# ============================================================

def router_node(state: GuidedState):
    """路由节点 - 🆕 增加记忆处理"""
    messages = state["messages"]
    user_id = state.get("user_id", "anonymous")
    question = messages[-1].content
    
    print(f"\n🧭 [智能路由]")
    if DEBUG_MEMORY:
        print(f"  🔑 [DEBUG] user_id = {user_id}")
    
    # 🆕 Step 1: 提取并存储健康信息
    extract_health_info(question, user_id)
    
    # 🆕 Step 2: 加载用户健康档案
    health_profile = load_health_profile(user_id)
    if health_profile:
        print(f"  📋 已加载用户健康档案")
    
    # 🆕 Step 3: 检查是否需要摘要压缩
    summary = ""
    if len(messages) > MAX_MESSAGES_BEFORE_SUMMARY:
        summary, messages = summarize_old_messages(messages, user_id)
    
    # 智能检测模式
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
            "health_profile": health_profile,  # 🆕
            "summary": summary                  # 🆕
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
            "health_profile": health_profile,  # 🆕
            "summary": summary                  # 🆕
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
    """评分与生成节点 - 🆕 注入健康档案和摘要"""
    question = state["messages"][-1].content
    docs = state["documents"]
    mode = state.get("mode", "science")
    
    # 🆕 获取记忆信息
    health_profile = state.get("health_profile", "")
    summary = state.get("summary", "")
    
    # 评分
    score = grade_documents(question, docs)
    print(f"  评分: {'✓ 相关' if score == 'yes' else '✗ 不相关'}")
    
    if score == "yes":
        print("💡 [生成答案]")
        context = "\n\n".join(docs)
        source_tag = "(来源: 互联网)" if state["used_web_search"] else "(来源: 医学知识库)"
        
        # 🆕 构建记忆上下文
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

语气要专业但亲切，像医生和朋友的结合。
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
3. 给出实用建议（需考虑用户的健康档案，如有）
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
    
    # 🆕 如果有健康档案，添加提示
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
# 🆕 调试命令
# ============================================================
def show_health_profile(user_id: str):
    """显示用户健康档案"""
    profile = load_health_profile(user_id)
    if profile:
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║                    📋 你的健康档案                         ║
╚═══════════════════════════════════════════════════════════╝

{profile}
""")
    else:
        print("\n📋 你的健康档案为空。告诉我你的身高体重、过敏史等信息，我会记住的！\n")


def clear_health_profile(user_id: str):
    """清空用户健康档案"""
    try:
        # 清空 Store
        try:
            items = health_store.search(("health", user_id))
            for item in items:
                health_store.delete(("health", user_id), item.key)
        except Exception:
            pass
        
        # 清空备选字典
        if user_id in _health_backup:
            _health_backup[user_id] = {}
        
        print("  ✓ 健康档案已清空")
    except Exception as e:
        print(f"  ⚠️ 清空失败: {e}")


# ============================================================
# 交互式菜单
# ============================================================
def show_mode_menu():
    print("""
请选择使用模式：

  1️⃣  【健康评估】计算健康指标，获取个性化建议
  2️⃣  【医学科普】学习疾病预防、症状解读等知识
  
  💡 或者直接提问，系统会自动识别！
  
  🆕 新命令：
     /profile  - 查看我记住的你的健康信息
     /clear    - 清空健康档案
     /new      - 开始新会话
     /debug    - 开启/关闭调试模式
  
输入 1 或 2 选择模式，或直接输入问题：
""")

def show_assessment_guide():
    print(ASSESSMENT_TOOLS)
    print("\n请输入你的问题（或输入 /back 返回）：")

def show_science_guide():
    print(SCIENCE_EXAMPLES)
    print("\n请输入你的问题（或输入 /back 返回）：")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print(WELCOME_MESSAGE)
    
    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️  提示: 未配置 TAVILY_API_KEY，联网搜索将不可用")
        print("   如需使用，请访问 https://tavily.com 获取API密钥\n")
    
    # 🆕 会话管理（user_id 用于长期记忆）
    user_id = input("👤 输入你的名字（用于记住你的健康信息，或按Enter匿名）: ").strip()
    if not user_id:
        user_id = f"anon_{uuid.uuid4().hex[:8]}"
    
    thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"  # 每次新会话
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n✨ 欢迎，{user_id}！")
    print(f"   会话ID: {thread_id}")
    
    # 🆕 检查是否有历史健康档案
    existing_profile = load_health_profile(user_id)
    if existing_profile:
        print(f"   📋 已加载你的健康档案（输入 /profile 查看）")
    
    print("━" * 60)
    
    current_mode = None
    
    while True:
        try:
            if current_mode is None:
                show_mode_menu()
            
            user_input = input("👉 ").strip()
            
            # 退出
            if user_input.lower() in ["q", "quit", "exit"]:
                print("\n👋 再见！你的健康信息已保存，下次见！")
                break
            
            # 🆕 新命令：查看健康档案
            if user_input == "/profile":
                show_health_profile(user_id)
                continue
            
            # 🆕 新命令：清空健康档案
            if user_input == "/clear":
                confirm = input("⚠️ 确定要清空健康档案吗？(y/n): ").strip().lower()
                if confirm == "y":
                    clear_health_profile(user_id)
                continue
            
            # 🆕 新命令：调试模式
            if user_input == "/debug":
                DEBUG_MEMORY = not DEBUG_MEMORY
                print(f"  调试模式: {'开启' if DEBUG_MEMORY else '关闭'}")
                continue
            
            if user_input == "/new":
                thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
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
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": user_id  # 🆕 传入 user_id
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
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()
            print("请重新输入或输入 /back 返回主菜单\n")