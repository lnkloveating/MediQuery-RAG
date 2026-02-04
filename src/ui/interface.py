"""
用户界面模块
负责：所有终端显示和用户交互

扩展指南：
- 修改欢迎界面：编辑 show_welcome()
- 修改命令：编辑 run_health_advisor() 或 run_science_qa()
- 添加新模式：创建新的 run_xxx() 函数
"""
import uuid
from langchain_core.messages import HumanMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import profile_store, load_health_profile

# 导入结构化问诊模块
from consultation.structured_consultation import (
    StructuredConsultation,
    RiskLevel,
    QuestionStage,
)


# ============================================================
# 全局变量
# ============================================================
_current_thread_id = ""

def set_current_thread_id(thread_id: str):
    global _current_thread_id
    _current_thread_id = thread_id


# ============================================================
# 欢迎界面
# ============================================================
def show_welcome():
    """显示主菜单"""
    print("""
╔══════════════════════════════════════════════════════════╗
║              🏥 科普医疗智能助手                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║   请选择服务模式：                                        ║
║                                                          ║
║   [1] 🩺 智能健康问诊（推荐）                             ║
║       • 系统引导式问诊，无需自己描述                       ║
║       • 自动评估症状风险等级                              ║
║       • 高危症状立即提醒就医                              ║
║                                                          ║
║   [2] 📚 医学科普问答                                    ║
║       • 无需登录，直接提问                                ║
║       • 基于医学知识库和网络搜索回答                       ║
║       • 适合了解疾病预防、健康知识等                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ============================================================
# 结构化问诊 - 打印问题
# ============================================================
def print_question(question: dict, index: int = None):
    """格式化打印问题"""
    print()
    if index:
        print(f"【问题 {index}】")
    
    print(f"🤖 {question['question']}")
    
    # 如果有选项，打印选项
    if question.get("options"):
        print()
        for i, opt in enumerate(question["options"], 1):
            print(f"   {i}. {opt}")
        print()
        print("   💡 输入数字选择，或直接输入内容")
    
    if question.get("placeholder"):
        print(f"   💡 示例：{question['placeholder']}")
    
    print()


# ============================================================
# 健康顾问模式（结构化问诊）
# ============================================================
def run_health_advisor(app, llm=None) -> str:
    """
    运行健康顾问模式 - 结构化问诊流程
    
    系统主导提问，用户只需回答
    
    Args:
        app: 编译后的 LangGraph app
        llm: 大模型实例（用于风险评估）
    
    Returns:
        "exit_program" 或 "back_to_menu"
    """
    print()
    print("=" * 58)
    print("  🩺 智能健康问诊")
    print("=" * 58)
    print()
    print("📋 本服务将通过结构化问诊收集您的健康信息")
    print("⚠️  本服务仅供参考，不能替代医生诊断")
    print()
    print("-" * 58)
    
    # 创建问诊实例，传入llm用于风险评估
    consultation = StructuredConsultation(llm=llm)
    
    # ========== 第1步：用户识别 ==========
    print()
    print("【第一步：用户识别】")
    print()
    print("请输入您的手机号（用于识别身份和保存档案）")
    print("💡 老用户输入相同手机号可恢复历史档案")
    print()
    
    while True:
        identifier = input("📱 您的手机号：").strip()
        
        if identifier.lower() in ['q', '/q']:
            return "back_to_menu"
        
        if identifier.lower() in ['qq', '/qq']:
            print("\n👋 再见！")
            return "exit_program"
        
        if not identifier:
            print("⚠️  请输入手机号")
            continue
        
        if len(identifier) < 6:
            print("⚠️  请输入有效的手机号")
            continue
        
        break
    
    # 识别用户
    user, is_new = consultation.identify_user(identifier)
    
    print()
    print("-" * 58)
    
    if is_new:
        print(f"\n👋 欢迎新用户！")
        print(f"   您的档案ID: {user.user_id[:8]}...")
        print(f"   首次问诊需要先收集基础信息")
    else:
        print(f"\n👋 欢迎回来！")
        print(f"   档案ID: {user.user_id[:8]}...")
        print(f"   上次访问: {user.last_visit}")
        
        # 显示已有信息
        if consultation.has_complete_profile():
            print(f"\n📋 您的已有档案：")
            print(f"   ├── 性别: {user.gender}")
            print(f"   ├── 年龄: {int(user.age)}岁")
            print(f"   ├── 身高: {user.height}cm | 体重: {user.weight}kg")
            if user.weight and user.height:
                bmi = round(user.weight / ((user.height/100) ** 2), 1)
                print(f"   ├── BMI: {bmi}")
            if user.allergies and user.allergies != ['无']:
                print(f"   ├── ⚠️ 过敏: {', '.join(user.allergies)}")
            if user.chronic_diseases and user.chronic_diseases != ['无']:
                print(f"   └── ⚠️ 慢性病: {', '.join(user.chronic_diseases)}")
            else:
                print(f"   └── 无已知慢性病")
            print()
            print("   ✅ 基础信息完整，直接进入症状描述")
    
    print()
    print("-" * 58)
    input("\n按回车键开始问诊（输入 q 可随时退出）...")
    
    # ========== 第2步：开始问诊 ==========
    session = consultation.start_session()
    
    stage_names = {
        QuestionStage.BASIC_INFO: "📊 基础信息采集",
        QuestionStage.MEDICAL_HISTORY: "📋 病史信息采集", 
        QuestionStage.CONSULTATION_TYPE: "🎯 咨询目的选择",
        QuestionStage.CURRENT_SYMPTOMS: "🩺 症状描述",
    }
    
    current_stage = None
    question_count = 0
    
    while True:
        # 获取当前问题
        question = consultation.get_current_question()
        
        if not question:
            # 没有更多问题，尝试切换阶段
            continue_flag, msg, risk = consultation._advance_stage()
            
            if msg:
                print(f"\n{'─'*40}")
                print(f"📌 {msg}")
                print(f"{'─'*40}")
            
            if not continue_flag:
                break
            continue
        
        # 检查是否进入新阶段
        stage = session.current_stage
        if stage != current_stage and stage in stage_names:
            current_stage = stage
            print()
            print("=" * 58)
            print(f"  {stage_names[stage]}")
            print("=" * 58)
        
        # 显示问题
        question_count += 1
        print_question(question, question_count)
        
        # 获取用户输入
        while True:
            answer = input("👤 您的回答：").strip()
            
            if answer.lower() in ['q', '/q']:
                print("\n⚠️  问诊已中断，您的信息已保存。")
                consultation.save_session()
                consultation.generate_history_markdown()  # 生成Markdown
                return "back_to_menu"
            
            if answer.lower() in ['qq', '/qq']:
                print("\n👋 再见！您的信息已保存。")
                consultation.save_session()
                consultation.generate_history_markdown()  # 生成Markdown
                return "exit_program"
            
            if not answer:
                print("⚠️  请输入您的回答")
                continue
            
            break
        
        # 处理回答
        continue_flag, msg, risk = consultation.process_answer(answer)
        
        if msg:
            print(f"\n{msg}")
        
        # 风险判断 - 高危立即退出
        if risk == RiskLevel.CRITICAL:
            print()
            print("!" * 58)
            print("  ⚠️  本次咨询已结束，请立即就医！")
            print("!" * 58)
            consultation.save_session()
            consultation.generate_history_markdown()  # 生成Markdown
            input("\n按回车键返回主菜单...")
            return "back_to_menu"
        
        if not continue_flag:
            break
    
    # ========== 第3步：评估与建议 ==========
    print()
    print("=" * 58)
    print("  📊 评估结果")
    print("=" * 58)
    
    risk_level = RiskLevel(session.risk_level) if session.risk_level else RiskLevel.LOW
    summary = consultation.get_consultation_summary()
    
    print(f"\n📋 问诊摘要：")
    print(f"   ├── 主诉: {summary['current_complaint']['chief_complaint']}")
    print(f"   ├── 持续时间: {summary['current_complaint']['duration']}")
    print(f"   ├── 严重程度: {summary['current_complaint']['severity']}/10")
    print(f"   └── 风险等级: {risk_level.value.upper()}")
    
    # 根据风险等级决定是否调用RAG
    if risk_level == RiskLevel.LOW:
        print()
        print("✅ 您的情况属于低风险，正在生成健康建议...")
        print()
        
        # 构造RAG查询
        query = _build_rag_query(summary)
        
        try:
            thread_id = f"{user.user_id}_{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}
            
            print("-" * 58)
            print("💡 健康建议：")
            print("-" * 58)
            
            for event in app.stream(
                {"messages": [HumanMessage(content=query)], "user_id": user.user_id},
                config
            ):
                if "summarizer" in event:
                    print(event["summarizer"]["final_answer"])
            
            print("-" * 58)
            
            # 保存建议
            session.advice_given = "已通过RAG生成建议"
            consultation.save_session()
            
        except Exception as e:
            print(f"⚠️  生成建议时出错: {e}")
            print("建议您咨询专业医生获取更详细的建议。")
    
    elif risk_level == RiskLevel.MEDIUM:
        print()
        print("⚠️  您的情况建议尽快就医检查")
        print()
        
        confirm = input("是否需要一些初步的健康建议作为参考？(y/n): ").strip().lower()
        
        if confirm == 'y':
            query = _build_rag_query(summary)
            
            try:
                thread_id = f"{user.user_id}_{uuid.uuid4().hex[:8]}"
                config = {"configurable": {"thread_id": thread_id}}
                
                print()
                print("-" * 58)
                print("💡 初步建议（仅供参考，请务必就医）：")
                print("-" * 58)
                
                for event in app.stream(
                    {"messages": [HumanMessage(content=query)], "user_id": user.user_id},
                    config
                ):
                    if "summarizer" in event:
                        print(event["summarizer"]["final_answer"])
                
                print("-" * 58)
                
            except Exception as e:
                print(f"⚠️  生成建议时出错: {e}")
    
    # 生成Markdown历史
    md_path = consultation.generate_history_markdown()
    
    print()
    print("=" * 58)
    print(f"📄 问诊记录已保存")
    print(f"   档案位置: user_data/{user.user_id[:8]}...")
    if md_path:
        print(f"   历史文档: history.md ✅")
    print("=" * 58)
    
    input("\n按回车键返回主菜单...")
    return "back_to_menu"


def _build_rag_query(summary: dict) -> str:
    """根据问诊摘要构建RAG查询"""
    parts = []
    
    # 用户基本情况
    profile = summary.get("user_profile", {})
    if profile.get("gender") and profile.get("age"):
        parts.append(f"患者是{int(profile['age'])}岁{profile['gender']}性")
    
    # 身体指标
    metrics = summary.get("health_metrics", {})
    if metrics:
        if metrics.get("BMI"):
            bmi = metrics["BMI"]
            parts.append(f"BMI为{bmi}")
            if bmi >= 28:
                parts.append("属于肥胖")
            elif bmi >= 24:
                parts.append("属于超重")
            elif bmi < 18.5:
                parts.append("属于偏瘦")
            else:
                parts.append("体重正常")
        
        if metrics.get("BMR"):
            parts.append(f"基础代谢率{metrics['BMR']}kcal/天")
        
        if metrics.get("IdealWeight"):
            parts.append(f"理想体重约{metrics['IdealWeight']}kg")
    
    # AI身体评估
    if summary.get("health_assessment"):
        parts.append(f"身体状况评估：{summary['health_assessment']}")
    
    # 病史
    if profile.get("chronic_diseases"):
        diseases = [d for d in profile["chronic_diseases"] if d and d != "无"]
        if diseases:
            parts.append(f"有{', '.join(diseases)}病史")
        else:
            parts.append("无慢性病史")
    
    if profile.get("allergies"):
        allergies = [a for a in profile["allergies"] if a and a != "无"]
        if allergies:
            parts.append(f"对{', '.join(allergies)}过敏")
        else:
            parts.append("无过敏史")
    
    # 构建查询
    context = "，".join(parts) if parts else "用户咨询健康问题"
    
    # 根据咨询类型生成不同的查询
    consultation_type = summary.get("consultation_type", "")
    complaint = summary.get("current_complaint", {})
    chief = complaint.get("chief_complaint", "")
    
    if consultation_type == "health_management":
        # 健康管理建议模式
        query = f"""
【用户情况】
{context}。

【咨询需求】
用户希望获得健康管理建议，请提供：

1. 根据BMI和基础代谢的体重管理建议
2. 适合该用户的饮食建议（每日热量摄入参考）
3. 运动建议（类型、频率、强度）
4. 生活习惯调整建议
5. 定期检查建议

【重要提示】
- 这是健康管理咨询，不是诊断
- 请结合用户的身体指标给出个性化建议
- 用通俗易懂的语言
"""
    else:
        # 症状咨询模式
        query = f"""
【患者情况】
{context}。

【症状描述】
主诉：{chief}
持续时间：{complaint.get('duration', '未知')}
严重程度：{complaint.get('severity', '未知')}/10分

【咨询需求】
请针对患者的症状「{chief}」提供健康建议：

1. 可能的原因分析
2. 日常调理和注意事项
3. 饮食和作息建议
4. 什么情况下需要就医

【重要提示】
- 这是健康科普咨询，不是诊断，请直接给出建议
- 不需要计算BMI等指标，患者信息已经提供
- 请用通俗易懂的语言，给出实用的建议
"""
    
    return query


# ============================================================
# 用户登录（保留兼容）
# ============================================================
def user_login() -> tuple:
    """
    用户登录/注册流程（旧版，保留兼容）
    """
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
        if profile_store.user_exists(user_input):
            user_info = profile_store.get_user_info(user_input)
            profile_store.update_last_active(user_input)
            records = profile_store.get_health_records(user_input)
            print(f"\n✅ 欢迎回来，{user_info['display_name']}！")
            if records:
                print(f"   已加载 {len(records)} 条健康记录")
            return user_input, user_info['display_name']
        else:
            print(f"\n❌ ID '{user_input}' 不存在")
            retry = input("   创建新账号？(y/n): ").strip().lower()
            if retry != 'y':
                return user_login()
    
    # 新用户注册
    display_name = input("\n📝 输入你的名字: ").strip() or "用户"
    user_id = f"{display_name}_{uuid.uuid4().hex[:8]}"
    profile_store.create_user(user_id, display_name)
    
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


# ============================================================
# 健康档案显示
# ============================================================
def show_health_profile(user_id: str):
    """显示用户健康档案"""
    profile = load_health_profile(user_id)
    user_info = profile_store.get_user_info(user_id)
    
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


# ============================================================
# 医学科普模式
# ============================================================
def run_science_qa(app) -> str:
    """
    运行医学科普问答模式
    
    Args:
        app: 编译后的 LangGraph app
    
    Returns:
        "exit_program" 或 "back_to_menu"
    """
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
    
    return "back_to_menu"
