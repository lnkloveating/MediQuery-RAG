"""
医疗建议模式 - 结构化问诊交互界面

流程：
1. 用户识别（手机号 → UUID）
2. 系统主导的问诊流程
3. 风险评估与分流
4. 低风险问题给出RAG建议
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consultation.structured_consultation import (
    StructuredConsultation,
    RiskLevel,
    QuestionStage,
)


def print_header():
    """打印界面头部"""
    print("\n" + "=" * 50)
    print("🏥 智能健康咨询系统 - 医疗建议模式")
    print("=" * 50)
    print()
    print("📋 本服务将通过结构化问诊收集您的健康信息")
    print("⚠️  本服务仅供参考，不能替代医生诊断")
    print()
    print("-" * 50)


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
    
    if question.get("placeholder"):
        print(f"   💡 示例：{question['placeholder']}")
    
    print()


def run_medical_advisor(app=None):
    """
    运行医疗建议模式
    
    Args:
        app: LangGraph应用实例（用于RAG查询）
    
    Returns:
        "exit_program" 表示退出整个程序
    """
    print_header()
    
    # 创建问诊实例
    consultation = StructuredConsultation()
    
    # ========== 第1步：用户识别 ==========
    print("【第一步：用户识别】")
    print()
    
    while True:
        identifier = input("请输入您的手机号或ID（输入q退出）：").strip()
        
        if identifier.lower() == 'q':
            return "exit_program"
        
        if not identifier:
            print("⚠️  请输入有效的手机号或ID")
            continue
        
        if len(identifier) < 4:
            print("⚠️  输入太短，请输入有效的手机号或ID")
            continue
        
        break
    
    # 识别用户
    user, is_new = consultation.identify_user(identifier)
    
    if is_new:
        print(f"\n👋 欢迎新用户！您的档案已创建。")
        print(f"   用户ID: {user.user_id[:8]}...")
    else:
        print(f"\n👋 欢迎回来！")
        print(f"   用户ID: {user.user_id[:8]}...")
        print(f"   上次访问: {user.last_visit}")
        
        # 显示已有信息
        if consultation.has_complete_profile():
            print(f"\n📋 您的基础信息：")
            print(f"   性别: {user.gender}")
            print(f"   年龄: {user.age}岁")
            print(f"   身高: {user.height}cm")
            print(f"   体重: {user.weight}kg")
            if user.weight and user.height:
                bmi = round(user.weight / ((user.height/100) ** 2), 1)
                print(f"   BMI: {bmi}")
    
    print("\n" + "-" * 50)
    
    # ========== 第2步：开始问诊 ==========
    session = consultation.start_session()
    
    stage_names = {
        QuestionStage.BASIC_INFO: "基础信息采集",
        QuestionStage.MEDICAL_HISTORY: "病史信息采集",
        QuestionStage.CURRENT_SYMPTOMS: "当前症状描述",
    }
    
    current_stage = None
    question_count = 0
    
    while True:
        # 获取当前问题
        question = consultation.get_current_question()
        
        if not question:
            # 没有更多问题，可能需要切换阶段
            continue_flag, msg, risk = consultation._advance_stage()
            
            if msg:
                print(f"\n📌 {msg}")
            
            if not continue_flag:
                # 问诊结束
                break
            
            continue
        
        # 检查是否进入新阶段
        stage = session.current_stage
        if stage != current_stage and stage in stage_names:
            current_stage = stage
            print(f"\n{'='*50}")
            print(f"📋 【{stage_names[stage]}】")
            print("=" * 50)
        
        # 显示问题
        question_count += 1
        print_question(question, question_count)
        
        # 获取用户输入
        while True:
            answer = input("您的回答：").strip()
            
            if answer.lower() == 'q':
                print("\n⚠️  问诊已中断，您的信息已保存。")
                consultation.save_session()
                return None
            
            if not answer:
                print("⚠️  请输入您的回答")
                continue
            
            break
        
        # 处理回答
        continue_flag, msg, risk = consultation.process_answer(answer)
        
        if msg:
            print(f"\n{msg}")
        
        # 风险判断
        if risk == RiskLevel.CRITICAL:
            # 危急情况，直接退出
            print("\n" + "!" * 50)
            print("本次咨询已结束，请立即就医。")
            print("!" * 50)
            consultation.save_session()
            return None
        
        if not continue_flag:
            break
    
    # ========== 第3步：评估与建议 ==========
    print("\n" + "=" * 50)
    print("📊 【评估结果】")
    print("=" * 50)
    
    risk_level = RiskLevel(session.risk_level) if session.risk_level else RiskLevel.LOW
    
    if risk_level == RiskLevel.LOW:
        print("\n✅ 您的情况属于低风险，可以提供健康建议。")
        
        # 获取问诊摘要用于RAG
        summary = consultation.get_consultation_summary()
        
        print("\n📋 问诊摘要：")
        print(f"   主诉: {summary['current_complaint']['chief_complaint']}")
        print(f"   持续时间: {summary['current_complaint']['duration']}")
        
        # 调用RAG生成建议（如果有app）
        if app:
            print("\n🔍 正在根据您的情况生成建议...\n")
            
            # 构造查询
            query = _build_rag_query(summary)
            
            try:
                # 调用RAG（需要根据实际的graph结构调整）
                result = app.invoke({
                    "messages": [{"role": "user", "content": query}],
                    "user_id": user.user_id,
                })
                
                # 提取回答
                if "messages" in result:
                    for msg in reversed(result["messages"]):
                        if hasattr(msg, 'content'):
                            print("💡 健康建议：")
                            print("-" * 40)
                            print(msg.content)
                            print("-" * 40)
                            
                            # 保存建议到session
                            session.advice_given = msg.content
                            consultation.save_session()
                            break
            except Exception as e:
                print(f"⚠️  生成建议时出错: {e}")
                print("建议您咨询专业医生获取更详细的建议。")
        else:
            print("\n💡 请根据您的情况，咨询专业医生或查阅相关健康资料。")
    
    elif risk_level == RiskLevel.MEDIUM:
        print("\n⚠️  您的情况建议尽快就医检查。")
        
        # 仍然可以提供一些初步建议
        confirm = input("\n是否需要一些初步的健康建议？(y/n): ").strip().lower()
        if confirm == 'y' and app:
            summary = consultation.get_consultation_summary()
            query = _build_rag_query(summary)
            
            try:
                result = app.invoke({
                    "messages": [{"role": "user", "content": query}],
                    "user_id": user.user_id,
                })
                
                if "messages" in result:
                    for msg in reversed(result["messages"]):
                        if hasattr(msg, 'content'):
                            print("\n💡 初步建议（仅供参考，请务必就医）：")
                            print("-" * 40)
                            print(msg.content)
                            print("-" * 40)
                            break
            except Exception as e:
                print(f"⚠️  生成建议时出错: {e}")
    
    # 生成历史档案
    consultation.generate_history_markdown()
    
    print("\n" + "=" * 50)
    print("📄 您的问诊记录已保存")
    print("=" * 50)
    
    return None


def _build_rag_query(summary: dict) -> str:
    """根据问诊摘要构建RAG查询"""
    parts = []
    
    # 用户基本情况
    profile = summary.get("user_profile", {})
    if profile.get("gender") and profile.get("age"):
        parts.append(f"患者是{profile['age']}岁{profile['gender']}性")
    
    if profile.get("bmi"):
        bmi = profile["bmi"]
        if bmi >= 28:
            parts.append("体重偏胖")
        elif bmi < 18.5:
            parts.append("体重偏瘦")
    
    # 病史
    if profile.get("chronic_diseases"):
        parts.append(f"有{', '.join(profile['chronic_diseases'])}病史")
    
    if profile.get("allergies"):
        parts.append(f"对{', '.join(profile['allergies'])}过敏")
    
    # 主诉
    complaint = summary.get("current_complaint", {})
    if complaint.get("chief_complaint"):
        parts.append(f"目前{complaint['chief_complaint']}")
    
    if complaint.get("duration"):
        parts.append(f"持续{complaint['duration']}")
    
    # 构建查询
    context = "，".join(parts) if parts else "用户"
    
    query = f"""
{context}。

请根据以上情况，提供健康建议：
1. 可能的原因分析
2. 日常注意事项
3. 饮食建议
4. 是否需要进一步检查

注意：这是健康科普建议，不是医疗诊断。
"""
    return query


if __name__ == "__main__":
    # 独立运行测试
    run_medical_advisor()
