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


# ============================================================
# 用户登录
# ============================================================
def user_login() -> tuple:
    """
    用户登录/注册流程
    
    Returns:
        (user_id, display_name)
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
# 健康顾问模式
# ============================================================
def run_health_advisor(app) -> str:
    """
    运行健康顾问模式
    
    Args:
        app: 编译后的 LangGraph app
    
    Returns:
        "exit_program" 或 "back_to_menu"
    """
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
            
            # 命令处理
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
                    profile_store.clear_health_records(user_id)
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
    
    return "back_to_menu"


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
