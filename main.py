"""
科普医疗智能助手 - 主程序入口

运行方式：
    python main.py

项目结构：
    config/         - 配置文件
    src/
        memory/     - 记忆模块（长期档案、健康提取、对话摘要）
        agents/     - Agent模块（节点定义、工作流）
        ui/         - 用户界面
        core/       - 核心工具函数
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# 导入模块
from src.ui import show_welcome, run_health_advisor, run_science_qa
from src.agents import create_nodes, build_graph
from src.tools import medical_tools_list

# 导入医学引擎
try:
    from src.medical_engine import llm, llm_with_tools, vectorstore, web_search_tool
except ImportError:
    print("❌ 错误: 无法导入医学引擎，请检查 medical_engine.py")
    sys.exit(1)


def main():
    """主函数"""
    # 检查 API Key
    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️ 提示: 未配置 TAVILY_API_KEY，联网搜索将不可用\n")
    
    # 创建节点
    nodes = create_nodes(
        llm=llm,
        llm_with_tools=llm_with_tools,
        vectorstore=vectorstore,
        web_search_tool=web_search_tool,
        medical_tools_list=medical_tools_list
    )
    
    # 构建工作流
    app = build_graph(nodes)
    
    # 主循环
    while True:
        show_welcome()
        choice = input("请选择 [1/2] (q退出): ").strip()
        
        if choice == "1":
            result = run_health_advisor(app)
            if result == "exit_program":
                break
            print()
        elif choice == "2":
            result = run_science_qa(app)
            if result == "exit_program":
                break
            print()
        elif choice.lower() in ["q", "quit", "exit"]:
            print("\n👋 再见！")
            break
        else:
            print("\n⚠️ 请输入 1 或 2\n")


if __name__ == "__main__":
    main()
