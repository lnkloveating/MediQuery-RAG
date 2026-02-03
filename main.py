"""
科普医疗智能助手 - 主程序入口
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
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
    

    


if __name__ == "__main__":
    main()
