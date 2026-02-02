"""
核心引擎：负责初始化模型、连接数据库、提供基础检索功能
"""
import sys
import os
# 导入工具列表，用于绑定给模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools import medical_tools_list

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

# --- 1. 基础配置 ---
DB_PATH = "./medical_db"

if not os.path.exists(DB_PATH):
    print(f"❌ 错误：向量库不存在 {DB_PATH}")
    print("请先运行 python3 src/ingest_medical.py")
    sys.exit(1)

# --- 2. 初始化共享资源 ---
print("⚙️ 正在初始化医学引擎 (LLM & VectorStore)...")

# ⚠️ 必须与入库时使用的模型一致
embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh")

# 初始化主模型
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# 初始化带工具的模型 (给 Tool Agent 用)
llm_with_tools = llm.bind_tools(medical_tools_list)

# 连接数据库
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# --- 3. 封装通用功能 ---

def search_knowledge_base(query: str, k: int = 3) -> str:
    """
    通用检索函数：输入问题，返回格式化后的上下文
    """
    print(f"  🔍 [引擎检索] 关键词: {query[:15]}...")
    try:
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return ""
        
        # 格式化输出
        context = "\n\n".join([
            f"【来源: {d.metadata.get('title', '未知')}】\n{d.page_content}" 
            for d in docs
        ])
        return context
    except Exception as e:
        print(f"检索出错: {e}")
        return ""