"""
核心引擎：负责初始化模型、连接数据库、提供基础检索功能
"""
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# 导入工具列表，用于绑定给模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools import medical_tools_list

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_tavily import TavilySearch as TavilySearchResults

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

web_search_tool = TavilySearchResults(k=3)

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
    

    # --- 4. Self-RAG 专用组件 ---

def grade_document_relevance(question: str, doc_content: str) -> str:
    """
    阅卷老师：判断文档是否与问题相关
    返回: 'yes' 或 'no'
    """
    prompt = f"""
    你是一名评分员，负责评估检索到的文档是否与用户问题相关。
    
    文档内容：
    {doc_content}
    
    用户问题：
    {question}
    
    如果文档包含与问题相关的关键词或语义，请回答 'yes'，否则回答 'no'。
    只输出 'yes' 或 'no'，不要有其他废话。
    """
    # 使用 invoke 调用模型
    score = llm.invoke(prompt).content.strip().lower()
    
    # 简单的清洗逻辑
    if "yes" in score: return "yes"
    return "no"

def rewrite_query(question: str) -> str:
    """
    改题专家：优化搜索关键词
    """
    print(f"  🔄 [Self-RAG] 正在重写查询: {question}")
    prompt = f"""
    你是一个搜索引擎优化专家。原问题检索不到相关信息。
    请根据原问题，推断其背后的语义意图，并构造一个更好的搜索查询词。
    
    原问题：{question}
    
    只输出新的查询词，不要有任何解释。
    """
    return llm.invoke(prompt).content.strip()