import re
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. 配置
DATA_FILE = "RAG超越百岁(原书)20250506.docx" 
DB_PATH = "./medical_db"

def parse_custom_format(file_path):
    """
    解析特定格式的医疗数据文件
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- 分割逻辑 ---
    # 使用 chunk_id 作为分割符
    chunks = re.split(r'chunk_id:', text)
    
    documents = []
    
    # 跳过第一个空的
    for chunk in chunks:
        if not chunk.strip(): continue
        
        try:
            # --- 提取 Title ---
            title_match = re.search(r'title:\s*(.*?)\n', chunk)
            title = title_match.group(1).strip() if title_match else "未命名"
            
            # --- 提取 Content (核心修复部分) ---
            content = ""
            content_match = re.search(r'content:\s*', chunk)
            
            if content_match:
                start_index = content_match.end()
                
                # 尝试找 content 后面紧跟的 "source:" 标签作为结束点
                end_index = chunk.find('source:', start_index)
                
                # 如果没找到 source，试着找 tags
                if end_index == -1:
                    end_index = chunk.find('tags:', start_index)
                
                if end_index != -1:
                    # 截取中间的内容
                    raw_content = chunk[start_index:end_index]
                    
                    # 【逻辑修复】检查内容里是否混入了标签，如果有，切掉它
                    if "source:" in raw_content or "tags:" in raw_content:
                        # 如果内容里混入了标签，在第一个标签处切断
                        raw_content = raw_content.split('source:')[0].split('tags:')[0]
                    
                    content = raw_content.strip()
                else:
                    # 如果后面没标签了，就取到最后
                    content = chunk[start_index:].strip()

            # --- 提取 Tags ---
            tags_match = re.search(r'tags:\s*(.*?)\n', chunk)
            tags = tags_match.group(1).strip() if tags_match else ""

            # --- 组装 Document ---
            if title or content:
                full_text = f"问题：{title}\n答案：{content}"
                
                doc = Document(
                    page_content=full_text,
                    metadata={
                        "title": title,
                        "tags": tags,
                        "source": "《超越百岁》"
                    }
                )
                documents.append(doc)
            
        except Exception as e:
            print(f"⚠️ 解析跳过一个块，原因: {e}")
            continue

    return documents

# 2. 执行入库
if __name__ == "__main__":
    # ⚠️ 请确认这里的文件路径是否正确
    # 如果你的 txt 文件在 data 文件夹下，请保持不变
    txt_path = "./data/medical_data.txt" 
    
    print(f"📂 准备读取文件: {txt_path}")
    
    docs = parse_custom_format(txt_path)
    
    if len(docs) > 0:
        print(f"🧹 解析成功！共清洗出 {len(docs)} 个知识块。")
        print(f"👀 预览第一条数据：\nTitle: {docs[0].metadata['title']}\nContent片段: {docs[0].page_content[:50]}...")
        
        print("\n💉 正在注入向量数据库 (Chroma)...")
        embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh")
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        print("🚀 数据库构建完成！")
    else:
        print("⚠️ 未提取到任何数据，请检查 txt_path 路径是否正确。")