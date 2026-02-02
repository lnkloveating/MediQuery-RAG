import sqlite3

# 连接数据库
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

try:
    # 查询 checkpoints 表（LangGraph 默认表名）
    # thread_id 通常保存在 thread_id 列中
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
    rows = cursor.fetchall()
    
    print("📊 数据库中已保存的会话 ID:")
    print("-" * 30)
    for row in rows:
        print(f"🆔 {row[0]}")
    print("-" * 30)
    print(f"共发现 {len(rows)} 个历史会话")

except Exception as e:
    print("数据库可能为空或表结构不同。", e)

conn.close()