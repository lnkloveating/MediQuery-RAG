"""
长期记忆模块 - 用户健康档案的持久化存储
负责：用户管理、健康记录的增删改查

扩展指南：
- 添加新的存储字段：修改 _init_db() 中的表结构
- 添加新的查询方法：在类中添加新方法
"""
import sqlite3
from typing import List, Optional
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


class ProfileStore:
    """
    用户健康档案的持久化存储
    使用 SQLite 数据库，关闭程序后数据不会丢失
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """
        初始化数据库表
        
        扩展提示：如需添加新字段，在此修改 CREATE TABLE 语句
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 健康记录表
        # 扩展提示：可添加字段如 source(来源), confidence(置信度) 等
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                important INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ==================== 用户管理 ====================
    
    def user_exists(self, user_id: str) -> bool:
        """检查用户是否存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def create_user(self, user_id: str, display_name: str) -> bool:
        """创建新用户"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
                (user_id, display_name)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_user_info(self, user_id: str) -> Optional[dict]:
        """获取用户信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, display_name, created_at, last_active FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "user_id": row[0],
                "display_name": row[1],
                "created_at": row[2],
                "last_active": row[3]
            }
        return None
    
    def update_last_active(self, user_id: str):
        """更新用户最后活跃时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
            (user_id,)
        )
        conn.commit()
        conn.close()
    
    # ==================== 健康记录管理 ====================
    
    def add_health_record(self, user_id: str, category: str, content: str, important: bool = False) -> bool:
        """
        添加健康记录
        
        Args:
            user_id: 用户ID
            category: 类别（如"身体指标"、"过敏信息"）
            content: 内容（如"身高170cm"）
            important: 是否重要（过敏、疾病史等标记为True）
        
        Returns:
            bool: 是否成功添加（重复内容返回False）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查是否已存在相同内容
        cursor.execute(
            "SELECT 1 FROM health_records WHERE user_id = ? AND category = ? AND content = ?",
            (user_id, category, content)
        )
        if cursor.fetchone():
            conn.close()
            return False  # 已存在，不重复添加
        
        cursor.execute(
            "INSERT INTO health_records (user_id, category, content, important) VALUES (?, ?, ?, ?)",
            (user_id, category, content, 1 if important else 0)
        )
        conn.commit()
        conn.close()
        return True
    
    def get_health_records(self, user_id: str) -> List[dict]:
        """
        获取用户所有健康记录
        
        Returns:
            按重要性和时间排序的记录列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT category, content, important, created_at 
               FROM health_records 
               WHERE user_id = ? 
               ORDER BY important DESC, created_at DESC""",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "category": row[0],
                "content": row[1],
                "important": bool(row[2]),
                "created_at": row[3]
            }
            for row in rows
        ]
    
    def get_records_by_category(self, user_id: str, category: str) -> List[dict]:
        """获取指定类别的记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content, important, created_at FROM health_records WHERE user_id = ? AND category = ?",
            (user_id, category)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"content": row[0], "important": bool(row[1]), "created_at": row[2]}
            for row in rows
        ]
    
    def clear_health_records(self, user_id: str):
        """清空用户所有健康记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_records WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
    
    def delete_record(self, user_id: str, category: str, content: str) -> bool:
        """删除指定的健康记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM health_records WHERE user_id = ? AND category = ? AND content = ?",
            (user_id, category, content)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


# 全局实例
profile_store = ProfileStore()