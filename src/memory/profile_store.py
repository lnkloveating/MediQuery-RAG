"""
长期记忆模块 - 用户健康档案的持久化存储
负责：用户管理、健康记录的增删改查

存储架构：
- SQLite数据库：快速查询、事务支持
- Markdown文件：人类可读、Git友好、便于管理

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
    
    双重存储机制：
    - SQLite数据库：程序查询（快速）
    - Markdown文件：人工查阅（直观）
    
    每次数据变更自动同步到Markdown
    """
    
    def __init__(self, db_path: str = DB_PATH, enable_markdown_sync: bool = True):
        """
        初始化ProfileStore
        
        Args:
            db_path: SQLite数据库路径
            enable_markdown_sync: 是否启用Markdown同步（默认启用）
        """
        self.db_path = db_path
        self.enable_markdown_sync = enable_markdown_sync
        self._markdown_manager = None
        self._init_db()
    
    @property
    def markdown_manager(self):
        """延迟加载Markdown管理器，避免循环导入"""
        if self._markdown_manager is None and self.enable_markdown_sync:
            from memory.user_profile_markdown import UserProfileMarkdown
            self._markdown_manager = UserProfileMarkdown()
        return self._markdown_manager
    
    def _sync_to_markdown(self, user_id: str):
        """
        将用户数据同步到Markdown文件
        
        Args:
            user_id: 用户ID
        """
        if not self.enable_markdown_sync or not self.markdown_manager:
            return
        
        try:
            user_info = self.get_user_info(user_id)
            if not user_info:
                return
            
            records = self.get_health_records(user_id)
            
            self.markdown_manager.save_profile(
                user_id=user_id,
                display_name=user_info.get("display_name", user_id),
                created_at=user_info.get("created_at", ""),
                records=records
            )
        except Exception as e:
            # Markdown同步失败不影响主流程
            print(f"  ⚠️ Markdown同步失败: {e}")
    
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
        """创建新用户，同时创建Markdown档案"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
                (user_id, display_name)
            )
            conn.commit()
            conn.close()
            
            # 同步到Markdown
            self._sync_to_markdown(user_id)
            
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
        添加健康记录，同时更新Markdown档案
        
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
        
        # 同步到Markdown
        self._sync_to_markdown(user_id)
        
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
        """清空用户所有健康记录，同时更新Markdown档案"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_records WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        
        # 同步到Markdown（清空后的状态）
        self._sync_to_markdown(user_id)
    
    def delete_record(self, user_id: str, category: str, content: str) -> bool:
        """删除指定的健康记录，同时更新Markdown档案"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM health_records WHERE user_id = ? AND category = ? AND content = ?",
            (user_id, category, content)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        # 同步到Markdown
        if deleted:
            self._sync_to_markdown(user_id)
        
        return deleted
    
    def delete_user(self, user_id: str) -> bool:
        """
        删除用户及其所有记录，同时删除Markdown档案
        
        Args:
            user_id: 用户ID
        
        Returns:
            是否成功删除
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 删除用户的所有健康记录
        cursor.execute("DELETE FROM health_records WHERE user_id = ?", (user_id,))
        # 删除用户
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        # 删除Markdown档案
        if deleted and self.markdown_manager:
            self.markdown_manager.delete_profile(user_id)
        
        return deleted
    
    def sync_all_to_markdown(self):
        """
        将所有用户数据同步到Markdown
        用于初始化或修复数据不一致
        """
        if not self.enable_markdown_sync or not self.markdown_manager:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users")
        users = cursor.fetchall()
        conn.close()
        
        for (user_id,) in users:
            self._sync_to_markdown(user_id)
        
        # 生成索引
        self.markdown_manager.generate_index()
        
        print(f"  ✅ 已同步 {len(users)} 个用户档案到Markdown")


# 全局实例（禁用Markdown同步，现在使用user_data/目录的JSON存储）
profile_store = ProfileStore(enable_markdown_sync=False)