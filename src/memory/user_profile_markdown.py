"""
用户健康档案 Markdown 管理模块
负责：将用户健康记录以结构化Markdown文件形式持久化存储

设计理念：
- 每个用户一个独立的Markdown文件，便于人工查阅和管理
- 与SQLite数据库同步，保证数据一致性
- 支持Git版本控制，可追踪用户档案变化历史

目录结构：
    user_profiles/
    ├── index.md          # 用户索引（可选）
    ├── user_001.md       # 用户001的健康档案
    ├── user_002.md       # 用户002的健康档案
    └── ...

文件格式：
    ---
    user_id: xxx
    display_name: xxx
    created_at: xxx
    last_updated: xxx
    ---
    
    # 用户健康档案
    
    ## ⚠️ 重要提醒
    - 青霉素过敏
    - 糖尿病史
    
    ## 身体指标
    - 身高170cm
    - 体重65kg
    
    ## 生活习惯
    - 每天运动30分钟
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
import re

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BASE_DIR

# 尝试从配置获取，如果没有则使用默认值
try:
    from config.settings import USER_PROFILES_DIR
except ImportError:
    USER_PROFILES_DIR = os.path.join(BASE_DIR, "user_profiles")


class UserProfileMarkdown:
    """
    用户健康档案的Markdown文件管理器
    
    特点：
    - 自动创建用户目录
    - 与SQLite保持同步
    - 结构化YAML frontmatter
    - 按类别组织健康记录
    """
    
    def __init__(self, profiles_dir: str = USER_PROFILES_DIR):
        """
        初始化Markdown管理器
        
        Args:
            profiles_dir: 用户档案存储目录
        """
        self.profiles_dir = profiles_dir
        self._ensure_dir_exists()
    
    def _ensure_dir_exists(self):
        """确保用户档案目录存在"""
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
            # 创建.gitkeep以便Git追踪空目录
            gitkeep_path = os.path.join(self.profiles_dir, ".gitkeep")
            with open(gitkeep_path, 'w') as f:
                f.write("")
            # 创建README说明
            self._create_readme()
    
    def _create_readme(self):
        """创建目录说明文件"""
        readme_content = """# 用户健康档案目录

此目录存储所有用户的健康档案（Markdown格式）。

## 文件命名规则
- 每个用户对应一个 `{user_id}.md` 文件
- 文件名与用户ID一致

## 文件结构
每个Markdown文件包含：
1. **YAML Frontmatter**: 用户元信息（ID、名称、创建时间等）
2. **重要提醒**: 过敏、疾病史等关键信息
3. **分类记录**: 按类别组织的健康信息

## 注意事项
- 这些文件由系统自动生成和更新
- 手动编辑可能导致与数据库不同步
- 如需修改，建议通过应用程序操作
"""
        readme_path = os.path.join(self.profiles_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _get_profile_path(self, user_id: str) -> str:
        """
        获取用户档案文件路径
        
        Args:
            user_id: 用户ID
        
        Returns:
            Markdown文件的完整路径
        """
        # 清理user_id中的特殊字符，确保文件名安全
        safe_id = re.sub(r'[^\w\-]', '_', user_id)
        return os.path.join(self.profiles_dir, f"{safe_id}.md")
    
    def _generate_markdown(
        self,
        user_id: str,
        display_name: str,
        created_at: str,
        records: List[Dict]
    ) -> str:
        """
        生成用户健康档案的Markdown内容
        
        Args:
            user_id: 用户ID
            display_name: 显示名称
            created_at: 创建时间
            records: 健康记录列表
        
        Returns:
            格式化的Markdown字符串
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # YAML Frontmatter
        lines = [
            "---",
            f"user_id: {user_id}",
            f"display_name: {display_name}",
            f"created_at: {created_at}",
            f"last_updated: {now}",
            f"total_records: {len(records)}",
            "---",
            "",
            f"# {display_name} 的健康档案",
            "",
        ]
        
        if not records:
            lines.append("*暂无健康记录*")
            return "\n".join(lines)
        
        # 按类别和重要性整理记录
        important_items = []
        categories = {}
        
        for record in records:
            category = record.get("category", "其他")
            content = record.get("content", "")
            important = record.get("important", False)
            record_time = record.get("created_at", "")
            
            if important:
                important_items.append({
                    "content": content,
                    "category": category,
                    "time": record_time
                })
            
            if category not in categories:
                categories[category] = []
            categories[category].append({
                "content": content,
                "important": important,
                "time": record_time
            })
        
        # 重要提醒部分（优先显示）
        if important_items:
            lines.append("## ⚠️ 重要提醒")
            lines.append("")
            lines.append("> **以下信息在医疗咨询中需特别注意**")
            lines.append("")
            for item in important_items:
                lines.append(f"- **{item['content']}** ({item['category']})")
            lines.append("")
        
        # 按类别显示详细记录
        # 定义类别显示顺序
        category_order = ["身体指标", "过敏信息", "疾病史", "用药情况", "生活习惯"]
        
        # 先显示已知类别，再显示其他
        sorted_categories = []
        for cat in category_order:
            if cat in categories:
                sorted_categories.append(cat)
        for cat in categories:
            if cat not in sorted_categories:
                sorted_categories.append(cat)
        
        for category in sorted_categories:
            items = categories[category]
            emoji = self._get_category_emoji(category)
            lines.append(f"## {emoji} {category}")
            lines.append("")
            
            for item in items:
                content = item['content']
                time_str = item['time'][:10] if item['time'] else ""
                
                if item['important']:
                    lines.append(f"- **{content}** `{time_str}`")
                else:
                    lines.append(f"- {content} `{time_str}`")
            
            lines.append("")
        
        # 页脚
        lines.append("---")
        lines.append(f"*此档案由 MediQuery-RAG 系统自动生成*")
        lines.append(f"*最后更新: {now}*")
        
        return "\n".join(lines)
    
    def _get_category_emoji(self, category: str) -> str:
        """获取类别对应的emoji"""
        emoji_map = {
            "身体指标": "📊",
            "过敏信息": "🚫",
            "疾病史": "🏥",
            "用药情况": "💊",
            "生活习惯": "🏃",
        }
        return emoji_map.get(category, "📋")
    
    def save_profile(
        self,
        user_id: str,
        display_name: str,
        created_at: str,
        records: List[Dict]
    ) -> str:
        """
        保存用户健康档案到Markdown文件
        
        Args:
            user_id: 用户ID
            display_name: 显示名称
            created_at: 用户创建时间
            records: 健康记录列表（从ProfileStore获取）
        
        Returns:
            保存的文件路径
        """
        markdown_content = self._generate_markdown(
            user_id=user_id,
            display_name=display_name,
            created_at=created_at,
            records=records
        )
        
        file_path = self._get_profile_path(user_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return file_path
    
    def delete_profile(self, user_id: str) -> bool:
        """
        删除用户的Markdown档案
        
        Args:
            user_id: 用户ID
        
        Returns:
            是否成功删除
        """
        file_path = self._get_profile_path(user_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    
    def profile_exists(self, user_id: str) -> bool:
        """检查用户的Markdown档案是否存在"""
        return os.path.exists(self._get_profile_path(user_id))
    
    def get_profile_path(self, user_id: str) -> Optional[str]:
        """
        获取用户档案路径（如果存在）
        
        Args:
            user_id: 用户ID
        
        Returns:
            文件路径，不存在则返回None
        """
        path = self._get_profile_path(user_id)
        return path if os.path.exists(path) else None
    
    def list_all_profiles(self) -> List[Dict]:
        """
        列出所有用户档案
        
        Returns:
            用户档案信息列表
        """
        profiles = []
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.md') and filename != 'README.md':
                file_path = os.path.join(self.profiles_dir, filename)
                user_id = filename[:-3]  # 去掉.md后缀
                
                # 获取文件修改时间
                mtime = os.path.getmtime(file_path)
                modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                profiles.append({
                    "user_id": user_id,
                    "file_path": file_path,
                    "last_modified": modified
                })
        
        return sorted(profiles, key=lambda x: x['last_modified'], reverse=True)
    
    def generate_index(self) -> str:
        """
        生成用户索引文件
        
        Returns:
            索引文件路径
        """
        profiles = self.list_all_profiles()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            "---",
            f"generated_at: {now}",
            f"total_users: {len(profiles)}",
            "---",
            "",
            "# 用户档案索引",
            "",
            f"共 **{len(profiles)}** 个用户档案",
            "",
            "| 用户ID | 最后更新 | 档案链接 |",
            "|--------|----------|----------|",
        ]
        
        for profile in profiles:
            user_id = profile['user_id']
            modified = profile['last_modified']
            link = f"[查看](./{user_id}.md)"
            lines.append(f"| {user_id} | {modified} | {link} |")
        
        lines.append("")
        lines.append("---")
        lines.append(f"*索引生成时间: {now}*")
        
        index_path = os.path.join(self.profiles_dir, "index.md")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        return index_path


# 不再创建全局实例，避免自动生成user_profiles目录
# 如需使用，请手动实例化：UserProfileMarkdown()
# user_profile_md = UserProfileMarkdown()
