"""
HITL (Human-in-the-Loop) 审核管理模块

设计理念：
- 使用Markdown文件作为审核队列，人类可直接阅读和编辑
- 三个状态目录：pending（待审核）、approved（已通过）、rejected（已拒绝）
- 支持多种审核类型：信息提取、档案修改、敏感回答

工作流程：
1. 系统生成变更请求 → 写入 pending/
2. 人工审核 → 修改状态字段或移动文件
3. 系统定期检查 → 处理已审核的请求

使用场景：
- 健康信息提取后，先进pending等待确认
- 用户请求修改档案，需要审批
- 涉及敏感医疗建议时，暂存待审核
"""

import os
import re
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Literal
from enum import Enum
from dataclasses import dataclass, asdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BASE_DIR


# ============================================================
# 配置
# ============================================================
HITL_BASE_DIR = os.path.join(BASE_DIR, "hitl_reviews")

# 审核类型
class ReviewType(str, Enum):
    EXTRACTION = "extraction"      # 健康信息提取
    PROFILE_EDIT = "profile_edit"  # 档案修改
    RESPONSE = "response"          # 敏感回答审核

# 审核状态
class ReviewStatus(str, Enum):
    PENDING = "pending"            # 待审核
    APPROVED = "approved"          # 已通过
    REJECTED = "rejected"          # 已拒绝
    AUTO_APPROVED = "auto_approved"  # 自动通过（低风险）

# 风险等级
class RiskLevel(str, Enum):
    LOW = "low"          # 低风险 - 可自动通过
    MEDIUM = "medium"    # 中风险 - 需要审核
    HIGH = "high"        # 高风险 - 必须审核


# ============================================================
# 审核请求数据结构
# ============================================================
@dataclass
class ReviewRequest:
    """审核请求"""
    request_id: str                    # 唯一ID
    review_type: ReviewType            # 审核类型
    user_id: str                       # 关联用户
    status: ReviewStatus               # 当前状态
    risk_level: RiskLevel              # 风险等级
    created_at: str                    # 创建时间
    
    # 内容
    title: str                         # 标题摘要
    content: Dict                      # 待审核的具体内容
    context: str                       # 上下文（对话摘要等）
    
    # 审核结果
    reviewed_at: Optional[str] = None  # 审核时间
    reviewer: Optional[str] = None     # 审核人
    review_note: Optional[str] = None  # 审核备注
    modified_content: Optional[Dict] = None  # 审核后修改的内容


# ============================================================
# HITL 管理器
# ============================================================
class HITLManager:
    """
    Human-in-the-Loop 审核管理器
    
    功能：
    - 创建审核请求（写入pending目录）
    - 检查审核状态（读取Markdown中的status字段）
    - 处理已审核请求（应用变更或拒绝）
    - 风险评估（决定是否需要人工审核）
    """
    
    def __init__(self, base_dir: str = HITL_BASE_DIR):
        self.base_dir = base_dir
        self.pending_dir = os.path.join(base_dir, "pending")
        self.approved_dir = os.path.join(base_dir, "approved")
        self.rejected_dir = os.path.join(base_dir, "rejected")
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """确保目录结构存在"""
        for dir_path in [self.pending_dir, self.approved_dir, self.rejected_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # 创建README
        readme_path = os.path.join(self.base_dir, "README.md")
        if not os.path.exists(readme_path):
            self._create_readme()
    
    def _create_readme(self):
        """创建审核目录说明"""
        content = """# HITL 审核队列

此目录用于管理需要人工审核的请求。

## 目录结构

```
hitl_reviews/
├── pending/      # 待审核 - 新请求放在这里
├── approved/     # 已通过 - 审核通过后移到这里
└── rejected/     # 已拒绝 - 审核拒绝后移到这里
```

## 审核流程

### 方式一：修改status字段（推荐）

1. 打开 `pending/` 中的文件
2. 修改 YAML frontmatter 中的 `status` 字段：
   - `approved` - 通过
   - `rejected` - 拒绝
3. 可选：填写 `reviewer` 和 `review_note`
4. 保存文件，系统会自动处理

### 方式二：移动文件

直接将文件从 `pending/` 移动到 `approved/` 或 `rejected/`

## 风险等级

- **low**: 低风险，系统可自动通过
- **medium**: 中风险，建议人工审核
- **high**: 高风险，必须人工审核（如过敏信息、用药建议）

## 审核类型

- **extraction**: 从对话中提取的健康信息
- **profile_edit**: 用户档案修改请求
- **response**: 涉及敏感内容的回答
"""
        with open(os.path.join(self.base_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_request_id(self, review_type: ReviewType, user_id: str) -> str:
        """生成唯一请求ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{review_type.value}_{user_id}_{timestamp}"
    
    def _get_file_path(self, request_id: str, status: ReviewStatus) -> str:
        """获取请求文件路径"""
        dir_map = {
            ReviewStatus.PENDING: self.pending_dir,
            ReviewStatus.APPROVED: self.approved_dir,
            ReviewStatus.REJECTED: self.rejected_dir,
            ReviewStatus.AUTO_APPROVED: self.approved_dir,
        }
        return os.path.join(dir_map[status], f"{request_id}.md")
    
    def _request_to_markdown(self, request: ReviewRequest) -> str:
        """将审核请求转换为Markdown格式"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            "---",
            f"request_id: {request.request_id}",
            f"review_type: {request.review_type.value}",
            f"user_id: {request.user_id}",
            f"status: {request.status.value}",
            f"risk_level: {request.risk_level.value}",
            f"created_at: {request.created_at}",
        ]
        
        if request.reviewed_at:
            lines.append(f"reviewed_at: {request.reviewed_at}")
        if request.reviewer:
            lines.append(f"reviewer: {request.reviewer}")
        if request.review_note:
            lines.append(f"review_note: \"{request.review_note}\"")
        
        lines.extend([
            "---",
            "",
            f"# {request.title}",
            "",
        ])
        
        # 风险提示
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        lines.append(f"**风险等级**: {risk_emoji.get(request.risk_level.value, '⚪')} {request.risk_level.value.upper()}")
        lines.append("")
        
        # 审核类型说明
        type_desc = {
            ReviewType.EXTRACTION: "从用户对话中提取的健康信息",
            ReviewType.PROFILE_EDIT: "用户档案修改请求",
            ReviewType.RESPONSE: "涉及敏感内容的回答",
        }
        lines.append(f"**类型**: {type_desc.get(request.review_type, request.review_type.value)}")
        lines.append("")
        
        # 上下文
        if request.context:
            lines.append("## 📝 上下文")
            lines.append("")
            lines.append(f"> {request.context}")
            lines.append("")
        
        # 待审核内容
        lines.append("## 📋 待审核内容")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(request.content, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        
        # 审核操作指南
        lines.append("## ✏️ 审核操作")
        lines.append("")
        lines.append("修改上方 `status` 字段为：")
        lines.append("- `approved` - 确认无误，同意添加")
        lines.append("- `rejected` - 信息有误，拒绝添加")
        lines.append("")
        lines.append("可选填写 `reviewer`（审核人）和 `review_note`（备注）")
        lines.append("")
        
        # 如果已审核
        if request.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED]:
            lines.append("---")
            lines.append("")
            lines.append("## ✅ 审核结果")
            lines.append("")
            lines.append(f"- **状态**: {request.status.value}")
            if request.reviewer:
                lines.append(f"- **审核人**: {request.reviewer}")
            if request.reviewed_at:
                lines.append(f"- **审核时间**: {request.reviewed_at}")
            if request.review_note:
                lines.append(f"- **备注**: {request.review_note}")
        
        return "\n".join(lines)
    
    def _parse_markdown(self, file_path: str) -> Optional[ReviewRequest]:
        """从Markdown文件解析审核请求"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析YAML frontmatter
            match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
            if not match:
                return None
            
            yaml_content = match.group(1)
            
            # 简单的YAML解析
            data = {}
            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    data[key] = value
            
            # 解析JSON内容块
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            content_data = {}
            if json_match:
                try:
                    content_data = json.loads(json_match.group(1))
                except:
                    pass
            
            # 提取上下文
            context_match = re.search(r'## 📝 上下文\n\n> (.*?)\n\n', content, re.DOTALL)
            context = context_match.group(1) if context_match else ""
            
            return ReviewRequest(
                request_id=data.get('request_id', ''),
                review_type=ReviewType(data.get('review_type', 'extraction')),
                user_id=data.get('user_id', ''),
                status=ReviewStatus(data.get('status', 'pending')),
                risk_level=RiskLevel(data.get('risk_level', 'medium')),
                created_at=data.get('created_at', ''),
                title=data.get('request_id', ''),  # 使用ID作为标题
                content=content_data,
                context=context,
                reviewed_at=data.get('reviewed_at'),
                reviewer=data.get('reviewer'),
                review_note=data.get('review_note'),
            )
        except Exception as e:
            print(f"解析审核文件失败: {e}")
            return None
    
    # ==================== 公开API ====================
    
    def assess_risk(self, review_type: ReviewType, content: Dict) -> RiskLevel:
        """
        评估风险等级
        
        规则：
        - 过敏信息、用药情况 → 高风险
        - 疾病史 → 中风险
        - 身体指标、生活习惯 → 低风险
        """
        if review_type == ReviewType.EXTRACTION:
            category = content.get('category', '')
            
            # 高风险类别
            if category in ['过敏信息', '用药情况']:
                return RiskLevel.HIGH
            
            # 中风险类别
            if category in ['疾病史']:
                return RiskLevel.MEDIUM
            
            # 内容关键词检查
            text = str(content).lower()
            high_risk_keywords = ['过敏', '禁忌', '不能吃', '不能用', '药物']
            if any(kw in text for kw in high_risk_keywords):
                return RiskLevel.HIGH
            
            return RiskLevel.LOW
        
        elif review_type == ReviewType.RESPONSE:
            # 回答审核默认中风险
            text = str(content).lower()
            if any(kw in text for kw in ['用药', '剂量', '诊断', '处方']):
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM
        
        return RiskLevel.MEDIUM
    
    def create_review(
        self,
        review_type: ReviewType,
        user_id: str,
        content: Dict,
        context: str = "",
        title: str = ""
    ) -> ReviewRequest:
        """
        创建审核请求
        
        Args:
            review_type: 审核类型
            user_id: 用户ID
            content: 待审核内容
            context: 上下文信息
            title: 标题
        
        Returns:
            创建的审核请求
        """
        request_id = self._generate_request_id(review_type, user_id)
        risk_level = self.assess_risk(review_type, content)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 低风险可自动通过
        if risk_level == RiskLevel.LOW:
            status = ReviewStatus.AUTO_APPROVED
        else:
            status = ReviewStatus.PENDING
        
        if not title:
            title = f"[{review_type.value}] {user_id} - {now[:10]}"
        
        request = ReviewRequest(
            request_id=request_id,
            review_type=review_type,
            user_id=user_id,
            status=status,
            risk_level=risk_level,
            created_at=now,
            title=title,
            content=content,
            context=context,
        )
        
        # 写入文件
        file_path = self._get_file_path(request_id, status)
        markdown = self._request_to_markdown(request)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        return request
    
    def get_pending_reviews(self, user_id: Optional[str] = None) -> List[ReviewRequest]:
        """获取待审核列表"""
        reviews = []
        
        for filename in os.listdir(self.pending_dir):
            if not filename.endswith('.md'):
                continue
            
            file_path = os.path.join(self.pending_dir, filename)
            request = self._parse_markdown(file_path)
            
            if request:
                if user_id is None or request.user_id == user_id:
                    reviews.append(request)
        
        return sorted(reviews, key=lambda x: x.created_at, reverse=True)
    
    def check_and_process_reviews(self) -> Dict[str, int]:
        """
        检查并处理已审核的请求
        
        扫描pending目录，处理status已变更的文件
        
        Returns:
            处理统计 {"approved": n, "rejected": n}
        """
        stats = {"approved": 0, "rejected": 0, "moved": 0}
        
        for filename in os.listdir(self.pending_dir):
            if not filename.endswith('.md'):
                continue
            
            file_path = os.path.join(self.pending_dir, filename)
            request = self._parse_markdown(file_path)
            
            if not request:
                continue
            
            # 检查状态是否已变更
            if request.status == ReviewStatus.APPROVED:
                # 移动到approved目录
                new_path = os.path.join(self.approved_dir, filename)
                shutil.move(file_path, new_path)
                stats["approved"] += 1
                stats["moved"] += 1
                
                # 触发回调（应用变更）
                self._on_approved(request)
                
            elif request.status == ReviewStatus.REJECTED:
                # 移动到rejected目录
                new_path = os.path.join(self.rejected_dir, filename)
                shutil.move(file_path, new_path)
                stats["rejected"] += 1
                stats["moved"] += 1
                
                # 触发回调（记录拒绝）
                self._on_rejected(request)
        
        return stats
    
    def _on_approved(self, request: ReviewRequest):
        """
        审核通过后的回调
        
        根据审核类型执行相应操作
        """
        print(f"  ✅ 审核通过: {request.request_id}")
        
        if request.review_type == ReviewType.EXTRACTION:
            # 将提取的信息写入用户档案
            try:
                from memory.profile_store import profile_store
                
                content = request.content
                if isinstance(content, dict) and 'category' in content:
                    profile_store.add_health_record(
                        user_id=request.user_id,
                        category=content['category'],
                        content=content['content'],
                        important=content.get('important', False)
                    )
                    print(f"    → 已添加到用户档案: {content['content']}")
            except Exception as e:
                print(f"    ⚠️ 添加档案失败: {e}")
    
    def _on_rejected(self, request: ReviewRequest):
        """审核拒绝后的回调"""
        print(f"  ❌ 审核拒绝: {request.request_id}")
        if request.review_note:
            print(f"    → 原因: {request.review_note}")
    
    def approve_review(self, request_id: str, reviewer: str = "", note: str = "") -> bool:
        """
        编程方式通过审核
        
        Args:
            request_id: 请求ID
            reviewer: 审核人
            note: 备注
        """
        return self._update_review_status(request_id, ReviewStatus.APPROVED, reviewer, note)
    
    def reject_review(self, request_id: str, reviewer: str = "", note: str = "") -> bool:
        """
        编程方式拒绝审核
        """
        return self._update_review_status(request_id, ReviewStatus.REJECTED, reviewer, note)
    
    def _update_review_status(
        self,
        request_id: str,
        new_status: ReviewStatus,
        reviewer: str = "",
        note: str = ""
    ) -> bool:
        """更新审核状态"""
        # 查找文件
        file_path = os.path.join(self.pending_dir, f"{request_id}.md")
        if not os.path.exists(file_path):
            return False
        
        request = self._parse_markdown(file_path)
        if not request:
            return False
        
        # 更新状态
        request.status = new_status
        request.reviewed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request.reviewer = reviewer or "system"
        request.review_note = note
        
        # 重写文件
        markdown = self._request_to_markdown(request)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        # 处理
        self.check_and_process_reviews()
        
        return True
    
    def get_review_stats(self) -> Dict:
        """获取审核统计"""
        return {
            "pending": len([f for f in os.listdir(self.pending_dir) if f.endswith('.md')]),
            "approved": len([f for f in os.listdir(self.approved_dir) if f.endswith('.md')]),
            "rejected": len([f for f in os.listdir(self.rejected_dir) if f.endswith('.md')]),
        }


# 全局实例
hitl_manager = HITLManager()
