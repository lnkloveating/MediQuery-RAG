"""
结构化问诊模块 - 医疗建议模式的核心流程

设计理念：
- 系统主导提问，不让用户自由输入
- 每一步提取关键信息存入JSON
- 风险评估：极端情况硬规则 + 其他情况大模型判断

问诊流程：
1. 用户识别（手机号/ID → UUID）
2. 基础信息采集（性别、年龄、身高体重）
3. 病史采集（家族病史、过敏史、用药史）
4. 症状采集（主诉、持续时间、部位、程度）
5. 风险评估 → 决定后续流程
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BASE_DIR


# ============================================================
# 配置
# ============================================================
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")

# 🔴 极端情况关键词 - 硬规则，直接退出（不经过大模型）
# 只保留自杀自残等必须立即干预的情况
EMERGENCY_KEYWORDS = [
    "想自杀", "不想活", "要自杀", "自杀", "自残", "自伤",
    "想死", "活不下去", "结束生命",
]

# 中等风险关键词（用于最终评估参考）
MEDIUM_RISK_KEYWORDS = [
    "持续疼痛", "反复发作", "越来越严重",
    "发烧", "高血压", "低血压", "心律不齐",
    "头晕", "眩晕", "恶心想吐",
    "皮疹", "过敏", "肿胀",
    "失眠严重", "焦虑", "抑郁",
]

# 低风险问题 - 可以直接给建议
LOW_RISK_TOPICS = [
    "减肥", "肥胖", "体重管理",
    "养生", "保健", "营养",
    "轻微感冒", "流鼻涕", "打喷嚏",
    "偶尔失眠", "睡眠质量",
    "久坐", "缺乏运动",
    "饮食习惯", "健康饮食",
]


# ============================================================
# 大模型风险评估 Prompt
# ============================================================
RISK_ASSESSMENT_PROMPT = """你是一名经验丰富的急诊分诊护士，需要根据患者描述判断紧急程度。

【患者信息】
- 年龄：{age}岁
- 性别：{gender}
- 慢性病史：{chronic_diseases}
- 过敏史：{allergies}
- 症状描述：{symptoms}

【判断标准】
- CRITICAL（危急）：需要立即拨打120或去急诊
  例如：胸痛+冒冷汗+呼吸困难、疑似心梗/中风、大量出血、严重过敏反应
  
- HIGH（紧急）：需要尽快就医（24小时内）
  例如：持续剧烈疼痛、症状快速加重、高烧不退
  
- MEDIUM（中等）：建议近期就医检查
  例如：反复发作的症状、影响日常生活、持续一周以上
  
- LOW（低风险）：可以继续咨询给建议
  例如：偶发轻微症状、有明确诱因（如天气热导致的轻微不适）、生活方式问题

【重要】请综合考虑：
1. 症状的严重程度描述（"有点"vs"非常"vs"剧烈"）
2. 是否有合理的诱因解释
3. 患者的年龄和基础病史
4. 症状的持续时间和变化趋势

请直接输出JSON格式（不要任何其他内容）：
{{"risk_level": "CRITICAL/HIGH/MEDIUM/LOW", "reason": "简短判断理由", "advice": "给患者的建议"}}"""


class RiskLevel(str, Enum):
    """风险等级"""
    CRITICAL = "critical"    # 危急 - 立即就医
    HIGH = "high"            # 高风险 - 强烈建议就医
    MEDIUM = "medium"        # 中等 - 建议就医+给建议
    LOW = "low"              # 低风险 - 直接给建议


class QuestionStage(str, Enum):
    """问诊阶段"""
    IDENTIFICATION = "identification"  # 用户识别
    BASIC_INFO = "basic_info"          # 基础信息
    MEDICAL_HISTORY = "medical_history" # 病史
    CURRENT_SYMPTOMS = "current_symptoms"  # 当前症状
    ASSESSMENT = "assessment"          # 评估
    ADVICE = "advice"                  # 建议
    COMPLETED = "completed"            # 完成


@dataclass
class UserProfile:
    """用户基础档案"""
    user_id: str
    phone_hash: str = ""           # 手机号哈希（隐私保护）
    created_at: str = ""
    last_visit: str = ""
    
    # 基础信息
    gender: str = ""               # 性别
    age: int = 0                   # 年龄
    height: float = 0.0            # 身高(cm)
    weight: float = 0.0            # 体重(kg)
    
    # 病史
    family_history: List[str] = field(default_factory=list)   # 家族病史
    allergies: List[str] = field(default_factory=list)        # 过敏史
    chronic_diseases: List[str] = field(default_factory=list) # 慢性病
    current_medications: List[str] = field(default_factory=list)  # 正在用药


@dataclass 
class ConsultationSession:
    """单次问诊会话"""
    session_id: str
    user_id: str
    start_time: str
    end_time: str = ""
    
    # 问诊阶段
    current_stage: QuestionStage = QuestionStage.IDENTIFICATION
    
    # 症状信息
    chief_complaint: str = ""              # 主诉
    symptom_location: str = ""             # 症状部位
    symptom_duration: str = ""             # 持续时间
    symptom_severity: str = ""             # 严重程度 (1-10)
    symptom_description: str = ""          # 详细描述
    
    # 评估结果
    risk_level: str = ""
    risk_keywords_found: List[str] = field(default_factory=list)
    llm_risk_reason: str = ""              # 大模型判断理由
    
    # 建议
    advice_given: str = ""
    referral_suggested: bool = False
    
    # 对话记录
    conversation: List[Dict] = field(default_factory=list)


# ============================================================
# 问诊问题定义
# ============================================================
QUESTIONS = {
    QuestionStage.BASIC_INFO: [
        {
            "field": "gender",
            "question": "请问您的性别是？",
            "options": ["男", "女"],
            "type": "choice"
        },
        {
            "field": "age",
            "question": "请问您的年龄是多少岁？",
            "type": "number",
            "validation": {"min": 0, "max": 120}
        },
        {
            "field": "height",
            "question": "请问您的身高是多少厘米(cm)？",
            "type": "number",
            "validation": {"min": 50, "max": 250}
        },
        {
            "field": "weight",
            "question": "请问您的体重是多少公斤(kg)？",
            "type": "number",
            "validation": {"min": 20, "max": 300}
        },
    ],
    QuestionStage.MEDICAL_HISTORY: [
        {
            "field": "family_history",
            "question": "请问您的直系亲属（父母、兄弟姐妹）有以下疾病吗？可多选，没有请输入'无'",
            "options": ["高血压", "糖尿病", "心脏病", "癌症", "脑卒中", "其他", "无"],
            "type": "multi_choice"
        },
        {
            "field": "allergies",
            "question": "请问您有药物或食物过敏吗？有请说明，没有请输入'无'",
            "type": "text",
            "placeholder": "例如：青霉素过敏、海鲜过敏"
        },
        {
            "field": "chronic_diseases",
            "question": "请问您有以下慢性病吗？可多选，没有请输入'无'",
            "options": ["高血压", "糖尿病", "高血脂", "心脏病", "哮喘", "其他", "无"],
            "type": "multi_choice"
        },
        {
            "field": "current_medications",
            "question": "请问您目前正在服用什么药物？没有请输入'无'",
            "type": "text",
            "placeholder": "例如：降压药、降糖药"
        },
    ],
    QuestionStage.CURRENT_SYMPTOMS: [
        {
            "field": "chief_complaint",
            "question": "请简单描述一下您今天咨询的主要问题是什么？",
            "type": "text",
            "important": True  # 这是风险评估的关键字段
        },
        {
            "field": "symptom_duration",
            "question": "这个症状/问题持续多长时间了？",
            "options": ["今天刚开始", "1-3天", "一周左右", "一个月以上", "很长时间了"],
            "type": "choice"
        },
        {
            "field": "symptom_severity",
            "question": "如果用1-10分表示严重程度（1最轻，10最重），您给自己的症状打几分？",
            "type": "number",
            "validation": {"min": 1, "max": 10}
        },
    ],
}


# ============================================================
# 核心类
# ============================================================
class StructuredConsultation:
    """
    结构化问诊管理器
    
    核心功能：
    - 用户识别与档案管理
    - 系统主导的问诊流程
    - 实时风险评估（大模型判断）
    - JSON档案存储
    """
    
    def __init__(self, data_dir: str = USER_DATA_DIR, llm=None):
        self.data_dir = data_dir
        self.llm = llm  # 大模型实例
        self._ensure_dirs()
        
        self.current_user: Optional[UserProfile] = None
        self.current_session: Optional[ConsultationSession] = None
        self.current_question_index: int = 0
    
    def set_llm(self, llm):
        """设置大模型实例"""
        self.llm = llm
    
    def _ensure_dirs(self):
        """确保目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _get_user_dir(self, user_id: str) -> str:
        """获取用户目录"""
        user_dir = os.path.join(self.data_dir, user_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            os.makedirs(os.path.join(user_dir, "sessions"))
        return user_dir
    
    def _generate_user_id(self, identifier: str) -> str:
        """从用户标识生成UUID"""
        hash_obj = hashlib.md5(identifier.encode())
        return str(uuid.UUID(hash_obj.hexdigest()))
    
    # ==================== 用户管理 ====================
    
    def identify_user(self, identifier: str) -> Tuple[UserProfile, bool]:
        """用户识别"""
        user_id = self._generate_user_id(identifier)
        user_dir = self._get_user_dir(user_id)
        profile_path = os.path.join(user_dir, "profile.json")
        
        is_new_user = not os.path.exists(profile_path)
        
        if is_new_user:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            profile = UserProfile(
                user_id=user_id,
                phone_hash=hashlib.sha256(identifier.encode()).hexdigest()[:16],
                created_at=now,
                last_visit=now
            )
            self._save_profile(profile)
        else:
            profile = self._load_profile(user_id)
            profile.last_visit = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_profile(profile)
        
        self.current_user = profile
        return profile, is_new_user
    
    def _save_profile(self, profile: UserProfile):
        """保存用户档案"""
        user_dir = self._get_user_dir(profile.user_id)
        profile_path = os.path.join(user_dir, "profile.json")
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
    
    def _load_profile(self, user_id: str) -> UserProfile:
        """加载用户档案"""
        user_dir = self._get_user_dir(user_id)
        profile_path = os.path.join(user_dir, "profile.json")
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return UserProfile(**data)
    
    def has_complete_profile(self) -> bool:
        """检查用户是否已有完整的基础档案"""
        if not self.current_user:
            return False
        
        return all([
            self.current_user.gender,
            self.current_user.age > 0,
            self.current_user.height > 0,
            self.current_user.weight > 0,
        ])
    
    # ==================== 问诊会话管理 ====================
    
    def start_session(self) -> ConsultationSession:
        """开始新的问诊会话"""
        if not self.current_user:
            raise ValueError("请先识别用户")
        
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        
        if self.has_complete_profile():
            start_stage = QuestionStage.CURRENT_SYMPTOMS
        else:
            start_stage = QuestionStage.BASIC_INFO
        
        session = ConsultationSession(
            session_id=session_id,
            user_id=self.current_user.user_id,
            start_time=now.strftime("%Y-%m-%d %H:%M:%S"),
            current_stage=start_stage
        )
        
        self.current_session = session
        self.current_question_index = 0
        
        return session
    
    def save_session(self):
        """保存当前会话"""
        if not self.current_session or not self.current_user:
            return
        
        user_dir = self._get_user_dir(self.current_user.user_id)
        session_path = os.path.join(
            user_dir, "sessions", 
            f"{self.current_session.session_id}.json"
        )
        
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.current_session), f, ensure_ascii=False, indent=2)
    
    # ==================== 问诊流程 ====================
    
    def get_current_question(self) -> Optional[Dict]:
        """获取当前问题"""
        if not self.current_session:
            return None
        
        stage = self.current_session.current_stage
        
        if stage not in QUESTIONS:
            return None
        
        questions = QUESTIONS[stage]
        
        if self.current_question_index >= len(questions):
            return None
        
        return questions[self.current_question_index]
    
    def process_answer(self, answer: str) -> Tuple[bool, Optional[str], Optional[RiskLevel]]:
        """处理用户回答"""
        if not self.current_session or not self.current_user:
            return False, "会话未初始化", None
        
        question = self.get_current_question()
        if not question:
            return False, "没有更多问题", None
        
        # 记录对话
        self.current_session.conversation.append({
            "role": "assistant",
            "content": question["question"]
        })
        self.current_session.conversation.append({
            "role": "user", 
            "content": answer
        })
        
        # 验证并存储答案
        field_name = question["field"]
        validated_answer = self._validate_answer(question, answer)
        
        if validated_answer is None:
            return True, f"输入无效，请重新回答：{question['question']}", None
        
        # 存储到相应位置
        self._store_answer(field_name, validated_answer)
        
        # 实时风险检测（针对症状描述）
        if question.get("important"):
            risk_level, risk_msg = self._assess_risk_realtime(answer)
            if risk_level == RiskLevel.CRITICAL:
                self.current_session.risk_level = risk_level.value
                self.current_session.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.save_session()
                return False, risk_msg, risk_level
        
        # 移动到下一个问题
        self.current_question_index += 1
        
        # 检查是否完成当前阶段
        stage = self.current_session.current_stage
        if self.current_question_index >= len(QUESTIONS.get(stage, [])):
            return self._advance_stage()
        
        return True, None, None
    
    def _validate_answer(self, question: Dict, answer: str) -> Optional[any]:
        """验证用户回答"""
        q_type = question.get("type", "text")
        
        if q_type == "choice":
            options = question.get("options", [])
            if answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            if answer in options:
                return answer
            return None
        
        elif q_type == "multi_choice":
            if answer == "无" or answer == "没有":
                return []
            selected = [a.strip() for a in answer.replace("，", ",").split(",")]
            options = question.get("options", [])
            valid = []
            for s in selected:
                if s.isdigit():
                    idx = int(s) - 1
                    if 0 <= idx < len(options):
                        valid.append(options[idx])
                elif s in options or s == "其他":
                    valid.append(s)
            return valid if valid else selected
        
        elif q_type == "number":
            try:
                num = float(answer)
                validation = question.get("validation", {})
                if validation:
                    if num < validation.get("min", float("-inf")):
                        return None
                    if num > validation.get("max", float("inf")):
                        return None
                return num
            except ValueError:
                return None
        
        else:  # text
            return answer.strip() if answer.strip() else None
    
    def _store_answer(self, field_name: str, value: any):
        """存储回答到相应位置"""
        stage = self.current_session.current_stage
        
        if stage == QuestionStage.BASIC_INFO:
            setattr(self.current_user, field_name, value)
            self._save_profile(self.current_user)
        
        elif stage == QuestionStage.MEDICAL_HISTORY:
            if isinstance(value, list):
                setattr(self.current_user, field_name, value)
            else:
                if value and value != "无":
                    setattr(self.current_user, field_name, [value])
                else:
                    setattr(self.current_user, field_name, [])
            self._save_profile(self.current_user)
        
        elif stage == QuestionStage.CURRENT_SYMPTOMS:
            setattr(self.current_session, field_name, value)
    
    def _advance_stage(self) -> Tuple[bool, Optional[str], Optional[RiskLevel]]:
        """进入下一阶段"""
        stage = self.current_session.current_stage
        self.current_question_index = 0
        
        if stage == QuestionStage.BASIC_INFO:
            self.current_session.current_stage = QuestionStage.MEDICAL_HISTORY
            return True, "基础信息已记录，接下来了解您的病史", None
        
        elif stage == QuestionStage.MEDICAL_HISTORY:
            self.current_session.current_stage = QuestionStage.CURRENT_SYMPTOMS
            return True, "病史信息已记录，请描述您今天的问题", None
        
        elif stage == QuestionStage.CURRENT_SYMPTOMS:
            self.current_session.current_stage = QuestionStage.ASSESSMENT
            return self._do_final_assessment()
        
        return False, "问诊完成", None
    
    # ==================== 风险评估 ====================
    
    def _assess_risk_realtime(self, text: str) -> Tuple[RiskLevel, Optional[str]]:
        """
        实时风险评估（两层判断）
        
        第一层：极端情况关键词（硬规则）→ 直接退出
        第二层：其他所有情况 → 大模型判断
        """
        text_lower = text.lower()
        
        # 第一层：极端情况（自杀自残等）- 硬规则，直接退出
        for keyword in EMERGENCY_KEYWORDS:
            if keyword in text_lower:
                self.current_session.risk_keywords_found = [keyword]
                msg = f"""
⚠️⚠️⚠️ 重要提醒 ⚠️⚠️⚠️

我注意到您提到了"{keyword}"，我非常担心您现在的状态。

【请立即寻求帮助】
• 全国心理援助热线：400-161-9995
• 北京心理危机研究与干预中心：010-82951332
• 或者告诉身边信任的人

您的生命很重要，请相信困难是暂时的。
如果您愿意，可以和我聊聊您的感受。
"""
                return RiskLevel.CRITICAL, msg
        
        # 第二层：调用大模型判断
        if self.llm:
            return self._llm_risk_assessment(text)
        
        # 如果没有大模型，返回低风险继续问诊
        return RiskLevel.LOW, None
    
    def _llm_risk_assessment(self, symptoms_text: str) -> Tuple[RiskLevel, Optional[str]]:
        """调用大模型进行风险评估"""
        user = self.current_user
        
        # 准备用户信息
        age = int(user.age) if user and user.age else "未知"
        gender = user.gender if user and user.gender else "未知"
        chronic = ", ".join(user.chronic_diseases) if user and user.chronic_diseases else "无"
        allergies = ", ".join(user.allergies) if user and user.allergies else "无"
        
        prompt = RISK_ASSESSMENT_PROMPT.format(
            age=age,
            gender=gender,
            chronic_diseases=chronic,
            allergies=allergies,
            symptoms=symptoms_text
        )
        
        try:
            print("  🤖 [AI正在分析症状严重程度...]")
            response = self.llm.invoke(prompt).content.strip()
            
            # 清理可能的markdown标记
            if "```" in response:
                parts = response.split("```")
                for part in parts:
                    if "{" in part:
                        response = part.replace("json", "").strip()
                        break
            
            result = json.loads(response)
            
            risk_map = {
                "CRITICAL": RiskLevel.CRITICAL,
                "HIGH": RiskLevel.HIGH,
                "MEDIUM": RiskLevel.MEDIUM,
                "LOW": RiskLevel.LOW,
            }
            
            level = risk_map.get(result.get("risk_level", "LOW").upper(), RiskLevel.LOW)
            reason = result.get("reason", "")
            advice = result.get("advice", "")
            
            # 保存判断理由
            self.current_session.llm_risk_reason = reason
            
            if level == RiskLevel.CRITICAL:
                msg = f"""
⚠️⚠️⚠️ 紧急提醒 ⚠️⚠️⚠️

根据您的描述，情况可能比较紧急。

【AI判断】{reason}

【建议】{advice}

请立即前往最近的医院急诊就医！
如有需要请拨打 120 急救电话。
"""
                return RiskLevel.CRITICAL, msg
            
            elif level == RiskLevel.HIGH:
                msg = f"""
⚠️ 健康提醒

【AI判断】{reason}

【建议】{advice}

建议您尽快（24小时内）前往医院就诊。
"""
                # HIGH级别不退出，但记录下来
                self.current_session.risk_keywords_found = ["AI判断为高风险"]
                return RiskLevel.HIGH, msg
            
            # MEDIUM 和 LOW 都继续问诊
            return RiskLevel.LOW, None
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️ AI返回格式错误，继续问诊")
            return RiskLevel.LOW, None
        except Exception as e:
            print(f"  ⚠️ AI判断出错: {e}，继续问诊")
            return RiskLevel.LOW, None
    
    def _do_final_assessment(self) -> Tuple[bool, str, RiskLevel]:
        """最终风险评估"""
        session = self.current_session
        user = self.current_user
        
        # 如果之前已经有高风险判断
        if session.llm_risk_reason:
            # 已经由大模型判断过了
            pass
        
        # 收集所有文本进行最终评估
        all_text = " ".join([
            session.chief_complaint or "",
            session.symptom_description or "",
        ])
        
        # 中等风险检测（关键词 + 严重程度）
        found_medium = [k for k in MEDIUM_RISK_KEYWORDS if k in all_text]
        severity = float(session.symptom_severity) if session.symptom_severity else 0
        
        if found_medium or severity >= 7:
            session.risk_level = RiskLevel.MEDIUM.value
            session.risk_keywords_found = found_medium
            session.referral_suggested = True
            self.save_session()
            
            return True, self._generate_medium_risk_message(found_medium), RiskLevel.MEDIUM
        
        # 低风险
        session.risk_level = RiskLevel.LOW.value
        self.save_session()
        
        return True, "感谢您提供的信息，我来为您分析一下...", RiskLevel.LOW
    
    def _generate_medium_risk_message(self, keywords: List[str]) -> str:
        """生成中等风险消息"""
        symptom_hint = f"（相关症状：{', '.join(keywords[:2])}）" if keywords else ""
        return f"""
📋 初步评估结果 {symptom_hint}

根据您提供的信息，建议您：
1. 近期安排时间到医院进行检查
2. 在此期间，我可以为您提供一些初步的健康建议

接下来我会根据医学知识库为您提供参考建议，
但请注意，这不能替代医生的专业诊断。

是否需要我为您提供一些初步建议？
"""
    
    def get_consultation_summary(self) -> Dict:
        """获取问诊摘要"""
        if not self.current_session or not self.current_user:
            return {}
        
        return {
            "user_profile": {
                "gender": self.current_user.gender,
                "age": self.current_user.age,
                "bmi": round(self.current_user.weight / ((self.current_user.height/100) ** 2), 1) 
                       if self.current_user.height > 0 else 0,
                "chronic_diseases": self.current_user.chronic_diseases,
                "allergies": self.current_user.allergies,
                "current_medications": self.current_user.current_medications,
            },
            "current_complaint": {
                "chief_complaint": self.current_session.chief_complaint,
                "duration": self.current_session.symptom_duration,
                "severity": self.current_session.symptom_severity,
            },
            "risk_assessment": {
                "level": self.current_session.risk_level,
                "keywords": self.current_session.risk_keywords_found,
                "llm_reason": self.current_session.llm_risk_reason,
            }
        }
    
    def generate_history_markdown(self) -> str:
        """生成用户历史的可读Markdown"""
        if not self.current_user:
            return ""
        
        user = self.current_user
        user_dir = self._get_user_dir(user.user_id)
        sessions_dir = os.path.join(user_dir, "sessions")
        
        lines = [
            f"# 用户健康档案",
            f"",
            f"**用户ID**: {user.user_id[:8]}...",
            f"**创建时间**: {user.created_at}",
            f"**最后访问**: {user.last_visit}",
            f"",
            f"## 基础信息",
            f"",
            f"| 项目 | 数值 |",
            f"|------|------|",
            f"| 性别 | {user.gender or '未填写'} |",
            f"| 年龄 | {user.age or '未填写'} |",
            f"| 身高 | {user.height}cm |" if user.height else "| 身高 | 未填写 |",
            f"| 体重 | {user.weight}kg |" if user.weight else "| 体重 | 未填写 |",
        ]
        
        if user.height and user.weight:
            bmi = round(user.weight / ((user.height/100) ** 2), 1)
            lines.append(f"| BMI | {bmi} |")
        
        lines.extend([
            f"",
            f"## 病史信息",
            f"",
            f"### 家族病史",
            f"{', '.join(user.family_history) if user.family_history else '无'}",
            f"",
            f"### 过敏史",
            f"{', '.join(user.allergies) if user.allergies else '无'}",
            f"",
            f"### 慢性病",
            f"{', '.join(user.chronic_diseases) if user.chronic_diseases else '无'}",
            f"",
            f"### 正在用药",
            f"{', '.join(user.current_medications) if user.current_medications else '无'}",
            f"",
        ])
        
        # 历史问诊记录
        if os.path.exists(sessions_dir):
            session_files = sorted(os.listdir(sessions_dir), reverse=True)
            if session_files:
                lines.extend([
                    f"## 问诊记录",
                    f"",
                ])
                for sf in session_files[:10]:
                    session_path = os.path.join(sessions_dir, sf)
                    try:
                        with open(session_path, 'r', encoding='utf-8') as f:
                            session = json.load(f)
                        lines.extend([
                            f"### {session.get('start_time', sf)}",
                            f"- **主诉**: {session.get('chief_complaint', '未记录')}",
                            f"- **风险等级**: {session.get('risk_level', '未评估')}",
                            f"- **AI判断**: {session.get('llm_risk_reason', '无')}",
                            f"",
                        ])
                    except:
                        pass
        
        # 保存Markdown
        md_path = os.path.join(user_dir, "history.md")
        content = "\n".join(lines)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return md_path


# 全局实例
consultation = StructuredConsultation()
