"""
结构化问诊模块 - 升级版 v3
- 集成自动身体指标计算与评估
- 咨询目的分流（健康管理 vs 身体不适）
- 多轮智能追问（最多3轮，由大模型决定是否追问及追问内容）
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
from tools import PURE_CALC_TOOLS


# ============================================================
# 配置
# ============================================================
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
MAX_FOLLOWUP_ROUNDS = 3  # 最多追问轮数

# 极端情况关键词 - 硬规则
EMERGENCY_KEYWORDS = [
    "想自杀", "不想活", "要自杀", "自杀", "自残", "自伤",
    "想死", "活不下去", "结束生命",
]

# 中等风险关键词
MEDIUM_RISK_KEYWORDS = [
    "持续疼痛", "反复发作", "越来越严重",
    "发烧", "高血压", "低血压", "心律不齐",
    "头晕", "眩晕", "恶心想吐",
    "皮疹", "过敏", "肿胀",
    "失眠严重", "焦虑", "抑郁",
]

# 大模型风险评估 Prompt
RISK_ASSESSMENT_PROMPT = """你是一名经验丰富的急诊分诊护士，需要根据患者描述判断紧急程度。

【患者信息】
- 年龄：{age}岁
- 性别：{gender}
- 慢性病史：{chronic_diseases}
- 过敏史：{allergies}
- 症状描述：{symptoms}

【判断标准】
- CRITICAL（危急）：需要立即拨打120或去急诊
- HIGH（紧急）：需要尽快就医（24小时内）
- MEDIUM（中等）：建议近期就医检查
- LOW（低风险）：可以继续咨询给建议

请直接输出JSON格式（不要任何其他内容）：
{{"risk_level": "CRITICAL/HIGH/MEDIUM/LOW", "reason": "简短判断理由", "advice": "给患者的建议"}}"""

# 大模型追问决策 Prompt
FOLLOWUP_DECISION_PROMPT = """你是一名专业的问诊医生。

【患者基本信息】
- 年龄：{age}岁
- 性别：{gender}
- 慢性病史：{chronic_diseases}

【已收集的症状信息】
{collected_info}

【任务】
判断是否还需要追问才能给出有效的健康建议。

【重要规则】
1. 每次只问一个问题，不要一次问多个问题
2. 问题要简短明确，不超过20个字
3. 如果提供选项，最多4个选项
4. 不要重复问已经收集到的信息
5. 最多追问3轮，如果信息已经足够就不要再追问

【判断标准】
- 如果症状描述清晰具体（如"左侧太阳穴跳痛"），不需要追问
- 如果缺少关键信息（如疼痛位置、性质），需要追问
- 如果已经追问过的信息，不要再问

【输出格式】
请直接输出JSON（不要任何其他内容）：
{{
    "need_followup": true或false,
    "question": "简短的追问问题（不超过20字）",
    "options": ["选项1", "选项2", "选项3", "选项4"]或null,
    "reason": "追问原因（简短）"
}}

示例输出：
{{"need_followup": true, "question": "头痛在什么位置？", "options": ["前额", "太阳穴", "后脑勺", "整个头"], "reason": "需要确定疼痛位置"}}
{{"need_followup": true, "question": "是什么样的疼法？", "options": ["跳痛", "胀痛", "刺痛", "闷痛"], "reason": "需要了解疼痛性质"}}
{{"need_followup": false, "question": "", "options": null, "reason": "信息已足够"}}"""


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QuestionStage(str, Enum):
    IDENTIFICATION = "identification"
    BASIC_INFO = "basic_info"
    MEDICAL_HISTORY = "medical_history"
    CONSULTATION_TYPE = "consultation_type"
    CURRENT_SYMPTOMS = "current_symptoms"
    FOLLOWUP = "followup"  # 新增：追问阶段
    ASSESSMENT = "assessment"
    ADVICE = "advice"
    COMPLETED = "completed"


class ConsultationType(str, Enum):
    HEALTH_MANAGEMENT = "health_management"
    SYMPTOM_CONSULTATION = "symptom_consultation"


@dataclass
class UserProfile:
    user_id: str
    phone_hash: str = ""
    created_at: str = ""
    last_visit: str = ""
    gender: str = ""
    age: int = 0
    height: float = 0.0
    weight: float = 0.0
    family_history: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    chronic_diseases: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)


@dataclass 
class ConsultationSession:
    session_id: str
    user_id: str
    start_time: str
    end_time: str = ""
    current_stage: QuestionStage = QuestionStage.IDENTIFICATION
    
    # 咨询类型
    consultation_type: str = ""
    
    # 症状信息
    chief_complaint: str = ""
    symptom_location: str = ""
    symptom_duration: str = ""
    symptom_severity: str = ""
    symptom_description: str = ""
    
    # 多轮追问记录
    followup_count: int = 0  # 已追问次数
    followup_qa: List[Dict] = field(default_factory=list)  # 追问问答记录
    current_followup_question: Dict = field(default_factory=dict)  # 当前追问问题
    
    # 评估结果
    risk_level: str = ""
    risk_keywords_found: List[str] = field(default_factory=list)
    llm_risk_reason: str = ""
    
    # 身体指标与评估
    health_metrics: Dict = field(default_factory=dict)
    health_assessment: str = ""
    
    advice_given: str = ""
    referral_suggested: bool = False
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
    QuestionStage.CONSULTATION_TYPE: [
        {
            "field": "consultation_type",
            "question": "请问您今天咨询的目的是？",
            "options": ["健康管理建议（减肥、养生、体检解读等）", "身体不适咨询（有具体症状需要咨询）"],
            "type": "choice",
            "mapping": {
                "健康管理建议（减肥、养生、体检解读等）": ConsultationType.HEALTH_MANAGEMENT.value,
                "身体不适咨询（有具体症状需要咨询）": ConsultationType.SYMPTOM_CONSULTATION.value,
            }
        },
    ],
    QuestionStage.CURRENT_SYMPTOMS: [
        {
            "field": "chief_complaint",
            "question": "请简单描述一下您哪里不舒服？",
            "type": "text",
            "important": True,
            "triggers_followup": True  # 标记：回答后触发追问判断
        },
    ],
    # 追问结束后的补充问题
    QuestionStage.FOLLOWUP: [
        {
            "field": "symptom_duration",
            "question": "这个症状持续多长时间了？",
            "options": ["今天刚开始", "1-3天", "一周左右", "一个月以上", "很长时间了"],
            "type": "choice"
        },
        {
            "field": "symptom_severity",
            "question": "如果用1-10分表示严重程度（1最轻，10最重），您给自己打几分？",
            "type": "number",
            "validation": {"min": 1, "max": 10}
        },
    ],
}


class StructuredConsultation:
    """结构化问诊管理器"""
    
    def __init__(self, data_dir: str = USER_DATA_DIR, llm=None):
        self.data_dir = data_dir
        self.llm = llm
        self._ensure_dirs()
        self.current_user: Optional[UserProfile] = None
        self.current_session: Optional[ConsultationSession] = None
        self.current_question_index: int = 0
    
    def set_llm(self, llm):
        self.llm = llm
    
    def _ensure_dirs(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _get_user_dir(self, user_id: str) -> str:
        user_dir = os.path.join(self.data_dir, user_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            os.makedirs(os.path.join(user_dir, "sessions"))
        return user_dir
    
    def _generate_user_id(self, identifier: str) -> str:
        hash_obj = hashlib.md5(identifier.encode())
        return str(uuid.UUID(hash_obj.hexdigest()))
    
    # ==================== 用户管理 ====================
    
    def identify_user(self, identifier: str) -> Tuple[UserProfile, bool]:
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
        user_dir = self._get_user_dir(profile.user_id)
        profile_path = os.path.join(user_dir, "profile.json")
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
    
    def _load_profile(self, user_id: str) -> UserProfile:
        user_dir = self._get_user_dir(user_id)
        profile_path = os.path.join(user_dir, "profile.json")
        with open(profile_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return UserProfile(**data)
    
    def has_complete_profile(self) -> bool:
        if not self.current_user:
            return False
        return all([
            self.current_user.gender,
            self.current_user.age > 0,
            self.current_user.height > 0,
            self.current_user.weight > 0,
        ])
    
    # ==================== 会话管理 ====================
    
    def start_session(self) -> ConsultationSession:
        if not self.current_user:
            raise ValueError("请先识别用户")
        
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        
        if self.has_complete_profile():
            start_stage = QuestionStage.CONSULTATION_TYPE
            session = ConsultationSession(
                session_id=session_id,
                user_id=self.current_user.user_id,
                start_time=now.strftime("%Y-%m-%d %H:%M:%S"),
                current_stage=start_stage
            )
            self.current_session = session
            self._perform_health_analysis()
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
        if not self.current_session:
            return None
        
        stage = self.current_session.current_stage
        
        # 如果在追问阶段
        if stage == QuestionStage.FOLLOWUP:
            # 优先返回AI生成的追问问题
            if self.current_session.current_followup_question:
                return self.current_session.current_followup_question
            # 否则返回固定的持续时间/严重程度问题
            followup_questions = QUESTIONS.get(QuestionStage.FOLLOWUP, [])
            if self.current_question_index < len(followup_questions):
                return followup_questions[self.current_question_index]
            return None
        
        if stage not in QUESTIONS:
            return None
        
        questions = QUESTIONS[stage]
        if self.current_question_index >= len(questions):
            return None
        
        return questions[self.current_question_index]
    
    def process_answer(self, answer: str) -> Tuple[bool, Optional[str], Optional[RiskLevel]]:
        if not self.current_session or not self.current_user:
            return False, "会话未初始化", None
        
        question = self.get_current_question()
        if not question:
            return False, "没有更多问题", None
        
        # 记录对话
        self.current_session.conversation.append({
            "role": "assistant",
            "content": question.get("question", "")
        })
        self.current_session.conversation.append({
            "role": "user",
            "content": answer
        })
        
        stage = self.current_session.current_stage
        
        # 处理追问阶段的回答
        if stage == QuestionStage.FOLLOWUP:
            return self._process_followup_answer(answer)
        
        # 验证并存储答案
        field_name = question.get("field", "")
        validated_answer = self._validate_answer(question, answer)
        
        if validated_answer is None:
            return True, f"输入无效，请重新回答：{question['question']}", None
        
        # 存储到相应位置
        self._store_answer(field_name, validated_answer, question)
        
        # 实时风险检测
        if question.get("important"):
            risk_level, risk_msg = self._assess_risk_realtime(answer)
            if risk_level == RiskLevel.CRITICAL:
                self.current_session.risk_level = risk_level.value
                self.current_session.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.save_session()
                return False, risk_msg, risk_level
        
        # 检查是否需要触发追问
        if question.get("triggers_followup") and self.llm:
            should_followup, followup_question = self._check_need_followup()
            if should_followup and followup_question:
                self.current_session.current_stage = QuestionStage.FOLLOWUP
                self.current_session.current_followup_question = followup_question
                return True, "🤔 我需要了解更多信息...", None
        
        # 移动到下一个问题
        self.current_question_index += 1
        
        # 检查是否完成当前阶段
        if self.current_question_index >= len(QUESTIONS.get(stage, [])):
            return self._advance_stage()
        
        return True, None, None
    
    def _process_followup_answer(self, answer: str) -> Tuple[bool, Optional[str], Optional[RiskLevel]]:
        """处理追问回答"""
        session = self.current_session
        question = self.get_current_question()
        
        # 如果是AI生成的追问（current_followup_question不为空）
        if session.current_followup_question:
            # 记录追问问答
            session.followup_qa.append({
                "question": session.current_followup_question.get("question", ""),
                "answer": answer
            })
            session.followup_count += 1
            
            # 风险检测
            risk_level, risk_msg = self._assess_risk_realtime(answer)
            if risk_level == RiskLevel.CRITICAL:
                session.risk_level = risk_level.value
                session.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.save_session()
                return False, risk_msg, risk_level
            
            # 检查是否需要继续追问
            if session.followup_count < MAX_FOLLOWUP_ROUNDS and self.llm:
                should_followup, followup_question = self._check_need_followup()
                if should_followup and followup_question:
                    session.current_followup_question = followup_question
                    return True, None, None
            
            # AI追问结束，清空当前追问问题，开始问固定的持续时间/严重程度问题
            session.current_followup_question = {}
            self.current_question_index = 0  # 重置索引，开始问FOLLOWUP阶段的固定问题
            return True, "✅ 好的，再问您几个问题就完成了", None
        
        # 处理固定问题（持续时间、严重程度）
        if question:
            field_name = question.get("field", "")
            validated_answer = self._validate_answer(question, answer)
            
            if validated_answer is None:
                return True, f"输入无效，请重新回答：{question['question']}", None
            
            # 存储到session
            setattr(session, field_name, validated_answer)
            
            # 移动到下一个问题
            self.current_question_index += 1
            
            # 检查是否完成FOLLOWUP阶段的固定问题
            followup_questions = QUESTIONS.get(QuestionStage.FOLLOWUP, [])
            if self.current_question_index >= len(followup_questions):
                # 所有问题问完，进入评估
                session.current_stage = QuestionStage.ASSESSMENT
                return self._do_final_assessment()
            
            return True, None, None
        
        # 没有问题了，进入评估
        session.current_stage = QuestionStage.ASSESSMENT
        return self._do_final_assessment()
    
    def _check_need_followup(self) -> Tuple[bool, Optional[Dict]]:
        """检查是否需要追问，并生成追问问题"""
        if not self.llm:
            return False, None
        
        user = self.current_user
        session = self.current_session
        
        # 构建已收集信息（清晰列出每一条）
        collected_info = []
        if session.chief_complaint:
            collected_info.append(f"• 主诉: {session.chief_complaint}")
        
        # 列出已追问的问答
        for i, qa in enumerate(session.followup_qa, 1):
            collected_info.append(f"• 追问{i}: {qa['question']}")
            collected_info.append(f"  回答: {qa['answer']}")
        
        collected_str = "\n".join(collected_info) if collected_info else "仅有主诉，无其他信息"
        
        prompt = FOLLOWUP_DECISION_PROMPT.format(
            age=int(user.age) if user.age else "未知",
            gender=user.gender or "未知",
            chronic_diseases=", ".join(user.chronic_diseases) if user.chronic_diseases else "无",
            collected_info=collected_str
        )
        
        try:
            print("  🤔 [AI正在判断是否需要追问...]")
            response = self.llm.invoke(prompt).content.strip()
            
            # 清理markdown
            if "```" in response:
                parts = response.split("```")
                for part in parts:
                    if "{" in part:
                        response = part.replace("json", "").strip()
                        break
            
            result = json.loads(response)
            
            if result.get("need_followup"):
                question_text = result.get("question", "")
                options = result.get("options")
                reason = result.get("reason", "")
                
                if question_text:
                    print(f"  💡 [追问原因: {reason}]")
                    
                    followup_q = {
                        "question": question_text,
                        "type": "choice" if options else "text",
                        "field": f"followup_{session.followup_count + 1}",
                    }
                    if options:
                        followup_q["options"] = options[:4]  # 最多4个选项
                    
                    return True, followup_q
            
            print("  ✅ [信息已足够，无需追问]")
            return False, None
            
        except json.JSONDecodeError:
            print("  ⚠️ AI返回格式错误，跳过追问")
            return False, None
        except Exception as e:
            print(f"  ⚠️ 追问判断出错: {e}")
            return False, None
    
    def _validate_answer(self, question: Dict, answer: str) -> Optional[any]:
        q_type = question.get("type", "text")
        
        if q_type == "choice":
            options = question.get("options", [])
            if answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            if answer in options:
                return answer
            # 对于追问的选择题，允许自由回答
            if question.get("field", "").startswith("followup_"):
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
    
    def _store_answer(self, field_name: str, value: any, question: Dict = None):
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
        
        elif stage == QuestionStage.CONSULTATION_TYPE:
            mapping = question.get("mapping", {}) if question else {}
            internal_value = mapping.get(value, value)
            self.current_session.consultation_type = internal_value
        
        elif stage == QuestionStage.CURRENT_SYMPTOMS:
            setattr(self.current_session, field_name, value)
    
    def _advance_stage(self) -> Tuple[bool, Optional[str], Optional[RiskLevel]]:
        """进入下一阶段"""
        stage = self.current_session.current_stage
        self.current_question_index = 0
        
        if stage == QuestionStage.BASIC_INFO:
            self._perform_health_analysis()
            self.current_session.current_stage = QuestionStage.MEDICAL_HISTORY
            return True, "基础信息已记录，正在分析您的身体状况...", None
        
        elif stage == QuestionStage.MEDICAL_HISTORY:
            self.current_session.current_stage = QuestionStage.CONSULTATION_TYPE
            return True, "病史信息已记录，请选择您今天的咨询目的", None
        
        elif stage == QuestionStage.CONSULTATION_TYPE:
            if self.current_session.consultation_type == ConsultationType.HEALTH_MANAGEMENT.value:
                self.current_session.current_stage = QuestionStage.ASSESSMENT
                self.current_session.risk_level = RiskLevel.LOW.value
                self.current_session.chief_complaint = "健康管理咨询"
                self.save_session()
                return False, "好的，我将根据您的身体状况为您提供健康管理建议...", RiskLevel.LOW
            else:
                self.current_session.current_stage = QuestionStage.CURRENT_SYMPTOMS
                return True, "请描述您的不适症状", None
        
        elif stage == QuestionStage.CURRENT_SYMPTOMS:
            # 如果没有追问，直接进入评估
            self.current_session.current_stage = QuestionStage.ASSESSMENT
            return self._do_final_assessment()
        
        return False, "问诊完成", None
    
    # ==================== 健康指标计算 ====================
    
    def _perform_health_analysis(self):
        user = self.current_user
        session = self.current_session
        
        if not (user.height and user.weight and user.age):
            return
        
        try:
            bmi_result = PURE_CALC_TOOLS["BMI"](user.height, user.weight)
            bmr_result = PURE_CALC_TOOLS["BMR"](user.weight, user.height, int(user.age), user.gender)
            ideal_result = PURE_CALC_TOOLS["IDEAL_WEIGHT"](user.height, user.gender)
            
            bmi = bmi_result.get("value")
            bmr = bmr_result.get("value")
            ideal = ideal_result.get("value")
            
            session.health_metrics = {
                "BMI": bmi,
                "BMR": bmr,
                "IdealWeight": ideal
            }
        except Exception as e:
            print(f"  ⚠️ 计算出错: {e}")
            return
        
        if self.llm:
            try:
                prompt = f"""你是一名专业健康管理师。请根据以下客观数据，用简练的语言判断该用户的身体状况。

【用户数据】
- {int(user.age)}岁 {user.gender}性
- 身高: {user.height}cm, 体重: {user.weight}kg
- BMI: {bmi} (正常范围18.5-24)
- BMR: {bmr} kcal/day (基础代谢)
- 理想体重约: {ideal}kg

【要求】
1. 判断体重状态（偏瘦/标准/超重/肥胖）
2. 一句话总结，例如"体重属于超重范围，基础代谢正常。"
3. 不要给建议，仅做事实判断。"""
                
                print("  🤖 [AI正在分析身体指标...]")
                assessment = self.llm.invoke(prompt).content.strip()
                session.health_assessment = assessment
            except Exception as e:
                print(f"  ⚠️ AI分析出错: {e}")
                session.health_assessment = "身体状况分析暂不可用"
    
    # ==================== 风险评估 ====================
    
    def _assess_risk_realtime(self, text: str) -> Tuple[RiskLevel, Optional[str]]:
        text_lower = text.lower()
        
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
"""
                return RiskLevel.CRITICAL, msg
        
        if self.llm:
            return self._llm_risk_assessment(text)
        
        return RiskLevel.LOW, None
    
    def _llm_risk_assessment(self, symptoms_text: str) -> Tuple[RiskLevel, Optional[str]]:
        user = self.current_user
        session = self.current_session
        
        # 整合所有症状信息
        all_symptoms = [symptoms_text]
        if session.chief_complaint and session.chief_complaint != symptoms_text:
            all_symptoms.insert(0, session.chief_complaint)
        for qa in session.followup_qa:
            all_symptoms.append(f"{qa['question']}: {qa['answer']}")
        
        symptoms_combined = "\n".join(all_symptoms)
        
        age = int(user.age) if user and user.age else "未知"
        gender = user.gender if user and user.gender else "未知"
        chronic = ", ".join(user.chronic_diseases) if user and user.chronic_diseases else "无"
        allergies = ", ".join(user.allergies) if user and user.allergies else "无"
        
        prompt = RISK_ASSESSMENT_PROMPT.format(
            age=age,
            gender=gender,
            chronic_diseases=chronic,
            allergies=allergies,
            symptoms=symptoms_combined
        )
        
        try:
            print("  🤖 [AI正在分析症状严重程度...]")
            response = self.llm.invoke(prompt).content.strip()
            
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
            
            self.current_session.llm_risk_reason = reason
            
            if level == RiskLevel.CRITICAL:
                msg = f"""
⚠️⚠️⚠️ 紧急提醒 ⚠️⚠️⚠️

根据您的描述，情况可能比较紧急。

【AI判断】{reason}
【建议】{advice}

请立即前往最近的医院急诊就医！
"""
                return RiskLevel.CRITICAL, msg
            
            elif level == RiskLevel.HIGH:
                msg = f"""
⚠️ 健康提醒

【AI判断】{reason}
【建议】{advice}

建议您尽快（24小时内）前往医院就诊。
"""
                self.current_session.risk_keywords_found = ["AI判断为高风险"]
                return RiskLevel.HIGH, msg
            
            return RiskLevel.LOW, None
            
        except json.JSONDecodeError:
            print("  ⚠️ AI返回格式错误，继续问诊")
            return RiskLevel.LOW, None
        except Exception as e:
            print(f"  ⚠️ AI判断出错: {e}，继续问诊")
            return RiskLevel.LOW, None
    
    def _do_final_assessment(self) -> Tuple[bool, str, RiskLevel]:
        session = self.current_session
        
        # 整合所有症状信息
        all_text = session.chief_complaint or ""
        for qa in session.followup_qa:
            all_text += f" {qa['answer']}"
        
        found_medium = [k for k in MEDIUM_RISK_KEYWORDS if k in all_text]
        severity = float(session.symptom_severity) if session.symptom_severity else 0
        
        if found_medium or severity >= 7:
            session.risk_level = RiskLevel.MEDIUM.value
            session.risk_keywords_found = found_medium
            session.referral_suggested = True
            self.save_session()
            
            symptom_hint = f"（相关症状：{', '.join(found_medium[:2])}）" if found_medium else ""
            return True, f"初步评估：建议近期就医检查{symptom_hint}。我也为您准备了一些参考建议。", RiskLevel.MEDIUM
        
        session.risk_level = RiskLevel.LOW.value
        self.save_session()
        
        return True, "感谢您的配合。我正在结合您的身体指标和症状生成建议...", RiskLevel.LOW
    
    # ==================== 摘要与导出 ====================
    
    def get_consultation_summary(self) -> Dict:
        if not self.current_session or not self.current_user:
            return {}
        
        # 整合追问信息到症状描述
        symptom_details = []
        if self.current_session.chief_complaint:
            symptom_details.append(f"主诉: {self.current_session.chief_complaint}")
        for qa in self.current_session.followup_qa:
            symptom_details.append(f"{qa['question']}: {qa['answer']}")
        
        return {
            "user_profile": {
                "gender": self.current_user.gender,
                "age": self.current_user.age,
                "height": self.current_user.height,
                "weight": self.current_user.weight,
                "chronic_diseases": self.current_user.chronic_diseases,
                "allergies": self.current_user.allergies,
                "current_medications": self.current_user.current_medications,
            },
            "health_metrics": self.current_session.health_metrics,
            "health_assessment": self.current_session.health_assessment,
            "consultation_type": self.current_session.consultation_type,
            "current_complaint": {
                "chief_complaint": self.current_session.chief_complaint,
                "symptom_details": symptom_details,  # 包含追问详情
                "duration": self.current_session.symptom_duration,
                "severity": self.current_session.symptom_severity,
            },
            "followup_qa": self.current_session.followup_qa,  # 追问记录
            "risk_assessment": {
                "level": self.current_session.risk_level,
                "keywords": self.current_session.risk_keywords_found,
                "llm_reason": self.current_session.llm_risk_reason,
            }
        }
    
    def generate_history_markdown(self) -> str:
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
            f"| 年龄 | {int(user.age) if user.age else '未填写'} |",
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
                lines.extend([f"## 问诊记录", f""])
                for sf in session_files[:10]:
                    session_path = os.path.join(sessions_dir, sf)
                    try:
                        with open(session_path, 'r', encoding='utf-8') as f:
                            session = json.load(f)
                        
                        consult_type = session.get('consultation_type', '')
                        type_label = "健康管理" if consult_type == "health_management" else "症状咨询"
                        
                        lines.extend([
                            f"### {session.get('start_time', sf)} [{type_label}]",
                            f"- **主诉**: {session.get('chief_complaint', '未记录')}",
                        ])
                        
                        # 显示追问记录
                        followup_qa = session.get('followup_qa', [])
                        if followup_qa:
                            lines.append(f"- **追问详情**:")
                            for qa in followup_qa:
                                lines.append(f"  - {qa['question']} → {qa['answer']}")
                        
                        lines.extend([
                            f"- **风险等级**: {session.get('risk_level', '未评估')}",
                            f"- **AI判断**: {session.get('llm_risk_reason', '无')}",
                            f"",
                        ])
                    except:
                        pass
        
        md_path = os.path.join(user_dir, "history.md")
        content = "\n".join(lines)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return md_path


# 全局实例
consultation = StructuredConsultation()