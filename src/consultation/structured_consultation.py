"""
结构化问诊模块 - 升级版 v2
- 集成自动身体指标计算与评估
- 新增咨询目的分流（健康管理 vs 身体不适）
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


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QuestionStage(str, Enum):
    IDENTIFICATION = "identification"
    BASIC_INFO = "basic_info"
    MEDICAL_HISTORY = "medical_history"
    CONSULTATION_TYPE = "consultation_type"  # 新增：咨询目的选择
    CURRENT_SYMPTOMS = "current_symptoms"
    ASSESSMENT = "assessment"
    ADVICE = "advice"
    COMPLETED = "completed"


class ConsultationType(str, Enum):
    """咨询类型"""
    HEALTH_MANAGEMENT = "health_management"  # 健康管理（减肥、养生等）
    SYMPTOM_CONSULTATION = "symptom_consultation"  # 身体不适咨询


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
    
    # 新增：咨询类型
    consultation_type: str = ""  # health_management 或 symptom_consultation
    
    # 症状信息（仅身体不适咨询时使用）
    chief_complaint: str = ""
    symptom_location: str = ""
    symptom_duration: str = ""
    symptom_severity: str = ""
    symptom_description: str = ""
    
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
    # 新增：咨询目的选择
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
            "important": True
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
        
        # 决定从哪个阶段开始
        if self.has_complete_profile():
            start_stage = QuestionStage.CONSULTATION_TYPE  # 老用户直接选咨询目的
            # 老用户直接计算指标
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
        self._store_answer(field_name, validated_answer, question)
        
        # 实时风险检测（仅针对症状描述）
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
            # 处理咨询目的选择，映射到内部值
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
            # 基础信息录完，计算指标
            self._perform_health_analysis()
            self.current_session.current_stage = QuestionStage.MEDICAL_HISTORY
            return True, "基础信息已记录，正在分析您的身体状况...", None
        
        elif stage == QuestionStage.MEDICAL_HISTORY:
            # 病史录完，进入咨询目的选择
            self.current_session.current_stage = QuestionStage.CONSULTATION_TYPE
            return True, "病史信息已记录，请选择您今天的咨询目的", None
        
        elif stage == QuestionStage.CONSULTATION_TYPE:
            # 根据咨询目的决定下一步
            if self.current_session.consultation_type == ConsultationType.HEALTH_MANAGEMENT.value:
                # 健康管理：跳过症状问题，直接进入评估
                self.current_session.current_stage = QuestionStage.ASSESSMENT
                self.current_session.risk_level = RiskLevel.LOW.value
                self.current_session.chief_complaint = "健康管理咨询"
                self.save_session()
                return False, "好的，我将根据您的身体状况为您提供健康管理建议...", RiskLevel.LOW
            else:
                # 身体不适：继续问症状
                self.current_session.current_stage = QuestionStage.CURRENT_SYMPTOMS
                return True, "请描述您的不适症状", None
        
        elif stage == QuestionStage.CURRENT_SYMPTOMS:
            self.current_session.current_stage = QuestionStage.ASSESSMENT
            return self._do_final_assessment()
        
        return False, "问诊完成", None
    
    # ==================== 健康指标计算 ====================
    
    def _perform_health_analysis(self):
        """执行后台计算和 AI 评估"""
        user = self.current_user
        session = self.current_session
        
        if not (user.height and user.weight and user.age):
            return
        
        # 1. 调用工具计算
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
        
        # 2. 调用 LLM 进行身体状态评估
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
        
        # 第一层：极端情况
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
        
        # 第二层：调用大模型判断
        if self.llm:
            return self._llm_risk_assessment(text)
        
        return RiskLevel.LOW, None
    
    def _llm_risk_assessment(self, symptoms_text: str) -> Tuple[RiskLevel, Optional[str]]:
        user = self.current_user
        
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
            
            # 清理markdown
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
        
        all_text = f"{session.chief_complaint} {session.symptom_description}"
        
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
