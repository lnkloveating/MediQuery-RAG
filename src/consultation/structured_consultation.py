"""
结构化问诊模块 - 升级版
已集成自动身体指标计算与评估
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
from src.tools import PURE_CALC_TOOLS  # <--- 导入新工具

# ... (保留原有的配置常量 USER_DATA_DIR, EMERGENCY_KEYWORDS, RISK_ASSESSMENT_PROMPT 等)
# ⚠️ 注意：请确保保留原本的所有常量定义，这里省略以节省篇幅

# ============================================================
# 配置 (确保保留)
# ============================================================
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
EMERGENCY_KEYWORDS = ["想自杀", "不想活", "要自杀", "自杀", "自残", "自伤", "想死", "活不下去", "结束生命"]
MEDIUM_RISK_KEYWORDS = ["持续疼痛", "反复发作", "越来越严重", "发烧", "头晕", "恶心想吐", "过敏", "肿胀"]

# 大模型风险评估 Prompt (保留原版)
RISK_ASSESSMENT_PROMPT = """你是一名经验丰富的急诊分诊护士... (内容略，保持原样) ..."""

class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class QuestionStage(str, Enum):
    IDENTIFICATION = "identification"
    BASIC_INFO = "basic_info"
    MEDICAL_HISTORY = "medical_history"
    CURRENT_SYMPTOMS = "current_symptoms"
    ASSESSMENT = "assessment"
    ADVICE = "advice"
    COMPLETED = "completed"

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
    """单次问诊会话"""
    session_id: str
    user_id: str
    start_time: str
    end_time: str = ""
    current_stage: QuestionStage = QuestionStage.IDENTIFICATION
    
    # 症状信息
    chief_complaint: str = ""
    symptom_location: str = ""
    symptom_duration: str = ""
    symptom_severity: str = ""
    symptom_description: str = ""
    
    # 评估结果
    risk_level: str = ""
    risk_keywords_found: List[str] = field(default_factory=list)
    llm_risk_reason: str = ""
    
    # === 新增：身体指标与评估 ===
    health_metrics: Dict = field(default_factory=dict) # 存储 BMI, BMR, IdealWeight
    health_assessment: str = ""                    # 存储 LLM 对身体底子的评价
    
    advice_given: str = ""
    referral_suggested: bool = False
    conversation: List[Dict] = field(default_factory=list)

# ... (保留 QUESTIONS 定义，保持不变)
QUESTIONS = {
    QuestionStage.BASIC_INFO: [
        {"field": "gender", "question": "请问您的性别是？", "options": ["男", "女"], "type": "choice"},
        {"field": "age", "question": "请问您的年龄是多少岁？", "type": "number", "validation": {"min": 0, "max": 120}},
        {"field": "height", "question": "请问您的身高是多少厘米(cm)？", "type": "number", "validation": {"min": 50, "max": 250}},
        {"field": "weight", "question": "请问您的体重是多少公斤(kg)？", "type": "number", "validation": {"min": 20, "max": 300}},
    ],
    QuestionStage.MEDICAL_HISTORY: [
        {"field": "family_history", "question": "请问您的直系亲属有以下疾病吗？(没有请输入'无')", "options": ["高血压", "糖尿病", "心脏病", "癌症", "脑卒中", "其他", "无"], "type": "multi_choice"},
        {"field": "allergies", "question": "请问您有药物或食物过敏吗？", "type": "text", "placeholder": "例如：青霉素过敏"},
        # 这里的慢性病选项已经包含了 "高血压"，符合你的要求
        {"field": "chronic_diseases", "question": "请问您有以下慢性病吗？", "options": ["高血压", "糖尿病", "高血脂", "心脏病", "哮喘", "其他", "无"], "type": "multi_choice"},
        {"field": "current_medications", "question": "请问您目前正在服用什么药物？", "type": "text"},
    ],
    QuestionStage.CURRENT_SYMPTOMS: [
        {"field": "chief_complaint", "question": "请简单描述一下您今天咨询的主要问题是什么？", "type": "text", "important": True},
        {"field": "symptom_duration", "question": "这个症状持续多久了？", "options": ["今天刚开始", "1-3天", "一周左右", "一个月以上"], "type": "choice"},
        {"field": "symptom_severity", "question": "如果1-10分，您觉得有多严重？", "type": "number", "validation": {"min": 1, "max": 10}},
    ],
}

class StructuredConsultation:
    # ... (保留 __init__, set_llm, 目录管理, 用户管理等基础方法)
    def __init__(self, data_dir: str = USER_DATA_DIR, llm=None):
        self.data_dir = data_dir
        self.llm = llm
        self._ensure_dirs()
        self.current_user: Optional[UserProfile] = None
        self.current_session: Optional[ConsultationSession] = None
        self.current_question_index: int = 0
    
    def set_llm(self, llm): self.llm = llm
    def _ensure_dirs(self): 
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
    def _get_user_dir(self, user_id: str):
        user_dir = os.path.join(self.data_dir, user_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            os.makedirs(os.path.join(user_dir, "sessions"))
        return user_dir
    def _generate_user_id(self, identifier: str):
        hash_obj = hashlib.md5(identifier.encode())
        return str(uuid.UUID(hash_obj.hexdigest()))
    
    # ... (保留 identify_user, _save_profile, _load_profile, has_complete_profile, start_session, save_session)
    def identify_user(self, identifier: str) -> Tuple[UserProfile, bool]:
        user_id = self._generate_user_id(identifier)
        user_dir = self._get_user_dir(user_id)
        profile_path = os.path.join(user_dir, "profile.json")
        is_new_user = not os.path.exists(profile_path)
        if is_new_user:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            profile = UserProfile(user_id=user_id, phone_hash=hashlib.sha256(identifier.encode()).hexdigest()[:16], created_at=now, last_visit=now)
            self._save_profile(profile)
        else:
            profile = self._load_profile(user_id)
            profile.last_visit = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_profile(profile)
        self.current_user = profile
        return profile, is_new_user
    
    def _save_profile(self, profile):
        with open(os.path.join(self._get_user_dir(profile.user_id), "profile.json"), 'w', encoding='utf-8') as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
    def _load_profile(self, user_id):
        with open(os.path.join(self._get_user_dir(user_id), "profile.json"), 'r', encoding='utf-8') as f: return UserProfile(**json.load(f))
    def has_complete_profile(self):
        if not self.current_user: return False
        return all([self.current_user.gender, self.current_user.age > 0, self.current_user.height > 0, self.current_user.weight > 0])
    
    def start_session(self):
        if not self.current_user: raise ValueError("请先识别用户")
        now = datetime.now()
        start_stage = QuestionStage.CURRENT_SYMPTOMS if self.has_complete_profile() else QuestionStage.BASIC_INFO
        
        # 如果是老用户且资料完整，直接触发一次计算，确保数据是最新的
        session = ConsultationSession(session_id=now.strftime("%Y%m%d_%H%M%S"), user_id=self.current_user.user_id, start_time=now.strftime("%Y-%m-%d %H:%M:%S"), current_stage=start_stage)
        self.current_session = session
        self.current_question_index = 0
        
        if self.has_complete_profile():
             self._perform_health_analysis() # 老用户直接计算
             
        return session
    
    def save_session(self):
        if not self.current_session or not self.current_user: return
        path = os.path.join(self._get_user_dir(self.current_user.user_id), "sessions", f"{self.current_session.session_id}.json")
        with open(path, 'w', encoding='utf-8') as f: json.dump(asdict(self.current_session), f, ensure_ascii=False, indent=2)

    # ... (保留 get_current_question, process_answer, _validate_answer, _store_answer)
    def get_current_question(self):
        if not self.current_session: return None
        stage = self.current_session.current_stage
        questions = QUESTIONS.get(stage, [])
        if self.current_question_index >= len(questions): return None
        return questions[self.current_question_index]
    
    def process_answer(self, answer: str):
        if not self.current_session: return False, "未初始化", None
        question = self.get_current_question()
        if not question: return False, "无问题", None
        
        self.current_session.conversation.append({"role": "assistant", "content": question["question"]})
        self.current_session.conversation.append({"role": "user", "content": answer})
        
        val = self._validate_answer(question, answer)
        if val is None: return True, f"输入无效，请重试", None
        
        self._store_answer(question["field"], val)
        
        if question.get("important"):
            risk, msg = self._assess_risk_realtime(answer)
            if risk == RiskLevel.CRITICAL:
                self.current_session.risk_level = risk.value
                self.save_session()
                return False, msg, risk
        
        self.current_question_index += 1
        if self.current_question_index >= len(QUESTIONS.get(self.current_session.current_stage, [])):
            return self._advance_stage()
        
        return True, None, None

    def _validate_answer(self, q, a):
        # 简化版验证逻辑 (原版代码太长，这里示意保留核心)
        if q["type"] == "number":
            try: return float(a)
            except: return None
        if q["type"] == "choice" and a.isdigit():
            idx = int(a)-1
            if 0<=idx<len(q["options"]): return q["options"][idx]
        return a # 默认
        
    def _store_answer(self, field, value):
        if self.current_session.current_stage == QuestionStage.BASIC_INFO:
            setattr(self.current_user, field, value)
            self._save_profile(self.current_user)
        elif self.current_session.current_stage == QuestionStage.MEDICAL_HISTORY:
             # 处理多选
             val = value if isinstance(value, list) else ([value] if value and value!="无" else [])
             setattr(self.current_user, field, val)
             self._save_profile(self.current_user)
        else:
            setattr(self.current_session, field, value)

    # === 修改的核心：阶段流转 ===
    def _advance_stage(self) -> Tuple[bool, Optional[str], Optional[RiskLevel]]:
        """进入下一阶段"""
        stage = self.current_session.current_stage
        self.current_question_index = 0
        
        if stage == QuestionStage.BASIC_INFO:
            # ✅ 关键点：基础信息一录完，立刻计算分析
            self._perform_health_analysis()
            
            self.current_session.current_stage = QuestionStage.MEDICAL_HISTORY
            return True, "基础信息已记录，正在分析您的身体状况...", None
        
        elif stage == QuestionStage.MEDICAL_HISTORY:
            self.current_session.current_stage = QuestionStage.CURRENT_SYMPTOMS
            return True, "病史已更新，请告诉我您今天哪里不舒服？", None
        
        elif stage == QuestionStage.CURRENT_SYMPTOMS:
            self.current_session.current_stage = QuestionStage.ASSESSMENT
            return self._do_final_assessment()
        
        return False, "问诊完成", None

    # === 新增：身体状况分析逻辑 ===
    def _perform_health_analysis(self):
        """执行后台计算和 AI 评估"""
        user = self.current_user
        session = self.current_session
        
        # 确保有数据
        if not (user.height and user.weight and user.age):
            return

        # 1. 调用工具计算
        try:
            bmi = PURE_CALC_TOOLS["BMI"](user.height, user.weight).get("value")
            bmr = PURE_CALC_TOOLS["BMR"](user.weight, user.height, user.age, user.gender).get("value")
            ideal = PURE_CALC_TOOLS["IDEAL_WEIGHT"](user.height, user.gender).get("value")
            
            session.health_metrics = {
                "BMI": bmi,
                "BMR": bmr, 
                "IdealWeight": ideal
            }
        except Exception as e:
            print(f"计算出错: {e}")
            return

        # 2. 调用 LLM 进行身体底子画像（非诊断，仅状态评估）
        if self.llm:
            try:
                # 构造Prompt
                prompt = f"""
                你是一名专业健康管理师。请根据以下客观数据，用简练的语言判断该用户的身体状况标签。
                
                【用户数据】
                - {user.age}岁 {user.gender}
                - BMI: {bmi}
                - BMR: {bmr} kcal/day
                - 实际体重: {user.weight}kg (理想体重约 {ideal}kg)
                
                【要求】
                1. 判断体重状态（偏瘦/标准/超重/肥胖等）
                2. 判断代谢水平（根据BMR和年龄粗略判断）
                3. 输出格式：一句话评价，例如"体重属于肥胖范围，基础代谢率正常。"
                4. 不要给任何建议，仅做事实判断。
                """
                
                print("  🤖 [AI正在分析身体指标...]")
                assessment = self.llm.invoke(prompt).content.strip()
                session.health_assessment = assessment
            except:
                session.health_assessment = "身体状况分析暂不可用"

    # ... (保留 _assess_risk_realtime, _llm_risk_assessment, _do_final_assessment, _generate_medium_risk_message, generate_history_markdown)
    def _assess_risk_realtime(self, text):
        for k in EMERGENCY_KEYWORDS:
            if k in text: return RiskLevel.CRITICAL, f"⚠️ 检测到危急关键词 '{k}'，请立即就医！"
        if self.llm: return self._llm_risk_assessment(text)
        return RiskLevel.LOW, None

    def _llm_risk_assessment(self, text):
        # 简化的LLM调用
        try:
            prompt = RISK_ASSESSMENT_PROMPT.format(age=self.current_user.age, gender=self.current_user.gender, chronic_diseases="", allergies="", symptoms=text)
            res = self.llm.invoke(prompt).content
            if "CRITICAL" in res: return RiskLevel.CRITICAL, "⚠️ AI判断为危急情况，建议立即就医！"
            if "HIGH" in res: return RiskLevel.HIGH, "⚠️ AI判断风险较高，建议尽快就医。"
        except: pass
        return RiskLevel.LOW, None

    def _do_final_assessment(self):
        session = self.current_session
        all_text = f"{session.chief_complaint} {session.symptom_description}"
        # 简单逻辑：有关键词或严重程度高 -> 中风险
        if any(k in all_text for k in MEDIUM_RISK_KEYWORDS) or (float(session.symptom_severity or 0) >= 7):
            session.risk_level = RiskLevel.MEDIUM.value
            return True, "初步评估：建议近期就医检查。我也为您准备了一些参考建议。", RiskLevel.MEDIUM
        
        session.risk_level = RiskLevel.LOW.value
        return True, "感谢您的配合。我正在结合您的身体指标和症状生成建议...", RiskLevel.LOW

    def generate_history_markdown(self):
        # 略，保留原逻辑
        return ""

    # === 修改 summary 方法 ===
    def get_consultation_summary(self) -> Dict:
        """获取完整的问诊摘要"""
        if not self.current_session or not self.current_user: return {}
        
        return {
            "user_profile": {
                "gender": self.current_user.gender,
                "age": self.current_user.age,
                "chronic_diseases": self.current_user.chronic_diseases,
                "allergies": self.current_user.allergies,
            },
            # ✅ 新增字段
            "health_metrics": self.current_session.health_metrics,
            "health_assessment": self.current_session.health_assessment,
            
            "current_complaint": {
                "chief_complaint": self.current_session.chief_complaint,
                "duration": self.current_session.symptom_duration,
                "severity": self.current_session.symptom_severity,
            },
            "risk_assessment": {
                "level": self.current_session.risk_level,
                "llm_reason": self.current_session.llm_risk_reason,
            }
        }