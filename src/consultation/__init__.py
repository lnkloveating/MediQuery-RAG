"""
问诊模块
- structured_consultation: 结构化问诊流程（医疗建议模式）
"""
from .structured_consultation import (
    StructuredConsultation,
    consultation,
    UserProfile,
    ConsultationSession,
    RiskLevel,
    QuestionStage,
)

__all__ = [
    'StructuredConsultation',
    'consultation',
    'UserProfile',
    'ConsultationSession',
    'RiskLevel',
    'QuestionStage',
]
