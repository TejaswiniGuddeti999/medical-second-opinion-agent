from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime

# Enums
class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"

class EvidenceQuality(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


# Input schemas

class PatientCase(BaseModel):
    """
    What the doctor uploads. Every field is optional except symptoms
    because real-world cases are always incomplete.
    """
    symptoms: str = Field(
        ...,
        description = "Primary symptoms , onset, duration",
        min_length = 10
    )
    lab_results: Optional[str] = Field(
        None,
        description = "Lab values, CBC, metabolic panel etc.",
    )
    imaging_notes : Optional[str] = Field(
        None,
        description = "Radiology reports, imaging findings"
    )
    current_medications: Optional[str] = Field(
        None,
        description = "Current medications for drug interaction check"
    )
    patint_age : Optional[int] = Field(None, ge=0, le=130)
    patient_sex: Optional[str] = Field(None, patter = "^(male|female|other)")
    clinical_history: Optional[str] = Field(
        None,
        description = "Relevant past medical history"
    )

# per agent output schemas

class PubMedArticle(BaseModel):
    """
    A single research article returned by ResearchAgent
    """
    title: str
    authors:list[str] =[]
    journal: str= ""
    year: Optional[int] = None
    pmid: str
    relevance_summary: str # why this article matters for this case
    evidence_quality: EvidenceQuality


class ResearchAgentOutput(BaseModel):
    articles: list[PubMedArticle]
    research_summary: str
    recommended_workup: list[str] = Field(
        default_factory = list,
        description = "Tests or investigations suggested by literature"
    )
    agent_position: str = Field(
        ...,
        description = "The agent's strong diagnostic position based on evidence"
    )

class DrugInteraction(BaseModel):
    drug_name:str
    interaction_type:str
    severity: str
    description: str

class SafetyAgentOutput(BaseModel):
    interactions_found: list[DrugInteraction] = []
    contraindications: list[str] =[]
    safety_summary: str
    is_safe_to_proceed: bool
    agent_position: str = Field(
        ...,
        description = "The agent's strong safety-based position"   
    )

class DiagnosisCandidate(BaseModel):
    condition: str
    confidence: float = Field(..., ge= 0.0, le = 1.0)
    confidence_level: ConfidenceLevel
    supporting_evidence: list[str]
    against_evidence: list[str]
    recommended_tests: list[str] =[]

class DiagnosisAgentOutput(BaseModel):
    primary_diagnosis: DiagnosisCandidate
    differential_diagnoses: list[DiagnosisCandidate]  # plural
    reasoning: str
    agent_position: str

# Synthesis (final report)

class AgentDisagreement(BaseModel):
    """Captures where agents disagreed - this is the most valuable part"""
    topic: str
    research_view: str
    safety_view: str
    diagnosis_view: str
    resolution: str
    resolution_confidence: ConfidenceLevel

class FinalReport(BaseModel):
    primary_diagnosis: str                    # ← correct spelling
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    research_output: ResearchAgentOutput
    safety_output: SafetyAgentOutput
    diagnosis_output: DiagnosisAgentOutput
    disagreements: list[AgentDisagreement] = []
    consensus_points: list[str] = []
    immediate_actions: list[str]
    further_investigations: list[str]
    red_flags: list[str] = []
    cited_sources: list[str] = []
    disclaimer: str = (
        "This is an AI-generated second opinion for informational purposes only. "
        "A licensed physician must make all clinical decisions."
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# API response wrappers

class AnalysisResponse(BaseModel):
    case_id: str
    status: str
    report: Optional[FinalReport] = None
    error: Optional[str] = None
    processing_time_seconds: Optional[float] = None

class HealthCheck(BaseModel):
    status: str = "ok"
    version: str
    database: str