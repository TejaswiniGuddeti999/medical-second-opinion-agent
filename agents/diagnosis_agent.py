import json
import asyncio
from openai import AsyncOpenAI
from app.config import get_settings
from app.schemas import (
    DiagnosisAgentOutput,
    DiagnosisCandidate,
    ConfidenceLevel,
)
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)


def confidence_to_level(score: float) -> ConfidenceLevel:
    if score >= 0.70:
        return ConfidenceLevel.HIGH
    elif score >= 0.40:
        return ConfidenceLevel.MODERATE
    else:
        return ConfidenceLevel.LOW


async def run_diagnosis_agent(case_summary: str) -> DiagnosisAgentOutput:
    logger.info("diagnosis_agent_started")

    prompt = f"""You are a DiagnosisAgent in a multi-agent medical second opinion system.
Your role: reason like an experienced clinician. Build a differential diagnosis.
You MUST take a strong, assertive primary diagnosis position.
Other agents have their own positions. Yours must be clearly stated so the debate is real.

IMPORTANT RULES:
- Reason from symptoms, labs, imaging, and history only
- Assign realistic confidence scores — do not give everything 0.9
- List specific evidence FOR and AGAINST each diagnosis
- Your primary diagnosis is the one you would stake your clinical reputation on

PATIENT CASE:
{case_summary}

Respond in this EXACT JSON format:
{{
  "primary_diagnosis": {{
    "condition": "specific condition name",
    "confidence": 0.75,
    "confidence_level": "high|moderate|low",
    "supporting_evidence": [
      "specific finding from the case that supports this"
    ],
    "against_evidence": [
      "finding that argues against this diagnosis"
    ],
    "recommended_tests": ["test to confirm", "test to rule out alternatives"]
  }},
  "differential_diagnoses": [
    {{
      "condition": "alternative condition",
      "confidence": 0.45,
      "confidence_level": "moderate",
      "supporting_evidence": ["finding that fits"],
      "against_evidence": ["finding that does not fit"],
      "recommended_tests": ["confirmatory test"]
    }}
  ],
  "reasoning": "Your step-by-step clinical reasoning. Walk through how you arrived at the primary diagnosis.",
  "agent_position": "One assertive sentence. Your single strongest clinical position."
}}

Rules for differential_diagnoses:
- Include 2-4 alternatives minimum
- Order by confidence score descending

Return ONLY the JSON. No preamble, no markdown fences."""

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    def parse_candidate(d: dict) -> DiagnosisCandidate:
        confidence = float(d.get("confidence", 0.5))
        return DiagnosisCandidate(
            condition=d["condition"],
            confidence=confidence,
            confidence_level=confidence_to_level(confidence),
            supporting_evidence=d.get("supporting_evidence", []),
            against_evidence=d.get("against_evidence", []),
            recommended_tests=d.get("recommended_tests", []),
        )

    primary = parse_candidate(data["primary_diagnosis"])

    # handle both plural and singular just in case LLM varies
    raw_differentials = (
        data.get("differential_diagnoses") or
        data.get("differential_diagnosis") or
        []
    )
    differentials = [parse_candidate(d) for d in raw_differentials]

    result = DiagnosisAgentOutput(
        primary_diagnosis=primary,
        differential_diagnoses=differentials,
        reasoning=data["reasoning"],
        agent_position=data["agent_position"],
    )

    logger.info(
        "diagnosis_agent_complete",
        primary=result.primary_diagnosis.condition,
        confidence=result.primary_diagnosis.confidence,
        differentials_count=len(result.differential_diagnoses),
    )

    return result