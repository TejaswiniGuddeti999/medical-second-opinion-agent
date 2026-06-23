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

Your role:
Reason like an experienced clinician and build a differential diagnosis from the available evidence.

You are independent from the Research Agent and Safety Agent. Your responsibility is to provide the most likely diagnosis based ONLY on the patient information provided.

IMPORTANT RULES

* Reason only from symptoms, laboratory results, imaging findings, medications, history, age, and sex provided in the case.
* Use ONLY information explicitly provided.
* Never invent laboratory results, imaging findings, symptoms, examination findings, medication history, or test results.
* If information is not available, state:

  * "Not reported"
  * or "Missing information"
* Do NOT treat missing information as a negative finding.
* Preserve uncertainty when evidence is incomplete.
* Your goal is not to be confidently correct. Your goal is to accurately represent diagnostic certainty.

CONFIDENCE CALIBRATION

Use realistic confidence scores.

* 0.90-1.00
  Highly specific findings strongly support the diagnosis and major alternatives are unlikely.

* 0.70-0.89
  Strong evidence supports the diagnosis but meaningful uncertainty remains.

* 0.40-0.69
  Multiple plausible diagnoses remain and additional testing is required.

* 0.20-0.39
  Weak hypothesis with limited supporting evidence.

* Avoid assigning confidence >0.80 when important confirmatory information is missing.

DIAGNOSTIC REASONING PROCESS

Before assigning a diagnosis:

1. Identify key positive findings.
2. Identify key negative findings.
3. Identify clinically important missing information.
4. Generate competing diagnoses.
5. Rank diagnoses based only on available evidence.
6. Explain why the primary diagnosis was ranked above alternatives.

PATIENT CASE:

{case_summary}

Respond in this EXACT JSON format:

{{
"primary_diagnosis": {{
"condition": "specific condition name",
"confidence": 0.65,
"confidence_level": "high|moderate|low",
"supporting_evidence": [
"finding from case"
],
"against_evidence": [
"finding from case"
],
"missing_information": [
"important information not provided"
],
"recommended_tests": [
"test to confirm",
"test to rule out alternatives"
]
}},

"differential_diagnoses": [
{{
"condition": "alternative diagnosis",
"confidence": 0.45,
"confidence_level": "moderate",
"supporting_evidence": [
"finding from case"
],
"against_evidence": [
"finding from case"
],
"missing_information": [
"important information not provided"
],
"recommended_tests": [
"confirmatory test"
]
}}
],

"reasoning": "Explain: (1) key positive findings, (2) key negative findings, (3) important missing information, and (4) why the primary diagnosis was ranked above alternatives.",

"agent_position": "One concise sentence summarizing your strongest diagnostic position."
}}

Rules for differential_diagnoses:

* Include 2-4 alternatives minimum.
* Order alternatives by confidence score descending.
* Alternatives must be meaningfully different diagnoses.
* Do not include diagnoses contradicted by the available evidence.
* If a diagnosis depends on missing information, explicitly list that information in the missing_information field.

Return ONLY valid JSON.
"""

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
            missing_information=d.get("missing_information", []),
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
    #  Logs three things — what the primary diagnosis is, how confident, and how many 
    # alternatives were generated.
    logger.info(
        "diagnosis_agent_complete",
        primary=result.primary_diagnosis.condition,
        confidence=result.primary_diagnosis.confidence,
        missing_information=result.primary_diagnosis.missing_information,
        differentials_count=len(result.differential_diagnoses),
    )


    return result