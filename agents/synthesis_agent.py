import json
from openai import AsyncOpenAI
from app.config import get_settings
from app.schemas import (
    FinalReport,
    ResearchAgentOutput,
    SafetyAgentOutput,
    DiagnosisAgentOutput,
    AgentDisagreement,
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


async def run_synthesis_agent(
    case_summary: str,
    research: ResearchAgentOutput,
    safety: SafetyAgentOutput,
    diagnosis: DiagnosisAgentOutput,
) -> FinalReport:
    """
    The senior consultant in the room.
    Receives all three agent outputs, finds disagreements,
    weighs evidence quality, and produces the final structured report.
    """
    logger.info("synthesis_agent_started")

    # Build a structured summary of what each agent said
    # This is what gets passed to the LLM — clean, organised, comparable
    agents_summary = f"""
RESEARCH AGENT POSITION:
{research.agent_position}

Evidence found: {len(research.articles)} PubMed articles
Research summary: {research.research_summary}
Recommended workup: {', '.join(research.recommended_workup)}

---

SAFETY AGENT POSITION:
{safety.agent_position}

Is safe to proceed: {safety.is_safe_to_proceed}
Interactions found: {len(safety.interactions_found)}
Safety summary: {safety.safety_summary}
Contraindications: {', '.join(safety.contraindications) if safety.contraindications else 'None identified'}
Drug interactions: {'; '.join([f"{i.drug_name} ({i.severity}): {i.description}" for i in safety.interactions_found]) if safety.interactions_found else 'None'}

---

DIAGNOSIS AGENT POSITION:
{diagnosis.agent_position}

Primary diagnosis: {diagnosis.primary_diagnosis.condition} ({diagnosis.primary_diagnosis.confidence:.0%} confidence)
Supporting evidence: {', '.join(diagnosis.primary_diagnosis.supporting_evidence)}
Against evidence: {', '.join(diagnosis.primary_diagnosis.against_evidence)}
Clinical reasoning: {diagnosis.reasoning}

Differential diagnoses:
{chr(10).join([f"  - {d.condition}: {d.confidence:.0%}" for d in diagnosis.differential_diagnoses])}
"""

    prompt = f"""You are a SynthesisAgent — the senior consultant in a multi-agent medical second opinion system.
You have received independent assessments from three specialist agents.
Your job is NOT to pick a winner. Your job is to reason carefully about agreements and disagreements,
weigh the quality of evidence behind each position, and produce a final structured second opinion.

ORIGINAL PATIENT CASE:
{case_summary}

THREE AGENT ASSESSMENTS:
{agents_summary}

Your task:
1. Identify where agents AGREE — these are high-confidence findings
2. Identify where agents DISAGREE — explain WHY they disagree and resolve it
3. Weigh evidence: PubMed citations > clinical reasoning > safety flags
4. Produce a final diagnosis with honest confidence score

Respond in this EXACT JSON format:
{{
  "primary_diagnosis": "final diagnosis condition name",
  "confidence": 0.75,
  "immediate_actions": [
    "specific action the doctor should take immediately",
    "another immediate action"
  ],
  "further_investigations": [
    "test to confirm diagnosis",
    "test to rule out alternatives"
  ],
  "red_flags": [
    "warning signs that would change the diagnosis urgently"
  ],
  "disagreements": [
    {{
      "topic": "what the agents disagreed about",
      "research_view": "what ResearchAgent said",
      "safety_view": "what SafetyAgent said",
      "diagnosis_view": "what DiagnosisAgent said",
      "resolution": "your reasoned resolution of this disagreement",
      "resolution_confidence": "high|moderate|low"
    }}
  ],
  "consensus_points": [
    "something all three agents agreed on",
    "another point of consensus"
  ],
  "cited_sources": [
    "PMID: 12345678 — brief description of what this paper contributes"
  ],
  "synthesis_reasoning": "Your full reasoning. Walk through how you weighed the three positions. Be specific about why you resolved disagreements the way you did. This is the most important field."
}}

Rules:
- confidence must reflect genuine uncertainty — do not give 0.95 unless truly warranted
- red_flags should be specific and actionable
- If safety agent flagged drug interactions, they MUST appear in immediate_actions
- cited_sources should only include PMIDs that actually appeared in research agent output

Return ONLY the JSON. No preamble, no markdown fences."""

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,   # lowest temp — final report must be consistent
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Parse disagreements
    disagreements = [
        AgentDisagreement(
            topic=d["topic"],
            research_view=d["research_view"],
            safety_view=d["safety_view"],
            diagnosis_view=d["diagnosis_view"],
            resolution=d["resolution"],
            resolution_confidence=ConfidenceLevel(d["resolution_confidence"]),
        )
        for d in data.get("disagreements", [])
    ]

    confidence = float(data.get("confidence", 0.5))

    report = FinalReport(
        primary_diagnosis=data["primary_diagnosis"],
        confidence=confidence,
        confidence_level=confidence_to_level(confidence),
        research_output=research,
        safety_output=safety,
        diagnosis_output=diagnosis,
        disagreements=disagreements,
        consensus_points=data.get("consensus_points", []),
        immediate_actions=data.get("immediate_actions", []),
        further_investigations=data.get("further_investigations", []),
        red_flags=data.get("red_flags", []),
        cited_sources=data.get("cited_sources", []),
    )

    logger.info(
        "synthesis_agent_complete",
        primary_diagnosis=report.primary_diagnosis,
        confidence=report.confidence,
        disagreements_found=len(report.disagreements),
        consensus_points=len(report.consensus_points),
    )

    return report