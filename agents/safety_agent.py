import httpx
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings
from app.schemas import SafetyAgentOutput, DrugInteraction
from app.logger import get_logger
from openai import AsyncOpenAI

logger = get_logger(__name__)
settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)

# OpenFDA API

@retry(stop = stop_after_attempt(3) , wait = wait_exponential(min=1, max=10))
async def fetch_drug_interactions(medications: list[str]) ->list[dict]:
    """
    Query OpenFDA for each medication.
    OpenFDA needs no API key - This is completely free and open
    """

    if not medications:
        return []

    all_interactions = []

    async with httpx.AsyncClient() as http:
        for drug in medications:
            try:
                response = await http.get(
                    f"{settings.openfda_base_url}/label.json",
                    params={
                        "search": f"openfda.brand_name:;{drug}+OR+openfda.generic_name:{drug}",
                        "limit":1,
                    },
                    timeout=15.0,
                )

                if response.status_code == 404:
                    # drug not found in FDA databse - not an error, just log it
                    logger.warning("drug_not_found_in_fda", drug=drug)
                    continue
                
                response.raise_for_status()
                data = response.json()

                if data.get("results"):
                    result = data['results'][0]
                    all_interactions.append({
                        "drug" : drug,
                        "warnings": result.get("warnings", [""])[0][:500] if result.get("warnings") else "",
                        "contraindications": result.get("contraindications", [""])[0][:500] if result.get("contraindications") else "",
                        "drug_interactions": result.get("drug_interactions", [""])[0][:500] if result.get("drug_interactions") else "",
                        "precautions": result.get("precautions", [""])[0][:500] if result.get("precautions") else "",
                    })
            except httpx.HTTPStatusError as e:
                logger.error("fda_api_error", drug=drug, status = e.response.status_code)
                continue

        logger.info("fda_fetch_complete",drugs_checked=len(medications), data_found = len(all_interactions))
        return all_interactions


def extract_medications(medications_text: str) -> list[str]:
    """
    Parse a free-text medication list into individual drug names.
    Handles: 'lisinopril 10mg, metformin 500mg' → ['lisinopril', 'metformin']
    """
    import re
    # Split on commas, newlines, semicolons
    raw = re.split(r'[,;\n]+', medications_text)
    drugs = []
    for item in raw:
        # Remove dosages like '10mg', '500 mg', 'twice daily'
        clean = re.sub(r'\d+\s*mg.*', '', item, flags=re.IGNORECASE)
        clean = re.sub(r'(once|twice|daily|morning|evening|oral|tablet|capsule)', '', clean, flags=re.IGNORECASE)
        clean = clean.strip()
        if clean and len(clean) > 2:
            drugs.append(clean)
    return drugs


# LLM analysis

async def analyze_safety_with_llm(
    case_summary: str,
    medications: list[str],
    fda_data: list[dict],
) -> SafetyAgentOutput:
    """
    Safety agent takes FDA data + case context and forms  a strong
    safety-based position. It will flag risks the other agetns might miss.
    """

    fda_text = json.dumps(fda_data, indent=2) if fda_data else "No FDA data found for these medications."

    prompt = f"""You are a SafetyAgent in a multi-agent medical second opinion system.
Your role: identify ALL safety concerns, drug interactions, and contraindications.
You are the most cautious agent. You MUST take a strong position on safety risks.
If something seems risky, flag it loudly. Other agents will weigh your warnings.

PATIENT CASE:
{case_summary}

MEDICATIONS: {', '.join(medications) if medications else 'None reported'}

FDA DATABASE DATA:
{fda_text}

Respond in this EXACT JSON format:
{{
  "interactions_found": [
    {{
      "drug_name": "string",
      "interaction_type": "contraindication|warning|caution",
      "severity": "major|moderate|minor",
      "description": "clear explanation of the risk"
    }}
  ],
  "contraindications": ["list of absolute contraindications"],
  "safety_summary": "2-3 sentence summary of the overall safety picture",
  "is_safe_to_proceed": true,
  "agent_position": "Your strong safety-based position. What risks must be addressed before any treatment? Be specific and assertive."
}}

Return ONLY the JSON. No preamble, no markdown fences."""

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # even lower — safety analysis must be consistent
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    interactions = [
        DrugInteraction(
            drug_name=i["drug_name"],
            interaction_type=i["interaction_type"],
            severity=i["severity"],
            description=i["description"],
        )
        for i in data.get("interactions_found", []
        )
    ]

    return SafetyAgentOutput(
        interactions_found=interactions,
        contraindications=data.get("contraindications", []),
        safety_summary=data["safety_summary"],
        is_safe_to_proceed=data["is_safe_to_proceed"],
        agent_position=data["agent_position"],
    )


# Main agent entry point

async def run_safety_agent(case_summary: str, medications_text: str = "") -> SafetyAgentOutput:
    """
    Full pipeline: extract drugs → check FDA → LLM safety analysis.
    If no medications provided, still runs safety analysis on the case itself.
    """
    logger.info("safety_agent_started")

    medications = extract_medications(medications_text) if medications_text else []
    logger.info("medications_extracted", count=len(medications), drugs=medications)

    fda_data = await fetch_drug_interactions(medications)

    result = await analyze_safety_with_llm(case_summary, medications, fda_data)
    logger.info(
        "safety_agent_complete",
        interactions_found=len(result.interactions_found),
        is_safe=result.is_safe_to_proceed,
    )

    return result