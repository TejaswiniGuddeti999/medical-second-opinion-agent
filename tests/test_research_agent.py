import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.research_agent import run_research_agent
from agents.safety_agent import run_safety_agent
from agents.diagnosis_agent import run_diagnosis_agent
from agents.synthesis_agent import run_synthesis_agent

case = """
18-year-old male presenting with chronic productive cough for 8 months,
night sweats, periodic high fevers, significant weight loss despite good appetite,
and fatigue. No known TB exposure. Currently taking paracetamol 500mg and
Reswas cough syrup.
"""

async def test_research():
    result = await run_research_agent(case)
    print("\n=== RESEARCH AGENT OUTPUT ===")
    print(f"Position: {result.agent_position}\n")
    print(f"Summary: {result.research_summary}\n")
    print(f"Articles found: {len(result.articles)}")
    for a in result.articles:
        print(f"  - [{a.evidence_quality.value}] {a.title[:80]}")
    print(f"Recommended workup: {result.recommended_workup}")
    return result

async def test_safety():
    result = await run_safety_agent(
        case_summary=case,
        medications_text="paracetamol 500mg, Reswas cough syrup"
    )
    print("\n=== SAFETY AGENT OUTPUT ===")
    print(f"Position: {result.agent_position}\n")
    print(f"Safety summary: {result.safety_summary}")
    print(f"Is safe to proceed: {result.is_safe_to_proceed}")
    print(f"Interactions found: {len(result.interactions_found)}")
    for i in result.interactions_found:
        print(f"  - [{i.severity}] {i.drug_name}: {i.description[:80]}")
    return result

async def test_diagnosis():
    result = await run_diagnosis_agent(case)
    print("\n=== DIAGNOSIS AGENT OUTPUT ===")
    print(f"Position: {result.agent_position}\n")
    print(f"Primary: {result.primary_diagnosis.condition} "
          f"({result.primary_diagnosis.confidence:.0%} confidence)\n")
    print("Supporting evidence:")
    for e in result.primary_diagnosis.supporting_evidence:
        print(f"  + {e}")
    print("Against evidence:")
    for e in result.primary_diagnosis.against_evidence:
        print(f"  - {e}")
    print(f"\nDifferentials:")
    for d in result.differential_diagnoses:
        print(f"  {d.condition}: {d.confidence:.0%}")
    print(f"\nReasoning: {result.reasoning[:300]}...")
    return result

async def test_synthesis():
    print("\n--- Running all 3 agents ---")
    research_result  = await run_research_agent(case)
    safety_result    = await run_safety_agent(
        case_summary=case,
        medications_text="paracetamol 500mg, Reswas cough syrup"
    )
    diagnosis_result = await run_diagnosis_agent(case)

    print("\n--- Running SynthesisAgent ---")
    report = await run_synthesis_agent(
        case_summary=case,
        research=research_result,
        safety=safety_result,
        diagnosis=diagnosis_result,
    )

    print("\n=== FINAL REPORT ===")
    print(f"Primary diagnosis : {report.primary_diagnosis}")
    print(f"Confidence        : {report.confidence:.0%} ({report.confidence_level.value})")

    print(f"\nConsensus points:")
    for point in report.consensus_points:
        print(f"  + {point}")

    print(f"\nDisagreements resolved: {len(report.disagreements)}")
    for d in report.disagreements:
        print(f"  Topic     : {d.topic}")
        print(f"  Resolution: {d.resolution}")
        print(f"  Confidence: {d.resolution_confidence.value}\n")

    print(f"Immediate actions:")
    for a in report.immediate_actions:
        print(f"  -> {a}")

    print(f"\nRed flags:")
    for r in report.red_flags:
        print(f"  !! {r}")

    print(f"\nCited sources:")
    for s in report.cited_sources:
        print(f"  [{s}]")

# run only synthesis — it calls all 3 agents internally
asyncio.run(test_synthesis())