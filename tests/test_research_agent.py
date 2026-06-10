import asyncio
from agents.research_agent import run_research_agent

case = """
A 17 year old female suffering with excessive , productive cough, weezing sounds
in lungs, weight loss, body pains and increse in temperature every single night.
She has good appetite but frequently feels fatigue and episodes of high fever periodically from the past 8 months.
What could the problem be?
"""

async def main():
    result = await run_research_agent(case)
    print("\n=== RESEARCH AGENT OUTPUT ===")
    print(f"Position: {result.agent_position}\n")
    print(f"Summary: {result.research_summary}\n")
    print(f"Articles found: {len(result.articles)}")
    for a in result.articles:
        print(f"  - [{a.evidence_quality.value}] {a.title[:80]}...")
    print(f"Recommended workup: {result.recommended_workup}")

asyncio.run(main())