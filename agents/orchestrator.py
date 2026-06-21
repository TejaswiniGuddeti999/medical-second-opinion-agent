import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, END
from app.config import get_settings
from app.schemas import (
    PatientCase,
    ResearchAgentOutput,
    SafetyAgentOutput,
    DiagnosisAgentOutput,
    FinalReport,
)
from agents.research_agent import run_research_agent
from agents.safety_agent import run_safety_agent
from agents.diagnosis_agent import run_diagnosis_agent
from agents.synthesis_agent import run_synthesis_agent
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── State ──────────────────────────────────────────────────────────────────────
# This is the shared box that gets passed between every node in the graph.
# Each agent reads from it and writes its output back into it.
# TypedDict means every key is typed — no silent KeyErrors.

class AgentState(TypedDict):
    case: PatientCase                              # original case — read by all agents
    case_summary: str                              # plain text version for LLM prompts
    research_output: ResearchAgentOutput | None    # written by research node
    safety_output: SafetyAgentOutput | None        # written by safety node
    diagnosis_output: DiagnosisAgentOutput | None  # written by diagnosis node
    final_report: FinalReport | None               # written by synthesis node
    error: str | None                              # captures any agent failure


# ── Node functions ─────────────────────────────────────────────────────────────
# Each node is just an async function that takes state and returns a dict.
# The dict contains only the keys this node updates — LangGraph merges it
# into the full state automatically.

async def research_node(state: AgentState) -> dict:
    """Runs ResearchAgent. Writes research_output into state."""
    try:
        logger.info("research_node_started")
        result = await run_research_agent(state["case_summary"])
        return {"research_output": result}
    except Exception as e:
        logger.error("research_node_failed", error=str(e))
        return {"error": f"ResearchAgent failed: {str(e)}"}


async def safety_node(state: AgentState) -> dict:
    """Runs SafetyAgent. Writes safety_output into state."""
    try:
        logger.info("safety_node_started")
        medications = state["case"].current_medications or ""
        result = await run_safety_agent(
            case_summary=state["case_summary"],
            medications_text=medications,
        )
        return {"safety_output": result}
    except Exception as e:
        logger.error("safety_node_failed", error=str(e))
        return {"error": f"SafetyAgent failed: {str(e)}"}


async def diagnosis_node(state: AgentState) -> dict:
    """Runs DiagnosisAgent. Writes diagnosis_output into state."""
    try:
        logger.info("diagnosis_node_started")
        result = await run_diagnosis_agent(state["case_summary"])
        return {"diagnosis_output": result}
    except Exception as e:
        logger.error("diagnosis_node_failed", error=str(e))
        return {"error": f"DiagnosisAgent failed: {str(e)}"}


async def parallel_agents_node(state: AgentState) -> dict:
    """
    Runs all three agents simultaneously using asyncio.gather.
    This is the key performance node — instead of running agents
    sequentially (3 x 10s = 30s), they run in parallel (10s total).
    
    If one agent fails, the others still complete.
    The error is captured and synthesis handles gracefully.
    """
    logger.info("parallel_agents_started")

    # Fire all three at the same time.Total time is slowest single agent
    results = await asyncio.gather(
        research_node(state),
        safety_node(state),
        diagnosis_node(state),
        return_exceptions=True,   # don't let one failure kill the others
    )

    # Merge all three result dicts into one
    merged = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error("agent_exception", error=str(result))
            merged["error"] = str(result)
        else:
            merged.update(result)

    logger.info(
        "parallel_agents_complete",
        research_done=merged.get("research_output") is not None, #is not None evaluates to true or fase
        safety_done=merged.get("safety_output") is not None,
        diagnosis_done=merged.get("diagnosis_output") is not None,
    )

    return merged


async def synthesis_node(state: AgentState) -> dict:
    """
    Runs SynthesisAgent after all three agents complete.
    Requires all three outputs — if any is missing, returns error.
    """
    try:
        logger.info("synthesis_node_started")

        # Validate all three outputs exist before synthesising
        # all() retuns True if every item in the list is True , if false - not all is True , exception is raised
        if not all([
            state.get("research_output"),
            state.get("safety_output"),
            state.get("diagnosis_output"),
        ]):
            missing = [
                k for k in ["research_output", "safety_output", "diagnosis_output"]
                if not state.get(k)
            ]
            raise ValueError(f"Missing agent outputs: {missing}")

        result = await run_synthesis_agent(
            case_summary=state["case_summary"],
            research=state["research_output"],
            safety=state["safety_output"],
            diagnosis=state["diagnosis_output"],
        )
        return {"final_report": result}

    except Exception as e:
        logger.error("synthesis_node_failed", error=str(e))
        return {"error": f"SynthesisAgent failed: {str(e)}"}


# ── Graph construction ─────────────────────────────────────────────────────────
# This is where LangGraph draws the flowchart.
# Nodes = functions. Edges = connections between them.

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    #registers the function under the name
    graph.add_node("parallel_agents", parallel_agents_node)
    graph.add_node("synthesis",       synthesis_node)

    # Draw edges — parallel_agents always leads to synthesis
    graph.set_entry_point("parallel_agents")
    graph.add_edge("parallel_agents", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


# Build the graph once at import time — not on every request
# This is the production pattern — graph compilation is expensive
medical_graph = build_graph()


# ── Main entry point ───────────────────────────────────────────────────────────

async def run_analysis(case: PatientCase) -> FinalReport:
    """
    This is what FastAPI calls.
    Takes a PatientCase, runs the full graph, returns FinalReport.
    """
    import time
    start = time.time()
    logger.info("analysis_started", symptoms_length=len(case.symptoms))

    # Build the plain text summary for LLM prompts
    case_summary = f"""
Patient: {case.patient_age or 'Unknown'} year old {case.patient_sex or 'unknown sex'}

Symptoms: {case.symptoms}

Lab results: {case.lab_results or 'Not provided'}

Imaging: {case.imaging_notes or 'Not provided'}

Current medications: {case.current_medications or 'None reported'}

Clinical history: {case.clinical_history or 'Not provided'}
""".strip()

    # Initial state
    initial_state: AgentState = {
        "case": case,
        "case_summary": case_summary,
        "research_output": None,
        "safety_output": None,
        "diagnosis_output": None,
        "final_report": None,
        "error": None,
    }

    # Run the graph
    final_state = await medical_graph.ainvoke(initial_state)

    elapsed = time.time() - start
    logger.info("analysis_complete", elapsed_seconds=round(elapsed, 2))

    if final_state.get("error") and not final_state.get("final_report"):
        raise RuntimeError(f"Analysis failed: {final_state['error']}")

    return final_state["final_report"]