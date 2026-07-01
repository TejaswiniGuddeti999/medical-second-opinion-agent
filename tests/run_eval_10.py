"""
run_eval_10.py
 
Automation script for running 10 diverse rheumatology eval cases through the
medical-second-opinion-agent pipeline, computing cosine similarity against
written reference answers, tracking latency, and summarizing results.
 
USAGE:
    Place this file in the project ROOT (same level as agents/, app/, db/).
    Place eval_cases_10.json in the same folder as this script, or update
    EVAL_CASES_PATH below.
 
    Run with:
        python run_eval_10.py
 
    Requires: pip install sentence-transformers --break-system-packages
 
WHAT THIS DOES:
    1. Loads all 10 cases from eval_cases_10.json
    2. Runs each through your orchestrator (agents.orchestrator.run_analysis)
       directly -- same pattern as test_research_agent.py, no HTTP/Streamlit
       needed.
    3. Times each case (wall-clock latency).
    4. Computes cosine similarity between the system's synthesis_reasoning
       (or primary_diagnosis + immediate_actions, as a fallback) and the
       human-written reference_answer for that case.
    5. Saves a full JSON result per case AND a summary CSV/table at the end.
 
WHAT THIS DOES NOT DO:
    - It does NOT judge clinical correctness. That's still your job (or your
      sister's, or a doctor's) -- fill in a "manual_match" column yourself
      afterward, the same way you did for the first 4 cases.
    - It does NOT use a second LLM as a judge (G-Eval). Deliberately skipped
      per the earlier discussion -- too fragile/expensive to build reliably
      right now, given everything found in score_one this session.
 
COST TRACKING NOTE:
    This script does NOT independently sum cost_usd from your structlog
    output, because structlog's PrintLoggerFactory writes to stdout, not
    a Python-accessible object, by default. Instead, this script prints
    a per-case marker (>>> CASE_START: <id> and >>> CASE_END: <id>) to
    stdout BEFORE and AFTER each case runs. If you've redirected stdout to
    a file (e.g. `python run_eval_10.py > eval_run.log 2>&1`), you can
    later isolate each case's token_usage lines by grepping between those
    markers, then sum cost_usd per case the same way you've been doing.
"""
 
import asyncio
import json
import time
import sys
import os
from pathlib import Path
 
# Ensure project root is importable 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
 
from agents.orchestrator import run_analysis
from app.schemas import PatientCase
 
EVAL_CASES_PATH = Path(__file__).parent / "eval_cases_10.json"
OUTPUT_DIR = Path(__file__).parent / "eval_results"
OUTPUT_DIR.mkdir(exist_ok=True)
 
# Load the same model the benchmark paper used
print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
 
 
def compute_cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two pieces of text, same method as the paper."""
    embeddings = embedder.encode([text_a, text_b])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(sim), 4)
 
 
def build_system_output_text(report) -> str:
    """
    Builds a single text blob from the system's report to compare against
    the reference answer. Prefers synthesis_reasoning if present and
    non-empty; falls back to a constructed summary otherwise.
    """
    if getattr(report, "synthesis_reasoning", None):
        return report.synthesis_reasoning
 
    # Fallback: construct from diagnosis + immediate actions + further investigations
    parts = [
        f"Primary diagnosis: {report.primary_diagnosis}",
        f"Confidence: {report.confidence}",
    ]
    if report.immediate_actions:
        parts.append("Immediate actions: " + "; ".join(report.immediate_actions))
    if report.further_investigations:
        parts.append("Further investigations: " + "; ".join(report.further_investigations))
    return " ".join(parts)
 
 
async def run_single_case(case_data: dict) -> dict:
    case_id = case_data["id"]
    print(f"\n>>> CASE_START: {case_id}", flush=True)
 
    patient_case = PatientCase(
        symptoms=case_data["case"]["symptoms"],
        lab_results=case_data["case"].get("lab_results"),
        current_medications=case_data["case"].get("current_medications"),
        patient_age=case_data["case"].get("patient_age"),
        patient_sex=case_data["case"].get("patient_sex"),
        clinical_history=case_data["case"].get("clinical_history"),
    )
 
    start = time.time()
    error = None
    report = None
    try:
        report = await run_analysis(patient_case)
    except Exception as e:
        error = str(e)
    elapsed = round(time.time() - start, 2)
 
    result = {
        "case_id": case_id,
        "difficulty": case_data.get("difficulty"),
        "expected_primary": case_data.get("expected_primary"),
        "acceptable": case_data.get("acceptable"),
        "elapsed_seconds": elapsed,
        "error": error,
    }
 
    if report:
        system_text = build_system_output_text(report)
        reference_text = case_data.get("reference_answer", "")
 
        cosine_score = (
            compute_cosine_similarity(system_text, reference_text)
            if reference_text else None
        )
 
        result.update({
            "system_diagnosis": report.primary_diagnosis,
            "system_confidence": report.confidence,
            "cosine_similarity": cosine_score,
            "immediate_actions": report.immediate_actions,
            "differentials": [
                {"condition": d.condition, "confidence": d.confidence}
                for d in report.diagnosis_output.differential_diagnoses
            ] if report.diagnosis_output else [],
            "cited_pmids": [a.pmid for a in report.research_output.articles] if report.research_output else [],
            "manual_match": None,  # <-- fill this in yourself: "Yes" / "Partial" / "No"
            "manual_notes": "",     # <-- your notes after review
        })
    else:
        result.update({
            "system_diagnosis": None,
            "system_confidence": None,
            "cosine_similarity": None,
            "manual_match": "No",
            "manual_notes": f"Case crashed: {error}",
        })
 
    print(f">>> CASE_END: {case_id} (elapsed={elapsed}s, error={error})", flush=True)
    return result
 
 
async def main():
    with open(EVAL_CASES_PATH) as f:
        cases = json.load(f)
 
    print(f"Loaded {len(cases)} cases. Running sequentially (not parallel, to keep logs separable)...\n")
 
    all_results = []
    for case_data in cases:
        result = await run_single_case(case_data)
        all_results.append(result)
 
        # Save individual result immediately, so a crash mid-run doesn't lose everything
        out_path = OUTPUT_DIR / f"{result['case_id']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
 
    # Save combined summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
 
    # Print a clean table to console
    print("\n" + "=" * 100)
    print(f"{'Case ID':<28} {'Diagnosis':<35} {'Cosine':<8} {'Latency(s)':<10} {'Error'}")
    print("=" * 100)
    for r in all_results:
        diag = (r.get("system_diagnosis") or "FAILED")[:33]
        cosine = r.get("cosine_similarity")
        cosine_str = f"{cosine:.3f}" if cosine is not None else "N/A"
        print(f"{r['case_id']:<28} {diag:<35} {cosine_str:<8} {r['elapsed_seconds']:<10} {r.get('error') or ''}")
 
    valid_cosines = [r["cosine_similarity"] for r in all_results if r.get("cosine_similarity") is not None]
    valid_latencies = [r["elapsed_seconds"] for r in all_results]
 
    print("=" * 100)
    if valid_cosines:
        print(f"Mean cosine similarity: {np.mean(valid_cosines):.4f} (std: {np.std(valid_cosines):.4f})")
    print(f"Mean latency: {np.mean(valid_latencies):.2f}s (min: {min(valid_latencies)}s, max: {max(valid_latencies)}s)")
    print(f"\nFull results saved to: {summary_path}")
    print("Per-case files saved to:", OUTPUT_DIR)
    print("\nNEXT STEP: open summary.json and fill in 'manual_match' and 'manual_notes' for each case.")
 
 
if __name__ == "__main__":
    asyncio.run(main())
 
