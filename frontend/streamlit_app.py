import streamlit as st #UI framework
import requests # To call FastAPI backend
import json
from datetime import datetime
import os


# ── Config ─────────────────────────────────────────────────────────────────────
# The URL of your FastAPI backend
# When running locally both run on your machine
# When deployed on Railway this becomes your live URL
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

# Page config - This should be the very first Streamlit call
# Streamlit throws error if anything else runs before tis
st.set_page_config(
    page_title="Medical Second Opinion",
    page_icon="",
    layout="wide", # which means app uses full browser instead of narrow centerd column
)


# ── Styling ────────────────────────────────────────────────────────────────────
# Streamlit allows injecting custom CSS.
#These classes get used when displaying confidence levels and disclaimers
st.markdown("""
<style>
    .confidence-high   { color: #0F6E56; font-weight: 500; }
    .confidence-medium { color: #854F0B; font-weight: 500; }
    .confidence-low    { color: #A32D2D; font-weight: 500; }
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .disclaimer {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def confidence_color(level: str) -> str:
    """Return CSS class based on confidence level."""
    mapping = {
        "high": "confidence-high",
        "moderate": "confidence-medium",
        "low": "confidence-low",
    }
    return mapping.get(level.lower(), "")


def display_report(report: dict, processing_time: float = None):
    """
    Takes the FinalReport dict and renders it as a structured UI.
    This is the most important display function.
    """

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # everything indented inside renders inside this column
        st.markdown("### Primary Diagnosis") # smaller heading
        st.markdown(f"## {report['primary_diagnosis']}") # larger one
 
    with col2:
        confidence = report['confidence']
        level = report['confidence_level']
        st.markdown("### Confidence")
        st.markdown(
            f"<span class='{confidence_color(level)}'>"
            f"{confidence:.0%} ({level})</span>",   
            unsafe_allow_html=True
        )

    with col3:
        if processing_time:
            st.markdown("### Analysis time")
            st.markdown(f"**{processing_time:.1f}s**")

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='disclaimer'>{report['disclaimer']}</div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    # ── Immediate actions ──────────────────────────────────────────────────────
    if report.get('immediate_actions'):
        st.markdown("### Immediate actions")
        for action in report['immediate_actions']:
            st.markdown(f"- {action}")

    # ── Red flags ──────────────────────────────────────────────────────────────
    if report.get('red_flags'):
        st.error("**Red flags — seek urgent attention if any of these appear:**")
        for flag in report['red_flags']:
            st.markdown(f"- {flag}")

    st.markdown("---")

    # ── Three columns for agent outputs ───────────────────────────────────────
    # This is where the debate becomes visible to the doctor
    st.markdown("### What each agent found")
    col_r, col_s, col_d = st.columns(3)

    with col_r:
        st.markdown("**Research agent**")
        research = report.get('research_output', {})
        st.caption(research.get('research_summary', ''))
        st.markdown("*Position:*")
        st.info(research.get('agent_position', ''))
        articles = research.get('articles', [])
        if articles:
            with st.expander(f"View {len(articles)} cited articles"):
                for a in articles:
                    st.markdown(
                        f"**{a['title'][:80]}...**  \n"
                        f"*{a['journal']}* ({a.get('year', '')})  \n"
                        f"[PMID: {a['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/)  \n"
                        f"{a['relevance_summary']}"
                    )
                    st.markdown("---")

    with col_s:
        st.markdown("**Safety agent**")
        safety = report.get('safety_output', {})
        st.caption(safety.get('safety_summary', ''))
        st.markdown("*Position:*")
        st.warning(safety.get('agent_position', ''))
        interactions = safety.get('interactions_found', [])
        if interactions:
            with st.expander(f"View {len(interactions)} drug interactions"):
                for i in interactions:
                    severity_color = {
                        "major": "red",
                        "moderate": "orange",
                        "minor": "blue"
                    }.get(i.get('severity', '').lower(), 'gray')
                    st.markdown(
                        f":{severity_color}[**{i['drug_name']}** "
                        f"({i['severity']})]  \n{i['description']}"
                    )

    with col_d:
        st.markdown("**Diagnosis agent**")
        diagnosis = report.get('diagnosis_output', {})
        primary = diagnosis.get('primary_diagnosis', {})
        st.caption(f"Primary: {primary.get('condition', '')} "
                   f"({primary.get('confidence', 0):.0%})")
        st.markdown("*Position:*")
        st.success(diagnosis.get('agent_position', ''))
        differentials = diagnosis.get('differential_diagnoses', [])
        if differentials:
            with st.expander(f"View {len(differentials)} differentials"):
                for d in differentials:
                    st.markdown(
                        f"**{d['condition']}** — {d['confidence']:.0%}  \n"
                        f"For: {', '.join(d.get('supporting_evidence', [])[:2])}  \n"
                        f"Against: {', '.join(d.get('against_evidence', [])[:2])}"
                    )
                    st.markdown("---")

    st.markdown("---")

    # ── Disagreements — the most valuable section ──────────────────────────────
    disagreements = report.get('disagreements', [])
    if disagreements:
        st.markdown("### Where agents disagreed")
        st.caption(
            "This is the most valuable section — "
            "it shows WHERE uncertainty exists and HOW it was resolved."
        )
        for d in disagreements:
            with st.expander(f"Disagreement: {d['topic']}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Research said:**")
                    st.markdown(d['research_view'])
                with c2:
                    st.markdown("**Safety said:**")
                    st.markdown(d['safety_view'])
                with c3:
                    st.markdown("**Diagnosis said:**")
                    st.markdown(d['diagnosis_view'])
                st.markdown("**Resolution:**")
                st.success(d['resolution'])

    # ── Consensus ──────────────────────────────────────────────────────────────
    consensus = report.get('consensus_points', [])
    if consensus:
        st.markdown("### Where all agents agreed")
        for point in consensus:
            st.markdown(f"- {point}")

    # ── Further investigations ─────────────────────────────────────────────────
    investigations = report.get('further_investigations', [])
    if investigations:
        st.markdown("### Further investigations")
        for inv in investigations:
            st.markdown(f"- {inv}")

# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    st.title("Medical Second Opinion Agent")
    st.caption(
        "Multi-agent AI system — ResearchAgent + SafetyAgent + DiagnosisAgent "
        "debate independently, then a SynthesisAgent resolves disagreements."
    )

    # Two tabs — Submit new case | View history
    tab1, tab2 = st.tabs(["New case", "Case history"])

    # ── Tab 1: Submit new case ─────────────────────────────────────────────────
    with tab1:
        st.markdown("### Patient case details")
        st.caption("Fill in as much detail as available. More detail = better analysis.")

        # Two column layout for the form
        col_left, col_right = st.columns(2)

        with col_left:
            symptoms = st.text_area(
                "Symptoms *",
                placeholder="Describe the primary symptoms, onset, and duration...",
                height=120,
            )
            lab_results = st.text_area(
                "Lab results",
                placeholder="CBC, metabolic panel, HbA1c, ESR...",
                height=100,
            )
            imaging_notes = st.text_area(
                "Imaging notes",
                placeholder="X-ray findings, CT reports, ultrasound...",
                height=80,
            )

        with col_right:
            current_medications = st.text_area(
                "Current medications",
                placeholder="Include dosages e.g. paracetamol 500mg twice daily...",
                height=100,
            )
            clinical_history = st.text_area(
                "Clinical history",
                placeholder="Relevant past conditions, surgeries, family history...",
                height=100,
            )
            col_age, col_sex = st.columns(2)
            with col_age:
                patient_age = st.number_input(
                    "Patient age",
                    min_value=0,
                    max_value=130,
                    value=None,
                )
            with col_sex:
                patient_sex = st.selectbox(
                    "Sex",
                    options=["", "male", "female", "other"],
                )

        # Submit button
        submit = st.button(
            "Get second opinion",
            type="primary",
            disabled=not symptoms,   # disabled until symptoms filled
        )

        if submit:
            if not symptoms.strip():
                st.error("Symptoms are required.")
            else:
                # Build the request payload
                payload = {
                    "symptoms": symptoms,
                    "lab_results": lab_results or None,
                    "imaging_notes": imaging_notes or None,
                    "current_medications": current_medications or None,
                    "clinical_history": clinical_history or None,
                    "patient_age": int(patient_age) if patient_age else None,
                    "patient_sex": patient_sex or None,
                }

                # Show spinner while waiting for all 4 agents
                with st.spinner(
                    "Running analysis... "
                    "ResearchAgent searching PubMed, "
                    "SafetyAgent checking FDA database, "
                    "DiagnosisAgent reasoning clinically..."
                ):
                    try:
                        response = requests.post(
                            f"{API_URL}/analyze",
                            json=payload,
                            timeout=120,   # agents can take up to 2 mins
                        )
                        response.raise_for_status()
                        data = response.json()

                        st.success(f"Analysis complete — case ID: {data['case_id']}")

                        if data.get("status") == "complete" and data.get("report"):
                            st.success(f"Analysis complete — case ID: {data['case_id']}")
                            display_report(
                                report=data['report'],
                                processing_time=data.get('processing_time_seconds'),
                            )
                        elif data.get("status") == "rejected":
                            st.error(f"Case rejected: {data.get('error', 'Submission could not be processed.')}")
                        elif data.get("status") == "insufficient_data":
                            st.warning(data.get('error', 'Please provide more detail.'))
                        else:
                            st.error(f"Unexpected response status: {data.get('status')}")


                    except requests.exceptions.Timeout:
                        st.error(
                            "Analysis timed out. "
                            "The agents are taking longer than expected. "
                            "Please try again."
                        )
                    except requests.exceptions.ConnectionError:
                        st.error(
                            "Cannot connect to backend. "
                            "Make sure FastAPI is running on port 8000."
                        )
                    except Exception as e:
                        st.error(f"Something went wrong: {str(e)}")

    # ── Tab 2: Case history ────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Recent cases")

        if st.button("Refresh"):
            st.rerun()

        try:
            response = requests.get(f"{API_URL}/cases", timeout=10)
            response.raise_for_status()
            cases = response.json()

            if not cases:
                st.info("No cases yet. Submit your first case in the New case tab.")
            else:
                for case in cases:
                    report = case.get('report', {})
                    if not report:
                        continue

                    with st.expander(
                        f"{report['primary_diagnosis']} — "
                        f"{report['confidence']:.0%} confidence — "
                        f"Case {case['case_id'][:8]}"
                    ):
                        display_report(
                            report=report,
                            processing_time=case.get('processing_time_seconds'),
                        )

        except requests.exceptions.ConnectionError:
            st.warning("Cannot load case history — backend not reachable.")
        except Exception as e:
            st.error(f"Error loading history: {str(e)}")


if __name__ == "__main__":
    main()