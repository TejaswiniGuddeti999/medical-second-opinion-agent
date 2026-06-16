# 🧬 Medical Second Opinion Agent

> An AI-powered multi-agent system that helps surface evidence-based insights for complex, hard-to-diagnose medical conditions — using peer-reviewed literature and FDA drug safety data.

---

## The Problem

Medical misdiagnosis is one of the most underreported crises in healthcare. Roughly **12 million patients are misdiagnosed annually** in the US alone, with nearly half of those errors causing serious harm. The problem is worst for complex conditions with overlapping symptoms — autoimmune diseases in particular.

Diseases like lupus, rheumatoid arthritis, Sjögren's syndrome, and multiple sclerosis share nearly identical presentations: fatigue, joint pain, inflammation, and organ dysfunction. On average, autoimmune patients see **3–4 doctors over 4+ years** before receiving an accurate diagnosis.

The bottleneck isn't physician skill — it's **information overload**. A patient arrives with years of history, multiple lab panels, imaging results, and a medication list. Cross-referencing all of that against current clinical literature, while accounting for drug interactions and comorbidities, is simply more than any one clinician can do in a standard appointment.

This project attempts to close that gap.

---

## How It Works

The problem is decomposed into two root causes of diagnostic failure:

### 1. Symptom + History + Lab Complexity
Patients present with a mix of subjective symptoms, historical events, and objective lab values. No single data point is conclusive — but together, they can point to a differential diagnosis if analyzed against up-to-date medical research.

### 2. Medication Confounding
Current medications can suppress, mimic, or exacerbate symptoms. Drug-induced lupus, steroid-masked inflammation, NSAID-altered lab values — a diagnosis made without accounting for the patient's medication profile risks being built on distorted signals.

---

## Architecture: Multi-Agent Pipeline

```
Patient Input (symptoms + history + labs + medications)
        │
        ▼
 ┌─────────────────┐
 │  Query Rewriter  │  →  Rewrites input into 3 targeted PubMed search queries
 └────────┬────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Research Agent    │     │    Safety Agent      │
│  (PubMed via        │     │  (OpenFDA API)       │
│   Entrez API)       │     │                      │
│                     │     │  Checks if current   │
│  Fetches 10 top     │     │  medications are     │
│  articles per query │     │  affecting symptoms  │
│  (abstracts, MeSH,  │     │  or flagged for      │
│  metadata)          │     │  adverse events      │
└──────────┬──────────┘     └──────────┬───────────┘
           │                           │
           └────────────┬──────────────┘
                        ▼
              ┌──────────────────┐
              │  Diagnose Agent  │  →  Pure LLM reasoning over structured patient data
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Synthesis Agent  │  →  Aggregates all signals → generates final report
              └──────────────────┘
```

---

## Data Sources

### PubMed
Maintained by the **National Library of Medicine (NLM)**, a division of the U.S. **National Institutes of Health (NIH)**. PubMed indexes over **36 million citations** from biomedical and life sciences literature and is the gold standard for peer-reviewed clinical research. Accessed via the free Entrez API.

### OpenFDA
A public API built and maintained by the **U.S. Food and Drug Administration (FDA)**. Provides structured access to drug adverse event reports, drug labeling data, recalls, and pharmacological information — all sourced from official FDA submissions.

---

## Tech Stack

- **Backend**: Python, FastAPI
- **Agents**: Custom multi-agent orchestration
- **APIs**: NCBI Entrez (PubMed), OpenFDA
- **Database**: PostgreSQL (via Docker)
- **Frontend**: React
- **Infra**: Docker Compose

---

## Project Structure

```
medical-second-opinion-agent/
├── agents/          # Research, Safety, Diagnose, Synthesis agents
├── app/             # FastAPI backend
├── db/              # Database models and migrations
├── frontend/        # React UI
├── tests/           # Test suite
├── docker-compose.yml
└── requirements.txt
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/TejaswiniGuddeti999/medical-second-opinion-agent.git
cd medical-second-opinion-agent

# Start all services
docker-compose up --build
```

---

## Known Limitations

| Limitation | Details |
|---|---|
| Article relevance | 10 abstracts per query is a starting point — not all may be clinically relevant to the specific case |
| Abstract-only extraction | Full paper content is not yet parsed; only abstracts, MeSH terms, and metadata are used |
| No longitudinal tracking | Each query is stateless; symptom progression over time is not tracked |
| Unstructured lab inputs | Lab values are entered as free text; no HL7/FHIR structured data support yet |
| No confidence scoring | The synthesis report doesn't indicate how many sources agreed vs. contradicted a hypothesis |

---

## Roadmap

- [ ] **RAG pipeline** — PDF retrieval → chunking → embedding → vector storage → reranking for full-text PubMed analysis
- [ ] **FHIR/EHR integration** — Structured ingestion of lab results via standard health data formats
- [ ] **Confidence scoring** — Evidence weighting in the synthesis report (how many sources support each hypothesis)
- [ ] **Drug-drug interaction analysis** — Expanded OpenFDA coverage beyond individual drug safety
- [ ] **Longitudinal patient sessions** — Track symptom changes across multiple queries

---

## Disclaimer

This tool is intended to assist clinical reasoning and surface relevant medical literature — it is **not a substitute for professional medical diagnosis or advice**. All outputs should be reviewed by a qualified healthcare provider.

---

