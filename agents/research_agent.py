import httpx
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings
from app.schemas import (
    ResearchAgentOutput,
    PubMedArticle,
    EvidenceQuality,
)
from app.logger import get_logger
from openai import AsyncOpenAI

logger = get_logger(__name__)
settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)


# ── PubMed API calls ───────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    async with httpx.AsyncClient() as http:
        response = await http.get(
            f"{settings.ncbi_base_url}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "api_key": settings.ncbi_api_key,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        pmids = data["esearchresult"]["idlist"]
        logger.info("pubmed_search_complete", query=query, results_found=len(pmids))
        return pmids


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
async def fetch_pubmed_metadata(pmids: list[str]) -> list[dict]:
    """
    Fetch article metadata (title, authors, journal, year) for a list of PMIDs.
    Uses esummary endpoint — returns structured JSON.
    """
    if not pmids:
        return []

    await asyncio.sleep(1)

    async with httpx.AsyncClient() as http:
        response = await http.get(
            f"{settings.ncbi_base_url}/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
                "api_key": settings.ncbi_api_key,
            },
            timeout=20.0,
        )
        response.raise_for_status()

        if not response.content or not response.text.strip():
            logger.warning("pubmed_metadata_empty", pmids=pmids)
            raise ValueError("PubMed returned empty metadata response")

        data = response.json()
        articles = []

        for pmid in pmids:
            if pmid in data.get("result", {}):
                article = data["result"][pmid]
                articles.append({
                    "pmid": pmid,
                    "title": article.get("title", "No title"),
                    "authors": [
                        a.get("name", "") for a in article.get("authors", [])
                    ],
                    "journal": article.get("fulljournalname", ""),
                    "year": article.get("pubdate", "")[:4],
                    "abstract": "",  # placeholder — filled by fetch_pubmed_abstracts
                })

        logger.info("pubmed_metadata_complete", articles_fetched=len(articles))
        return articles


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
async def fetch_pubmed_abstracts(pmids: list[str]) -> dict[str, str]:
    """
    Fetch actual abstract text for each PMID using PubMed's efetch endpoint.
    Returns a dict of {pmid: abstract_text}.
    This is real paper content — not LLM hallucination.
    """
    if not pmids:
        return {}

    await asyncio.sleep(1)

    async with httpx.AsyncClient() as http:
        response = await http.get(
            f"{settings.ncbi_base_url}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "text",
                "api_key": settings.ncbi_api_key,
            },
            timeout=20.0,
        )
        response.raise_for_status()

        raw_text = response.text
        if not raw_text.strip():
            logger.warning("pubmed_abstracts_empty", pmids=pmids)
            return {}

        # PubMed returns all abstracts concatenated with PMID markers
        abstracts = {}
        current_pmid = None
        current_lines = []

        for line in raw_text.split('\n'):
            for pmid in pmids:
                if pmid in line:
                    if current_pmid and current_lines:
                        abstracts[current_pmid] = ' '.join(current_lines).strip()
                    current_pmid = pmid
                    current_lines = []
                    break
            else:
                if current_pmid:
                    current_lines.append(line.strip())

        # Capture the last abstract
        if current_pmid and current_lines:
            abstracts[current_pmid] = ' '.join(current_lines).strip()

        logger.info("abstracts_fetched", count=len(abstracts))
        return abstracts


async def fetch_pubmed_details(pmids: list[str]) -> list[dict]:
    """
    Orchestrates metadata + abstracts fetching in parallel.
    No @retry here — each individual function handles its own retries.
    """
    if not pmids:
        return []

    # Fire both API calls simultaneously — asyncio.gather runs them in parallel
    metadata, abstracts = await asyncio.gather(
        fetch_pubmed_metadata(pmids),
        fetch_pubmed_abstracts(pmids),
    )

    # Merge abstract text into each article dict
    for article in metadata:
        pmid = article["pmid"]
        article["abstract"] = abstracts.get(pmid, "Abstract not available")

    print("RAW METADATA FROM PUBMED:", metadata[:2])  # show first 2 articles

    return metadata


# ── LLM analysis ───────────────────────────────────────────────────────────────

async def analyze_articles_with_llm(
    case_summary: str,
    articles: list[dict],
) -> ResearchAgentOutput:
    """
    Send case + real abstracts to GPT.
    Agent is prompted to take a STRONG position — creates the debate.
    """

    # If PubMed returned nothing, be honest — don't hallucinate
    if not articles:
        logger.warning("no_articles_to_analyze")
        return ResearchAgentOutput(
            articles=[],
            research_summary="No PubMed articles found for this query.",
            recommended_workup=[],
            agent_position="Insufficient literature evidence found for this case. Clinical judgment from other agents should take precedence.",
        )

    articles_text = "\n\n".join([
        f"PMID: {a['pmid']}\n"
        f"Title: {a['title']}\n"
        f"Journal: {a['journal']} ({a['year']})\n"
        f"Abstract: {a['abstract'][:1000]}"
        for a in articles
    ])

    # List real PMIDs explicitly so LLM cannot use others
    real_pmids = [a["pmid"] for a in articles]
    real_pmids_str = ", ".join(real_pmids)

    prompt = f"""You are a ResearchAgent in a multi-agent medical second opinion system.
Your role: find the strongest evidence-based position for this case.
You MUST take a clear, assertive diagnostic position — not sit on the fence.
Other agents will challenge you. State your position strongly so the debate is meaningful.
Base your position ONLY on the abstracts provided below — do not use outside knowledge.

PATIENT CASE:
{case_summary}

PUBMED ARTICLES FOUND:
{articles_text}

REAL PMIDS (use ONLY these exact PMIDs, never invent new ones): {real_pmids_str}

Respond in this EXACT JSON format:
{{
  "articles": [
    {{
      "pmid": "use only PMIDs from the list above",
      "title": "exact title from the articles above",
      "authors": ["string"],
      "journal": "exact journal from the articles above",
      "year": 2024,
      "relevance_summary": "why this article matters for THIS case specifically",
      "evidence_quality": "strong|moderate|weak"
    }}
  ],
  "research_summary": "2-3 sentence summary of what the literature says about this case",
  "recommended_workup": ["test1", "test2"],
  "agent_position": "Your strong, assertive position based strictly on the abstracts above. Be specific. Take a stand."
}}

Rules:
- You MUST only use PMIDs from this list: {real_pmids_str}
- Do NOT invent or modify PMIDs
- Do NOT use PMIDs like 12345678 or 87654321 — those are fake examples

Return ONLY the JSON. No preamble, no markdown fences."""

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Override LLM output with real PubMed data — never trust LLM for identifiers
    llm_articles = data.get("articles", [])
    articles_parsed = []

    for i, llm_article in enumerate(llm_articles):
        if i < len(articles):
            # Use real data from PubMed for all factual fields
            real = articles[i]
            articles_parsed.append(
                PubMedArticle(
                    pmid=real["pmid"],           # real PMID — never LLM's
                    title=real["title"],         # real title — never LLM's
                    authors=real.get("authors", []),
                    journal=real.get("journal", ""),
                    year=int(real["year"]) if str(real.get("year", "")).isdigit() else None,
                    relevance_summary=llm_article.get("relevance_summary", ""),  # LLM's reasoning is fine
                    evidence_quality=EvidenceQuality(
                        llm_article.get("evidence_quality", "moderate")
                    ),
                )
            )

    return ResearchAgentOutput(
        articles=articles_parsed,
        research_summary=data["research_summary"],
        recommended_workup=data.get("recommended_workup", []),
        agent_position=data["agent_position"],
    )


# ── Main agent entry point ─────────────────────────────────────────────────────

async def run_research_agent(case_summary: str) -> ResearchAgentOutput:
    """
    Full pipeline: build 3 diverse queries → search PubMed → fetch metadata + abstracts → LLM analysis.
    3 queries targeting different diagnostic angles prevents confirmation bias.
    """
    logger.info("research_agent_started")

    # Generate 3 queries targeting DIFFERENT diagnostic possibilities
    query_prompt = f"""You are a medical librarian helping a doctor investigate a complex case.
Generate exactly 3 different PubMed search queries for this case.
Each query MUST target a completely different diagnostic angle or clinical concern.
Think broadly — consider the most dangerous possibilities first, not the most obvious.

Rules:
- Each query must be 3-6 words
- Each query must be meaningfully different from the others
- Consider: primary diagnosis, alternative diagnoses, drug complications, comorbidities
- Use standard medical terminology

Case: {case_summary}

Return ONLY this JSON format:
{{"queries": ["query one", "query two", "query three"]}}"""

    query_response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": query_prompt}],
        temperature=0.4,   # slightly higher — we want diverse queries
        response_format={"type": "json_object"},
    )

    raw_queries = json.loads(query_response.choices[0].message.content)
    queries = raw_queries.get("queries", [])

    # Fallback if parsing fails
    if not queries:
        queries = [case_summary[:50]]

    logger.info("search_queries_built", queries=queries)

    # Search all 3 queries and merge PMIDs
    all_pmids = []
    for query in queries[:3]:
        try:
            pmids = await search_pubmed(query, max_results=3)
            logger.info("query_results", query=query, found=len(pmids))
            all_pmids.extend(pmids)
        except Exception as e:
            logger.warning("query_failed", query=query, error=str(e))
            continue

    # Deduplicate while preserving order
    seen = set()
    unique_pmids = []
    for pmid in all_pmids:
        if pmid not in seen:
            seen.add(pmid)
            unique_pmids.append(pmid)

    logger.info("total_unique_pmids", count=len(unique_pmids))

    # Fetch details only if we have PMIDs
    articles = []
    if unique_pmids:
        articles = await fetch_pubmed_details(unique_pmids[:9])  # max 9 articles
    else:
        logger.warning("pubmed_no_results_all_queries", queries=queries)

    result = await analyze_articles_with_llm(case_summary, articles)
    logger.info(
        "research_agent_complete",
        articles_fetched=len(articles),
        position_length=len(result.agent_position),
    )

    return result