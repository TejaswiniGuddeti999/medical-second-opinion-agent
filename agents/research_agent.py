import httpx #async http client like requests
import json #parse json responses
import asyncio # run multiple API calls at the same time (parallel)
from tenacity import retry, stop_after_attempt, wait_exponential #retry library. Automatically retries if an API call fails
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
# retry decorator- If a func throws error , try again upto 3 times , waiting for 1 , 2 and 4 seconds between attempts
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def search_pubmed(query: str, max_results: int = 3) -> list[str]:
    async with httpx.AsyncClient() as http:
        response = await http.get(
            f"{settings.ncbi_base_url}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": f"{query} AND humans[MH] AND English[lang] AND (clinical trial[pt] OR meta-analysis[pt] OR randomized controlled trial[pt] OR systematic review[pt] OR review[pt])",
                "retmax": max_results,
                "retmode" :  "json",
                "api_key": settings.ncbi_api_key,
                "datetype": "pdat",
                "mindate": "2015",
                "maxdate": "2026",
            },
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        print("PUBMED RAW RESPONSE:", json.dumps(data, indent=2))

        # ── ADD THIS BLOCK ──────────────────────────────────────
        if "esearchresult" not in data:
            logger.error(
                "pubmed_unexpected_response",
                query=query,
                response_keys=list(data.keys()),   # shows what keys ARE there
                response_body=str(data)[:500],     # shows the actual error message
            )
            raise ValueError(f"PubMed returned unexpected response: {data}")
        # ────────────────────────────────────────────────────────

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
    
    # creates a http client. Like opening a browser session but it automatically closes and cleans up the connectoin
    async with httpx.AsyncClient() as http:
        # makes a GET request to PubMed's esummary endpoint
        response = await http.get(
            f"{settings.ncbi_base_url}/esummary.fcgi",
            params={
                # tells API which NCBI database to search because it hosts many datavases and pubmed has literature
                "db": "pubmed",
                # Pubmed accepts multiple PMIDs as list of comma seperated valuse
                "id": ",".join(pmids),
                # tells API to return in JOSON format instal of default XML
                "retmode": "json",
                # without key - 3 requests per socond, with key - 10 requests per second
                "api_key": settings.ncbi_api_key,
            },
            # If PubMed doesn't respond within 20 secods, it raises timeout error, which is caught
            # by retry decorator and retires
            timeout=20.0,
        )
        # if http response code is an error, this line throws an exception immediately and continues
        response.raise_for_status()

        # if search results are empty, raises an error which is again coaught by retry logic
        if not response.content or not response.text.strip():
            logger.warning("pubmed_metadata_empty", pmids=pmids)
            raise ValueError("PubMed returned empty metadata response")

        # parse raw response text into a Python dictinary
        data = response.json()
        # Initialize empty list to store articles
        articles = []

        for pmid in pmids:
            # gets result key from response, if it doesnt exist return {}
            # and checks if pmid is present in the article
            if pmid in data.get("result", {}):
                article = data["result"][pmid]
                articles.append({
                    "pmid": pmid,
                    "title": article.get("title", "No title"),
                    "authors": [
                        a.get("name", "") for a in article.get("authors", [])
                    ],
                    #un abbreviated journal name, source is abbreviated
                    "journal": article.get("fulljournalname", ""),
                    "year": article.get("pubdate", "")[:4],
                    "abstract": "",  # placeholder — filled by fetch_pubmed_abstracts
                })

        logger.info("pubmed_metadata_complete", articles_fetched=len(articles))
        return articles


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
async def fetch_pubmed_abstracts(pmids: list[str]) -> dict[str, str]:
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
                "retmode": "xml",          # ← changed from "text"
                "api_key": settings.ncbi_api_key,
            },
            timeout=20.0,
        )
        response.raise_for_status()

        raw_xml = response.text
        if not raw_xml.strip():
            logger.warning("pubmed_abstracts_empty", pmids=pmids)
            return {}

        # Parse XML — no manual line-by-line splitting needed
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw_xml)

        abstracts = {}

        for article in root.findall(".//PubmedArticle"):
            # PMID — clean, unambiguous
            pmid_el = article.find(".//PMID")
            if pmid_el is None:
                continue
            pmid = pmid_el.text.strip()

            # Abstract — may have multiple sections (Background, Methods, etc.)
            abstract_texts = article.findall(".//AbstractText")
            if not abstract_texts:
                abstracts[pmid] = "Abstract not available"
                continue

            # Some abstracts have labeled sections, some don't
            sections = []
            for section in abstract_texts:
                label = section.get("Label")      # e.g. "BACKGROUND", "METHODS"
                text = section.text or ""
                if label:
                    sections.append(f"{label}: {text}")
                else:
                    sections.append(text)

            abstracts[pmid] = " ".join(sections).strip()

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
        # we left a blank placeholder for abstracts in metadata, now we try to replace it.
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
        f"Abstract: {a['abstract'][:2000]}"
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

    
    # Build lookup from LLM response by PMID
    llm_lookup = {
        item.get("pmid"): item
        for item in data.get("articles", [])
    }

    # Iterate over ALL real PubMed articles — not LLM's selection
    articles_parsed = []
    for real in articles:
        pmid = real["pmid"]
        llm_data = llm_lookup.get(pmid, {})  # get LLM's reasoning if it exists, else {}

        articles_parsed.append(
            PubMedArticle(
                pmid=real["pmid"],
                title=real["title"],
                authors=real.get("authors", []),
                journal=real.get("journal", ""),
                year=int(real["year"]) if str(real.get("year", "")).isdigit() else None,
                relevance_summary=llm_data.get("relevance_summary", "No summary provided"),
                evidence_quality=EvidenceQuality(
                    llm_data.get("evidence_quality", "moderate")
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
- Focus on diagnostic criteria and clinical presentation papers
- NOT molecular mechanisms or basic science

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
            pmids = await search_pubmed(query, max_results=5)
            logger.info("query_results", query=query, found=len(pmids))
            all_pmids.extend(pmids)
        except Exception as e:
            logger.error(            # ← was logger.warning
                "query_failed",
                query=query,
                error=str(e),
                exc_info=True        # ← ADD THIS — logs full stack trace
            )
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
        articles = await fetch_pubmed_details(unique_pmids[:15])  # max 9 articles
    else:
        logger.warning("pubmed_no_results_all_queries", queries=queries)

    result = await analyze_articles_with_llm(case_summary, articles)
    logger.info(
        "research_agent_complete",
        articles_fetched=len(articles),
        position_length=len(result.agent_position),
    )

    return result