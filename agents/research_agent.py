import httpx          # like requests, but async — can do multiple API calls simultaneously
from tenacity import retry, stop_after_attempt, wait_exponential  # retry logic
from app.config import get_settings    # your .env values, loaded once
from app.schemas import (ResearchAgentOutput,
                        PubMedArticle,
                        EvidenceQuality,)            # the Pydantic models you wrote yesterday
from app.logger import get_logger      # structured logging, not print()
from openai import AsyncOpenAI         # async OpenAI client


logger = get_logger(__name__)
settings=get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)


#PubMed API calls
@retry(stop=stop_after_attempt(3), wait = wait_exponential(min=1, max=10))
async def search_pubmed(query: str, max_results: int= 5) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    async with httpx.AsyncClient() as http:
        response = await http.get(
            f"{settings.ncbi_base_url}/esearch.fcgi",
            params = {
                "db":"pubmed",
                "term": query,
                "retmax": max_results,
                "retmode":"json",
                "api_key": settings.ncbi_api_key,
            },
            timeout = 15.0
        )
        response.raise_for_status()
        data = response.json()
        pmids = data["esearchresult"]["idlist"]
        logger.info("pubmed_search_complete", query=query, results_found=len(pmids))
        return pmids
   
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
async def fetch_pubmed_details(pmids: list[str]) -> list[dict]:
    """Fetch article details for a list of PMIDs."""
    if not pmids:
        return []

    import asyncio
    await asyncio.sleep(1)  # PubMed needs a moment between esearch and esummary

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

        # Guard against empty response before parsing
        if not response.content or not response.text.strip():
            logger.warning("pubmed_empty_response", pmids=pmids)
            raise ValueError("PubMed returned empty response")

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
                })

        logger.info("pubmed_fetch_complete", articles_fetched=len(articles))
        return articles

#LLM analysis
async def analyze_articles_with_llm(
    case_summary : str,
    articles: list[dict],
) -> ResearchAgentOutput:
    """
    send case +articles to GPT - 40.
    The agent is prompted to take a STRONG position- not just summarize.
    This is what creates the debate in the orchestrator.
    """

    articles_text = "\n\n".join([
        f"PMID: {a['pmid']}\nTitle: {a['title']}\n"
        f"Journal: {a['journal']}\nYear: {a['year']}\n"
        f"Authors: {', '.join(a['authors'][:3])}"
        for a in articles
    ])

    prompt = f"""You are a ResearchAgent in a multi-agent medical secong opinion system.
Your role: find the strongest evidence-based position for this case.
You MUST take a clear, assertive diagnostic position — not sit on the fence.
Other agents will challenge you. State your position strongly so the debate is meaningful.

PATIENT_CASE:
{case_summary}

PUBMED ARTICLES FOUND:
{articles_text}

Respond in this Exact format:
{{
    "articles" :[
    {{
     "pmid": "string",
      "title": "string",
      "authors": ["string"],
      "journal": "string",
      "year": 2024,
      "relevance_summary": "why this article matters for THIS case specifically",
      "evidence_quality": "strong|moderate|weak"  
    }}
    ],
    "research_summary": "2-3 sentence summary about what the liteature ssays about this case",
    "recommended_workup": ["test1", "test2"],
    "agent_position": "Your strong, assertive position on what this patient likely has and why the evidence supports it. Be specific. Take a stand."
}}

Return only the Json. No preamle, no markdown fences."""

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,   # low temp = consistent, reliable medical reasoning
        response_format={"type": "json_object"}, 
    )
    import json
    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Map raw dicts to typed Pydantic models
    articles_parsed = [
        PubMedArticle(
            pmid=a["pmid"],
            title=a["title"],
            authors=a.get("authors", []),
            journal=a.get("journal", ""),
            year=int(a["year"]) if str(a.get("year", "")).isdigit() else None,
            relevance_summary=a["relevance_summary"],
            evidence_quality=EvidenceQuality(a["evidence_quality"]),
        )
        for a in data.get("articles", [])
    ]
    return ResearchAgentOutput(
        articles=articles_parsed,
        research_summary=data["research_summary"],
        recommended_workup=data.get("recommended_workup", []),
        agent_position=data["agent_position"],
    )

#main agent entry point

async def run_research_agent(case_summary: str) -> ResearchAgentOutput:
    """
    Full pipeline: build query → search PubMed → fetch details → LLM analysis.
    This is what the orchestrator calls.
    """
    logger.info("research_agent_started")

    # Build a focused search query from the case
    query_prompt = f"""Extract a focused PubMed search query (max 8 words) from this case.
Return ONLY the search query, nothing else.

Case: {case_summary}"""

    query_response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": query_prompt}],
        temperature=0.1,
    )
    search_query = query_response.choices[0].message.content.strip()
    logger.info("search_query_built", query=search_query)

    # Search and fetch
    pmids = await search_pubmed(search_query)
    articles = await fetch_pubmed_details(pmids)

    # LLM analysis with strong position
    result = await analyze_articles_with_llm(case_summary, articles)
    logger.info("research_agent_complete", position_length=len(result.agent_position))

    return result
    









