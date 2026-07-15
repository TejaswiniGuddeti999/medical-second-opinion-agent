import time 
import uuid
import re
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas import PatientCase, AnalysisResponse, HealthCheck
from app.config import get_settings
from app.logger import get_logger
from agents.orchestrator import run_analysis
from db.database import get_db
from db import crud
from openai import AsyncOpenAI



logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

moderation_client = AsyncOpenAI(api_key = settings.openai_api_key)


INJECTION_PATTERNS = [
    r"ignore (all |everything|previous|prior|above)",
    r"forget (everything|all|previous)",
    r"disregard (your|the|all)",
    r"you are now",
    r"new instructions",
    r"act as if",
    r"system prompt",
]

HARM_INTENT_PATTERNS = [
    r"how to (kill|harm|hurt|poison|injure)",
    r"how (do i|can i) (kill|harm|hurt|poison)",
    r"(lethal|fatal) dose",
    r"undetectable (poison|method)",
]

def looks_like_injection(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS)

def looks_like_harmful_intent(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in HARM_INTENT_PATTERNS)
    

async def check_moderation(text: str) -> bool:
    """Returns True if flagged by OpenAI's default threshold OR our own stricter violence threshold."""
    try:
        response = await moderation_client.moderations.create(input=text)
        result = response.results[0]
        violence_score = result.category_scores.violence

        logger.info(
            "moderation_result",
            flagged=result.flagged,
            violence_score=violence_score,
        )

        return result.flagged or violence_score > 0.5
    except Exception as e:
        logger.error("moderation_check_failed", error=str(e))
        return False

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Always the first endpoint you build.
    Load balancers, Railway, Docker — all ping this to know if your app is alive.
    Returns 200 OK if the service is running.
    """
    return HealthCheck(status="ok", version=settings.app_version, database = "connected")


@router.post("/analyze", response_model = AnalysisResponse)
async def analyze_case(
    case: PatientCase,
    db: AsyncSession = Depends(get_db),
):
    """
    Main endpoint. Doctor submits a case, gets a structured second opinion.
    
    Depends(get_db) — FastAPI automatically creates a DB session for this
    request and closes it when the request finishes. You never manage
    ssessions manually. This is dependency injection.
    """
    case_id = str(uuid.uuid4()) #generates unique id for every case
    # Require minimum case information before running expensive agents
    if len(case.symptoms.strip()) < 20:
        return AnalysisResponse(
            case_id=case_id,
            status="insufficient_data",
            error="Please provide more detailed symptoms for a meaningful analysis. Minimum 20 characters required.",
        )
    # Check 1 — injection pre-screen (cheap, instant, no API call)
    if looks_like_injection(case.symptoms):
        logger.warning("injection_pattern_detected", case_id=case_id)
        return AnalysisResponse(
            case_id=case_id,
            status="rejected",
            error="The submitted case description does not appear to contain valid clinical information. Please describe actual patient symptoms.",
        )

    # Check 2 — harmful-intent pre-screen (cheap, instant, no API call)
    if looks_like_harmful_intent(case.symptoms):
        logger.warning("harmful_intent_detected", case_id=case_id)
        return AnalysisResponse(
            case_id=case_id,
            status="rejected",
            error="This submission cannot be processed.",
        )

    # Check 3 — Moderation API (costs a network call, runs last since the
    # two checks above are free and catch the most obvious cases first)
    is_flagged = await check_moderation(case.symptoms)
    if is_flagged:
        logger.warning("moderation_flagged", case_id=case_id)
        return AnalysisResponse(
            case_id=case_id,
            status="rejected",
            error="The submitted content could not be processed. Please ensure your submission contains only clinical case information.",
        )
    start = time.time()

    logger.info("analysis_request_received", case_id = case_id)

    try:
        await crud.save_case(db, case_id = case_id , case = case)
        report = await run_analysis(case)

        elapsed = round(time.time() - start, 2)

        # Save the completed report to DB
        await crud.save_report(db, case_id=case_id, report=report, processing_time=elapsed)
        logger.info(
            "analyze_request_complete",
            case_id=case_id,
            elapsed=elapsed,
            diagnosis=report.primary_diagnosis,
        )

        return AnalysisResponse(
            case_id=case_id,
            status="complete",
            report=report,
            processing_time_seconds=elapsed,
        )

    except Exception as e:
        logger.error("analyze_request_failed", case_id=case_id, error=str(e))
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@router.get("/cases", response_model=list[AnalysisResponse],dependencies=[Depends(verify_api_key)])
async def list_cases(
    limit:int = 10,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns recent cases. Doctors can review past second opinions.
    The limit parameter lets the frontend request more or fewer cases.
    """
    cases = await crud.get_recent_cases(db, limit=limit)
    return cases

@router.get("/cases/{case_id}", response_model =AnalysisResponse , dependencies=[Depends(verify_api_key)])  
async def get_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch a single case by ID.
    {case_id} in the URL is a path parameter — FastAPI extracts it automatically.
    """
    case = await crud.get_case(db, case_id= case_id)
    if not case:
        raise HTTPException(status_code = 404, detail = "Case not found")
    return case