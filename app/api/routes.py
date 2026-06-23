import time 
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas import PatientCase, AnalysisResponse, HealthCheck
from app.config import get_settings
from app.logger import get_logger
from agents.orchestrator import run_analysis
from db.database import get_db
from db import crud

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cases", response_model=list[AnalysisResponse])
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

@router.get("/cases/{case_id}", response_model =AnalysisResponse)  
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