import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.models import CaseRecord, ReportRecord
from app.schemas import PatientCase, FinalReport, AnalysisResponse
from app.logger import get_logger

logger = get_logger(__name__)


async def save_case(db: AsyncSession, case_id: str, case: PatientCase):
    """Save incoming case to DB before analysis starts."""
    record = CaseRecord(
        id=case_id,
        symptoms=case.symptoms,
        lab_results=case.lab_results,
        imaging_notes=case.imaging_notes,
        current_medications=case.current_medications,
        patient_age=str(case.patient_age) if case.patient_age else None,
        patient_sex=case.patient_sex,
        clinical_history=case.clinical_history,
    )
    db.add(record)
    await db.commit()
    logger.info("case_saved", case_id=case_id)


async def save_report(
    db: AsyncSession,
    case_id: str,
    report: FinalReport,
    processing_time: float = None,
):
    """Save completed report after analysis finishes."""
    record = ReportRecord(
        id=str(uuid.uuid4()),
        case_id=case_id,
        primary_diagnosis=report.primary_diagnosis,
        confidence=report.confidence,
        full_report=report.model_dump(mode="json"),  # convert to JSON-serializable dict
        processing_time=processing_time,
    )
    db.add(record)
    await db.commit()
    logger.info("report_saved", case_id=case_id, diagnosis=report.primary_diagnosis)


async def get_case(db: AsyncSession, case_id: str) -> AnalysisResponse | None:
    """Fetch a single case + its report by case_id."""
    result = await db.execute(
        select(ReportRecord).where(ReportRecord.case_id == case_id)
    )
    record = result.scalar_one_or_none()

    if not record:
        return None

    report = FinalReport(**record.full_report)
    return AnalysisResponse(
        case_id=record.case_id,
        status="complete",
        report=report,
        processing_time_seconds=record.processing_time,
    )


async def get_recent_cases(
    db: AsyncSession,
    limit: int = 10,
) -> list[AnalysisResponse]:
    """Fetch most recent cases ordered by creation time."""
    result = await db.execute(
        select(ReportRecord)
        .order_by(ReportRecord.created_at.desc())
        .limit(limit)
    )
    records = result.scalars().all()

    responses = []
    for record in records:
        report = FinalReport(**record.full_report)
        responses.append(AnalysisResponse(
            case_id=record.case_id,
            status="complete",
            report=report,
            processing_time_seconds=record.processing_time,
        ))

    return responses