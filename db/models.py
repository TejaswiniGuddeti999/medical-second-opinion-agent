from sqlalchemy import Column, String, Float, DateTime, JSON, Text
from sqlalchemy.sql import func
from db.database import Base


class CaseRecord(Base):
    """
    Stores every patient case submitted to the system.
    JSON columns store the full structured data — flexible, no schema migrations
    needed when we add fields to PatientCase.
    """
    __tablename__ = "cases"

    id = Column(String, primary_key=True)          # the uuid we generate
    symptoms = Column(Text, nullable=False)
    lab_results = Column(Text, nullable=True)
    imaging_notes = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    patient_age = Column(String, nullable=True)
    patient_sex = Column(String, nullable=True)
    clinical_history = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ReportRecord(Base):
    """
    Stores the final report for each case.
    Linked to CaseRecord by case_id.
    full_report stores the entire FinalReport as JSON —
    means we can always reconstruct the full object.
    """
    __tablename__ = "reports"

    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False, index=True)
    primary_diagnosis = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    full_report = Column(JSON, nullable=False)      # entire FinalReport stored here
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())