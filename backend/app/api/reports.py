from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.schemas import AgentReportContent
from app.agent.service import generate_report, latest_report, narrator_choice
from app.api.deps import get_current_user
from app.database.session import get_db
from app.models import AgentReport, User

router = APIRouter(prefix="/reports", tags=["reports"])


class FactOut(BaseModel):
    id: str
    statement: str


class ReportOut(BaseModel):
    id: int
    created_at: datetime
    period_start: date
    period_end: date
    content: AgentReportContent
    facts: list[FactOut]        # what the report is allowed to say; lets the UI show sources
    narrator: dict              # {"used": "claude"|"template", "model", "usage"}
    verification: dict          # number-check details: attempts, dropped parts, fell_back, error


class ReportSummary(BaseModel):
    id: int
    created_at: datetime
    period_end: date
    headline: str


class NarratorStatus(BaseModel):
    narrator: str   # "claude" | "openrouter" | "template"
    model: str | None


def _to_out(r: AgentReport) -> ReportOut:
    s = r.structured_report
    return ReportOut(
        id=r.id, created_at=r.created_at, period_start=r.period_start, period_end=r.period_end,
        content=s["content"], facts=s["facts"], narrator=s["narrator"], verification=s["verification"],
    )


@router.get("/status", response_model=NarratorStatus)
def narrator_status(current_user: User = Depends(get_current_user)) -> NarratorStatus:
    """Which writer will produce the next report: Claude, OpenRouter, or the built-in template."""
    kind, model = narrator_choice()
    return NarratorStatus(narrator=kind, model=model)


@router.post("/generate", response_model=ReportOut, status_code=status.HTTP_201_CREATED)
def generate(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ReportOut:
    """Builds facts from the ML outputs, has the narrator explain them, verifies every number, and stores the report.
    With an API key this makes a paid Claude call, so it is only triggered explicitly (never on page load)."""
    report = generate_report(db, current_user.id)
    if report is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No spending data to analyze yet. Import transactions first.")
    return _to_out(report)


@router.get("/latest", response_model=ReportOut | None)
def latest(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ReportOut | None:
    report = latest_report(db, current_user.id)
    return _to_out(report) if report else None


@router.get("", response_model=list[ReportSummary])
def list_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ReportSummary]:
    stmt = select(AgentReport).where(AgentReport.user_id == current_user.id).order_by(AgentReport.id.desc()).limit(limit)
    return [ReportSummary(id=r.id, created_at=r.created_at, period_end=r.period_end, headline=r.summary) for r in db.scalars(stmt)]
