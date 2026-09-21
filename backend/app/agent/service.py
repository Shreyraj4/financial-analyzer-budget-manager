"""Generates and stores an explanation report for a user.

Flow: facts (code) -> narrator (Claude or template) -> number verification
(retry once with feedback) -> drop what still fails -> store in agent_reports.
Any Claude failure falls back to the template narrator, never to a 500.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.facts import FactSet, build_facts
from app.agent.narrators import ClaudeNarrator, NarrationResult, TemplateNarrator
from app.agent.openrouter import OpenRouterClient, OpenRouterNarrator
from app.agent.schemas import AgentReportContent
from app.agent.verify import drop_failed_parts, verify_report
from app.config import get_settings
from app.models import AgentReport

logger = logging.getLogger(__name__)


def narrator_choice(settings=None) -> tuple[str, str | None]:
    """Which writer will be used, as (kind, model): "claude" | "openrouter" | "template"."""
    s = settings or get_settings()
    mode = s.report_narrator
    if mode == "template":
        return "template", None
    if mode in ("auto", "claude") and s.anthropic_api_key:
        return "claude", s.anthropic_model
    if mode in ("auto", "openrouter") and s.openrouter_api_key:
        return "openrouter", s.openrouter_model
    return "template", None


def get_narrator():
    settings = get_settings()
    kind, model = narrator_choice(settings)
    if kind == "claude":
        import anthropic  # imported lazily: the app runs without this SDK or a key

        return ClaudeNarrator(anthropic.Anthropic(api_key=settings.anthropic_api_key), model)
    if kind == "openrouter":
        return OpenRouterNarrator(OpenRouterClient(settings.openrouter_api_key, settings.openrouter_base_url), model)
    return TemplateNarrator()


def narrate_verified(facts: FactSet, narrator) -> tuple[NarrationResult, dict]:
    """Runs a narrator, verifies every number, retries once, drops unverifiable parts."""
    template = TemplateNarrator()
    info: dict = {"attempts": 0, "fell_back": False, "error": None, "dropped_parts": []}
    ids = facts.numbers_by_id()

    try:
        result = narrator.narrate(facts)
        info["attempts"] = 1
        verdict = verify_report(result.content, ids)
        if not verdict.ok and getattr(narrator, "supports_feedback", False):
            info["first_attempt_issues"] = [f"{i.part}: {i.problem}" for i in verdict.issues]
            result = narrator.narrate(facts, previous=result.content, feedback=verdict.feedback())
            info["attempts"] = 2
            verdict = verify_report(result.content, ids)
    except Exception as exc:  # anthropic.APIError, NarrationError, network problems ...
        logger.exception("Narrator %s failed; using the template report", narrator.name)
        info.update(fell_back=True, error=f"{type(exc).__name__}: {exc}"[:300])
        result = template.narrate(facts)
        verdict = verify_report(result.content, ids)

    cleaned, text_failed = drop_failed_parts(result.content, verdict)
    if not verdict.ok:
        info["dropped_parts"] = sorted(verdict.bad_parts())
    if text_failed:
        # A headline/summary with unverifiable figures is replaced by the (always correct) template wording.
        base = template.narrate(facts).content
        cleaned = cleaned.model_copy(update={"headline": base.headline, "summary": base.summary})
        info["headline_summary_replaced"] = True
    info["verified"] = True
    info["numbers_checked"] = verify_report(cleaned, ids).checked_numbers
    result.content = cleaned
    return result, info


def generate_report(db: Session, user_id: int, narrator=None) -> AgentReport | None:
    facts = build_facts(db, user_id)
    if facts is None:
        return None
    narrator = narrator or get_narrator()
    result, verification = narrate_verified(facts, narrator)

    used = "template" if verification["fell_back"] else result.narrator
    report = AgentReport(
        user_id=user_id,
        period_start=facts.period_start,
        period_end=facts.period_end,
        summary=result.content.headline,
        structured_report={
            "content": result.content.model_dump(),
            "facts": [{"id": f.id, "statement": f.statement} for f in facts.facts],
            "narrator": {"used": used, "model": result.model, "usage": result.usage},
            "verification": verification,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def latest_report(db: Session, user_id: int) -> AgentReport | None:
    stmt = select(AgentReport).where(AgentReport.user_id == user_id).order_by(AgentReport.id.desc()).limit(1)
    return db.scalars(stmt).first()


def report_content(report: AgentReport) -> AgentReportContent:
    return AgentReportContent.model_validate(report.structured_report["content"])
