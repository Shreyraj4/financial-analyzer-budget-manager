"""Checks that every number in a narrated report comes from the facts.

The LLM may only *restate* figures the deterministic code produced. Each
insight/action cites fact ids; every number in its text must appear (within
rounding) in a cited fact. The headline and summary may draw on any fact.
Anything that fails is reported so the caller can retry or drop it.
"""
import re
from dataclasses import dataclass, field

from app.agent.schemas import AgentReportContent

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
MAX_INSIGHTS = 6
MAX_ACTIONS = 5


def extract_numbers(text: str) -> list[float]:
    """Numbers in ``text`` ('₹12,345', '12.5%', '3 months'); calendar years are ignored."""
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group().rstrip(",").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if 1900 <= value <= 2100 and "." not in raw:
            continue
        out.append(value)
    return out


def _matches(x: float, allowed: list[float]) -> bool:
    """Equal up to rounding: half a rupee for big figures, 0.5% or 0.05 for small ones."""
    for v in allowed:
        tol = 0.5 if abs(v) >= 100 else max(0.05, 0.005 * abs(v))
        if abs(x - v) <= tol:
            return True
    return False


def unsupported_numbers(text: str, allowed: list[float]) -> list[float]:
    return [x for x in extract_numbers(text) if not _matches(x, allowed)]


@dataclass
class Issue:
    part: str            # e.g. "insight[2]", "summary"
    problem: str


@dataclass
class VerificationResult:
    issues: list[Issue] = field(default_factory=list)
    checked_numbers: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    def bad_parts(self) -> set[str]:
        return {i.part for i in self.issues}

    def feedback(self) -> str:
        lines = [f"- {i.part}: {i.problem}" for i in self.issues]
        return (
            "Your report failed verification against the facts:\n" + "\n".join(lines) +
            "\nRewrite the whole report. Use only figures that appear verbatim in the cited facts, cite the "
            "right fact ids, and do not compute new figures (no sums, differences or percentages of your own)."
        )


def verify_report(report: AgentReportContent, facts: dict[str, list[float]]) -> VerificationResult:
    """``facts`` maps fact id -> the numbers that fact contains."""
    result = VerificationResult()
    everything = [n for nums in facts.values() for n in nums]

    def check(part: str, text: str, allowed: list[float]) -> None:
        nums = extract_numbers(text)
        result.checked_numbers += len(nums)
        bad = unsupported_numbers(text, allowed)
        if bad:
            result.issues.append(Issue(part, f"numbers not found in the facts: {', '.join(f'{b:g}' for b in bad)}"))

    def allowed_for(part: str, ids: list[str]) -> list[float] | None:
        unknown = [i for i in ids if i not in facts]
        if unknown:
            result.issues.append(Issue(part, f"cites unknown fact ids: {', '.join(unknown)}"))
            return None
        return [n for i in ids for n in facts[i]]

    check("headline", report.headline, everything)
    check("summary", report.summary, everything)
    if len(report.insights) > MAX_INSIGHTS:
        result.issues.append(Issue("insights", f"too many insights ({len(report.insights)}); at most {MAX_INSIGHTS}"))
    if len(report.suggested_actions) > MAX_ACTIONS:
        result.issues.append(Issue("suggested_actions", f"too many actions ({len(report.suggested_actions)}); at most {MAX_ACTIONS}"))
    for i, ins in enumerate(report.insights):
        allowed = allowed_for(f"insight[{i}]", ins.fact_ids)
        if allowed is not None:
            check(f"insight[{i}]", f"{ins.title}. {ins.text}", allowed)
    for i, act in enumerate(report.suggested_actions):
        allowed = allowed_for(f"action[{i}]", act.fact_ids)
        if allowed is not None:
            check(f"action[{i}]", act.text, allowed)
    return result


def drop_failed_parts(report: AgentReportContent, result: VerificationResult) -> tuple[AgentReportContent, bool]:
    """Removes insights/actions that failed verification. Returns (cleaned report, headline_or_summary_failed)."""
    bad = result.bad_parts()
    cleaned = report.model_copy(update={
        "insights": [x for i, x in enumerate(report.insights[:MAX_INSIGHTS]) if f"insight[{i}]" not in bad],
        "suggested_actions": [x for i, x in enumerate(report.suggested_actions[:MAX_ACTIONS]) if f"action[{i}]" not in bad],
    })
    return cleaned, bool(bad & {"headline", "summary"})
