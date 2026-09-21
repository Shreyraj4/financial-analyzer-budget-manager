"""The structured report the narrator must produce (also the LLM's output schema).

Kept free of length/range constraints on purpose: structured-output schemas
support only a subset of JSON Schema, so limits are enforced in verify.py.
"""
from typing import Literal

from pydantic import BaseModel


class Insight(BaseModel):
    title: str
    text: str
    severity: Literal["info", "positive", "watch", "alert"]
    fact_ids: list[str]


class SuggestedAction(BaseModel):
    text: str
    fact_ids: list[str]


class AgentReportContent(BaseModel):
    headline: str
    summary: str
    insights: list[Insight]
    suggested_actions: list[SuggestedAction]
