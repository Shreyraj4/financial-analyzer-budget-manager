import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from app.agent.facts import build_facts_from_parts
from app.agent.narrators import SYSTEM_PROMPT, NarrationError
from app.agent.openrouter import (
    JSON_FORMAT,
    OpenRouterClient,
    OpenRouterError,
    OpenRouterNarrator,
    list_free_models,
    parse_report,
)
from app.agent.service import narrate_verified
from app.agent.verify import verify_report

VALID = {
    "headline": "Total spending in the latest month (Sep 2026) was ₹29,000.",
    "summary": "The data covers 3 months of spending, up to Sep 2026.",
    "insights": [{"title": "Big category", "text": "Housing was spending category number 1 in Sep 2026.", "severity": "info", "fact_ids": ["spend.top_1"]}],
    "suggested_actions": [],
}


def _facts():
    months = pd.date_range("2026-07-01", periods=3, freq="MS")
    rows = [{"user_id": 1, "category": c, "month": m, "spend": v} for m, (a, b) in zip(months, [(5000.0, 20000.0), (6000.0, 20000.0), (9000.0, 20000.0)])
            for c, v in (("Food & Dining", a), ("Housing", b))]
    return build_facts_from_parts(pd.DataFrame(rows), None, None, [], date(2026, 9, 28))


class FakeTransport:
    """Scripted stand-in for the network: records requests, returns (status, body) pairs."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": headers, "body": json.loads(body) if body else None})
        status, payload = self.replies.pop(0)
        return status, json.dumps(payload).encode()


def _ok(text, model="vendor/free-model:free", prompt=100, completion=50):
    return 200, {"model": model, "choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def _narrator(*replies):
    transport = FakeTransport(*replies)
    return OpenRouterNarrator(OpenRouterClient("sk-or-secret", transport=transport), "vendor/free-model:free"), transport


# ---- request shape -------------------------------------------------------------------------
def test_request_goes_to_chat_completions_with_key_only_in_the_auth_header():
    narrator, transport = _narrator(_ok(json.dumps(VALID)))
    narrator.narrate(_facts())
    req = transport.requests[0]
    assert req["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-or-secret"
    assert "sk-or-secret" not in json.dumps(req["body"])
    body = req["body"]
    assert body["model"] == "vendor/free-model:free" and body["max_tokens"] == 4000
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT + JSON_FORMAT
    assert "spend.latest_total" in body["messages"][1]["content"]     # facts sent as a data payload


def test_result_carries_model_and_token_usage():
    narrator, _ = _narrator(_ok(json.dumps(VALID), model="meta/whatever", prompt=1200, completion=340))
    result = narrator.narrate(_facts())
    assert result.narrator == "openrouter" and result.model == "meta/whatever"
    assert result.usage == {"input_tokens": 1200, "output_tokens": 340}


# ---- tolerant parsing of weaker models' output ----------------------------------------------------
def test_parse_report_accepts_fences_prose_and_reasoning_blocks():
    raw = "<think>let me plan</think>Sure! Here is the report:\n```json\n" + json.dumps(VALID) + "\n```\nHope that helps."
    assert parse_report(raw).headline == VALID["headline"]


def test_parse_report_normalizes_common_slips():
    slip = {**VALID, "insights": [{"title": "t", "text": "x", "severity": "URGENT", "fact_ids": "spend.top_1"}]}
    del slip["suggested_actions"]
    report = parse_report(json.dumps(slip))
    assert report.insights[0].severity == "info" and report.insights[0].fact_ids == ["spend.top_1"] and report.suggested_actions == []


@pytest.mark.parametrize("raw, message", [("no json here", "no JSON object"), ("{not: valid}", "not valid JSON"), ('{"headline": "x"}', "required shape")])
def test_parse_report_rejects_unusable_output_with_a_message_for_the_model(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_report(raw)


def test_formatting_slip_gets_one_repair_round_and_tokens_are_summed():
    narrator, transport = _narrator(_ok("Sorry, here you go: nothing", prompt=100, completion=10), _ok(json.dumps(VALID), prompt=150, completion=60))
    result = narrator.narrate(_facts())
    assert result.content.headline == VALID["headline"] and len(transport.requests) == 2
    repair = transport.requests[1]["body"]["messages"]
    assert [m["role"] for m in repair] == ["system", "user", "assistant", "user"] and "could not be used" in repair[3]["content"]
    assert result.usage == {"input_tokens": 250, "output_tokens": 70}


def test_two_unusable_replies_raise_narration_error():
    narrator, _ = _narrator(_ok("nope"), _ok("still nope"))
    with pytest.raises(NarrationError, match="usable report"):
        narrator.narrate(_facts())


def test_feedback_from_verification_is_sent_back_with_the_previous_answer():
    narrator, transport = _narrator(_ok(json.dumps(VALID)))
    facts = _facts()
    previous = parse_report(json.dumps(VALID))
    narrator.narrate(facts, previous=previous, feedback="fix numbers")
    msgs = transport.requests[0]["body"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"] and msgs[3]["content"] == "fix numbers"


# ---- HTTP errors --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "status, fragment",
    [(401, "rejected the API key"), (402, "insufficient credits"), (404, "not found"), (429, "rate limited"), (503, "problem")],
)
def test_http_errors_become_actionable_messages(status, fragment):
    client = OpenRouterClient("k", transport=FakeTransport((status, {"error": {"message": "provider said no"}})))
    with pytest.raises(OpenRouterError, match=fragment) as exc:
        client.chat("m", [{"role": "user", "content": "hi"}])
    assert exc.value.status == status and "provider said no" in str(exc.value)


def test_error_body_with_http_200_is_still_an_error():
    client = OpenRouterClient("k", transport=FakeTransport((200, {"error": {"message": "upstream exploded"}})))
    with pytest.raises(OpenRouterError, match="upstream exploded"):
        client.chat("m", [])


def test_missing_message_content_is_a_narration_error():
    narrator, _ = _narrator((200, {"choices": []}))
    with pytest.raises(NarrationError):
        narrator.narrate(_facts())


# ---- through the orchestration layer ----------------------------------------------------------------
def test_good_report_passes_verification_in_one_attempt():
    narrator, _ = _narrator(_ok(json.dumps(VALID)))
    facts = _facts()
    result, info = narrate_verified(facts, narrator)
    assert info["attempts"] == 1 and not info["fell_back"] and result.narrator == "openrouter"
    assert verify_report(result.content, facts.numbers_by_id()).ok


def test_invented_figure_from_a_weak_model_is_retried_then_dropped():
    bad = {**VALID, "insights": [{"title": "x", "text": "You overspent by ₹98,765.", "severity": "watch", "fact_ids": ["spend.top_1"]}]}
    narrator, transport = _narrator(_ok(json.dumps(bad)), _ok(json.dumps(bad)))
    result, info = narrate_verified(_facts(), narrator)
    assert info["attempts"] == 2 and info["dropped_parts"] == ["insight[0]"] and result.content.insights == []
    assert "failed verification" in transport.requests[1]["body"]["messages"][3]["content"]


def test_provider_outage_falls_back_to_template():
    narrator, _ = _narrator((429, {"error": {"message": "slow down"}}))
    result, info = narrate_verified(_facts(), narrator)
    assert info["fell_back"] and "rate limited" in info["error"] and result.narrator == "template"


# ---- free-model listing ----------------------------------------------------------------------------------
def test_list_free_models_filters_by_price_and_sorts_by_context():
    payload = {"data": [
        {"id": "a/paid", "context_length": 200000, "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
        {"id": "b/free-small", "context_length": 8000, "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "c/free-big:free", "context_length": 128000, "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "d/half-free", "context_length": 64000, "pricing": {"prompt": "0", "completion": "0.000001"}},
    ]}
    models = list_free_models(transport=FakeTransport((200, payload)))
    assert [m["id"] for m in models] == ["c/free-big:free", "b/free-small"]
    with pytest.raises(OpenRouterError):
        list_free_models(transport=FakeTransport((500, {})))
