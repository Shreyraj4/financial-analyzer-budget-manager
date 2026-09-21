"""OpenRouter narrator: lets free/open models write the report.

OpenRouter (https://openrouter.ai) exposes many models behind one
OpenAI-style HTTP API. This talks to it over plain HTTPS (stdlib only), so it
needs no SDK. Free models rarely support schema-constrained output, so the
prompt asks for a JSON object and we validate it strictly with Pydantic; the
same number verification and template fallback used for Claude apply on top.
"""
import json
import re
import urllib.error
import urllib.request

from pydantic import ValidationError

from app.agent.facts import FactSet
from app.agent.narrators import SYSTEM_PROMPT, NarrationError, NarrationResult, facts_message
from app.agent.schemas import AgentReportContent

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
SEVERITIES = {"info", "positive", "watch", "alert"}

JSON_FORMAT = """

Respond with ONLY one JSON object, no markdown fences and no text before or after it, in exactly this shape:
{
  "headline": "one sentence",
  "summary": "two to four sentences",
  "insights": [{"title": "short title", "text": "one or two sentences", "severity": "info|positive|watch|alert", "fact_ids": ["fact.id"]}],
  "suggested_actions": [{"text": "one sentence", "fact_ids": ["fact.id"]}]
}"""


class OpenRouterError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _urllib_transport(url: str, headers: dict, body: bytes | None, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        raise OpenRouterError(f"Could not reach OpenRouter: {e.reason}") from e


_STATUS_HELP = {
    401: "OpenRouter rejected the API key (check OPENROUTER_API_KEY in .env)",
    402: "OpenRouter says the account has insufficient credits for this model",
    403: "OpenRouter refused the request (key or model not permitted)",
    404: "the model was not found or is currently unavailable (run: python -m app.agent.check --list-free)",
    408: "the request timed out at OpenRouter",
    429: "rate limited: free models have low limits; wait a minute or pick another model",
}


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0, transport=None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport or _urllib_transport

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Personal Finance Analyzer",  # optional attribution shown on OpenRouter
        }

    def chat(self, model: str, messages: list[dict], max_tokens: int = 4000, temperature: float = 0.2) -> dict:
        # "reasoning" asks thinking models to keep it short; free models may burn the whole budget on thoughts otherwise.
        body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                           "reasoning": {"effort": "low"}}).encode()
        status, raw = self.transport(f"{self.base_url}/chat/completions", self._headers(), body, self.timeout)
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {}
        if status != 200:
            detail = (payload.get("error") or {}).get("message") if isinstance(payload.get("error"), dict) else None
            reason = _STATUS_HELP.get(status) or ("OpenRouter or the model provider had a problem" if status >= 500 else "request rejected")
            raise OpenRouterError(f"OpenRouter HTTP {status}: {reason}" + (f" ({detail})" if detail else ""), status)
        if payload.get("error"):  # OpenRouter can return 200 with an error body
            err = payload["error"]
            raise OpenRouterError(f"OpenRouter error: {err.get('message', err) if isinstance(err, dict) else err}")
        return payload


def list_free_models(timeout: float = 30.0, transport=None) -> list[dict]:
    """Public endpoint (no key needed): models whose prompt and completion prices are both zero."""
    status, raw = (transport or _urllib_transport)(f"{DEFAULT_BASE_URL}/models", {"Accept": "application/json"}, None, timeout)
    if status != 200:
        raise OpenRouterError(f"Could not list models (HTTP {status})", status)
    free = []
    for m in json.loads(raw).get("data", []):
        pricing = m.get("pricing") or {}
        if str(pricing.get("prompt")) in {"0", "0.0"} and str(pricing.get("completion")) in {"0", "0.0"}:
            free.append({"id": m["id"], "context_length": m.get("context_length") or 0})
    return sorted(free, key=lambda m: -m["context_length"])


def parse_report(text: str) -> AgentReportContent:
    """Tolerant JSON extraction + strict validation. Raises ValueError with a message fit to show the model."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)        # reasoning models
    text = re.sub(r"```(?:json)?", "", text)                                       # markdown fences
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the reply")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as e:
        raise ValueError(f"the reply is not valid JSON ({e})") from e

    # Weaker models often slip on small things: normalize instead of failing.
    for key in ("insights", "suggested_actions"):
        for item in data.get(key) or []:
            if isinstance(item, dict):
                ids = item.get("fact_ids")
                item["fact_ids"] = [ids] if isinstance(ids, str) else (ids or [])
                if key == "insights":
                    sev = str(item.get("severity", "info")).lower()
                    item["severity"] = sev if sev in SEVERITIES else "info"
    data.setdefault("insights", [])
    data.setdefault("suggested_actions", [])
    try:
        return AgentReportContent.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:4])
        raise ValueError(f"the JSON does not match the required shape ({problems})") from e


class OpenRouterNarrator:
    name = "openrouter"
    supports_feedback = True

    def __init__(self, client: OpenRouterClient, model: str, max_tokens: int = 12000):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _ask(self, messages: list[dict]) -> tuple[str, dict]:
        payload = self.client.chat(self.model, messages, max_tokens=self.max_tokens)
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise NarrationError("OpenRouter returned no message content") from e
        usage = payload.get("usage") or {}
        return text, {"model": payload.get("model", self.model), "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}

    def narrate(self, facts: FactSet, previous: AgentReportContent | None = None, feedback: str | None = None) -> NarrationResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT + JSON_FORMAT}, {"role": "user", "content": facts_message(facts)}]
        if previous is not None and feedback:
            messages += [{"role": "assistant", "content": previous.model_dump_json()}, {"role": "user", "content": feedback}]

        text, meta = self._ask(messages)
        totals = {"input_tokens": meta["input_tokens"], "output_tokens": meta["output_tokens"]}
        try:
            content = parse_report(text)
        except ValueError as first_error:
            # One repair round for formatting slips (fences, prose around the JSON, wrong shape).
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That reply could not be used: {first_error}. Reply again with ONLY the JSON object."}]
            text, meta2 = self._ask(messages)
            totals = {k: totals[k] + meta2[k] for k in totals}
            try:
                content = parse_report(text)
            except ValueError as e:
                raise NarrationError(f"the model did not return a usable report: {e}") from e
        return NarrationResult(content=content, narrator=self.name, model=meta["model"], usage=totals)
