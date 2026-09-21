"""Checks your LLM provider setup with one tiny request.

Usage (from backend/):
  python -m app.agent.check               test whichever provider is configured (Claude or OpenRouter)
  python -m app.agent.check --list-free   list free OpenRouter models (no key needed)
"""
import sys

from app.agent.openrouter import OpenRouterClient, OpenRouterError, list_free_models
from app.agent.service import narrator_choice
from app.config import get_settings


def check_openrouter(settings) -> int:
    client = OpenRouterClient(settings.openrouter_api_key, settings.openrouter_base_url, timeout=60)
    try:
        payload = client.chat(settings.openrouter_model, [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=200)
    except OpenRouterError as e:
        print(f"FAILED: {e}")
        if e.status in (404, 400):
            print("Tip: pick another model with  python -m app.agent.check --list-free  and set OPENROUTER_MODEL in .env")
        return 1
    text = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = payload.get("usage") or {}
    print(f"Success. {payload.get('model', settings.openrouter_model)} answered: {text.strip()[:60]!r}")
    print(f"Tokens: {usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out")
    return 0


def check_claude(settings) -> int:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        reply = client.messages.create(model=settings.anthropic_model, max_tokens=200, messages=[{"role": "user", "content": "Reply with exactly: OK"}])
    except anthropic.AuthenticationError:
        print("FAILED: the Anthropic API key was rejected.")
    except anthropic.NotFoundError:
        print(f"FAILED: model '{settings.anthropic_model}' not found. Check ANTHROPIC_MODEL.")
    except anthropic.BadRequestError as e:
        print(f"FAILED: request rejected (often no credit). {e.message}")
    except anthropic.APIConnectionError:
        print("FAILED: could not reach the API.")
    except anthropic.APIStatusError as e:
        print(f"FAILED: API error {e.status_code}: {e.message}")
    else:
        print(f"Success. Model {reply.model} answered: {next((b.text for b in reply.content if b.type == 'text'), '').strip()!r}")
        return 0
    return 1


def main(argv: list[str]) -> int:
    if "--list-free" in argv:
        try:
            models = list_free_models()
        except OpenRouterError as e:
            print(f"FAILED: {e}")
            return 1
        print(f"{len(models)} free models (largest context first). Set OPENROUTER_MODEL in .env to one of these ids:")
        for m in models[:40]:
            print(f"  {m['id']:60s} context {m['context_length']:>9,}")
        return 0

    settings = get_settings()
    kind, model = narrator_choice(settings)
    if kind == "template":
        print("No LLM key found in .env (ANTHROPIC_API_KEY / OPENROUTER_API_KEY), or REPORT_NARRATOR=template.")
        print("The app will use the built-in template report. See docs/openrouter_setup.md or docs/anthropic_setup.md.")
        return 1
    print(f"Provider: {kind}   Model: {model}")
    return check_openrouter(settings) if kind == "openrouter" else check_claude(settings)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
