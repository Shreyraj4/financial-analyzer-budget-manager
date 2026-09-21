"""Checks that your Anthropic API key and model work, with one tiny request.

Usage (from backend/):  python -m app.agent.check
Costs a fraction of a cent. Prints what is wrong (missing key, bad key, no credit, unknown model) in plain words.
"""
import sys

import anthropic

from app.config import get_settings


def main() -> int:
    settings = get_settings()
    if not settings.anthropic_api_key:
        print("No ANTHROPIC_API_KEY found in .env - the app will use the built-in template report.")
        print("See docs/anthropic_setup.md to add one.")
        return 1
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        reply = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=200,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
    except anthropic.AuthenticationError:
        print("The API key was rejected. Re-copy it from the Anthropic Console into .env (no quotes or spaces).")
    except anthropic.PermissionDeniedError:
        print("This key is not allowed to use that model or feature. Check the workspace in the Console.")
    except anthropic.NotFoundError:
        print(f"Model '{settings.anthropic_model}' was not found. Check ANTHROPIC_MODEL in .env.")
    except anthropic.RateLimitError:
        print("Rate limited. Wait a minute and try again.")
    except anthropic.BadRequestError as e:
        print(f"Request rejected (often: no credit on the account). Details: {e.message}")
    except anthropic.APIConnectionError:
        print("Could not reach the API. Check your internet connection or proxy.")
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}")
    else:
        text = next((b.text for b in reply.content if b.type == "text"), "")
        print(f"Success. Model {reply.model} answered: {text.strip()!r}")
        print(f"Tokens used: {reply.usage.input_tokens} in / {reply.usage.output_tokens} out")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
