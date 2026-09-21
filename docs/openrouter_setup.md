# Using OpenRouter (free models) for the explanation report

The report writer can be Claude (Anthropic API), an OpenRouter model, or the built-in
template. With `REPORT_NARRATOR=auto` the app picks: Anthropic key -> OpenRouter key ->
template. So adding only an OpenRouter key is enough.

## 1. Add your key

Edit `C:\panel\.env` (git-ignored; create it from `.env.example` if it doesn't exist) and add:

```
OPENROUTER_API_KEY=sk-or-...your-key...
OPENROUTER_MODEL=openrouter/free
REPORT_NARRATOR=auto
```

No quotes, no spaces around `=`. Never commit the key or paste it into chat or frontend code.

## 2. Pick a model

`openrouter/free` is OpenRouter's auto-router: it sends each request to whichever free
model is available right now, which suits a moving free tier. To pin a specific model
(more consistent results), list what is free today:

```
cd C:\panel\backend
..\.venv\Scripts\python -m app.agent.check --list-free
```

Choose a **text/chat** model id from the list (ignore music, vision-only or safety-classifier
models) and set `OPENROUTER_MODEL` to it. Larger instruction-tuned models, such as the
Gemma, Qwen or Nemotron entries, generally follow the JSON format and number rules better
than tiny ones. The free lineup changes often; an id that worked last week may be gone.

## 3. Check it works

```
..\.venv\Scripts\python -m app.agent.check
```

Expected: `Provider: openrouter   Model: ...` then `Success. ... answered: 'OK'`. Otherwise
the message says what to fix:

| Message | Meaning / fix |
|---|---|
| rejected the API key | key mistyped or revoked; re-copy it |
| insufficient credits | that model is not free for your account; choose a `:free` model |
| model was not found or unavailable | pick another id from `--list-free` |
| rate limited | free models have low per-minute/per-day limits; wait or switch model |

## 4. Generate a report

Start the API (`uvicorn app.main:app --reload` from `backend/`), log in for a token, then:

- `GET /reports/status` -> `{"narrator": "openrouter", "model": "openrouter/free"}`
- `POST /reports/generate` -> writes and stores a report
- `GET /reports/latest`

Inspect `verification` in the response: `attempts` (1 or 2), `dropped_parts`, `fell_back`
and `error`.

## What to expect from free models

- **Quality varies.** They invent numbers or break the JSON format more often than Claude.
  The app defends itself: it repairs formatting once, checks every number against the facts,
  retries once with feedback, drops any sentence it cannot verify, and falls back to the
  template report if the call fails. The result is always safe, but may be plainer.
- **Limits.** Free models are rate limited and may be slow or briefly unavailable; the report
  then simply uses the template and records why in `verification.error`.
- **Privacy.** Prompts sent to free models may be logged or used for training by the provider.
  The report prompt contains your spending figures and merchant names. Fine for the synthetic
  demo data; think before using it with real bank data (paid or privacy-guaranteed models,
  or `REPORT_NARRATOR=template`, avoid this).
