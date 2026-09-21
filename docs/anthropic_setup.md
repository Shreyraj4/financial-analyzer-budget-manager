# Adding your Anthropic (Claude) API key

The app works **without** a key: reports are then written by a built-in template
(same structure, plainer wording). Adding a key makes Claude write the report in
natural language. Nothing else in the project needs the key.

## 1. Get the key

1. Go to the Anthropic Console: https://console.anthropic.com (if it redirects, follow it) and sign up or log in.
   - Note: a Claude.ai chat subscription is **separate** from API access. The API is billed on its own.
2. Add credit: in the Console open **Billing** (or **Plans & Billing**) and add prepaid credit.
   A few dollars is plenty for development. Set a monthly spend limit there too.
3. Open **API Keys** -> **Create Key**. Name it something like `panel-dev`.
4. **Copy the key immediately.** It starts with `sk-ant-` and is shown only once.

## 2. Put it in the project

Open (or create) `C:\panel\.env` - the file is git-ignored - and set:

```
ANTHROPIC_API_KEY=sk-ant-...your-key...
ANTHROPIC_MODEL=claude-opus-5
REPORT_NARRATOR=auto
```

- No quotes, no spaces around `=`.
- `ANTHROPIC_MODEL`: `claude-opus-5` is the default. `claude-sonnet-5` is cheaper and
  fine for this task if you want to save money.
- `REPORT_NARRATOR`: `auto` = use Claude when a key exists; `template` = never call
  the API (good for demos or if credit runs out).
- Never commit `.env`, never paste the key in chat/screenshots, and never put it in
  frontend code. If it leaks, delete it in the Console and create a new one.

The SDK is already in `backend/requirements.txt` (`anthropic`). If you set up a fresh
environment: `pip install -r backend/requirements.txt`.

## 3. Check it works (costs a fraction of a cent)

From `C:\panel\backend`:

```
..\.venv\Scripts\python -m app.agent.check
```

Success looks like: `Success. Model claude-opus-5 answered: 'OK'`. Otherwise it tells you
what is wrong:

| Message | Fix |
|---|---|
| No ANTHROPIC_API_KEY found | `.env` not saved, wrong folder, or a typo in the variable name |
| The API key was rejected | Re-copy the key; no quotes/spaces |
| Request rejected (often: no credit) | Add credit in Billing |
| Model '...' was not found | Fix `ANTHROPIC_MODEL` |
| Rate limited / Could not reach the API | Wait, or check your network |

## 4. Use it

Start the server (`uvicorn app.main:app --reload` from `backend/`), log in to get a token,
then (with `Authorization: Bearer <token>`):

- `GET  /reports/status`   -> `{"narrator": "claude", "model": "claude-opus-5"}` when the key is active
- `POST /reports/generate` -> creates and stores a report (one paid Claude call)
- `GET  /reports/latest`   -> the most recent stored report
- `GET  /reports`          -> list of past report headlines

Reports are generated only when you call `POST /reports/generate` (never on page load), so you
control the cost. Each report is roughly a few thousand input tokens and about a thousand
output tokens: cents per report on Opus, less on Sonnet. The response shows the exact token
usage under `narrator.usage`.

## 5. What happens if something goes wrong

The app never fails because of the LLM. If Claude errors, refuses, or produces a number that
is not in the facts twice in a row, the report falls back to the template (or drops the
unverifiable sentence). Look at `verification` in the response to see what happened:
`attempts`, `fell_back`, `error`, `dropped_parts`.
