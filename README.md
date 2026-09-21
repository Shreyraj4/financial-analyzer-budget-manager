# Personal Finance Analyzer + Budgeting Agent

A production-style personal finance analytics platform with an AI budgeting agent.

**Core design principle:** the LLM is never the source of truth for financial numbers.
All spending calculations, comparisons, and anomaly detection are deterministic
(Python/SQL). The AI agent only reasons over and narrates verified structured
metrics via tool calls, then produces a validated structured report.

## Status

Backend (auth, import, ML categorization / forecasting / anomalies / clustering /
budget recommendations, verified LLM report) and a plain-JS dashboard are working.
Architecture and interview notes: `docs/study/00_project_pitch_and_architecture.md`.

## Stack

- Backend: Python, FastAPI, SQLAlchemy, Alembic, Pandas
- Database: PostgreSQL (Supabase)
- Frontend: HTML/CSS/JavaScript, Chart.js
- AI: Anthropic Claude (tool calling, structured output)
- Deployment: Railway (backend), Supabase (database)

## Local development

```
cp .env.example .env            # fill in DATABASE_URL, JWT_SECRET, and optionally an LLM key
cd backend
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --port 8000

cd ../frontend                   # in a second terminal
python -m http.server 5500       # open http://localhost:5500
```

The dashboard talks to `http://localhost:8000` (override with `window.API_BASE`).
Reports use Claude, OpenRouter (free models work), or a built-in template, chosen from `.env`.
