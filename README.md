# Personal Finance Analyzer + Budgeting Agent

A production-style personal finance analytics platform with an AI budgeting agent.

**Core design principle:** the LLM is never the source of truth for financial numbers.
All spending calculations, comparisons, and anomaly detection are deterministic
(Python/SQL). The AI agent only reasons over and narrates verified structured
metrics via tool calls, then produces a validated structured report.

## Status

Project scaffolding in progress. See `docs/architecture.md` (coming soon) for the
full design.

## Stack

- Backend: Python, FastAPI, SQLAlchemy, Alembic, Pandas
- Database: PostgreSQL (Supabase)
- Frontend: HTML/CSS/JavaScript, Chart.js
- AI: Anthropic Claude (tool calling, structured output)
- Deployment: Railway (backend), Supabase (database)

## Local development

Setup instructions will be added as the backend comes online (Milestone 2).
