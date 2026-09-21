from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.anomalies import router as anomalies_router
from app.api.auth import router as auth_router
from app.api.budgets import router as budgets_router
from app.api.forecast import router as forecast_router
from app.api.profile import router as profile_router
from app.api.recommendations import router as recommendations_router
from app.api.reports import router as reports_router
from app.api.transactions import router as transactions_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(title="Personal Finance Analyzer + Budgeting Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(anomalies_router)
app.include_router(budgets_router)
app.include_router(transactions_router)
app.include_router(forecast_router)
app.include_router(profile_router)
app.include_router(recommendations_router)
app.include_router(reports_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Personal Finance Analyzer API",
        "status": "running",
    }