from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str = ""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"
    # OpenRouter (https://openrouter.ai): one key for many models, including free ones.
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Who writes the report: "auto" = Claude if ANTHROPIC_API_KEY is set, else OpenRouter if
    # OPENROUTER_API_KEY is set, else the built-in template. Or force "claude" | "openrouter" | "template".
    report_narrator: str = "auto"
    cors_origins: str = "http://localhost:5500"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def sqlalchemy_database_url(self) -> str:
        # Supabase gives a "postgresql://" URI; psycopg3 needs the
        # "postgresql+psycopg://" dialect prefix for SQLAlchemy to pick the
        # right driver.
        return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
