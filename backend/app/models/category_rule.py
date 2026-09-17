from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class CategoryRule(Base):
    """Maps a merchant name/pattern to a category so the pipeline stays
    config-driven instead of hardcoding merchant checks in code."""

    __tablename__ = "category_rules"

    id: Mapped[int] = mapped_column(primary_key=True)
    merchant_pattern: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(100))
    subcategory: Mapped[str | None] = mapped_column(String(100), nullable=True)
