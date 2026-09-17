"""add check constraint on budget period

Revision ID: 8148c02131fa
Revises: 03d18fb56d49
Create Date: 2026-09-17 14:08:37.121724

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8148c02131fa'
down_revision: Union[str, Sequence[str], None] = '03d18fb56d49'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_check_constraint(
        "check_budget_period",
        "budgets",
        "period IN ('weekly','monthly')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("check_budget_period", "budgets", type_="check")
