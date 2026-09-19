"""add category_source and category_confidence to transactions

Revision ID: b7c1d2e3f4a5
Revises: 8148c02131fa
Create Date: 2026-09-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7c1d2e3f4a5'
down_revision: Union[str, Sequence[str], None] = '8148c02131fa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema (additive, nullable: existing rows are unaffected)."""
    op.add_column('transactions', sa.Column('category_source', sa.String(length=20), nullable=True))
    op.add_column('transactions', sa.Column('category_confidence', sa.Numeric(precision=4, scale=3), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('transactions', 'category_confidence')
    op.drop_column('transactions', 'category_source')
