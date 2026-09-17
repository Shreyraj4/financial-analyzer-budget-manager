from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.session import Base
from app.models import Budget, User


@pytest.fixture
def db_session():
    # In-memory SQLite, isolated per test, enforces CHECK constraints
    # just like Postgres does - good enough for this model-level rule.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[User.__table__, Budget.__table__])
    with Session(engine) as session:
        yield session


def _make_user(session: Session) -> User:
    user = User(name="Test User", email="test@example.com", password_hash="hashed")
    session.add(user)
    session.flush()  # assigns user.id without committing
    return user


def test_valid_period_is_accepted(db_session):
    user = _make_user(db_session)
    budget = Budget(
        user_id=user.id, category="Dining", amount="5000.00",
        period="monthly", start_date=date(2026, 9, 1), end_date=date(2026, 9, 30),
    )
    db_session.add(budget)
    db_session.commit()  # should not raise
    assert budget.id is not None


def test_invalid_period_is_rejected(db_session):
    user = _make_user(db_session)
    budget = Budget(
        user_id=user.id, category="Dining", amount="5000.00",
        period="yearly", start_date=date(2026, 1, 1), end_date=date(2026, 12, 31),
    )
    db_session.add(budget)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
