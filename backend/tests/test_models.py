from app.database.session import Base
from app.models import AgentReport, Budget, CategoryRule, Transaction, User


def test_all_models_are_registered_on_base_metadata():
    table_names = set(Base.metadata.tables.keys())
    assert table_names == {
        "users",
        "transactions",
        "budgets",
        "agent_reports",
        "category_rules",
    }


def test_models_can_be_instantiated():
    user = User(name="Test User", email="test@example.com", password_hash="hashed")
    assert user.email == "test@example.com"

    transaction = Transaction(
        transaction_date="2026-09-01",
        description="SWIGGY",
        amount="450.00",
        transaction_type="debit",
    )
    assert transaction.description == "SWIGGY"

    budget = Budget(category="Dining", amount="5000.00", period="monthly",
                     start_date="2026-09-01", end_date="2026-09-30")
    assert budget.category == "Dining"

    report = AgentReport(period_start="2026-09-01", period_end="2026-09-07",
                          summary="ok", structured_report={"key": "value"})
    assert report.structured_report == {"key": "value"}

    rule = CategoryRule(merchant_pattern="SWIGGY", category="Dining")
    assert rule.category == "Dining"
