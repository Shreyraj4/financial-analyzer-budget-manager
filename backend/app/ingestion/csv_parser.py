import io

import pandas as pd

REQUIRED_COLUMNS = {"date", "description", "amount"}


def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parses raw CSV bytes into a DataFrame of strings (no type coercion
    yet - that happens in cleaning.py, where we control exactly how dates
    and amounts are interpreted instead of trusting pandas' guesses)."""
    if not file_bytes.strip():
        raise ValueError("The uploaded file is empty")

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Could not parse file as CSV: {exc}") from exc

    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}. "
            f"Expected at least: {', '.join(sorted(REQUIRED_COLUMNS))}"
        )

    if df.empty:
        raise ValueError("CSV has headers but no data rows")

    return df
