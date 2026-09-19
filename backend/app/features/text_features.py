from sklearn.feature_extraction.text import TfidfVectorizer

from app.preprocessing.text import normalize_description


def build_text_vectorizer() -> TfidfVectorizer:
    """Character n-gram TF-IDF over normalized narrations.

    Character n-grams (rather than words) tolerate truncated or misspelled
    merchant names ('STARBU', 'AMAZN') and generalize to unseen merchants
    that share sub-strings with known ones ('... PHARMACY', '... FUEL').
    """
    return TfidfVectorizer(
        preprocessor=normalize_description,
        analyzer="char_wb",
        ngram_range=(2, 5),
        sublinear_tf=True,
        min_df=2,
    )
