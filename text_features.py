from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_text_vectorizer():
    """Build the text representation used by all suitability models."""
    word_features = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 3),
        sublinear_tf=True,
    )

    char_features = TfidfVectorizer(
        analyzer="char_wb",
        max_features=10000,
        ngram_range=(3, 5),
        sublinear_tf=True,
    )

    return FeatureUnion(
        [
            ("word", word_features),
            ("char", char_features),
        ]
    )
