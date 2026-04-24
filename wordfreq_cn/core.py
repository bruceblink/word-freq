"""
wordfreq_cn.core
----------------
兼容门面：对外保持原有 API，内部实现位于 wordfreq_cn._core。
"""

from ._core.config import DEFAULT_MAX_FEATURES, DEFAULT_NGRAM_RANGE, DEFAULT_TOKEN_PATTERN
from ._core.frequency import count_word_frequency
from ._core.models import KeywordItem, TfIdfResult
from ._core.segment import segment_text
from ._core.stopwords import load_stopwords
from ._core.text import clean_text, preprocess_text
from ._core.tfidf import extract_keywords_tfidf, extract_keywords_tfidf_per_doc
from ._core.wordclouding import generate_trend_wordcloud


def extract_keywords(
    data: str | list[str],
    method: str = "tfidf",
    top_k: int = 20,
    stopwords: set[str] | None = None,
    **kwargs,
) -> list[KeywordItem] | list[list[KeywordItem]]:
    method = method.lower()
    if method == "tfidf":
        if not isinstance(data, list):
            raise TypeError("tfidf requires corpus list[str] as input")
        res = extract_keywords_tfidf(data, top_k=top_k, stopwords=stopwords, **kwargs)
        return res.keywords
    if method == "tfidf_per_doc":
        if not isinstance(data, list):
            raise TypeError("tfidf_per_doc requires corpus list[str] as input")
        return extract_keywords_tfidf_per_doc(data, top_k=top_k, stopwords=stopwords, **kwargs)
    raise ValueError(f"Unknown method: {method}. Supported: tfidf, tfidf_per_doc")


__all__ = [
    "DEFAULT_MAX_FEATURES",
    "DEFAULT_NGRAM_RANGE",
    "DEFAULT_TOKEN_PATTERN",
    "KeywordItem",
    "TfIdfResult",
    "load_stopwords",
    "clean_text",
    "preprocess_text",
    "segment_text",
    "extract_keywords_tfidf",
    "extract_keywords_tfidf_per_doc",
    "count_word_frequency",
    "generate_trend_wordcloud",
    "extract_keywords",
]
