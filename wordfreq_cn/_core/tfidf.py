from typing import Any, Iterable, Iterator

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DEFAULT_MAX_FEATURES, DEFAULT_NGRAM_RANGE, DEFAULT_TOKEN_PATTERN
from .models import KeywordItem, TfIdfResult
from .text import preprocess_text


def _iter_processed_corpus(corpus: Iterable[str], stopwords: set[str] | None = None) -> Iterator[str]:
    for doc in corpus:
        yield " ".join(preprocess_text(doc, stopwords=stopwords))


def _fit_tfidf_matrix(
    corpus: list[str],
    ngram_range: tuple[int, int],
    stopwords: set[str] | None,
    max_features: int,
    min_df: int,
    max_df: float,
    sublinear_tf: bool,
    token_pattern: str,
) -> tuple[TfidfVectorizer, Any, Any]:
    n_docs = len(corpus)
    adjusted_max_df = 1.0 if n_docs == 1 else max_df

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=adjusted_max_df,
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(_iter_processed_corpus(corpus, stopwords=stopwords))
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, X, feature_names


def extract_keywords_tfidf(
    corpus: list[str],
    top_k: int | None = 20,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    stopwords: set[str] | None = None,
    max_features: int = DEFAULT_MAX_FEATURES,
    min_df: int = 1,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    token_pattern: str = DEFAULT_TOKEN_PATTERN,
    include_artifacts: bool = True,
) -> TfIdfResult:
    if not corpus:
        return TfIdfResult([], None, None)

    vectorizer, X, feature_names = _fit_tfidf_matrix(
        corpus=corpus,
        ngram_range=ngram_range,
        stopwords=stopwords,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        token_pattern=token_pattern,
    )

    weights_array = np.asarray(X.sum(axis=0)).ravel()
    doc_counts = np.diff(X.tocsc().indptr)

    kw_items = [
        KeywordItem(word=feature_names[i], weight=float(weights_array[i]), count=int(doc_counts[i]))
        for i in range(len(feature_names))
    ]

    kw_items.sort(key=lambda x: x.weight, reverse=True)

    if top_k:
        top_keywords = kw_items[:top_k]
    else:
        top_keywords = kw_items

    if not include_artifacts:
        return TfIdfResult(keywords=top_keywords, vectorizer=None, matrix=None)

    return TfIdfResult(keywords=top_keywords, vectorizer=vectorizer, matrix=X)


def extract_keywords_tfidf_per_doc(corpus: list[str], top_k: int = 5, **kwargs) -> list[list[KeywordItem]]:
    if not corpus:
        return []

    _, X, feature_names = _fit_tfidf_matrix(
        corpus=corpus,
        ngram_range=kwargs.get("ngram_range", DEFAULT_NGRAM_RANGE),
        stopwords=kwargs.get("stopwords"),
        max_features=kwargs.get("max_features", DEFAULT_MAX_FEATURES),
        min_df=kwargs.get("min_df", 1),
        max_df=kwargs.get("max_df", 0.95),
        sublinear_tf=kwargs.get("sublinear_tf", True),
        token_pattern=kwargs.get("token_pattern", DEFAULT_TOKEN_PATTERN),
    )

    results: list[list[KeywordItem]] = []

    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            results.append([])
            continue

        values = row.data
        indices = row.indices

        if top_k and row.nnz > top_k:
            top_positions = np.argpartition(values, -top_k)[-top_k:]
            sorted_positions = top_positions[np.argsort(values[top_positions])[::-1]]
        else:
            sorted_positions = np.argsort(values)[::-1]

        keywords = [
            KeywordItem(
                word=feature_names[indices[pos]],
                weight=float(values[pos]),
                count=1,
            )
            for pos in sorted_positions
        ]

        results.append(keywords)

    return results
