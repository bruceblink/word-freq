from collections import Counter
from typing import Iterable, Iterator

from .text import preprocess_text


def _generate_ngrams(words: list[str], n: int) -> Iterator[str]:
    if n <= 1:
        for word in words:
            yield word
        return
    for i in range(len(words) - n + 1):
        yield "".join(words[i : i + n])


def count_word_frequency(
    corpus: Iterable[str],
    stopwords: set[str] | None = None,
    min_len: int = 2,
    ngram_range: tuple[int, int] = (1, 1),
) -> Counter:
    counter = Counter()
    for text in corpus:
        words = preprocess_text(text, stopwords=stopwords, min_len=min_len)
        for n in range(ngram_range[0], ngram_range[1] + 1):
            if n == 1:
                counter.update(g for g in words if len(g) >= min_len)
            else:
                counter.update(g for g in _generate_ngrams(words, n) if len(g) >= min_len)
    return counter
