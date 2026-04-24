import argparse
import random
import string
import time
import tracemalloc
from typing import Callable, Any

from wordfreq_cn.core import (
    count_word_frequency,
    extract_keywords_tfidf,
    extract_keywords_tfidf_per_doc,
)


def _random_chinese_word(length: int = 2) -> str:
    return "".join(chr(random.randint(0x4E00, 0x9FFF)) for _ in range(length))


def _random_english_word(length: int = 6) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def build_corpus(docs: int, words_per_doc: int, seed: int) -> list[str]:
    random.seed(seed)
    zh_vocab = [_random_chinese_word(random.randint(2, 3)) for _ in range(3000)]
    en_vocab = [_random_english_word(random.randint(4, 9)) for _ in range(2000)]
    vocab = zh_vocab + en_vocab

    corpus: list[str] = []
    for _ in range(docs):
        tokens = [random.choice(vocab) for _ in range(words_per_doc)]
        corpus.append(" ".join(tokens))
    return corpus


def run_with_peak_memory(name: str, func: Callable[[], Any]) -> tuple[str, float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    _ = func()
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return name, elapsed, peak_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory benchmark for wordfreq-cn")
    parser.add_argument("--docs", type=int, default=3000)
    parser.add_argument("--words-per-doc", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    corpus = build_corpus(args.docs, args.words_per_doc, args.seed)

    benches = [
        (
            "tfidf_global",
            lambda: extract_keywords_tfidf(
                corpus,
                top_k=args.topk,
                include_artifacts=False,
            ),
        ),
        (
            "tfidf_per_doc",
            lambda: extract_keywords_tfidf_per_doc(
                corpus,
                top_k=5,
            ),
        ),
        (
            "word_frequency",
            lambda: count_word_frequency(corpus),
        ),
    ]

    print(f"dataset: docs={args.docs}, words_per_doc={args.words_per_doc}")
    print("benchmark\telapsed_s\tpeak_py_mem_mb")

    for name, fn in benches:
        bench_name, elapsed, peak_mb = run_with_peak_memory(name, fn)
        print(f"{bench_name}\t{elapsed:.3f}\t{peak_mb:.2f}")


if __name__ == "__main__":
    main()
