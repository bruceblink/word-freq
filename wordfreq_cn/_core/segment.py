import contextlib
import io
import logging
import os
from functools import lru_cache

from .config import DEFAULT_PKUSEG_MODEL, DEFAULT_SEGMENTER

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _current_segmenter() -> str:
    mode = os.getenv("WORDFREQ_SEGMENTER", DEFAULT_SEGMENTER).strip().lower()
    return mode if mode in {"pkuseg", "simple"} else "pkuseg"


def _simple_segment(text: str) -> list[str]:
    return [w for w in text.split() if w.strip()]


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str = DEFAULT_PKUSEG_MODEL):
    if _current_segmenter() == "simple":
        return None

    import spacy_pkuseg

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        tokenizer = spacy_pkuseg.pkuseg(model_name=model_name)
    msg = buf.getvalue().strip()
    if msg:
        logger.debug("pkuseg loader message: %s", msg)
    return tokenizer


@lru_cache(maxsize=128)
def _cached_cut(text: str) -> tuple[str, ...]:
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return tuple(_simple_segment(text))
    return tuple(tokenizer.cut(text))


def segment_text(text: str) -> list[str]:
    if not text:
        return []

    if _current_segmenter() == "simple":
        return _simple_segment(text)

    if len(text) > 500:
        tokenizer = get_tokenizer()
        if tokenizer is None:
            return _simple_segment(text)
        return [w for w in tokenizer.cut(text) if w.strip()]

    return [w for w in _cached_cut(text) if w.strip()]
