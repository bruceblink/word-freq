import importlib
import re

import contractions

from .segment import segment_text as _segment_text

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_APOSTROPHES_CHARS = "''’＇`´!"
_RE_KEEP_CHARS = re.compile(fr"[^A-Za-z0-9一-鿿{re.escape(_APOSTROPHES_CHARS)}]+")
_RE_ISOLATED_APOSTROPHE = re.compile(r"(?<![A-Za-z])'|'(?![A-Za-z])")
_RE_DIGITS = re.compile(r"\d+")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_digits: bool = False,
) -> str:
    if not text:
        return ""
    s = text

    if remove_urls:
        s = _RE_URL.sub(" ", s)

    if remove_emails:
        s = _RE_EMAIL.sub(" ", s)

    apostrophes_chars = "'’＇`´!"
    escaped_apostrophes = re.escape(apostrophes_chars)
    s = re.sub(fr"[^A-Za-z0-9一-鿿{escaped_apostrophes}]+", " ", s)

    for apostrophe in apostrophes_chars:
        s = s.replace(apostrophe, "'")
    s = _RE_ISOLATED_APOSTROPHE.sub(" ", s)

    if remove_digits:
        s = _RE_DIGITS.sub(" ", s)

    s = _RE_WHITESPACE.sub(" ", s).strip()

    return s.lower()


def _resolve_segment_text():
    try:
        core_mod = importlib.import_module("wordfreq_cn.core")
        return core_mod.segment_text
    except Exception:
        return _segment_text


def preprocess_text(text: str, stopwords: set[str] | None = None, min_len: int = 2) -> list[str]:
    cleaned = clean_text(text)
    cleaned = contractions.fix(cleaned)
    words = _resolve_segment_text()(cleaned)

    if stopwords:
        words = [w for w in words if w and w.lower() not in stopwords and len(w) >= min_len]
    else:
        words = [w for w in words if w and len(w) >= min_len]

    return words
