from dataclasses import asdict, dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class KeywordItem:
    word: str
    weight: float
    count: int | None = None


@dataclass
class TfIdfResult:
    keywords: list[KeywordItem]
    vectorizer: TfidfVectorizer | None
    matrix: Any

    def keywords_to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        if not self.keywords:
            return "[]"
        import json

        return json.dumps([asdict(k) for k in self.keywords], indent=indent, ensure_ascii=ensure_ascii)
