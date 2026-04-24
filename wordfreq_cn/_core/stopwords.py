import logging
import os
from importlib.resources import files

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_stopwords(custom_file: str | None = None, hit_file: str | None = None) -> set[str]:
    stopwords = set()

    def _load_from_path(path: str):
        if not path or not os.path.exists(path):
            logger.debug("Stopwords file not found: %s", path)
            return
        try:
            with open(path, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    stopwords.add(line.lower())
        except Exception as e0:
            logger.warning("Failed to load stopwords from %s: %s", path, e0)

    if hit_file:
        _load_from_path(hit_file)
    if custom_file:
        _load_from_path(custom_file)

    try:
        stopwords_file = files("wordfreq_cn.data") / "cn_stopwords.txt"
        _load_from_path(str(stopwords_file))
    except Exception as e:
        logger.debug("Failed to load builtin stopwords: %s", e)

    return stopwords
