import os

DEFAULT_MAX_FEATURES = 2000
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_TOKEN_PATTERN = r"(?u)[一-鿿A-Za-z0-9]+"
DEFAULT_PKUSEG_MODEL = "news"
DEFAULT_SEGMENTER = os.getenv("WORDFREQ_SEGMENTER", "pkuseg").strip().lower() or "pkuseg"
DEFAULT_FONT_CANDIDATES = [
    "SourceHanSansHWSC-VF.ttf",
    "SourceHanSansSC-Regular.otf",
    "NotoSansCJK-Regular.ttc",
    "msyh.ttc",
]
