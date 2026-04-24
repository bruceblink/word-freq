import logging
import os
import uuid
from collections import Counter
from datetime import datetime
from importlib.resources import files
from io import BytesIO
from typing import Any

from wordcloud import WordCloud

from .config import DEFAULT_FONT_CANDIDATES
from .frequency import count_word_frequency

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_default_font_path() -> str:
    try:
        fonts_pkg = files("wordfreq_cn.data.fonts")
        for name in DEFAULT_FONT_CANDIDATES:
            candidate = fonts_pkg / name
            if candidate.exists():
                return str(candidate)
    except Exception as e:
        logger.debug("package fonts not available: %s", e)

    sys_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:\\Windows\\Fonts\\msyh.ttf",
    ]
    for p in sys_candidates:
        if os.path.exists(p):
            return p

    raise RuntimeError("No suitable font found for WordCloud. Please provide font_path.")


def _generate_wordcloud(
    frequencies: Counter,
    output_path: str | None,
    font_path: str | None = None,
    width: int = 900,
    height: int = 600,
    background_color: str = "white",
    colormap: str | None = None,
    max_words: int | None = 100,
    mask: Any | None = None,
) -> str | bytes:
    if not frequencies:
        raise ValueError("frequencies is empty")

    font_path = font_path or _get_default_font_path()

    wc_kwargs: dict[str, Any] = dict(
        font_path=font_path,
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        mask=mask,
    )
    if colormap:
        wc_kwargs["colormap"] = colormap

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(frequencies)
    img = wc.to_image()

    if output_path:
        img.save(output_path, format="PNG")
        return output_path

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_trend_wordcloud(
    news_by_date: dict[str, list[str]],
    stopwords: set[str] | None = None,
    min_len: int = 2,
    ngram_range: tuple[int, int] = (1, 1),
    output_dir: str | None = None,
    font_path: str | None = None,
    width: int = 900,
    height: int = 600,
    max_words: int | None = 100,
    background_color: str = "white",
    return_bytes: bool = False,
) -> list[str | bytes]:
    current_date = datetime.now()
    font_path = font_path or _get_default_font_path()

    results: list[str | bytes] = []

    for date_str, texts in sorted(news_by_date.items()):
        if not texts:
            continue
        date_str = date_str or current_date.strftime("%Y-%m-%d")
        counter = count_word_frequency(
            texts,
            stopwords=stopwords,
            min_len=min_len,
            ngram_range=ngram_range,
        )

        if not counter:
            continue

        if return_bytes:
            img_bytes = _generate_wordcloud(
                counter,
                output_path=None,
                font_path=font_path,
                width=width,
                height=height,
                max_words=max_words,
                background_color=background_color,
            )
            results.append(img_bytes)
            continue

        output_dir_final = os.path.join(output_dir, date_str) if output_dir else os.path.join("wordclouds", date_str)
        os.makedirs(output_dir_final, exist_ok=True)

        out_file = os.path.join(output_dir_final, f"wordcloud_{uuid.uuid4().hex}.png")

        _generate_wordcloud(
            counter,
            out_file,
            font_path=font_path,
            width=width,
            height=height,
            max_words=max_words,
            background_color=background_color,
        )
        results.append(out_file)

    return results
