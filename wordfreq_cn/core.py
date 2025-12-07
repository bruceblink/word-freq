"""
wordfreq_cn.core
----------------
优化后的关键词提取与词云工具集合，基于 jieba3 + sklearn + wordcloud。

主要功能：
- 停用词加载 (load_stopwords)
- 文本清洗 (clean_text) 与预处理 (preprocess_text)
- 分词（缓存）(segment_text)
- 全局 / per-doc TF-IDF 关键词提取 (extract_keywords_tfidf, extract_keywords_tfidf_per_doc)
- 词频统计 (count_word_frequency)
- 词云生成 (generate_wordcloud, generate_trend_wordcloud)
"""

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from functools import lru_cache
from importlib.resources import files
from io import BytesIO
from typing import Any

import jieba3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------
# Configuration defaults
# ---------------------------
DEFAULT_MAX_FEATURES = 2000
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_TOKEN_PATTERN = r"(?u)[\u4e00-\u9fffA-Za-z0-9]+"
DEFAULT_FONT_CANDIDATES = [
    "SourceHanSansHWSC-VF.ttf",
    "SourceHanSansSC-Regular.otf",
    "NotoSansCJK-Regular.ttc",
    "msyh.ttc"  # Windows fallback
]

jieba3_tokenizer = jieba3.jieba3()


# ---------------------------
# Helper dataclasses
# ---------------------------

@dataclass
class KeywordItem:
    word: str
    weight: float
    count: int | None = None  # optional: available for TF-IDF (global counts)


@dataclass
class TfIdfResult:
    keywords: list[KeywordItem]
    vectorizer: TfidfVectorizer | None
    matrix: Any  # sparse matrix returned by fit_transform

    def keywords_to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """
        将 keywords 字段转成 JSON 字符串。
        """
        if not self.keywords:
            return "[]"
        import json
        return json.dumps([asdict(k) for k in self.keywords], indent=indent, ensure_ascii=ensure_ascii)


# ---------------------------
# Stopwords
# ---------------------------

def load_stopwords(custom_file: str | None = None, hit_file: str | None = None) -> set[str]:
    """
    加载停用词集合（hit_file -> custom_file -> package 内置）

    - 忽略空行和以 '#' 开头的注释行
    - 返回小写化的词列表
    """
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

    # package 内置停用词（确保 package 数据存在）
    try:
        stopwords_file = files('wordfreq_cn.data') / 'cn_stopwords.txt'
        _load_from_path(str(stopwords_file))
    except Exception as e:
        logger.debug("Failed to load builtin stopwords: %s", e)

    return stopwords


# ---------------------------
# Text cleaning / preprocessing
# ---------------------------

def clean_text(text: str, remove_urls: bool = True, remove_emails: bool = True, remove_digits: bool = False) -> str:
    """
    基础清洗：去掉非中文/英文/数字字符, 允许英文缩写的撇号保留、合并空白，并可选删除 URL / email / 数字（或保留数字）。
    返回小写形式（英文）。
    """
    if not text:
        return ""
    s = text

    if remove_urls:
        s = re.sub(r"https?://\S+|www\.\S+", " ", s)

    if remove_emails:
        s = re.sub(r"\S+@\S+", " ", s)

    # 保留中文、英文、数字和撇号
    s = re.sub(r"[^\w\u4e00-\u9fff'’]", " ", s)

    # 删除不在单词内部的撇号（确保 didn't / isn't 这种才被保留）
    s = re.sub(r"(?<![A-Za-z])[’']+|[’']+(?![A-Za-z])", " ", s)

    if remove_digits:
        s = re.sub(r"\d+", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


import contractions


def preprocess_text(
        text: str,
        stopwords: list[str] | None = None,
        min_len: int = 2
) -> list[str]:
    """
    预处理管道：clean_text -> 扩展英文缩写 -> 分词 -> 停用词 & 长度过滤
    返回词列表（原始词形，不再小写中文）
    """

    # 1. 清洗中文/英文/符号
    cleaned = clean_text(text)

    # 2. 展开英文缩写（don't -> do not）
    cleaned = contractions.fix(cleaned)

    # 3. 分词（中文/英文混合）
    words = segment_text(cleaned)

    # 4. 停词过滤 + 词长过滤
    if stopwords:
        sw = set(w.lower() for w in stopwords)
        words = [
            w for w in words
            if w and w.lower() not in sw and len(w) >= min_len
        ]
    else:
        words = [w for w in words if w and len(w) >= min_len]

    return words


# ---------------------------
# Segment (jieba3) with caching
# ---------------------------

@lru_cache(maxsize=65536)
def _cached_cut(text: str) -> tuple[str, ...]:
    """
    内部缓存分词结果（不可变 tuple），减少重复分词成本。
    """
    return tuple(jieba3_tokenizer.cut_text(text))


def segment_text(text: str) -> list[str]:
    """
    对字符串进行分词，返回词列表（去除空字符串）。
    使用 lru_cache 缓存分词结果。
    """
    if not text:
        return []
    return [w for w in _cached_cut(text) if w.strip()]


# ---------------------------
# TF-IDF: 全局 & per-doc 提取
# ---------------------------

def extract_keywords_tfidf(
        corpus: list[str],
        top_k: int|None = 20,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
        stopwords: set[str] | None = None,
        max_features: int = DEFAULT_MAX_FEATURES,
        min_df: int = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        token_pattern: str = DEFAULT_TOKEN_PATTERN,
) -> TfIdfResult:
    """
    全局 TF-IDF：基于整个语料库计算 TF-IDF，自动清洗文本 + 展开缩写 + 分词。

    返回 TfIdfResult:
      - keywords: list of KeywordItem (word, weight, count)
      - vectorizer: 训练好的 TfidfVectorizer（便于后续 transform）
      - matrix: 稀疏矩阵 X (n_docs, n_features)
    """
    if not corpus:
        return TfIdfResult([], None, None)

    # ----------------------------
    # 文本预处理
    # ----------------------------
    processed_corpus = [
        " ".join(preprocess_text(doc, stopwords=stopwords))
        for doc in corpus
    ]

    # ----------------------------
    # 修复单文档时 max_df 问题
    # ----------------------------
    n_docs = len(processed_corpus)
    adjusted_max_df = max_df
    if n_docs == 1:
        adjusted_max_df = 1.0

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, token_pattern=token_pattern,
                                 sublinear_tf=sublinear_tf, min_df=min_df, max_df=adjusted_max_df)

    X = vectorizer.fit_transform(processed_corpus)  # shape: (n_docs, n_features)
    feature_names = vectorizer.get_feature_names_out()

    # 按列求和得到每个特征在所有文档中的 TF-IDF 总权重
    weights_array = np.asarray(X.sum(axis=0)).ravel()

    # 统计每个 token 在多少个文档中出现（非零计数）
    doc_counts = np.asarray((X > 0).sum(axis=0)).ravel()

    # 包装结果
    kw_items = [
        KeywordItem(word=feature_names[i], weight=float(weights_array[i]), count=int(doc_counts[i]))
        for i in range(len(feature_names))
    ]

    # 排序
    kw_items.sort(key=lambda x: x.weight, reverse=True)

    # 截断 top_k
    if top_k:
        top_keywords = kw_items[:top_k]
    else:
        top_keywords = kw_items  # 返回所有关键词

    return TfIdfResult(keywords=top_keywords, vectorizer=vectorizer, matrix=X)


def extract_keywords_tfidf_per_doc(
        corpus: list[str],
        top_k: int = 5,
        **kwargs
) -> list[list[KeywordItem]]:
    """
    对每篇文档分别提取 TF-IDF top_k 关键词 （基于全局 TF-IDF）。
    返回列表：每个元素对应原 corpus 中一篇文档的 top_k 关键词列表（KeywordItem）。
    """
    tfidf_result = extract_keywords_tfidf(corpus, top_k=None, **kwargs)

    if tfidf_result.vectorizer is None:
        return []

    X = tfidf_result.matrix         # shape = (n_docs, n_features)
    feature_names = tfidf_result.vectorizer.get_feature_names_out()

    results = []

    for row in X:  # 每行是一个文档的 TF-IDF 权重
        row = row.toarray().ravel()
        idx = row.argsort()[::-1][:top_k]  # top_k 索引

        keywords = [
            KeywordItem(
                word=feature_names[i],
                weight=float(row[i]),
                count=1
            )
            for i in idx if row[i] > 0    # 忽略权重为 0 的词
        ]

        results.append(keywords)

    return results


# ---------------------------
# 词频统计（支持 n-gram）
# ---------------------------

def _generate_ngrams(words: list[str], n: int) -> list[str]:
    if n <= 1:
        return words
    # 中文常用连接方式，无空格
    return ["".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def count_word_frequency(
        corpus: list[str],
        stopwords: set[str] | None = None,
        min_len: int = 2,
        ngram_range: tuple[int, int] = (1, 1)
) -> Counter:
    """
    统计词频。支持 ngram_range，例如 (1,2) 同时统计 unigram + bigram。
    返回 Counter: {token: freq}
    """
    counter = Counter()
    for text in corpus:
        words = preprocess_text(text, stopwords=stopwords, min_len=min_len)
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngrams = words if n == 1 else _generate_ngrams(words, n)
            # 过滤长度太短的 gram
            counter.update([g for g in ngrams if len(g) >= min_len])
    return counter


# ---------------------------
# WordCloud（单图 + 按日期批量）
# ---------------------------

def _get_default_font_path() -> str:
    """
    从包资源中找到第一个可用字体；若失败，则尝试常见系统路径，最终抛出异常。
    """
    # 先尝试包内字体
    try:
        fonts_pkg = files('wordfreq_cn.data.fonts')
        for name in DEFAULT_FONT_CANDIDATES:
            candidate = fonts_pkg / name
            if candidate.exists():
                return str(candidate)
    except Exception as e:
        logger.debug(f"package fonts not available: {e}")

    # 尝试常见系统路径（简单尝试）
    sys_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:\\Windows\\Fonts\\msyh.ttf"
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
        mask: Any | None = None
) -> str | bytes:
    """
    生成单张词云图片。
    frequencies: Counter 或 dict {word: freq}
    返回输出文件路径
    """
    if not frequencies:
        raise ValueError("frequencies is empty")

    font_path = font_path or _get_default_font_path()

    wc = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        background_color=background_color,
        max_words = max_words,
        mask=mask,
    )
    if colormap:
        # WordCloud 会使用 colormap 参数通过 recolor
        wc.recolor(colormap=colormap)

    wc.generate_from_frequencies(frequencies)
    img = wc.to_image()

    # 模式1：输出文件
    if output_path:
        img.save(output_path, format="PNG")
        return output_path

    # 模式2：返回 bytes
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
        return_bytes: bool = False,     # ⭐ 新增
) -> list[str | bytes]:
    """
    根据日期生成多张词云（按 date_str key 顺序）。
    news_by_date: {"2025-01-01": [text1, text2, ...], ...}

    - 如果 return_bytes=False（默认），返回文件路径列表。
    - 如果 return_bytes=True，则返回 PNG bytes 列表。
    """
    from datetime import datetime
    import uuid

    current_date = datetime.now()  # 当前日期和时间
    font_path = font_path or _get_default_font_path()

    results: list[str] | list[bytes] = []

    for date_str, texts in sorted(news_by_date.items()):
        if not texts:
            continue
        # 如果date_str不存在则直接获取当前日期
        date_str = date_str or current_date.strftime('%Y-%m-%d')
        counter = count_word_frequency(
            texts,
            stopwords=stopwords,
            min_len=min_len,
            ngram_range=ngram_range
        )

        if not counter:
            continue

        # ⭐ 模式1：返回 bytes（不写文件）
        if return_bytes:
            img_bytes = _generate_wordcloud(
                counter,
                output_path=None,   # <- 关键
                font_path=font_path,
                width=width,
                height=height,
                max_words=max_words,
                background_color=background_color,
            )
            results.append(img_bytes)
            continue

        # ⭐ 模式2：写文件（原始行为）
        output_dir_final = (
            f"{output_dir}{date_str}"
            if output_dir else
            f"wordclouds/{date_str}"
        )
        os.makedirs(output_dir_final, exist_ok=True)

        out_file = os.path.join(
            output_dir_final,
            f"wordcloud_{uuid.uuid4().hex}.png"
        )

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



# ---------------------------
# Unified high-level interface
# ---------------------------

def extract_keywords(
        data: str | list[str],
        method: str = "tfidf",
        top_k: int = 20,
        stopwords: set[str] | None = None,
        **kwargs
) -> list[KeywordItem] | list[list[KeywordItem]]:
    """
    统一关键词提取接口：
      - method = "tfidf" -> expects data: list[str] (corpus) and returns TfIdfResult.keywords
      - method = "tfidf_per_doc" -> expects data: list[str] and returns list of per-doc keywords

    kwargs 会传递给对应的子函数（例如 ngram_range, min_df 等）
    """
    method = method.lower()
    if method == "tfidf":
        if not isinstance(data, list):
            raise TypeError("tfidf requires corpus list[str] as input")
        res = extract_keywords_tfidf(data, top_k=top_k, stopwords=stopwords, **kwargs)
        return res.keywords
    elif method == "tfidf_per_doc":
        if not isinstance(data, list):
            raise TypeError("tfidf_per_doc requires corpus list[str] as input")
        return extract_keywords_tfidf_per_doc(data, top_k=top_k, stopwords=stopwords, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: tfidf, tfidf_per_doc")
