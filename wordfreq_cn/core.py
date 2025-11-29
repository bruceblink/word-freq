"""
wordfreq_cn.core
----------------
优化后的关键词提取与词云工具集合，基于 jieba + sklearn + wordcloud。

主要功能：
- 停用词加载 (load_stopwords)
- 文本清洗 (clean_text) 与预处理 (preprocess_text)
- 分词（缓存）(segment_text)
- 全局 / per-doc TF-IDF 关键词提取 (extract_keywords_tfidf, extract_keywords_tfidf_per_doc)
- TextRank 关键词提取 (extract_keywords_textrank)
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
from typing import Any

import jieba
import jieba.analyse
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
            with open(path, "r", encoding="utf-8") as f:
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
    基础清洗：去掉非中文/英文/数字字符、合并空白，并可选删除 URL / email / 数字（或保留数字）。
    返回小写形式（英文）。
    """
    if not text:
        return ""
    s = text

    if remove_urls:
        # 简单的 url pattern 去除
        s = re.sub(r"https?://\S+|www\.\S+", " ", s)

    if remove_emails:
        s = re.sub(r"\S+@\S+", " ", s)

    # 去掉除了中文/字母/数字之外的字符
    s = re.sub(r"[^\w\u4e00-\u9fff]", " ", s)

    if remove_digits:
        s = re.sub(r"\d+", " ", s)

    s = re.sub(r"\s+", " ", s).strip()

    # 对英文小写化（中文不受影响）
    s = s.lower()
    return s


def preprocess_text(text: str, stopwords: list[str] | None = None, min_len: int = 2) -> list[str]:
    """
    预处理管道：clean_text -> 分词 -> 停用词 & 长度过滤
    返回词列表（原始词形，不再小写中文）
    """
    cleaned = clean_text(text)
    words = segment_text(cleaned)
    if stopwords:
        sw = set(w.lower() for w in stopwords)
        words = [w for w in words if w and w.lower() not in sw and len(w) >= min_len]
    else:
        words = [w for w in words if w and len(w) >= min_len]
    return words


# ---------------------------
# Segment (jieba) with caching
# ---------------------------

@lru_cache(maxsize=65536)
def _cached_cut(text: str) -> tuple[str, ...]:
    """
    内部缓存分词结果（不可变 tuple），减少重复分词成本。
    """
    return tuple(jieba.cut(text))


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
        top_k: int = 20,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
        stopwords: set[str] | None = None,
        max_features: int = DEFAULT_MAX_FEATURES,
        min_df: int = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        token_pattern: str = DEFAULT_TOKEN_PATTERN,
) -> TfIdfResult:
    """
    全局 TF-IDF：基于整个语料库计算 TF-IDF，然后返回 top_k 全局关键词 + 其总权重与出现次数。

    返回 TfIdfResult:
      - keywords: list of KeywordItem (word, weight, count)
      - vectorizer: 训练好的 TfidfVectorizer（便于后续 transform）
      - matrix: 稀疏矩阵 X (n_docs, n_features)
    """
    if not corpus:
        return TfIdfResult([], None, None)

    # ----------------------------
    # 修复单文档时 max_df 问题
    # ----------------------------
    n_docs = len(corpus)
    adjusted_max_df = max_df
    if n_docs == 1:
        # 单文档时 max_df 不能小于 min_df
        adjusted_max_df = 1.0

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=list(stopwords),
        token_pattern=token_pattern,
        lowercase=True,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=adjusted_max_df
    )

    X = vectorizer.fit_transform(corpus)  # shape: (n_docs, n_features)
    feature_names = vectorizer.get_feature_names_out()

    # 按列求和得到每个特征在所有文档中的 TF-IDF 总权重
    weights_array = np.asarray(X.sum(axis=0)).ravel()  # shape: (n_features,)

    # 统计每个 token 在多少个文档中出现（非零计数）
    doc_counts = np.asarray((X > 0).sum(axis=0)).ravel()

    # 包装结果
    kw_items = [KeywordItem(word=feature_names[i], weight=float(weights_array[i]), count=int(doc_counts[i]))
                for i in range(len(feature_names))]

    # 排序
    kw_items.sort(key=lambda x: x.weight, reverse=True)

    # 截断 top_k
    top_keywords = kw_items[:top_k]

    return TfIdfResult(keywords=top_keywords, vectorizer=vectorizer, matrix=X)


def extract_keywords_tfidf_per_doc(
        corpus: list[str],
        top_k: int = 10,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
        stopwords: set[str] | None = None,
        max_features: int = DEFAULT_MAX_FEATURES,
        min_df: int = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        token_pattern: str = DEFAULT_TOKEN_PATTERN,
) -> list[list[KeywordItem]]:
    """
    对每篇文档分别提取 TF-IDF top_k 关键词。
    返回列表：每个元素对应原 corpus 中一篇文档的 top_k 关键词列表（KeywordItem）。
    """
    if not corpus:
        return []

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stopwords,
        token_pattern=token_pattern,
        lowercase=True,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df
    )

    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    results: list[list[KeywordItem]] = []
    # 对每一行（文档）寻找 top_k 非零特征
    for doc_idx in range(X.shape[0]):
        row = X[doc_idx]  # sparse row
        if row.nnz == 0:
            results.append([])
            continue
        # 排序取 top_k
        idx_sorted = np.argsort(row.data)[::-1][:top_k]
        doc_keywords = [KeywordItem(word=feature_names[row.indices[i]], weight=float(row.data[i])) for i in idx_sorted]
        results.append(doc_keywords)

    return results


# ---------------------------
# TextRank 关键词（增强：预处理 + stopwords + POS 过滤）
# ---------------------------

def extract_keywords_textrank(
        text: str,
        top_k: int = 20,
        with_weight: bool = True,
        stopwords: set[str] | None = None,
        min_len: int = 2,
        allow_pos: tuple[str, ...] = ("ns", "n", "vn", "v")
) -> list[str | KeywordItem]:
    """
    使用 TextRank 提取单文档关键词，内部做清洗与 stopwords 过滤。

    参数:
        text: 原始文本
        top_k: 返回数量
        with_weight: 是否返回权重
        stopwords: 停用词列表
        min_len: 词最小长度
        allow_pos: 允许的词性（jieba 的 POS tag）

    返回:
        若 with_weight=True, 返回 List[KeywordItem]; 否则返回 List[str]
    """
    if not text:
        return []

    cleaned = clean_text(text)
    # 先直接调用 textrank（它内部有分词），但我们会过滤结果
    try:
        # jieba.analyse.textrank 支持 allowPOS 参数
        candidates = jieba.analyse.textrank(
            cleaned, topK=top_k * 3, withWeight=True, allowPOS=allow_pos
        )
    except TypeError:
        # 兼容旧版本 jieba 不支持 allowPOS 的情况
        candidates = jieba.analyse.textrank(cleaned, topK=top_k * 3, withWeight=True)

    results: list[str | KeywordItem] = []
    sw = set(w.lower() for w in stopwords) if stopwords else set()
    for word, weight in candidates:
        w = word.strip()
        if not w or len(w) < min_len or w.lower() in sw:
            continue
        if with_weight:
            results.append(KeywordItem(word=w, weight=float(weight), count=None))
        else:
            results.append(w)
        if len(results) >= top_k:
            break

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


def generate_wordcloud(
        frequencies: Counter,
        output_path: str,
        font_path: str | None = None,
        width: int = 900,
        height: int = 600,
        background_color: str = "white",
        colormap: str | None = None,
        mask: Any | None = None
) -> str:
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
        mask=mask,
    )
    if colormap:
        # WordCloud 会使用 colormap 参数通过 recolor
        wc.recolor(colormap=colormap)

    wc.generate_from_frequencies(frequencies)
    wc.to_file(output_path)
    return output_path


def generate_trend_wordcloud(
        news_by_date: dict[str, list[str]],
        stopwords: set[str] | None = None,
        min_len: int = 2,
        ngram_range: tuple[int, int] = (1, 1),
        output_dir: str = "wordclouds",
        font_path: str | None = None,
        width: int = 900,
        height: int = 600,
        background_color: str = "white"
) -> list[str]:
    """
    根据日期生成多张词云（按 date_str key 顺序）。
    news_by_date: {"2025-01-01": [text1, text2, ...], ...}
    返回生成的文件路径列表（按输入 dict 的 key 顺序）
    """
    os.makedirs(output_dir, exist_ok=True)
    font_path = font_path or _get_default_font_path()
    file_list: list[str] = []
    for date_str, texts in sorted(news_by_date.items()):
        if not texts:
            continue
        counter = count_word_frequency(texts, stopwords=stopwords, min_len=min_len, ngram_range=ngram_range)
        if counter:
            out_file = os.path.join(output_dir, f"wordcloud_{date_str}.png")
            generate_wordcloud(counter,
                               out_file,
                               font_path=font_path,
                               width=width,
                               height=height,
                               background_color=background_color)
            file_list.append(out_file)
    return file_list


# ---------------------------
# Unified high-level interface
# ---------------------------

def extract_keywords(
        data: str | list[str],
        method: str = "textrank",
        top_k: int = 20,
        stopwords: set[str] | None = None,
        **kwargs
) -> list[KeywordItem] | list[list[KeywordItem]]:
    """
    统一关键词提取接口：
      - method = "textrank" -> expects data: str (single doc)
      - method = "tfidf" -> expects data: list[str] (corpus) and returns TfIdfResult.keywords
      - method = "tfidf_per_doc" -> expects data: list[str] and returns list of per-doc keywords

    kwargs 会传递给对应的子函数（例如 ngram_range, min_df 等）
    """
    method = method.lower()
    if method == "textrank":
        if not isinstance(data, str):
            raise TypeError("textrank requires a single text string as input")
        return extract_keywords_textrank(data, top_k=top_k, stopwords=stopwords, **kwargs)
    elif method == "tfidf":
        if not isinstance(data, list):
            raise TypeError("tfidf requires corpus list[str] as input")
        res = extract_keywords_tfidf(data, top_k=top_k, stopwords=stopwords, **kwargs)
        return res.keywords
    elif method == "tfidf_per_doc":
        if not isinstance(data, list):
            raise TypeError("tfidf_per_doc requires corpus list[str] as input")
        return extract_keywords_tfidf_per_doc(data, top_k=top_k, stopwords=stopwords, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: textrank, tfidf, tfidf_per_doc")
