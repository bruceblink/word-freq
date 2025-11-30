# wordfreq_cn/__init__.py

"""
wordfreq_cn
===========

新闻文本关键词提取、词频统计与词云生成工具集。
基于 jieba + sklearn + wordcloud。

模块结构：
- core.py: 核心算法实现
- cli.py: 命令行接口
"""

# CLI 主入口
from .cli import main as cli_main
# 暴露核心函数和类
from .core import (
    extract_keywords,
    extract_keywords_tfidf,
    extract_keywords_tfidf_per_doc,
    count_word_frequency,
    generate_trend_wordcloud,
    load_stopwords,
    KeywordItem,
    TfIdfResult,
    clean_text,
    preprocess_text,
    segment_text
)

__all__ = [
    "extract_keywords",
    "extract_keywords_tfidf",
    "extract_keywords_tfidf_per_doc",
    "count_word_frequency",
    "generate_trend_wordcloud",
    "load_stopwords",
    "KeywordItem",
    "TfIdfResult",
    "clean_text",
    "preprocess_text",
    "segment_text",
    "cli_main"
]
