# wordfreq_cn/core.py

import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import os
from wordcloud import WordCloud
from importlib.resources import files


# ============================================================
# 停用词加载
# ============================================================

def load_stopwords(custom_file=None, hit_file=None):
    """
    加载停用词列表（命中词 + 自定义词 + 内置词）

    参数:
        custom_file (str): 用户自定义停用词文件路径
        hit_file (str): 高频命中停用词文件（动态收集）

    返回:
        list[str]: 停用词（统一小写）
    """
    stopwords = set()

    # 用户动态屏蔽词
    if hit_file and os.path.exists(hit_file):
        with open(hit_file, "r", encoding="utf-8") as f:
            stopwords.update(line.strip().lower() for line in f)

    # 用户自定义停用词
    if custom_file and os.path.exists(custom_file):
        with open(custom_file, "r", encoding="utf-8") as f:
            stopwords.update(line.strip().lower() for line in f)

    # 包内置默认停用词
    stopwords_file = files('wordfreq_cn.data') / 'cn_stopwords.txt'
    with open(str(stopwords_file), "r", encoding="utf-8") as f:
        stopwords.update(line.strip().lower() for line in f)

    return list(stopwords)


# ============================================================
# 文本预处理
# ============================================================

def clean_text(text: str) -> str:
    """
    清洗文本，只保留中文、数字、英文，移除无效符号。

    参数:
        text (str): 原始文本字符串

    返回:
        str: 清洗后的文本
    """
    # 移除非字母数字及中文字符
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)
    # 合并多余空格
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def segment_text(text: str) -> list[str]:
    """
    使用 jieba 分词，返回去除空白词的序列。

    参数:
        text (str): 输入文本

    返回:
        list[str]: 分词结果
    """
    return [w for w in jieba.cut(text) if w.strip()]


# ============================================================
# TF-IDF 关键词提取
# ============================================================

def extract_keywords_tfidf(
        corpus: list[str],
        top_k=20,
        ngram_range=(1, 2),
        stopwords=None
):
    """
    使用 TF-IDF 从一组新闻文本中提取最高权重关键词。

    ⚠ TF-IDF 是"语料级"算法，输入必须是多篇文章。

    参数:
        corpus (list[str]): 新闻文本列表
        top_k (int): 返回的关键词数量
        ngram_range (tuple): n-gram 范围 (1,2 表示 unigram + bigram)
        stopwords (list[str]): 停用词列表

    返回:
        list[(str, float)]: (关键词, 权重值)
    """
    if not corpus:
        return []

    vectorizer = TfidfVectorizer(
        max_features=2000,  # 提升特征数量，新闻数据更适合 2000+
        ngram_range=ngram_range,
        stop_words=stopwords,
        token_pattern=r"(?u)\b\w+\b",  # 兼容中英文的 token pattern
        lowercase=True,
        sublinear_tf=True  # 重要优化：log(TF) 缓解高频词影响
    )

    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # 计算每个词的 TF-IDF 总权重
    weights = X.sum(axis=0)
    word_weights = [
        (feature_names[i], weights[0, i])
        for i in range(len(feature_names))
    ]

    # 按权重降序
    word_weights.sort(key=lambda x: x[1], reverse=True)
    return word_weights[:top_k]


# ============================================================
# TextRank 关键词提取
# ============================================================

def extract_keywords_textrank(text: str, top_k=20, with_weight=True):
    """
    使用 TextRank 对单文章文本提取关键字。

    参数:
        text (str): 文本
        top_k (int): 关键词数量
        with_weight (bool): 是否返回权重

    返回:
        list 或 list[(str, float)]
    """
    return jieba.analyse.textrank(text, topK=top_k, withWeight=with_weight)


# ============================================================
# 词频统计
# ============================================================

def count_word_frequency(
        corpus: list[str],
        stopwords=None,
        min_len=2
) -> Counter:
    """
    对语料进行分词并统计词频。

    参数:
        corpus (list[str]): 文本列表
        stopwords (list[str]): 停用词
        min_len (int): 单词最小长度（过滤无意义短词）

    返回:
        Counter: {词: 频次}
    """
    counter = Counter()

    for text in corpus:
        words = segment_text(clean_text(text))
        if stopwords:
            words = [
                w.lower() for w in words
                if w.lower() not in stopwords and len(w) >= min_len
            ]
        else:
            words = [w for w in words if len(w) >= min_len]

        counter.update(words)

    return counter


# ============================================================
# 按日期生成多张词云
# ============================================================

def generate_trend_wordcloud(
        news_by_date: dict,
        stopwords=None,
        min_len=2,
        output_dir="wordclouds"
) -> list[str]:
    """
    根据日期 → 新闻内容生成趋势词云图。
    例如：
        news_by_date = {
            "2025-01-01": [新闻1, 新闻2, ...],
            "2025-01-02": [...],
        }

    参数:
        news_by_date (dict): 日期到新闻文本列表的映射
        stopwords (list[str]): 停用词
        min_len (int): 最短词长度
        output_dir (str): 输出目录

    返回:
        list[str]: 生成的图片文件路径列表
    """
    file_list = []

    os.makedirs(output_dir, exist_ok=True)

    # 内置字体（思源黑体）
    font_path = files('wordfreq_cn.data.fonts') / 'SourceHanSansHWSC-VF.ttf'

    for date_str, texts in news_by_date.items():
        counter = count_word_frequency(texts, stopwords=stopwords, min_len=min_len)
        if not counter:
            continue

        wc = WordCloud(
            font_path=str(font_path),
            width=900,
            height=600,
            background_color="white"
        ).generate_from_frequencies(counter)

        out_file = os.path.join(output_dir, f"wordcloud_{date_str}.png")
        wc.to_file(out_file)
        file_list.append(out_file)

    return file_list
