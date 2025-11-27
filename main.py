# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import os
import urllib.request

# ------------------------------
# 配置
# ------------------------------
STOP_WORDS_FILE = "stopwords.txt"  # 自定义停用词，每行一个词
HIT_STOPWORDS_URL = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"
HIT_STOPWORDS_FILE = "cn_stopwords.txt"  # 哈工大中文停用词表

# ------------------------------
# 下载停用词表（如果没有）
# ------------------------------
if not os.path.exists(HIT_STOPWORDS_FILE):
    print("下载哈工大中文停用词表...")
    urllib.request.urlretrieve(HIT_STOPWORDS_URL, HIT_STOPWORDS_FILE)

# ------------------------------
# 工具函数
# ------------------------------
def load_stopwords(custom_file=None, hit_file=None):
    stopwords = set()
    # 哈工大停用词
    if hit_file and os.path.exists(hit_file):
        with open(hit_file, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word.lower())
    # 自定义停用词
    if custom_file and os.path.exists(custom_file):
        with open(custom_file, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word.lower())
    return list(stopwords)  # 返回 list，兼容 sklearn

def clean_text(text):
    """文本清洗：去除标点和多余空格"""
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def segment_text(text):
    """中文分词，兼容旧版 jieba"""
    return [w for w in jieba.cut(text) if w.strip()]

# ------------------------------
# 1. TF-IDF 高权重词
# ------------------------------
def tfidf_keywords(corpus, top_k=20, ngram_range=(1,2), stopwords=None):
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=ngram_range,
        stop_words=stopwords,
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True
    )
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    sums = X.sum(axis=0)
    word_weights = [(feature_names[i], sums[0, i]) for i in range(len(feature_names))]
    word_weights.sort(key=lambda x: x[1], reverse=True)
    return word_weights[:top_k]

# ------------------------------
# 2. TextRank 关键词
# ------------------------------
def textrank_keywords(text, top_k=20, with_weight=True):
    return jieba.analyse.textrank(text, topK=top_k, withWeight=with_weight)

# ------------------------------
# 3. 词频统计
# ------------------------------
def count_words(corpus, stopwords=None, min_len=2):
    counter = Counter()
    for text in corpus:
        words = segment_text(clean_text(text))
        if stopwords:
            words = [w.lower() for w in words if w.lower() not in stopwords and len(w) >= min_len]
        else:
            words = [w for w in words if len(w) >= min_len]
        counter.update(words)
    return counter

# ------------------------------
# 示例运行
# ------------------------------
if __name__ == "__main__":
    news_list = [
        "人工智能技术在医疗领域的应用取得突破",
        "全球气候变化加剧，联合国发布最新报告",
        "新冠疫苗接种率提升，儿童群体受关注",
        "AI 驱动的自动驾驶技术正在快速发展"
    ]

    stopwords = load_stopwords(custom_file=STOP_WORDS_FILE, hit_file=HIT_STOPWORDS_FILE)

    # 1. TF-IDF
    tfidf_res = tfidf_keywords(news_list, top_k=10, ngram_range=(1,2), stopwords=stopwords)
    print("=== TF-IDF 高权重词 ===")
    for word, weight in tfidf_res:
        print(word, f"{weight:.4f}")

    # 2. TextRank
    print("\n=== TextRank 关键词 ===")
    for text in news_list:
        kws = textrank_keywords(text, top_k=5)
        print(f"标题: {text}")
        for kw, weight in kws:
            print(f"  {kw} ({weight:.4f})")

    # 3. 词频统计
    counter = count_words(news_list, stopwords=stopwords)
    print("\n=== 词频统计 ===")
    for word, freq in counter.most_common(10):
        print(word, freq)
