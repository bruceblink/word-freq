# wordfreq_cn/core.py
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import re
import os
import urllib.request
from wordcloud import WordCloud

# 停用词加载
def load_stopwords(custom_file=None, hit_file=None):
    stopwords = set()
    if hit_file and os.path.exists(hit_file):
        with open(hit_file, "r", encoding="utf-8") as f:
            for line in f:
                stopwords.add(line.strip().lower())
    if custom_file and os.path.exists(custom_file):
        with open(custom_file, "r", encoding="utf-8") as f:
            for line in f:
                stopwords.add(line.strip().lower())
    # 默认的stopwords list
    with open("cn_stopwords.txt", "r", encoding="utf-8") as f:
        for line in f:
            stopwords.add(line.strip().lower())
    return list(stopwords)

# 文本处理函数
def clean_text(text):
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def segment_text(text):
    return [w for w in jieba.cut(text) if w.strip()]

# TF-IDF
def tfidf_keywords(corpus, top_k=20, ngram_range=(1,2), stopwords=None):
    if not corpus:
        return []
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

# TextRank
def textrank_keywords(text, top_k=20, with_weight=True):
    return jieba.analyse.textrank(text, topK=top_k, withWeight=with_weight)

# 词频统计
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

# 时间窗口词云
def generate_trend_wordcloud(news_by_date, stopwords=None, min_len=2, output_dir="wordclouds") -> list[str]:
    file_list = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for date_str, texts in news_by_date.items():
        counter = count_words(texts, stopwords=stopwords, min_len=min_len)
        if not counter:
            continue
        wc = WordCloud(font_path="simhei.ttf", width=800, height=600, background_color="white")
        wc.generate_from_frequencies(counter)
        out_file = os.path.join(output_dir, f"wordcloud_{date_str}.png")
        wc.to_file(out_file)
        file_list.append(out_file)
    return file_list
