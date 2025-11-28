# wordfreq_cn/cli.py

import argparse
from collections import defaultdict
from .core import (
    extract_keywords_tfidf,
    extract_keywords_textrank,
    count_word_frequency,
    generate_trend_wordcloud,
    load_stopwords
)

# ============================================================
# 工具函数
# ============================================================

def load_news(args):
    """从 --news 或 --input-file 加载文本"""
    if args.news:
        return args.news
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError("需要提供 --news 或 --input-file")


def print_kw_list(title, kw_list):
    print(f"\n=== {title} ===")
    for w, score in kw_list:
        print(f"{w}\t{score:.4f}")


# ============================================================
# 子命令对应函数（只包装 core）
# ============================================================

def run_tfidf(args):
    news = load_news(args)
    stopwords = load_stopwords(args.stopwords)

    result = extract_keywords_tfidf(
        corpus=news,
        top_k=args.topk,
        stopwords=stopwords
    )
    print("TF-IDF 关键词", result.keywords)


def run_textrank(args):
    news = load_news(args)

    print("\n=== TextRank 关键词（逐条新闻） ===")
    for text in news:
        kws = extract_keywords_textrank(
            text=text,
            top_k=args.topk,
            with_weight=True
        )
        print(f"\n【新闻】{text[:40]}...")
        for w, s in kws:
            print(f"{w}\t{s:.4f}")


def run_wordfreq(args):
    news = load_news(args)
    stopwords = load_stopwords(args.stopwords)
    counter = count_word_frequency(news, stopwords)

    print("\n=== 词频统计 ===")
    for w, c in counter.most_common(args.topk):
        print(f"{w}\t{c}")


def run_wordcloud(args):
    news = load_news(args)
    stopwords = load_stopwords(args.stopwords)

    news_by_date = defaultdict(list)
    for i, text in enumerate(news):
        news_by_date[f"day{i+1}"].append(text)

    print("正在生成词云图...")
    files = generate_trend_wordcloud(news_by_date, stopwords=stopwords)

    print("\n生成的文件：")
    for f in files:
        print(" -", f)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="新闻词频分析工具 wordfreq-cn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 追加公共参数
    def add_common(p):
        p.add_argument("--news", nargs="+", help="新闻文本")
        p.add_argument("--input-file", type=str, help="从文件加载新闻")
        p.add_argument("--stopwords", type=str, help="自定义停用词")
        p.add_argument("--topk", type=int, default=20, help="关键词数量")

    # TF-IDF
    p1 = subparsers.add_parser("tfidf", help="使用 TF-IDF 提取关键词")
    add_common(p1)
    p1.set_defaults(func=run_tfidf)

    # TextRank
    p2 = subparsers.add_parser("textrank", help="使用 TextRank 提取关键词")
    add_common(p2)
    p2.set_defaults(func=run_textrank)

    # Word Frequency
    p3 = subparsers.add_parser("freq", help="统计词频")
    add_common(p3)
    p3.set_defaults(func=run_wordfreq)

    # WordCloud
    p4 = subparsers.add_parser("wordcloud", help="生成词云图")
    add_common(p4)
    p4.set_defaults(func=run_wordcloud)

    args = parser.parse_args()
    args.func(args)
