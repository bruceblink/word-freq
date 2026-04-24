# wordfreq_cn/cli.py

import argparse
import json
import os
from typing import Iterator

from .core import (
    extract_keywords_tfidf,
    count_word_frequency,
    generate_trend_wordcloud,
    load_stopwords
)


# ============================================================
# 工具函数
# ============================================================

def _iter_input_file_lines(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text


def load_news(args):
    """从 --news 或 --input-file 加载文本"""
    if args.news:
        return args.news
    if args.input_file:
        return list(_iter_input_file_lines(args.input_file))
    raise ValueError("需要提供 --news 或 --input-file")


# ============================================================
# 子命令对应函数（只包装 core）
# ============================================================

def run_tfidf(args):
    news = load_news(args)
    stopwords = load_stopwords(args.stopwords)

    low_memory = os.getenv("WORDFREQ_LOW_MEMORY", "").strip().lower() in {"1", "true", "yes"}
    max_features = 800 if low_memory else 2000
    ngram_range = (1, 1) if low_memory else (1, 2)

    result = extract_keywords_tfidf(
        corpus=news,
        top_k=args.topk,
        stopwords=stopwords,
        include_artifacts=False,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    # 输出 JSON 或文本
    if args.json:
        print(result.keywords_to_json())
    else:
        print("=== TF-IDF 关键词 ===")
        for keyword_item in result.keywords:  #
            print(f"{keyword_item.word}\t{keyword_item.weight:.4f}")


def run_wordfreq(args):
    stopwords = load_stopwords(args.stopwords)
    news = args.news if args.news else _iter_input_file_lines(args.input_file)
    counter = count_word_frequency(news, stopwords)

    if args.json:
        print(json.dumps(counter, ensure_ascii=False, indent=2))
    else:
        print("\n=== 词频统计 ===")
        for w, c in counter.most_common(args.topk):
            print(f"{w}\t{c}")


def run_wordcloud(args):
    """
    - 默认：生成多张趋势词云，输出文件名
    - 加 --bin：输出单张趋势词云 PNG bytes 到 stdout
    """
    import sys
    from collections import defaultdict

    news = load_news(args)
    stopwords = load_stopwords(args.stopwords)

    # 按日期分组
    news_by_date = defaultdict(list)
    for i, text in enumerate(news):
        news_by_date[f"day{i + 1}"].append(text)

    if args.bin:
        # binary 模式：返回 bytes list
        print("正在生成趋势词云图 byte...", file=sys.stderr)  # 提示信息写 stderr
        files = generate_trend_wordcloud(
            news_by_date,
            stopwords=stopwords,
            font_path=getattr(args, "font_path", None),
            return_bytes=True
        )
        # 写入 stdout.buffer
        for item in files:
            sys.stdout.buffer.write(item)
        return  # 直接返回，不打印文件名

    # 非 bin 模式：返回文件路径
    print("正在生成趋势词云图...")
    files = generate_trend_wordcloud(
        news_by_date,
        stopwords=stopwords,
        font_path=getattr(args, "font_path", None),
    )

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
        p.add_argument("--json", action="store_true", help="输出 JSON 格式")
        # 新增：输出为二进制
        p.add_argument("--bin", "-b", action="store_true", help="Output PNG bytes to stdout")

    # TF-IDF
    p1 = subparsers.add_parser("tfidf", help="使用 TF-IDF 提取关键词")
    add_common(p1)
    p1.set_defaults(func=run_tfidf)

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
