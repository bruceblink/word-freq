# wordfreq_cn/cli.py

import argparse
import json

from .core import (
    extract_keywords_tfidf,
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
        with open(args.input_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError("需要提供 --news 或 --input-file")


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
    # 输出 JSON 或文本
    if args.json:
        print(result.keywords_to_json())
    else:
        print("=== TF-IDF 关键词 ===")
        for keyword_item in result.keywords:  #
            print(f"{keyword_item.word}\t{keyword_item.weight:.4f}")


def run_wordfreq(args):
    news = load_news(args)
    stopwords = load_stopwords(args.stopwords)
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
