# wordfreq_cn/cli.py
import argparse
from .core import extract_keywords_tfidf, extract_keywords_textrank, count_word_frequency, generate_trend_wordcloud, load_stopwords
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="新闻词频分析工具")
    parser.add_argument("--news", nargs="+", help="新闻标题或正文列表", required=True)
    parser.add_argument("--stopwords", type=str, help="自定义停用词文件", default=None)
    parser.add_argument("--topk", type=int, help="关键词数量", default=10)
    args = parser.parse_args()

    stopwords = load_stopwords(custom_file=args.stopwords)

    # TF-IDF
    tfidf_res = extract_keywords_tfidf(args.news, top_k=args.topk, stopwords=stopwords)
    print("=== TF-IDF 高权重词 ===")
    for word, weight in tfidf_res:
        print(word, f"{weight:.4f}")

    # TextRank
    print("\n=== TextRank 关键词 ===")
    for text in args.news:
        kws = extract_keywords_textrank(text, top_k=args.topk)
        print(f"标题: {text}")
        for kw, weight in kws:
            print(f"  {kw} ({weight:.4f})")

    # 词频统计
    counter = count_word_frequency(args.news, stopwords=stopwords)
    print("\n=== 词频统计 ===")
    for word, freq in counter.most_common(args.topk):
        print(word, freq)

    # 简单示例：按日期生成词云
    news_by_date = defaultdict(list)
    for i, text in enumerate(args.news):
        news_by_date[f"day{i+1}"].append(text)
    files = generate_trend_wordcloud(news_by_date, stopwords=stopwords)
    print(files)
