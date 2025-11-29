# tests/test_core.py

import os
import tempfile
from collections import Counter

from wordfreq_cn.core import (
    TfIdfResult,
    extract_keywords_tfidf,
    extract_keywords_textrank,
    count_word_frequency,
    load_stopwords,
    generate_trend_wordcloud
)


class TestTFIDFKeywords:
    def test_tfidf_basic(self, sample_news):
        """测试 TF-IDF 基础功能"""
        result = extract_keywords_tfidf(sample_news, top_k=5)

        assert isinstance(result, TfIdfResult)
        assert len(result.keywords) <= 5
        for key_item in result.keywords:
            assert isinstance(key_item.word, str)
            assert isinstance(key_item.weight, float)
            assert isinstance(key_item.count, int|None)
            assert key_item.weight > 0

    def test_tfidf_with_stopwords(self, sample_news, stopwords_file):
        """测试 TF-IDF 停用词过滤"""
        stopwords = load_stopwords(custom_file=stopwords_file)
        result = extract_keywords_tfidf(sample_news, stopwords=stopwords, top_k=10)

        stopwords_list = ["的", "了", "是", "在", "与"]
        for keyword_item in result.keywords:
            assert keyword_item.word not in stopwords_list

    def test_tfidf_empty_input(self):
        """测试空输入"""
        result = extract_keywords_tfidf([], top_k=5)
        assert result.keywords == []

    def test_tfidf_single_document(self):
        """测试单文档输入"""
        result = extract_keywords_tfidf(["单一文档测试"], top_k=3)
        assert len(result.keywords) <= 3

class TestTextRankKeywords:
    def test_textrank_basic(self, sample_text):
        """测试 TextRank 基础功能"""
        result = extract_keywords_textrank(sample_text, top_k=5)

        assert isinstance(result, list)
        assert len(result) <= 5
        for keyword_item in result:
            assert isinstance(keyword_item.word, str)
            assert isinstance(keyword_item.weight, float)

    def test_textrank_empty_text(self):
        """测试空文本"""
        result = extract_keywords_textrank("", top_k=5)
        assert result == []

    def test_textrank_short_text(self):
        """测试短文本"""
        result = extract_keywords_textrank("人工智能", top_k=5)
        assert len(result) <= 1

class TestCountWords:
    def test_count_words_basic(self, sample_news):
        """测试词频统计基础功能"""
        counter = count_word_frequency(sample_news)

        assert isinstance(counter, Counter)
        assert len(counter) > 0

        common_words = counter.most_common(3)
        for word, count in common_words:
            assert isinstance(word, str)
            assert isinstance(count, int)
            assert count > 0

    def test_count_words_with_stopwords(self, sample_news, stopwords_file):
        """测试带停用词的词频统计"""
        stopwords = load_stopwords(custom_file=stopwords_file)
        counter = count_word_frequency(sample_news, stopwords=stopwords)

        stopwords_list = ["的", "了", "是"]
        for stopword in stopwords_list:
            assert counter.get(stopword, 0) == 0

    def test_count_words_empty(self):
        """测试空输入"""
        counter = count_word_frequency([])
        assert len(counter) == 0

class TestLoadStopwords:
    def test_load_default_stopwords(self):
        """测试加载默认停用词"""
        stopwords = load_stopwords()
        assert isinstance(stopwords, list)
        assert len(stopwords) > 0
        assert "的" in stopwords
        assert "了" in stopwords

    def test_load_custom_stopwords(self, stopwords_file):
        """测试加载自定义停用词"""
        stopwords = load_stopwords(custom_file=stopwords_file)
        assert isinstance(stopwords, list)
        assert "的" in stopwords
        assert "了" in stopwords
        assert "是" in stopwords

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        stopwords = load_stopwords(custom_file="nonexistent.txt")
        assert isinstance(stopwords, list)
        assert len(stopwords) > 0

class TestGenerateTrendWordcloud:
    def test_generate_wordcloud_basic(self, mock_news_by_date, tmp_path):
        """测试生成词云基础功能"""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = generate_trend_wordcloud(
                mock_news_by_date,
                stopwords=list(),
                output_dir=temp_dir
            )

            assert isinstance(files, list)
            for file_path in files:
                assert os.path.exists(file_path)
                assert file_path.endswith('.png')

    def test_generate_wordcloud_custom_dir(self, mock_news_by_date, tmp_path):
        """测试自定义输出目录"""
        custom_dir = tmp_path / "wordclouds"
        custom_dir.mkdir()

        files = generate_trend_wordcloud(
            mock_news_by_date,
            stopwords=list(),
            output_dir=str(custom_dir)
        )

        for file_path in files:
            assert str(custom_dir) in file_path
            assert os.path.exists(file_path)

    def test_generate_wordcloud_empty_data(self, tmp_path):
        """测试空数据"""
        files = generate_trend_wordcloud(
            {},
            stopwords=set(),
            output_dir=str(tmp_path)
        )
        assert files == []
