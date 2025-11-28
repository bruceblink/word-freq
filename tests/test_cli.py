import pytest
from wordfreq_cn.cli import main
import argparse
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

class TestCLI:
    def test_cli_help(self):
        """测试帮助信息"""
        with patch('sys.argv', ['wordfreq-cn', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_cli_missing_required_args(self):
        """测试缺少必需参数"""
        with patch('sys.argv', ['wordfreq-cn']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0

    @patch('wordfreq_cn.cli.tfidf_keywords')
    @patch('wordfreq_cn.cli.textrank_keywords')
    @patch('wordfreq_cn.cli.count_words')
    @patch('wordfreq_cn.cli.generate_trend_wordcloud')
    @patch('wordfreq_cn.cli.load_stopwords')
    def test_cli_integration(self, mock_load_stopwords, mock_generate_wordcloud,
                             mock_count_words, mock_textrank_keywords, mock_tfidf_keywords):
        """测试 CLI 集成"""
        # 设置 mock 返回值
        mock_load_stopwords.return_value = {"的", "了", "是"}
        mock_tfidf_keywords.return_value = [("人工智能", 0.8), ("技术", 0.6)]
        mock_textrank_keywords.return_value = [("机器学习", 0.9), ("深度", 0.7)]
        mock_counter = MagicMock()
        mock_counter.most_common.return_value = [("人工智能", 5), ("技术", 3)]
        mock_count_words.return_value = mock_counter
        mock_generate_wordcloud.return_value = ["wordcloud_day1.png"]

        # 模拟命令行参数
        test_args = [
            'wordfreq-cn',
            '--news', '新闻一', '新闻二', '新闻三',
            '--topk', '5'
        ]

        with patch('sys.argv', test_args):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()

                output = mock_stdout.getvalue()

                # 验证输出包含预期内容
                assert "TF-IDF 高权重词" in output
                assert "TextRank 关键词" in output
                assert "词频统计" in output
                assert "人工智能" in output

        # 验证函数调用
        mock_tfidf_keywords.assert_called_once()
        mock_count_words.assert_called_once()
        assert mock_textrank_keywords.call_count == 3  # 每个新闻调用一次

    @patch('wordfreq_cn.cli.load_stopwords')
    def test_cli_with_stopwords_file(self, mock_load_stopwords):
        """测试带停用词文件的 CLI"""
        mock_load_stopwords.return_value = {"的", "了"}

        test_args = [
            'wordfreq-cn',
            '--news', '测试新闻',
            '--stopwords', 'custom_stopwords.txt'
        ]

        with patch('sys.argv', test_args):
            with patch('wordfreq_cn.cli.tfidf_keywords') as mock_tfidf:
                with patch('wordfreq_cn.cli.textrank_keywords'):
                    with patch('wordfreq_cn.cli.count_words'):
                        with patch('wordfreq_cn.cli.generate_trend_wordcloud'):
                            main()

        # 验证停用词加载函数被正确调用
        mock_load_stopwords.assert_called_once_with(custom_file='custom_stopwords.txt')