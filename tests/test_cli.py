from io import StringIO, BytesIO
from unittest.mock import patch, MagicMock

import pytest

from wordfreq_cn import KeywordItem, TfIdfResult
from wordfreq_cn.cli import main


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

    def test_cli_tfidf(self):
        """测试 TF-IDF 子命令"""
        with patch('wordfreq_cn.cli.load_stopwords') as mock_load_stopwords, \
                patch('wordfreq_cn.cli.extract_keywords_tfidf') as mock_extract_tfidf, \
                patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            mock_load_stopwords.return_value = {"的", "了", "是"}
            top_keywords = [KeywordItem("人工智能", 0.8), KeywordItem("技术", 0.6)]
            mock_extract_tfidf.return_value = TfIdfResult(keywords=top_keywords, vectorizer=None, matrix=None)

            test_args = [
                'wordfreq-cn', 'tfidf',
                '--news', '新闻一', '新闻二',
                '--topk', '5'
            ]

            with patch('sys.argv', test_args):
                main()
                output = mock_stdout.getvalue()

            assert "TF-IDF 关键词" in output
            assert "人工智能" in output

    def test_cli_freq(self):
        """测试词频统计子命令"""
        with patch('wordfreq_cn.cli.load_stopwords') as mock_load_stopwords, \
                patch('wordfreq_cn.cli.count_word_frequency') as mock_count_words, \
                patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            mock_load_stopwords.return_value = {"的", "了"}
            mock_counter = MagicMock()
            mock_counter.most_common.return_value = [("人工智能", 5), ("技术", 3)]
            mock_count_words.return_value = mock_counter

            test_args = [
                'wordfreq-cn', 'freq',
                '--news', '新闻X', '新闻Y',
                '--topk', '2'
            ]

            with patch('sys.argv', test_args):
                main()
                output = mock_stdout.getvalue()

            assert "词频统计" in output
            assert "人工智能" in output

    def test_cli_wordcloud(self):
        """测试词云生成子命令"""
        with patch('wordfreq_cn.cli.load_stopwords') as mock_load_stopwords, \
                patch('wordfreq_cn.cli.generate_trend_wordcloud') as mock_gen_wc, \
                patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            mock_load_stopwords.return_value = {"的", "了"}
            mock_gen_wc.return_value = ["wordcloud_day1.png"]

            test_args = [
                'wordfreq-cn', 'wordcloud',
                '--news', '新闻1', '新闻2',
                '--topk', '5'
            ]

            with patch('sys.argv', test_args):
                main()
                output = mock_stdout.getvalue()

            assert "正在生成趋势词云图" in output
            assert "wordcloud_day1.png" in output

    def test_cli_with_stopwords_file(self):
        """测试带停用词文件的 CLI"""
        with patch('wordfreq_cn.cli.load_stopwords') as mock_load_stopwords, \
                patch('wordfreq_cn.cli.extract_keywords_tfidf'), \
                patch('wordfreq_cn.cli.count_word_frequency'), \
                patch('wordfreq_cn.cli.generate_trend_wordcloud'):
            mock_load_stopwords.return_value = {"的", "了"}

            test_args = [
                'wordfreq-cn', 'tfidf',
                '--news', '测试新闻',
                '--stopwords', 'custom_stopwords.txt'
            ]

            with patch('sys.argv', test_args):
                main()

            mock_load_stopwords.assert_called_once_with('custom_stopwords.txt')

    def test_cli_json_output(self):
        """测试 JSON 输出模式"""
        with patch('wordfreq_cn.cli.load_stopwords') as mock_load_stopwords, \
                patch('wordfreq_cn.cli.extract_keywords_tfidf') as mock_extract_tfidf, \
                patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            mock_load_stopwords.return_value = {"的", "了"}
            top_keywords = [KeywordItem("人工智能", 0.8), KeywordItem("技术", 0.6)]
            mock_extract_tfidf.return_value = TfIdfResult(keywords=top_keywords, vectorizer=None, matrix=None)

            test_args = [
                'wordfreq-cn', 'tfidf',
                '--news', '测试新闻',
                '--topk', '2',
                '--json'
            ]

            with patch('sys.argv', test_args):
                main()
                output = mock_stdout.getvalue()

            assert '"人工智能"' in output
            assert '"技术"' in output
            assert '"weight": 0.8' in output
            assert '"weight": 0.6' in output

    def test_cli_wordcloud_bin(self):
        """测试 wordcloud --bin 输出 PNG bytes"""
        with patch("wordfreq_cn.cli.load_stopwords") as mock_load_stopwords, \
             patch("wordfreq_cn.cli.generate_trend_wordcloud") as mock_gen_wc, \
             patch("sys.stdout", new_callable=lambda: StdoutBuffer()) as mock_stdout:

            mock_load_stopwords.return_value = set()
            mock_gen_wc.return_value = [
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00"
            ]

            test_args = [
                "wordfreq-cn", "wordcloud",
                "--news", "新闻1", "新闻2",
                "--bin"
            ]

            with patch("sys.argv", test_args):
                main()

            output = mock_stdout.buffer.getvalue()
            assert output.startswith(b"\x89PNG\r\n\x1a\n")
            assert "生成".encode("utf-8") not in output
            assert ".png".encode("utf-8") not in output


class StdoutBuffer:
    def __init__(self):
        self.buffer = BytesIO()
    def write(self, s):
        self.buffer.write(s)
    def flush(self):
        pass