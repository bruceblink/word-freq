import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_news():
    return [
        "人工智能技术快速发展",
        "机器学习在医疗领域的应用",
        "深度学习改变人工智能发展",
        "大数据与人工智能结合"
    ]

@pytest.fixture
def sample_text():
    return "自然语言处理是人工智能的重要分支，深度学习推动了自然语言处理的发展"

@pytest.fixture
def stopwords_file(tmp_path):
    stopwords_content = "的\n了\n是\n我\n有\n和\n在\n与\n"
    stopwords_file = tmp_path / "cn_stopwords.txt"
    stopwords_file.write_text(stopwords_content, encoding='utf-8')
    return str(stopwords_file)

@pytest.fixture
def mock_news_by_date():
    return {
        "day1": ["新闻一内容", "新闻一标题"],
        "day2": ["新闻二内容", "新闻二标题"],
        "day3": ["新闻三内容", "新闻三标题"]
    }