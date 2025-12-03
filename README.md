# [WordFreq-CN](https://github.com/bruceblink/word-freq) [![PyPI](https://img.shields.io/pypi/v/wordfreq-cn.svg)](https://pypi.org/project/wordfreq-cn/) [![PyPI Downloads](https://img.shields.io/pypi/dm/wordfreq-cn.svg?label=PyPI%20downloads)](https://pypi.org/project/wordfreq-cn/)

**中文新闻词频分析与趋势词可视化工具**

---

## 功能

* 中文新闻标题/正文的 **TF-IDF 高频词提取**
* **词频统计**
* 按 **时间窗口生成趋势词云**
* 支持自定义停用词表，过滤中文虚词
* 可通过命令行工具 `wordfreq-cn` 直接运行
* 也可以通过`wordfreq-cn`API函数使用

---

## 安装

```bash
# 安装本地源码包（如果你有源码）
pip install .

# 或直接从 PyPI 安装
pip install wordfreq-cn
```

---

## 快速开始示例（命令行）

```bash
wordfreq-cn tfidf --news "人工智能技术在医疗领域的应用取得突破" "全球气候变化加剧" --topk 5
wordfreq-cn freq --news "人工智能技术在医疗领域的应用取得突破" --topk 10
wordfreq-cn wordcloud --news "人工智能技术在医疗领域的应用取得突破" "全球气候变化加剧"
wordfreq-cn freq --news "人工智能技术在医疗领域的应用取得突破" --json
wordfreq-cn wordcloud --news "人工智能技术在医疗领域的应用取得突破" "全球气候变化加剧" --bin
```

### 示例输出

**TF-IDF 高频词：**

```
人工智能技术 1.0000
医疗 0.8349
应用 0.6730
...
```

**词频统计：**

```
技术 2
人工智能 1
医疗 1
...
```

**json输出**

```json
{
  "人工智能技术": 1,
  "医疗": 1,
  "应用": 1,
  "突破": 1
}

```

**词云输出目录：**

```
wordclouds/wordcloud_day1.png
wordclouds/wordcloud_day2.png
```

---

## Python API 使用示例

```python
from collections import defaultdict
from wordfreq_cn import (
    extract_keywords,
    count_word_frequency,
    generate_trend_wordcloud,
    load_stopwords
)

# 示例新闻数据
news_list = [
    ("2025-11-25", "人工智能技术在医疗领域的应用取得突破"),
    ("2025-11-25", "全球气候变化加剧，联合国发布最新报告")
]

# 加载自定义停用词
stopwords = load_stopwords("stopwords.txt")

# ---------------------------
# TF-IDF 关键词提取
# ---------------------------
texts = [text for _, text in news_list]
tfidf_res = extract_keywords(texts, method="tfidf", top_k=5, stopwords=stopwords)
print("TF-IDF:", tfidf_res)

# ---------------------------
# 词频统计
# ---------------------------
counter = count_word_frequency(texts, stopwords=stopwords)
print("词频统计:", counter)

# ---------------------------
# 按日期生成趋势词云
# ---------------------------
news_by_date = defaultdict(list)
for date, text in news_list:
    news_by_date[date].append(text)

generate_trend_wordcloud(news_by_date, stopwords=stopwords) # 生成图片和存放的路径list
# 词云图片默认保存到 wordclouds/ 目录
generate_trend_wordcloud(news_by_date, stopwords=stopwords, return_bytes=True) # 返回二进制byte数据
```

---

## 快速流程图示

```
                    ┌─────────────┐
                    │ 输入新闻列表 │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    TF-IDF   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  输出关键词  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   词频统计   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  生成词云图  │
                    └─────────────┘
```

---

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_core.py -v

# 运行特定测试类
pytest tests/test_core.py::TestTFIDFKeywords -v

# 带覆盖率报告
pytest --cov=wordfreq_cn

# 生成 HTML 覆盖率报告
pytest --cov=wordfreq_cn --cov-report=html
```

---

## 文件说明

| 文件名                                 | 说明                     |
|-------------------------------------|------------------------|
| `wordfreq_cn/`                      | Python 包目录，包含核心逻辑和 CLI |
| `wordfreq_cn/data/stopwords.txt`    | 可选自定义停用词文件             |
| `wordfreq_cn/data/cn_stopwords.txt` | 哈工大中文停用词表              |
| `wordfreq_cn/data/fonts/`           | 中文字体文件（如思源黑体）用于生成词云    |
| `wordclouds/`                       | 默认存放生成的词云图片            |
| `tests/`                            | 单元测试代码                 |

---

## 注意事项

* 新闻量大时，可调整 `extract_keywords` 的 `top_k` 或 TF-IDF 的 `max_features` 参数
* 停用词表建议包含常用虚词（如“的”“在”“是”）以获得更干净的词频统计结果
* 安装后直接使用 `wordfreq-cn` 命令，无需手动运行 `python cli.py`

