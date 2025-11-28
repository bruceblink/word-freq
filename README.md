# [新闻词频分析与趋势词可视化](https://github.com/bruceblink/word-freq)

## 功能

* 中文新闻标题/正文的 **TF-IDF 高频词提取**
* 基于 **TextRank** 的关键词提取
* **词频统计**
* 按 **时间窗口生成趋势词云**
* 支持自定义停用词表，过滤中文虚词
* 可直接通过命令行工具 `wordfreq-cn` 运行

---

## 安装

```bash
# 安装 Python 依赖
pip install jieba scikit-learn wordcloud matplotlib

# 安装本地包（如果使用源代码）
pip install .

# 在线安装
pip install wordfreq-cn
```
---

## 使用方法

### 1. 命令行运行

```bash
  wordfreq-cn --news "人工智能技术在医疗领域的应用取得突破" "全球气候变化加剧" --topk 5
```

* `--news`：新闻标题或正文列表，可传多个
* `--topk`：输出前 N 个关键词（默认 10）
* 会在 `wordclouds/` 生成每条新闻或按日期聚合的趋势词云图片

示例输出：

```
=== TF-IDF 高权重词 ===
人工智能技术 1.0000
医疗 0.8349
应用 0.6730
...

=== TextRank 关键词 ===
标题: 人工智能技术在医疗领域的应用取得突破
  领域 (1.0000)
  医疗 (0.8349)
  取得 (0.6746)
  应用 (0.6730)
  突破 (0.5175)

=== 词频统计 ===
技术 2
人工智能 1
医疗 1
...
```

---

### 2. Python 调用

```python
from wordfreq_cn import tfidf_keywords, textrank_keywords, count_words, generate_trend_wordcloud, load_stopwords

news_list = [
    ("2025-11-25", "人工智能技术在医疗领域的应用取得突破"),
    ("2025-11-25", "全球气候变化加剧，联合国发布最新报告")
]

stopwords = load_stopwords(custom_file="stopwords.txt")

# TF-IDF
tfidf_res = tfidf_keywords([text for _, text in news_list], top_k=5, stopwords=stopwords)
print(tfidf_res)

# 词频统计
counter = count_words([text for _, text in news_list], stopwords=stopwords)
print(counter)

# 按日期生成词云
from collections import defaultdict

news_by_date = defaultdict(list)
for date, text in news_list:
    news_by_date[date].append(text)
generate_trend_wordcloud(news_by_date, stopwords=stopwords)
```
词云示例

![2015-11-25](wordclouds/wordcloud_2025-11-25.png)
![2015-11-26](wordclouds/wordcloud_2025-11-26.png)

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

## 文件说明

| 文件名                                               | 说明                     |
|---------------------------------------------------|------------------------|
| `wordfreq_cn/`                                    | Python 包目录，包含核心逻辑和 CLI |
| `wordfreq_cn/data/stopwords.txt`                  | 可选自定义停用词文件             |
| `wordfreq_cn/data/cn_stopwords.txt`               | 哈工大中文停用词表（脚本可自动加载）     |
| `wordfreq_cn/data/fonts/SourceHanSansHWSC-VF.ttf` | 《思源黑体》中文字体文件，用于生成中文词云  |
| `wordclouds/`                                     | 存放生成的词云图片              |
| `tests/`                                          | 单元测试代码                 |

---

## 注意事项

* 如果新闻量大，可在 `tfidf_keywords` 函数中调整 `max_features` 和 `top_k` 参数。
* 建议停用词表包含常用虚词（如“的”“在”“是”）以获得更干净的词频统计结果。
* 安装后，可以直接使用 `wordfreq-cn` 命令，无需运行 `python main.py`或者`python wordfreq_cn/cli.py` 之类的命令使用。

