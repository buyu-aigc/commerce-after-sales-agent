# 数据

| 目录 | 内容 | 来源 | 许可 |
| --- | --- | --- | --- |
| `olist_derived/` | 订单事实 `order_facts_index.json`（98,666 笔订单）、类目风险索引、评测用例 | [Olist 巴西电商公开数据集](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | CC BY-NC-SA 4.0：需要署名，不得商用，以相同方式共享 |
| `rescommons_derived/` | 售后知识检索语料 `support_corpus.jsonl`、检索评测用例 | [rescommons/Full-Ecom-Chatbot-Dataset](https://huggingface.co/datasets/rescommons/Full-Ecom-Chatbot-Dataset) | MIT（见 `metadata.json`） |
| `v1rtucious_derived/` | 电商客服 Agent 评测用例 | [V1rtucious/Ecom-Chatbot-Test-Set](https://huggingface.co/datasets/V1rtucious/Ecom-Chatbot-Test-Set) | MIT（见 `metadata.json`） |
| `bitext_derived/` | 意图评测用例 | [Bitext customer support LLM chatbot training dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset) | 以来源页面为准 |
| `knowledge_base/` | 演示用的售后政策和常见问题 | 上游项目编写的演示内容，不属于 Olist 数据 | 和本仓库相同 |

`order_facts_index.json` 和 `support_corpus.jsonl` 就是评测时实际使用的版本。也可以自己重新生成：

```bash
uv sync --extra data
python scripts/download_olist_data.py && python scripts/build_olist_dataset.py
python scripts/download_rescommons_data.py && python scripts/build_rescommons_dataset.py
```
