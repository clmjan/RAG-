# 离线评测结果

以下结果来自本地命令 `python -m app.seed_demo` 与
`python -m app.evaluate --compare` 的一次实际运行（2026-09-19）。评测集
`data/evaluation_questions.json` 共 64 题：validation 11 题、test 53 题。
表中指标来自 test split；延迟为本机单进程运行时间，不代表线上 SLA。

| 方法 | Recall@1 | Recall@5 | MRR | 回答准确率 | 拒答准确率 | 引用覆盖率 | 平均延迟 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 70.59% | 100.00% | 0.8015 | 66.04% | 84.21% | 64.71% | 0.67 ms | 0.99 ms |
| bm25 | 73.53% | 76.47% | 0.7500 | 64.15% | 78.95% | 73.53% | 2.01 ms | 2.70 ms |
| embedding* | 73.53% | 76.47% | 0.7500 | 64.15% | 78.95% | 73.53% | 4.32 ms | 11.70 ms |
| hybrid* | 73.53% | 76.47% | 0.7500 | 64.15% | 78.95% | 73.53% | 16.80 ms | 28.37 ms |

`*` 当前运行未配置 Embedding Provider（`embedding_provider_configured=false`），
因此按设计回退到 BM25；这不是外部向量模型的效果声明。完整评测记录会写入
SQLite 的 `evaluation_runs` 表，并保存参数快照和数据集 SHA-256。

复现命令：

```powershell
python -m app.seed_demo
python -m app.evaluate --compare
```

数据集 SHA-256：
`a473fff5ee93e6128cfd3a186940ec59b959a0d0b8c39fb79938c6b5b8fe611d`
