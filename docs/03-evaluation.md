# 评测与证据

## 1. 确定性测试

```powershell
uv run pytest -q
```

当前结果：`63 passed`。覆盖上传解析、表格/图片、智能分块、query 改写、检索融合、冲突处理、范围消歧、无关问题隔离、内置操作指南、引用、SSE、幂等、取消和运行指标。

## 2. 检索评测

评测集位于 `data/evaluation_cases.jsonl`，包含 20 条合成用例：明确答案、同义问法、地区范围、资料冲突、资料外问题、提示词攻击和普通对话。

```powershell
uv run knowledge-agent-eval --db runtime/knowledge.db --output examples/evaluation-report.json
```

当前检索结果：

| 指标 | 结果 |
| --- | --- |
| Case 数 | 20 |
| Recall@5 | 0.95 |
| MRR | 0.95 |
| 评测模式 | Mock 配置、真实本地索引 |

`parking` 用例仍召回了泛化制度片段，是当前最明确的检索缺口；其他用例符合预期。

如需运行完整 Agent 状态评测：

```powershell
uv run knowledge-agent-eval --db runtime/knowledge.db --include-agent
```

该命令会为每个案例调用真实模型，耗时和费用明显更高，因此不作为默认 CI。

## 3. 真实 Trace

| 场景 | 文件 | 结果 |
| --- | --- | --- |
| 正常回答 | `examples/normal/trace.json` | `completed`，3 个 Agent 步骤，3 次模型调用 |
| 资料不足 | `examples/insufficient/trace.json` | `not_found`，没有伪造答案 |
| 工具失败 | `examples/tool_error/trace.json` | 坏图片解析失败，`tool_error` |

前两组使用真实 OpenAI-compatible 模型运行，包含 SSE token、阶段事件、工具参数、工具观察、最终状态和 metrics。第三组使用受控坏图片触发真实解析失败。

## 4. 指标定义

- Recall@K：期望文档标题被前 K 个候选命中的比例。
- MRR：第一个相关候选排名倒数的均值。
- status accuracy：Agent 终态与评测集期望状态的一致率，仅在 `--include-agent` 时计算。
- citation faithfulness：最终引用必须来自 `read_document` 原文块。
- E2E latency：run 从创建到终态的耗时，来自 `metrics.duration_ms`。

## 5. 限制

- 检索集只有 20 条合成数据，不能代表生产分布。
- 当前指标以文档标题作为相关性标签，不能替代人工段落级标注。
- 真实模型结果存在随机性和供应商波动；提交时应固定模型、提示词和索引版本。
- 流式事件数、tokens 和耗时随供应商变化，不应当作固定快照逐字比对。
