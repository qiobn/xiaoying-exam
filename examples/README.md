# 示例证据

- `normal/trace.json`：真实模型正常检索、读取原文并回答。
- `insufficient/trace.json`：真实模型确认知识库资料不足，返回 `not_found`。
- `tool_error/trace.json`：坏图片解析失败，返回 `tool_error`。
- `evaluation-report.json`：20 条检索用例的 Recall@5、MRR 和逐案例结果。

Trace 保留结构化动作、工具参数、观察摘要、SSE 阶段事件和运行指标，不包含 API Key、系统提示词或隐藏思维链。
