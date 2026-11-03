# AI 使用说明

## 1. 使用范围

- 使用 AI 辅助梳理需求、拆分模块、编写初版实现和测试。
- 使用 OpenAI-compatible 模型作为运行时问答、query 改写、智能分块、大纲重建和可选视觉理解能力。
- AI 生成内容均需通过代码审查、自动测试和真实运行验证后才作为结论。

## 2. 人工控制点

- 模型只能生成 query、切片行号、标题层级或回答草稿，不能直接修改原文。
- 最终引用只取自 `read_document` 返回的原始 chunk。
- 冲突检测、引用数字校验、状态映射、步数上限和取消逻辑由代码执行。
- 提示词、密钥和本地路径不写入示例、Trace 或 Git。
- 用户上传内容和工具结果作为不可信数据处理，不允许覆盖系统约束。

## 3. 模型配置

| 能力 | 默认 | 可配置 |
| --- | --- | --- |
| 主 Agent | 当前 OpenAI-compatible 文本模型 | `OPENAI_MODEL`、`OPENAI_BASE_URL` |
| Query 改写 | 复用主模型，失败回退规则 | `QUERY_REWRITE_ENABLED` |
| 智能分块 | 普通长文按需调用，标题/表格优先结构化 | `CHUNK_STRATEGY`、`CHUNK_LLM_MODEL` |
| 大纲重建 | 仅处理扁平候选标题 | `OUTLINE_LLM_*` |
| 视觉理解 | 默认关闭 | `VISION_ENABLED`、`VISION_MODEL` |
| Embedding | 默认关闭，回退 BM25 | `EMBEDDING_*` |
| Rerank | 本地 CrossEncoder，失败回退 | `RERANK_*` |

## 4. 安全边界

- 不展示模型隐藏 reasoning，只展示路由、动作、工具观察和校验状态。
- 不把系统提示词、API Key 或内部配置返回给浏览器。
- 不将上传文件中的指令当作用户指令执行。
- 真实模型不可用时，Mock 只验证控制流，不宣称具有真实模型回答能力。
