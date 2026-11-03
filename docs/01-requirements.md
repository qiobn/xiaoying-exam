# 企业知识问答 Agent：需求规格

## 1. 功能需求

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| F01 | 创建、删除知识库，上传 TXT、Markdown、CSV、PDF、DOCX 和常见图片 | API/Web 集成测试 |
| F02 | 保存原始上传文件，支持文档删除、重新解析和知识库重建 | `test_p0_p1_features.py` |
| F03 | 文本支持结构感知分块和 LLM 行号语义分块，失败时自动降级 | `test_semantic_chunking.py` |
| F04 | CSV/DOCX/PDF 表格转换为 Markdown；超长表格按行组拆分并在检索文本中带表头 | `test_table_and_image_processing.py` |
| F05 | 图片支持 OCR，可选视觉模型描述图表、截图和复杂版面 | 图片解析测试 |
| F06 | 普通对话直接回复；企业知识问题按需调用 RAG 工具 | `test_conversation_architecture.py` |
| F07 | 多轮追问支持规则与 LLM query rewrite，失败回退规则 | `test_p0_p1_features.py` |
| F08 | 检索返回 BM25/dense/RRF/rerank 诊断，rerank 失败回退 | `test_hardening.py` |
| F09 | Agent Trace 记录路由、query 改写、工具动作、观察和终态 | Trace JSON 与 API |
| F10 | 通过 SSE 实时传输阶段摘要和最终回答 token | `test_real_chat_streams_token_events` |
| F11 | 支持 `Idempotency-Key` 防止重复 run，支持协作式取消 | 运行 API 测试 |
| F12 | 输出 answer、citations、gaps、status、question_to_user、trace、metrics | Agent 集成测试 |

## 2. 非功能需求

- 本地单机运行，不依赖云数据库或消息队列。
- 默认 Mock 可完整运行；真实模型密钥只放 `settings/.env` 或进程环境。
- 提示词、密钥、内部配置不得出现在回答、Trace 或示例文件中。
- 文档内容视为不可信数据，不能执行其中的指令。
- 本地 Embedding/Rerank 默认离线加载缓存，失败时回退 BM25。
- 每个关键失败必须产生明确终态：`not_found`、`tool_error`、`step_limit`、`cancelled` 或 `needs_clarification`。

## 3. 非目标

- 多租户、SSO、ACL、分布式 Worker、Kafka/RocketMQ、Kubernetes。
- 生产级 SLA 和跨地域高可用。
- 向用户展示模型隐藏思维链；只展示可审计的决策摘要和工具观察。

## 4. 验收状态

| 状态 | 含义 |
| --- | --- |
| `completed` | 已形成有效回答，或普通对话完成 |
| `needs_clarification` | 证据冲突或缺少必要前提 |
| `not_found` | 当前知识库没有足够证据 |
| `out_of_scope` | 问题明显超出企业知识库与系统操作范围 |
| `tool_error` | 工具、解析或模型关键步骤失败 |
| `step_limit` | 达到步数或工具预算 |
| `cancelled` | 用户主动取消 |
