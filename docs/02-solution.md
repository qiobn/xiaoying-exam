# 企业知识问答 Agent 技术方案

> 状态：本地可运行的实现方案。接口与选型以仓库当前代码和 `/api/capabilities` 为准。

## 1. 设计目标与取舍

项目需同时满足笔试验收与可演示的真实使用体验：员工能在本地 Web 工作台上传资料、构建知识库、查看解析分块与检索过程，并通过会话获得有原文依据的答案。底层仍须证明模型能观察工具结果并调整下一步；失败、冲突和资料缺失能被如实表达。评审可从 UI 和结构化 Trace 复现决策路径。

参考 kb-service 的知识组织、召回文本构建、混合检索、候选去重、rerank 降级和检索回归测试；参考 Agent Builder 的模型/工具调用分层、上下文精简和结构化 Trace；参考 agent-eval 的固定场景、回放证据和评估归档。本项目吸收这些原则，采用本地优先的单机全栈：FastAPI 提供文档、知识库、会话和事件 API；React/Vite 提供上传、知识库检查、运行日志与交互问答界面；SQLite 保存元数据、会话和运行事件；本地可替换的检索适配器与可选 OpenAI-compatible 模型客户端构成 Agent 核心。CLI 作为开发、诊断和离线评测入口保留，不再是唯一交付。首版不部署云服务、外部数据库、分布式向量库或多 Agent。

## 2. 技术选择

| 能力 | 选择 | 理由与边界 |
| --- | --- | --- |
| 语言 | Python 3.12+ | 参考仓库均采用 Python，模型及测试生态成熟 |
| 运行入口 | 单命令启动本地 Web 应用；CLI 保留用于离线评测和诊断 | 评审可浏览器交互，也可复制 CLI 命令复现关键用例 |
| Agent 循环 | LangGraph StateGraph + ToolNode；LangChain tools/chat model 接口 | 复用成熟编排与工具协议；业务策略和证据校验只保留必要定制 |
| 模型接入 | LangChain ChatOpenAI adapter，兼容 OpenAI API endpoint；离线 Mock 复用同一 LangGraph 图 | 配置简单、供应商可替换；Mock 只验证工作流，不代表真实决策 |
| 数据契约 | Pydantic v2 | 校验输入、文档、动作、响应和 Trace |
| 文档与运行状态 | SQLite 保存知识库、文档、chunk、会话和 run/event；JSONL 用于合成种子资料与评测导入导出 | UI 可检查实际索引和运行过程；单文件易复制、清理和演示 |
| 检索首选 | 入库时生成标题/适用范围 breadcrumb 检索文本；查询侧使用 jieba + BM25，按稳定 chunk ID 去重后截断 top-k，再回填原文 | 无外部服务仍可启动；检索文本增强主题召回，但引用永远来自原始 chunk |
| 检索扩展 | dense、rerank、精确词路由保留为后续可替换适配器 | 只有在固定验证集证明收益后再增加复杂度；rerank 失败必须回退 BM25 候选 |
| 测试 | pytest | 确定性 Fake 覆盖流程，真实模型评测单独标记 |
| 质量工具 | Ruff | 格式和静态错误检查，依赖轻量 |
| 配置 | 环境变量；真实模型变量写入 .env.example | 无密钥默认运行 Mock；密钥不进代码或 Trace |

默认问答模式为离线 Mock，不包含模型权重；真实模式调用 LangChain ChatOpenAI，默认模型名为 `gpt-4o-mini`，可由 OpenAI-compatible 环境变量覆盖。当前 embedding 为 `none`，检索采用 jieba + BM25，不存在向量索引。PDF 使用可选 MinerU 优先、PyMuPDF 文本抽取回退；无文本的扫描 PDF 和图片使用已安装的 Tesseract OCR。

## 3. 逻辑架构

    data/documents.jsonl
      -> DocumentRepository
      -> Retriever (摘要命中与稳定文档 ID)
      -> AgentLoop <-> ModelPolicy (选择下一工具动作)
            |             +-- OpenAI-compatible adapter / Offline FakeModel
            +-- ToolRegistry
                  +-- search_documents(query, top_k)
                  +-- read_document(document_id, query)
      -> EvidenceVerifier / ConflictResolver
      -> AnswerAssembler -> JSON response + JSONL Trace

建议代码结构（全栈 Web + 可选 CLI）：

    src/knowledge_agent/
      api.py                 # FastAPI 路由、请求校验与 SSE run 事件
      cli.py                 # 参数、输入输出和进程退出码
      config.py              # 校验运行设置
      schemas.py             # 请求、文档、动作、响应和 Trace
      agent.py               # LangGraph StateGraph 与有界决策路由
      model_client.py        # LangChain ChatModel 配置和真实/离线策略
      tools.py               # LangChain @tool 与结构化工具结果
      repository.py          # 读取源文档
      ingestion.py           # 切分和索引流程
      parsers.py             # TXT/Markdown/DOCX/PDF/图片统一解析，MinerU/Tesseract 适配
      persistence.py         # SQLite schema、事务与 run journal
      retrieval.py           # 规范化、扩展和排序
      evidence.py            # 原文引用校验和冲突处理
      prompts/               # 决策与最终作答提示词
    frontend/                # React/Vite 工作台，生产构建由 FastAPI 托管
    tests/                   # API、检索、路径、证据、失败、UI 契约和 CLI 测试
    data/                    # 合成 JSONL 资料
    examples/cases|outputs|traces/
    docs/                    # 需求、方案、评估、AI 使用和复盘

避免每个类独立建文件。让 5-8 个稳定职责模块承载主要行为，测试从公开接口穿过同一条链路。

## 4. 模块接口与控制权

### AgentLoop

对外接口为 run(request) -> response。模型策略、工具表、执行限制和 Trace 记录器从构造时注入。每轮将用户问题、有限历史、已观察结果和工具 schema 交给模型，获取一个合法工具动作或最终答复。

LangGraph 管理状态迁移和循环，LangChain ToolNode 执行白名单工具。代码负责输入参数校验、步数/调用上限、失败结果归一化、引用存在性与响应 schema；真实模型负责每次观察后的下一步选择。代码不按检索命中数硬编码“下一步必须 read/finish”，避免把固定分支冒充模型决策。

### ModelPolicy 与 ToolRegistry

模型通过 LangChain ChatModel.bind_tools 产生结构化工具调用；LangGraph ToolNode 执行工具并将观察回送模型。离线 Fake policy 根据图状态驱动相同节点和工具协议。Fake Trace 明确标为模拟，不声称代表真实模型行为。

ToolRegistry 只提供 search_documents 和 read_document。模型不能指定任意文件路径、shell 命令或数据库表达式。top_k、查询长度、返回大小及可读 document ID 均受限。测试可注入故障工具。

### Retriever、Repository、EvidenceVerifier

Retriever 接口为 search(query, top_k, filters=None) -> list[SearchHit]。SearchHit 含 document_id、标题、日期、范围、摘要、排序分和命中字段，不含可作为最终引文的伪造片段。Repository 按 ID 读取原文。EvidenceVerifier 检查 quote 是否为源内容的连续原文；适用范围或优先级必须由已读证据支持。日期只作为引用元数据，不自动消解冲突。

## 5. 模型决策循环

1. CLI 校验请求并规范化 history。历史只帮助模型补全省略式问题，不是事实证据。
2. 模型选择 search_documents、read_document 或结束/追问。提示要求先搜索并读取来源，不能从摘要直接生成引用。
3. 工具执行后返回精简观察；Trace 保存实际参数、结果 ID/数量/摘要、耗时和错误码，不保留 chain-of-thought、密钥或无关全文。
4. 模型根据观察决定再次换词搜索、读取其他命中文档、追问或结束。
5. 完成答案先经过引用校验和状态校验，再输出 answer、citations、gaps、status、question_to_user 和 trace。
6. 达到限制时由循环返回 step_limit，保留已读证据与停止原因。

默认限制建议：最多 6 次模型决策、最多 4 次工具调用、相同查询最多 2 次、top_k 不超过 5、单次观察限制长度。具体值可配置并通过测试触发验证。

## 6. 检索方案

### 文档结构

JSONL 每行至少有 document_id、title、version_date、scope、content、synthetic 标记，可选 section_path。content 为可信原文；标题和章节路径可以补强召回，但增强字段与原文分离，不能出现在引用 quote 中。

### 召回与排序

首版对标题、章节路径、范围、关键词和正文做字段加权词项匹配，同义词表覆盖“报销期限/几天内交/提交时限”。jieba 可作为可选分词器；未安装时用标准库字符 n-gram 或词组匹配保底。排序和同分 tie-breaker 固定，Trace 可显示命中字段。

先建立固定 query set 测 recall@k 和 MRR。只有当词项检索的漏召回案例证明确有收益时，再加 BM25 (rank-bm25)。若再评估向量检索，固定 embedding 模型、维度、归一化与索引产物并对照基线；对这组小资料，发现冲突后读取全部相关原文比无条件引入 reranker 更关键。

真实模型可自主生成补充 query 并选是否再搜，离线 Fake 按场景验证相同循环协议。检索不能静默挑选最新版；日期不同但规则冲突且没有优先级依据时仍须追问。

## 7. 冲突、缺口和上下文

- 报销 30 天/15 天且无适用范围：读取双方原文，引用两份证据，报告缺口并返回 needs_clarification。
- 资料外问题：充分检索后返回 not_found，具体说明资料库未覆盖的主题。
- 相似改写：通过同义词扩展和模型发起的补充查询命中同一条款。
- 上下文追问：由模型根据 history 补全查询，再调用 search 和 read；不能复述历史答案冒充新证据。
- 会话持久化和长期记忆不在首版范围；每次 CLI 调用可接收本轮有限 history。

## 8. 故障和安全

工具异常以结构化 observation 返回 ok、error_code、retryable 和安全摘要。是否改写查询、换工具或停止由模型依据观察选择；循环不暗中自动重试。关键搜索/读取失败返回 tool_error。可选 reranker 若未来加入，失败时回退初始候选并在 Trace 标记降级。

用户文档和工具结果是不可信内容，与系统指令隔离。工具白名单固定，不接受任意路径或代码。限制输入、history 和观察长度。结构化日志记录 request_id、model、action、tool、latency_ms 和 status；不记录密钥、完整提示词或非必要个人信息。真实模型密钥只从环境读取。

## 9. Trace 和评估

每个决策 Trace 至少含 step、动作、参数、观察摘要、下一动作、简短 decision_summary、模式/模型名、耗时和累计调用数。依据只写“首轮只返回摘要，需读取原文”这类可审计理由，不写模型隐藏思维链。

至少保存这些运行材料：

1. 正常：search -> read -> finish，一条文档逐字支持答案。
2. 冲突：search -> read D01 -> 补充检索/读取 D02 -> needs_clarification。
3. 工具失败：失败观察进入下一轮，由模型依据失败选择停止或替代动作；无成功证据时为 tool_error。
4. 上下文追问：显示意图补全、重新检索和新引用。
5. 相似改写及资料替换：把 30 天改成 45 天后答案和引用同步改变。

自动测试覆盖 Schema、检索排序、原文读取、精确引用、冲突不选边、工具参数、两条动态路径、超时/异常、上限、JSON CLI/API 和资料替换。离线 Mock 与真实模型结果分开报告；真实模型 Trace 记录模型 ID、日期、参数及随机性限制。


## 10. 开发阶段与完成门槛

不设固定工时上限，依赖顺序与验收退出条件决定推进：

1. **规格与金标语料**：Schema、版本/范围、冲突对、改写、上下文、故障注入和 golden query set；每验收点映射到测试。
2. **可运行垂直切片**：FastAPI、SQLite、search/read、LangGraph、FakeModel、Trace 和预算；CLI 保持离线可复现。
3. **证据安全**：精确引用、claim 支持、冲突处理、状态和缺口；建立关键错误不可回归测试。
4. **检索迭代**：baseline 后按漏召回类型增加规范化、字段权重、BM25/融合/可选 rerank；保留对照指标。
5. **恢复能力**：SQLite RunJournal、限次重试、deadline、Ctrl-C/进程重启恢复；Web SSE 事件日志支持断线游标回放。
6. **真实模型验证**：OpenAI-compatible tool calling；至少两种真实观察引发不同后续动作，保存脱敏 Trace。
7. **交付审阅**：依赖锁定、README、变量样例、至少三组实测 examples、评估/AI 使用/复盘文档、干净环境复现和最终 SHA。

阶段结束须更新评估文档，记录实际结果、失败、限制和下一步。界面验收与 Agent 核心链路并行推进，不能用静态页面或 Mock Trace 替代真实 Agent 证据。

## 13. 进入编码后确认的参数

实际 Python/Node 与 SDK 版本、真实 endpoint 与 model ID、模型 tool calling 支持、CLI/API 参数、文件格式边界、同义词覆盖、默认限额和检索权重应以安装结果和固定测试集验证后记录；SQLite journal 与恢复参数以进程中断测试结果锁定；不得在设计阶段编造准确率。


## 11. 当前不引入的企业设施

| 参考设施 | 本题暂缓原因 |
| --- | --- |
| Milvus、PostgreSQL、Redis、OSS | 文档库用 JSONL、运行恢复用本地 SQLite；没有并发入库或生产存储要求，不引入外部数据库与对象存储 |
| Docker、Kubernetes、外部异步队列 | 本地单机 Web 可由一个进程启动；无需容器编排或分布式任务设施 |
| 多租户、ACL、Entitlement、网关熔断 | 无真实租户和企业数据，增加安全与部署工作而不覆盖评分点 |
| LangGraph、AgentScope、ADK | 本项目优先使用 LangGraph StateGraph/ToolNode；仅不引入多 Agent 和复杂分布式 checkpoint |
| 向量库、rerank、LLM-as-Judge | 先由检索基线和案例证明问题及改进收益，不以技术堆叠替代验证 |

## 12. 参考项目吸收

| 来源 | 吸收 | 不照搬 |
| --- | --- | --- |
| kb-service | 检查了 docs/design.md、app/services/retrieval_text_builder.py、app/routers/retrieve.py 与 tests/test_retrieve_rule_router.py：吸收原文/元数据分离、标题/章节路径召回增强、候选去重、可选增强失败回退和故障回归 | 不引入 Milvus 集群、多租户、数据库/对象存储、多模型迁移和导入队列 |
| Agent Builder | 检查了 docs/ARCHITECTURE.md、app/tools/、app/runtime/ 与模型适配模块：吸收模型策略与工具注册分离、工具调用 Trace、上下文和观察精简 | 不引入 Supervisor/子 Agent、分布式 durable run、Postgres/Redis 事件总线；本地 SQLite checkpoint 只用于小规模可恢复执行 |
| agent-eval | 检查了 docs/TURN_REPLAY_AND_CALL_TRACE.md、docs/EVALUATION_CONFIG_REFERENCE.md 与 tests/test_traces.py：吸收固定输入、运行回放、输出/证据归档和评估结果引用案例 | 不引入独立评测平台、批量调度、云端存储和 Judge 模型 |


## 14. 加强版 Agent 主循环

### 显式运行状态

每次 run 持有 request_id、规范化问题、有限历史、工具观察 observations、已读取证据 evidence、已尝试查询 attempted_queries、模型/工具/时间/token 预算和终态。不要只靠把所有内容拼到提示词里表达状态。每条 observation 关联 tool_call_id、工具名、参数摘要、返回状态、证据 ID 和耗时。

### 单步协议

每次模型调用返回且只返回一种结果：工具调用（调用 ID、白名单工具名、结构化参数），或结束草稿（回答、citation IDs、gaps、状态、追问）。程序先做 Pydantic/schema 校验，再执行工具；将结果和相同调用 ID 返回给模型，下一次模型选择才算新一步。首版每轮只执行一个工具，便于核对“动作-观察-调整”。以后并行只允许独立只读搜索，并把每个结果分别记入 Trace。

无效 JSON、未知工具名、非法参数或调用 ID 不匹配时视作协议错误；最多将校验错误作为观察交还模型纠正一次。再次失败则 model_error。重复相同查询限制次数；没有工具白名单的动作绝不执行。

### 完成闸门

模型的 finish 只是建议。代码检查：completed 必须有成功读取的原文；每个 citation ID 必须属于当前 run 已读证据；quote 必须是来源文档连续、逐字的原文；回答中的关键数值、日期和适用范围要由引用支持；已发现但未解决的冲突不能 completed。needs_clarification 要明确缺失前提并提出可回答问题。not_found 仅适用于检索成功但无充分证据；关键工具不可用应为 tool_error。达到执行预算时返回 step_limit 并保留已读证据。最终答案若校验失败，只允许一次带校验错误的修正调用，仍失败则安全降级，不循环自修复。

同时限制模型决策轮数、工具调用数、重复查询数、每次超时、总 deadline、输入/输出 token 和观察长度；每步 Trace 记录剩余额度。所有重试时间计入总 deadline。CLI 用非零退出码标记未完成/错误状态；stdout 输出机器可读 JSON，诊断写 stderr。

## 15. 深化 RAG：从问题到可引用证据

### 检索流水线

1. 保留原问，并生成 standalone query。省略式追问从 history 补全主题和指代；地点、人员类别、日期、产品版本等适用条件保留为结构化 filters。不能确定关键前提时保留歧义并追问，不能擅自补事实。
2. 对文档 ID、型号、错误码、日期、数字等精确实体做词面匹配，避免语义改写损坏关键 token。
3. 词法召回分别对标题、章节路径、范围、FAQ 问句、关键词和正文评分；中文 jieba 分词，加字符 n-gram 兜底；同义词扩展需记录 query source。
4. 语料规模和失败样本证明有收益后再增加 embedding 语义召回。固定模型版本和归一化方式；相似度是排序信号，不是事实证据或唯一拒答阈值。
5. 多通道候选用 RRF 或明确定义的归一化方法融合，按 document_id/chunk_id 去重，保留每路排名、分数与匹配字段。变体索引命中时返回 canonical 原文 ID，同时记录变体命中原因。
6. 可选 cross-encoder reranker 只对有限候选重排。超时或失败时回退初始排序并标记 degraded；不改成空命中。小型合成语料首版可以不接 reranker。
7. 对范围和版本做扩展检索；规则值冲突时保留各方候选，不能仅按日期选最新版。为防止 top-k 只返回一方，可用冲突组约束保证同事实候选成组保留。
8. Agent 选择 read_document 后，从 canonical 文档取得原文和定位信息。标题/章节 breadcrumb 可以帮助召回，却与 canonical content 分离，禁止作为 quote。
9. 证据打包按相关性和 token budget 限长，但同一冲突组完整保留；每片带 document ID、版本、scope、chunk ID/原文 offset。
10. 最终答案按 claim 验证：quote 精确匹配原文；含数值/日期/范围的 claim 必须有支持证据；不支持的内容删除、追问或拒答。

### 文档与切片

JSONL 资料至少含 document_id、title、version_date、scope、status、source_type、content、synthetic；可选 section_path、valid_from、valid_to、supersedes、topic/entity tags。考试提供 FAQ/制度和片段读取工具，首版直接以明确条款/FAQ 为切片，不开发通用 PDF 入库。未来支持文件时保存原始 hash、解析器/版本、切片参数与原文 offset；按标题、段落、表格结构切，避免前提和规则值分离；表格转文本时保留列名与来源定位。

### 检索评测

建立 golden query set：原问、改写问、期望 document/chunk、必须共同召回的冲突组、相似干扰项。记录 Recall@k、MRR、nDCG@k、冲突双方共同召回率和 query-level 错误。生成质量、引用忠实度与检索召回分开报告。仅当 baseline 暴露具体漏召回类型且对照评测证明改善后，再增加 BM25、embedding、RRF 或 rerank。

## 16. 失败分类、重试与降级

| 阶段 | 故障 | 重试和最终处理 |
| --- | --- | --- |
| 配置/协议 | 缺 endpoint、无效模型输出、未知工具 | 配置错误不重试；格式协议最多一次纠正，之后 model_error |
| 模型传输 | 连接重置、超时、429、部分 5xx | 有界指数退避+随机抖动；仅 deadline 充足时重试；耗尽为 model_error |
| 模型业务 | 400 参数错、401/403、上下文超限 | 不重放相同请求；明确报错，要求修正配置/上下文 |
| 搜索 | 本地语料损坏、远端检索失败 | 仅瞬态错误有限重试；失败不是空列表，返回 tool_error |
| reranker | 可选重排超时/不可用 | 回退初始召回顺序，Trace 标 degraded，仍保留命中 |
| 原文读取 | ID 不存在/存储失败 | 不存在不重试；瞬态存储错误限次重试；未读原文绝不引用 |
| 资料冲突 | 两条适用规则给出不同值 | 不重试；引用双方并 needs_clarification |
| 用户中断 | Ctrl-C/取消 | 不自动重放半完成请求；flush 部分 Trace，状态 interrupted |

传输层 retry 与业务逻辑重试分开，集中配置一个 retry owner，避免 SDK retries 与外层循环叠乘。设置 timeout、attempt 上限、总 deadline、指数退避和 jitter；如 SDK 提供 Retry-After，遵守且不得越过 deadline。Trace 记录 attempt 数和最终错误类别。

当前 search/read 都是只读工具，但重试仍计入调用预算。未来若工具产生副作用，必须传幂等键；断线后结果未知不能盲目重发。搜索失败不得归为 not_found；rerank 降级不等于无命中；部分证据不得包装成完整答案。

## 17. 断线、中断和重连语义

### 本地 RunJournal 与恢复点

Web 服务与 CLI 共用 SQLite 单文件保存 run header、状态快照、模型调用摘要、tool intent/result、Trace event 和最终响应。每次状态变化用事务提交；敏感原文仅保存引用校验所需最小证据。run_id 区分任务，thread_id 区分对话，client_request_id 做重复提交幂等映射。事件 seq 单调递增、event_id 稳定唯一，last_seq 可续读本地 journal。

执行工具前先持久化 TOOL_PENDING、call_id 和参数 hash；工具完成后持久化 TOOL_COMMITTED 与 observation。恢复时已提交结果直接复用，不重复调用。纯本地只读检索的 PENDING 可重新执行并增加 attempt；未来副作用工具必须使用下游幂等键，结果不确定时标记 outcome_unknown 并停止，不能盲目重放。通用远程副作用不承诺 exactly-once，目标是持久 intent/result、幂等调用和去重。

模型 stream 未收到完整终止事件或完整 tool call 时，不能执行半截工具动作；记录失败和非敏感事件摘要，有限次以新 attempt 重发模型请求，并标记 regenerated，不承诺 token 级续传一致。完整 tool call 校验后先落 intent，再执行；客户端断开不改变 run 状态。

### RunJournal 与 CLI 行为

浏览器通过 SSE 订阅 run 事件，SQLite event journal 支持断线后按游标回放；本地 CLI 也可读取同一 run journal。命令可用 --resume RUN_ID 从最近安全检查点继续；新问题创建新 run_id。完整结果写 stdout，Trace 事件逐步写 journal 并可导出 JSONL。Ctrl-C 时保存 interrupted 状态、已完成观察和部分证据，不输出 completed 答案。恢复时已提交的工具结果直接复用，未完成的只读工具 intent 可按策略重做；不完整模型流以新 attempt 重发并标记 regenerated。

### Web/SSE 事件接口

SSE 是单向服务端流。订阅断开不代表 Agent run 取消；重连也不能直接从头重跑。实现时将 run 执行和客户端订阅分离：

- POST 创建 run，接受 idempotency key；同一 key 重复提交回到同一 run。
- 每个 run 将事件先追加到 event log 再发布。事件有单调 seq 和 event_id；公开事件只含阶段、工具参数/结果摘要和终态，不含密钥、完整 prompt 或思维链。
- GET /runs/{id}/events 用 Last-Event-ID 或 after_seq 回放游标后的持久事件，再订阅新事件；客户端按 event_id 去重并检查 seq gap。
- heartbeat 防代理空闲断连；客户端网络重连有上限、指数退避和 jitter。HTTP 200 仅说明流建立，不表示 run 完成。
- run 状态独立于连接状态，明确 queued/running/completed/failed/interrupted/step_limit。
- event log 只解决客户端回放。要在进程崩溃后继续执行，还要持久化 Agent checkpoint：模型轮次、待执行/已完成 tool_call_id、工具观察、证据、预算。恢复先核实工具动作是否已完成，防重复执行。
- 若无 checkpoint，服务器可继续后台 run、客户端重连回放；进程崩溃则标 interrupted/unknown，不能伪称可恢复。
- seq 分配和事件追加须原子化或具幂等去重；subscriber 离线不得阻塞 Agent 执行。

LangGraph checkpointer 可持久化 thread 状态和恢复中断，但不自动提供 SSE event log、客户端去重、恰好一次副作用或取消语义。本地 Web 与 CLI 共用 SQLite journal，不需要外部数据库或事件总线。

## 18. 评测方法与扩展验收矩阵

检索数据固定版本，分别评估 Recall@k、MRR/Hit@k、冲突候选召回率、citation exact-match、evidence support/faithfulness。测试集覆盖改写、错别字、编号/型号精确词、文档版本冲突和上下文代词。用消融表比较 lexical、BM25、dense、RRF、rerank，避免只记录最终答案主观感受。

Agent 测试评估动作序列正确性（tool name/args/观察后动作）、成功终止与有限终止、失败路由、预算、恢复去重；答复评估明确正确、引用支持、冲突不选边、缺口表述。Ragas 指标可作为后续实验参考，不在极小样本上宣称统计有效；本题主验收采用人工可核验合成 gold 和确定性断言，不用 LLM-as-Judge 替代引用逐字校验。


- Retry 矩阵：400/401 不重试；429/5xx/连接错误有限重试；总 deadline 生效。
- Reranker 抛错仍保留候选和原排序，并在 Trace 标 degraded。
- 摘要命中必须触发 read；read 失败不能产生 citation。
- 扩展检索应找回冲突双方，引用可定位到两份原文。
- citation 字符级匹配和 claim 支持检查；无依据数值不得通过。
- Ctrl-C 保留 partial trace，不输出 completed。
- CLI 恢复测试覆盖 Ctrl-C、模型流中断、工具执行前/后进程退出；已提交工具 observation 不重复执行。SSE 覆盖断线游标回放、幂等创建、事件去重和 run/连接状态独立。
- Mock 和真实模型路径分别归档，模拟轨迹不能作为真实模型决策证明。

## 19. 成熟项目与一手资料

外部库的实现行为应在锁定依赖版本后复核。

- LangChain Agent：模型在工具调用与工具结果之间循环，prompt/tools/middleware 构成 harness。https://docs.langchain.com/oss/python/langchain/agents
- LangChain Tools：工具是有明确定义输入输出的 callable，模型决定调用时机和参数。https://docs.langchain.com/oss/python/langchain/tools
- LangChain Structured Output：用 schema 校验结构化响应。https://docs.langchain.com/oss/python/langchain/structured-output
- LangGraph Fault Tolerance：timeout、retry 和 error handler 的职责与顺序；异常分类决定 retry。https://docs.langchain.com/oss/python/langgraph/fault-tolerance
- LangGraph Persistence：checkpointer 保存 thread 状态，store 承载跨 thread 长期数据。https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Event Streaming：消息、状态更新和自定义事件可单独投影；stream 不等同持久状态。https://docs.langchain.com/oss/python/langgraph/event-streaming
- Qdrant Hybrid Queries：dense/sparse 查询可通过融合策略组合，RRF 是一种秩融合方式。https://qdrant.tech/documentation/search/hybrid-queries/
- Ragas Metrics：Context Precision/Recall、Faithfulness、Response Relevancy 分别衡量不同阶段，不能合并成一个模糊总分。https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- OpenAI Retrieval guide：语义搜索可弥补词面不匹配，再用模型综合结果。https://platform.openai.com/docs/guides/retrieval
- Sentence Transformers Retrieve & Re-Rank：bi-encoder 先召回，cross-encoder 精排有限候选。https://www.sbert.net/examples/applications/retrieve_rerank/README.html
- MDN SSE：事件流和重连协议基础；应用仍须定义运行状态和回放语义。https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
- OpenAI Python SDK README：默认请求重试可通过 max_retries 配置；stream 消费超时会抛错，stream consumption 不自动重试以免重复交付，应用需决定如何恢复。https://github.com/openai/openai-python#streaming-responses
- 本地 kb-service：docs/design.md、app/services/retrieval_text_builder.py、app/routers/retrieve.py、tests/test_retrieve_rule_router.py、tests/test_retrieve_rerank_fallback.py。
- 本地 Agent Builder：docs/ARCHITECTURE.md、app/runtime、app/tools 和 model adapter。
- 本地 agent-eval：docs/TURN_REPLAY_AND_CALL_TRACE.md、tests/test_traces.py。
## 20. 本地 Web 工作台：目标体验与页面结构

产品形态参考纽酷项目的完整应用体验，但只复用“打开即用、导航清晰、状态可见、交互闭环”的产品原则，不复制其业务、视觉资产或庞大技术栈。本项目定位为本地单机知识库工作台：用户运行一个命令，在浏览器完成资料导入、知识库构建、检索检查和问答；无需账号、云部署或外部基础设施。

### 主导航与工作区

- **知识库**：知识库列表、创建/删除、文档数、chunk 数、最近构建时间与状态；进入后有拖拽/选择文件上传入口和构建按钮。
- **文档与分块**：文档解析状态、来源、页数/字符数、错误信息；可筛选查看 chunk 原文、章节/页码、chunk 序号、长度与元数据。支持删除、重新解析；重新构建须版本化并避免半成品覆盖可用索引。
- **检索调试**：输入 query 后查看 query 改写（若启用）、候选 chunk、检索通道/分数、RRF 名次、过滤条件和耗时；选择候选可回看原文。日志需区分真实检索证据和 Mock/演示数据。
- **问答会话**：创建/切换/重命名会话，展示用户问题、流式回答、引用卡片、证据原文、资料缺口和任务状态。点击引用跳转到文档原文及 chunk。
- **运行详情/事件**：按时间展示 Agent 选择的动作、工具名与参数、耗时、结果摘要、下一动作、结束原因和累计预算。可从会话消息打开对应 run；不得显示隐藏思维链或 API key。
- **设置**：模型模式（离线 Mock / 真实模型）、endpoint/model 名称、索引和 chunk 配置；密钥仅从环境变量读取，不在浏览器回显或写入运行日志。

### 交互状态

每个页面须覆盖 loading、empty、success、partial、error 状态；上传显示文件级进度与失败原因，构建显示 queued/parsing/chunking/indexing/completed/failed；问答显示 queued/running/completed/needs_clarification/not_found/tool_error/step_limit。页面刷新或 SSE 断线后按 run_id 与 last_seq 恢复，不将连接断开误判为 Agent 取消。

### 本地启动与演示

交付一个根目录启动器（例如 `python run.py`），负责检查配置、启动本地 API 并打开/打印浏览器地址；生产/评审模式由 FastAPI 托管前端构建产物，开发模式允许 Vite 热更新。README 必须给出从克隆、安装依赖、初始化合成示例库到启动和提问的逐步命令，并提供无需密钥的 Mock 演示。另保留 `python -m knowledge_agent ...` 的 CLI 评测命令，作为 UI 之外的确定性复现路径。

Web UI 是可运行功能而非截图或静态原型。前端展示的检索、chunk、日志必须由同一后端业务链路产生；不允许仅在浏览器硬编码一份假 Trace 冒充实时结果。离线 Mock 可驱动完整演示，但界面和文档要明确标注 Mock。

## 21. Web/API 与数据链路

建议的最小 API 面：

| 方法与路径 | 用途 | 关键行为 |
| --- | --- | --- |
| `POST /api/knowledge-bases` | 创建知识库 | 返回稳定 ID 与配置 |
| `GET /api/knowledge-bases` | 知识库列表/状态 | 汇总文档、chunk 和构建状态 |
| `POST /api/knowledge-bases/{id}/documents` | 上传资料 | 多文件、类型/大小校验、幂等文件 hash；不得执行上传内容 |
| `POST /api/knowledge-bases/{id}/build` | 构建/重建索引 | 同步返回耗时和索引版本；记录开始/成功/失败事件，失败时 SQLite 事务回滚 |
| `GET /api/knowledge-bases/{id}/documents` | 文档列表 | 返回解析状态、来源和统计 |
| `GET /api/documents/{id}/chunks` | chunk 浏览 | 分页返回原文片段和定位元数据 |
| `GET /api/knowledge-bases/{id}/build-events` | 构建/检索活动日志 | 返回该知识库最近的可观测事件 |
| `POST /api/knowledge-bases/{id}/retrieve` | 检索调试 | 返回 BM25 候选、匹配词、分数、索引版本与耗时 |
| `POST /api/sessions`、`GET /api/sessions` | 会话管理 | 会话绑定知识库，历史有界 |
| `POST /api/sessions/{id}/runs` | 创建问答 run | 幂等键防重复提交，立即返回 run_id |
| `GET /api/runs/{id}` | run 状态/最终结果 | 状态与连接生命周期分离 |
| `GET /api/runs/{id}/events` | SSE 事件流 | seq/event_id 游标回放、重连去重与终态事件 |

文档摄取链路：文件校验 -> 文本解析 -> chunk 切分 -> 生成 breadcrumb 检索文本 -> SQLite 事务写入索引。每个 chunk 保留 document_id、chunk_id、原文 offset 和检索文本，使引用能反向定位；标题和适用范围只进入检索字段，不伪装成 quote。格式先覆盖 TXT/Markdown/PDF/DOCX；扫描 PDF/OCR、表格结构恢复可列为明确限制，不能静默当作空文档成功。

问答链路沿用 AgentLoop：用户/有限历史 -> search -> 模型观察结果并决定 read/补充 search/ask_user/finish -> 引用验证与冲突汇总 -> SSE + 持久 Trace。检索调试 API 与 Agent 工具调用复用同一 Retriever，避免两套排序逻辑。文档原文和工具输出均视为不可信输入并与系统指令隔离。

安全与资源边界：限制上传文件类型、单文件/总大小、解析时间、chunk 数、上下文长度和并发 run；文件名净化，使用随机存储名，防路径穿越；不渲染不可信 HTML；删除文档时同步清理其 chunk 与索引关联；默认仅绑定 localhost。真实模型密钥只在后端进程读取并脱敏，SQLite/上传目录放在用户数据目录且加入 gitignore。

## 22. 调整后的开发阶段与门槛

按可演示垂直切片推进，每阶段可运行、可验证，不先建设高复杂度设施：

1. **契约与种子资料**：状态/证据契约、合成冲突语料、gold query、API schema；验收映射到自动测试。
2. **后端可运行闭环**：JSONL/SQLite、search/read、真实/离线 ModelPolicy、AgentLoop、预算、Trace；CLI 完成正常、冲突、资料外和工具失败用例。
3. **知识摄取**：多格式上传、解析、chunk、索引版本和失败隔离；API 测试保证坏文件不会污染可用知识库。
4. **工作台骨架**：React 页面布局、知识库与文档列表、chunk 预览、空/错误/加载态；与真实 API 对接，不做纯展示假数据。
5. **检索与问答交互**：检索调试、会话和逐字引用、SSE Trace、断线游标恢复；UI 端到端覆盖 upload -> build -> retrieve -> ask -> inspect citation。
6. **评估与交付**：检索消融、至少两条不同模型观察驱动路径、三组以上输入/输出/Trace、Mock 与真实模型证据分开、Windows 单命令干净启动、README 和提交材料。

Agent 编排优先使用 LangChain/LangGraph，不自行重写通用工具协议或图运行时。任何阶段都不得以 UI 完整度掩盖 Agent 要求：至少两条真实模型路径必须体现工具结果改变后续动作；Fake/Mock 仅用于可重复流程测试。首版不做登录、多用户、云部署、外部向量数据库、分布式任务队列、通用 OCR 或生产 SLA。

## 8. 附件问答流程

聊天 run 接受 multipart 多文件上传和普通 JSON 问答两种输入。run 创建后，SSE 依次发布 `attachment_received`、`attachment_parsed`、`attachment_indexing`、`attachment_indexed`，之后进入 LangGraph 工具轨迹并以 `run_finished` 结束。附件写入当前会话知识库，当前 run 的检索限制在本次附件 document IDs，引用仍由原始 chunk 生成。解析器和模型能力通过 `/api/capabilities` 暴露，UI 不会将未安装的 OCR/MinerU 显示为可用。
