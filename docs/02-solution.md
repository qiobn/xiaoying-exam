# 企业知识问答 Agent 技术方案

> 状态：待实现的设计基线。本文定义项目如何满足笔试要求，不代表功能或评测已完成。

## 1. 设计目标与取舍

项目需证明三件事：员工问题能从合成知识中获得准确且可核对的答案；模型能观察工具结果并调整下一步；失败、冲突和资料缺失能被如实表达。评审应能从结构化 Trace 复现决策路径。

参考 kb-service 的知识组织、召回文本构建、混合检索、候选去重、rerank 降级和检索回归测试；参考 Agent Builder 的模型/工具调用分层、上下文精简和结构化 Trace；参考 agent-eval 的固定场景、回放证据和评估归档。本项目保留这些原则但缩小实现：本地合成文档、纯 Python 检索适配器、CLI、可选 OpenAI-compatible 模型客户端。首版不部署 Milvus、Redis、MySQL、FastAPI、LangGraph 或多 Agent。

## 2. 技术选择

| 能力 | 选择 | 理由与边界 |
| --- | --- | --- |
| 语言 | Python 3.12+ | 参考仓库均采用 Python，模型及测试生态成熟 |
| 运行入口 | CLI，JSON 输入/输出 | 评审可复制运行，无需网络服务或前端 |
| Agent 循环 | 原生 Python ReAct loop | 透明、易测，直接证明模型动态选择动作 |
| 模型接入 | OpenAI Python SDK 的 tool calling，兼容 OpenAI API 的 endpoint | 真实模型与供应商可配置；离线 Fake 不访问网络 |
| 数据契约 | Pydantic v2 | 校验输入、文档、动作、响应和 Trace |
| 文档储存 | JSONL 合成语料 | 可读、可替换，可用临时语料验证输出随内容改变 |
| 初期检索 | 字段加权词项检索、同义词扩展 | 小语料离线、可解释、可复现 |
| 可选检索增强 | jieba；评测证明不足后再加 BM25 或本地 embedding | 不为了技术名词引入未验证依赖 |
| 测试 | pytest | 确定性 Fake 覆盖流程，真实模型评测单独标记 |
| 质量工具 | Ruff | 格式和静态错误检查，依赖轻量 |
| 配置 | 环境变量；真实模型变量写入 .env.example | 无密钥默认运行 Mock；密钥不进代码或 Trace |

进入编码后记录实际安装和验证的版本，不在方案中虚构版本号或模型效果。

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

建议代码结构：

    src/knowledge_agent/
      cli.py                 # 参数、输入输出和进程退出码
      config.py              # 校验运行设置
      schemas.py             # 请求、文档、动作、响应和 Trace
      agent.py               # 有上限的决策循环
      model_client.py        # ModelPolicy 与真实/离线适配器
      tools.py               # 工具 schema、分发和安全错误结果
      repository.py          # 读取源文档
      retrieval.py           # 规范化、扩展和排序
      evidence.py            # 原文引用校验和冲突处理
      prompts/               # 决策与最终作答提示词
    tests/                   # 检索、路径、证据、失败和 CLI 测试
    data/                    # 合成 JSONL 资料
    examples/cases|outputs|traces/
    docs/                    # 需求、方案、评估、AI 使用和复盘

避免每个类独立建文件。让 5-8 个稳定职责模块承载主要行为，测试从公开接口穿过同一条链路。

## 4. 模块接口与控制权

### AgentLoop

对外接口为 run(request) -> response。模型策略、工具表、执行限制和 Trace 记录器从构造时注入。每轮将用户问题、有限历史、已观察结果和工具 schema 交给模型，获取一个合法工具动作或最终答复。

固定由代码负责：动作名称白名单、参数验证、工具分发、步数/调用上限、失败结果归一化、引用存在性与响应 schema。模型负责每次观察后的下一步选择。代码不按检索命中数硬编码“下一步必须 read/finish”，避免把固定分支冒充模型决策。

### ModelPolicy 与 ToolRegistry

ModelPolicy 每次返回 ToolCall(name, arguments) 或 FinalDraft。真实模型适配使用 function/tool calling 并校验结构；离线 Fake 根据前一轮实际观察返回测试策略。Fake Trace 明确标为模拟，不声称代表真实模型行为。

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

自动测试覆盖 Schema、检索排序、原文读取、精确引用、冲突不选边、工具参数、两条动态路径、超时/异常、上限、JSON CLI 和资料替换。离线 Mock 与真实模型结果分开报告；真实模型 Trace 记录模型 ID、日期、参数及随机性限制。

## 10. 开发阶段与门槛

| 阶段 | 时间预算 | 工作 | 完成门槛 |
| --- | --- | --- | --- |
| 0 规格固化 | 15 分钟 | Schema、合成语料、验收映射、CLI 契约 | 每项验收点对应测试或场景 |
| 1 离线闭环 | 50 分钟 | JSONL、search/read、FakeModel 循环、Trace、上限 | 离线可跑 search/read/finish、tool_error、step_limit |
| 2 证据与冲突 | 30 分钟 | 引用校验、冲突与状态汇总 | 30/15 触发追问，引用精确匹配 |
| 3 检索改进 | 30 分钟 | 字段权重、同义词、查询扩展与对照评测 | 改写和替换资料回归通过，保留前后数据 |
| 4 真实模型 | 35 分钟 | OpenAI-compatible tools、配置、超时和脱敏日志 | 真实模型至少两条不同观察驱动的路径 |
| 5 交付 | 20 分钟 | 三组以上 examples、文档、干净环境验证和 commit | 评审可离线复制运行，Mock/真实证据清楚 |

三小时优先完成阶段 0-3、一次真实模型运行和阶段 5。网络或配额不可用时应如实记录真实路径未验证；Mock Trace 不能替代真实决策证据。

## 11. 当前不引入的企业设施

| 参考设施 | 本题暂缓原因 |
| --- | --- |
| Milvus、PostgreSQL、Redis、OSS | 数据量小，无并发导入或生产存储要求，JSONL 足以复现 |
| FastAPI、Docker、Kubernetes、异步队列 | 无在线服务或长任务验收点，CLI 更易检查 |
| 多租户、ACL、Entitlement、网关熔断 | 无真实租户和企业数据，增加安全与部署工作而不覆盖评分点 |
| LangGraph、AgentScope、ADK | 首版原生循环更直接地显示模型动作；需要 checkpoint/复杂状态时再评估 |
| 向量库、rerank、LLM-as-Judge | 先由检索基线和案例证明问题及改进收益，不以技术堆叠替代验证 |

## 12. 参考项目吸收

| 来源 | 吸收 | 不照搬 |
| --- | --- | --- |
| kb-service | 检查了 docs/design.md、app/services/retrieval_text_builder.py、app/routers/retrieve.py 与 tests/test_retrieve_rule_router.py：吸收原文/元数据分离、标题/章节路径召回增强、候选去重、可选增强失败回退和故障回归 | 不引入 Milvus 集群、多租户、数据库/对象存储、多模型迁移和导入队列 |
| Agent Builder | 检查了 docs/ARCHITECTURE.md、app/tools/、app/runtime/ 与模型适配模块：吸收模型策略与工具注册分离、工具调用 Trace、上下文和观察精简 | 不引入 Supervisor/子 Agent、checkpoint、durable run、Postgres/Redis 事件总线 |
| agent-eval | 检查了 docs/TURN_REPLAY_AND_CALL_TRACE.md、docs/EVALUATION_CONFIG_REFERENCE.md 与 tests/test_traces.py：吸收固定输入、运行回放、输出/证据归档和评估结果引用案例 | 不引入独立评测平台、批量调度、云端存储和 Judge 模型 |

## 13. 进入编码后确认的参数

实际 Python 和 SDK 版本、真实 endpoint 与 model ID、模型 tool calling 支持、CLI 参数名、同义词覆盖、默认限额与检索权重应以安装结果和固定测试集验证后记录；不得在设计阶段编造准确率。
