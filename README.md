# 小影知识库问答 Agent

基于 LangChain、LangGraph、FastAPI 和 React/Vite 的本地知识库工作台。支持上传资料、构建索引、检查分块与检索日志，并在聊天中上传附件问答。

## 快速启动（Windows）

首次运行：

```powershell
uv sync --extra dev
npm install --prefix frontend
Copy-Item settings/.env.example settings/.env
```

编辑 `settings/.env` 后启动：

```powershell
uv run python run.py
```

默认模式为离线 Mock，无需 API Key。工作台通常在 `http://127.0.0.1:8000/` 打开。

不用 uv 时也可以安装标准依赖清单：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
npm install --prefix frontend
python run.py
```

`requirements.txt` 是运行依赖，`requirements-dev.txt` 额外包含测试和静态检查依赖；uv 的项目元数据仍以 `pyproject.toml` 为准，并通过 `uv.lock` 固定可复现依赖版本。

## 配置

本地真实配置放在 `settings/.env`，该文件已加入 `.gitignore`，不会提交到 Git。模板和配置说明保存在 `settings/.env.example`、`settings/README.md`。不要将 API Key、MinerU Token 或机器相关路径写入模板或提交到仓库。环境变量优先级为：当前进程环境变量 > `settings/.env` > 根目录旧版 `.env`（兼容）。可通过 `KNOWLEDGE_AGENT_SETTINGS_DIR` 指定另一配置目录。

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `KNOWLEDGE_AGENT_MODE` | `mock` | `mock` 离线流程；`real` 调用真实 ChatOpenAI |
| `OPENAI_API_KEY` | 空 | OpenAI-compatible 服务凭据，仅保存在本地配置 |
| `OPENAI_BASE_URL` | 空 | 兼容服务地址 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 真实问答模型名 |
| `OPENAI_TIMEOUT_SECONDS` | `45` | 单次问答模型请求超时 |
| `QUERY_REWRITE_ENABLED` | `true` | 多轮上下文追问是否在检索前补全 query |
| `MODEL_LOCAL_FILES_ONLY` | `true` | 本地 Embedding/Rerank 仅使用已缓存权重，禁止运行时联网 |
| `EMBEDDING_BACKEND` | `none` | `none`、`local`、`openai_compatible`；默认关闭以保证零额外下载 |
| `EMBEDDING_MODEL` | `none` | 默认关闭；启用 local 后推荐 `BAAI/bge-small-zh-v1.5`，切换模型后需重新构建知识库 |
| `EMBEDDING_DEVICE` / `EMBEDDING_BATCH_SIZE` | `cpu` / `16` | 本地向量编码设备和批大小 |
| `MODEL_CACHE_DIR` | `runtime/models` | 本地模型权重缓存目录 |
| `RERANK_ENABLED` | `true` | 是否对候选结果做二阶段重排 |
| `RERANK_BACKEND` | `local` | `local`、`dashscope`、`http`、`none`；未安装可选依赖时自动降级 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 本地中文 CrossEncoder 推荐模型 |
| `RERANK_DEVICE` / `RERANK_TOP_K` | `cpu` / `8` | 重排设备和候选数量 |
| `RERANK_TIMEOUT_SECONDS` | `20` | 远程 rerank 请求超时 |
| `RERANK_API_URL` / `RERANK_API_KEY` | 空 | 通用 HTTP rerank 服务及本地密钥 |
| `KNOWLEDGE_AGENT_DB` | `runtime/knowledge.db` | SQLite 元数据、文档、索引、会话与运行事件 |
| `KNOWLEDGE_AGENT_MAX_UPLOAD_BYTES` | `10485760` | 单附件大小上限（10 MiB） |
| `KNOWLEDGE_AGENT_CHUNK_SIZE` / `KNOWLEDGE_AGENT_CHUNK_OVERLAP` | `420` / `60` | 文档切块长度和重叠 |
| `CHUNK_STRATEGY` | `auto` | `auto`、`structured`、`llm`；自动模式对普通长文使用 LLM 语义分块 |
| `CHUNK_LLM_MODEL` | 空 | 分块模型，留空复用问答模型 |
| `CHUNK_LLM_MIN_CHARS` / `CHUNK_LLM_MAX_CHARS` | `1600` / `12000` | 自动分块触发长度与单次窗口上限 |
| `CHUNK_LLM_TIMEOUT_SECONDS` | `60` | 单次智能分块超时 |
| `PARSE_PDF_TABLES` / `PARSE_EMBEDDED_IMAGES` | `true` / `true` | 是否提取 PDF 表格和文档内嵌图片 |
| `VISION_ENABLED` / `VISION_MODEL` | `false` / 空 | 是否用多模态模型描述图表，留空复用问答模型 |
| `VISION_API_URL` / `VISION_API_KEY` | 空 | 可与文字模型不同的视觉服务地址和密钥 |
| `VISION_TIMEOUT_SECONDS` / `MAX_EMBEDDED_IMAGES` | `60` / `20` | 视觉调用超时和单文档图片上限 |
| `KNOWLEDGE_AGENT_MAX_STEPS` / `KNOWLEDGE_AGENT_MAX_TOOL_CALLS` | `6` / `8` | Agent 图的执行上限 |
| `MINERU_COMMAND` | 空 | 可选 MinerU CLI；配置后优先处理 PDF |
| `MINERU_SERVER_URL` / `MINERU_METHOD` | 空 / `auto` | MinerU HTTP 后端地址和解析方式 |
| `MINERU_TIMEOUT_SECONDS` | `300` | MinerU/OCR 最大执行时间 |
| `TESSERACT_COMMAND` | `tesseract` | 图片和扫描 PDF 的可选 OCR 命令 |
| `TESSERACT_LANG` | `chi_sim+eng` | OCR 识别语言包 |

## 模型与解析选型

- **问答模型**：仓库不包含模型权重。默认 `mock` 是 LangGraph 图中的确定性离线策略，用于端到端验证；它不是 LLM。设置 `KNOWLEDGE_AGENT_MODE=real`、`OPENAI_API_KEY`，并按供应商设置 `OPENAI_BASE_URL` 和 `OPENAI_MODEL` 后，使用 LangChain `ChatOpenAI`。未指定模型名时默认 `gpt-4o-mini`。 Settings 页面现在可编辑并立即保存这些配置到被忽略的 `settings/.env`，也可在保存前测试当前填写的供应商连通性。要获得真实模型回答，仍需用户提供可用的模型服务地址与凭据；仓库本身不含模型密钥。
- **Embedding/检索**：默认仍是无需下载的 jieba + BM25，适合考试现场先打通流程。需要更好语义召回时，在 Settings 选择 `local`，推荐 `BAAI/bge-small-zh-v1.5`；安装 `sentence-transformers` 后构建知识库会把向量写入 SQLite，查询使用 BM25 + dense + RRF。精确编码、错误码和长数字仍保留词法召回优先。
- **Rerank**：本地推荐 `BAAI/bge-reranker-base`，CPU 成本较高，可关闭或将候选数调小；也可配置 DashScope 或通用 HTTP 服务。重排服务失败会保留第一阶段结果，并在检索诊断和运行日志中显示回退原因。
- **可选依赖**：默认 `requirements.txt` 不安装 PyTorch。需要本地模型时执行 `uv sync --extra local-rag` 或 `pip install -r requirements-local-rag.txt`；首次下载时临时关闭离线模式，未安装依赖、模型未缓存或加载失败都会回退 BM25。
- **解析**：TXT/Markdown 本地解码，CSV 转换为 Markdown 表格；DOCX 按正文顺序解析段落、表格和内嵌图片；PDF 由 PyMuPDF 提取文本与表格。配置 MinerU CLI 后优先使用 MinerU；图片及扫描 PDF 通过 Tesseract OCR，可选视觉模型补充图表、截图和复杂版面的描述。MinerU/Tesseract 是外部系统依赖，不会由 Python requirements 自动安装。
- **能力状态**：Settings 页面以及 `/api/capabilities` 返回运行时模式、配置模型和 MinerU/OCR 实际可用状态。

本地模型首次下载时可临时设置 `MODEL_LOCAL_FILES_ONLY=false`，下载完成后改回 `true`。服务会优先解析 `runtime/models` 中已存在的 Hugging Face snapshot 路径并直接离线加载，避免缓存命中后仍访问 Hub 校验。

## 对话与 RAG 架构

- **按需检索**：`routing.py` 的 `RetrievalGate` 区分普通交流和企业知识问题。问候、能力介绍、写作类请求直接由主 Agent 回复；制度、流程、产品规则等问题才绑定检索工具。
- **提示词保护**：`prompt_guard.py` 在模型和检索前拦截系统提示词提取、密钥索取和指令覆盖尝试；`prompts.py` 明确工具结果和上传文档是不可信资料，文档中的命令不得执行。
- **上下文改写**：`query_rewrite.py` 的 `ConversationQueryRewriter` 将“那国外出差呢”补全为自包含 query；改写产生的新实体必须出现在候选原文中，避免泛化词造成误召回。
- **结构感知分块**：`chunking.py` 的 `StructuredChunker` 识别 Markdown 标题层级，在 chunk 中保存 `section_path` 和精确原文 offset，再把标题、适用范围与章节路径写入检索文本。
- **LLM 智能分块**：`semantic_chunking.py` 参考纽酷的行号分段方案，模型只返回 `start_line/end_line/title`，正文由本地按偏移回填，不允许模型改写。普通长文在 `auto` 模式下使用 LLM；标题结构清晰、Mock、无密钥、模型异常或 JSON 不合法时自动降级为确定性分块。
- **表格与图表**：`document_parsers.py` 将表格转为 Markdown 并保持表头与数据行原子分块；视觉模型提取图表标题、坐标轴、图例、关键数值和 OCR 文本，图片描述进入检索文本。服务端记录 `table_count/image_count`，分块记录 `chunk_method`，用于界面核对。
- **混合检索与重排**：`retrieval.py` 使用 BM25、可选 dense embedding、RRF 和精确 token 路由；`KnowledgeRetrievalService` 是 Agent 工具与检索调试 API 的共用入口。Rerank 失败自动回退第一阶段候选。
- **可审计轨迹**：Trace 记录 `retrieval_route`、`query_rewrite`、`search_documents`、`read_document`、`finish` 或 `guardrail`，不记录隐藏思维过程。

## CLI

```powershell
uv run knowledge-agent --question "报销期限是多久？" --data data/documents.jsonl
```

## 目录

- `src/knowledge_agent/`：Agent、按需检索路由、提示词保护、query 改写、LLM/结构化分块、表格与图片解析、检索、API、索引、持久化及 CLI。
- `frontend/`：React/Vite 工作台。
- `settings/`：本地配置模板和使用说明；实际 `.env` 被忽略。
- `data/`：合成演示资料。
- `docs/`：项目需求和技术方案。
- `tests/`：解析、检索、Agent 和 API 测试。
