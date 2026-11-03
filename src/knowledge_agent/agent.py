from __future__ import annotations

import json
import re
from typing import Annotated, TypedDict
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import Settings
from .persistence import Store
from .prompt_guard import PromptGuard
from .prompts import CHAT_PROMPT, PROMPT_GUARD_ANSWER, SYSTEM_PROMPT
from .query_rewrite import ConversationQueryRewriter, RewriteResult
from .rag import KnowledgeRetrievalService
from .retrieval import terms
from .routing import RetrievalGate
from .schemas import AgentResult

GENERIC_TERMS = {"员工", "申请", "提交", "需要", "应该", "规定", "制度", "工作日", "多少", "几天", "多久"}


def chat_model_options(settings: Settings) -> dict:
    """Build provider-specific options without changing other OpenAI-compatible APIs."""
    options = {
        "model": settings.model or "gpt-4o-mini",
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "temperature": 0,
        "max_retries": 1,
        "timeout": settings.llm_timeout_seconds,
    }
    host = urlparse(settings.base_url or "").hostname or ""
    if host == "deepseek.com" or host.endswith(".deepseek.com"):
        # DeepSeek requires reasoning_content from prior assistant turns when thinking is on.
        # LangChain's tool-message round trip does not retain that provider-only field.
        options["extra_body"] = {"thinking": {"type": "disabled"}}
    return options


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    history: list[dict]
    trace: list[dict]
    step: int
    tool_calls: int
    mode: str
    final_text: str
    needs_retrieval: bool
    routing_reason: str
    rewritten_query: str


def make_tools(
    store: Store,
    kb_id: str,
    settings: Settings,
    question: str,
    rewritten_query: str,
    document_ids: list[str] | None = None,
    required_terms: tuple[str, ...] = (),
) -> list[BaseTool]:
    retriever = KnowledgeRetrievalService(store, kb_id, settings, document_ids, required_terms)
    guard = PromptGuard()

    @tool
    def search_documents(query: str, top_k: int = 5) -> str:
        """Search indexed knowledge chunks. Returns summaries, not citation-grade evidence."""
        effective_query = query.strip()
        if not effective_query or effective_query == question:
            effective_query = rewritten_query or question
        outcome = retriever.retrieve(effective_query, top_k)
        warnings = sorted({
            warning
            for hit in outcome.hits
            for warning in guard.untrusted_warnings(hit.get("text", ""))
        })
        return json.dumps({
            "ok": True,
            "query": query,
            "effective_query": effective_query,
            "count": len(outcome.hits),
            "hits": outcome.hits,
            "diagnostics": outcome.diagnostics,
            "security_warnings": warnings,
            "source_boundary": "untrusted_evidence",
        }, ensure_ascii=False)

    @tool
    def read_document(document_id: str, query: str = "") -> str:
        """Read citation-grade source text for one document."""
        doc = store.get_document(document_id)
        if document_ids and document_id not in document_ids:
            return json.dumps({"ok": False, "error_code": "document_not_in_run", "document_id": document_id}, ensure_ascii=False)
        if not doc or doc["kb_id"] != kb_id:
            return json.dumps({"ok": False, "error_code": "document_not_found", "document_id": document_id}, ensure_ascii=False)
        chunks = store.chunks(document_id)
        query_terms = terms(query)
        matched = [
            chunk for chunk in chunks
            if query_terms & terms(chunk.get("retrieval_text") or chunk["text"])
        ]
        chosen = matched[:3] or chunks[:2]
        warnings = sorted({
            warning
            for chunk in chosen
            for warning in guard.untrusted_warnings(chunk["text"])
        })
        return json.dumps({
            "ok": True,
            "document_id": document_id,
            "title": doc["title"],
            "version_date": doc["version_date"],
            "scope": doc["scope"],
            "chunks": chosen,
            "security_warnings": warnings,
            "source_boundary": "untrusted_evidence",
        }, ensure_ascii=False)

    return [search_documents, read_document]


def _last_tool_observation(messages: list[BaseMessage]):
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            try:
                return json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                return {"ok": False, "error_code": "invalid_tool_output"}
    return None


def _mock_decision(state: AgentState, tools: list[BaseTool]) -> AIMessage:
    question = state["question"]
    retrieval_query = state.get("rewritten_query") or question
    messages = state.get("messages", [])
    read_ids = []
    search_payload = None
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            payload = json.loads(msg.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if msg.name == "read_document" and payload.get("ok"):
            read_ids.append(payload.get("document_id"))
        elif msg.name == "search_documents":
            search_payload = payload
    if search_payload is None:
        return AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": retrieval_query, "top_k": 5}, "id": f"mock-search-{state.get('step', 0)}"}])
    if search_payload.get("ok"):
        hits = search_payload.get("hits", [])
        best_score = hits[0].get("fusion_score", hits[0]["score"]) if hits else 0
        candidates = list(dict.fromkeys(
            hit["document_id"] for hit in hits
            if hit.get("fusion_score", hit["score"]) >= best_score * 0.55
        ))
        unread = [doc_id for doc_id in candidates if doc_id not in read_ids]
        if unread and len(read_ids) < 2:
            doc_id = unread[0]
            return AIMessage(content="", tool_calls=[{"name": "read_document", "args": {"document_id": doc_id, "query": retrieval_query}, "id": f"mock-read-{doc_id}-{state.get('step', 0)}"}])
    return AIMessage(content="__FINAL__")


def _mock_chat_response(question: str) -> AIMessage:
    lowered = question.lower()
    if any(term in lowered for term in ("你是谁", "你能做什么", "你会什么", "介绍一下你")):
        return AIMessage(content="我是“知问”企业知识助手。可以直接和我交流，也可以在你需要时检索企业制度、流程和产品资料，并展示原文依据。")
    if any(term in lowered for term in ("谢谢", "感谢")):
        return AIMessage(content="不客气，有需要继续问我。")
    return AIMessage(content="你好，我是“知问”企业知识助手。当前是离线 Mock 模式，你可以继续提问；涉及企业资料时我会走知识库检索流程。")


def _build_graph(store: Store, kb_id: str, settings: Settings, tools: list[BaseTool]):
    tool_node = ToolNode(tools)

    def execute_tools(state: AgentState):
        result = tool_node.invoke(state)
        messages = result.get("messages", [])
        trace = list(state.get("trace", []))
        if messages and isinstance(messages[-1], ToolMessage) and trace:
            last = messages[-1]
            try:
                payload = json.loads(last.content)
            except (TypeError, json.JSONDecodeError):
                payload = {"ok": False, "summary": str(last.content)[:240]}
            diagnostics = payload.get("diagnostics", {})
            summary = f"{payload.get('count')} 个候选"
            if diagnostics.get("method"):
                summary += f" · {diagnostics['method']}"
            if diagnostics.get("fallback"):
                summary += f" · 回退：{diagnostics['fallback']}"
            trace[-1] = {
                **trace[-1],
                "observation": {
                    "ok": payload.get("ok", True),
                    "summary": summary if "hits" in payload else (
                        f"读取 {len(payload.get('chunks', []))} 个原文块"
                        if "chunks" in payload
                        else str(payload.get("error_code", "工具完成"))
                    ),
                    "document_ids": list(dict.fromkeys(
                        hit.get("document_id") for hit in payload.get("hits", [])
                    )) if "hits" in payload else [payload.get("document_id")] if payload.get("document_id") else [],
                    "security_warnings": payload.get("security_warnings", []),
                },
            }
        return {**result, "trace": trace}

    def agent_node(state: AgentState):
        step = state.get("step", 0) + 1
        messages = state.get("messages", [])
        needs_retrieval = state.get("needs_retrieval", True)
        if not needs_retrieval:
            if settings.mode == "real":
                if not settings.api_key:
                    raise ValueError("KNOWLEDGE_AGENT_MODE=real 时必须配置 OPENAI_API_KEY")
                from langchain_openai import ChatOpenAI

                model = ChatOpenAI(**chat_model_options(settings))
                response = model.invoke([SystemMessage(content=CHAT_PROMPT), *messages])
            else:
                response = _mock_chat_response(state["question"])
        else:
            if settings.mode == "real":
                if not settings.api_key:
                    raise ValueError("KNOWLEDGE_AGENT_MODE=real 时必须配置 OPENAI_API_KEY")
                from langchain_openai import ChatOpenAI

                model = ChatOpenAI(**chat_model_options(settings))
                prompt = SYSTEM_PROMPT.format(
                    retrieval_query=state.get("rewritten_query") or state["question"]
                )
                response = model.bind_tools(tools).invoke([SystemMessage(content=prompt), *messages])
                has_searched = any(
                    isinstance(message, ToolMessage) and message.name == "search_documents"
                    for message in messages
                )
                if not response.tool_calls and not has_searched:
                    response = AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "search_documents",
                            "args": {
                                "query": state.get("rewritten_query") or state["question"],
                                "top_k": 8,
                            },
                            "id": f"required-knowledge-search-{step}",
                        }],
                    )
            else:
                response = _mock_decision(state, tools)
        trace = list(state.get("trace", []))
        action = "finish" if not response.tool_calls else response.tool_calls[0]["name"]
        trace.append({"step": step, "action": action, "arguments": response.tool_calls[0].get("args", {}) if response.tool_calls else {}, "mode": settings.mode})
        return {"messages": [response], "step": step, "trace": trace, "tool_calls": state.get("tool_calls", 0) + len(response.tool_calls)}

    def route(state: AgentState):
        if state.get("step", 0) >= settings.max_steps:
            return END
        last = state.get("messages", [])[-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and last.tool_calls:
            if state.get("tool_calls", 0) >= settings.max_tool_calls:
                return END
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", execute_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


def _evidence(messages: list[BaseMessage]):
    citations = []
    reads = []
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name != "read_document":
            continue
        try:
            payload = json.loads(msg.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if payload.get("ok"):
            reads.append(payload)
            for chunk in payload.get("chunks", []):
                citations.append({"document_id": payload["document_id"], "title": payload["title"], "version_date": payload["version_date"], "quote": chunk["text"], "chunk_id": chunk["id"]})
    return reads, citations


def _assemble(
    question: str,
    messages: list[BaseMessage],
    trace: list[dict],
    settings: Settings,
    needs_retrieval: bool,
) -> AgentResult:
    final_message = next(
        (
            msg
            for msg in reversed(messages)
            if isinstance(msg, AIMessage)
            and not msg.tool_calls
            and msg.content
            and msg.content != "__FINAL__"
        ),
        None,
    )
    if not needs_retrieval:
        answer = str(final_message.content).strip() if final_message else "你好，我是“知问”企业知识助手。"
        return AgentResult(answer, "completed", [], [], None, trace, settings.mode)

    reads, citations = _evidence(messages)
    if not reads:
        search_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "search_documents":
                try:
                    search_results.append(json.loads(msg.content))
                except (TypeError, json.JSONDecodeError):
                    pass
        if any(not result.get("ok", True) for result in search_results):
            return AgentResult("检索工具执行失败，无法确认资料库是否包含答案。", "tool_error", [], ["检索未成功，不能将工具失败当作资料缺失。"], None, trace, settings.mode)
        return AgentResult("当前知识库没有找到足以回答该问题的原文依据。", "not_found", [], ["没有读取到可引用的原文。"], None, trace, settings.mode)
    values = []
    for read in reads:
        values.extend([c["text"] for c in read.get("chunks", [])])
    query_topics = terms(question) - GENERIC_TERMS
    relevant_reads = [
        read
        for read in reads
        if any(query_topics & (terms(chunk["text"]) - GENERIC_TERMS) for chunk in read.get("chunks", []))
    ]
    numeric_facts = {
        tuple(re.findall(r"\d+(?:\.\d+)?", chunk["text"]))
        for read in relevant_reads
        for chunk in read.get("chunks", [])
        if re.search(r"\d", chunk["text"])
    }
    conflict = len(relevant_reads) > 1 and len(numeric_facts) > 1
    if conflict:
        return AgentResult("检索到多份制度原文，当前证据存在可能冲突，不能在缺少适用范围或优先级时擅自选择。", "needs_clarification", citations, ["需要确认适用地区、员工范围或制度优先级。"], "请确认本问题适用的地区/员工范围，或指定优先制度。", trace, settings.mode)
    answer = str(final_message.content).strip() if final_message else "根据知识库原文：" + "；".join(dict.fromkeys(values))
    cited_text = " ".join(values)
    if any(number not in cited_text for number in re.findall(r"\d+(?:\.\d+)?", answer)):
        answer = "根据知识库原文：" + "；".join(dict.fromkeys(values))
    return AgentResult(answer, "completed", citations, [], None, trace, settings.mode)


def run_agent(store: Store, kb_id: str, question: str, history: list[dict] | None, settings: Settings, on_event=None, document_ids: list[str] | None = None) -> AgentResult:
    guard = PromptGuard()
    guard_decision = guard.inspect(question)
    if not guard_decision.allowed:
        trace = [{
            "step": 0,
            "action": "guardrail",
            "arguments": {"reason": guard_decision.reason},
            "mode": settings.mode,
        }]
        if on_event:
            on_event("trace", trace[0])
        return AgentResult(PROMPT_GUARD_ANSWER, "completed", [], [], None, trace, settings.mode)

    question = guard_decision.normalized_text
    route = RetrievalGate().decide(question, history, bool(document_ids))
    rewrite = (
        ConversationQueryRewriter().rewrite(question, history)
        if route.needed and settings.query_rewrite_enabled
        else RewriteResult(question, False)
    )
    trace: list[dict] = []
    if not route.needed:
        trace.append({
            "step": 0,
            "action": "retrieval_route",
            "arguments": {"needed": False, "reason": route.reason},
            "mode": settings.mode,
        })
    elif rewrite.applied:
        trace.append({
            "step": 0,
            "action": "query_rewrite",
            "arguments": {"original": question, "rewritten": rewrite.query},
            "result": rewrite.reason,
            "mode": settings.mode,
        })

    filtered_history = []
    for item in (history or [])[-6:]:
        if item.get("role") not in {"user", "assistant"}:
            continue
        decision = guard.inspect(str(item.get("content", "")))
        if decision.allowed:
            filtered_history.append({"role": item["role"], "content": decision.normalized_text})
        else:
            trace.append({
                "step": 0,
                "action": "history_guardrail",
                "arguments": {"role": item["role"], "reason": decision.reason},
                "mode": settings.mode,
            })

    tools = make_tools(
        store,
        kb_id,
        settings,
        question,
        rewrite.query,
        document_ids,
        rewrite.focus_terms,
    )
    graph = _build_graph(store, kb_id, settings, tools)
    messages: Annotated[list[BaseMessage], add_messages] = []
    for item in filtered_history:
        messages.append(
            HumanMessage(content=item["content"])
            if item["role"] == "user"
            else AIMessage(content=item["content"])
        )
    messages.append(HumanMessage(content=question))
    state: AgentState = {
        "messages": messages,
        "question": question,
        "history": filtered_history,
        "trace": trace,
        "step": 0,
        "tool_calls": 0,
        "mode": settings.mode,
        "needs_retrieval": route.needed,
        "routing_reason": route.reason,
        "rewritten_query": rewrite.query,
    }
    try:
        result = state
        emitted_trace = 0
        emitted_observations = set()
        for snapshot in graph.stream(state, stream_mode="values"):
            result = snapshot
            current_trace = snapshot.get("trace", [])
            if on_event:
                while emitted_trace < len(current_trace):
                    on_event("trace", current_trace[emitted_trace])
                    emitted_trace += 1
                for index, item in enumerate(current_trace):
                    if item.get("observation") and index not in emitted_observations:
                        on_event("observation", {"step": item["step"], **item["observation"]})
                        emitted_observations.add(index)
    except Exception as exc:  # noqa: BLE001 - graph failures must become an explicit tool_error result
        trace = state["trace"] + [{"step": state["step"] + 1, "action": "error", "error": str(exc)[:240], "mode": settings.mode}]
        return AgentResult("Agent 执行失败，未生成可靠答案。", "tool_error", [], ["请检查模型配置或工具运行日志。"], None, trace, settings.mode)
    final = _assemble(
        question,
        result.get("messages", []),
        result.get("trace", []),
        settings,
        route.needed,
    )
    last_action = result.get("trace", [{}])[-1].get("action") if result.get("trace") else None
    if result.get("step", 0) >= settings.max_steps and last_action != "finish" and final.status != "tool_error":
        final.status = "step_limit"
        final.gaps.append("达到 Agent 步数上限，结果未完成验证。")
    return final
