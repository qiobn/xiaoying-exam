from __future__ import annotations

import re
from dataclasses import dataclass

_CHAT = re.compile(
    r"^(?:你好|您好|嗨|哈喽|hi|hello|早上好|下午好|晚上好|谢谢|感谢|好的|收到|再见|拜拜)"
    r"|(?:你是谁|你能做什么|你会什么|介绍一下你|讲个笑话|聊聊天|测试一下|在吗)",
    re.IGNORECASE,
)
_TASK_WITHOUT_KB = re.compile(r"(?:写|翻译|润色|总结).{0,12}(?:一段|一封|这个句子|英文|中文)")
_KB_REFERENCE = re.compile(r"根据|按照|依据|引用|检索|查找|制度|政策|规定|条款|文档|知识库")
_QUESTION = re.compile(r"[？?]|(?:吗|呢|多少|多久|几天|几点|哪里|哪个|如何|怎么|是否|有没有|能不能)")
_KNOWLEDGE_TERMS = {
    "制度", "政策", "规则", "流程", "报销", "差旅", "发票", "年假", "请假", "考勤",
    "薪资", "绩效", "培训", "审批", "采购", "合同", "员工", "办公", "产品", "价格",
    "参数", "功能", "错误码", "故障", "账号", "权限", "faq", "sop", "知识库", "文档",
}


@dataclass(frozen=True)
class RetrievalDecision:
    needed: bool
    reason: str


class RetrievalGate:
    """Conservative pre-router: skip RAG for clear social chat, allow it for business questions."""

    def decide(self, question: str, history: list[dict] | None = None, has_attachments: bool = False) -> RetrievalDecision:
        text = (question or "").strip()
        lowered = text.lower()
        if has_attachments:
            return RetrievalDecision(True, "attachments_require_retrieval")
        if not text:
            return RetrievalDecision(False, "empty_chat")
        if _TASK_WITHOUT_KB.search(text) and not _KB_REFERENCE.search(text):
            return RetrievalDecision(False, "general_writing_task")
        if any(term in lowered for term in _KNOWLEDGE_TERMS):
            return RetrievalDecision(True, "knowledge_term")
        if _CHAT.search(text):
            return RetrievalDecision(False, "social_or_capability_chat")
        if _QUESTION.search(text):
            previous = " ".join(
                str(item.get("content", ""))
                for item in (history or [])
                if item.get("role") == "user"
            )
            if any(term in previous.lower() for term in _KNOWLEDGE_TERMS):
                return RetrievalDecision(True, "contextual_knowledge_question")
            return RetrievalDecision(True, "unresolved_factual_question")
        return RetrievalDecision(False, "ordinary_chat")
