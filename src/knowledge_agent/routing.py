from __future__ import annotations

import re
from dataclasses import dataclass

_CHAT = re.compile(
    r"^(?:你好|您好|嗨|哈喽|hi|hello|早上好|下午好|晚上好|谢谢|感谢|好的|收到|再见|拜拜)"
    r"|(?:你是谁|你能做什么|你能干什么|你会什么|介绍一下你|讲个笑话|聊聊天|测试一下|在吗)",
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
_OPERATION_TERMS = {
    "上传", "构建知识库", "知识库构建", "分块", "检索调试", "停止运行",
    "取消运行", "设置", "配置", "embedding", "rerank", "模型连接", "操作",
    "使用说明", "怎么使用", "日志", "会话", "导出", "重建", "重新解析",
    "删除文档", "附件",
}
_OUT_OF_SCOPE = re.compile(
    r"(?:天气|气温|下雨|新闻|股票|基金|彩票|比赛|比分|电影|音乐|菜谱|做饭|"
    r"旅游攻略|航班|火车票|汇率|写代码|编程题)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalDecision:
    needed: bool
    reason: str
    kind: str = "chat"


class RetrievalGate:
    """Conservative pre-router: skip RAG for clear social chat, allow it for business questions."""

    def decide(self, question: str, history: list[dict] | None = None, has_attachments: bool = False) -> RetrievalDecision:
        text = (question or "").strip()
        lowered = text.lower()
        if has_attachments:
            return RetrievalDecision(True, "attachments_require_retrieval", "knowledge")
        if not text:
            return RetrievalDecision(False, "empty_chat", "chat")
        if any(term in lowered for term in _OPERATION_TERMS):
            return RetrievalDecision(True, "operation_guide", "operation")
        if _TASK_WITHOUT_KB.search(text) and not _KB_REFERENCE.search(text):
            return RetrievalDecision(False, "general_writing_task", "chat")
        if any(term in lowered for term in _KNOWLEDGE_TERMS):
            return RetrievalDecision(True, "knowledge_term", "knowledge")
        if _CHAT.search(text):
            return RetrievalDecision(False, "social_or_capability_chat", "chat")
        if _OUT_OF_SCOPE.search(text):
            return RetrievalDecision(False, "outside_knowledge_scope", "out_of_scope")
        if _QUESTION.search(text):
            previous = " ".join(
                str(item.get("content", ""))
                for item in (history or [])
                if item.get("role") == "user"
            )
            if any(term in previous.lower() for term in _OPERATION_TERMS):
                return RetrievalDecision(True, "contextual_operation_question", "operation")
            if any(term in previous.lower() for term in _KNOWLEDGE_TERMS):
                return RetrievalDecision(True, "contextual_knowledge_question", "knowledge")
            return RetrievalDecision(True, "unresolved_factual_question", "knowledge")
        return RetrievalDecision(False, "ordinary_chat", "chat")
