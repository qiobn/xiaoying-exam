from __future__ import annotations

import re
from dataclasses import dataclass

import jieba

_FOLLOW_UP = re.compile(r"^(?:那|这个|那个|它|其|上述|刚才|还有|另外)|(?:呢|如何|怎么样)[？?]?$")
_STOPWORDS = {
    "什么", "怎么", "怎样", "如何", "多久", "多少", "是否", "可以", "需要", "应该",
    "请问", "一下", "这个", "那个", "这些", "那些", "我们", "你们", "他们", "以及",
    "还有", "但是", "如果", "因为", "所以", "关于", "相关", "进行", "一个", "有没",
}


@dataclass(frozen=True)
class RewriteResult:
    query: str
    applied: bool
    reason: str | None = None
    focus_terms: tuple[str, ...] = ()


class ConversationQueryRewriter:
    """Resolve short contextual follow-ups without inventing new facts."""

    def rewrite(self, question: str, history: list[dict] | None) -> RewriteResult:
        current = (question or "").strip()
        if not current or not history:
            return RewriteResult(current, False)
        if not _FOLLOW_UP.search(current):
            return RewriteResult(current, False)

        previous = next(
            (
                str(item.get("content", "")).strip()
                for item in reversed(history)
                if item.get("role") == "user" and str(item.get("content", "")).strip()
            ),
            "",
        )
        if not previous:
            return RewriteResult(current, False)

        previous_terms = self._keywords(previous)
        current_terms = self._keywords(current)
        if not previous_terms:
            return RewriteResult(current, False)

        # Keep the current request's new entity first, then restore the omitted topic.
        additions = [term for term in previous_terms if term not in current_terms]
        ordered = list(dict.fromkeys([*current_terms, *additions]))
        separator = "" if all(re.fullmatch(r"[\u4e00-\u9fff]+", term) for term in ordered) else " "
        query = separator.join(ordered)
        return RewriteResult(
            query or current,
            query != current,
            "context_follow_up",
            tuple(current_terms),
        )

    @staticmethod
    def _keywords(text: str) -> list[str]:
        values: list[str] = []
        for token in jieba.cut(text.lower(), cut_all=False):
            token = token.strip()
            if len(token) < 2 or token in _STOPWORDS:
                continue
            if re.fullmatch(r"[\W_]+", token):
                continue
            values.append(token)
            if len(values) >= 8:
                break
        return list(dict.fromkeys(values))
