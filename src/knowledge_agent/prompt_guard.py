from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_INSTRUCTION_OVERRIDE = re.compile(
    r"(?:忽略|无视|绕过|覆盖).{0,20}(?:之前|以上|前面|系统|开发者).{0,12}(?:指令|规则|约束)"
    r"|ignore.{0,30}(?:previous|system|developer).{0,20}(?:instruction|prompt|message)"
    r"|jailbreak|越狱模式|角色覆盖",
    re.IGNORECASE,
)
_SECRET_EXTRACTION = re.compile(
    r"(?:输出|显示|泄露|打印|告诉我|复述).{0,20}(?:系统提示词|system\s*prompt|隐藏指令|开发者消息|api\s*key|密钥|token)"
    r"|(?:system\s*prompt|developer\s*message).{0,20}(?:reveal|print|show|output)",
    re.IGNORECASE,
)
_DOCUMENT_INJECTION = re.compile(
    r"(?:忽略|无视|覆盖).{0,16}(?:系统|之前|以上).{0,12}(?:指令|规则)"
    r"|ignore.{0,24}(?:previous|system).{0,16}(?:instruction|prompt)"
    r"|(?:调用|执行).{0,12}(?:shell|命令|工具).{0,12}(?:泄露|删除|上传)"
    r"|(?:reveal|print).{0,16}(?:system prompt|api key|secret)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    normalized_text: str
    reason: str | None = None


class PromptGuard:
    """Cheap deterministic guard for direct prompt extraction and instruction override."""

    def normalize(self, text: str) -> str:
        value = unicodedata.normalize("NFKC", text or "")
        value = _ZERO_WIDTH.sub("", value)
        return re.sub(r"[ \t]+", " ", value).strip()

    def inspect(self, text: str) -> GuardDecision:
        normalized = self.normalize(text)
        if _INSTRUCTION_OVERRIDE.search(normalized):
            return GuardDecision(False, normalized, "instruction_override")
        if _SECRET_EXTRACTION.search(normalized):
            return GuardDecision(False, normalized, "sensitive_instruction_extraction")
        return GuardDecision(True, normalized)

    def untrusted_warnings(self, text: str) -> list[str]:
        return ["document_contains_instruction_like_text"] if _DOCUMENT_INJECTION.search(text or "") else []
