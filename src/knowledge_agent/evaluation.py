from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import run_agent
from .config import Settings
from .persistence import Store
from .rag import KnowledgeRetrievalService


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    expected_titles: list[str]
    expected_status: str


@dataclass
class EvaluationResult:
    case_id: str
    question: str
    retrieved_titles: list[str]
    recall: float
    reciprocal_rank: float
    status: str | None = None
    status_match: bool | None = None


def load_cases(path: Path) -> list[EvaluationCase]:
    cases = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        cases.append(EvaluationCase(
            id=value["id"],
            question=value["question"],
            expected_titles=list(value.get("expected_titles", [])),
            expected_status=value.get("expected_status", "completed"),
        ))
    return cases


def evaluate(
    store: Store,
    kb_id: str,
    settings: Settings,
    cases: list[EvaluationCase],
    *,
    include_agent: bool = False,
    top_k: int = 5,
) -> dict:
    retriever = KnowledgeRetrievalService(store, kb_id, settings)
    details: list[EvaluationResult] = []
    for case in cases:
        outcome = retriever.retrieve(case.question, top_k)
        titles = list(dict.fromkeys(hit["title"] for hit in outcome.hits))
        expected = set(case.expected_titles)
        if expected:
            recall = len(expected & set(titles)) / len(expected)
            rank = next(
                (index + 1 for index, title in enumerate(titles) if title in expected),
                0,
            )
            reciprocal_rank = 1.0 / rank if rank else 0.0
        else:
            recall = 1.0 if not titles else 0.0
            reciprocal_rank = 1.0 if not titles else 0.0
        status = None
        status_match = None
        if include_agent:
            result = run_agent(store, kb_id, case.question, [], settings)
            status = result.status
            status_match = status == case.expected_status
        details.append(EvaluationResult(
            case_id=case.id,
            question=case.question,
            retrieved_titles=titles,
            recall=recall,
            reciprocal_rank=reciprocal_rank,
            status=status,
            status_match=status_match,
        ))
    count = max(1, len(details))
    summary = {
        "cases": len(details),
        "recall_at_k": round(sum(item.recall for item in details) / count, 4),
        "mrr": round(sum(item.reciprocal_rank for item in details) / count, 4),
        "agent_status_accuracy": (
            round(sum(bool(item.status_match) for item in details) / count, 4)
            if include_agent else None
        ),
        "top_k": top_k,
        "include_agent": include_agent,
        "mode": settings.mode,
    }
    return {"summary": summary, "details": [asdict(item) for item in details]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and optional Agent behavior")
    parser.add_argument("--db", default="runtime/knowledge.db")
    parser.add_argument("--kb", default="")
    parser.add_argument("--cases", default="data/evaluation_cases.jsonl")
    parser.add_argument("--output", default="")
    parser.add_argument("--include-agent", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    settings = Settings(db_path=Path(args.db))
    store = Store(settings.db_path)
    kb = store.get_kb(args.kb) if args.kb else (store.list_kbs() or [None])[0]
    if not kb:
        raise SystemExit("没有可评测的知识库")
    report = evaluate(
        store,
        kb["id"],
        settings,
        load_cases(Path(args.cases)),
        include_agent=args.include_agent,
        top_k=args.top_k,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
