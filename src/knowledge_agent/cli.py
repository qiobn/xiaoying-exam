from __future__ import annotations

import argparse
import json

from .agent import run_agent
from .config import load_settings
from .persistence import Store


def main():
    parser = argparse.ArgumentParser(description="LangGraph enterprise knowledge agent")
    parser.add_argument("--question", required=True)
    parser.add_argument("--data", default="data/documents.jsonl")
    parser.add_argument("--kb", default="cli-demo")
    args = parser.parse_args()
    settings = load_settings()
    store = Store(settings.db_path)
    kb = next((x for x in store.list_kbs() if x["name"] == args.kb), None) or store.create_kb(args.kb)
    if not store.documents(kb["id"]):
        from pathlib import Path

        from .ingestion import build_kb, sha256
        for line in Path(args.data).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item=json.loads(line); raw=item["content"].encode(); store.add_document(kb["id"], item["document_id"]+".md", item["title"], item["version_date"], item["scope"], item["content"], sha256(raw))
        build_kb(store, kb["id"], settings)
    print(json.dumps(run_agent(store, kb["id"], args.question, [], settings).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
