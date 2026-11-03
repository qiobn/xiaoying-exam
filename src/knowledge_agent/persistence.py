from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from .schemas import now_iso


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(self):
        with self.connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS knowledge_bases(
                id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'empty', index_version INTEGER NOT NULL DEFAULT 0);
              CREATE TABLE IF NOT EXISTS documents(
                id TEXT PRIMARY KEY, kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                filename TEXT NOT NULL, title TEXT NOT NULL, version_date TEXT NOT NULL,
                scope TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL,
                sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
                parser_name TEXT NOT NULL DEFAULT '', parser_method TEXT NOT NULL DEFAULT '', parse_warnings TEXT NOT NULL DEFAULT '[]');
              CREATE TABLE IF NOT EXISTS chunks(
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, text TEXT NOT NULL, start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL, retrieval_text TEXT NOT NULL DEFAULT '',
                section_path TEXT NOT NULL DEFAULT '');
              CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                title TEXT NOT NULL, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS runs(
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                question TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL,
                attachments_json TEXT NOT NULL DEFAULT '[]');
              CREATE TABLE IF NOT EXISTS events(
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, seq INTEGER NOT NULL,
                event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(run_id, seq));
              CREATE TABLE IF NOT EXISTS build_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS chunk_embeddings(
                chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                model TEXT NOT NULL, vector_json TEXT NOT NULL, created_at TEXT NOT NULL);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(chunks)").fetchall()}
            if "retrieval_text" not in columns:
                db.execute("ALTER TABLE chunks ADD COLUMN retrieval_text TEXT NOT NULL DEFAULT ''")
            if "section_path" not in columns:
                db.execute("ALTER TABLE chunks ADD COLUMN section_path TEXT NOT NULL DEFAULT ''")
            document_columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)").fetchall()}
            for name, definition in (("parser_name", "TEXT NOT NULL DEFAULT ''"), ("parser_method", "TEXT NOT NULL DEFAULT ''"), ("parse_warnings", "TEXT NOT NULL DEFAULT '[]'")):
                if name not in document_columns:
                    db.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")
            run_columns = {row["name"] for row in db.execute("PRAGMA table_info(runs)").fetchall()}
            if "attachments_json" not in run_columns:
                db.execute("ALTER TABLE runs ADD COLUMN attachments_json TEXT NOT NULL DEFAULT '[]'")

    def create_kb(self, name: str) -> dict:
        name = name.strip()
        if not name:
            raise ValueError("知识库名称不能为空")
        if len(name) > 80:
            raise ValueError("知识库名称不能超过 80 个字符")
        row = {"id": str(uuid.uuid4()), "name": name, "created_at": now_iso(), "status": "empty", "index_version": 0}
        with self.connect() as db:
            db.execute("INSERT INTO knowledge_bases(id,name,created_at,status,index_version) VALUES(?,?,?,?,?)", tuple(row.values()))
        return row

    def list_kbs(self):
        with self.connect() as db:
            rows = db.execute("""SELECT k.*, count(DISTINCT d.id) document_count,
                count(c.id) chunk_count FROM knowledge_bases k
                LEFT JOIN documents d ON d.kb_id=k.id LEFT JOIN chunks c ON c.document_id=d.id
                GROUP BY k.id ORDER BY k.created_at""").fetchall()
        return [dict(r) for r in rows]

    def get_kb(self, kb_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
        return dict(row) if row else None

    def add_document(self, kb_id: str, filename: str, title: str, version_date: str, scope: str, content: str, sha256: str, parser_name: str = "", parser_method: str = "", parse_warnings: list[str] | None = None):
        result = self.add_documents(kb_id, [{
            "filename": filename, "title": title, "version_date": version_date, "scope": scope,
            "content": content, "sha256": sha256, "parser_name": parser_name,
            "parser_method": parser_method, "parse_warnings": parse_warnings or [],
        }])
        return result[0]

    def add_documents(self, kb_id: str, documents: list[dict]):
        if not self.get_kb(kb_id):
            raise ValueError("知识库不存在")
        results = []
        with self.connect() as db:
            for item in documents:
                duplicate = db.execute(
                    "SELECT id FROM documents WHERE kb_id=? AND sha256=?",
                    (kb_id, item["sha256"]),
                ).fetchone()
                if duplicate:
                    results.append((duplicate["id"], False))
                    continue
                doc_id = str(uuid.uuid4())
                db.execute(
                    """INSERT INTO documents(id,kb_id,filename,title,version_date,scope,content,status,sha256,created_at,parser_name,parser_method,parse_warnings)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (doc_id, kb_id, item["filename"], item["title"], item["version_date"],
                     item["scope"], item["content"], "uploaded", item["sha256"], now_iso(),
                     item.get("parser_name", ""), item.get("parser_method", ""),
                     json.dumps(item.get("parse_warnings", []), ensure_ascii=False)),
                )
                results.append((doc_id, True))
            if any(created for _, created in results):
                db.execute("UPDATE knowledge_bases SET status='dirty' WHERE id=?", (kb_id,))
        return results

    def build(self, kb_id: str, chunker):
        with self.connect() as db:
            docs = db.execute("SELECT * FROM documents WHERE kb_id=?", (kb_id,)).fetchall()
            for doc in docs:
                db.execute("DELETE FROM chunks WHERE document_id=?", (doc["id"],))
                for i, piece in enumerate(chunker(doc["content"])):
                    if isinstance(piece, tuple):
                        text, start, end = piece
                        section_path = ""
                    else:
                        text, start, end = piece.text, piece.start, piece.end
                        section_path = piece.section_path
                    retrieval_text = "\n".join(filter(None, (doc["title"], doc["scope"], section_path, text)))
                    db.execute(
                        "INSERT INTO chunks(id,document_id,ordinal,text,start_offset,end_offset,retrieval_text,section_path) VALUES(?,?,?,?,?,?,?,?)",
                        (str(uuid.uuid4()), doc["id"], i, text, start, end, retrieval_text, section_path),
                    )
                db.execute("UPDATE documents SET status='indexed' WHERE id=?", (doc["id"],))
            db.execute("UPDATE knowledge_bases SET status='ready', index_version=index_version+1 WHERE id=?", (kb_id,))
        return len(docs)

    def clear_embeddings(self, kb_id: str):
        with self.connect() as db:
            db.execute("DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.kb_id=?)", (kb_id,))

    def save_embeddings(self, values: list[tuple[str, str, list[float]]]):
        if not values:
            return
        with self.connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO chunk_embeddings(chunk_id,model,vector_json,created_at) VALUES(?,?,?,?)",
                [(chunk_id, model, json.dumps(vector), now_iso()) for chunk_id, model, vector in values],
            )

    def documents(self, kb_id: str):
        with self.connect() as db:
            rows = db.execute("""SELECT d.id,d.kb_id,d.filename,d.title,d.version_date,d.scope,d.status,d.parser_name,d.parser_method,d.parse_warnings,
                length(d.content) character_count,count(c.id) chunk_count
                FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
                WHERE d.kb_id=? GROUP BY d.id ORDER BY d.created_at""", (kb_id,)).fetchall()
        values = [dict(r) for r in rows]
        for item in values:
            item["parse_warnings"] = json.loads(item["parse_warnings"] or "[]")
        return values

    def chunks(self, document_id: str):
        with self.connect() as db:
            rows = db.execute("SELECT c.*,e.model embedding_model,e.vector_json embedding_vector FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id WHERE c.document_id=? ORDER BY c.ordinal", (document_id,)).fetchall()
        return [dict(r) for r in rows]

    def all_chunks(self, kb_id: str):
        with self.connect() as db:
            rows = db.execute("""SELECT c.*,d.title,d.version_date,d.scope,d.content document_content,e.model embedding_model,e.vector_json embedding_vector
                FROM chunks c JOIN documents d ON d.id=c.document_id LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id
                WHERE d.kb_id=? AND d.status='indexed' ORDER BY d.id,c.ordinal""", (kb_id,)).fetchall()
        return [dict(r) for r in rows]

    def add_build_event(self, kb_id: str, event_type: str, payload: dict):
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO build_events(kb_id,event_type,payload_json,created_at) VALUES(?,?,?,?)",
                (kb_id, event_type, json.dumps(payload, ensure_ascii=False), now_iso()),
            )
            return {"id": cursor.lastrowid, "kb_id": kb_id, "event_type": event_type, **payload}

    def build_events(self, kb_id: str, limit: int = 100):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM build_events WHERE kb_id=? ORDER BY id DESC LIMIT ?",
                (kb_id, max(1, min(limit, 500))),
            ).fetchall()
        return [
            {"id": row["id"], "kb_id": row["kb_id"], "event_type": row["event_type"],
             **json.loads(row["payload_json"]), "created_at": row["created_at"]}
            for row in reversed(rows)
        ]

    def get_document(self, document_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return dict(row) if row else None

    def create_session(self, kb_id: str, title: str):
        if not self.get_kb(kb_id):
            raise ValueError("知识库不存在")
        title = title.strip() or "新会话"
        row = {"id": str(uuid.uuid4()), "kb_id": kb_id, "title": title, "created_at": now_iso()}
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?,?)", tuple(row.values()))
        return row

    def sessions(self):
        with self.connect() as db:
            rows = db.execute("""SELECT s.*, (SELECT count(*) FROM runs r WHERE r.session_id=s.id) run_count
                FROM sessions s ORDER BY created_at DESC""").fetchall()
        return [dict(r) for r in rows]

    def session_runs(self, session_id: str):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,question,status,result_json,created_at,attachments_json FROM runs WHERE session_id=? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item.pop("result_json")) if item["result_json"] else None
            item["attachments"] = json.loads(item.pop("attachments_json") or "[]")
            values.append(item)
        return values

    def create_run(self, session_id: str, question: str, attachments: list[str] | None = None):
        run_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs(id,session_id,question,status,result_json,created_at,attachments_json) VALUES(?,?,?,?,?,?,?)",
                (run_id, session_id, question, "running", None, now_iso(), json.dumps(attachments or [], ensure_ascii=False)),
            )
        return run_id

    def finish_run(self, run_id: str, result: dict):
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,result_json=? WHERE id=?", (result["status"],json.dumps(result,ensure_ascii=False),run_id))

    def add_event(self, run_id: str, event_type: str, payload: dict):
        with self.connect() as db:
            seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (run_id,seq,event_type,json.dumps(payload,ensure_ascii=False),now_iso()))
        return {"run_id":run_id,"seq":seq,"event_type":event_type,**payload}

    def run(self, run_id: str):
        with self.connect() as db:
            row=db.execute("SELECT * FROM runs WHERE id=?",(run_id,)).fetchone()
        if not row: return None
        data=dict(row)
        data["result"] = json.loads(data.pop("result_json")) if data["result_json"] else None
        data["attachments"] = json.loads(data.pop("attachments_json", "[]") or "[]")
        return data

    def events(self, run_id: str, after_seq: int = 0):
        with self.connect() as db:
            rows=db.execute("SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq",(run_id,after_seq)).fetchall()
        return [{"run_id":r["run_id"],"seq":r["seq"],"event_type":r["event_type"],**json.loads(r["payload_json"])} for r in rows]
