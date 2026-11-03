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
                parser_name TEXT NOT NULL DEFAULT '', parser_method TEXT NOT NULL DEFAULT '',
                parse_warnings TEXT NOT NULL DEFAULT '[]', parse_metadata TEXT NOT NULL DEFAULT '{}',
                source_path TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT '');
              CREATE TABLE IF NOT EXISTS chunks(
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, text TEXT NOT NULL, start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL, retrieval_text TEXT NOT NULL DEFAULT '',
                section_path TEXT NOT NULL DEFAULT '', chunk_method TEXT NOT NULL DEFAULT 'structured');
              CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                title TEXT NOT NULL, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS runs(
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                question TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL,
                attachments_json TEXT NOT NULL DEFAULT '[]', idempotency_key TEXT NOT NULL DEFAULT '',
                request_id TEXT NOT NULL DEFAULT '', cancel_requested INTEGER NOT NULL DEFAULT 0,
                metrics_json TEXT NOT NULL DEFAULT '{}', finished_at TEXT NOT NULL DEFAULT '');
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
            if "chunk_method" not in columns:
                db.execute("ALTER TABLE chunks ADD COLUMN chunk_method TEXT NOT NULL DEFAULT 'structured'")
            document_columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)").fetchall()}
            for name, definition in (
                ("parser_name", "TEXT NOT NULL DEFAULT ''"),
                ("parser_method", "TEXT NOT NULL DEFAULT ''"),
                ("parse_warnings", "TEXT NOT NULL DEFAULT '[]'"),
                ("parse_metadata", "TEXT NOT NULL DEFAULT '{}'"),
                ("source_path", "TEXT NOT NULL DEFAULT ''"),
                ("updated_at", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in document_columns:
                    db.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")
            run_columns = {row["name"] for row in db.execute("PRAGMA table_info(runs)").fetchall()}
            for name, definition in (
                ("attachments_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("idempotency_key", "TEXT NOT NULL DEFAULT ''"),
                ("request_id", "TEXT NOT NULL DEFAULT ''"),
                ("cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
                ("metrics_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("finished_at", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in run_columns:
                    db.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
            db.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_session_idempotency
                ON runs(session_id, idempotency_key) WHERE idempotency_key <> ''"""
            )

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

    def add_document(
        self,
        kb_id: str,
        filename: str,
        title: str,
        version_date: str,
        scope: str,
        content: str,
        sha256: str,
        parser_name: str = "",
        parser_method: str = "",
        parse_warnings: list[str] | None = None,
        parse_metadata: dict | None = None,
        source_path: str = "",
    ):
        result = self.add_documents(kb_id, [{
            "filename": filename, "title": title, "version_date": version_date, "scope": scope,
            "content": content, "sha256": sha256, "parser_name": parser_name,
            "parser_method": parser_method, "parse_warnings": parse_warnings or [],
            "parse_metadata": parse_metadata or {}, "source_path": source_path,
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
                    """INSERT INTO documents(id,kb_id,filename,title,version_date,scope,content,status,sha256,created_at,parser_name,parser_method,parse_warnings,parse_metadata,source_path,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (doc_id, kb_id, item["filename"], item["title"], item["version_date"],
                     item["scope"], item["content"], "uploaded", item["sha256"], now_iso(),
                     item.get("parser_name", ""), item.get("parser_method", ""),
                     json.dumps(item.get("parse_warnings", []), ensure_ascii=False),
                     json.dumps(item.get("parse_metadata", {}), ensure_ascii=False),
                     item.get("source_path", ""), now_iso()),
                )
                results.append((doc_id, True))
            if any(created for _, created in results):
                db.execute("UPDATE knowledge_bases SET status='dirty' WHERE id=?", (kb_id,))
        return results

    def build(self, kb_id: str, chunker):
        chunk_count = 0
        chunk_methods: dict[str, int] = {}
        with self.connect() as db:
            docs = db.execute("SELECT * FROM documents WHERE kb_id=?", (kb_id,)).fetchall()
            for doc in docs:
                db.execute("DELETE FROM chunks WHERE document_id=?", (doc["id"],))
                for i, piece in enumerate(chunker(doc["content"])):
                    if isinstance(piece, tuple):
                        text, start, end = piece
                        section_path = ""
                        chunk_method = "structured"
                    else:
                        text, start, end = piece.text, piece.start, piece.end
                        section_path = piece.section_path
                        chunk_method = getattr(piece, "method", "structured")
                    retrieval_context = getattr(piece, "retrieval_context", "") if not isinstance(piece, tuple) else ""
                    retrieval_text = "\n".join(filter(None, (
                        doc["title"], doc["scope"], section_path, retrieval_context, text,
                    )))
                    db.execute(
                        "INSERT INTO chunks(id,document_id,ordinal,text,start_offset,end_offset,retrieval_text,section_path,chunk_method) VALUES(?,?,?,?,?,?,?,?,?)",
                        (str(uuid.uuid4()), doc["id"], i, text, start, end, retrieval_text, section_path, chunk_method),
                    )
                    chunk_count += 1
                    chunk_methods[chunk_method] = chunk_methods.get(chunk_method, 0) + 1
                db.execute("UPDATE documents SET status='indexed' WHERE id=?", (doc["id"],))
            db.execute("UPDATE knowledge_bases SET status='ready', index_version=index_version+1 WHERE id=?", (kb_id,))
        return {"documents": len(docs), "chunks": chunk_count, "chunk_methods": chunk_methods}

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
            rows = db.execute("""SELECT d.id,d.kb_id,d.filename,d.title,d.version_date,d.scope,d.status,d.parser_name,d.parser_method,d.parse_warnings,d.source_path,d.updated_at,
                d.parse_metadata,length(d.content) character_count,count(c.id) chunk_count
                FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
                WHERE d.kb_id=? GROUP BY d.id ORDER BY d.created_at""", (kb_id,)).fetchall()
        values = [dict(r) for r in rows]
        for item in values:
            item["parse_warnings"] = json.loads(item["parse_warnings"] or "[]")
            item["parse_metadata"] = json.loads(item["parse_metadata"] or "{}")
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

    def set_document_source(self, document_id: str, source_path: str):
        with self.connect() as db:
            db.execute(
                "UPDATE documents SET source_path=?,updated_at=? WHERE id=?",
                (source_path, now_iso(), document_id),
            )

    def replace_document_parse(
        self,
        document_id: str,
        *,
        content: str,
        parser_name: str,
        parser_method: str,
        parse_warnings: list[str],
        parse_metadata: dict,
    ) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT kb_id FROM documents WHERE id=?",
                (document_id,),
            ).fetchone()
            if row is None:
                return None
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            db.execute(
                """UPDATE documents SET content=?,status='uploaded',parser_name=?,parser_method=?,
                parse_warnings=?,parse_metadata=?,updated_at=? WHERE id=?""",
                (
                    content,
                    parser_name,
                    parser_method,
                    json.dumps(parse_warnings, ensure_ascii=False),
                    json.dumps(parse_metadata, ensure_ascii=False),
                    now_iso(),
                    document_id,
                ),
            )
            db.execute("UPDATE knowledge_bases SET status='dirty' WHERE id=?", (row["kb_id"],))
            return row["kb_id"]

    def delete_document(self, document_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT kb_id FROM documents WHERE id=?", (document_id,)).fetchone()
            if row is None:
                return None
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
            db.execute("UPDATE knowledge_bases SET status='dirty' WHERE id=?", (row["kb_id"],))
            return row["kb_id"]

    def delete_knowledge_base(self, kb_id: str) -> bool:
        with self.connect() as db:
            cursor = db.execute("DELETE FROM knowledge_bases WHERE id=?", (kb_id,))
            return cursor.rowcount > 0

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
                """SELECT id,question,status,result_json,created_at,attachments_json,request_id,
                cancel_requested,metrics_json,finished_at FROM runs WHERE session_id=? ORDER BY created_at""",
                (session_id,),
            ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item.pop("result_json")) if item["result_json"] else None
            item["attachments"] = json.loads(item.pop("attachments_json") or "[]")
            item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
            item["cancel_requested"] = bool(item["cancel_requested"])
            values.append(item)
        return values

    def create_run(
        self,
        session_id: str,
        question: str,
        attachments: list[str] | None = None,
        *,
        idempotency_key: str = "",
        request_id: str = "",
    ):
        run_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute(
                """INSERT INTO runs(id,session_id,question,status,result_json,created_at,
                attachments_json,idempotency_key,request_id) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, session_id, question, "running", None, now_iso(),
                    json.dumps(attachments or [], ensure_ascii=False),
                    idempotency_key, request_id,
                ),
            )
        return run_id

    def find_run_by_idempotency(self, session_id: str, idempotency_key: str):
        if not idempotency_key:
            return None
        with self.connect() as db:
            row = db.execute(
                "SELECT id FROM runs WHERE session_id=? AND idempotency_key=?",
                (session_id, idempotency_key),
            ).fetchone()
        return row["id"] if row else None

    def request_cancel(self, run_id: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                return False
            if row["status"] == "running":
                db.execute("UPDATE runs SET cancel_requested=1 WHERE id=?", (run_id,))
            return True

    def is_cancel_requested(self, run_id: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT cancel_requested FROM runs WHERE id=?", (run_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def finish_run(self, run_id: str, result: dict):
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET status=?,result_json=?,finished_at=? WHERE id=?",
                (result["status"], json.dumps(result, ensure_ascii=False), now_iso(), run_id),
            )

    def update_run_metrics(self, run_id: str, metrics: dict):
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET metrics_json=? WHERE id=?",
                (json.dumps(metrics, ensure_ascii=False), run_id),
            )

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
        data["metrics"] = json.loads(data.pop("metrics_json", "{}") or "{}")
        data["cancel_requested"] = bool(data["cancel_requested"])
        return data

    def events(self, run_id: str, after_seq: int = 0):
        with self.connect() as db:
            rows=db.execute("SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq",(run_id,after_seq)).fetchall()
        return [{"run_id":r["run_id"],"seq":r["seq"],"event_type":r["event_type"],**json.loads(r["payload_json"])} for r in rows]
