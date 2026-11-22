"""SQLite FTS5 index over notes and prose. Stdlib only, no server, no embeddings.

Why not a vector database, which is what "memory palace" usually means: the
questions actually asked of this corpus are "what did X say about Y", "when did
we last discuss Z", "what did I decide about W". Those are keyword + entity +
time-range queries. BM25 with metadata filters answers them directly and
explains itself; embeddings answer "what paragraph is near this paragraph",
which is a different question and a worse fit for three of the four.

Vectors are still the right addition LATER, for the paraphrase case ("that
thing about phones dropping off after a firmware push" when nobody wrote
"firmware"). The schema in schema.sql leaves a pgvector column ready. But
hybrid-with-BM25-first is the design, not vectors-first, and at 14.9 MB of prose
the keyword layer alone is fast enough to feel instant.

Layout: two FTS5 tables (notes, turns) plus a plain metadata table per side, so
filters are real SQL and ranking is real BM25. The whole index is derived — it
can be deleted and rebuilt from the prose files and note directories at any
time, which is why nothing is stored only here.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import scrub
import store

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    project        TEXT,
    cwd            TEXT,
    git_branch     TEXT,
    title          TEXT,
    started        TEXT,
    ended          TEXT,
    parent_session TEXT,
    turns          INTEGER,
    human_turns    INTEGER,
    source_path    TEXT
);
CREATE INDEX IF NOT EXISTS sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS sessions_started ON sessions(started);

-- Prose. UNINDEXED columns are filters and citation data, not search targets.
CREATE VIRTUAL TABLE IF NOT EXISTS turns USING fts5(
    text,
    session_id UNINDEXED, i UNINDEXED, role UNINDEXED,
    ts UNINDEXED, project UNINDEXED, sidechain UNINDEXED,
    tokenize = 'porter unicode61'
);

-- Notes: description and body are both searchable; description is what a
-- listing shows, so a hit there is worth more than a hit deep in a body.
CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5(
    name, description, body,
    type UNINDEXED, project UNINDEXED, source UNINDEXED, session UNINDEXED,
    occurred UNINDEXED, recorded UNINDEXED, people UNINDEXED,
    entities UNINDEXED, supersedes UNINDEXED, sensitivity UNINDEXED,
    confidence UNINDEXED, path UNINDEXED, links UNINDEXED,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# --- building ---------------------------------------------------------------

def rebuild_prose(conn: sqlite3.Connection, prose_dir: Path,
                  manifest_path: Path) -> tuple[int, int]:
    """Load every extracted session's prose. Full rebuild, not incremental:
    it takes seconds and an incremental path that silently diverges from the
    files is worse than a rebuild that cannot."""
    conn.execute("DELETE FROM turns")
    conn.execute("DELETE FROM sessions")

    manifest: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {m["session_id"]: m for m in manifest}

    n_sessions = n_turns = 0
    for f in prose_dir.rglob("*.jsonl"):
        rows: list[tuple] = []
        meta: dict[str, Any] = {}
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_meta" in d:
                    meta = d["_meta"]
                    continue
                text, _n = scrub.redact_known(d.get("text") or "")
                rows.append((text, meta.get("session_id"),
                             d.get("i"), d.get("role"), d.get("ts"),
                             meta.get("project"), 1 if d.get("sidechain") else 0))
        if not meta:
            continue
        sid = meta["session_id"]
        m = by_id.get(sid, meta)
        # Titles are model-written summaries of the first prompt and can quote
        # it; they are served too, so they get the same scrub as the prose.
        title = scrub.scrub(m.get("title") or "").text or None
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, project, cwd, git_branch,"
            " title, started, ended, parent_session, turns, human_turns, source_path)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, m.get("project"), m.get("cwd"), m.get("git_branch"), title,
             m.get("started"), m.get("ended"), m.get("parent_session"),
             m.get("turns"), m.get("human_turns"), m.get("source_path")))
        conn.executemany(
            "INSERT INTO turns (text, session_id, i, role, ts, project, sidechain)"
            " VALUES (?,?,?,?,?,?,?)", rows)
        n_sessions += 1
        n_turns += len(rows)
    conn.commit()
    return n_sessions, n_turns


def rebuild_notes(conn: sqlite3.Connection, notes: Iterable[store.Note]) -> int:
    conn.execute("DELETE FROM notes")
    n = 0
    for note in notes:
        r = store.as_row(note)
        # Notes from the runtime memory folders are read where they live and
        # were never scrubbed; a cleartext device password reached recall that
        # way on 2026-10-10. Scrub at the index so every note is covered,
        # whoever wrote it.
        r["description"] = scrub.scrub(r["description"] or "").text
        r["body"] = scrub.scrub(r["body"] or "").text
        conn.execute(
            "INSERT INTO notes (name, description, body, type, project, source,"
            " session, occurred, recorded, people, entities, supersedes,"
            " sensitivity, confidence, path, links)"
            " VALUES (:name,:description,:body,:type,:project,:source,:session,"
            ":occurred,:recorded,:people,:entities,:supersedes,:sensitivity,"
            ":confidence,:path,:links)", r)
        n += 1
    conn.commit()
    return n


def set_meta(conn: sqlite3.Connection, **kv: Any) -> None:
    conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                     [(k, json.dumps(v)) for k, v in kv.items()])
    conn.commit()


def get_meta(conn: sqlite3.Connection) -> dict[str, Any]:
    return {r["key"]: json.loads(r["value"])
            for r in conn.execute("SELECT key, value FROM meta")}


# --- querying ---------------------------------------------------------------

def _fts_query(user_query: str) -> str:
    """Turn a natural phrase into an FTS5 MATCH expression.

    FTS5 treats bare punctuation as syntax, so `who owns voip-mcp?` is a syntax
    error rather than a search. Each word becomes a quoted term (safe), and the
    terms are ANDed — narrow beats broad here, because a question typed in full
    otherwise matches everything that shares its function words.
    """
    words = [w for w in (t.strip("?!.,;:()[]\"'") for t in user_query.split())
             if len(w) > 1]
    if not words:
        return '""'
    return " AND ".join(f'"{w}"' for w in words)


def _fts_query_or(user_query: str) -> str:
    words = [w for w in (t.strip("?!.,;:()[]\"'") for t in user_query.split())
             if len(w) > 2]
    return " OR ".join(f'"{w}"' for w in words) if words else '""'


def _run(conn: sqlite3.Connection, sql: str, args: list[Any]) -> list[dict[str, Any]]:
    """Execute a search, tolerating ONLY a malformed user query.

    The first version of this swallowed every OperationalError and returned [],
    which hid a real bug for a full build cycle: aliasing the FTS table as `t`
    made `turns MATCH ?` ambiguous, so every prose search raised and silently
    reported "no results". A search that finds nothing and a search that is
    broken must not look the same. Only FTS5 query-syntax errors — which come
    from what the caller typed — degrade to empty; anything else is ours and
    raises.
    """
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    except sqlite3.OperationalError as e:
        if "fts5: syntax error" in str(e).lower():
            return []
        raise


def search_notes(conn: sqlite3.Connection, query: str, *, project: str | None = None,
                 type_: str | None = None, since: str | None = None,
                 limit: int = 10) -> list[dict[str, Any]]:
    """Notes matching `query`, best first.

    Tries all-terms first and falls back to any-term: an engineer typing a full
    sentence should not get zero results because one word is absent.
    """
    for expr in (_fts_query(query), _fts_query_or(query)):
        sql = ("SELECT name, description, body, type, project, source, session,"
               " occurred, recorded, people, entities, sensitivity, path,"
               " bm25(notes, 4.0, 8.0, 1.0) AS score"
               " FROM notes WHERE notes MATCH ?")
        args: list[Any] = [expr]
        if project:
            sql += " AND project = ?"; args.append(project)
        if type_:
            sql += " AND type = ?"; args.append(type_)
        if since:
            sql += " AND occurred >= ?"; args.append(since)
        sql += " ORDER BY score LIMIT ?"
        args.append(limit)
        rows = _run(conn, sql, args)
        if rows:
            return rows
    return []


def search_turns(conn: sqlite3.Connection, query: str, *, project: str | None = None,
                 role: str | None = None, since: str | None = None,
                 limit: int = 10, context: int = 1) -> list[dict[str, Any]]:
    """Prose matching `query`, with neighbouring turns for readability.

    A single matching turn is often an answer with no question attached, so each
    hit carries `context` turns either side — that is usually the difference
    between a quotation and a usable recollection.
    """
    # bm25() takes the real TABLE name and MATCH cannot be aliased, so the FTS
    # search and its ranking happen in a subquery and the session metadata is
    # joined onto the result. Aliasing the FTS table directly fails two
    # different ways ("no such column: t" / "ambiguous column name: turns") and
    # both were silently swallowed as "no results" before _run was narrowed.
    for expr in (_fts_query(query), _fts_query_or(query)):
        inner = ("SELECT text, session_id, i, role, ts, project, sidechain,"
                 " bm25(turns, 1.0) AS score FROM turns WHERE turns MATCH ?")
        args: list[Any] = [expr]
        if project:
            inner += " AND project = ?"; args.append(project)
        if role:
            inner += " AND role = ?"; args.append(role)
        if since:
            inner += " AND ts >= ?"; args.append(since)
        inner += " ORDER BY score LIMIT ?"
        args.append(limit)
        sql = (f"SELECT q.*, s.title, s.git_branch, s.cwd FROM ({inner}) q"
               " LEFT JOIN sessions s ON s.session_id = q.session_id"
               " ORDER BY q.score")
        hits = _run(conn, sql, args)
        if not hits:
            continue
        if context:
            for h in hits:
                h["context"] = [
                    dict(r) for r in conn.execute(
                        "SELECT role, text, i FROM turns WHERE session_id = ?"
                        " AND i BETWEEN ? AND ? ORDER BY i",
                        (h["session_id"], h["i"] - context, h["i"] + context))]
        return hits
    return []


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "sessions": q("SELECT COUNT(*) FROM sessions"),
        "subagent_sessions": q("SELECT COUNT(*) FROM sessions WHERE parent_session IS NOT NULL"),
        "turns": q("SELECT COUNT(*) FROM turns"),
        "notes": q("SELECT COUNT(*) FROM notes"),
        "projects": q("SELECT COUNT(DISTINCT project) FROM sessions"),
        "earliest": q("SELECT MIN(started) FROM sessions"),
        "latest": q("SELECT MAX(ended) FROM sessions"),
        "notes_by_type": {r["type"]: r["n"] for r in conn.execute(
            "SELECT type, COUNT(*) AS n FROM notes GROUP BY type ORDER BY n DESC")},
        **get_meta(conn),
    }
