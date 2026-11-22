"""Memory Palace — MCP server over this machine's AI memory.

Two modes, same read tools:

  local (default)  stdio, started by Claude Code on this laptop. No auth, no
                   network. Adds the write tools (remember, reindex, rebuild).
  hosted           PALACE_MODE=hosted: streamable HTTP behind Entra, for
                   claude.ai and every other Claude surface. Read-only; serves
                   the scrubbed index the laptop uploads nightly. See hosted.py.

Two layers, and the tool descriptions say so, because the model choosing between
them is the difference between an answer and a wall of transcript:

  notes  — distilled, human-correctable facts. Ask these first.
  prose  — verbatim session turns. Cited, the fallback.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

import hosted
import index
import store
from instrumentation import TelemetryMiddleware, publish_identity
from telemetry import from_env as telemetry_from_env

HERE = Path(__file__).parent
HOSTED = os.environ.get("PALACE_MODE", "local").strip().lower() == "hosted"
_CFG = hosted.Config.from_env() if HOSTED else None
DB = _CFG.db_path if _CFG else HERE / "index" / "palace.db"
NOTES = HERE / "notes"
CLAUDE_HOME = Path.home() / ".claude"
logger = logging.getLogger("memory_palace")

_READ_ONLY = {"readOnlyHint": True, "destructiveHint": False,
              "idempotentHint": True, "openWorldHint": False}

# Per-tool-call usage to the OTEL collector (mcp-telemetry-kit). A no-op unless
# OTEL_EXPORTER_OTLP_ENDPOINT is set, which only the hosted stack does.
_telemetry = telemetry_from_env("memory-palace")
_telemetry.start()

mcp = FastMCP(
    name="memory-palace",
    instructions=(
        "Long-term memory across every Claude Code session, note and agent run on "
        "this machine (call `palace_stats` for current size and coverage). "
        "Use `recall` for almost everything — it "
        "searches the curated notes FIRST and falls back to verbatim session "
        "prose, and every hit is cited with its session, project and date. Use "
        "`timeline` for 'when did we…' / 'how did this evolve'; `decisions` for "
        "'what did I decide about…'; `remember` to write a new durable note when "
        "something worth keeping is established. Prefer recall over asking the "
        "user to re-explain history they have already explained once. Prose and "
        "notes are credential-scrubbed (patterns plus every secret value known "
        "on this machine), but the scrub is not perfect: treat any "
        "[REDACTED:…] marker as proof that secrets DO transit this corpus, never "
        "try to reconstruct one, and never repeat a credential-looking string "
        "a result contains."
        + (" This is the hosted, read-only copy: `remember` is not available "
           "here; notes are written from Claude Code on the owner's laptop." if HOSTED else "")
    ),
    auth=hosted.build_auth(_CFG) if HOSTED else None,
    middleware=[TelemetryMiddleware(_telemetry)],
)


def _conn():
    if not DB.exists():
        raise RuntimeError(
            "The hosted index has not been uploaded yet." if HOSTED else
            f"No index at {DB}. Build it first: python {HERE / 'build.py'}")
    if HOSTED:
        # Read-only: the file is swapped whole by /ingest, never written here.
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    return index.connect(DB)


def _since(days: int | None, since: str | None) -> str | None:
    """Accept either an explicit ISO date or a day count, because a model asked
    for 'the last two weeks' should not have to do date arithmetic."""
    if since:
        return since[:10]
    if days:
        return (date.today() - timedelta(days=days)).isoformat()
    return None


def _trim(text: str, limit: int) -> tuple[str, bool]:
    text = text.strip()
    return (text, False) if len(text) <= limit else (text[:limit].rstrip() + " …", True)


@mcp.tool(annotations={"title": "Recall", **_READ_ONLY})
def recall(query: str, project: str | None = None, days: int | None = None,
           since: str | None = None, limit: int = 6, include_prose: bool = True,
           snippet_chars: int = 700) -> dict[str, Any]:
    """Search everything remembered: distilled notes first, then verbatim session
    prose. THE default tool — reach for it whenever the answer might already be
    known rather than asking the user to repeat themselves.

    `query` is plain words (all terms must match; it retries with any-term if
    that finds nothing). Narrow with `project` (a projects/ slug), `days` (last N
    days) or `since` (ISO date). `include_prose=false` returns only the curated
    notes, which is the right choice when you want the settled answer rather
    than the discussion around it.

    Every hit carries where it came from — note path, or session id + project +
    date — so a claim can be traced instead of trusted. Prose hits include the
    surrounding turns, because a matching answer with no question attached is
    usually not enough to act on.
    """
    conn = _conn()
    cutoff = _since(days, since)
    notes = index.search_notes(conn, query, project=project, since=cutoff, limit=limit)
    out_notes = []
    for n in notes:
        body, cut = _trim(n["body"], snippet_chars)
        out_notes.append({
            "name": n["name"], "type": n["type"], "description": n["description"],
            "project": n["project"], "occurred": n["occurred"],
            "entities": [e for e in (n["entities"] or "").split(",") if e],
            "body": body, "body_truncated": cut, "path": n["path"],
        })

    out_prose = []
    if include_prose:
        for h in index.search_turns(conn, query, project=project, since=cutoff,
                                    limit=limit, context=1):
            text, cut = _trim(h["text"], snippet_chars)
            out_prose.append({
                "said_by": h["role"], "on": (h["ts"] or "")[:10],
                "session": h["session_id"], "session_title": h["title"],
                "project": h["project"], "subagent_run": bool(h["sidechain"]),
                "text": text, "text_truncated": cut,
                "before_after": [
                    {"role": c["role"], "text": _trim(c["text"], 240)[0]}
                    for c in h.get("context", []) if c["i"] != h["i"]],
            })

    return {
        "query": query,
        "scope": {"project": project or "all", "since": cutoff or "all time"},
        "notes": out_notes,
        "prose": out_prose,
        "counts": {"notes": len(out_notes), "prose": len(out_prose)},
        "guidance": ("Notes are curated and correctable; prose is verbatim and may "
                     "be superseded by a later session. When they disagree, the "
                     "note wins unless the prose is more recent."
                     if out_notes and out_prose else None),
    }


@mcp.tool(annotations={"title": "Timeline", **_READ_ONLY})
def timeline(subject: str, days: int | None = None, since: str | None = None,
             limit: int = 20) -> dict[str, Any]:
    """How a subject evolved, oldest to newest — for "when did we…", "how did we
    end up here", "has this come up before".

    This is the query a similarity search answers badly: it is chronological,
    not thematic. Returns notes and prose mentions on one time axis, so a
    decision and the discussion that led to it sit next to each other.
    """
    conn = _conn()
    cutoff = _since(days, since)
    events: list[dict[str, Any]] = []
    for n in index.search_notes(conn, subject, since=cutoff, limit=limit):
        events.append({"on": n["occurred"], "kind": f"note:{n['type']}",
                       "what": n["description"] or n["name"],
                       "where": n["project"], "ref": n["path"]})
    for h in index.search_turns(conn, subject, since=cutoff, limit=limit, context=0):
        events.append({"on": (h["ts"] or "")[:10], "kind": f"said:{h['role']}",
                       "what": _trim(h["text"], 260)[0],
                       "where": h["project"], "ref": h["session_id"]})
    events.sort(key=lambda e: e["on"] or "")
    return {"subject": subject, "since": cutoff or "all time",
            "event_count": len(events), "timeline": events}


@mcp.tool(annotations={"title": "Decisions", **_READ_ONLY})
def decisions(topic: str | None = None, project: str | None = None,
              limit: int = 20) -> dict[str, Any]:
    """Decisions and standing preferences on record — "what did I decide about X",
    "what's our position on Y", "did we already rule this out".

    Reads only notes typed `decision` or `feedback`, so it returns settled
    positions rather than the argument that produced them. An empty result means
    nothing was ever written down, NOT that no decision was made — say so rather
    than inventing one.
    """
    conn = _conn()
    rows: list[dict[str, Any]] = []
    for t in ("decision", "feedback"):
        rows += index.search_notes(conn, topic or "the", project=project,
                                   type_=t, limit=limit)
    rows.sort(key=lambda r: r["occurred"] or "", reverse=True)
    return {
        "topic": topic or "(all)",
        "count": len(rows),
        "decisions": [{"on": r["occurred"], "type": r["type"], "name": r["name"],
                       "what": r["description"], "detail": _trim(r["body"], 600)[0],
                       "project": r["project"], "path": r["path"]}
                      for r in rows[:limit]],
        "caveat": ("Only what was written down. Absence here is absence of a NOTE, "
                   "not absence of a decision."),
    }


# Write tools exist only in local mode. On the hosted copy they are not
# registered at all (absent from tools/list), since the next nightly upload
# would silently overwrite anything written there.
_local_tool = (lambda **_kw: (lambda f: f)) if HOSTED else mcp.tool


@_local_tool(annotations={"title": "Remember", "readOnlyHint": False,
                       "destructiveHint": False, "idempotentHint": False,
                       "openWorldHint": False})
def remember(name: str, description: str, body: str, type: str = "insight",
             project: str | None = None, entities: list[str] | None = None,
             occurred: str | None = None, supersedes: str | None = None,
             confidence: str = "stated") -> dict[str, Any]:
    """Write a durable note into the palace — the write half of memory.

    Use when something worth keeping is ESTABLISHED: a decision, a corrected
    assumption, a non-obvious fact about a system, a preference the user stated.
    One claim per note; a note covering three things is retrievable for none of
    them.

    Do NOT use it for: what the code already records, what only matters inside
    the current conversation, or a guess. `type` is decision | insight | project
    | feedback | user | reference. `supersedes` takes an existing note's name
    when this replaces it — that is how the palace stays current instead of
    accumulating contradictions.

    The note lands as a markdown file and is searchable after the next build (or
    immediately via `reindex_notes`).
    """
    try:
        note = store.Note(
            name=name, description=description, body=body, type=type,
            project=project, source="claude-code", entities=entities or [],
            occurred=occurred, supersedes=supersedes, confidence=confidence)
    except ValueError as e:
        return {"status": "rejected", "reason": str(e)}
    path = store.write_note(note, NOTES)
    return {"status": "written", "path": str(path), "name": note.name,
            "type": note.type, "occurred": note.occurred,
            "next": "Call reindex_notes to make it searchable now."}


@_local_tool(annotations={"title": "Reindex Notes", "readOnlyHint": False,
                       "destructiveHint": False, "idempotentHint": True,
                       "openWorldHint": False})
def reindex_notes() -> dict[str, Any]:
    """Re-read every note directory into the index. Seconds, and safe to repeat.

    Only the NOTE side — re-extracting session prose is the slower batch job in
    build.py. Call this after `remember`, or after editing a note by hand.
    """
    conn = _conn()
    dirs = store.note_dirs(CLAUDE_HOME, NOTES)
    n = index.rebuild_notes(conn, store.iter_notes(dirs))
    return {"status": "ok", "notes_indexed": n, "note_dirs": len(dirs)}


@mcp.tool(annotations={"title": "Palace Stats", **_READ_ONLY})
def palace_stats() -> dict[str, Any]:
    """What is actually in here — corpus size, coverage, date range, note mix.

    Worth calling before telling the user the palace does not know something:
    a project with no sessions indexed is a coverage gap, not an absence of
    history.
    """
    s = index.stats(_conn())
    s["biggest_projects"] = [
        dict(r) for r in _conn().execute(
            "SELECT project, COUNT(*) AS sessions, SUM(turns) AS turns"
            " FROM sessions GROUP BY project ORDER BY turns DESC LIMIT 8")]
    # Freshness, so a stalled nightly shows up in the answer instead of
    # quietly serving last week's memory as if it were current.
    s["freshness"] = hosted.index_health(DB)
    s.pop("note_dirs", None)       # laptop paths; noise to a reader
    return s


@mcp.tool(annotations={"title": "What can this do", **_READ_ONLY})
def capabilities() -> dict[str, Any]:
    """Call this when asked what the Memory Palace is, what it can answer, or
    how to use it."""
    return {
        "what": ("The owner's long-term working memory: every Claude Code session, "
                 "distilled note and agent run since April 2026, searchable and "
                 "cited, with credentials scrubbed."),
        "ask_it": [
            "Have we hit this before? (a symptom, an error code, a customer)",
            "When did we decide X, and why?",
            "What do we know about <customer / system / vendor>?",
            "How did <project> evolve?",
        ],
        "tools": {"recall": "almost everything; notes first, then cited prose",
                  "timeline": "chronological: when did we, how did this evolve",
                  "decisions": "settled positions and stated preferences only",
                  "palace_stats": "coverage and freshness; check before saying it doesn't know"},
        "limits": [
            "Keyword search: use the words the conversation would have used.",
            "Prose is verbatim and can be superseded by a later session; notes win when they disagree.",
            ("Read-only here; notes are written from Claude Code on the owner's laptop."
             if HOSTED else "remember writes notes; reindex_notes makes them searchable now."),
        ],
    }


@mcp.prompt(title="Have We Seen This")
def seen_this_before(symptom: str) -> str:
    """Check whether a problem has come up before and how it was resolved."""
    return (f"Use the memory-palace connector for this, and only that connector: its "
            f"tools are recall, timeline, decisions and palace_stats. Has this come up "
            f"before: {symptom}? Call recall with the most distinctive words from it "
            f"(error codes, hostnames, customer names), then timeline if there are "
            f"several hits. Cite each finding with its session or note and date. If "
            f"nothing matches, say so plainly and check palace_stats for coverage "
            f"before concluding it is new.")


@mcp.prompt(title="Decision Log")
def decision_log(topic: str) -> str:
    """What was decided about a topic, and when."""
    return (f"Use the memory-palace connector for this, and only that connector. "
            f"Call decisions for '{topic}', then recall for the discussion behind the "
            f"most recent one. List each decision with its date and source. An empty "
            f"result means nothing was written down, not that nothing was decided; "
            f"say that rather than inferring one.")


@_local_tool(annotations={"title": "Rebuild Palace", "readOnlyHint": False,
                       "destructiveHint": False, "idempotentHint": True,
                       "openWorldHint": False})
def rebuild(full: bool = False) -> dict[str, Any]:
    """Re-extract new session prose and rebuild the whole index.

    Incremental by default (only transcripts newer than the last watermark), so
    a normal run is seconds; `full=true` re-reads all 447 transcripts and takes
    about half a minute. The nightly distiller calls this, so a manual run is
    only needed to pick up sessions from the last few hours.
    """
    proc = subprocess.run(
        ["python", str(HERE / "build.py")] + (["--full"] if full else []),
        capture_output=True, text=True, timeout=900, cwd=HERE)
    return {"status": "ok" if proc.returncode == 0 else "failed",
            "output": (proc.stdout or "").strip().splitlines(),
            "errors": (proc.stderr or "").strip().splitlines()[-5:] or None}


if HOSTED:
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        # Always 200 while the process is up (the container healthcheck), with
        # the index state in the body: missing / stale is reported, not hidden.
        return JSONResponse({"status": "ok", **hosted.index_health(DB)})

    @mcp.custom_route("/ingest", methods=["POST"])
    async def ingest(request: Request) -> JSONResponse:
        """The laptop's nightly upload of the finished, scrubbed palace.db
        (gzipped). Bearer token AND a private source address, because this is
        the one route that changes what the server serves."""
        import anyio
        client = request.client.host if request.client else None
        if hosted.via_proxy(request.headers) or not hosted.client_allowed(_CFG, client):
            logger.warning("ingest refused: source %s not in allowlist", client)
            return JSONResponse({"status": "forbidden"}, status_code=403)
        if not hosted.token_ok(_CFG, request.headers.get("authorization")):
            logger.warning("ingest refused: bad token from %s", client)
            return JSONResponse({"status": "unauthorized"}, status_code=401)
        # Spool the body to the data volume as it arrives, never into memory:
        # await request.body() held the whole upload (twice, while joining)
        # and the 512 MB container was killed mid-ingest on 2026-11-05.
        import tempfile
        DB.parent.mkdir(parents=True, exist_ok=True)
        fd, spool = tempfile.mkstemp(prefix=".upload-", suffix=".gz", dir=DB.parent)
        os.close(fd)
        try:
            received = 0
            async with await anyio.open_file(spool, "wb") as fh:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > hosted.MAX_DB_BYTES:
                        logger.warning("ingest rejected from %s: body over ceiling", client)
                        return JSONResponse({"status": "rejected",
                                             "reason": "upload exceeds size ceiling"},
                                            status_code=413)
                    await fh.write(chunk)

            def _install():
                with open(spool, "rb") as src:
                    return hosted.install_upload(src, DB)
            try:
                result = await anyio.to_thread.run_sync(_install)
            except ValueError as e:      # a bad upload: report, keep serving the old index
                logger.warning("ingest rejected from %s: %s", client, e)
                return JSONResponse({"status": "rejected", "reason": str(e)}, status_code=422)
        finally:
            try:
                os.unlink(spool)
            except FileNotFoundError:    # already gone: nothing to clean up
                pass
        logger.info("ingest installed from %s: %s", client,
                    {k: result[k] for k in ("bytes", "sessions", "turns", "notes")})
        return JSONResponse(result)


def main() -> None:
    if HOSTED:
        _CFG.validate()
        logger.info("memory-palace hosted on %s:%s%s, auth=%s, db=%s",
                    _CFG.host, _CFG.port, _CFG.path,
                    "Entra JWT (resource-server)" if _CFG.auth_enabled else "DISABLED",
                    DB)
        # Before run: an idle server must still appear in Grafana, with a zero
        # row per tool, or "which tools does nobody use" has no answer.
        publish_identity(mcp, _telemetry)
        mcp.run(transport="http", host=_CFG.host, port=_CFG.port,
                path=_CFG.path, stateless_http=True)
        return
    # `remember` writes whatever a chat hands it; make sure it can redact every
    # secret value already on disk. The transcript sweep is too slow for a
    # server start and is left to build.py.
    import known_secrets
    known_secrets.install(include_transcripts=False)
    mcp.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
