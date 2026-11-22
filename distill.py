"""Turn session prose into atomic notes — the batch half of memory capture.

    python distill.py --limit 5            # distill up to 5 pending sessions
    python distill.py --dry-run            # show what it WOULD distill
    python distill.py --session <id>       # one specific session

Design: the SessionEnd hook is O(1) (append a session id to a queue) and this
drains that queue on a schedule. A hook that called a model on every session end
would tax every exit to serve a nightly need, and would fire mid-work every time
a session was cleared or resumed.

The model is invoked through the `claude -p` CLI rather than the API directly, so
it uses the existing OAuth login and there is no API key to store or rotate.

Two guards worth knowing about:

* The queue is not trusted as the only source of truth. A hook that silently
  breaks looks exactly like a quiet week, so pending work is the UNION of the
  queue and any indexed session that has never been distilled. The queue is an
  optimisation, not the ledger.

* Distilled state is recorded per session id, so re-running is cheap and safe.
  A session is never distilled twice unless --force says so.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import known_secrets
import store

HERE = Path(__file__).parent
QUEUE = HERE / "state" / "queue.jsonl"
DISTILLED = HERE / "state" / "distilled.json"
PROSE = HERE / "index" / "prose"
MANIFEST = HERE / "index" / "manifest.json"
NOTES = HERE / "notes"

# Below this, a session is a fragment — a one-question lookup or an aborted
# start. Distilling it costs a model call to produce nothing worth keeping.
MIN_CHARS = 2000
MIN_HUMAN_TURNS = 2
# Cap the prose sent per session. The longest sessions here run past 200 KB of
# prose; the distiller wants the shape of the work, not every word.
MAX_PROMPT_CHARS = 60_000

INSTRUCTIONS = """\
You are the distiller for a long-term memory store. You are reading the prose of \
ONE Claude Code session (tool calls and output already stripped) and writing the \
few durable facts worth remembering months from now.

Return ONLY a JSON array, no prose around it. Each element:

  {"name": "kebab-case-slug",
   "description": "one line, <=140 chars, what this note asserts",
   "type": "decision|insight|project|feedback|reference",
   "body": "2-6 sentences. State the fact. If it is a correction or a trap, say \
what the wrong assumption was and why it was wrong.",
   "entities": ["system-or-tool-names", "customer-names"],
   "confidence": "stated|inferred|verified"}

WRITE a note for:
  * a decision made, and the reason for it
  * a non-obvious fact about a system, verified during the session
  * a corrected assumption or a trap someone would hit again
  * a preference or working instruction the user stated
  * an unresolved commitment that outlives the session

DO NOT write a note for:
  * anything the code, tests, README or git history already record
  * what was done step by step — this is not a changelog
  * routine successes ("the tests passed", "the build worked")
  * anything you are guessing at. Fewer, truer notes beat more notes.

Use "verified" only when the session actually proved it (a live call, a passing \
test, an observed output). One claim per note. Most sessions yield 0-3 notes; an \
empty array [] is a valid and common answer.\
"""


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def queued_sessions() -> list[str]:
    if not QUEUE.exists():
        return []
    out: list[str] = []
    for line in QUEUE.read_text(encoding="utf-8").splitlines():
        try:
            sid = json.loads(line).get("session_id")
        except json.JSONDecodeError:
            continue
        if sid and sid not in out:
            out.append(sid)
    return out


def pending(force: bool = False) -> list[dict]:
    """Sessions worth distilling: the queue UNION anything never distilled.

    The union matters. If the SessionEnd hook breaks, the queue goes quiet and
    looks identical to a week with no sessions — so the manifest is checked too
    and the queue is treated as a hint about ordering, not as the ledger.
    """
    manifest = _load_json(MANIFEST, [])
    done = set() if force else set(_load_json(DISTILLED, {}).get("sessions", []))
    by_id = {m["session_id"]: m for m in manifest}

    ordered = [by_id[s] for s in queued_sessions() if s in by_id]
    ordered += [m for m in manifest if m["session_id"] not in
                {o["session_id"] for o in ordered}]

    out = []
    for m in ordered:
        if m["session_id"] in done:
            continue
        if m.get("chars", 0) < MIN_CHARS or m.get("human_turns", 0) < MIN_HUMAN_TURNS:
            continue
        out.append(m)
    return out


def prose_for(session: dict) -> str:
    f = PROSE / session["project"] / f"{session['session_id']}.jsonl"
    if not f.exists():
        return ""
    parts: list[str] = []
    for line in f.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_meta" in d:
            continue
        who = "USER" if d.get("role") == "human" else "ASSISTANT"
        parts.append(f"[{who}] {d.get('text','').strip()}")
    text = "\n\n".join(parts)
    if len(text) <= MAX_PROMPT_CHARS:
        return text
    # Keep the head and the tail: a session's framing is at the start and its
    # conclusions are at the end. The middle is where the grinding happens.
    half = MAX_PROMPT_CHARS // 2
    return text[:half] + "\n\n[... middle of session omitted ...]\n\n" + text[-half:]


def call_model(prompt: str, model: str = "sonnet", timeout: int = 300) -> str:
    """Ask `claude -p` for the notes.

    MUST run from a normal terminal or a scheduled task, NOT from inside a
    sandboxed Claude Code Bash call: a nested `claude` cannot reach the parent's
    OAuth credentials and comes back `401 OAuth access token has been revoked`.
    Verified the hard way — this is why the failure message below names the cause
    instead of printing a bare exit code.

    Note the CLI reports API failures as `is_error` in a SUCCESSFUL exit, so the
    return code alone is not a health check.
    """
    proc = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--model", model,
         "--allowed-tools", ""],
        input=prompt, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace")

    payload = {}
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {}

    if payload.get("is_error") or payload.get("api_error_status"):
        detail = payload.get("result") or f"status {payload.get('api_error_status')}"
        hint = ""
        if str(payload.get("api_error_status")) == "401":
            hint = (" — run distill.py from your own terminal (or a scheduled "
                    "task); a nested claude inside a Claude Code sandbox cannot "
                    "use the parent session's login")
        raise RuntimeError(f"model call failed: {detail}{hint}")
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exited {proc.returncode}: "
                           f"{(proc.stderr or proc.stdout or '')[-400:]}")
    return payload.get("result") or ""


def parse_notes(raw: str) -> list[dict]:
    """Pull the JSON array out of the model's reply.

    Tolerates a fenced code block or a sentence before the array, because a
    strict parse that throws away a good answer over a stray backtick would make
    the whole pipeline look flaky.
    """
    text = raw.strip()
    if "```" in text:
        blocks = [b for b in text.split("```") if b.strip()]
        for b in blocks:
            b = b.removeprefix("json").strip()
            if b.startswith("["):
                text = b
                break
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("name")]


def distill_session(session: dict, model: str, dry_run: bool) -> list[Path]:
    text = prose_for(session)
    if not text:
        return []
    header = (f"SESSION {session['session_id']}\n"
              f"PROJECT {session.get('project')}\n"
              f"REPO    {session.get('cwd')} (branch {session.get('git_branch')})\n"
              f"TITLE   {session.get('title')}\n"
              f"DATE    {(session.get('started') or '')[:10]}\n")
    prompt = f"{INSTRUCTIONS}\n\n---\n{header}\n---\n\n{text}"
    if dry_run:
        print(f"  would send {len(prompt):,} chars for {session['session_id']}")
        return []

    raw = call_model(prompt, model=model)
    written: list[Path] = []
    for spec in parse_notes(raw):
        try:
            note = store.Note(
                name=str(spec["name"]),
                description=str(spec.get("description", ""))[:200],
                body=str(spec.get("body", "")).strip(),
                type=str(spec.get("type", "insight")),
                project=session.get("project"),
                source="claude-code",
                session=session["session_id"],
                occurred=(session.get("started") or "")[:10] or None,
                entities=[str(e) for e in (spec.get("entities") or [])][:8],
                confidence=str(spec.get("confidence", "stated")),
            )
        except (ValueError, KeyError) as e:
            print(f"  rejected a malformed note ({e})", file=sys.stderr)
            continue
        written.append(store.write_note(note, NOTES))
    return written


def mark_done(session_ids: list[str]) -> None:
    state = _load_json(DISTILLED, {"sessions": []})
    known = set(state.get("sessions", []))
    known.update(session_ids)
    DISTILLED.parent.mkdir(parents=True, exist_ok=True)
    DISTILLED.write_text(json.dumps(
        {"sessions": sorted(known),
         "last_run": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--session", help="distill one session id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-distill sessions already marked done")
    args = ap.parse_args()

    # Notes are scrubbed as they are written; give that scrub every known
    # secret value on disk too. The transcript sweep is build.py's job.
    known_secrets.install(include_transcripts=False)

    todo = pending(force=args.force)
    if args.session:
        todo = [t for t in todo if t["session_id"].startswith(args.session)] or [
            t for t in _load_json(MANIFEST, [])
            if t["session_id"].startswith(args.session)]
    todo = todo[:args.limit]

    if not todo:
        print("nothing pending")
        return
    print(f"{len(todo)} session(s) to distill (model={args.model})")

    done: list[str] = []
    total_notes = 0
    for s in todo:
        label = f"{s['session_id'][:8]} {(s.get('title') or '')[:44]}"
        try:
            paths = distill_session(s, args.model, args.dry_run)
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            # Per-session degradation, loudly: one bad session must not stop the
            # batch, and must not be silently marked done either.
            print(f"  FAILED {label}: {e}", file=sys.stderr)
            continue
        total_notes += len(paths)
        print(f"  {label} -> {len(paths)} note(s)")
        for p in paths:
            print(f"      {p.name}")
        done.append(s["session_id"])

    if done and not args.dry_run:
        mark_done(done)
        print(f"\n{total_notes} notes written from {len(done)} session(s). "
              f"Run `python build.py` (or the reindex_notes tool) to index them.")


if __name__ == "__main__":
    main()
