"""SessionEnd hook: record that a session ended, and do nothing else.

This runs on every session exit, so it is deliberately O(1) and dependency-free:
read the hook JSON from stdin, append one line to a queue file, exit. No model
call, no extraction, no index write. The expensive work (distillation) is a
batch job that drains this queue on a schedule — a hook that spawned a model
call on every session end would tax every single exit to serve a nightly need.

Failure is silent BY DESIGN here, and only here: a memory-capture side channel
must never be the reason a session cannot close. The cost of that choice is that
a broken hook looks like an idle one, so distill.py cross-checks the queue
against the transcript directory and reports sessions it never saw queued.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

QUEUE = Path(__file__).resolve().parent.parent / "state" / "queue.jsonl"


def main() -> int:
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    entry = {
        "session_id": payload.get("session_id") or payload.get("sessionId"),
        "cwd": payload.get("cwd") or os.getcwd(),
        "reason": payload.get("reason") or payload.get("hook_event_name"),
        "queued": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not entry["session_id"]:
        return 0        # nothing identifiable to queue; not an error

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 — see the module docstring: never block an exit
        sys.exit(0)
