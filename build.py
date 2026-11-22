"""Build the palace: extract prose, then index prose + notes.

    python build.py            # incremental extract, full reindex
    python build.py --full     # re-extract every transcript from scratch

Incremental is the default because extraction is the slow half (32s over 447
transcripts) while indexing is seconds. A transcript is re-extracted when its
mtime is newer than the last run's watermark; the index is always rebuilt whole,
so it can never silently drift from the files it describes.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import extract
import index
import known_secrets
import store

HERE = Path(__file__).parent
CLAUDE_HOME = Path.home() / ".claude"
PROJECTS = CLAUDE_HOME / "projects"
PROSE = HERE / "index" / "prose"
MANIFEST = HERE / "index" / "manifest.json"
DB = HERE / "index" / "palace.db"
NOTES = HERE / "notes"
STATE = HERE / "state" / "watermark.json"


def _load_watermark() -> float:
    if STATE.exists():
        try:
            return float(json.loads(STATE.read_text(encoding="utf-8")).get("mtime", 0))
        except (ValueError, OSError):
            return 0.0
    return 0.0


def _save_watermark(mtime: float, counts: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"mtime": mtime, "ran": datetime.now(timezone.utc).isoformat(),
         **counts}, indent=1), encoding="utf-8")


def extract_all(full: bool = False) -> dict:
    """Extract prose from transcripts newer than the watermark."""
    watermark = 0.0 if full else _load_watermark()
    newest = watermark
    manifest: list[dict] = []
    if MANIFEST.exists() and not full:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {m["session_id"]: m for m in manifest}

    done = skipped = 0
    redactions: dict[str, int] = {}
    for path, project in extract.iter_transcripts(PROJECTS):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        newest = max(newest, mtime)
        if mtime <= watermark and path.stem in known:
            skipped += 1
            continue
        session = extract.parse_session(path, project)
        if session is None:
            continue
        row = session.manifest_row()
        row["parent_session"] = extract.parent_session_of(path)
        extract.write_prose(session, PROSE)
        known[session.session_id] = row
        for k, v in session.redactions.items():
            redactions[k] = redactions.get(k, 0) + v
        done += 1

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(list(known.values()), indent=1), encoding="utf-8")
    return {"extracted": done, "unchanged": skipped, "newest_mtime": newest,
            "redactions": redactions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="re-extract every transcript, ignoring the watermark")
    args = ap.parse_args()

    # Exact secret values first, so extraction AND the index both redact every
    # copy of a credential, whatever shape it was written in. Counts only.
    ks = known_secrets.install()
    print(f"secrets: {ks['total']} known values (credentials {ks['credentials']}, "
          f"config {ks['config_files']}, claude.json {ks['claude_json']}, "
          f"transcripts {ks.get('transcripts', 0)}) in {ks['seconds']}s")

    t0 = time.time()
    ex = extract_all(full=args.full)
    print(f"extract: {ex['extracted']} transcripts ({ex['unchanged']} unchanged) "
          f"in {time.time()-t0:.0f}s")
    if ex["redactions"]:
        print(f"         redactions this run: {ex['redactions']}")

    t1 = time.time()
    conn = index.connect(DB)
    sessions, turns = index.rebuild_prose(conn, PROSE, MANIFEST)
    dirs = store.note_dirs(CLAUDE_HOME, NOTES)
    notes = index.rebuild_notes(conn, store.iter_notes(dirs))
    index.set_meta(conn, note_dirs=[str(d) for d in dirs],
                   built=datetime.now(timezone.utc).isoformat())
    print(f"index:   {sessions} sessions, {turns:,} turns, {notes} notes "
          f"from {len(dirs)} note dirs in {time.time()-t1:.0f}s")
    print(f"db:      {DB} ({DB.stat().st_size/1024/1024:.1f} MB)")
    _save_watermark(ex["newest_mtime"],
                    {"sessions": sessions, "turns": turns, "notes": notes})


if __name__ == "__main__":
    main()
