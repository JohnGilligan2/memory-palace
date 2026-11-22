"""Upload the finished index to the hosted palace, but only if it is provably
clean.

    python push_palace.py            # gate, snapshot, upload, confirm
    python push_palace.py --check    # gate only, no upload

The gate is the point. Before a byte leaves this laptop, every text row in the
index is checked against every secret value known on this machine (see
known_secrets.py) and against the full pattern scrub. One hit and nothing is
sent: a hosted copy reachable from any Claude chat must never hold a
credential, and "the scrub usually works" is not the bar.

Config: ~/.claude/credentials/memory-palace.env holds PALACE_INGEST_URL and
PALACE_INGEST_TOKEN. Nothing here prints either, or any row text.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import known_secrets
import scrub

HERE = Path(__file__).parent
DB = HERE / "index" / "palace.db"
CONFIG = Path.home() / ".claude" / "credentials" / "memory-palace.env"


def _config() -> dict[str, str]:
    out = {}
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def gate(db: Path) -> dict[str, int]:
    """Counts of rows that still hold a known value or would change under the
    full scrub. All zero or no upload."""
    stats = known_secrets.install()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = [r[0] or "" for r in conn.execute("SELECT text FROM turns")]
        for col in ("name", "description", "body"):
            rows += [r[0] or "" for r in conn.execute(f"SELECT {col} FROM notes")]
        rows += [r[0] or "" for r in conn.execute("SELECT title FROM sessions")]
    finally:
        conn.close()
    known = sum(1 for t in rows if scrub._KNOWN is not None and scrub._KNOWN.search(t))
    pattern = sum(1 for t in rows if scrub.scrub(t).text != t)
    return {"rows": len(rows), "known_value_rows": known, "pattern_rows": pattern,
            "known_values": stats["total"]}


def snapshot(db: Path) -> bytes:
    """A consistent single-file copy (no WAL), gzipped."""
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "palace.db"
        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        out = sqlite3.connect(dest)
        try:
            src.backup(out)
            out.execute("PRAGMA journal_mode=DELETE")
            out.commit()
            # A page-level copy carries free pages, which can still hold text
            # from earlier rebuilds that the row-level gate never sees. VACUUM
            # rewrites only live rows (and shrinks the upload: 110 MB -> ~75).
            out.execute("VACUUM")
        finally:
            out.close()
            src.close()
        raw = dest.read_bytes()
    return raw


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the gate only")
    args = ap.parse_args()

    g = gate(DB)
    print(f"gate: {g['rows']} rows checked against {g['known_values']} known values; "
          f"known-value rows {g['known_value_rows']}, pattern rows {g['pattern_rows']}")
    if g["known_value_rows"] or g["pattern_rows"]:
        print("REFUSING TO UPLOAD: the index is not clean. Rebuild, then retry.", file=sys.stderr)
        return 2
    if args.check:
        return 0

    if not CONFIG.exists():
        print(f"push not configured ({CONFIG} missing); nothing uploaded", file=sys.stderr)
        return 3
    cfg = _config()
    raw = snapshot(DB)
    payload = gzip.compress(raw, compresslevel=6)
    sha = hashlib.sha256(raw).hexdigest()
    req = urllib.request.Request(cfg["PALACE_INGEST_URL"], data=payload, method="POST",
                                 headers={"Authorization": f"Bearer {cfg['PALACE_INGEST_TOKEN']}",
                                          "Content-Type": "application/gzip"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"upload failed: HTTP {e.code} {e.read()[:200]!r}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"upload failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if result.get("sha256") != sha:
        print("upload failed: server checksum does not match", file=sys.stderr)
        return 1
    print(f"uploaded: {len(raw)/1e6:.1f} MB ({len(payload)/1e6:.1f} MB gzipped); "
          f"server holds {result.get('sessions')} sessions, {result.get('turns')} turns, "
          f"{result.get('notes')} notes; checksum verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
