"""Harvest the exact secret VALUES that exist on this machine, so the scrub can
redact every copy of them whatever the surrounding text looks like.

Why this exists: pattern rules catch `DB_PASSWORD=...` and "password <value>",
but not a credential written as `netadmin`/`<pw>`, "Try user <pw>" or
"log in (`email` / `<pw>`)". Measured 2026-10-26 after the pattern rules were
tightened: about 15 real credentials still sat verbatim in the index, in exactly
those shapes. No regex catches them without also eating ordinary prose, but the
VALUES are known: they sit in the credential files, the .env files, and in
some other transcript where a pattern did catch them once. Know the value
once, redact it everywhere.

Sources, most to least certain:
  1. ~/.claude/credentials/*           every plausible value in every file
  2. .env / *.env / config JSON under ~/ClaudeCode   secret-named keys only
  3. ~/.claude.json                    secret-named keys and bearer headers
  4. raw transcripts                   every value a pattern rule matched

The harvested set lives in memory only. It is never printed, logged or written
anywhere, since it is by construction a list of every secret on the machine.
Only counts leave this module.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import scrub

HOME = Path.home()
CREDENTIALS = HOME / ".claude" / "credentials"
# Where your projects live; .env and config JSON under it are harvested. Override per machine.
CODE_ROOT = Path(os.environ.get("PALACE_CODE_ROOT") or (HOME / "ClaudeCode"))
CLAUDE_JSON = HOME / ".claude.json"
PROJECTS = HOME / ".claude" / "projects"

_PRUNE = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build",
          ".pytest_cache", "index", "site-packages"}
_MAX_FILE = 512 * 1024
# A value pattern-caught in more distinct transcripts than this is a recurring
# identifier (a status line, a temp filename), not a credential. Credential-file
# values are exempt: those are secrets by where they live.
MAX_TRANSCRIPTS = 30

_EMAIL = re.compile(r"\A[^@\s]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\Z")
_HOST = re.compile(r"\A[a-z0-9.-]+\.[a-z]{2,}\Z", re.I)
_IPV4 = re.compile(r"\A\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\Z")
_FILENAME = re.compile(r"\A[\w.-]+\.(?:txt|json|env|py|ps1|md|log|csv|pfx|pem|key|ya?ml|sh|js|ts|html)\Z", re.I)
_BASE64_DATA = re.compile(r'"data":\s*"[A-Za-z0-9+/=]{200,}"')
_PLACEHOLDER = re.compile(r"xxx|changeme|example|your[_-]|placeholder|redacted|<|\$\{|\.\.\.", re.I)


def plausible(v: str, trusted: bool = False) -> bool:
    """Could this be a secret, as opposed to an id, host, path or word?
    `trusted` loosens the shape test for values from the credentials folder."""
    v = v.strip().strip("'\"`")
    if len(v) < 6 or len(v) > 4096 or any(c.isspace() for c in v) or "\\" in v:
        return False
    if "://" in v or v.startswith("/") or v.lower() in ("true", "false", "null", "none"):
        return False
    if v.isdigit() or scrub._UUID.fullmatch(v) or _IPV4.match(v) or _EMAIL.match(v):
        return False
    if _HOST.match(v) or _FILENAME.match(v) or _PLACEHOLDER.search(v):
        return False
    if "A-Z" in v or "a-z" in v or "0-9" in v:     # a regex character class
        return False
    # Code, not a credential: `token = secrets.token_urlsafe(48)` was harvested
    # as a "value" from a coding session and would blank that line out of recall.
    if re.search(r"[A-Za-z_]\w*\(|\.\w+\[", v):
        return False
    has_alpha = any(c.isalpha() for c in v)
    has_digit = any(c.isdigit() for c in v)
    has_symbol = any(c in "!@#$%^&*+=~?" for c in v)
    mixed = any(c.isupper() for c in v) and any(c.islower() for c in v)
    # A letters-only value is usually a username, tenant or CamelCase
    # identifier ("Report_Tool"), and redacting it would blank that word
    # out of every sentence that names it. An all-letters password is the
    # accepted miss.
    if trusted:
        return has_alpha and (has_digit or has_symbol)
    return has_alpha and (has_digit or (has_symbol and mixed))


# Keys in a credentials file whose values are identifiers, not secrets.
_IDENTIFIER_KEY = re.compile(
    r"user(?!.*pass)|name|url|uri|host|server|_id\b|_id_|\bid\b|tenant|domain|from|_db\b|database|"
    r"\bto\b|email|mail|port|base|endpoint|region|org|account(?!.*key)", re.I)


def _walk_json(obj: Any, key: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_json(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v, key)
    elif isinstance(obj, str):
        yield key, obj


def _read(p: Path) -> str | None:
    try:
        if p.stat().st_size > _MAX_FILE:
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def from_credentials() -> set[str]:
    """Everything plausible in the credentials folder: KEY=VALUE values, bare
    single-token lines, and private-key body lines."""
    out: set[str] = set()
    if not CREDENTIALS.is_dir():
        return out
    for p in CREDENTIALS.rglob("*"):
        if not p.is_file() or p.suffix in (".pub", ".pfx", ".example") or p.name.endswith(".example"):
            continue
        text = _read(p)
        if text is None or "\x00" in text:
            continue
        # Filtered like everything else: unfiltered, this let 6-letter words
        # and hyphenated usernames through, and they were blanked out of 35-42
        # sessions each in the first measurement.
        out.update(v for v in scrub.secret_values(text) if plausible(v, trusted=True))
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-----"):
                continue
            m = re.match(r"\A(?:export\s+)?([A-Za-z_][A-Za-z0-9_.\-]*)\s*[:=]\s*(.+)\Z", line)
            if m:
                key, value = m.group(1), m.group(2).strip().strip("'\"`")
                if scrub._is_secret_key(key):
                    keep = plausible(value, trusted=True)
                elif _IDENTIFIER_KEY.search(key):
                    keep = False
                else:
                    keep = plausible(value)
            else:
                value, keep = line.strip("'\"`"), plausible(line, trusted=True)
            if keep:
                out.add(value)
        try:
            out.update(v for k, v in _walk_json(json.loads(text))
                       if (scrub._is_secret_key(k) and plausible(v, trusted=True))
                       or (not _IDENTIFIER_KEY.search(k) and plausible(v)))
        except ValueError:
            pass
    return out


def identifier_values() -> set[str]:
    """Values the credentials folder stores under identifier keys."""
    out: set[str] = set()
    if not CREDENTIALS.is_dir():
        return out
    for p in CREDENTIALS.rglob("*"):
        if not p.is_file() or p.suffix in (".pub", ".pfx"):
            continue
        text = _read(p)
        if text is None or "\x00" in text:
            continue
        for line in text.splitlines():
            m = re.match(r"\A\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.\-]*)\s*[:=]\s*(.+)\Z", line)
            if m and _IDENTIFIER_KEY.search(m.group(1)) and not scrub._is_secret_key(m.group(1)):
                out.add(m.group(2).strip().strip("'\"`"))
    return out


def _config_files() -> Iterator[Path]:
    if not CODE_ROOT.is_dir():
        return
    stack = [CODE_ROOT]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name not in _PRUNE:
                    stack.append(e)
                continue
            n = e.name.lower()
            if n.endswith(".example") or n.endswith(".sample"):
                continue
            if (n == ".env" or n.endswith(".env") or n.startswith(".env.")
                    or (n.endswith(".json") and any(w in n for w in ("config", "secret", "cred", "settings")))):
                yield e


def from_config_files() -> set[str]:
    """Secret-named values from .env and config JSON in the code tree."""
    out: set[str] = set()
    for p in _config_files():
        text = _read(p)
        if text is None:
            continue
        out.update(v for v in scrub.secret_values(text) if plausible(v))
        if p.suffix.lower() == ".json":
            try:
                out.update(v for k, v in _walk_json(json.loads(text))
                           if scrub._is_secret_key(k) and plausible(v))
            except ValueError:
                pass
    return out


def from_claude_json() -> set[str]:
    text = _read(CLAUDE_JSON) if CLAUDE_JSON.exists() else None
    if text is None:
        try:
            text = CLAUDE_JSON.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return set()
    out = {v for v in scrub.secret_values(text) if plausible(v)}
    try:
        out.update(v for k, v in _walk_json(json.loads(text))
                   if scrub._is_secret_key(k) and plausible(v))
    except ValueError:
        pass
    return out


# Every pattern rule needs one of these nearby. Running the rules only on
# windows around a trigger is what makes a full-corpus harvest affordable:
# measured on a 22.7 MB transcript, the assignment rule alone took 11.7 s
# because every JSON "key": "value" pair is a candidate match.
_TRIGGER = re.compile(
    r"passw|pwd|secret|token|api.?key|auth|bearer|basic |credential|private|pass\b|_pass|"
    r"sk-|gh[pousr]_|re_|xox|AIza|eyJ", re.I)
# Key names run at most ~40 chars before their trigger word and values at most
# a few hundred after it; a tight window keeps the work proportional to the
# number of trigger words, not the size of the transcript.
_BEFORE, _AFTER = 60, 300


# Transcript fields that are bulk machine data and never carry a credential:
# token accounting (whose "input_tokens" keys would trigger every window),
# thinking-block signatures and base64 screenshots.
_SKIP_KEYS = {"usage", "signature", "data", "uuid", "parentUuid", "requestId",
              "sessionId", "timestamp", "id", "tool_use_id", "message_id"}


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in _SKIP_KEYS:
                yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str) and len(obj) >= 6:
        yield obj


def _content(raw: str) -> str:
    """The human-meaningful strings of a JSONL transcript, unescaped."""
    out: list[str] = []
    for line in raw.splitlines():
        try:
            out.extend(_strings(json.loads(line)))
        except ValueError:
            continue
    return "\n".join(out)


def _windows(text: str) -> str:
    """The text around every trigger, merged, joined by blank lines."""
    spans: list[list[int]] = []
    for m in _TRIGGER.finditer(text):
        a, b = max(0, m.start() - _BEFORE), m.end() + _AFTER
        if spans and a <= spans[-1][1]:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    return "\n\n".join(text[a:b] for a, b in spans)


def from_transcripts(projects_dir: Path = PROJECTS) -> dict[str, int]:
    """Every value a pattern rule matched in a raw transcript, with the number
    of transcripts it appeared in (for the recurring-identifier guard)."""
    files = list(projects_dir.rglob("*.jsonl"))
    seen: dict[str, int] = {}
    # One process per core: the pattern rules are pure-Python regex work, about
    # a second per trigger-dense megabyte, over a corpus past 1.4 GB.
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        for found in pool.map(_harvest_file, files, chunksize=4):
            for v in found:
                seen[v] = seen.get(v, 0) + 1
    return seen


def _harvest_file(f: Path) -> set[str]:
    try:
        raw = f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    text = _windows(_content(raw))
    found = {v for v in scrub.secret_values(text, blobs=False) if _plausible_from_transcript(v)}
    # "password <value>" is the strongest signal a transcript gives, strong
    # enough for 6-7 char values: a user's 6-char password survived the
    # 8-char floor in the first run.
    for m in scrub._INLINE_CREDENTIAL.finditer(text):
        v = m.group("value").strip(scrub._EDGE_PUNCT)
        if plausible(v) and any(c.isdigit() for c in v) and not v.startswith(("$", "{", "<", "%", "[", "http")):
            found.add(v)
    return found


def _plausible_from_transcript(v: str) -> bool:
    """Stricter than plausible(): a pattern hit in a transcript is weaker
    evidence than a value sitting in a credential file. Short or digit-free
    values ("False", a policy name, a 6-char db name) were blanked out of 20-37
    sessions each in the first harvest; real credentials almost always carry a
    digit."""
    return plausible(v) and len(v) >= 8 and any(c.isdigit() for c in v)


def load(include_transcripts: bool = True) -> tuple[set[str], dict[str, Any]]:
    """The full known-values set plus counts-only stats for logging."""
    t0 = time.time()
    creds = from_credentials()
    configs = from_config_files()
    cj = from_claude_json()
    values = creds | configs | cj
    stats: dict[str, Any] = {"credentials": len(creds), "config_files": len(configs),
                             "claude_json": len(cj)}
    if include_transcripts:
        tx = from_transcripts()
        kept = {v for v, n in tx.items() if n <= MAX_TRANSCRIPTS or v in values}
        stats["transcripts"] = len(kept)
        stats["transcripts_dropped_recurring"] = len(tx) - len(kept)
        values |= kept
    # A value a credentials file itself labels as an identifier (APP_DB,
    # *_USERNAME, *_TENANT) is not a secret, whatever a transcript pattern
    # thought: one app's database name was blanked out of 58 places that way.
    ids = identifier_values()
    stats["dropped_identifiers"] = len(values & ids)
    values -= ids
    stats["total"] = len(values)
    stats["seconds"] = round(time.time() - t0, 1)
    return values, stats


def install(include_transcripts: bool = True) -> dict[str, Any]:
    """Harvest and install into scrub. Returns counts only."""
    values, stats = load(include_transcripts)
    scrub.set_known_secrets(values)
    return stats
