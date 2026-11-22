"""Notes: the curated layer, and the thing recall reads first.

Two layers exist in this palace and the distinction is the whole design:

  prose  — 19,927 turns extracted from 447 transcripts. Verbatim, greppable,
           and mostly NOT what you want back. It is the fallback and the
           citation trail.
  notes  — atomic, distilled, human-correctable facts. One claim per note.
           This is what a question should hit first.

The reason notes beat raw retrieval is supersession. A vector store has no
opinion about which of five contradictory statements is current; a note gets
EDITED when it turns out wrong, and carries `supersedes` when it replaces an
earlier one. That is why this machine's existing 113 hand-maintained notes are
more useful than 623 MB of transcript, and scaling that is the point.

The frontmatter here is deliberately Option B's Postgres schema written as
YAML — every field maps to a column in schema.sql. Notes are the seed rows for
the database, so the migration is a loader, not a redesign.

Frontmatter is parsed with a deliberately small hand-rolled reader (stdlib
only, no PyYAML) supporting exactly what the schema uses: scalars, one level of
nesting, and inline `[a, b]` lists. Anything fancier is a schema change, and
should be a considered one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any

import scrub

# The controlled vocabularies. Kept small on purpose: a type system nobody can
# remember gets ignored, and then the filters are worthless.
TYPES = ("decision", "insight", "project", "feedback", "user", "reference")
SOURCES = ("claude-code", "chat", "email", "helpdesk", "manual")
SENSITIVITY = ("internal", "customer", "secret-adjacent")

_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_LIST = re.compile(r"\A\[(.*)\]\Z")
_SLUG_BAD = re.compile(r"[^a-z0-9]+")


@dataclass
class Note:
    """One durable claim.

    `occurred` vs `recorded` is not pedantry: "we decided X" belongs on the day
    of the decision, not the day a distiller noticed it, or every timeline query
    collapses onto the night the backfill ran.
    """
    name: str
    description: str
    body: str
    type: str = "insight"
    project: str | None = None
    source: str = "claude-code"
    session: str | None = None
    occurred: str | None = None          # ISO date of the underlying event
    recorded: str | None = None          # ISO date this note was written
    people: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    supersedes: str | None = None
    sensitivity: str = "internal"
    confidence: str = "stated"           # stated | inferred | verified
    path: Path | None = None

    def __post_init__(self) -> None:
        self.recorded = self.recorded or date.today().isoformat()
        self.occurred = self.occurred or self.recorded
        if self.type not in TYPES:
            raise ValueError(f"type must be one of {TYPES}, got {self.type!r}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.sensitivity not in SENSITIVITY:
            raise ValueError(f"sensitivity must be one of {SENSITIVITY}")

    @property
    def links(self) -> list[str]:
        """[[wiki-links]] to other notes. A link to a note that does not exist
        yet is legitimate — it marks something worth writing."""
        return re.findall(r"\[\[([^\]]+)\]\]", self.body)

    def to_markdown(self) -> str:
        def fmt(v: Any) -> str:
            if isinstance(v, list):
                return "[" + ", ".join(str(x) for x in v) + "]"
            return "" if v is None else str(v)

        keys = ("name", "description", "type", "project", "source", "session",
                "occurred", "recorded", "people", "entities", "supersedes",
                "sensitivity", "confidence")
        lines = ["---"]
        for k in keys:
            v = getattr(self, k)
            if v in (None, "", []):
                continue
            lines.append(f"{k}: {fmt(v)}")
        lines += ["---", "", self.body.strip(), ""]
        return "\n".join(lines)


def slugify(text: str, max_len: int = 60) -> str:
    s = _SLUG_BAD.sub("-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "note"


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if m := _LIST.match(raw):
        inner = m.group(1).strip()
        return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()] if inner else []
    if raw.startswith(('"', "'")) and raw.endswith(('"', "'")) and len(raw) > 1:
        return raw[1:-1]
    return raw


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (fields, body). Handles one level of nesting by flattening it.

    Flattening matters for reading the notes this machine already has: those
    were written with `metadata:\\n  type: project`, and recall should treat
    that `type` the same as a top-level one rather than ignoring it.
    """
    m = _FM.match(text)
    if not m:
        return {}, text.strip()
    fields: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if not value.strip():          # a nesting header; children flatten in
            continue
        fields.setdefault(key, _parse_scalar(value))
    return fields, m.group(2).strip()


def load_note(path: Path) -> Note | None:
    """Read a note from disk, tolerating the older hand-written shape.

    The 113 pre-existing notes on this machine predate this schema: they carry
    name/description plus a nested metadata.type and no dates. They are still
    the most valuable content in the palace, so this coerces rather than
    rejects, and falls back to file mtime for `occurred`.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fields, body = parse_frontmatter(raw)
    if not body and not fields:
        return None
    mtime = date.fromtimestamp(path.stat().st_mtime).isoformat()
    ntype = str(fields.get("type") or "insight").strip().lower()
    src = str(fields.get("source") or "claude-code").strip().lower()
    sens = str(fields.get("sensitivity") or "internal").strip().lower()
    try:
        return Note(
            name=str(fields.get("name") or path.stem),
            description=str(fields.get("description") or "").strip('"'),
            body=body,
            type=ntype if ntype in TYPES else "insight",
            project=fields.get("project") or _project_of(path),
            source=src if src in SOURCES else "claude-code",
            session=fields.get("session"),
            occurred=str(fields.get("occurred") or mtime)[:10],
            recorded=str(fields.get("recorded") or mtime)[:10],
            people=fields.get("people") or [],
            entities=fields.get("entities") or [],
            supersedes=fields.get("supersedes"),
            sensitivity=sens if sens in SENSITIVITY else "internal",
            confidence=str(fields.get("confidence") or "stated"),
            path=path,
        )
    except ValueError:
        return None


def _project_of(path: Path) -> str | None:
    """Recover the project from a harness memory path.

    Pre-existing notes live at ~/.claude/projects/<slug>/memory/<name>.md, so
    the slug two levels up is the project they belong to.
    """
    parts = path.parts
    if "memory" in parts:
        i = parts.index("memory")
        if i >= 1 and parts[i - 1] != ".claude":
            return parts[i - 1]
    return None


def note_dirs(claude_home: Path, palace_notes: Path) -> list[Path]:
    """Every directory holding notes worth searching.

    Deliberately includes the harness's own per-project memory dirs and the
    global one. Those are managed by the runtime and must NOT be moved into the
    palace — the runtime would rewrite them. Read them where they live.
    """
    dirs = [palace_notes, claude_home / "memory"]
    projects = claude_home / "projects"
    if projects.is_dir():
        dirs += [p / "memory" for p in projects.iterdir() if (p / "memory").is_dir()]
    return [d for d in dirs if d.is_dir()]


def iter_notes(dirs: list[Path]):
    """Every note in every note directory, MEMORY.md indexes excluded.

    The indexes are pointers to notes, not notes; indexing them would return
    a table of contents as a search hit for every query it lists.
    """
    for d in dirs:
        for f in sorted(d.rglob("*.md")):
            if f.name.upper() in ("MEMORY.MD", "README.MD"):
                continue
            n = load_note(f)
            if n is not None:
                yield n


def write_note(note: Note, notes_dir: Path) -> Path:
    """Write a note, namespaced by project so two projects can hold the same
    slug without collision."""
    # `remember` writes whatever a chat window hands it, so the file on disk is
    # scrubbed too, not just the index row.
    note.description = scrub.scrub(note.description).text
    note.body = scrub.scrub(note.body).text
    sub = notes_dir / (note.project or "cross-cutting")
    sub.mkdir(parents=True, exist_ok=True)
    dest = sub / f"{slugify(note.name)}.md"
    dest.write_text(note.to_markdown(), encoding="utf-8")
    note.path = dest
    return dest


def as_row(note: Note) -> dict[str, Any]:
    """Flat dict for the index and, later, a Postgres INSERT."""
    d = asdict(note)
    d["path"] = str(note.path) if note.path else None
    d["people"] = ",".join(note.people)
    d["entities"] = ",".join(note.entities)
    d["links"] = ",".join(note.links)
    return d
