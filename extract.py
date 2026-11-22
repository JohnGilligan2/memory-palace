"""Turn Claude Code session transcripts into the 13% of them that is prose.

Measured on this machine before writing this: 623 MB across 458 sessions and 60
projects, of which

    tool results          25.4%
    structural noise      46.0%
    assistant tool calls  15.3%
    human prose            8.4%   <- wanted
    assistant prose        4.9%   <- wanted

Embedding or indexing the raw corpus means 87% of the index is machine chatter
that looks near-identical from one session to the next, and it buries the part
worth recalling. So extraction is aggressive: tool calls, tool results,
thinking blocks and harness-injected context are dropped, not summarised.

What survives is a turn stream per session with enough metadata to answer
"when", "where" and "which project" without a model: session id, project, cwd,
git branch, title, timestamps, and whether the turn came from a subagent.

Everything goes through scrub.scrub() on the way out. Nothing reaches the index
unscrubbed — see scrub.py for why that is not optional.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import scrub

# Harness-injected wrappers that appear INSIDE a user turn. They are context the
# runtime added, not something a person typed, and they repeat across thousands
# of turns — the single biggest source of false recall if left in.
_INJECTED = re.compile(
    r"<(system-reminder|local-command-stdout|local-command-stderr|command-name|"
    r"command-message|command-args|task-notification|budget:[a-z-]+)>.*?</\1>"
    r"|<(system-reminder|task-notification)>.*?\Z",
    re.DOTALL | re.IGNORECASE,
)

# A turn that is nothing but one of these is an artifact, not content.
_ARTIFACT_ONLY = re.compile(
    r"\A\s*(\[Request interrupted[^\]]*\]|<[a-z-]+>\s*</[a-z-]+>|"
    r"(no content|\(no content\)|caveat:.*))\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

MIN_TURN_CHARS = 12


@dataclass
class Turn:
    index: int
    role: str                 # "human" | "assistant"
    text: str
    ts: str | None
    sidechain: bool = False

    def as_row(self) -> dict[str, Any]:
        return {"i": self.index, "role": self.role, "ts": self.ts,
                "sidechain": self.sidechain, "text": self.text}


@dataclass
class Session:
    session_id: str
    project: str                        # the projects/ dir slug
    path: Path
    cwd: str | None = None
    git_branch: str | None = None
    title: str | None = None
    started: str | None = None
    ended: str | None = None
    turns: list[Turn] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)
    dropped_blocks: int = 0

    @property
    def human_turns(self) -> int:
        return sum(1 for t in self.turns if t.role == "human")

    @property
    def chars(self) -> int:
        return sum(len(t.text) for t in self.turns)

    def manifest_row(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "project": self.project,
            "cwd": self.cwd, "git_branch": self.git_branch, "title": self.title,
            "started": self.started, "ended": self.ended,
            "turns": len(self.turns), "human_turns": self.human_turns,
            "chars": self.chars, "redactions": self.redactions,
            "dropped_blocks": self.dropped_blocks,
            "source_path": str(self.path),
        }


def _clean(text: str) -> str:
    """Strip harness-injected context and normalise whitespace."""
    text = _INJECTED.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _text_blocks(message: Any) -> Iterator[str]:
    """Yield only human/assistant PROSE from a message payload.

    Explicitly skips tool_use, tool_result, thinking and image blocks. A string
    content field (the common shape for a typed prompt) is prose by definition.
    """
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if isinstance(content, str):
        yield content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, str):
            yield block
        elif isinstance(block, dict) and block.get("type") == "text":
            yield block.get("text") or ""


def parse_session(path: Path, project: str) -> Session | None:
    """Read one .jsonl transcript into a Session, or None if it has no prose."""
    s = Session(session_id=path.stem, project=project, path=path)
    idx = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line happens when a session is still live.
                continue
            kind = d.get("type")

            if kind in ("ai-title", "custom-title"):
                s.title = d.get("customTitle") or d.get("aiTitle") or s.title
                continue
            if kind not in ("user", "assistant"):
                continue

            s.cwd = s.cwd or d.get("cwd")
            s.git_branch = s.git_branch or d.get("gitBranch")
            ts = d.get("timestamp")
            if ts:
                s.started = s.started or ts
                s.ended = ts

            # isMeta marks runtime-injected user turns (reminders, hook output).
            if kind == "user" and d.get("isMeta"):
                s.dropped_blocks += 1
                continue

            role = "human" if kind == "user" else "assistant"
            for raw in _text_blocks(d.get("message")):
                cleaned = _clean(raw)
                if len(cleaned) < MIN_TURN_CHARS or _ARTIFACT_ONLY.match(cleaned):
                    s.dropped_blocks += 1
                    continue
                res = scrub.scrub(cleaned)
                for k, v in res.redactions.items():
                    s.redactions[k] = s.redactions.get(k, 0) + v
                s.turns.append(Turn(index=idx, role=role, text=res.text, ts=ts,
                                    sidechain=bool(d.get("isSidechain"))))
                idx += 1
    return s if s.turns else None


def parent_session_of(path: Path) -> str | None:
    """The session that spawned a subagent transcript, if this is one.

    Subagent runs are NOT stored beside the main sessions — they live at
    `<project>/<session-uuid>/subagents/agent-<id>.jsonl`, three quarters of
    every transcript on this machine (347 of 458). A top-level glob misses all
    of them, which would have silently dropped every chief-of-staff brief, researcher
    report and deck-team draft from the corpus.
    """
    parts = path.parts
    if "subagents" in parts:
        i = parts.index("subagents")
        if i >= 1:
            return parts[i - 1]
    return None


def iter_transcripts(projects_dir: Path) -> Iterator[tuple[Path, str]]:
    """Every transcript under ~/.claude/projects at any depth, newest first.

    Newest-first matters for incremental runs: the watermark walk can stop as
    soon as it reaches a file it has already seen.
    """
    files: list[tuple[float, Path, str]] = []
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for f in project_dir.rglob("*.jsonl"):
            try:
                files.append((f.stat().st_mtime, f, project_dir.name))
            except OSError:
                continue
    for _, f, project in sorted(files, key=lambda t: -t[0]):
        yield f, project


def write_prose(session: Session, out_dir: Path) -> Path:
    """Persist a session's prose as one JSONL per session (turn per line).

    Kept as flat files rather than only in SQLite so the corpus stays greppable,
    diffable and portable — the index is a derived artifact that can always be
    thrown away and rebuilt.
    """
    dest = out_dir / session.project / f"{session.session_id}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": session.manifest_row()}, ensure_ascii=False) + "\n")
        for t in session.turns:
            fh.write(json.dumps(t.as_row(), ensure_ascii=False) + "\n")
    return dest
