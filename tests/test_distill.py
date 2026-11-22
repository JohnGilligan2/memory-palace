"""The distiller's parsing and note-writing, tested without a model call.

The one link that cannot be exercised from inside a Claude Code sandbox is the
`claude -p` subprocess itself (a nested claude gets 401 — the parent's OAuth is
not reachable). Everything on either side of that call is tested here, so the
untested surface is exactly one subprocess invocation and nothing else.
"""
import json

import distill
import store

# What the model actually returns, in the four shapes seen in practice.
CLEAN = '[{"name":"a-fact","description":"d","type":"insight","body":"b",' \
        '"entities":["pbx"],"confidence":"verified"}]'
FENCED = f"```json\n{CLEAN}\n```"
CHATTY = f"Here are the notes I extracted:\n\n{CLEAN}\n\nLet me know if you want more."
EMPTY = "[]"


def test_clean_json_parses():
    notes = distill.parse_notes(CLEAN)
    assert [n["name"] for n in notes] == ["a-fact"]


def test_fenced_and_chatty_replies_still_parse():
    """A stray code fence or a preamble must not throw away a good answer —
    that would make the whole pipeline look flaky for a cosmetic reason."""
    assert len(distill.parse_notes(FENCED)) == 1
    assert len(distill.parse_notes(CHATTY)) == 1


def test_empty_array_is_a_valid_answer_not_a_failure():
    """Most sessions yield nothing worth keeping. That has to be expressible."""
    assert distill.parse_notes(EMPTY) == []


def test_unparseable_reply_yields_nothing_rather_than_raising():
    assert distill.parse_notes("I could not find anything useful.") == []
    assert distill.parse_notes("") == []


def test_notes_without_a_name_are_dropped():
    assert distill.parse_notes('[{"description":"no name"},{"name":"kept"}]') == \
        [{"name": "kept"}]


def test_a_distilled_note_round_trips_through_the_store(tmp_path):
    """Write it, read it back, and confirm the metadata that Option B's schema
    depends on survives the trip."""
    spec = json.loads(CLEAN)[0]
    note = store.Note(
        name=spec["name"], description=spec["description"], body=spec["body"],
        type=spec["type"], project="proj-x", source="claude-code",
        session="sess-1", occurred="2026-08-19",
        entities=spec["entities"], confidence=spec["confidence"])
    path = store.write_note(note, tmp_path)
    back = store.load_note(path)
    assert back is not None
    assert (back.name, back.type, back.session) == ("a-fact", "insight", "sess-1")
    assert back.occurred == "2026-08-19"
    assert back.entities == ["pbx"]
    assert back.confidence == "verified"


def test_an_invalid_type_is_rejected_at_construction():
    """The vocabulary is a contract with the index and with Postgres later. A
    typo must fail here, not become a silent third category."""
    try:
        store.Note(name="n", description="d", body="b", type="thought")
    except ValueError:
        return
    raise AssertionError("an unknown note type was accepted")


def test_a_session_too_small_to_be_worth_a_model_call_is_skipped(monkeypatch, tmp_path):
    """A one-question lookup should not cost a model call."""
    manifest = [
        {"session_id": "tiny", "project": "p", "chars": 100, "human_turns": 1},
        {"session_id": "real", "project": "p", "chars": 50_000, "human_turns": 9},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(distill, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(distill, "DISTILLED", tmp_path / "distilled.json")
    monkeypatch.setattr(distill, "QUEUE", tmp_path / "queue.jsonl")
    assert [s["session_id"] for s in distill.pending()] == ["real"]


def test_pending_does_not_rely_on_the_queue_alone(monkeypatch, tmp_path):
    """A silently broken SessionEnd hook looks exactly like a quiet week. Pending
    work must therefore include indexed sessions that were never queued."""
    manifest = [{"session_id": "never-queued", "project": "p",
                 "chars": 9000, "human_turns": 4}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "queue.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(distill, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(distill, "DISTILLED", tmp_path / "distilled.json")
    monkeypatch.setattr(distill, "QUEUE", tmp_path / "queue.jsonl")
    assert [s["session_id"] for s in distill.pending()] == ["never-queued"]


def test_already_distilled_sessions_are_not_redone(monkeypatch, tmp_path):
    manifest = [{"session_id": "done-already", "project": "p",
                 "chars": 9000, "human_turns": 4}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "distilled.json").write_text(
        json.dumps({"sessions": ["done-already"]}), encoding="utf-8")
    monkeypatch.setattr(distill, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(distill, "DISTILLED", tmp_path / "distilled.json")
    monkeypatch.setattr(distill, "QUEUE", tmp_path / "queue.jsonl")
    assert distill.pending() == []
    assert [s["session_id"] for s in distill.pending(force=True)] == ["done-already"]
