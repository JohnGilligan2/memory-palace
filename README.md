# Memory Palace

Long-term memory over every Claude Code session on your machine, served back to Claude Code as
an MCP server. Ask "have we hit this 404 before?" and it finds the session where you did, with
the date and project.

## Why I built it

On 2026-08-02 I wrote a fix for one of my MCP servers and left it on a worktree branch that
never got merged. Sixteen days later I re-derived the same fix from scratch, while the deployed
connector returned 404 the whole time. The finding had been sitting in a transcript all along,
where one search would have surfaced it.

## Two layers

| Layer | Role |
|---|---|
| notes | Curated facts, one claim each, easy to correct. Searched first. |
| prose | Verbatim session turns, cited with session, project and date. The fallback and the audit trail. |

It isn't a vector database, on purpose. The questions I actually ask of my history are "what
did we say about X", "when did we last touch Y" and "what did I decide about Z". Those are
keyword-and-date questions, and BM25 with metadata filters answers them directly and can explain
why a hit matched. Embeddings answer "which paragraph is near this paragraph",
which is a different question. They're the planned addition for paraphrase misses, and
`schema.sql` already has the column.

It doesn't index raw transcripts either. On my machine's 623 MB of transcripts, human prose was
8.4 percent of the bytes and assistant prose 4.9 percent. The rest was tool results, tool calls
and structural noise that looks nearly identical from one session to the next. Extraction keeps
the prose only, and 623 MB became 14.9 MB.

## The scrub carries the security

Before building anything I measured the corpus: 128 of 458 transcripts contained something
matching a secret assignment. An index served by an MCP tool would hand those back to any chat
that asked.

So `scrub.py` runs on every string before it reaches disk or the index. It redacts named
assignments, bearer headers, provider-shaped tokens (`sk-ant-`, `ghp_`, JWTs, PEM blocks) and
bare high-entropy blobs, and keeps the sentence around them. "I set DB_PASSWORD in the stack env"
is worth remembering. The value isn't. Redactions are typed and visible (`[REDACTED:...]`), so a
reader can tell something was removed.

The rules were tuned against the real corpus. My first bare-blob rule fired 2,368 times over 121
transcripts and was wrong nearly every time, because `/` in its character class let one match run
across a whole URL or Windows path. After tightening it fired 145 times, and one of those was a
real DER-encoded private key.

Patterns weren't enough. A recall once returned a network-device password in clear,
because notes had never been scrubbed (only prose was) and the password was written in words
with no `=` for a rule to anchor on. Even after fixing both, about 15 real credentials survived
in shapes no regex can safely catch, like a `user`/`password` pair in backticks.

That's what `known_secrets.py` is for. At run time it harvests the exact secret values on your
own machine: everything in `~/.claude/credentials`, secret-named keys in `.env` and config JSON
under your code folder (`PALACE_CODE_ROOT`, default `~/ClaudeCode`), `~/.claude.json`, and every
value a pattern ever caught in a raw transcript. The scrub then redacts every copy of those
values, whatever surrounds them. The set lives in memory only. Nothing in this repo holds a
secret value, and the harvester only ever logs counts.

The raw transcripts under `~/.claude/projects` are left as they are. This never modifies them.

## Run it

```
pip install fastmcp==3.4.5
python build.py
claude mcp add memory-palace --scope user -- python /abs/path/to/mcp_server.py
```

`build.py` reads `~/.claude/projects` and `~/.claude/memory` for whoever runs it and builds a
SQLite FTS5 index of the prose in `index/`. It's incremental (`--full` re-extracts everything) and
needs no API key or network.

Then ask in plain language and the model picks the tool:

| Tool | For |
|---|---|
| `recall` | Almost everything. Notes first, then cited prose. |
| `timeline` | "When did we..." and how something evolved, in date order. |
| `decisions` | Settled positions and stated preferences only. |
| `remember` | Write a durable note. One claim per note. |
| `reindex_notes` | Make a just-written note searchable now. |
| `palace_stats` | Coverage. Check it before deciding the palace doesn't know something. |
| `rebuild` | Re-extract recent sessions and rebuild the index. |

`distill.py --limit 10` reads sessions that haven't been distilled and writes atomic notes with
`claude -p`. Run it from your own terminal: a `claude` nested inside a Claude Code session can't
reach the parent's login. A `SessionEnd` hook (`hooks/queue_session.py`) queues each session as
it ends with one file append and no model call. The distiller drains the queue, but treats any
indexed session that was never distilled as pending too, because a broken hook looks exactly
like a quiet week.

Tests: `python -m pytest -q`.

## Hosted mode (optional)

`PALACE_MODE=hosted` serves the same read tools over streamable HTTP behind Microsoft Entra, for
claude.ai and other Claude surfaces. The write tools aren't registered there. The server never
builds anything and never sees a transcript. The machine that owns the transcripts builds the
index, and `push_palace.py` checks every row against every known secret value plus the full
pattern scrub. It refuses to upload on a single hit. A clean index is gzipped and POSTed to
`/ingest`, which needs a bearer token and a private source address, and the server swaps it in
atomically.

Entra needs two app registrations: a resource app with the `access_as_user` scope and a separate
client app for Claude. With a single app there's no refresh token and the connector drops when
the access token expires. `docker-compose.yml` lists every variable; the tenant-specific
provisioning scripts I used aren't included.

`instrumentation.py` and `telemetry.py` are copies of
[mcp-telemetry-kit](https://github.com/johngilligan2/mcp-telemetry-kit). They do nothing unless
`OTEL_EXPORTER_OTLP_ENDPOINT` is set.

## Layout

```
scrub.py          credential scrub; everything passes through it
known_secrets.py  harvests exact secret values from your machine at run time (memory only)
extract.py        transcripts to prose turns plus session metadata
store.py          note schema (frontmatter matches schema.sql columns), read/write
index.py          SQLite FTS5 index, BM25 search
build.py          extract + index (incremental)
distill.py        prose to notes via `claude -p`
mcp_server.py     the MCP server: local stdio, or hosted HTTP
hosted.py         hosted mode: Entra auth, /healthz, /ingest validation
push_palace.py    clean gate, then upload of the index to the hosted copy
schema.sql        Postgres + pgvector layout for when keyword search stops being enough
hooks/            SessionEnd queue hook
index/, state/    derived and run state, created locally, never committed
notes/            your distilled notes, created locally, never committed
```

## License

MIT
