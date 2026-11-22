-- Option B: the palace on Postgres, reachable from claude.ai as well as the CLI.
--
-- This is deliberately NOT a redesign. Every column here already exists as a
-- field in a note's YAML frontmatter or a row in the SQLite index, so migrating
-- is a loader over notes/ and index/prose/, not a rewrite. Run Option A until
-- one of these is actually true, and not before:
--
--   * you want recall from claude.ai chats, not just Claude Code on this laptop
--   * a teammate needs read access to a shared slice
--   * keyword search starts missing things you know are in there (the
--     paraphrase problem — "phones dropping after a firmware push" when nobody
--     wrote "firmware"), which is when embeddings start earning their cost
--
-- Target: Postgres 16 + pgvector on a VM, or a container beside the existing
-- MCP servers. Embeddings local (bge-m3 / nomic-embed via Ollama) — sending
-- customer email and chat content to a hosted embedding API is a data-egress
-- decision that is harder to defend to a customer than it is to avoid.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- --------------------------------------------------------------------------
-- Vocabularies. Small on purpose: a taxonomy nobody can remember gets ignored,
-- and then the filters are worthless.
-- --------------------------------------------------------------------------
CREATE TYPE note_type   AS ENUM ('decision','insight','project','feedback','user','reference');
CREATE TYPE mem_source  AS ENUM ('claude-code','chat','email','helpdesk','manual');
CREATE TYPE sensitivity AS ENUM ('internal','customer','secret-adjacent');
CREATE TYPE confidence  AS ENUM ('stated','inferred','verified');

-- --------------------------------------------------------------------------
-- People and things, so "what did Sam say about X" is a join and not a
-- string search. This is the part SQLite FTS cannot do well and the reason
-- Option C (a temporal graph) is a natural next step from here rather than a
-- separate project.
-- --------------------------------------------------------------------------
CREATE TABLE person (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT UNIQUE,
    display     TEXT NOT NULL,
    org         TEXT
);

CREATE TABLE entity (
    id      BIGSERIAL PRIMARY KEY,
    slug    TEXT UNIQUE NOT NULL,      -- 'pbx', 'billing', 'examplesite'
    kind    TEXT,                      -- system | customer | project | vendor | tool
    display TEXT
);

-- --------------------------------------------------------------------------
-- Sources: one row per session / thread / message the corpus was built from.
-- Mirrors the `sessions` table in index.py.
-- --------------------------------------------------------------------------
CREATE TABLE source_doc (
    id             BIGSERIAL PRIMARY KEY,
    source         mem_source NOT NULL,
    external_id    TEXT NOT NULL,          -- session id, chat message id, mail message id
    project        TEXT,
    title          TEXT,
    cwd            TEXT,
    git_branch     TEXT,
    parent_id      BIGINT REFERENCES source_doc(id) ON DELETE SET NULL,  -- subagent runs
    started        TIMESTAMPTZ,
    ended          TIMESTAMPTZ,
    turns          INT,
    human_turns    INT,
    ingested       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, external_id)
);
CREATE INDEX source_doc_project_idx ON source_doc (project);
CREATE INDEX source_doc_started_idx ON source_doc (started DESC);

-- --------------------------------------------------------------------------
-- Prose: the verbatim layer. Scrubbed at ingest — see scrub.py. NOTHING lands
-- here that has not been through it; the measured baseline was 128 of 458
-- transcripts carrying credential material.
-- --------------------------------------------------------------------------
CREATE TABLE turn (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       BIGINT NOT NULL REFERENCES source_doc(id) ON DELETE CASCADE,
    idx          INT NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('human','assistant')),
    said_at      TIMESTAMPTZ,
    sidechain    BOOLEAN NOT NULL DEFAULT false,
    body         TEXT NOT NULL,
    redactions   JSONB NOT NULL DEFAULT '{}'::jsonb,   -- kind -> count, never values
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,
    embedding    vector(1024),                          -- NULL until embedded
    UNIQUE (doc_id, idx)
);
CREATE INDEX turn_tsv_idx  ON turn USING GIN (tsv);
CREATE INDEX turn_time_idx ON turn (said_at DESC);
-- Add only when the corpus is big enough to need it; at 20k turns a seq scan
-- over vectors is already fast and an ANN index just adds recall loss:
-- CREATE INDEX turn_vec_idx ON turn USING hnsw (embedding vector_cosine_ops);

-- --------------------------------------------------------------------------
-- Notes: the curated layer, and what recall reads first. One claim per row.
-- Column-for-column the note frontmatter in store.py.
-- --------------------------------------------------------------------------
CREATE TABLE note (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL,
    body         TEXT NOT NULL,
    type         note_type   NOT NULL DEFAULT 'insight',
    source       mem_source  NOT NULL DEFAULT 'claude-code',
    sensitivity  sensitivity NOT NULL DEFAULT 'internal',
    confidence   confidence  NOT NULL DEFAULT 'stated',
    project      TEXT,
    doc_id       BIGINT REFERENCES source_doc(id) ON DELETE SET NULL,
    -- occurred vs recorded is not pedantry: "we decided X" belongs on the day of
    -- the decision, or every timeline collapses onto the night the backfill ran.
    occurred     DATE NOT NULL,
    recorded     DATE NOT NULL DEFAULT CURRENT_DATE,
    -- Supersession is the thing a plain vector store cannot express, and the
    -- reason 114 curated notes beat 623 MB of transcript. A superseded note is
    -- kept, not deleted: how a position CHANGED is often the answer.
    supersedes   BIGINT REFERENCES note(id) ON DELETE SET NULL,
    superseded_by BIGINT REFERENCES note(id) ON DELETE SET NULL,
    file_path    TEXT,                    -- the markdown file this came from
    tsv          tsvector GENERATED ALWAYS AS (
                     setweight(to_tsvector('english', coalesce(name,'')), 'A') ||
                     setweight(to_tsvector('english', coalesce(description,'')), 'A') ||
                     setweight(to_tsvector('english', coalesce(body,'')), 'B')) STORED,
    embedding    vector(1024),
    UNIQUE (project, name)
);
CREATE INDEX note_tsv_idx      ON note USING GIN (tsv);
CREATE INDEX note_occurred_idx ON note (occurred DESC);
CREATE INDEX note_live_idx     ON note (type) WHERE superseded_by IS NULL;

CREATE TABLE note_entity (
    note_id   BIGINT NOT NULL REFERENCES note(id) ON DELETE CASCADE,
    entity_id BIGINT NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    PRIMARY KEY (note_id, entity_id)
);

CREATE TABLE note_person (
    note_id   BIGINT NOT NULL REFERENCES note(id) ON DELETE CASCADE,
    person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    role      TEXT,                      -- said_it | subject | owner
    PRIMARY KEY (note_id, person_id, role)
);

-- [[wiki-links]] between notes. A link to a note that does not exist yet is
-- legitimate: it marks something worth writing.
CREATE TABLE note_link (
    from_id   BIGINT NOT NULL REFERENCES note(id) ON DELETE CASCADE,
    to_name   TEXT NOT NULL,
    to_id     BIGINT REFERENCES note(id) ON DELETE SET NULL,
    PRIMARY KEY (from_id, to_name)
);

-- --------------------------------------------------------------------------
-- Ingest bookkeeping, so a broken adapter is visible rather than quiet.
-- --------------------------------------------------------------------------
CREATE TABLE ingest_run (
    id          BIGSERIAL PRIMARY KEY,
    source      mem_source NOT NULL,
    started     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished    TIMESTAMPTZ,
    docs_seen   INT DEFAULT 0,
    docs_new    INT DEFAULT 0,
    turns_new   INT DEFAULT 0,
    notes_new   INT DEFAULT 0,
    redactions  JSONB DEFAULT '{}'::jsonb,
    errors      JSONB DEFAULT '[]'::jsonb,
    watermark   TEXT
);

-- --------------------------------------------------------------------------
-- Hybrid retrieval: BM25-ish full text UNION vector similarity, notes weighted
-- above prose. Keyword-first is the design — vectors are the paraphrase
-- fallback, not the primary index.
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION recall(
    q            TEXT,
    q_embedding  vector(1024) DEFAULT NULL,
    p_project    TEXT DEFAULT NULL,
    p_since      DATE DEFAULT NULL,
    p_limit      INT  DEFAULT 10
) RETURNS TABLE (
    layer TEXT, ref BIGINT, occurred DATE, headline TEXT, body TEXT, score REAL
) LANGUAGE sql STABLE AS $$
    WITH kw_notes AS (
        SELECT 'note'::TEXT, n.id, n.occurred, n.description, n.body,
               ts_rank(n.tsv, websearch_to_tsquery('english', q)) * 2.0 AS score
        FROM note n
        WHERE n.tsv @@ websearch_to_tsquery('english', q)
          AND n.superseded_by IS NULL
          AND (p_project IS NULL OR n.project = p_project)
          AND (p_since   IS NULL OR n.occurred >= p_since)
    ),
    kw_turns AS (
        SELECT 'prose'::TEXT, t.id, t.said_at::DATE, d.title, t.body,
               ts_rank(t.tsv, websearch_to_tsquery('english', q)) AS score
        FROM turn t JOIN source_doc d ON d.id = t.doc_id
        WHERE t.tsv @@ websearch_to_tsquery('english', q)
          AND (p_project IS NULL OR d.project = p_project)
          AND (p_since   IS NULL OR t.said_at >= p_since)
    ),
    vec_notes AS (
        SELECT 'note~'::TEXT, n.id, n.occurred, n.description, n.body,
               (1 - (n.embedding <=> q_embedding)) * 1.5 AS score
        FROM note n
        WHERE q_embedding IS NOT NULL AND n.embedding IS NOT NULL
          AND n.superseded_by IS NULL
        ORDER BY n.embedding <=> q_embedding
        LIMIT p_limit
    )
    SELECT * FROM (
        SELECT * FROM kw_notes
        UNION ALL SELECT * FROM kw_turns
        UNION ALL SELECT * FROM vec_notes
    ) hits
    ORDER BY score DESC
    LIMIT p_limit;
$$;
