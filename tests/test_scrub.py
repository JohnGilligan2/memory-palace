"""The scrub is the security control the whole store rests on, so its two
failure modes get pinned separately.

A MISS puts a live credential in an index that an MCP tool serves to any chat
window. A FALSE POSITIVE mangles the prose we are keeping — and the corpus is
full of things that look like secrets and are not: git SHAs, session uuids,
Entra client ids, timestamps, `pwd`.

The negative cases below have the shapes of real strings from the author's transcripts,
with identifiers swapped for placeholders.
Three of them were live bugs the first time this ran: JSON-style
`"client_secret": "..."` slipped through because the key's closing quote was not
in the pattern, and `ENTRA_CLIENT_ID=<uuid>` plus
`registration_expires_time=2026-08-19` were both mangled because `=` sat in the
bare-blob character class and swallowed the key along with the value.
"""
import scrub

MUST_REDACT = [
    ("env assignment", "DB_PASSWORD=hunter2SwordFish99"),
    ("resend key", "RESEND_API_KEY=re_ab12CD34ef56GH78ij90KL"),
    ("json client secret", '"client_secret": "Xk8Q~7yTbN2mPqR4vL9wZ1aS3dF6gH0j"'),
    ("bearer header", 'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.'
                      'eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk'),
    ("sip password", "sip_password: 8Kd93mZqLp01"),
    ("github pat", "the token is ghp_AbCdEfGhIjKlMnOpQrStUvWxYz012345"),
    ("openai key", "sk-proj-Ab12Cd34Ef56Gh78Ij90Kl12Mn34Op56"),
    ("private key block", "-----BEGIN RSA PRIVATE KEY-----"),
    ("device auth key", 'authentication_key="Zq83Kd0lPw92Ms"'),
    # The 2026-10-10 miss: credentials dictated in words or pasted as device
    # config, with a space where the assignment rule wants `=` or `:`.
    # Values here are synthetic; the shapes are the real ones.
    ("dictated device login", "username is netadmin password qz7wk2mx then run enable"),
    ("dictated enable", "you run enable and enter password r8tq4vzn"),
    ("ssh in prose", "ssh 192.0.2.12 user user password Tr0ub4dor&3x"),
    ("router sip auth", "sip-authentication password k3jd8wq0zm2xp"),
    ("router register", 'register 3105550100 auth-name "3105550100" password k3jd8wq0zm2xp'),
    ("cisco type 7", "username admin password 7 0822455D0A16"),
    ("cisco enable secret", "enable secret 5 $1$mERr$hx5rVt7rPNoS4wqbXKX7m0"),
    ("password is", "the portal password is Wint3r2026!"),
]

MUST_KEEP = [
    ("entra client id", "ENTRA_CLIENT_ID=00000000-0000-0000-0000-000000000001"),
    ("git shas", "commit ce44dd6 and b3f2102a9c8d4e5f6071829304a5b6c7d8e9f0a1"),
    ("env var reference", "DB_PASSWORD=${DB_PASSWORD}"),
    ("prose about a secret", "portal password is set in the stack env"),
    ("session uuid", "session 00000000-0000-0000-0000-000000000002 wrote the note"),
    ("placeholder", "pwd=<redacted>"),
    ("the pwd command", "I ran pwd and got /c/Users/user"),
    ("switch timestamp", "registration_expires_time=2026-08-19 05:14:44"),
    ("webhook id", "GitHub webhook 123456789 -> stack redeploy, 204"),
    ("long identifier", "mcp__00000000-0000-0000-0000-000000000003__list_devices"),
    # Prose ABOUT passwords, which the inline rule must not eat.
    ("password field", "the password field is blank on the device row"),
    ("password required", "sudo: a password is required"),
    ("password reset", "send the password reset link to the user"),
    ("password protected", "the PDF is password-protected"),
    ("password env ref", "password ${DB_PASSWORD} comes from the stack env"),
    ("cmdlet", "Add-MgApplicationPassword -ApplicationId abc"),
    ("plural", "rotate the passwords for all 12 devices"),
    # Real shapes from the first measurement run, all false positives then.
    ("markdown heading", "- **Password** reset goes through the portal"),
    ("sentence end", "Don't paste the secret. Ask for the vault link."),
    ("client id and secret", "the client ID and secret rotate together"),
    ("no password", "the app refuses to start with no password-file set"),
]


def test_pass_key_segment():
    """APP_CERT_PASS slipped through: the key list had `passw`, not `pass`."""
    assert scrub.scrub("APP_CERT_PASS=Qk83mZ0pLw92Ms").total == 1
    assert scrub.scrub("SMTP_PASS: Qk83mZ0pLw92Ms").total == 1
    for harmless in ("bypass=Enabled2026x", "compass_heading=north123x", "passthrough=on123456",
                     "pass: Pass-Fail-Check", "TokenPolicyName=MCP-AccessToken-24h",
                     "APP_CERT_PASSWORD_ENV=APP_CERT_PASS",
                     "HV_POWER_TOKEN_ID=svc@realm!example01"):
        assert scrub.scrub(harmless).total == 0, harmless


def test_known_values_are_redacted_in_any_shape():
    """The shapes no pattern can catch: a pair, a dictated retry, a login
    hint. Once the value is known, every copy goes."""
    try:
        scrub.set_known_secrets(["Zx9pairPw1"])
        for text in ("creds are `netadmin`/`Zx9pairPw1` for devices",
                     "Try user Zx9pairPw1",
                     "log in (`john@example.org` / `Zx9pairPw1`)"):
            r = scrub.scrub(text)
            assert "Zx9pairPw1" not in r.text, text
            assert r.redactions.get("known-secret") == 1
        # Whole-token only: a longer word that merely contains it is kept.
        assert scrub.scrub("aZx9pairPw1b").total == 0
    finally:
        scrub.set_known_secrets([])


def test_harvest_plausibility():
    """What the harvester may treat as a secret value. Ids, hosts and paths
    would otherwise be blanked out of every sentence that names them."""
    import known_secrets as ks
    for keep in ("Zx9pairPw1", "k3jd8wq0zm2xp", "AbCd!fGhIjKl"):
        assert ks.plausible(keep), keep
    for drop in ("00000000-0000-0000-0000-000000000001", "john@example.org",
                 "core1.pbx.example.com", "192.0.2.10", "notes.json",
                 "netadmin", "Report_Tool", "3105550100", "${DB_PASSWORD}",
                 "https://x.example/a1", "changeme123"):
        assert not ks.plausible(drop), drop
    assert not ks.plausible("CallHistoryTool", trusted=True)
    assert ks.plausible("Wint3r2026", trusted=True)


def test_scrub_is_idempotent():
    """Notes and prose are now scrubbed again at index time, so a second pass
    over already-scrubbed text must change nothing and count nothing."""
    once = scrub.scrub("DB_PASSWORD=hunter2SwordFish99 and user x password qz7wk2mx").text
    again = scrub.scrub(once)
    assert again.text == once
    assert again.total == 0


def test_notes_are_scrubbed_at_index_time():
    """The 2026-10-10 leak came through a NOTE, not prose: notes from the
    runtime memory folders were indexed exactly as written. The index must
    scrub every note it serves, whoever wrote it."""
    import sqlite3
    import index
    import store
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(index.SCHEMA)
    note = store.Note(name="device-login", description="router login",
                      body="username is netadmin password qz7wk2mx", type="reference")
    index.rebuild_notes(conn, [note])
    body = conn.execute("SELECT body FROM notes").fetchone()["body"]
    assert "qz7wk2mx" not in body
    assert "[REDACTED:" in body


def test_credentials_are_redacted():
    for label, text in MUST_REDACT:
        assert scrub.scrub(text).total > 0, f"MISSED a credential in: {label}"


def test_lookalikes_are_left_alone():
    for label, text in MUST_KEEP:
        r = scrub.scrub(text)
        assert r.total == 0, f"false positive on {label}: {r.text}"


def test_the_sentence_survives_the_redaction():
    """Keeping "I set DB_PASSWORD in the stack env" is the point; only the value
    goes. A scrub that dropped the line would cost us the recall."""
    r = scrub.scrub("I set DB_PASSWORD=hunter2SwordFish99 in the stack env today")
    assert "in the stack env today" in r.text
    assert "DB_PASSWORD" in r.text
    assert "hunter2SwordFish99" not in r.text


def test_redaction_is_typed_and_visible():
    """A reader must be able to tell something was removed, rather than trusting
    a doctored sentence."""
    r = scrub.scrub("key: ghp_AbCdEfGhIjKlMnOpQrStUvWxYz012345")
    assert "[REDACTED:" in r.text


def test_counts_are_reported_but_values_never_are():
    r = scrub.scrub("DB_PASSWORD=Sup3rS3cretValue99x SMTP_PWD=An0therS3cretValue88")
    assert r.total == 2
    assert all(isinstance(v, int) for v in r.redactions.values())
    assert "Sup3rS3cretValue99x" not in repr(r.redactions)


def test_base64_encoded_prose_is_not_mistaken_for_a_key():
    """A pasted base64 diff or doc decodes to readable ASCII; key material does
    not. Without this, every encoded attachment gets redacted."""
    import base64
    encoded = base64.b64encode(
        b"the quick brown fox jumps over the lazy dog repeatedly and at length"
    ).decode()
    assert scrub.scrub(encoded).total == 0
