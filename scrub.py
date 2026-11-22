"""Credential scrub for anything entering the palace.

This is the load-bearing security control of the whole store. Measured before
building it: of 458 Claude Code session transcripts on this machine, 128 contain
something matching a secret assignment and 1 contains a cleartext SIP
credential field. Those transcripts are the ingest corpus. An MCP tool over an
unscrubbed index would hand any chat window a semantic search over every secret
that has ever crossed the terminal — it would undo, from the other end, the
transport-layer scrub the source servers already do.

So: nothing is written to the index or a note until it has been through here.
The scrub is a DENY list plus patterns, not an allow list, because we cannot
enumerate what a future tool will print. It fails toward redaction: an
ambiguous match is redacted, not kept.

What it deliberately does NOT do is drop the surrounding text. The sentence
"I set DB_PASSWORD in the stack env" is worth keeping; the value is not. Every
redaction leaves a typed marker so a later reader can see that something was
removed rather than silently reading a doctored sentence.
"""
from __future__ import annotations

import base64
import math
import re
from typing import NamedTuple

# Field/variable names whose VALUE is always a credential. Substring match on a
# lowercased key, so DB_PASSWORD, db_old_password and "smtp_pwd" all land.
_SECRET_KEY_PARTS = (
    "passw", "pwd", "secret", "token", "api_key", "apikey", "api-key",
    "auth_key", "authentication_key", "access_key", "private_key",
    "subscriber_pin", "client_secret", "bearer", "credential",
)

# Names that look secret but are routinely harmless identifiers. Without this,
# every mention of a client_id or a public key fingerprint gets mangled.
_SECRET_KEY_EXCEPTIONS = ("client_id", "public_key", "key_name", "keyboard",
                          "token_count", "tokens", "key_vault_name",
                          # Measured 2026-10-26 when these values were harvested
                          # as "secrets" and blanked out of up to 25 sessions:
                          # TokenPolicyName holds a policy name, PasswordNotRequired
                          # a boolean, *_PASSWORD_ENV the NAME of an env var, and
                          # *_TOKEN_ID an id (the secret is its own key).
                          "name", "policy", "required", "_env", "_file", "_path",
                          "token_id", "_type", "expir", "lifetime")

# KEY=value / KEY: value / "key": "value" — the three shapes secrets arrive in.
_ASSIGNMENT = re.compile(
    r"""(?P<key>[A-Za-z_][A-Za-z0-9_.\-]{2,40})
        (?P<keyq>["']?)                      # JSON writes "key": "value"
        (?P<sep>\s*(?:=>|[:=])\s*)
        (?P<quote>["']?)
        (?P<value>[^\s"',;}\)]{6,200})
        (?P=quote)""",
    re.VERBOSE,
)

_BEARER = re.compile(r"\b(?:Bearer|Basic)\s+([A-Za-z0-9._~+/=-]{16,})", re.IGNORECASE)

# A credential stated in words or in device config, with no `=`/`:` for the
# assignment rule to anchor on. Found 2026-10-10 when a recall returned the
# network-device enable password in clear: it had been dictated as
# "username is netadmin password <value>", and router config arrives as
# `sip-authentication password <value>`. Both slipped through because the key
# and value are separated by a space, and an 8-char password is far below the
# bare-blob floor.
#
# What separates "password Xk3m9pQz" from "password field", "password reset"
# and "**Password** is required" is _looks_like_inline_value() below: a letter
# AND a digit, or mixed case AND a real symbol, judged after markdown and
# sentence punctuation are stripped. The first cut accepted any symbol and
# measured 128 fires on the real corpus, most of them `**Password**` headings
# and "secret." at the end of a sentence. Prose about passwords is everywhere
# here and is worth keeping. An all-letters password stated in prose still gets
# through; that is the accepted cost of not mangling every sentence about one.
_INLINE_CREDENTIAL = re.compile(
    r"""(?P<lead>\b(?:password|passwd|passphrase|pwd|secret)
           (?:\s+(?:is|was|of|to|set\s+to))?\s*[:=]?\s+)
        (?P<quote>["'`]?)
        (?P<value>[^\s"'`,;)]{4,64})
        (?P=quote)""",
    re.VERBOSE | re.IGNORECASE,
)
_EDGE_PUNCT = "*_.:?!()[]"


def _looks_like_inline_value(value: str) -> bool:
    core = value.strip(_EDGE_PUNCT)
    if len(core) < 4 or not any(c.isalpha() for c in core):
        return False
    if any(c.isdigit() for c in core):
        return True
    mixed = any(c.isupper() for c in core) and any(c.islower() for c in core)
    return mixed and any(c in "!@#$%^&~+=" for c in core)

# Cisco-style encrypted-type syntax: `password 7 0822455D0A16`,
# `enable secret 5 $1$abcd$...`, `username x privilege 15 secret 9 $9$...`.
# The type digit defeats the rule above, and type-7 is trivially reversible.
_TYPED_DEVICE_SECRET = re.compile(
    r"\b(?P<lead>(?:password|secret)\s+[0-9]{1,2}\s+)(?P<value>\S{4,})", re.IGNORECASE)

# Provider-shaped tokens worth catching by their own prefix, since they can
# appear with no key name at all (pasted bare into a prompt).
_KNOWN_TOKEN_SHAPES = (
    ("resend", re.compile(r"\bre_[A-Za-z0-9_]{16,}\b")),
    ("openai", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("google", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b")),
    ("jwt", re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)

# A bare high-entropy blob. Deliberately conservative (>=32 chars, mixed
# classes, high Shannon entropy) because git SHAs, uuids and file hashes are
# everywhere in this corpus and are NOT secrets.
#
# `=` is NOT in the body class, only as trailing base64 padding. With `=` in the
# body, `ENTRA_CLIENT_ID=<uuid>` matched as ONE token, which defeated the uuid
# and hex exemptions below and redacted a public identifier (caught in test).
#
# `/` is not in the body class either, and the floor is 40 chars rather than 32.
# Measured on 121 real transcripts: with `/` included and a 32-char floor this
# rule fired 2,368 times and was wrong essentially every time — it was matching
# URLs, Windows paths, project slugs and `Get-`-prefixed cmdlets, because a path
# separator let one match run across a whole URL. The named-assignment and
# provider-shape rules caught 133 items over the same corpus and were precise.
# A bare secret with no key name AND no provider prefix is rare; corrupting
# thousands of prose spans to maybe catch one is a bad trade.
_BLOB = re.compile(r"\b[A-Za-z0-9+_-]{40,}={0,2}(?![A-Za-z0-9+_-])")

# A 4-digit year anywhere means a timestamp or a dated slug, not key material.
_YEARISH = re.compile(r"(19|20)\d{2}")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_HEXISH = re.compile(r"\A[0-9a-f]+\Z", re.I)


# Exact secret VALUES known to exist on this machine (credential files, .env
# files, and every value the patterns below have ever caught in a raw
# transcript). Patterns cannot catch a password written as `netadmin`/`<pw>` or
# "Try user <pw>"; measured 2026-10-26, about 15 real credentials survived the
# pattern rules in exactly those shapes. Once a value is known, every copy of
# it goes, whatever surrounds it. Set by known_secrets.load(); held in memory
# only, never written anywhere.
_KNOWN: re.Pattern[str] | None = None
_KNOWN_COUNT = 0


def set_known_secrets(values) -> int:
    """Install exact values to redact everywhere. Longest first so a value that
    contains another is replaced whole. Returns how many were installed."""
    global _KNOWN, _KNOWN_COUNT
    vals = sorted({v for v in values if v and len(v) >= 6}, key=len, reverse=True)
    _KNOWN_COUNT = len(vals)
    _KNOWN = (re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(map(re.escape, vals))
                         + r")(?![A-Za-z0-9])") if vals else None)
    return _KNOWN_COUNT


def redact_known(text: str) -> tuple[str, int]:
    """Only the known-values pass. Prose was fully scrubbed at extraction, but
    the known set grows as new secrets appear, so the index re-applies this
    cheap pass to every turn on every rebuild."""
    if _KNOWN is None or not text:
        return text, 0
    out, n = _KNOWN.subn("[REDACTED:known-secret]", text)
    return out, n


class Result(NamedTuple):
    text: str
    redactions: dict[str, int]      # kind -> count. Never the values themselves.

    @property
    def total(self) -> int:
        return sum(self.redactions.values())


def _shannon(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_secret_key(key: str) -> bool:
    k = key.lower()
    if any(x in k for x in _SECRET_KEY_EXCEPTIONS):
        return False
    # `_PASS` as a whole segment (APP_CERT_PASS, SMTP_PASS) slipped through
    # `passw`; a bare substring would hit bypass, compass and passthrough.
    # A key that is just `pass` is Python's statement or a test verdict far
    # more often than a password (37 false sessions in the first harvest).
    if k != "pass" and re.search(r"(?:^|[_.\-])pass(?:$|[_.\-])", k):
        return True
    return any(part in k for part in _SECRET_KEY_PARTS)


def _looks_like_secret_blob(s: str) -> bool:
    """A bare token with no key to identify it.

    Excluded: uuids (session ids are everywhere in this corpus), pure hex (git
    SHAs, md5/sha digests), and anything whose entropy is low enough to be
    prose, a path, or base64 of prose. Requires >= 3.6 bits/char AND at least
    one digit AND mixed case or symbol — a real key, not a long word.
    """
    if _UUID.match(s) or _HEXISH.match(s):
        return False
    # Identifier-shaped, not key-shaped. An EMBEDDED uuid means a generated name
    # (MCP tool names are `mcp__<server-uuid>__<tool>`, worktree branches are
    # `claude/<slug>-<hash>`), and `__` is a naming convention no key uses.
    if _UUID.search(s) or "__" in s:
        return False
    # Hyphen-rich strings are slugs, branch names and dotted identifiers, not
    # keys. Provider tokens carry at most one separator and are caught by shape
    # above anyway. This plus the year check is what killed the false-positive
    # flood from project slugs like `C--Users-user-ClaudeCode-...`.
    if s.count("-") > 2 or _YEARISH.search(s):
        return False
    if len(s) < 40 or _shannon(s) < 3.6:
        return False
    if not any(c.isdigit() for c in s):
        return False
    has_upper, has_lower = any(c.isupper() for c in s), any(c.islower() for c in s)
    if not (has_upper and has_lower) and not any(c in "+/=_-" for c in s):
        return False
    # base64 that decodes to readable ASCII prose is almost always encoded text
    # (a diff, a doc) rather than key material.
    try:
        raw = base64.b64decode(s + "=" * (-len(s) % 4), validate=False)
        printable = sum(32 <= b < 127 for b in raw)
        if raw and printable / len(raw) > 0.9:
            return False
    except Exception:  # noqa: BLE001 — not base64 at all; that is fine, keep testing
        pass
    return True


def scrub(text: str) -> Result:
    """Redact credential material, keeping the sentence around it.

    Order matters: exact known values first (no pattern can beat certainty),
    then named assignments (the most reliable pattern signal), then
    provider-shaped tokens, then bare blobs, so a value that an earlier rule
    already identified is not double-counted as an anonymous blob.
    """
    if not text:
        return Result(text or "", {})
    counts: dict[str, int] = {}

    def bump(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    out = text
    if _KNOWN is not None:
        def repl_known_value(_m: re.Match[str]) -> str:
            bump("known-secret")
            return "[REDACTED:known-secret]"
        out = _KNOWN.sub(repl_known_value, out)

    def repl_assignment(m: re.Match[str]) -> str:
        if not _is_secret_key(m.group("key")):
            return m.group(0)
        value = m.group("value")
        # Placeholders and env-var references are not secrets and are useful to
        # keep readable: DB_PASSWORD=${DB_PASSWORD}, pwd=<redacted>, key=xxx.
        # `[` covers an earlier [REDACTED:...] marker, so re-scrubbing already
        # scrubbed text (notes, re-indexed prose) changes and counts nothing.
        if value.startswith(("$", "${", "<", "%", "[")) or set(value) <= set("x*.•"):
            return m.group(0)
        bump("named-assignment")
        return (f'{m.group("key")}{m.group("keyq")}{m.group("sep")}'
                f'{m.group("quote")}[REDACTED:credential]{m.group("quote")}')

    out = _ASSIGNMENT.sub(repl_assignment, out)

    def repl_bearer(m: re.Match[str]) -> str:
        bump("bearer")
        return m.group(0).split()[0] + " [REDACTED:token]"

    out = _BEARER.sub(repl_bearer, out)

    def repl_typed(m: re.Match[str]) -> str:
        if m.group("value").startswith("[REDACTED"):
            return m.group(0)
        bump("device-secret")
        return f'{m.group("lead")}[REDACTED:credential]'

    out = _TYPED_DEVICE_SECRET.sub(repl_typed, out)

    def repl_inline(m: re.Match[str]) -> str:
        value = m.group("value")
        # Same placeholder carve-out as assignments, plus URLs and an earlier
        # redaction marker, which this rule would otherwise count twice.
        if (value.startswith(("$", "{", "<", "%", "[", "http")) or "REDACTED" in value
                or not _looks_like_inline_value(value)):
            return m.group(0)
        bump("inline-credential")
        return f'{m.group("lead")}{m.group("quote")}[REDACTED:credential]{m.group("quote")}'

    out = _INLINE_CREDENTIAL.sub(repl_inline, out)

    for kind, pattern in _KNOWN_TOKEN_SHAPES:
        def repl_known(_m: re.Match[str], _kind: str = kind) -> str:
            bump(f"token:{_kind}")
            return f"[REDACTED:{_kind}]"
        out = pattern.sub(repl_known, out)

    def repl_blob(m: re.Match[str]) -> str:
        s = m.group(0)
        if not _looks_like_secret_blob(s):
            return s
        bump("high-entropy-blob")
        return "[REDACTED:blob]"

    out = _BLOB.sub(repl_blob, out)
    return Result(out, counts)


def secret_values(text: str, blobs: bool = True) -> set[str]:
    """The VALUES the pattern rules would redact in `text`, for harvesting into
    the known-values set. Callers must never print or persist what this returns.

    `blobs=False` skips the bare-blob rule, which dominates the cost on raw
    transcripts (megabytes of base64 screenshots) and adds little: a blob's
    other copies are blobs too, and the pattern catches those anyway."""
    found: set[str] = set()
    for m in _ASSIGNMENT.finditer(text):
        v = m.group("value")
        if _is_secret_key(m.group("key")) and not v.startswith(("$", "<", "%", "[")):
            found.add(v)
    for m in _INLINE_CREDENTIAL.finditer(text):
        v = m.group("value").strip(_EDGE_PUNCT)
        if _looks_like_inline_value(v) and not v.startswith(("$", "{", "<", "%", "[", "http")):
            found.add(v)
    for m in _TYPED_DEVICE_SECRET.finditer(text):
        found.add(m.group("value"))
    for m in _BEARER.finditer(text):
        found.add(m.group(1))
    for _kind, pattern in _KNOWN_TOKEN_SHAPES[:-1]:      # not the PEM header line
        found.update(m.group(0) for m in pattern.finditer(text))
    if blobs:
        found.update(m.group(0) for m in _BLOB.finditer(text) if _looks_like_secret_blob(m.group(0)))
    return {v for v in found if "REDACTED" not in v}
