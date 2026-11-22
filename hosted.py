"""The hosted half of the palace: Entra auth, health, and the upload endpoint.

Local mode (the default) is a stdio server reading this laptop's index. Hosted
mode (`PALACE_MODE=hosted`) is the same tools over streamable HTTP behind
Entra, for claude.ai and every other Claude surface.

The hosted copy never builds anything. The laptop builds and scrubs the index,
proves it clean, and uploads the finished palace.db to POST /ingest; the server
validates it and swaps it in atomically. Raw transcripts never leave the
laptop, so the only thing ever on this server is already-scrubbed text.

Read-only by design: `remember`, `reindex_notes` and `rebuild` are not
registered here, because notes live on the laptop and a write on the server
would be overwritten by the next upload.
"""
from __future__ import annotations

import gzip
import hashlib
import hmac
import ipaddress
import logging
import os
import sqlite3
import tempfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger("memory_palace.hosted")

# An index older than this means the nightly upload has stopped. Reported as
# stale rather than healthy: fail-unknown, never fail-clean.
STALE_HOURS = 36
# Uncompressed ceiling for an upload (~137 MB on 2026-11-05). Also applied to
# the compressed body as it streams in, so an oversized POST stops early.
MAX_DB_BYTES = 1024 * 1024 * 1024
# Streaming unit for the body and the decompression: memory per upload is a
# few of these, independent of the size of the index.
CHUNK = 1024 * 1024
_SQLITE_MAGIC = b"SQLite format 3\x00"
_REQUIRED_TABLES = {"notes", "turns", "sessions", "meta"}


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    path: str = "/mcp"
    db_path: Path = Path("/data/palace.db")

    auth_enabled: bool = True
    tenant_id: str = ""
    client_id: str = ""            # the RESOURCE app (this server's API)
    identifier_uri: str = ""
    base_url: str = ""
    scope_name: str = "access_as_user"
    advertise_offline_access: bool = True

    ingest_token: str = ""
    ingest_allow: list[ipaddress._BaseNetwork] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Config":
        allow = _env("PALACE_INGEST_ALLOW_CIDRS", "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.0/8")
        return cls(
            host=_env("MCP_HOST", "0.0.0.0"),
            port=int(_env("MCP_PORT", "8000")),
            path=_env("MCP_PATH", "/mcp"),
            db_path=Path(_env("PALACE_DB", "/data/palace.db")),
            auth_enabled=_env("MCP_AUTH_ENABLED", "true").lower() != "false",
            tenant_id=_env("ENTRA_TENANT_ID"),
            client_id=_env("ENTRA_CLIENT_ID"),
            identifier_uri=_env("ENTRA_IDENTIFIER_URI"),
            base_url=_env("MCP_BASE_URL"),
            scope_name=_env("ENTRA_SCOPE_NAME", "access_as_user"),
            advertise_offline_access=_env("ENTRA_ADVERTISE_OFFLINE_ACCESS", "true").lower() != "false",
            ingest_token=_env("PALACE_INGEST_TOKEN"),
            ingest_allow=[ipaddress.ip_network(c.strip()) for c in allow.split(",") if c.strip()],
        )

    def validate(self) -> None:
        """Refuse to boot rather than serve this corpus open or half-configured.
        It is a year of an engineer's working memory, including customer
        incident detail; an auth typo must stop the deploy, not degrade it."""
        missing = []
        if self.auth_enabled:
            missing += [n for n, v in (("ENTRA_TENANT_ID", self.tenant_id),
                                       ("ENTRA_CLIENT_ID", self.client_id),
                                       ("ENTRA_IDENTIFIER_URI", self.identifier_uri),
                                       ("MCP_BASE_URL", self.base_url)) if not v]
        if len(self.ingest_token) < 32:
            missing.append("PALACE_INGEST_TOKEN (>= 32 chars)")
        if missing:
            raise SystemExit("memory-palace hosted: refusing to start, missing " + ", ".join(missing))
        if not self.auth_enabled:
            logger.warning("MCP_AUTH_ENABLED=false: UNAUTHENTICATED. Local testing only.")


def build_auth(cfg: Config):
    """Entra resource-server mode: validate audience-bound v2 JWTs, never issue
    them. Claude signs in through the separate CLIENT app registration; this
    server only knows the RESOURCE app (house rule: two apps, never one)."""
    if not cfg.auth_enabled:
        return None
    from fastmcp.server.auth import RemoteAuthProvider
    from fastmcp.server.auth.providers.azure import AzureJWTVerifier
    from pydantic import AnyHttpUrl

    verifier = AzureJWTVerifier(
        tenant_id=cfg.tenant_id, client_id=cfg.client_id,
        identifier_uri=cfg.identifier_uri, required_scopes=[cfg.scope_name])
    scopes = [f"{cfg.identifier_uri}/{cfg.scope_name}"]
    if cfg.advertise_offline_access:
        # Without it Claude never gets a refresh token and the connector dies
        # at access-token expiry.
        scopes.append("offline_access")
    return RemoteAuthProvider(
        token_verifier=verifier,
        authorization_servers=[AnyHttpUrl(f"https://login.microsoftonline.com/{cfg.tenant_id}/v2.0")],
        base_url=cfg.base_url,
        scopes_supported=scopes,
    )


def index_health(db_path: Path) -> dict[str, Any]:
    """What /healthz and palace_stats report about the served index."""
    if not db_path.exists():
        return {"index": "missing", "detail": "no upload received yet"}
    out: dict[str, Any] = {"index": "present", "bytes": db_path.stat().st_size}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key='built'").fetchone()
        finally:
            conn.close()
        built = (row[0] or "").strip('"') if row else ""
        out["built"] = built or None
        if built:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(built)).total_seconds() / 3600
            out["age_hours"] = round(age, 1)
            out["index"] = "stale" if age > STALE_HOURS else "fresh"
        else:
            out["index"] = "unknown-age"
    except (sqlite3.Error, ValueError) as e:   # report, never crash the probe
        out["index"] = "unreadable"
        out["detail"] = type(e).__name__
    return out


_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")


def via_proxy(headers) -> bool:
    """True when the request came through the public reverse proxy. The proxy
    connects from a private address, so the source-CIDR check alone passed every
    public request (measured 2026-10-27: an unauthenticated POST /ingest through
    the public hostname reached the token check). Uploads come straight from the
    laptop on the LAN and never carry these headers."""
    return any(h in headers for h in _PROXY_HEADERS)


def client_allowed(cfg: Config, host: str | None) -> bool:
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return any(ip in net for net in cfg.ingest_allow)


def token_ok(cfg: Config, header: str | None) -> bool:
    if not header or not header.startswith("Bearer ") or not cfg.ingest_token:
        return False
    return hmac.compare_digest(header[7:].strip().encode(), cfg.ingest_token.encode())


def install_upload(source: BinaryIO, db_path: Path) -> dict[str, Any]:
    """Validate a gzipped SQLite upload and swap it in atomically.

    `source` is a readable binary stream of the gzip (the spooled request body
    on the server, a BytesIO in tests). It is decompressed straight to disk in
    CHUNK-sized pieces, hashing as it goes: the whole database is never held in
    memory. Holding it was what killed the 512 MB container mid-ingest on
    2026-11-05 (67 MB body, joined, plus the 137 MB decompressed copy).

    Validation before the swap, never after: a truncated or wrong file must
    leave the served index exactly as it was."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".incoming-", suffix=".db", dir=db_path.parent)
    sha = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            try:
                with gzip.GzipFile(fileobj=source, mode="rb") as gz:
                    while chunk := gz.read(CHUNK):
                        size += len(chunk)
                        if size > MAX_DB_BYTES:
                            raise ValueError("upload exceeds size ceiling")
                        sha.update(chunk)
                        fh.write(chunk)
            # BadGzipFile is an OSError; a truncated stream is EOFError; a
            # corrupt deflate body is zlib.error, which is neither.
            except (OSError, EOFError, zlib.error) as e:
                raise ValueError(f"not gzip: {type(e).__name__}") from None
        with open(tmp, "rb") as fh:
            if fh.read(len(_SQLITE_MAGIC)) != _SQLITE_MAGIC:
                raise ValueError("not a SQLite database")
        try:
            conn = sqlite3.connect(tmp)
        except sqlite3.Error as e:
            raise ValueError(f"unreadable database: {type(e).__name__}") from None
        try:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("integrity check failed")
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not _REQUIRED_TABLES <= tables:
                raise ValueError(f"missing tables: {sorted(_REQUIRED_TABLES - tables)}")
            counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("sessions", "turns", "notes")}
            # Readers open mode=ro; a WAL file next to the swapped db would be
            # orphaned, so the uploaded copy must be a single self-contained file.
            conn.execute("PRAGMA journal_mode=DELETE")
        # A file with a valid header but damaged pages fails here, not above:
        # a bad upload, reported as one, never a 500 that looks like a crash.
        except sqlite3.DatabaseError as e:
            raise ValueError(f"unreadable database: {type(e).__name__}") from None
        finally:
            conn.close()
        os.replace(tmp, db_path)
        tmp = ""
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return {"status": "installed", "bytes": size, "sha256": sha.hexdigest(), **counts}
