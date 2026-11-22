"""The hosted server: upload validation, the ingest gates, and a real boot.

The boot test starts mcp_server.py in hosted mode as a subprocess with stub
Entra values and drives it over HTTP: health, discovery document, the 401
challenge on POST /mcp, and an actual upload. House rule: unit-green is not
deploy-green, so the entry point itself gets exercised before any push.
"""
import gzip
import io
import ipaddress
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import tracemalloc
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import hosted

HERE = Path(__file__).resolve().parent.parent
TOKEN = "t" * 40


def _make_db(path: Path, tables=("notes", "turns", "sessions", "meta")) -> bytes:
    conn = sqlite3.connect(path)
    for t in tables:
        conn.execute(f"CREATE TABLE {t} (x TEXT)")
    if "meta" in tables:
        conn.execute("DROP TABLE meta")
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('built', ?)",
                     (json.dumps("2026-10-26T00:00:00+00:00"),))
    conn.commit()
    conn.close()
    return gzip.compress(path.read_bytes())


def test_install_upload_swaps_a_valid_db(tmp_path):
    payload = _make_db(tmp_path / "src.db")
    dest = tmp_path / "data" / "palace.db"
    r = hosted.install_upload(io.BytesIO(payload), dest)
    assert r["status"] == "installed" and dest.exists()
    assert r["sessions"] == 0 and len(r["sha256"]) == 64


def test_bad_uploads_leave_the_served_index_untouched(tmp_path):
    dest = tmp_path / "palace.db"
    good = _make_db(tmp_path / "good.db")
    hosted.install_upload(io.BytesIO(good), dest)
    before = dest.read_bytes()
    raw = gzip.decompress(good)
    corrupt_deflate = bytearray(good)
    corrupt_deflate[20:40] = bytes([0xFF]) * 20     # zlib.error, not OSError
    damaged_pages = gzip.compress(raw[:100] + bytes([0xFF]) * (len(raw) - 100))
    for bad in (b"not gzip", b"", gzip.compress(b"not sqlite"), gzip.compress(b""),
                good[: len(good) // 2],                    # truncated stream
                bytes(corrupt_deflate), damaged_pages,
                _make_db(tmp_path / "partial.db", tables=("notes",))):
        with pytest.raises(ValueError):
            hosted.install_upload(io.BytesIO(bad), dest)
    assert dest.read_bytes() == before
    assert not list(tmp_path.glob(".incoming-*")), "temp file left behind"


def test_install_upload_streams_instead_of_holding_the_db(tmp_path):
    # The 2026-11-05 nightly killed the 512 MB container: the old path held the
    # body and the whole decompressed database in memory. Python-heap peak must
    # stay a small multiple of CHUNK however big the upload is.
    src = tmp_path / "big.db"
    _make_db(src)
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE filler (b BLOB)")
    conn.executemany("INSERT INTO filler VALUES (?)",
                     ((os.urandom(64 * 1024),) for _ in range(40 * 16)))   # ~40 MB
    conn.commit()
    conn.close()
    gz = tmp_path / "big.db.gz"
    with open(src, "rb") as fi, gzip.open(gz, "wb", compresslevel=1) as fo:
        while b := fi.read(hosted.CHUNK):
            fo.write(b)
    tracemalloc.start()
    try:
        with open(gz, "rb") as fh:
            r = hosted.install_upload(fh, tmp_path / "data" / "palace.db")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert r["status"] == "installed" and r["bytes"] == src.stat().st_size
    assert peak < 8 * hosted.CHUNK, f"peak {peak / 1e6:.1f} MB: the upload is being held in memory"


def test_ingest_gates():
    cfg = hosted.Config(ingest_token=TOKEN,
                        ingest_allow=[ipaddress.ip_network("192.0.2.0/24")])
    assert hosted.token_ok(cfg, f"Bearer {TOKEN}")
    assert not hosted.token_ok(cfg, "Bearer wrong")
    assert not hosted.token_ok(cfg, None)
    assert hosted.client_allowed(cfg, "192.0.2.11")
    assert not hosted.client_allowed(cfg, "203.0.113.50")
    assert not hosted.client_allowed(cfg, None)


def test_refuses_to_boot_half_configured():
    with pytest.raises(SystemExit):
        hosted.Config(auth_enabled=True, ingest_token=TOKEN).validate()
    with pytest.raises(SystemExit):
        hosted.Config(auth_enabled=False, ingest_token="short").validate()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _req(url, data=None, headers=None, method=None):
    r = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_hosted_boot_end_to_end(tmp_path):
    port = _free_port()
    host = "https://palace.test.example"
    env = {**os.environ, "PALACE_MODE": "hosted", "MCP_HOST": "127.0.0.1",
           "MCP_PORT": str(port), "PALACE_DB": str(tmp_path / "data" / "palace.db"),
           "ENTRA_TENANT_ID": "00000000-0000-0000-0000-000000000001",
           "ENTRA_CLIENT_ID": "00000000-0000-0000-0000-000000000002",
           "ENTRA_IDENTIFIER_URI": f"{host}/mcp", "MCP_BASE_URL": host,
           "PALACE_INGEST_TOKEN": TOKEN, "PYTHONUTF8": "1"}
    # Output to a file, not a PIPE nobody reads: once the pipe buffer fills
    # with banner and access-log lines the server blocks on its next log write
    # and every later request times out.
    server_log = open(tmp_path / "server.log", "wb")
    proc = subprocess.Popen([sys.executable, str(HERE / "mcp_server.py")], env=env,
                            cwd=HERE, stdout=server_log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                status, _, body = _req(f"{base}/healthz")
                if status == 200:
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("server never answered /healthz")
        assert json.loads(body)["index"] == "missing"

        # Discovery: full-URL scope plus offline_access, or the connector dies.
        status, _, body = _req(f"{base}/.well-known/oauth-protected-resource/mcp")
        assert status == 200
        scopes = json.loads(body)["scopes_supported"]
        assert f"{host}/mcp/access_as_user" in scopes and "offline_access" in scopes

        # The challenge only comes from a POST (GET is 405 on fastmcp 3.4.5).
        status, headers, _ = _req(f"{base}/mcp", data=b"{}", method="POST",
                                  headers={"Content-Type": "application/json"})
        assert status == 401
        assert any(k.lower() == "www-authenticate" for k in headers)

        # Upload: wrong token refused, right token installs, health turns present.
        payload = _make_db(tmp_path / "up.db")
        status, _, _ = _req(f"{base}/ingest", data=payload, method="POST",
                            headers={"Authorization": "Bearer nope"})
        assert status == 401
        # Through the public proxy: refused even with the right token.
        status, _, _ = _req(f"{base}/ingest", data=payload, method="POST",
                            headers={"Authorization": f"Bearer {TOKEN}",
                                     "X-Forwarded-For": "203.0.113.50"})
        assert status == 403
        status, _, body = _req(f"{base}/ingest", data=payload, method="POST",
                               headers={"Authorization": f"Bearer {TOKEN}"})
        assert status == 200, body
        assert json.loads(body)["status"] == "installed"
        # A multi-chunk body through the real route (streamed, spooled to disk)
        # comes back with the checksum the laptop computed.
        big = tmp_path / "big.db"
        _make_db(big)
        conn = sqlite3.connect(big)
        conn.execute("CREATE TABLE filler (b BLOB)")
        conn.executemany("INSERT INTO filler VALUES (?)",
                         ((os.urandom(64 * 1024),) for _ in range(64)))   # ~4 MB
        conn.commit()
        conn.close()
        raw = big.read_bytes()
        status, _, body = _req(f"{base}/ingest", data=gzip.compress(raw), method="POST",
                               headers={"Authorization": f"Bearer {TOKEN}"})
        assert status == 200, body
        import hashlib
        assert json.loads(body)["sha256"] == hashlib.sha256(raw).hexdigest()
        # A corrupt body is a 422 and leaves no spool file behind.
        status, _, _ = _req(f"{base}/ingest", data=b"\x1f\x8b garbage", method="POST",
                            headers={"Authorization": f"Bearer {TOKEN}"})
        assert status == 422
        assert not list((tmp_path / "data").glob(".upload-*")), "spool file left behind"
        assert not list((tmp_path / "data").glob(".incoming-*")), "temp db left behind"
        status, _, body = _req(f"{base}/healthz")
        assert json.loads(body)["index"] in ("fresh", "stale")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        server_log.close()
