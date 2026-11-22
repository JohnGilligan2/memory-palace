"""Every variable hosted.Config reads must be in the compose `environment:`
block. Stack managers such as Portainer only substitute into compose; a variable the
code reads but compose omits silently falls back to its default in production."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def test_compose_carries_every_env_var_the_code_reads():
    code = (HERE / "hosted.py").read_text(encoding="utf-8") + \
        (HERE / "mcp_server.py").read_text(encoding="utf-8")
    read = set(re.findall(r'_env\("([A-Z_]+)"', code)) | \
        set(re.findall(r'environ\.get\("([A-Z_]+)"', code))
    compose = (HERE / "docker-compose.yml").read_text(encoding="utf-8")
    declared = set(re.findall(r"^\s{6}([A-Z_]+):", compose, re.M))
    missing = read - declared
    assert not missing, f"read by code but missing from compose: {sorted(missing)}"


def test_image_ships_every_module_the_server_imports():
    docker = (HERE / "Dockerfile").read_text(encoding="utf-8")
    copied = set(re.findall(r"(\w+)\.py", docker.split("COPY mcp_server.py", 1)[1].splitlines()[0]))
    copied.add("mcp_server")
    local = {p.stem for p in HERE.glob("*.py")}
    imported = set()
    for mod in copied:
        src = (HERE / f"{mod}.py").read_text(encoding="utf-8")
        imported |= {m for m in re.findall(r"^\s*import (\w+)", src, re.M) if m in local}
        imported |= {m for m in re.findall(r"^\s*from (\w+) import", src, re.M) if m in local}
    # known_secrets is imported lazily, only in local mode.
    imported.discard("known_secrets")
    assert imported <= copied, f"imported but not COPY'd: {sorted(imported - copied)}"
