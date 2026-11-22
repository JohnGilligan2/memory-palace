"""fastmcp middleware that records what each tool call cost.

COPY THIS FILE, together with telemetry.py, into an MCP server's package. Copied
rather than packaged on purpose: a shared pinned dependency across these repos is
exactly what caused an earlier connector incident. Neither file imports
anything outside the standard library plus `requests` and `fastmcp`.

`usage_log.py` is OPTIONAL and only exists where tool calls are billable. Servers
without it pass usage=None and lose nothing but the billing sink.


Hooks two things:

  `on_call_tool`   — one call: who, which tool, how long, how many bytes came
                     back, and whether it failed. The bytes matter most: they
                     land in the caller's context window.
  `on_list_tools`  — the size of our tool schemas. This is the invisible cost:
                     the definitions are re-sent on EVERY request in the
                     conversation, called or not, so 18 tools of JSON schema is
                     a fixed tax per turn.

Everything here is wrapped so telemetry can never break a tool. If recording
fails, the tool result still returns.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastmcp.server.middleware import Middleware, MiddlewareContext

from telemetry import Telemetry, estimate_tokens

if TYPE_CHECKING:                # pragma: no cover
    from usage_log import UsageLog

logger = logging.getLogger(__name__)

# Metric names. Dots become underscores in Prometheus, so these arrive as
# mcp_tool_calls_total, mcp_tool_response_bytes_total, etc.
M_CALLS = "mcp.tool.calls"
M_DURATION_MS = "mcp.tool.duration_ms"
M_RESPONSE_BYTES = "mcp.tool.response_bytes"
M_RESPONSE_TOKENS = "mcp.tool.response_tokens_est"
M_SCHEMA_TOKENS = "mcp.schema.tokens_est"
M_SCHEMA_BYTES = "mcp.schema.bytes"
# Counters, not gauges. See the note in on_list_tools for why both are needed:
# the gauges give the SIZE of a schema, these give how often it was actually
# served and the cumulative context cost of doing so.
M_SCHEMA_REQUESTS = "mcp.schema.requests"
M_SCHEMA_TOKENS_SERVED = "mcp.schema.tokens_served_est"
M_TOOL_COUNT = "mcp.schema.tool_count"

# --- error classification ----------------------------------------------------
# A single "error rate" number turned out to be worse than useless. The live
# figure read 6.4% and every one of the underlying failures was a tool correctly
# telling the caller "this account has three domains, pick one" — the model then
# picked one and got its answer. Nothing was broken. Meanwhile a real fault would
# have been buried in the same number.
#
# So errors are classified, and the two are never added together:
#
#   bad_input     the caller asked for something we could not act on, and the
#                 message tells it how to fix that. Self-correcting. Expected on
#                 any multi-domain account and NOT a fault.
#   denied        the scope boundary refused. Also working as designed, but worth
#                 watching separately: a rise means someone is probing.
#   upstream      a backend service we depend on failed. Real, but not our bug.
#   internal      an exception we did not anticipate. This is the only kind that
#                 means "go and look at the code", and it is the one that should
#                 page someone.
#
# Classified by exception CLASS NAME rather than by importing the exception
# types, because this module is copied verbatim into every house MCP server and
# must not depend on any one server's modules.
_BAD_INPUT_NAMES = frozenset({"ToolError", "ValueError", "ValidationError",
                              "McpError", "TypeError", "KeyError"})
_DENIED_NAMES = frozenset({"ScopeError", "PermissionError", "AuthError",
                           "Unauthorized", "Forbidden"})
_UPSTREAM_NAMES = frozenset({
    "UpstreamAPIError", "ExportUnavailable", "HTTPError", "Timeout",
    "ConnectTimeout", "ReadTimeout", "ConnectionError", "RequestException",
    "SSLError", "Fault", "ProtocolError", "TimeoutError", "OSError",
})


def classify_error(exc: BaseException) -> str:
    """Bucket an exception so a dashboard can separate faults from guidance."""
    for cls in type(exc).__mro__:
        name = cls.__name__
        if name in _DENIED_NAMES:
            return "denied"
        if name in _UPSTREAM_NAMES:
            return "upstream"
        if name in _BAD_INPUT_NAMES:
            return "bad_input"
    return "internal"


def _caller() -> str:
    """A low-cardinality identity for the calling user.

    Prefers the Entra UPN so a Grafana breakdown reads as people rather than
    GUIDs. Falls back to 'anonymous' when auth is off (local testing) and
    'unknown' if the claim shape surprises us — never raises, and never puts a
    raw token anywhere near a metric label.
    """
    try:
        from fastmcp.server.dependencies import get_access_token

        tok = get_access_token()
        if tok is None:
            return "anonymous"
        claims = getattr(tok, "claims", None) or {}
        for key in ("preferred_username", "upn", "unique_name", "email", "sub"):
            v = claims.get(key)
            if v:
                return str(v)
        return getattr(tok, "client_id", None) or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _sizeof(result: Any) -> int:
    """Bytes of the payload as the caller will receive it.

    Best-effort: serialise whatever the tool returned. A tool result that will
    not serialise is measured by its repr, which is close enough for a cost
    signal and cannot fail.
    """
    try:
        content = getattr(result, "content", None)
        structured = getattr(result, "structured_content", None)
        if content is not None or structured is not None:
            total = 0
            for block in (content or []):
                text = getattr(block, "text", None)
                total += len(text.encode("utf-8")) if text else len(repr(block))
            if structured is not None:
                total += len(json.dumps(structured, default=str).encode("utf-8"))
            return total
        return len(json.dumps(result, default=str).encode("utf-8"))
    except Exception:  # noqa: BLE001
        try:
            return len(repr(result).encode("utf-8"))
        except Exception:  # noqa: BLE001
            return 0


class TelemetryMiddleware(Middleware):
    """Records per-tool-call cost signals. Never changes behaviour."""

    def __init__(self, telemetry: Telemetry,
                 usage: "UsageLog | None" = None) -> None:
        self._t = telemetry
        # Two sinks, deliberately different guarantees: telemetry is lossy and
        # async (observability), usage is synchronous and durable (billing).
        self._u = usage

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool = getattr(getattr(context, "message", None), "name", None) or "unknown"
        user = _caller()
        started = time.perf_counter()
        status = "ok"
        error_kind = ""
        result = None
        try:
            result = await call_next(context)
            return result
        except Exception as exc:
            status = "error"
            error_kind = classify_error(exc)
            raise
        finally:
            # In a finally block so an erroring tool is still counted — an error
            # rate you cannot see is worse than no error rate.
            try:
                ms = (time.perf_counter() - started) * 1000.0
                # error_kind is a label on the call counter so one panel can
                # show real faults and another can show self-correcting guidance,
                # instead of one number that means neither.
                self._t.count(M_CALLS, 1, tool=tool, status=status, user=user,
                              error_kind=error_kind or "none")
                self._t.count(M_DURATION_MS, ms, tool=tool, status=status)
                n = _sizeof(result) if (status == "ok" and result is not None) else 0
                if n:
                    self._t.count(M_RESPONSE_BYTES, n, tool=tool)
                    self._t.count(M_RESPONSE_TOKENS, estimate_tokens(n), tool=tool)
                if self._u is not None:
                    # Billable events are recorded for FAILED calls too: a
                    # rejected out-of-scope request is evidence, and a month of
                    # errors is something a customer may ask about.
                    self._u.record(tool=tool, user=user, status=status,
                                   duration_ms=ms, response_bytes=n,
                                   tokens_est=estimate_tokens(n),
                                   error_kind=error_kind)
            except Exception as e:  # noqa: BLE001
                logger.debug("failed to record tool metrics: %s", e)

    async def on_list_tools(self, context: MiddlewareContext, call_next):
        tools = await call_next(context)
        try:
            # Measure what the client is actually handed, since that is what
            # occupies context on every subsequent request.
            payload = []
            for t in tools or []:
                payload.append({
                    "name": getattr(t, "name", ""),
                    "description": getattr(t, "description", "") or "",
                    "inputSchema": getattr(t, "inputSchema", None)
                                   or getattr(t, "input_schema", None) or {},
                })
            raw = json.dumps(payload, default=str).encode("utf-8")
            self._t.gauge(M_SCHEMA_BYTES, len(raw))
            self._t.gauge(M_SCHEMA_TOKENS, estimate_tokens(len(raw)))
            self._t.gauge(M_TOOL_COUNT, len(payload))
            # The gauges above say how BIG the schema is. They cannot say how
            # OFTEN it was handed to a client, because a gauge holds its last
            # value forever and the exporter re-sends it on a fixed cycle
            # whether or not anything happened. So `count_over_time()` on a
            # gauge measures the export cadence, not schema loads — a trap that
            # produces a confidently wrong answer (every server looks equally
            # busy, including idle ones).
            #
            # This counter is the actual event. Read it as SESSIONS, not
            # requests: a client calls list_tools once on connect, caches the
            # result, and then carries the schemas in every later request
            # WITHOUT asking us again. We never see that number.
            #
            # Verified empirically 2026-08-10: reconnecting a server from two
            # fresh sessions produced exactly 2 loads; further requests inside
            # those sessions produced none.
            #
            # It is still the metric that matters, from the other direction —
            # zero loads means nothing ever fetched these schemas, so they
            # cannot be sitting in anyone's context. That is what distinguishes
            # "in circulation" from "container merely running", which the
            # gauge alone could never tell you.
            self._t.count(M_SCHEMA_REQUESTS, 1)
            self._t.count(M_SCHEMA_TOKENS_SERVED, estimate_tokens(len(raw)))
        except Exception as e:  # noqa: BLE001
            logger.debug("failed to record schema metrics: %s", e)
        return tools


def publish_identity(mcp: Any, telemetry: Telemetry) -> None:
    """Announce a server's existence and schema cost at STARTUP.

    Call this from main(), BEFORE mcp.run().

    Without it a server only appears in Grafana once something calls list_tools,
    because that is what produced the schema gauges. A freshly restarted server
    that nobody has used therefore reported nothing at all — it dropped out of
    the dashboard's server picker, the `$service` variable resolved to no values,
    and every panel read "No data".

    That is exactly backwards for the question the dashboard exists to answer.
    "Is anyone actually using this tool?" requires an UNUSED server to be visible
    with zero calls against its name. A missing row is indistinguishable from a
    broken exporter or a dead container; a row of zeros is an answer.

    Never raises: a telemetry problem must not stop a server booting.
    """
    if not telemetry.enabled:
        return
    try:
        import asyncio

        # run_middleware=False: at boot there is no caller, and an access gate
        # in the middleware chain (mcp_gate in enforce mode) correctly answers
        # an anonymous tools/list with []. Through middleware, ufw-mcp announced
        # 0 tools in production (2026-10-27) and no zero rows, so its unused
        # tools were invisible. Older fastmcp lacks the keyword.
        try:
            tools = asyncio.run(mcp.list_tools(run_middleware=False))
        except TypeError:
            tools = asyncio.run(mcp.list_tools())
        payload = [{"name": getattr(x, "name", ""),
                    "description": getattr(x, "description", "") or "",
                    "inputSchema": getattr(x, "inputSchema", None)
                                   or getattr(x, "input_schema", None) or {}}
                   for x in (tools or [])]
        raw = json.dumps(payload, default=str).encode("utf-8")
        telemetry.gauge(M_SCHEMA_BYTES, len(raw))
        telemetry.gauge(M_SCHEMA_TOKENS, estimate_tokens(len(raw)))
        telemetry.gauge(M_TOOL_COUNT, len(payload))
        # Zero-init the schema counters — deliberately 0, not 1.
        #
        # This boot-time announce calls mcp.list_tools() directly and so does NOT
        # pass through TelemetryMiddleware.on_list_tools. That is the behaviour we
        # want: a server describing itself at startup is not a client asking for
        # its schemas, and counting it as one would make every restart look like
        # real usage. Restart-heavy deploys (GitOps rebuilds on every push) would
        # then manufacture exactly the signal you are trying to measure.
        #
        # But the series still has to EXIST at zero, for the same
        # counter-first-appears-at-1 reason described below.
        telemetry.count(M_SCHEMA_REQUESTS, 0)
        telemetry.count(M_SCHEMA_TOKENS_SERVED, 0)
        # A zero-valued counter FOR EVERY TOOL, so each series exists at 0 before
        # anyone calls it.
        #
        # This is not cosmetic. A Prometheus counter that first appears
        # mid-window already at 1 reports an increase() of ZERO, because
        # Prometheus never observed it at 0 and so has no transition to measure.
        # Without these, the first call to any tool is invisible on the
        # dashboard, and a tool used exactly once looks identical to a tool
        # nobody has ever touched — which is the single question this dashboard
        # exists to answer.
        #
        # It also means an unused tool is a visible row of zeros rather than an
        # absent series, for the same reason publish_identity exists at all.
        for spec in payload:
            name = spec.get("name")
            if name:
                telemetry.count(M_CALLS, 0, tool=name, status="ok", user="none",
                                error_kind="none")
        telemetry.count(M_CALLS, 0, tool="none", status="startup", user="none",
                        error_kind="none")
        logger.info("published identity: %d tools, ~%d schema tokens per request",
                    len(payload), estimate_tokens(len(raw)))
    except Exception as e:  # noqa: BLE001 — never block startup on telemetry
        logger.warning("could not publish startup identity: %s", e)
