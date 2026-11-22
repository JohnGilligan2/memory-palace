"""Per-tool-call metrics, emitted to the house OTEL collector.

WHAT THIS CAN AND CANNOT TELL YOU ABOUT TOKENS
----------------------------------------------
An MCP server never calls a model. Claude calls *us*. So this server has no
Anthropic API key, spends no tokens, and cannot report "its" usage the way an
app that calls the API can.

What it CAN measure is its own contribution to somebody else's context, which is
the honest and useful number:

  * `response_bytes` / `tokens_est` — every byte we hand back is pasted into the
    caller's context window. This is the dominant, controllable cost. It is also
    exactly what the CSV-export work was about: a week of call records was
    ~530,000 tokens inline versus a one-line download link.
  * `schema_tokens_est` — our tool definitions are re-sent on EVERY request in
    that conversation. 18 tools of JSON schema is a fixed tax on every single
    turn, whether a tool is called or not, and nobody ever looks at it.

The estimate is `bytes / 4`, the standard rough ratio for English+JSON. It is an
ESTIMATE and labelled as one; do not bill anyone from it. For actual spend, the
Claude Code telemetry already flowing into the same Prometheus
(`claude_code_token_usage_tokens_total`, `claude_code_cost_usage_USD_total`) is
the authoritative source, broken down per user and model. The two answer
different questions: Claude Code says what was spent, this says how much of the
input we were responsible for.

DESIGN CONSTRAINTS
------------------
This runs inside a customer-facing server, so it is built to be ignorable:

  * ZERO new dependencies. The collector's OTLP/HTTP receiver accepts JSON
    (verified against the live collector: 200 `{"partialSuccess":{}}`), so this
    hand-rolls the payload with the standard library rather than pulling the
    opentelemetry SDK into a dependency set that was just deliberately pinned.
  * NEVER blocks a tool call. Points go on a bounded queue; a daemon thread
    flushes them. A full queue drops points rather than waiting.
  * NEVER raises. Any failure is counted and swallowed. Telemetry must not be
    able to break a customer's invoice lookup.
  * OFF unless configured. No endpoint, no thread, no overhead.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import json
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Rough chars-per-token for JSON + English. Used only for the *_est metrics.
CHARS_PER_TOKEN = 4
_QUEUE_MAX = 5000
_FLUSH_SECONDS = 15.0
_BATCH_MAX = 500
# How long a brand-new counter series is exported at 0 before its real total.
# Must cover at least two Prometheus scrapes (a 30s scrape interval is assumed),
# so Prometheus is guaranteed to store the 0. See the note in _flush.
_NEW_SERIES_HOLD_SECONDS = 75.0


def _parse_headers(raw: str) -> dict[str, str]:
    """Parse OTEL_EXPORTER_OTLP_HEADERS ('K=V,K2=V2').

    Same format Claude Code's managed-settings.json uses, so one convention
    covers both senders. Values may contain '=' (bearer tokens do not, but be
    safe), so split once only.
    """
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass
class _Point:
    metric: str
    value: float
    attrs: dict[str, str]
    is_gauge: bool = False
    at_ns: int = field(default_factory=time.time_ns)


class Telemetry:
    """Fire-and-forget OTLP/JSON metric emitter."""

    def __init__(self, *, endpoint: str | None, headers: dict[str, str] | None = None,
                 service_name: str = "mcp-server",
                 resource_attrs: dict[str, str] | None = None,
                 timeout: float = 10.0) -> None:
        base = (endpoint or "").rstrip("/")
        # Accept either the collector root or a full /v1/metrics URL, because
        # OTEL_EXPORTER_OTLP_ENDPOINT conventionally means the root.
        self._url = base if base.endswith("/v1/metrics") else (base + "/v1/metrics" if base else "")
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._service = service_name
        self._resource = {"service.name": service_name, **(resource_attrs or {})}
        self._timeout = timeout
        self._q: queue.Queue[_Point] = queue.Queue(maxsize=_QUEUE_MAX)
        # Running totals per (metric, attrs). REQUIRED, not an optimisation:
        # the collector's Prometheus exporter does not accumulate DELTA sums
        # into a rising counter, so emitting deltas leaves every counter flat
        # and `increase()`/`rate()` return 0 forever — a dashboard of zeros
        # that looks healthy. We therefore accumulate here and emit CUMULATIVE,
        # which is what an OTel SDK does.
        self._totals: dict[tuple, float] = {}
        # Last value of each gauge, re-sent on every flush. Gauges here are
        # emitted only when the event that produces them happens — the tool
        # schema size comes from a list_tools call, which a client may do once
        # and then not again for hours. Without a heartbeat the series goes
        # stale after ~5 minutes and BOTH the panel and the dashboard's server
        # picker (which lists servers via label_values on a gauge) go empty
        # while the server is simply idle rather than broken.
        self._last_gauges: dict[tuple, float] = {}
        # When each counter series was first exported (monotonic seconds).
        self._born: dict[tuple, float] = {}
        self._started = False
        self._lock = threading.Lock()
        self._start_ns = time.time_ns()
        self.dropped = 0
        self.failed_flushes = 0
        self.sent_points = 0

    # --- lifecycle -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        with self._lock:
            if self._started:
                return
            t = threading.Thread(target=self._run, name="mcp-telemetry", daemon=True)
            t.start()
            self._started = True
        logger.info("telemetry -> %s (service=%s)", self._url, self._service)

    # --- recording -----------------------------------------------------------

    def count(self, metric: str, value: float = 1, **attrs: Any) -> None:
        self._put(_Point(metric, float(value), _clean_attrs(attrs)))

    def gauge(self, metric: str, value: float, **attrs: Any) -> None:
        self._put(_Point(metric, float(value), _clean_attrs(attrs), is_gauge=True))

    def _put(self, p: _Point) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(p)
        except queue.Full:
            # Losing a metric is always preferable to slowing a tool call.
            self.dropped += 1

    # --- export --------------------------------------------------------------

    def _run(self) -> None:
        while True:
            batch: list[_Point] = []
            deadline = time.monotonic() + _FLUSH_SECONDS
            while len(batch) < _BATCH_MAX:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            # Flush even when `batch` is empty: the heartbeat below is the
            # whole point, and an idle server produces no new points.
            if batch or self._last_gauges or self._totals:
                try:
                    self._flush(batch)
                except Exception as e:  # noqa: BLE001 — telemetry never raises
                    self.failed_flushes += 1
                    logger.debug("telemetry flush failed: %s", e)

    def _flush(self, batch: list[_Point]) -> None:
        # Aggregate identical (metric, attrs) pairs so one flush is one point
        # per series rather than one per call — keeps the payload small and the
        # collector happy.
        sums: dict[tuple, float] = {}
        gauges: dict[tuple, tuple[float, int]] = {}
        for p in batch:
            key = (p.metric, tuple(sorted(p.attrs.items())))
            if p.is_gauge:
                gauges[key] = (p.value, p.at_ns)
                self._last_gauges[key] = p.value
            else:
                sums[key] = sums.get(key, 0.0) + p.value
        # Heartbeat: re-assert every gauge we have ever seen, so an idle server
        # keeps reporting instead of vanishing from the dashboard.
        now_ns = time.time_ns()
        for key, val in self._last_gauges.items():
            gauges.setdefault(key, (val, now_ns))
        # ...and every COUNTER too. Without this a counter is only written in
        # flushes that had activity, so five minutes after the last tool call
        # Prometheus marks the series stale and it disappears: an instant query
        # for mcp_tool_calls_total returns nothing at all, and any last-value
        # panel reads empty on a server that simply is not busy right now. Since
        # these are cumulative totals, re-sending the same total is free of
        # meaning-change — rate() and increase() see a flat line, which is the
        # truth — and it keeps the series alive between bursts.
        for key in self._totals:
            sums.setdefault(key, 0.0)

        now = time.time_ns()
        mono = time.monotonic()
        metrics: list[dict[str, Any]] = []
        for (name, attrs), delta in sums.items():
            key = (name, attrs)
            self._totals[key] = self._totals.get(key, 0.0) + delta
            total = self._totals[key]
            # A NEW series is exported at 0 first, and its real total only once
            # Prometheus has scraped that 0. A counter that first appears already
            # at 1 has no transition to measure, so increase() reports ZERO for
            # it. That lost the first call of every (tool, user) pair, and
            # publish_identity cannot pre-create those series because users are
            # unknown at boot (and a gateway cannot list upstream tools without
            # a caller). Seen in production: users who had made tool calls
            # read as zero calls on the dashboard. The same
            # hold makes a restart visible as a reset, rather than a new process
            # resuming at a total that happens to be >= the old one.
            born = self._born.setdefault(key, mono)
            if mono - born < _NEW_SERIES_HOLD_SECONDS:
                total = 0.0
            metrics.append({
                # NO "unit" field on purpose. The collector's Prometheus
                # exporter appends a unit suffix, and unit "1" becomes
                # `_ratio` — so `mcp.schema.tokens_est` arrived as
                # `mcp_schema_tokens_est_ratio` and broke every dashboard
                # query. Omitting it leaves the name alone.
                "name": name,
                "sum": {
                    # CUMULATIVE. See the note on self._totals: DELTA leaves
                    # Prometheus counters flat and every rate panel at zero.
                    "aggregationTemporality": 2,
                    "isMonotonic": True,
                    "dataPoints": [{
                        "asDouble": total,
                        "startTimeUnixNano": str(self._start_ns),
                        "timeUnixNano": str(now),
                        "attributes": _otlp_attrs(dict(attrs)),
                    }],
                },
            })
        for (name, attrs), (val, at) in gauges.items():
            metrics.append({
                "name": name,          # see the note above: no unit suffix
                "gauge": {"dataPoints": [{
                    "asDouble": val,
                    "timeUnixNano": str(at),
                    "attributes": _otlp_attrs(dict(attrs)),
                }]},
            })
        if not metrics:
            return
        payload = {"resourceMetrics": [{
            "resource": {"attributes": _otlp_attrs(self._resource)},
            "scopeMetrics": [{"scope": {"name": "mcp.telemetry"},
                              "metrics": metrics}],
        }]}
        # urllib, not requests. This file is copied into every house MCP server
        # and `requests` is NOT a dependency of all of them — assuming it was
        # took three servers down on boot with
        # ModuleNotFoundError at import time. A telemetry module that adds a
        # dependency to nine repos will eventually break one of them, so it now
        # imports nothing outside the standard library.
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self._url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in (self._headers or {}).items():
            if k.lower() != "content-type":
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                if r.status >= 300:
                    raise RuntimeError(f"collector returned {r.status}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"collector returned {e.code}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"collector unreachable: {e.reason}") from e
        self.sent_points += len(metrics)

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "queued": self._q.qsize(),
                "sent_points": self.sent_points, "series": len(self._totals),
                "dropped": self.dropped, "failed_flushes": self.failed_flushes}


def _clean_attrs(attrs: dict[str, Any]) -> dict[str, str]:
    """Stringify label values and drop empties.

    Every label becomes a Prometheus dimension, so keep cardinality sane: no
    ids, no tokens, no free text. Callers pass tool names, statuses and a user
    principal — all bounded sets.
    """
    return {k: str(v) for k, v in attrs.items() if v not in (None, "")}


def _otlp_attrs(d: dict[str, str]) -> list[dict[str, Any]]:
    return [{"key": k, "value": {"stringValue": str(v)}} for k, v in d.items()]


def estimate_tokens(n_bytes: int) -> int:
    return int(n_bytes / CHARS_PER_TOKEN)


def from_env(service_name: str, **resource_attrs: str) -> Telemetry:
    """Build a Telemetry from the standard OTEL_* environment variables.

    Uses the same variable names and header format as the Claude Code telemetry
    already pointed at this collector, so there is one convention to remember
    and the bearer token can be copied verbatim.

    Disabled (a no-op object) when OTEL_EXPORTER_OTLP_ENDPOINT is unset — which
    is the default, so telemetry is opt-in per deployment.
    """
    if not _env_bool("MCP_TELEMETRY_ENABLED", True):
        return Telemetry(endpoint=None, service_name=service_name)
    return Telemetry(
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
        headers=_parse_headers(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")),
        service_name=os.environ.get("OTEL_SERVICE_NAME") or service_name,
        resource_attrs=resource_attrs,
    )


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")
