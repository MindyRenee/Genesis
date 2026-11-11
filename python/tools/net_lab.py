"""Net lab — reachability, naming, and raw HTTP.

`ping` uses ICMP when the ping binary exists, else a TCP-connect
probe (honest about which it is). `dns` resolves names through the
system resolver. `http_request` is the real instrument — urllib with
method/headers/body, returning status, headers, and body with
timing. `port_check` is a TCP connect_ex.

Docs: https://docs.python.org/3/library/urllib.request.html ·
https://docs.python.org/3/library/socket.html
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

from tools.framework import ToolResult


def ping(host: str, count: int = 3, timeout: int = 5) -> ToolResult:
    """Is a host reachable? ICMP ping when available, TCP :443 probe else.

    https://man7.org/linux/man-pages/man8/ping.8.html
    """
    count = max(1, min(count, 10))
    exe = shutil.which("ping")
    if exe:
        try:
            r = subprocess.run(
                [exe, "-c", str(count), "-W", str(timeout), host],
                capture_output=True,
                text=True,
                timeout=count * timeout + 5,
                stdin=subprocess.DEVNULL,
            )
            return ToolResult(
                success=r.returncode == 0,
                output=r.stdout,
                error=r.stderr,
                data={"method": "icmp"},
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return ToolResult(success=False, error=str(e))
    # Fallback: TCP connect to :443 — measures reachability, not latency stats.
    try:
        t0 = time.monotonic()
        with socket.create_connection((host, 443), timeout=timeout):
            ms = (time.monotonic() - t0) * 1000
        return ToolResult(
            success=True,
            output=f"{host}: tcp/443 reachable in {ms:.0f} ms",
            data={"method": "tcp", "ms": ms},
        )
    except OSError as e:
        return ToolResult(success=False, error=f"{host}: unreachable ({e})")


def dns(name: str) -> ToolResult:
    """Resolve a name through the system resolver — all address families.

    https://docs.python.org/3/library/socket.html#socket.getaddrinfo
    """
    try:
        infos = socket.getaddrinfo(name, None)
    except socket.gaierror as e:
        return ToolResult(success=False, error=f"{name}: {e}")
    addrs = sorted({str(i[4][0]) for i in infos})
    return ToolResult(
        success=True,
        output="\n".join(addrs),
        data={"name": name, "addresses": addrs},
    )


def http_request(
    url: str,
    method: str = "GET",
    body: str = "",
    headers: str = "",
    timeout: int = 20,
) -> ToolResult:
    """A real HTTP instrument — method, headers, body, timing.

    ``headers`` is 'Key: Value; Key2: Value2'. POST/PUT/PATCH send
    ``body`` as UTF-8. Returns status, response headers (capped), and
    body (first 64 KiB) with round-trip ms.
    https://docs.python.org/3/library/urllib.request.html
    """
    hdrs: dict[str, str] = {}
    for h in headers.split(";"):
        if ":" in h:
            k, v = h.split(":", 1)
            hdrs[k.strip()] = v.strip()
    data = body.encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=max(1, min(timeout, 120))) as resp:
            raw = resp.read(1 << 16)
            ms = (time.monotonic() - t0) * 1000
            text = raw.decode("utf-8", errors="replace")
            return ToolResult(
                success=True,
                output=f"{resp.status} {resp.reason} — {len(raw)} bytes in {ms:.0f} ms",
                data={
                    "status": resp.status,
                    "headers": dict(list(resp.headers.items())[:32]),
                    "body": text,
                    "ms": ms,
                },
            )
    except urllib.error.HTTPError as e:
        return ToolResult(
            success=False,
            error=f"HTTP {e.code} {e.reason}",
            data={"status": e.code, "body": e.read(4096).decode("utf-8", errors="replace")},
        )
    except (urllib.error.URLError, OSError, ValueError) as e:
        return ToolResult(success=False, error=str(e))


def port_check(host: str, port: int, timeout: int = 5) -> ToolResult:
    """Is a TCP port open? connect_ex with timing."""
    try:
        t0 = time.monotonic()
        with socket.create_connection((host, int(port)), timeout=max(1, timeout)):
            ms = (time.monotonic() - t0) * 1000
        return ToolResult(
            success=True,
            output=f"{host}:{port} open ({ms:.0f} ms)",
            data={"open": True, "ms": ms},
        )
    except OSError as e:
        return ToolResult(
            success=False,
            error=f"{host}:{port} closed ({e})",
            data={"open": False},
        )


NET_TOOLS = (
    (
        "ping",
        "Is a host reachable? Real ICMP ping when the binary exists, "
        "else an honest TCP-connect probe to :443.",
        ping,
    ),
    (
        "dns",
        "Resolve a hostname through the system resolver — every address family, deduplicated.",
        dns,
    ),
    (
        "http_request",
        "Real HTTP: method=, url, headers= 'K: V; K2: V2', body=. "
        "Returns status, headers, first 64KiB of body, round-trip ms.",
        http_request,
    ),
    (
        "port_check",
        "Is a TCP port open on a host? connect_ex with latency. host, port, timeout= seconds.",
        port_check,
    ),
)
