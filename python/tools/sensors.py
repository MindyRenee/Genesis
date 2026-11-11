"""Sensors — proprioception of the host beyond CPU effort.

Thermal zones, network interfaces, mounted filesystems, and the
hardware census — all read straight from /sys and /proc, the kernel's
own accounting. Read-only; a sensor never changes the machine.

Docs: https://www.kernel.org/doc/html/latest/filesystems/sysfs.html ·
https://man7.org/linux/man-pages/man5/proc.5.html
"""

from __future__ import annotations

import pathlib
import shutil
import socket
from typing import Any

from tools.framework import ToolResult


def thermal(**_kwargs: Any) -> ToolResult:
    """Read every thermal zone the kernel exposes — her temperature sense.

    /sys/class/thermal/thermal_zone*/{type,temp}; temps are
    millidegrees C. Absent on VMs without thermal hardware — reported
    honestly. https://www.kernel.org/doc/html/latest/driver-api/thermal/
    """
    zones: list[dict[str, Any]] = []
    try:
        for z in sorted(pathlib.Path("/sys/class/thermal").glob("thermal_zone*")):
            try:
                kind = (z / "type").read_text().strip()
                milli = int((z / "temp").read_text().strip())
                zones.append({"zone": z.name, "type": kind, "celsius": milli / 1000.0})
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    if not zones:
        return ToolResult(success=False, error="no thermal zones exposed on this host")
    return ToolResult(
        success=True,
        output="; ".join(f"{z['type']} {z['celsius']:.1f}°C" for z in zones),
        data={"zones": zones},
    )


def net_interfaces(**_kwargs: Any) -> ToolResult:
    """Network interface census — links, states, addresses.

    /sys/class/net for link state and MTU; `ip -o addr` for addresses
    when iproute2 exists; hostname resolution as fallback for the
    primary IPv4. Read-only.
    https://man7.org/linux/man-pages/man8/ip-address.8.html
    """
    ifaces: dict[str, dict[str, Any]] = {}
    try:
        for i in sorted(pathlib.Path("/sys/class/net").iterdir()):
            name = i.name
            try:
                operstate = (i / "operstate").read_text().strip()
            except OSError:
                operstate = "?"
            try:
                mtu = int((i / "mtu").read_text().strip())
            except (OSError, ValueError):
                mtu = 0
            try:
                mac = (i / "address").read_text().strip()
            except OSError:
                mac = ""
            ifaces[name] = {"state": operstate, "mtu": mtu, "mac": mac, "addrs": []}
    except OSError:
        pass
    if not ifaces:
        # /sys is not mounted in every container/proot — fall back to
        # the kernel's socket API, which always answers.
        try:
            for _idx, name in socket.if_nameindex():
                ifaces[name] = {"state": "up", "mtu": 0, "mac": "", "addrs": []}
        except OSError:
            pass
    ip = shutil.which("ip")
    if ip:
        import subprocess

        try:
            out = subprocess.run(
                [ip, "-o", "addr", "show"],
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
            for line in out.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 4 and parts[1] in ifaces:
                    ifaces[parts[1]]["addrs"].append(parts[3])
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not ifaces:
        return ToolResult(success=False, error="no network interfaces visible")
    lines = [
        f"{n} {v['state']} mtu={v['mtu']} {','.join(v['addrs']) or v['mac']}"
        for n, v in ifaces.items()
    ]
    return ToolResult(success=True, output="\n".join(lines), data={"interfaces": ifaces})


def disk_usage(**_kwargs: Any) -> ToolResult:
    """Every mounted real filesystem's usage — her `df`.

    Reads /proc/mounts, keeps real filesystems (ext4/xfs/btrfs/vfat/
    f2fs/tmpfs), reports total/used/free per mountpoint.
    """
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    try:
        for line in pathlib.Path("/proc/mounts").read_text().splitlines():
            dev, mount, fstype, *_ = line.split()
            if fstype not in (
                "ext4",
                "ext3",
                "xfs",
                "btrfs",
                "vfat",
                "f2fs",
                "exfat",
                "ntfs",
                "tmpfs",
                "overlay",
            ):
                continue
            if mount in seen:
                continue
            seen.add(mount)
            try:
                du = shutil.disk_usage(mount)
            except OSError:
                continue
            rows.append(
                {
                    "device": dev,
                    "mount": mount,
                    "fstype": fstype,
                    "total": du.total,
                    "used": du.used,
                    "free": du.free,
                }
            )
    except OSError as e:
        return ToolResult(success=False, error=str(e))
    if not rows:
        return ToolResult(success=False, error="no real filesystems found")
    gib = 1 << 30
    lines = [
        f"{r['mount']:24} {r['fstype']:7} {r['used'] / gib:7.1f}/{r['total'] / gib:7.1f} GiB "
        f"({100 * r['used'] / max(r['total'], 1):.0f}%)"
        for r in rows
    ]
    return ToolResult(success=True, output="\n".join(lines), data={"filesystems": rows})


def hardware_info(**_kwargs: Any) -> ToolResult:
    """Hardware census: CPU model/cores, memory, block devices, GPUs.

    /proc/cpuinfo + /proc/meminfo + /sys/block — the same sources
    lscpu/lsblk read. Read-only.
    """
    data: dict[str, Any] = {}
    try:
        model = ""
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("model name", "Model", "Hardware")):
                model = model or line.split(":", 1)[1].strip()
        data["cpu_model"] = model
        data["cpu_cores"] = __import__("os").cpu_count()
    except OSError:
        pass
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                data["mem_total_bytes"] = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        pass
    try:
        blocks = []
        for b in sorted(pathlib.Path("/sys/block").iterdir()):
            try:
                sectors = int((b / "size").read_text().strip())
                blocks.append({"name": b.name, "bytes": sectors * 512})
            except (OSError, ValueError):
                continue
        data["block_devices"] = blocks
    except OSError:
        pass
    gpus = (
        [c.name for c in pathlib.Path("/sys/class/drm").glob("card[0-9]*")]
        if pathlib.Path("/sys/class/drm").is_dir()
        else []
    )
    data["gpus"] = gpus
    gib = 1 << 30
    parts = [data.get("cpu_model", "cpu?"), f"{data.get('cpu_cores', '?')} cores"]
    if "mem_total_bytes" in data:
        parts.append(f"{data['mem_total_bytes'] / gib:.1f} GiB")
    if data.get("block_devices"):
        total = sum(b["bytes"] for b in data["block_devices"])
        parts.append(f"{total / gib:.0f} GiB storage across {len(data['block_devices'])} devices")
    return ToolResult(success=True, output="; ".join(parts), data=data)


SENSOR_TOOLS = (
    (
        "thermal",
        "Every thermal zone's temperature (°C) from /sys/class/thermal "
        "— her sense of the machine's heat. Read-only.",
        thermal,
    ),
    (
        "net_interfaces",
        "Network interface census: link state, MTU, MAC, and IPv4/IPv6 "
        "addresses when iproute2 is installed. Read-only.",
        net_interfaces,
    ),
    (
        "disk_usage",
        "Usage of every real mounted filesystem (ext4/xfs/btrfs/tmpfs/"
        "…) — total, used, free per mountpoint. Her df.",
        disk_usage,
    ),
    (
        "hardware_info",
        "Hardware census: CPU model and core count, total memory, "
        "block devices with sizes, DRM GPU nodes. Read-only.",
        hardware_info,
    ),
)
