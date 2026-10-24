"""Krita bridge — the connection between her and the real program.

Krita is a drawing application on the machine (flatpak
``org.kde.krita``). This module installs a small PyKrita extension,
``genesis_studio``, which watches a command queue on disk and executes
painting operations inside a live Krita document with Krita's own
brushes, colors, and paper — the same way a human uses the program.

The channel is plain files under Krita's own data directory, so it
works regardless of what Qt modules the flatpak bundles:

- ``~/.var/app/org.kde.krita/data/genesis_studio/queue/op_<seq>.json``
  — operations written by Genesis,
- ``ack_<seq>.json`` — results written back by the plugin,
- ``heartbeat`` — refreshed every second while the studio is alive.

Everything here is best-effort: if Krita is absent or unreachable,
the canvas's built-in rasterizer still lets her paint.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["KritaStudio"]

_PLUGIN_ID = "genesis_studio"

# ── The plugin source ─────────────────────────────────────────────
# Installed into Krita's pykrita directory on first use. Plain
# stdlib + QtCore (QTimer) + the krita module — nothing else assumed.

_PLUGIN_INIT = """from krita import Krita
from .studio import GenesisStudio

Krita.instance().addExtension(GenesisStudio(Krita.instance()))
"""

_PLUGIN_DESKTOP = """[Desktop Entry]
Type=Service
ServiceTypes=Krita/PythonPlugin
X-KDE-Library=genesis_studio
X-Python-2-Compatible=false
Name=Genesis Studio
Comment=Genesis's painting surface — executes ops from queue files.
"""

_PLUGIN_STUDIO = '''"""Genesis Studio — a PyKrita extension that paints what she sends.

Watches ~/.var/app/org.kde.krita/data/genesis_studio/queue for
op_*.json files, executes each op inside a live Krita document, and
writes ack_<seq>.json back. A heartbeat file is refreshed every
second while Krita is up so the client can tell the studio is alive.
"""

import glob
import json
import os
import time

from krita import Krita, Extension, InfoObject, ManagedColor
from PyQt5.QtCore import QPointF, QRect, QRectF, QTimer
from PyQt5.QtGui import QPainterPath

_STUDIO_DIR = os.path.expanduser(
    "~/.var/app/org.kde.krita/data/genesis_studio"
)
_QUEUE = os.path.join(_STUDIO_DIR, "queue")
_HEARTBEAT = os.path.join(_STUDIO_DIR, "heartbeat")

_FILL_STYLES = {"fill": "ForegroundColor", "none": "None"}


class GenesisStudio(Extension):
    """Extension entry point: a queue consumer on a QTimer."""

    def __init__(self, parent):
        super().__init__(parent)
        self.app = None
        self.doc = None
        self.paint_node = None
        self._timer = None
        self._beat = None

    def setup(self):
        self.app = Krita.instance()
        os.makedirs(_QUEUE, exist_ok=True)
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(150)
        self._beat = QTimer()
        self._beat.timeout.connect(self._heartbeat)
        self._beat.start(1000)
        self._heartbeat()

    def createActions(self, window):
        pass

    def _heartbeat(self):
        try:
            with open(_HEARTBEAT, "w") as f:
                f.write(str(time.time()))
        except OSError:
            pass

    def _poll(self):
        for path in sorted(glob.glob(os.path.join(_QUEUE, "op_*.json"))):
            seq = 0
            try:
                with open(path) as f:
                    msg = json.load(f)
                seq = int(msg.get("seq", 0))
            except (OSError, json.JSONDecodeError, ValueError):
                try:
                    os.unlink(path)
                except OSError:
                    pass
                continue
            try:
                result = self._exec(msg)
            except Exception as e:  # noqa: BLE001
                result = {"seq": seq, "ok": False, "error": str(e)}
            try:
                ack = os.path.join(_QUEUE, "ack_%d.json" % seq)
                with open(ack, "w") as f:
                    json.dump(result, f)
            except OSError:
                pass
            try:
                os.unlink(path)
            except OSError:
                pass

    # ── Helpers ───────────────────────────────────────────────────

    def _window(self):
        win = self.app.activeWindow()
        if win is None and self.app.windows():
            win = self.app.windows()[0]
        return win

    def _view(self):
        win = self._window()
        return win.activeView() if win is not None else None

    def _node(self):
        if self.doc is None:
            return None
        node = self.doc.activeNode()
        return node if node is not None else self.paint_node

    def _set_color_components(self, r, g, b, a):
        color = ManagedColor("RGBA", "U8", "")
        color.setComponents(
            [r / 255.0, g / 255.0, b / 255.0, float(a)]
        )
        view = self._view()
        if view is not None:
            view.setForeGroundColor(color)

    def _find_preset(self, keyword):
        presets = self.app.resources("preset")
        key = keyword.lower()
        for name in sorted(presets):
            if key in name.lower():
                return presets[name]
        return None

    # ── Ops ───────────────────────────────────────────────────────

    def _exec(self, msg):
        op = msg.get("op")
        self._ensure_doc_if_needed(op)
        if op == "ping":
            return {"seq": msg.get("seq"), "ok": True}
        if op == "new":
            return self._op_new(msg)
        if op == "color":
            self._set_color_components(
                msg.get("r", 0), msg.get("g", 0), msg.get("b", 0),
                msg.get("a", 1.0),
            )
            return {"seq": msg.get("seq"), "ok": True}
        if op == "tool":
            return self._op_tool(msg)
        if op == "brush":
            return self._op_brush(msg)
        if op == "stroke":
            return self._op_stroke(msg)
        if op == "line":
            return self._op_line(msg)
        if op == "ellipse":
            return self._op_shape(msg, "ellipse")
        if op == "rect":
            return self._op_shape(msg, "rect")
        if op == "polygon":
            return self._op_polygon(msg)
        if op == "fill":
            return self._op_fill(msg)
        if op == "clear":
            return self._op_clear()
        if op == "export":
            return self._op_export(msg)
        return {"seq": msg.get("seq"), "ok": False,
                "error": "unknown op %s" % op}

    def _ensure_doc_if_needed(self, op):
        if self.doc is not None or op in ("ping",):
            return
        if op == "new":
            return
        # An op arrived before a sheet — make one at default size.
        self._make_document(640, 480, (250, 250, 248))

    def _make_document(self, w, h, paper):
        app = self.app
        doc = app.createDocument(
            int(w), int(h), "Genesis", "RGBA", "U8", "", 300.0
        )
        win = self._window()
        if win is not None:
            win.addView(doc)
        self.doc = doc
        # Paint the paper color onto the background layer.
        root = doc.rootNode()
        kids = root.childNodes()
        bg = kids[-1] if kids else None
        if bg is not None:
            doc.setActiveNode(bg)
            self._set_color_components(*paper, 1.0)
            bg.paintRectangle(QRect(0, 0, int(w), int(h)),
                              "ForegroundColor", "None")
        layer = doc.createNode("genesis", "paintlayer")
        root.addChildNode(layer, None)
        doc.setActiveNode(layer)
        self.paint_node = layer
        doc.refreshProjection()

    def _op_new(self, msg):
        w = msg.get("width", 640)
        h = msg.get("height", 480)
        paper = msg.get("paper") or (250, 250, 248)
        self._make_document(w, h, tuple(paper[:3]))
        return {"seq": msg.get("seq"), "ok": True}

    def _op_tool(self, msg):
        name = str(msg.get("tool", ""))
        size = float(msg.get("size") or 0)
        view = self._view()
        preset = self._find_preset(name)
        if view is not None and preset is not None:
            view.activateResource(preset)
        if view is not None and size:
            try:
                view.setBrushSize(size)
            except AttributeError:
                pass
        return {"seq": msg.get("seq"), "ok": True,
                "resolved": preset.name() if preset is not None else None}

    def _op_brush(self, msg):
        view = self._view()
        if view is None:
            return {"seq": msg.get("seq"), "ok": False, "error": "no view"}
        size = float(msg.get("size") or 0)
        opacity = float(msg.get("opacity") or 0)
        if size:
            try:
                view.setBrushSize(size)
            except AttributeError:
                pass
        if opacity:
            try:
                view.setPaintingOpacity(opacity)
            except AttributeError:
                pass
        blend = msg.get("blend") or ""
        try:
            if msg.get("erase"):
                view.setCurrentBlendingMode("erase")
            elif blend:
                view.setCurrentBlendingMode(blend)
            else:
                view.setCurrentBlendingMode("normal")
        except AttributeError:
            if msg.get("erase"):
                preset = self._find_preset("eraser")
                if preset is not None:
                    view.activateResource(preset)
        return {"seq": msg.get("seq"), "ok": True}

    def _op_stroke(self, msg):
        node = self._node()
        if node is None:
            return {"seq": msg.get("seq"), "ok": False, "error": "no node"}
        pts = msg.get("points") or []
        if not pts:
            return {"seq": msg.get("seq"), "ok": True}
        path = QPainterPath()
        path.moveTo(QPointF(pts[0][0], pts[0][1]))
        for x, y in pts[1:]:
            path.lineTo(QPointF(x, y))
        node.paintPath(path, "None", "ForegroundColor")
        return {"seq": msg.get("seq"), "ok": True}

    def _op_line(self, msg):
        node = self._node()
        if node is None:
            return {"seq": msg.get("seq"), "ok": False, "error": "no node"}
        node.paintLine(
            QPointF(msg["x1"], msg["y1"]), QPointF(msg["x2"], msg["y2"]),
            "ForegroundColor",
        )
        return {"seq": msg.get("seq"), "ok": True}

    def _op_shape(self, msg, kind):
        node = self._node()
        if node is None:
            return {"seq": msg.get("seq"), "ok": False, "error": "no node"}
        rect = QRectF(msg["x"], msg["y"], msg["w"], msg["h"])
        fill = "ForegroundColor" if msg.get("fill") else "None"
        stroke = "None" if msg.get("fill") else "ForegroundColor"
        if kind == "ellipse":
            node.paintEllipse(rect, fill, stroke)
        else:
            node.paintRectangle(rect.toRect(), fill, stroke)
        return {"seq": msg.get("seq"), "ok": True}

    def _op_polygon(self, msg):
        node = self._node()
        if node is None:
            return {"seq": msg.get("seq"), "ok": False, "error": "no node"}
        pts = [QPointF(x, y) for x, y in msg.get("points") or []]
        if len(pts) < 3:
            return {"seq": msg.get("seq"), "ok": True}
        fill = "ForegroundColor" if msg.get("fill") else "None"
        stroke = "None" if msg.get("fill") else "ForegroundColor"
        node.paintPolygon(pts, fill, stroke)
        return {"seq": msg.get("seq"), "ok": True}

    def _op_fill(self, msg):
        node = self._node()
        if node is None or self.doc is None:
            return {"seq": msg.get("seq"), "ok": False, "error": "no node"}
        x = int(msg.get("x", 0))
        y = int(msg.get("y", 0))
        w, h = self.doc.width(), self.doc.height()
        data = bytes(node.pixelData(0, 0, w, h))
        px = bytearray(data)
        i0 = (y * w + x) * 4
        target = bytes(px[i0:i0 + 3])
        new = bytes(px[i0:i0 + 4])
        # Current view color as the fill value.
        view = self._view()
        if view is not None:
            mc = view.foreGroundColor()
            c = mc.components()
            if len(c) >= 3:
                new = bytes(
                    [int(c[2] * 255), int(c[1] * 255),
                     int(c[0] * 255), 255]
                )
        tol = 32
        seen = bytearray(w * h)
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            idx = cy * w + cx
            if seen[idx]:
                continue
            seen[idx] = 1
            o = idx * 4
            if all(abs(px[o + c] - target[c]) <= tol for c in range(3)):
                px[o:o + 4] = new
                stack.extend(
                    ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1))
                )
        node.setPixelData(bytes(px), 0, 0, w, h)
        self.doc.refreshProjection()
        return {"seq": msg.get("seq"), "ok": True}

    def _op_clear(self):
        if self.paint_node is None or self.doc is None:
            return {"seq": 0, "ok": False, "error": "no node"}
        w, h = self.doc.width(), self.doc.height()
        self.paint_node.setPixelData(bytes(w * h * 4), 0, 0, w, h)
        self.doc.refreshProjection()
        return {"seq": 0, "ok": True}

    def _op_export(self, msg):
        if self.doc is None:
            return {"seq": msg.get("seq"), "ok": False, "error": "no doc"}
        path = str(msg.get("path"))
        ok = bool(self.doc.exportImage(path, InfoObject()))
        return {"seq": msg.get("seq"), "ok": ok, "path": path}
'''


def _pykrita_dir() -> Path:
    return Path.home() / ".var/app/org.kde.krita/data/krita/pykrita"


def _studio_dir() -> Path:
    return Path.home() / ".var/app/org.kde.krita/data/genesis_studio"


class KritaStudio:
    """Client for the Genesis Studio extension inside Krita.

    The connection is a command queue on disk. ``connect`` reports
    whether a live studio answered recently; ``send`` queues one op;
    ``export`` asks the studio to write out the working document.
    """

    def __init__(self) -> None:
        self._queue = _studio_dir() / "queue"
        self._heartbeat = _studio_dir() / "heartbeat"
        self._seq = 0
        self.alive = False

    # ── Install / enable ──────────────────────────────────────────

    def ensure_plugin(self) -> bool:
        """Install the extension into Krita's pykrita dir + enable it."""
        pykrita = _pykrita_dir()
        if not pykrita.parent.exists():
            return False
        pkg = pykrita / _PLUGIN_ID
        try:
            pkg.mkdir(parents=True, exist_ok=True)
            (pkg / "__init__.py").write_text(_PLUGIN_INIT)
            (pkg / "studio.py").write_text(_PLUGIN_STUDIO)
            (pykrita / f"{_PLUGIN_ID}.desktop").write_text(_PLUGIN_DESKTOP)
        except OSError as e:
            logger.debug("krita plugin install failed: %s", e)
            return False
        self._enable_in_kritarc()
        return True

    @staticmethod
    def _enable_in_kritarc() -> None:
        kritarc = Path.home() / ".var/app/org.kde.krita/config/kritarc"
        entry = f"enable_{_PLUGIN_ID}=true"
        try:
            text = kritarc.read_text() if kritarc.exists() else ""
        except OSError:
            return
        if entry in text:
            return
        try:
            if "[python]" in text:
                text = text.replace("[python]", "[python]\n" + entry, 1)
            else:
                text = text.rstrip() + "\n\n[python]\n" + entry + "\n"
            kritarc.write_text(text)
        except OSError as e:
            logger.debug("kritarc enable failed: %s", e)

    # ── Connection ────────────────────────────────────────────────

    def _heartbeat_fresh(self, max_age: float = 4.0) -> bool:
        try:
            return time.time() - self._heartbeat.stat().st_mtime < max_age
        except OSError:
            return False

    def connect(self, timeout: float = 2.0, launch: bool = False) -> bool:
        """True when a live studio answered inside ``timeout``."""
        self._queue.mkdir(parents=True, exist_ok=True)
        if not self._heartbeat_fresh():
            if not launch:
                return False
            if not self.ensure_plugin():
                return False
            self._launch()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._heartbeat_fresh():
                self.alive = True
                return True
            time.sleep(0.25)
        return self._heartbeat_fresh()

    def _launch(self) -> None:
        """Start Krita via flatpak, detached."""
        try:
            subprocess.Popen(
                ["flatpak", "run", "org.kde.krita"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, shutil.Error) as e:
            logger.debug("krita launch failed: %s", e)

    # ── Ops ───────────────────────────────────────────────────────

    def send(self, op: str, **args) -> bool:
        """Queue one op; True when the queue accepted it."""
        if not self._heartbeat_fresh():
            self.alive = False
            return False
        self._seq += 1
        msg = {"seq": self._seq, "op": op}
        msg.update(args)
        try:
            path = self._queue / f"op_{self._seq:06d}.json"
            path.write_text(json.dumps(msg))
            return True
        except OSError:
            return False

    def export(self, path: str, timeout: float = 10.0) -> bool:
        """Ask the studio to write the document out to ``path``."""
        seq_before = self._seq
        if not self.send("export", path=path):
            return False
        ack = self._queue / f"ack_{self._seq:06d}.json"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if ack.exists():
                try:
                    data = json.loads(ack.read_text())
                    return bool(data.get("ok"))
                except (OSError, json.JSONDecodeError):
                    return False
            time.sleep(0.15)
        _ = seq_before
        return False
