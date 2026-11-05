"""
Detailed debug recording for the kiosk.

Every kiosk run gets its own folder under logs/debug/<date-time>/:

    events.jsonl   one JSON object per line: every recognition result (change
                   score and colour verdict per square, the candidate moves
                   with their scores), board checks, references, robot moves,
                   prompts shown and buttons pressed - each with a timestamp
    frames/        full camera frames and rectified board images at every
                   event that matters (reference set, move detected, check
                   failed, ...), named <event number>_<tag>.jpg and referenced
                   from events.jsonl
    feed/          the raw camera feed at DEBUG_FEED_FPS (default 1 per second)

Writing happens on a background thread, so the game never waits for the
disk. Old sessions are deleted once everything under logs/debug/ exceeds
DEBUG_LOG_MAX_MB (the current session's feed is thinned first).

Replay a session:  .venv/bin/python debuglog.py logs/debug/<session>
"""

import json
import logging
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config

log = logging.getLogger("debuglog")


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return np.round(v, 3).tolist() if v.dtype.kind == "f" else v.tolist()
    if isinstance(v, (np.floating,)):
        return round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float):
        return round(v, 3)
    if v is None or isinstance(v, (str, int, bool)):
        return v
    return str(v)


class Recorder:
    def __init__(self, root: Path = None, enabled: bool = None):
        self.enabled = config.DEBUG_LOG if enabled is None else enabled
        self.root = Path(root or config.LOG_DIR / "debug")
        self.dir: Optional[Path] = None
        self._q: "queue.Queue" = queue.Queue(maxsize=400)
        self._n = 0
        self._lock = threading.Lock()
        self._last_feed = 0.0
        self._dropped = 0
        if not self.enabled:
            return
        self.dir = self.root / time.strftime("%Y%m%d-%H%M%S")
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "feed").mkdir(exist_ok=True)
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)
        threading.Thread(target=self._writer, daemon=True, name="debuglog").start()
        self._prune()
        log.info("debug recording to %s", self.dir)

    # ---- public API (cheap; safe from any thread) ---------------------- #
    def event(self, kind: str, **data) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._n += 1
            n = self._n
        rec = {"n": n, "t": round(time.time(), 3),
               "time": time.strftime("%H:%M:%S"), "kind": kind, **data}
        self._put(("event", rec))

    def image(self, tag: str, img: Optional[np.ndarray]) -> Optional[str]:
        """Queue an image; returns its file name (relative to the session)."""
        if not self.enabled or img is None:
            return None
        with self._lock:
            self._n += 1
            name = f"frames/{self._n:06d}_{tag}.jpg"
        self._put(("image", name, img.copy(), 92))
        return name

    def feed(self, frame: Optional[np.ndarray], t: float) -> None:
        if not self.enabled or frame is None or config.DEBUG_FEED_FPS <= 0:
            return
        if t - self._last_feed < 1.0 / config.DEBUG_FEED_FPS:
            return
        self._last_feed = t
        name = "feed/" + time.strftime("%H%M%S", time.localtime(t)) + f"_{int(t * 1000) % 1000:03d}.jpg"
        self._put(("image", name, frame.copy(), 80))

    # ---- internals ------------------------------------------------------ #
    def _put(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1   # never block the game for the log

    def _writer(self) -> None:
        last_prune = time.time()
        while True:
            item = self._q.get()
            try:
                if item[0] == "event":
                    if self._dropped:
                        item[1]["dropped_before"] = self._dropped
                        self._dropped = 0
                    self._events.write(json.dumps(_jsonable(item[1]), separators=(",", ":")) + "\n")
                else:
                    _, name, img, q = item
                    cv2.imwrite(str(self.dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, q])
            except Exception as e:  # a full disk must not stop the exhibition
                log.warning("debug log write failed: %s", e)
            if time.time() - last_prune > 300:
                last_prune = time.time()
                self._prune()

    def _prune(self) -> None:
        limit = config.DEBUG_LOG_MAX_MB * 1024 * 1024
        try:
            sessions = sorted(p for p in self.root.iterdir() if p.is_dir())
            sizes = {p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in sessions}
            total = sum(sizes.values())
            for p in sessions:
                if total <= limit:
                    return
                if p == self.dir:
                    # thin this session's feed (oldest first) as a last resort
                    for f in sorted((p / "feed").glob("*.jpg"))[::2]:
                        total -= f.stat().st_size
                        f.unlink()
                    return
                shutil.rmtree(p, ignore_errors=True)
                total -= sizes[p]
                log.info("debug log: removed old session %s", p.name)
        except Exception as e:
            log.warning("debug log pruning failed: %s", e)


_NULL = Recorder(enabled=False)
_current: Recorder = _NULL


def get() -> Recorder:
    return _current


def start() -> Recorder:
    global _current
    if _current is _NULL and config.DEBUG_LOG:
        try:
            _current = Recorder()
        except Exception as e:
            log.warning("debug recording disabled: %s", e)
    return _current


def _replay(session: Path) -> None:
    """Print a readable summary of a recorded session."""
    for line in open(session / "events.jsonl"):
        e = json.loads(line)
        k = e["kind"]
        if k == "inference":
            r = e["result"]
            top = r["top"][0] if r.get("top") else {}
            print(f"{e['time']} {k:10s} {e.get('phase', ''):8s} {r['status']:9s} "
                  f"changed={r.get('changed')} best={top.get('line')} {r.get('note', '')}")
        else:
            rest = {x: y for x, y in e.items() if x not in ("n", "t", "time", "kind")}
            print(f"{e['time']} {k:10s} {json.dumps(rest)[:160]}")


if __name__ == "__main__":
    _replay(Path(sys.argv[1]))
