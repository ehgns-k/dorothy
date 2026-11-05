"""
Top-down chessboard vision.

Instead of classifying every square as occupied / empty with a fixed
threshold (fragile under exhibition lighting), we exploit what a fixed
overhead camera gives us for free:

  1. After every ply we store a REFERENCE image of the rectified board.
  2. We wait until the scene is STILL (no hand / arm in view).
  3. We measure how much every square CHANGED versus the reference
     (colour distance in Lab, after compensating global lighting drift and
     tiny camera nudges).
  4. Every LEGAL move predicts exactly which squares must change (from/to,
     plus rook squares for castling, plus the captured pawn for en passant).
     The move whose prediction best explains the observed change wins.

This sees captures (the destination changes colour/shape), needs no
per-lighting threshold tuning, and lets us verify the robot's own moves.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import cv2
import numpy as np

import config

log = logging.getLogger("vision")

S = config.WARP_SQUARE_PX
BOARD_PX = 8 * S
PAD = S  # margin around the board: approaching hands, pieces leaning outward


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
class Camera:
    """Background-grabbing UVC camera that always serves the newest frame
    and reconnects by itself if the device drops out."""

    def __init__(self, source):
        self.source = source
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._stop = threading.Event()
        self.error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")

    def start(self) -> "Camera":
        self._open()
        self._thread.start()
        deadline = time.time() + 5
        while self._frame is None and time.time() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            raise RuntimeError(f"camera {self.source!r} delivers no frames")
        return self

    def _open(self) -> None:
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.source!r}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not config.CAMERA_AUTOFOCUS:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cap = cap

    def _run(self) -> None:
        fails = 0
        while not self._stop.is_set():
            ok, frame = (self._cap.read() if self._cap is not None else (False, None))
            if ok and frame is not None:
                fails = 0
                self.error = None
                with self._lock:
                    self._frame, self._t = frame, time.time()
                    self._seq += 1
                continue
            fails += 1
            if fails >= 10:
                self.error = "camera disconnected"
                log.warning("camera read failing - reconnecting")
                try:
                    if self._cap is not None:
                        self._cap.release()
                    time.sleep(config.CAMERA_RECONNECT_S)
                    self._open()
                    fails = 0
                except Exception as e:
                    log.warning("camera reconnect failed: %s", e)
                    self._cap = None
            else:
                time.sleep(0.02)

    def read(self):
        """(frame_bgr, timestamp, sequence_number) - frame may be None."""
        with self._lock:
            return self._frame, self._t, self._seq

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


# --------------------------------------------------------------------------- #
# Board geometry
# --------------------------------------------------------------------------- #
class BoardGeometry:
    """Outer board corners in image pixels, in the order a1, h1, h8, a8.

    `home` is where the board was when the ARM was calibrated (relative to
    the camera). If the board itself slides away from home the arm will miss
    squares; if only the camera moved, home moves with it."""

    def __init__(self, corners: Sequence[Sequence[float]], home=None, lean_k: float = 0.0):
        self.lean_k = float(lean_k or 0.0)   # how far pieces appear to lean (see fit_lean)
        self.corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        self.home = self.corners.copy() if home is None else \
            np.asarray(home, dtype=np.float32).reshape(4, 2)
        a1, h1, h8, a8 = self.corners
        src = np.float32([a8, h8, h1, a1])
        dst = np.float32([[PAD, PAD], [PAD + BOARD_PX, PAD],
                          [PAD + BOARD_PX, PAD + BOARD_PX], [PAD, PAD + BOARD_PX]])
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def load(cls, path=config.BOARD_FILE) -> Optional["BoardGeometry"]:
        try:
            d = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        return cls(d["corners"], d.get("home"), d.get("lean_k", 0.0))

    def save(self, path=config.BOARD_FILE) -> None:
        path.write_text(json.dumps({"corners": self.corners.tolist(),
                                    "home": self.home.tolist(),
                                    "order": "a1,h1,h8,a8",
                                    "lean_k": self.lean_k,
                                    "saved": time.strftime("%Y-%m-%d %H:%M:%S")},
                                   indent=2))

    def moved(self, corners) -> "BoardGeometry":
        """Same home, new current position."""
        return BoardGeometry(corners, self.home, self.lean_k)

    def home_offset(self) -> float:
        return float(np.linalg.norm(self.corners - self.home, axis=1).max())


    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Rectified board (a8 top-left) with a PAD margin around it."""
        n = BOARD_PX + 2 * PAD
        return cv2.warpPerspective(frame, self.H, (n, n), flags=cv2.INTER_LINEAR)

    def square_polygon(self, sq: int) -> np.ndarray:
        """The square's outline in image pixels."""
        c, r = chess.square_file(sq), 7 - chess.square_rank(sq)
        x0, y0 = PAD + c * S, PAD + r * S
        pts = np.float32([[x0, y0], [x0 + S, y0], [x0 + S, y0 + S], [x0, y0 + S]])
        return cv2.perspectiveTransform(pts[None], self.H_inv)[0]


HOME_BG_FILE = config.BOARD_FILE.with_name("board_home.png")


def _bg_mask(shape, polys, grow: float = 1.25) -> np.ndarray:
    """255 on the background, 0 on (and right around) the board."""
    mask = np.full(shape[:2], 255, np.uint8)
    for p in polys:
        c = np.asarray(p, np.float32)
        ctr = c.mean(axis=0)
        cv2.fillPoly(mask, [(ctr + (c - ctr) * grow).astype(np.int32)], 0)
    return mask


def save_home_background(frame: np.ndarray, geo: BoardGeometry) -> None:
    cv2.imwrite(str(HOME_BG_FILE), frame)


_ORB = None


def camera_shift(frame: np.ndarray, geo: BoardGeometry):
    """Translation of the BACKGROUND (table, robot base) vs. the home frame
    in px, or None if it can't be measured. Background moved <=> camera moved.
    ORB features + consensus translation: robust to people walking past."""
    global _ORB
    home = cv2.imread(str(HOME_BG_FILE))
    if home is None or home.shape != frame.shape:
        return None
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
    mask = _bg_mask(frame.shape, [geo.home, geo.corners])
    ka, da = _ORB.detectAndCompute(cv2.cvtColor(home, cv2.COLOR_BGR2GRAY), mask)
    kb, db = _ORB.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 12:
        return None
    d = np.float32([np.subtract(kb[m.trainIdx].pt, ka[m.queryIdx].pt) for m in matches])
    # consensus: the translation agreed on by the most matches
    best, best_n = None, 0
    for cand in d[np.random.default_rng(0).choice(len(d), min(len(d), 200), replace=False)]:
        n = int((np.linalg.norm(d - cand, axis=1) < 2.5).sum())
        if n > best_n:
            best, best_n = cand, n
    if best_n < max(10, 0.2 * len(matches)):
        return None
    inl = d[np.linalg.norm(d - best, axis=1) < 2.5]
    return np.median(inl, axis=0).astype(np.float32)


def auto_find_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """Find the outer corners of an EMPTY board from its 7x7 inner corners.
    Returns 4 image points in an arbitrary rotation (caller fixes a1)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok, pts = cv2.findChessboardCornersSB(
        gray, (7, 7), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        return None
    pts = pts.reshape(7, 7, 2)
    ideal = np.float32([[i + 1, j + 1] for j in range(7) for i in range(7)])
    H, _ = cv2.findHomography(ideal, pts.reshape(-1, 2).astype(np.float32))
    ox, oy = grid_offset(frame, H)
    outer = np.float32([[0, 8], [8, 8], [8, 0], [0, 0]]) + np.float32([ox, oy])
    return cv2.perspectiveTransform(outer[None], H)[0]


GRID_CELL_PX = 24


def _grid_patches(frame: np.ndarray, H: np.ndarray, lo: int, hi: int):
    """Lab colour near the four corners of every grid cell in [lo, hi)^2
    (grid units; the detected board is [0, 8)). A piece stands in the middle
    of its square, so the corners show the square's own colour even on a
    crowded rank. Returns (k, k, 4, 3) medians and a (k, k) in-frame mask."""
    px = GRID_CELL_PX
    n = (hi - lo) * px
    M = H @ np.array([[1.0 / px, 0, lo], [0, 1.0 / px, lo], [0, 0, 1]])
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    w = cv2.warpPerspective(lab, M, (n, n), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
    inside = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), M, (n, n),
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    k = hi - lo
    m, q = px // 8, px // 4                       # skip the grid lines
    cells = w.reshape(k, px, k, px, 3).astype(np.float32)
    patches = [cells[:, r0:r0 + q, :, c0:c0 + q] for r0 in (m, px - m - q)
               for c0 in (m, px - m - q)]
    med = np.stack([np.median(p.transpose(0, 2, 1, 3, 4).reshape(k, k, -1, 3), axis=2)
                    for p in patches], axis=2)
    valid = inside.reshape(k, px, k, px).min(axis=(1, 3)) > 0
    return med, valid


def grid_offset(frame: np.ndarray, H: np.ndarray, search: int = 1):
    """Which 8x8 block of the detected grid is the chessboard?

    The corner detector only sees a 7x7 lattice of corners. When pieces hide
    one edge row of corners it can latch onto a lattice one row/column off,
    using the board's printed border as the missing row - the board then
    comes out shifted by a whole square (one rank cut off, one row of
    margin added). Same problem as chesscog's "inferring missing lines"
    step (Wölflein & Arandjelović 2021, sec. 4.1.3); here it is solved with
    the checker pattern itself:

    * the light and dark square colours are learned from the middle of the
      detected grid (board for sure, whichever way it is off)
    * every corner patch of every cell is classified light / dark / neither
      (pieces, table clutter -> neither, so they can't tip the balance)
    * an 8x8 window scores +1 for each patch with the colour its square
      should have and -1 for the opposite colour; a plain margin cancels
      out, and a window that has board squares just outside it (= is one
      square off) is penalised for them.

    Returns the (dx, dy) grid shift, in squares, of the best window."""
    lo, hi = -search - 1, 9 + search       # + the row just outside every window
    try:
        med, valid = _grid_patches(frame, H, lo, hi)
    except cv2.error:
        return 0, 0
    k = hi - lo
    par = np.add.outer(np.arange(k) + lo, np.arange(k) + lo) % 2     # 0/1 per cell
    core = np.zeros((k, k), bool)
    core[2 - lo:6 - lo, 2 - lo:6 - lo] = True
    core &= valid
    if core.sum() < 8:
        return 0, 0
    c0 = np.median(med[core & (par == 0)].reshape(-1, 3), axis=0)
    c1 = np.median(med[core & (par == 1)].reshape(-1, 3), axis=0)
    gap = float(np.linalg.norm(c0 - c1))
    if gap < 15:                               # no clear checker pattern
        return 0, 0
    d0 = np.linalg.norm(med - c0, axis=-1)
    d1 = np.linalg.norm(med - c1, axis=-1)
    near = 0.45 * gap
    vote = (d0 < near).astype(np.float32) - (d1 < near)              # +1 colour 0, -1 colour 1
    sig = np.where(valid, vote.sum(axis=2) * np.where(par == 0, 1.0, -1.0), 0.0)
    best, best_s, scores = (0, 0), -np.inf, {}
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            y0, x0 = dy - lo, dx - lo
            ys, xs = slice(y0, y0 + 8), slice(x0, x0 + 8)
            s = float(sig[ys, xs].sum())
            # just outside the real board there is no checker pattern
            s -= sum(max(0.0, float(e.sum())) for e in (sig[y0 - 1, xs], sig[y0 + 8, xs],
                                                        sig[ys, x0 - 1], sig[ys, x0 + 8]))
            scores[dx, dy] = s
            if s > best_s:
                best, best_s = (dx, dy), s
    if best != (0, 0):
        log.info("board detector was off by %s squares - corrected (scores %s)", best,
                 {k: round(v) for k, v in scores.items()})
    return best


# --------------------------------------------------------------------------- #
# Change analysis
# --------------------------------------------------------------------------- #
def _center_mask(frac: float = config.SQUARE_INNER_FRAC, sigma_frac: float = None) -> np.ndarray:
    ax = (np.arange(S) + 0.5) / S - 0.5
    xx, yy = np.meshgrid(ax, ax)
    sigma = 0.5 * frac / 2.0 if sigma_frac is None else sigma_frac
    m = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    m[(np.abs(xx) > frac / 2) | (np.abs(yy) > frac / 2)] = 0
    return (m / m.sum()).astype(np.float32)


_MASK = _center_mask()
_MASK_TOP = _center_mask(0.5, 0.1)          # small spot: the top of a piece
_INNER = _center_mask(0.7, 10.0) > 0        # flat inner region for percentiles
# row-major (row 0 = rank 8) grid position -> chess square index
_GRID_TO_SQ = np.array([chess.square(c, 7 - r) for r in range(8) for c in range(8)])
_NUDGE = int(config.NUDGE_TOLERANCE_PX)
_SHIFTS = (-_NUDGE, -_NUDGE // 2, 0, _NUDGE // 2, _NUDGE)
PADDED_PX = BOARD_PX + 2 * PAD
MAX_LEAN_PX = 0.45 * S


def _lab(board_bgr: np.ndarray) -> np.ndarray:
    f = cv2.GaussianBlur(board_bgr, (5, 5), 1.2).astype(np.float32) / 255.0
    return cv2.cvtColor(f, cv2.COLOR_BGR2Lab)


def _per_square(img: np.ndarray) -> np.ndarray:
    """(8S, 8S) -> 64 weighted means, chess.SQUARES order."""
    g = img.reshape(8, S, 8, S).transpose(0, 2, 1, 3)
    vals = (g * _MASK).sum(axis=(2, 3)).reshape(64)
    out = np.empty(64, np.float32)
    out[_GRID_TO_SQ] = vals
    return out


def crop_board(padded: np.ndarray) -> np.ndarray:
    return padded[PAD:PAD + BOARD_PX, PAD:PAD + BOARD_PX]


def _padded(img: np.ndarray) -> np.ndarray:
    """Accept a padded rectified board, or a bare (cropped) one."""
    if img.shape[0] == PADDED_PX:
        return img
    return cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)


_BASE_CENTRES = np.float32([[PAD + (chess.square_file(s) + 0.5) * S,
                             PAD + (7 - chess.square_rank(s) + 0.5) * S] for s in chess.SQUARES])


def square_centres(lean: Optional[np.ndarray] = None) -> np.ndarray:
    """Where each square's piece is SEEN in the padded rectified image
    (x, y; chess.SQUARES order). A top-down camera sees tall pieces leaning
    away from the point straight below it, so on outer squares the piece
    appears shifted outward - sample it there, not at the square's centre."""
    c = _BASE_CENTRES if lean is None else _BASE_CENTRES + lean
    lo, hi = S // 2 + _NUDGE, PADDED_PX - S // 2 - _NUDGE
    return np.clip(c, lo, hi - 1)


def lean_offsets(geo: "BoardGeometry", k: float, frame_shape=None) -> np.ndarray:
    """Per-square sampling offset for lean factor k (offset = k x distance
    from the camera's nadir, taken as the image centre)."""
    if not k:
        return np.zeros((64, 2), np.float32)
    h, w = (frame_shape or (config.CAMERA_HEIGHT, config.CAMERA_WIDTH))[:2]
    nadir = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), geo.H)[0, 0]
    off = (_BASE_CENTRES - nadir) * k
    n = np.linalg.norm(off, axis=1, keepdims=True)
    return (off * np.minimum(1.0, MAX_LEAN_PX / np.maximum(n, 1e-6))).astype(np.float32)


def _gather(img: np.ndarray, centres: np.ndarray, half: int) -> np.ndarray:
    """(N, N, C) image -> (64, 2*half, 2*half, C) patches around `centres`."""
    c = np.round(centres).astype(np.int32)
    ar = np.arange(-half, half, dtype=np.int32)
    ys = c[:, 1, None] + ar
    xs = c[:, 0, None] + ar
    return img[ys[:, :, None], xs[:, None, :]]


def _align(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Undo a small global shift of `cur` against `ref` (camera nudge)."""
    g_ref = cv2.cvtColor(crop_board(ref), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_cur = cv2.cvtColor(crop_board(cur), cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(g_ref.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(g_ref, g_cur, win)
    if abs(dx) <= config.MAX_ALIGN_SHIFT_PX and abs(dy) <= config.MAX_ALIGN_SHIFT_PX \
            and (abs(dx) > 0.3 or abs(dy) > 0.3):
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        cur = cv2.warpAffine(cur, M, cur.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    return cur


def square_features(lab: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Per-square appearance (64 x 6): mean L, a, b around the piece, L at the
    piece's top, and the 10th / 90th percentile of L (dark / bright parts)."""
    p = _gather(lab, centres, S // 2)                         # (64, S, S, 3)
    mean = (p * _MASK[None, :, :, None]).sum(axis=(1, 2))      # (64, 3)
    top = (p[..., 0] * _MASK_TOP).sum(axis=(1, 2))
    inner = p[..., 0][:, _INNER]                               # (64, n)
    lo, hi = np.percentile(inner, [10, 90], axis=1)
    return np.column_stack([mean, top, lo, hi]).astype(np.float32)


@dataclass
class Analysis:
    scores: np.ndarray                 # per-square change vs. the reference
    features: np.ndarray               # per-square appearance now (64 x 6)
    l_shift: float                     # global lightness drift vs. the reference


def analyze(ref: np.ndarray, cur: np.ndarray, centres: Optional[np.ndarray] = None) -> Analysis:
    """Change of every square between two rectified boards, plus what each
    square looks like now."""
    ref, cur = _padded(ref), _padded(cur)
    if centres is None:
        centres = square_centres()
    cur = _align(ref, cur)
    a, b = _lab(ref), _lab(cur)
    half = S // 2
    pa = _gather(a, centres, half + _NUDGE)                   # (64, S+2n, S+2n, 3)
    pb = _gather(b, centres, half)                            # (64, S, S, 3)
    # global lighting drift: most squares are unchanged -> median shift
    n = _NUDGE
    dl = ((pb[..., 0] - pa[:, n:n + S, n:n + S, 0]) * _MASK).sum(axis=(1, 2))
    l_shift = float(np.median(dl))
    pb = pb.copy()
    pb[..., 0] -= l_shift
    # Shift tolerance: a piece that was merely nudged still matches itself
    # after a small local shift; an emptied / newly filled square does not.
    best = None
    w = config.LIGHTNESS_WEIGHT
    for dy in _SHIFTS:
        for dx in _SHIFTS:
            d = pa[:, n + dy:n + dy + S, n + dx:n + dx + S] - pb
            dist = np.sqrt((w * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
            sc = (dist * _MASK).sum(axis=(1, 2))
            best = sc if best is None else np.minimum(best, sc)
    feats = square_features(b, centres)
    return Analysis(best.astype(np.float32), feats, l_shift)


def square_changes(ref_board: np.ndarray, cur_board: np.ndarray,
                   centres: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-square change score (chess.SQUARES order) between two rectified
    board images (padded or bare, BGR)."""
    return analyze(ref_board, cur_board, centres).scores


def change_threshold(scores: np.ndarray) -> float:
    return float(max(config.CHANGE_THRESHOLD,
                     config.CHANGE_NOISE_MULT * np.median(scores)))


# --------------------------------------------------------------------------- #
# Piece colour: empty / white / black per square
# --------------------------------------------------------------------------- #
EMPTY, WHITE, BLACK = 0, 1, 2
STATE_NAMES = ("empty", "white", "black")
_PARITY = np.array([(chess.square_file(s) + chess.square_rank(s)) % 2 for s in chess.SQUARES])
# minimum spread per feature (Lab units) so a model learned from few, very
# similar samples is not over-confident
_FEAT_FLOOR = np.float32([3.0, 2.0, 2.0, 4.0, 4.0, 4.0])


def square_state(board: chess.Board, sq: int) -> int:
    p = board.piece_at(sq)
    return EMPTY if p is None else WHITE if p.color == chess.WHITE else BLACK


def board_states(board: chess.Board) -> np.ndarray:
    return np.array([square_state(board, s) for s in chess.SQUARES])


class ColourModel:
    """What an empty square, a white piece and a black piece look like, per
    square colour (light / dark). Learned from images whose position is KNOWN
    (the start position and every confirmed position since), so it adapts to
    the exhibition's lighting and to the actual piece set - cream pieces on a
    white square differ mostly in colour, not brightness."""

    KEEP = 6                       # recent positions kept (plus the start position)

    def __init__(self):
        self.start: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.recent: List[Tuple[np.ndarray, np.ndarray]] = []
        self.mean = None           # (2 parities, 3 states, 6)
        self.var = None

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def reset(self) -> None:
        self.__init__()

    def add(self, feats: np.ndarray, states: np.ndarray, start: bool = False) -> None:
        if start:
            self.start = (feats.copy(), states.copy())
            self.recent = []
        else:
            self.recent = (self.recent + [(feats.copy(), states.copy())])[-self.KEEP:]
        self._fit()

    def _fit(self) -> None:
        data = ([self.start] if self.start else []) + self.recent
        if not data:
            return
        F = np.concatenate([d[0] for d in data])
        St = np.concatenate([d[1] for d in data])
        P = np.tile(_PARITY, len(data))
        mean = np.full((2, 3, F.shape[1]), np.nan, np.float32)
        var = np.full_like(mean, np.nan)
        for par in (0, 1):
            for st in (EMPTY, WHITE, BLACK):
                x = F[(P == par) & (St == st)]
                if len(x) >= 2:
                    mean[par, st] = np.median(x, axis=0)
                    mad = 1.4826 * np.median(np.abs(x - mean[par, st]), axis=0)
                    var[par, st] = (np.maximum(mad, _FEAT_FLOOR) * 1.5) ** 2
        # a piece colour never seen on one square colour: borrow the other
        for st in (EMPTY, WHITE, BLACK):
            for par in (0, 1):
                if np.isnan(mean[par, st, 0]) and not np.isnan(mean[1 - par, st, 0]):
                    mean[par, st] = mean[1 - par, st]
                    var[par, st] = var[1 - par, st] * 2.0
        if np.isnan(mean).any():
            return                 # not enough data yet (keep the old model)
        self.mean, self.var = mean, var

    def posterior(self, feats: np.ndarray, l_shift: float = 0.0) -> Optional[np.ndarray]:
        """P(empty / white / black) per square (64 x 3), or None if untrained."""
        if not self.ready:
            return None
        f = feats.copy()
        f[:, [0, 3, 4, 5]] -= l_shift          # lighting drift since the reference
        m = self.mean[_PARITY]                  # (64, 3, 6)
        v = self.var[_PARITY]
        z2 = (f[:, None, :] - m) ** 2 / v
        ll = -0.5 * np.minimum(z2, 16.0).sum(axis=2) - 0.5 * np.log(v).sum(axis=2)
        ll = ll / 2.0                           # features are correlated
        ll -= ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        p = p / p.sum(axis=1, keepdims=True)
        return np.clip(p, 1e-4, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Move inference
# --------------------------------------------------------------------------- #
def changed_squares(board: chess.Board, move: chess.Move) -> frozenset:
    """Squares whose physical content differs after `move` (pre-move board)."""
    sqs = {move.from_square, move.to_square}
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if board.is_kingside_castling(move):
            sqs = {chess.square(4, rank), chess.square(6, rank),
                   chess.square(7, rank), chess.square(5, rank)}
        else:
            sqs = {chess.square(4, rank), chess.square(2, rank),
                   chess.square(0, rank), chess.square(3, rank)}
    elif board.is_en_passant(move):
        sqs.add(chess.square(chess.square_file(move.to_square),
                             chess.square_rank(move.from_square)))
    return frozenset(sqs)


_NEIGH = [frozenset(s2 for s2 in chess.SQUARES if chess.square_distance(s, s2) == 1)
          for s in chess.SQUARES]
BLEED_DISCOUNT = 0.6   # tall pieces lean into neighbouring squares
KEEP_DISCOUNT = 0.7    # changed square whose colour still matches -> probably a nudge
COLOUR_WEIGHT = 1.0    # colour evidence vs. change evidence in the ranking
AMBIG_MARGIN = 4.0     # score lead that settles two otherwise-possible moves
INSIDE_FRAC = 0.85     # a must-change square may be this much under the threshold


def explain(board: chess.Board, line: Sequence[chess.Move]):
    """Physical effect of playing `line` from `board`.
    Returns (must, maybe, end_board): `must` squares change between empty /
    white / black; `maybe` squares were touched but end in the same colour
    state (e.g. a piece captured and recaptured on the same square)."""
    b = board.copy(stack=False)
    touched = set()
    for mv in line:
        touched |= changed_squares(b, mv)
        b.push(mv)
    must = frozenset(s for s in touched if square_state(board, s) != square_state(b, s))
    return must, frozenset(touched) - must, b


@dataclass
class Hypothesis:
    line: Tuple[chess.Move, ...]
    must: frozenset
    maybe: frozenset
    accepted: bool
    total: float          # change + colour evidence (higher = better)
    colour: float         # colour evidence alone
    margin: float
    worst_outside: Optional[int]
    after: Optional[Tuple] = None   # colour states of must|maybe after the line


@dataclass
class Inference:
    status: str                          # "still" | "move" | "unclear" | "ambiguous"
    moves: List[chess.Move] = field(default_factory=list)   # best first (last move of each line)
    changed: List[int] = field(default_factory=list)        # squares over threshold
    scores: Optional[np.ndarray] = None
    threshold: float = 0.0
    confidence: float = 0.0
    guesses: List[chess.Move] = field(default_factory=list)  # best-explaining legal moves
    lines: List[Tuple[chess.Move, ...]] = field(default_factory=list)
    posterior: Optional[np.ndarray] = None
    ranked: List[Hypothesis] = field(default_factory=list)
    note: str = ""

    def summary(self, board: Optional[chess.Board] = None) -> dict:
        """JSON-friendly record for the debug log."""
        def line_str(ln):
            return " ".join(m.uci() for m in ln)
        out = {"status": self.status, "threshold": round(self.threshold, 2),
               "confidence": round(self.confidence, 3), "note": self.note,
               "changed": {chess.square_name(s): round(float(self.scores[s]), 1)
                           for s in self.changed} if self.scores is not None else {},
               "lines": [line_str(ln) for ln in self.lines[:6]],
               "top": [{"line": line_str(h.line), "ok": h.accepted, "total": round(h.total, 2),
                        "colour": round(h.colour, 2), "margin": round(h.margin, 2),
                        "worst_outside": None if h.worst_outside is None
                        else chess.square_name(h.worst_outside)}
                       for h in self.ranked[:8]]}
        if self.scores is not None:
            out["scores"] = [round(float(x), 1) for x in self.scores]
        if self.posterior is not None:
            out["colour"] = "".join("ewb"[int(i)] for i in self.posterior.argmax(axis=1))
            out["colour_conf"] = [round(float(x), 2) for x in self.posterior.max(axis=1)]
        return out


_NEIGH_MASK = np.array([[s2 in _NEIGH[s] for s2 in chess.SQUARES] for s in chess.SQUARES])


def _score(board: chess.Board, line, scores, T, logit, post, before, eff_base) -> Hypothesis:
    must, maybe, end = explain(board, line)
    lik, col, inside_ok = 0.0, 0.0, bool(must)
    for s in must:
        b, a = before[s], square_state(end, s)
        lik += float(logit[s])
        if post is not None:
            col += float(np.clip(np.log(post[s, a]) - np.log(post[s, b]), -4.0, 4.0))
        ok_change = scores[s] >= T * INSIDE_FRAC
        # colour confirms it even though the change score is weak (e.g. a
        # cream piece on a white square seen at an angle)
        ok_colour = post is not None and post[s, a] > 0.8 and post[s, b] < 0.2 \
            and scores[s] >= 0.5 * T
        if not (ok_change or ok_colour):
            inside_ok = False
    for s in maybe:
        if post is not None:
            col += float(min(0.0, np.log(post[s, square_state(end, s)]) - np.log(0.5)))
    region = must | maybe
    idx = list(region)
    near = _NEIGH_MASK[idx].any(axis=0) if idx else np.zeros(64, bool)
    eff = eff_base * np.where(near, BLEED_DISCOUNT, 1.0)
    eff[idx] = -1.0
    worst = int(np.argmax(eff))
    outside = max(0.0, float(eff[worst]))
    accepted = inside_ok and outside < T
    inside_min = min((float(scores[s]) for s in must), default=0.0)
    margin = min(inside_min - T, T - outside)
    after = tuple(sorted((s, square_state(end, s)) for s in region))
    return Hypothesis(tuple(line), must, maybe, accepted, lik + COLOUR_WEIGHT * col, col,
                      margin, worst if outside > 0 else None, after)


def infer_lines(board: chess.Board, lines, scores: np.ndarray,
                post: Optional[np.ndarray] = None) -> Inference:
    """Which of the candidate move sequences (`lines`, from `board`) best
    explains the observed change? Uses where things changed AND, when the
    colour model is trained, whether each square now holds the expected
    empty / white / black."""
    T = change_threshold(scores)
    w = T / config.CHANGE_LOGIT_SCALE
    logit = np.clip((scores - T) / w, -4.0, 4.0)
    changed = [int(s) for s in np.argsort(-scores) if scores[s] >= T]
    if not changed:
        return Inference("still", [], [], scores, T, posterior=post)

    before = board_states(board)
    # a changed square whose colour still matches what was there: probably nudged
    eff_base = scores.astype(np.float64).copy()
    if post is not None:
        eff_base[post[np.arange(64), before] > 0.85] *= KEEP_DISCOUNT
    ranked = [_score(board, ln, scores, T, logit, post, before, eff_base) for ln in lines]
    ranked.sort(key=lambda h: (h.accepted, h.total), reverse=True)
    guesses = [h.line[-1] for h in ranked[:8]]
    acc = [h for h in ranked if h.accepted]
    if not acc:
        return Inference("unclear", guesses, changed, scores, T, 0.0, guesses,
                         [h.line for h in ranked[:8]], post, ranked)
    best = acc[0]
    same = [h for h in acc if h.after == best.after]      # e.g. promotion choices
    other = [h for h in acc if h.after != best.after]
    conf = float(np.clip(best.margin / T, 0.0, 1.0))
    if other:
        lead = best.total - other[0].total
        if post is None or lead < AMBIG_MARGIN:
            return Inference("ambiguous", [h.line[-1] for h in acc], changed, scores, T, conf,
                             guesses, [h.line for h in acc], post, ranked)
        note = f"colour settled it (lead {lead:.1f})"
    else:
        note = ""
    return Inference("move", [h.line[-1] for h in same], changed, scores, T, conf, guesses,
                     [h.line for h in same], post, ranked, note)


def infer_move(board: chess.Board, scores: np.ndarray,
               post: Optional[np.ndarray] = None) -> Inference:
    return infer_lines(board, [(mv,) for mv in board.legal_moves], scores, post)


def fit_lean(padded: np.ndarray, board: chess.Board, geo: "BoardGeometry",
             frame_shape=None) -> Tuple[float, Dict[float, float]]:
    """Find how far pieces appear to lean outward, from an image whose
    position is known: the lean that best separates occupied from empty
    squares wins (0 = sample at square centres)."""
    lab = _lab(_padded(padded))
    st = board_states(board)
    occ = st != EMPTY
    if occ.all() or not occ.any():
        return 0.0, {}
    res = {}
    for k in np.round(np.arange(0.0, 0.205, 0.02), 3):
        f = square_features(lab, square_centres(lean_offsets(geo, k, frame_shape)))
        d = np.zeros(64)
        for par in (0, 1):
            m = _PARITY == par
            ref = np.median(f[m & ~occ, :3], axis=0)
            dd = f[:, :3] - ref
            d[m] = np.sqrt((config.LIGHTNESS_WEIGHT * dd[m, 0]) ** 2 + dd[m, 1] ** 2 + dd[m, 2] ** 2)
        # Fisher separation, with the WORST occupied squares weighted in:
        # a lean that loses the edge pieces must not win on the average
        a, b = d[occ], d[~occ]
        res[float(k)] = float((np.percentile(a, 10) - np.percentile(b, 90)) / (a.std() + b.std() + 1e-6)
                              + (a.mean() - b.mean()) / (a.std() + b.std() + 1e-6))
    best = max(res, key=res.get)
    return best, res


# --------------------------------------------------------------------------- #
# Vision system: processing thread + reference handling
# --------------------------------------------------------------------------- #
class Vision:
    def __init__(self, camera, geometry: Optional[BoardGeometry]):
        self.camera = camera
        self.lean_k = geometry.lean_k if geometry is not None else 0.0
        self.lean_pending = True     # learn piece lean at the first start position
        self.centres = square_centres()
        self._geometry = None
        self.geometry = geometry
        self.colour = ColourModel()
        self.recorder = None                          # debuglog.Recorder (optional)
        self.reference: Optional[np.ndarray] = None   # rectified board BGR (padded)
        self.reference_board: Optional[chess.Board] = None
        self._ref_feats: Optional[np.ndarray] = None
        # live empty / white / black verdict per square, for the UI overlay
        self.detected: Optional[np.ndarray] = None    # (64, 3) probabilities
        self._detected_t = 0.0
        self.start_reference: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._prev_small = None
        self._last_seq = -1
        self.motion = 1.0
        self.stable_since: Optional[float] = None
        self.last_frame_t = 0.0
        self.board_img: Optional[np.ndarray] = None   # latest rectified board
        self.frame: Optional[np.ndarray] = None
        self.frame_seq = 0
        self.highlight: Dict[int, tuple] = {}          # sq -> BGR for preview
        self.show_home = False                         # draw "push the board here"
        self.show_grid = True                          # off while the board is lost
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="vision")

    def start(self) -> "Vision":
        self._thread.start()
        return self

    @property
    def geometry(self) -> Optional[BoardGeometry]:
        return self._geometry

    @geometry.setter
    def geometry(self, geo: Optional[BoardGeometry]) -> None:
        if geo is not None:
            geo.lean_k = self.lean_k
            frame = getattr(self, "frame", None)
            self.centres = square_centres(lean_offsets(
                geo, self.lean_k, None if frame is None else frame.shape))
        self._geometry = geo

    def stop(self) -> None:
        self._stop.set()

    # ---- background processing ------------------------------------------ #
    def _run(self) -> None:
        period = 1.0 / config.VISION_HZ
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._step()
            except Exception:
                log.exception("vision step failed")
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _step(self) -> None:
        frame, frame_t, seq = self.camera.read()
        if frame is None or seq == self._last_seq:
            return
        self._last_seq = seq
        geo = self.geometry
        if geo is None:
            with self._lock:
                self.frame, self.frame_seq = frame, seq
            if self.recorder is not None:
                self.recorder.feed(frame, frame_t)
            return
        padded = geo.warp(frame)
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(padded, None, fx=0.25, fy=0.25,
                                                         interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2GRAY), (3, 3), 0)
        now = frame_t  # CAPTURE time: a frame grabbed before the arm left must not count
        if self._prev_small is not None and self._prev_small.shape == small.shape:
            moving = np.mean(cv2.absdiff(small, self._prev_small) > config.MOTION_PIXEL_DIFF)
        else:
            moving = 1.0
        self._prev_small = small
        with self._lock:
            self.motion = float(moving)
            self.last_frame_t = now
            if moving > config.MOTION_FRACTION:
                self.stable_since = None
            elif self.stable_since is None:
                self.stable_since = now
            self.board_img = padded
            self.frame, self.frame_seq = frame, seq
        if self.colour.ready and time.time() - self._detected_t >= 1.0 / config.DETECT_OVERLAY_HZ:
            self._detected_t = time.time()
            feats = square_features(_lab(padded), self.centres)
            ref = self._ref_feats
            shift = 0.0 if ref is None else float(np.median(feats[:, 0] - ref[:, 0]))
            self.detected = self.colour.posterior(feats, shift)
        rec = self.recorder
        if rec is not None:
            rec.feed(frame, frame_t)

    # ---- queries -------------------------------------------------------- #
    @property
    def ready(self) -> bool:
        return self.geometry is not None and self.board_img is not None

    def is_stable(self, min_s: float = config.STABLE_TIME_S) -> bool:
        with self._lock:
            return self.stable_since is not None and \
                self.last_frame_t - self.stable_since >= min_s

    def hand_in_view(self) -> bool:
        with self._lock:
            return self.stable_since is None

    def wait_stable(self, timeout: float = 10.0, min_s: float = config.STABLE_TIME_S,
                    cancel=None) -> bool:
        """Wait until the scene has been still for `min_s`, counting only
        frames captured after this call (an arm that just left the view must
        not count as 'still')."""
        t_call = time.time()
        deadline = t_call + timeout
        while time.time() < deadline:
            if cancel is not None and cancel():
                return False
            with self._lock:
                since, last = self.stable_since, self.last_frame_t
            # judged on capture times: only frames taken after this call count
            if since is not None and last - max(since, t_call) >= min_s:
                return True
            time.sleep(0.05)
        return False

    def current_board(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self.board_img is None else self.board_img.copy()

    def set_reference(self, board: Optional[chess.Board] = None, trusted: bool = True) -> None:
        """Remember the board as it looks now. With `board` (the position it
        shows) the colour model also learns from it - unless `trusted` is
        False (a visitor overrode a check, so the image may not match)."""
        img = self.current_board()
        if img is None:
            raise RuntimeError("no camera image")
        with self._lock:
            self.reference = img
            self.reference_board = board.copy(stack=False) if board is not None else None
        feats = square_features(_lab(img), self.centres)
        self._ref_feats = feats
        if board is not None and trusted:
            self.colour.add(feats, board_states(board))
        if self.recorder is not None:
            self.recorder.event("reference", fen=board.fen() if board is not None else None,
                                trusted=trusted, image=self.recorder.image("reference", img),
                                frame=self.recorder.image("reference_frame", self.frame))

    def learn_start_position(self, board: Optional[chess.Board] = None) -> None:
        """The pieces are known to be in `board` (default: the starting
        position): learn how far pieces lean and what empty / white / black
        squares look like."""
        board = board or chess.Board()
        img = self.current_board()
        geo = self.geometry
        if img is None or geo is None:
            return
        k, table = self.lean_k, None
        if self.lean_pending:
            # once per camera calibration (start-up or the Calibrate button),
            # never re-learned behind the operator's back between games
            k, table = fit_lean(img, board, geo,
                                self.frame.shape if self.frame is not None else None)
            if abs(k - self.lean_k) > 1e-6:
                log.info("piece lean: %.2f (was %.2f)", k, self.lean_k)
            self.lean_k = k
            self.lean_pending = False
            self.geometry = geo                  # recompute the sampling points
        feats = square_features(_lab(img), self.centres)
        self.colour.add(feats, board_states(board), start=True)
        if self.recorder is not None:
            self.recorder.event("learn_start", lean_k=k, lean_scores=table,
                                lean_learned=table is not None,
                                colour_ready=self.colour.ready,
                                image=self.recorder.image("start", img))

    def analyze(self) -> Optional[Analysis]:
        cur = self.current_board()
        with self._lock:
            ref = self.reference
        if cur is None or ref is None:
            return None
        return analyze(ref, cur, self.centres)

    def changes(self) -> Optional[np.ndarray]:
        a = self.analyze()
        return None if a is None else a.scores

    def posterior(self, a: Analysis) -> Optional[np.ndarray]:
        return self.colour.posterior(a.features, a.l_shift) if config.USE_COLOUR else None

    def infer(self, board: chess.Board) -> Optional[Inference]:
        a = self.analyze()
        return None if a is None else infer_move(board, a.scores, self.posterior(a))

    def infer_lines(self, board: chess.Board, lines) -> Optional[Inference]:
        a = self.analyze()
        return None if a is None else infer_lines(board, lines, a.scores, self.posterior(a))

    def check_robot_move(self, board: chess.Board, move: chess.Move) -> Optional[Inference]:
        """After the robot played `move` from `board` (the reference shows
        `board`): is that all that happened - or did the visitor already
        answer while the arm was busy? Lines: the robot's move alone, and the
        robot's move followed by every legal reply."""
        after = board.copy(stack=False)
        after.push(move)
        lines = [(move,)] + [(move, r) for r in after.legal_moves]
        return self.infer_lines(board, lines)

    def verify_expected(self, expected: Sequence[int]):
        """After the robot moved: did exactly `expected` squares change?
        Returns (ok, missing_squares, unexpected_squares)."""
        sc = self.changes()
        if sc is None:
            return False, list(expected), []
        T = change_threshold(sc)
        exp = set(expected)
        missing = [s for s in exp if sc[s] < T * 0.8]
        near = set().union(*(_NEIGH[s] for s in exp)) - exp
        extra = [s for s in chess.SQUARES if s not in exp and
                 sc[s] * (BLEED_DISCOUNT if s in near else 1.0) >= T * 1.2]
        return (not missing and not extra), missing, extra

    # ---- board / camera movement --------------------------------------- #
    def alignment_shift(self) -> float:
        """Cheap drift check: how far (px in the rectified view) the board
        appears shifted against the reference image."""
        cur = self.current_board()
        with self._lock:
            ref = self.reference
        if cur is None or ref is None:
            return 0.0
        ref, cur = crop_board(ref), crop_board(cur)
        a = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
        return float(np.hypot(dx, dy))

    def _ecc_relocate(self) -> Optional[np.ndarray]:
        """Track the board with ECC against the reference image (works
        mid-game when pieces hide the grid). Returns new corners or None."""
        cur = self.current_board()
        with self._lock:
            ref = self.reference
        geo = self.geometry
        if cur is None or ref is None or geo is None:
            return None
        ref, cur = crop_board(ref), crop_board(cur)
        k = 0.5
        a = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(ref, None, fx=k, fy=k), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        b = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(cur, None, fx=k, fy=k), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        W = np.eye(3, dtype=np.float32)
        try:
            cc, W = cv2.findTransformECC(a.astype(np.float32), b.astype(np.float32), W,
                                         cv2.MOTION_HOMOGRAPHY,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                                         None, 5)
        except cv2.error:
            return None
        if cc < 0.75:
            return None
        Sc = np.diag([k, k, 1.0]).astype(np.float64)
        Wf = np.linalg.inv(Sc) @ W.astype(np.float64) @ Sc          # full-res crop coords
        T = np.array([[1, 0, -PAD], [0, 1, -PAD], [0, 0, 1]], np.float64)
        M = geo.H_inv @ np.linalg.inv(T) @ Wf @ T                    # padded dst -> image
        dst = np.float32([[PAD, PAD + BOARD_PX], [PAD + BOARD_PX, PAD + BOARD_PX],
                          [PAD + BOARD_PX, PAD], [PAD, PAD]])        # a1, h1, h8, a8
        return cv2.perspectiveTransform(dst[None], M)[0]

    def relocate(self, attempts: int = 3) -> Optional[np.ndarray]:
        """Full re-registration: chessboard detector first, ECC fallback."""
        geo = self.geometry
        raw = find_board(self.camera, attempts=attempts, assume_start=False)
        if raw is not None and geo is not None:
            rots = [np.roll(raw, -k, axis=0) for k in range(4)]
            return min(rots, key=lambda c: float(np.linalg.norm(c - geo.corners, axis=1).sum()))
        return self._ecc_relocate()

    def board_status(self, tol_px: float = config.BOARD_HOME_TOL_PX, attempts: int = 3) -> dict:
        st = self._board_status(tol_px, attempts)
        if self.recorder is not None:
            geo = self.geometry
            self.recorder.event("board_status", result=st["kind"],
                                **{k: v for k, v in st.items() if k != "kind"},
                                corners=None if geo is None else geo.corners.round(1).tolist())
        return st

    def _board_status(self, tol_px: float, attempts: int) -> dict:
        """Where is the board relative to the camera and to the robot?

        kind: "ok"            board at home (geometry refreshed)
              "camera_moved"  camera moved, board did not -> re-registered
              "board_moved"   board slid away from where the arm expects it
              "lost"          cannot find the board at all
        """
        geo = self.geometry
        corners = self.relocate(attempts)
        if corners is None or geo is None:
            return {"kind": "lost"}
        if config.BOARD_FIXED_TO_ROBOT:
            # the board cannot move relative to the arm: any shift is the camera
            moved = float(np.linalg.norm(corners - geo.corners, axis=1).max())
            if moved > config.BOARD_UPDATE_PX:
                # (sub-pixel detector jitter is ignored: re-warping would only
                # add noise against the stored reference image)
                self.geometry = BoardGeometry(corners)
            if moved > 1.5:
                log.info("camera moved: board re-registered (%.1f px)", moved)
                return {"kind": "camera_moved", "offset": moved}
            return {"kind": "ok", "offset": moved}
        self.geometry = geo = geo.moved(corners)
        off = geo.home_offset()
        frame, _, _ = self.camera.read()
        cam = camera_shift(frame, geo) if frame is not None and off > 1.5 else None
        cam_vec = np.zeros(2, np.float32) if cam is None else cam
        # how far the board moved relative to the WORLD (= the robot)
        rel = float(np.linalg.norm(geo.corners - (geo.home + cam_vec), axis=1).max())
        if rel > tol_px:
            return {"kind": "board_moved", "offset": rel,
                    "camera": None if cam is None else cam.tolist()}
        if cam is not None and float(np.linalg.norm(cam)) > 1.5:
            self.geometry = BoardGeometry(geo.corners, geo.home + cam)
            save_home_background(frame, self.geometry)
            log.info("camera moved by %s px - re-registered", cam.round(1).tolist())
            return {"kind": "camera_moved", "offset": rel, "shift": cam.tolist()}
        return {"kind": "ok", "offset": rel}

    def remember_start_position(self) -> None:
        img = self.current_board()
        if img is not None:
            self.start_reference = img

    def start_position_mismatch(self) -> Optional[List[int]]:
        """Squares that look different from the remembered start position
        (or, with a trained colour model, hold the wrong colour)."""
        cur = self.current_board()
        if self.start_reference is None or cur is None:
            return None
        a = analyze(self.start_reference, cur, self.centres)
        T = change_threshold(a.scores) * 1.3
        bad = {int(s) for s in chess.SQUARES if a.scores[s] >= T}
        post = self.posterior(a)
        if post is not None:
            want = board_states(chess.Board())
            bad |= {int(s) for s in chess.SQUARES if post[s, want[s]] < 0.05
                    and a.scores[s] >= 0.6 * T}
        return sorted(bad)

    # ---- preview for the UI -------------------------------------------- #
    def preview(self, size=(480, 360), full: bool = False,
                board: Optional[chess.Board] = None) -> Optional[np.ndarray]:
        """'What the robot sees': the camera frame (whole, or cropped around
        the board) with the grid and highlighted squares projected back. With
        `board` (the game position) it also marks what the colour model
        currently sees on every square - see _draw_pieces."""
        with self._lock:
            frame = self.frame
            highlight = dict(self.highlight)
        if frame is None:
            return None
        img = frame.copy()
        geo = self.geometry
        sc = min(size[0] / img.shape[1], size[1] / img.shape[0]) if full else None
        if geo is not None and self.show_grid:
            overlay = img.copy()
            for sq, col in highlight.items():
                cv2.fillPoly(overlay, [geo.square_polygon(sq).astype(np.int32)], col)
            img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
            if self.show_home:
                cv2.polylines(img, [geo.home.astype(np.int32)], True, (0, 220, 255), 4, cv2.LINE_AA)
                cv2.polylines(img, [geo.corners.astype(np.int32)], True, (60, 60, 255), 3, cv2.LINE_AA)
                for a, b in zip(geo.corners, geo.home):
                    if np.linalg.norm(a - b) > 4:
                        cv2.arrowedLine(img, tuple(a.astype(int)), tuple(b.astype(int)),
                                        (0, 220, 255), 3, cv2.LINE_AA, tipLength=0.3)
            for i in range(9):
                for a, b in (((i, 0), (i, 8)), ((0, i), (8, i))):
                    p = np.float32([[PAD + a[0] * S, PAD + a[1] * S],
                                    [PAD + b[0] * S, PAD + b[1] * S]])
                    q = cv2.perspectiveTransform(p[None], geo.H_inv)[0].astype(int)
                    cv2.line(img, tuple(q[0]), tuple(q[1]), (90, 220, 255), 1, cv2.LINE_AA)
            det = self.detected
            if board is not None and det is not None:
                img = self._draw_pieces(img, geo, det, board, sc or 0.5)
            if not full:
                x, y, w, h = cv2.boundingRect(np.vstack([geo.corners, geo.home]).astype(np.int32))
                m = int(0.12 * max(w, h))
                x0, y0 = max(0, x - m), max(0, y - m)
                x1, y1 = min(img.shape[1], x + w + m), min(img.shape[0], y + h + m)
                img = img[y0:y1, x0:x1]
        th, tw = size[1], size[0]
        sc = min(tw / img.shape[1], th / img.shape[0])
        img = cv2.resize(img, (max(1, int(img.shape[1] * sc)), max(1, int(img.shape[0] * sc))),
                         interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    def _draw_pieces(self, img, geo, det, board: chess.Board, scale: float) -> np.ndarray:
        """Semi-transparent marker where each piece is SEEN (the lean-corrected
        sampling point): a light disc = white piece, a dark disc = black
        piece, nothing = empty. The letter is the piece the game says is on
        that square (the camera only tells the colour). Red ring = the camera
        disagrees with the game: wrong colour, a piece where the square
        should be empty ("?"), or nothing seen where a piece should be."""
        pts = cv2.perspectiveTransform(self.centres[None].astype(np.float32), geo.H_inv)[0]
        sq_px = float(np.linalg.norm(geo.corners[1] - geo.corners[0])) / 8
        r = max(3, int(0.28 * sq_px))
        th = max(1, int(round(1.5 / scale)))          # outline: ~1.5 px once shrunk
        text_th = max(1, r // 8)
        font = r / 22
        overlay = img.copy()
        state = det.argmax(axis=1)
        for s in chess.SQUARES:
            x, y = int(pts[s][0]), int(pts[s][1])
            want = square_state(board, s)
            got = int(state[s])
            bad = got != want
            if got == EMPTY:
                if bad:   # a piece should be here but none is seen
                    cv2.circle(overlay, (x, y), r, (40, 40, 235), th * 2, cv2.LINE_AA)
                continue
            fill, ink = ((235, 240, 245), (40, 40, 40)) if got == WHITE else \
                ((35, 35, 35), (235, 235, 235))
            cv2.circle(overlay, (x, y), r, fill, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), r, (40, 40, 235) if bad else ink, th * (2 if bad else 1),
                       cv2.LINE_AA)
            p = board.piece_at(s)
            txt = p.symbol().upper() if p is not None and not bad else "?"
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font, text_th)
            cv2.putText(overlay, txt, (x - tw // 2, y + tht // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font, ink, text_th, cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.6, img, 0.4, 0)


def orient_corners(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Given the 4 outer corners in any rotation, return them as a1,h1,h8,a8
    by assuming the pieces are in the STARTING position: White (light
    pieces) on ranks 1-2, Black on ranks 7-8, middle empty. Uses the colour
    contrast of each square vs. the empty middle of the board."""
    best, best_score = None, -1e9
    for k in range(4):
        c = np.roll(np.asarray(corners, np.float32), -k, axis=0)
        geo = BoardGeometry(c)
        lab = _lab(crop_board(geo.warp(frame)))
        L = _per_square(lab[..., 0])
        chroma = _per_square(np.hypot(lab[..., 1], lab[..., 2]))
        empty = list(range(16, 48))
        # occupied squares differ from same-coloured empty squares
        dev = np.zeros(64)
        for s in chess.SQUARES:
            light = (chess.square_file(s) + chess.square_rank(s)) % 2
            ref = [e for e in empty if (chess.square_file(e) + chess.square_rank(e)) % 2 == light]
            dev[s] = abs(L[s] - np.median(L[ref])) + abs(chroma[s] - np.median(chroma[ref]))
        occ_score = dev[:16].mean() + dev[48:].mean() - 2 * dev[16:48].mean()
        light_score = L[:16].mean() - L[48:].mean()   # white pieces brighter
        score = occ_score + 0.5 * light_score
        if score > best_score:
            best, best_score = c, score
    return best


def find_board(camera, attempts: int = 30, assume_start: bool = True) -> Optional[np.ndarray]:
    """Detect the board on live frames (retrying while exposure settles).
    Returns corners a1,h1,h8,a8 or None."""
    last_seq = -1
    for _ in range(attempts):
        frame, _, seq = camera.read()
        if frame is None or seq == last_seq:
            time.sleep(0.05)
            continue
        last_seq = seq
        c = auto_find_corners(frame)
        if c is not None:
            return orient_corners(frame, c) if assume_start else c
        time.sleep(0.1)
    return None


def refine_geometry(camera, saved: Optional[BoardGeometry],
                    assume_start: bool) -> Optional[BoardGeometry]:
    """Re-detect the board (the camera may have been bumped or moved).
    Orientation is taken from the saved geometry when there is one (nearest
    rotation), else inferred from the starting position."""
    raw = find_board(camera, attempts=20, assume_start=False)
    if raw is None:
        return saved
    if saved is not None:
        rots = [np.roll(raw, -k, axis=0) for k in range(4)]
        best = min(rots, key=lambda c: float(np.linalg.norm(c - saved.corners, axis=1).sum()))
        moved = float(np.linalg.norm(best - saved.corners, axis=1).max())
        if moved > 2.0:
            log.info("board re-detected; corners moved up to %.1f px", moved)
        return saved.moved(best)
    if not assume_start:
        return None
    frame, _, _ = camera.read()
    return BoardGeometry(orient_corners(frame, raw))
