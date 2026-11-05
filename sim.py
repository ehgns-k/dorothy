"""
Hardware-free simulation: a synthetic overhead camera looking at a
"physical" board that the simulated robot and the visitor (mouse clicks on
the on-screen board) can change - including hands and the arm occluding the
view, lighting drift and sensor noise. Used by `kiosk.py --sim` and the
self-test, so the whole vision -> inference -> game pipeline can be exercised
without the arm or camera.
"""

import random
import threading
import time
from typing import Dict, Optional

import chess
import cv2
import numpy as np

TEX_SQ = 80
TEX_MARGIN = 80           # room around the board for pieces leaning outward
W, H = 1280, 720
# outer corners a1, h1, h8, a8 in the synthetic camera image (mild perspective)
SIM_CORNERS = [[452, 640], [878, 632], [858, 214], [470, 208]]


# style "real": like the exhibition set - white/green vinyl board, cream and
# black pieces with real heights that lean outward from the camera
STYLES = {
    "flat": dict(light=(181, 217, 240), dark=(99, 136, 181),
                 white=((225, 230, 232), (90, 90, 90)), black=((40, 38, 42), (150, 150, 150))),
    "real": dict(light=(236, 240, 240), dark=(95, 150, 20),
                 white=((165, 208, 228), (120, 165, 190)), black=((34, 33, 36), (75, 75, 78))),
}
PIECE_HEIGHT_MM = {chess.PAWN: 48, chess.KNIGHT: 62, chess.BISHOP: 68, chess.ROOK: 52,
                   chess.QUEEN: 82, chess.KING: 92}
SIM_SQUARE_MM = 37.5
SIM_CAMERA_MM = 700.0     # camera height above the board (real setup: ~0.7 m)


class SimWorld:
    def __init__(self, noise: float = 3.0, drift: float = 0.04, seed: Optional[int] = None,
                 style: str = "flat", lean: bool = False, place_jitter: int = 5):
        self.style = STYLES[style]
        self.place_jitter = place_jitter
        self.lean = lean
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.pieces: Dict[int, chess.Piece] = dict(chess.Board().piece_map())
        self.offsets: Dict[int, tuple] = {sq: self._jitter() for sq in self.pieces}
        self.noise = noise
        self.drift = drift
        self._lock = threading.Lock()
        self._tex = None
        self._hand_until = 0.0
        self._hand_sq = 0
        self._arm = False
        self._seq = 0
        self._noise_bank = [self.np_rng.normal(0, noise, (H, W, 3)).astype(np.float32)
                            for _ in range(4)]
        dst = np.float32(SIM_CORNERS)
        src = np.float32([[0, 8 * TEX_SQ], [8 * TEX_SQ] * 2, [8 * TEX_SQ, 0], [0, 0]])
        self._src = src
        self.board_offset = np.float32([0, 0])    # board slid on the table
        self.camera_offset = np.float32([0, 0])   # camera bumped
        self._H = cv2.getPerspectiveTransform(src, dst)
        # camera nadir (image centre) in texture coordinates, for piece lean
        self._nadir = cv2.perspectiveTransform(np.float32([[[W / 2, H / 2]]]),
                                               np.linalg.inv(self._H))[0, 0]
        self._t0 = time.time()
        # static, textured surroundings (table edge, robot base, clutter)
        bg = np.full((H, W, 3), (70, 80, 95), np.uint8)
        r = np.random.default_rng(7)
        for _ in range(40):
            x, y = int(r.integers(0, W)), int(r.integers(0, H))
            col = tuple(int(c) for c in r.integers(30, 200, 3))
            cv2.rectangle(bg, (x, y), (x + int(r.integers(20, 160)), y + int(r.integers(20, 120))), col, -1)
        cv2.rectangle(bg, (960, 300), (1120, 520), (235, 235, 235), -1)   # robot base
        self._bg = bg

    def _jitter(self, amount: Optional[int] = None) -> tuple:
        # a placed piece is never perfectly centred - but it stays put
        a = getattr(self, "place_jitter", 5) if amount is None else amount
        return (self.rng.randint(-a, a), self.rng.randint(-a, a))

    def _place(self, sq: int, p: chess.Piece) -> None:
        self.pieces[sq] = p
        self.offsets[sq] = self._jitter()

    # ---- physical actions --------------------------------------------- #
    def reset(self) -> None:
        with self._lock:
            self.pieces = dict(chess.Board().piece_map())
            self.offsets = {sq: self._jitter() for sq in self.pieces}
            self._tex = None

    def transport(self, src: str, dst: str) -> None:
        with self._lock:
            p = self.pieces.pop(chess.parse_square(src), None)
            if p is not None and dst != "capture":
                self._place(chess.parse_square(dst), p)
            self._tex = None

    def human_move(self, frm: int, to: int, promotion=chess.QUEEN) -> None:
        """Physically move a piece (captures the target, handles castling /
        en passant like a real player would)."""
        with self._lock:
            p = self.pieces.pop(frm, None)
            if p is None:
                return
            if p.piece_type == chess.KING and abs(chess.square_file(frm) - chess.square_file(to)) == 2:
                r = chess.square_rank(frm)
                rf, rt = (7, 5) if chess.square_file(to) == 6 else (0, 3)
                rook = self.pieces.pop(chess.square(rf, r), None)
                if rook:
                    self._place(chess.square(rt, r), rook)
            if p.piece_type == chess.PAWN and chess.square_file(frm) != chess.square_file(to) \
                    and to not in self.pieces:
                self.pieces.pop(chess.square(chess.square_file(to), chess.square_rank(frm)), None)
            if p.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7):
                p = chess.Piece(promotion, p.color)
            self._place(to, p)
            self._tex = None
            self._hand_sq = to
            self._hand_until = time.time() + 0.6

    def replace_piece(self, sq: int, piece: chess.Piece) -> None:
        with self._lock:
            self._place(sq, piece)
            self._tex = None

    def set_arm_visible(self, v: bool) -> None:
        self._arm = v

    # ---- rendering ---------------------------------------------------- #
    def _texture(self) -> np.ndarray:
        if self._tex is not None:
            return self._tex
        st = self.style
        M = TEX_MARGIN
        full = np.zeros((8 * TEX_SQ + 2 * M, 8 * TEX_SQ + 2 * M, 3), np.uint8)
        t = full[M:M + 8 * TEX_SQ, M:M + 8 * TEX_SQ]   # view: board coordinates
        for sq in chess.SQUARES:
            c, r = chess.square_file(sq), 7 - chess.square_rank(sq)
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            t[r * TEX_SQ:(r + 1) * TEX_SQ, c * TEX_SQ:(c + 1) * TEX_SQ] = \
                st["light"] if light else st["dark"]
        t = full
        off = np.float32([M, M])
        # outer pieces lean further: draw from the centre outwards
        order = sorted(self.pieces.items(), key=lambda kv: np.hypot(
            *(self._sq_centre(kv[0]) - self._nadir)))
        for sq, p in order:
            cx, cy = self._sq_centre(sq) + off
            dx, dy = self.offsets.get(sq, (0, 0))
            base = np.float32([cx + dx, cy + dy])
            rad = int(TEX_SQ * (0.26 if p.piece_type == chess.PAWN else 0.34))
            body, ring = st["white"] if p.color else st["black"]
            top = base
            if self.lean:
                h = PIECE_HEIGHT_MM[p.piece_type]
                top = base + (base - off - self._nadir) * h / (SIM_CAMERA_MM - h)
                # the piece's side, from its base up to its (narrower) top
                for i in range(9):
                    f = i / 8
                    q = base + (top - base) * f
                    cv2.circle(t, (int(q[0]), int(q[1])), int(rad * (1 - 0.35 * f)),
                               tuple(int(v * 0.8) for v in body), -1, cv2.LINE_AA)
                rad = int(rad * 0.7)
            tx, ty = int(top[0]), int(top[1])
            cv2.circle(t, (tx, ty), rad, body, -1, cv2.LINE_AA)
            cv2.circle(t, (tx, ty), rad, ring, 3, cv2.LINE_AA)
            cv2.putText(t, p.symbol().upper(), (tx - 10, ty + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, ring, 2, cv2.LINE_AA)
        self._tex = t
        return t

    @staticmethod
    def _tex_mask(tex: np.ndarray) -> np.ndarray:
        m = (tex.max(axis=2) > 0).astype(np.uint8) * 255
        M = TEX_MARGIN
        m[M:-M, M:-M] = 255
        return m

    @staticmethod
    def _sq_centre(sq: int) -> np.ndarray:
        c, r = chess.square_file(sq), 7 - chess.square_rank(sq)
        return np.float32([c * TEX_SQ + TEX_SQ // 2, r * TEX_SQ + TEX_SQ // 2])

    def render(self) -> np.ndarray:
        with self._lock:
            tex = self._texture()
            hand = time.time() < self._hand_until
            hand_sq = self._hand_sq
        frame = self._bg.copy()
        Hb = cv2.getPerspectiveTransform(self._src + TEX_MARGIN,
                                         np.float32(SIM_CORNERS) + self.board_offset)
        warped = cv2.warpPerspective(tex, Hb, (W, H))
        mask = cv2.warpPerspective(self._tex_mask(tex), Hb, (W, H)) > 0
        frame[mask] = warped[mask]
        if hand:
            c, r = chess.square_file(hand_sq), 7 - chess.square_rank(hand_sq)
            p = cv2.perspectiveTransform(np.float32([[[c * TEX_SQ + 40, r * TEX_SQ + 40]]]), self._H)[0, 0]
            cv2.ellipse(frame, (int(p[0]), int(p[1]) + 90), (70, 140), 10, 0, 360, (120, 160, 210), -1)
        if self._arm:
            cv2.rectangle(frame, (600, 0), (720, 420), (60, 60, 60), -1)
        if self.camera_offset.any():
            M = np.float32([[1, 0, self.camera_offset[0]], [0, 1, self.camera_offset[1]]])
            frame = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        t = time.time() - self._t0
        gain = 1.0 + self.drift * np.sin(t / 7.0)
        self._seq += 1
        out = frame.astype(np.float32) * gain + self._noise_bank[self._seq % 4]
        return np.clip(out, 0, 255).astype(np.uint8)


class SimCamera:
    """Same interface as vision.Camera."""

    def __init__(self, world: SimWorld, fps: float = 15.0):
        self.world = world
        self.fps = fps
        self.error = None
        self._lock = threading.Lock()
        self._frame, self._t, self._seq = None, 0.0, 0
        self._stop = threading.Event()

    def start(self) -> "SimCamera":
        threading.Thread(target=self._run, daemon=True, name="simcam").start()
        while self._frame is None:
            time.sleep(0.01)
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            f = self.world.render()
            with self._lock:
                self._frame, self._t = f, time.time()
                self._seq += 1
            time.sleep(1.0 / self.fps)

    def read(self):
        with self._lock:
            return self._frame, self._t, self._seq

    def stop(self) -> None:
        self._stop.set()
