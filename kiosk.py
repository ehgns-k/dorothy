"""
Exhibition kiosk front-end for the chess robot.

A full-screen, touch/keyboard GUI meant to run unattended on a dedicated
laptop. It REUSES the existing framework unchanged - ChessVision /
RobotMotion from module.py and the move-inference logic from main.py - and
adds a hardened game loop on top:

  * All blocking hardware work runs on a background worker thread so the UI
    never freezes; the worker talks to Tk via queues + root.after().
  * Camera, serial (arm) and engine failures are caught and recovered with
    on-screen "please wait / attendant" overlays instead of crashing.
  * Any unhandled error parks the arm, is logged to kiosk.log, and offers a
    Restart button - the kiosk is designed to never need a console.
  * Sabotage hardening: window close disabled, full-screen + always-on-top,
    unbound keys do nothing, operator-only exit / new-game shortcuts.

Pre-requisites (set up ONCE by the team, then frozen):
    device_config.json   (python find_port.py)
    calibration.json     (python calibration.py)

Public controls:
    SPACE / ENTER ........ "I've moved" / proceed / start
    number keys 1-9 ...... pick a move / promotion piece (big buttons too)
Operator controls (hidden):
    Ctrl+Shift+N ......... force a new game
    Ctrl+Shift+Q ......... quit the kiosk
    F11 .................. toggle full-screen

Launch with pythonw.exe for a console-less kiosk:
    pythonw kiosk.py
"""

import os
import sys
import json
import time
import queue
import logging
import threading
from pathlib import Path

import tkinter as tk

import chess
import chess.engine

# Importing these does NOT modify the originals; module.py installs the
# chesscog sys.path shim on import, so the chesscog imports below resolve.
from module import ChessVision, RobotMotion, DEVICE_CONFIG_FILE, OCCUPANCY_THRESHOLD
from main import (
    infer_move_candidates,
    ROBOT_COLOR,
    ENGINE_PATH,
    ENGINE_THINK_TIME_S,
    DETECTION_ITERATIONS,
)
from chesscog.corner_detection import find_corners
from recap import CfgNode as CN


# --------------------------------------------------------------------------- #
# Configuration / appearance
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "kiosk.log"
# Written on a deliberate operator quit so game.bat can tell an intentional
# shutdown (stop relaunching) from a crash / killed window (relaunch).
STOP_SENTINEL = HERE / "kiosk.stop"

UI_POLL_MS = 50                 # how often Tk drains worker->UI messages
GAMEOVER_AUTORESET_S = 90       # auto-start a new game this long after game over
CAMERA_FAILS_BEFORE_RECOVER = 3 # consecutive read failures -> rebuild camera

BG = "#1b1b1b"
PANEL_BG = "#1b1b1b"
TITLE_FG = "#ffffff"
SUB_FG = "#bdbdbd"
LIGHT_SQ = "#eeeed2"
DARK_SQ = "#6f9f57"
LIGHT_HL = "#f6f669"
DARK_HL = "#b9ca43"
WHITE_PIECE = "#ffffff"
BLACK_PIECE = "#1c1c1c"
LABEL_FG = "#3d4a32"

BTN_BG = "#3a7bd5"
BTN_FG = "#ffffff"
BTN_ACTIVE = "#5a93e0"
OVERLAY_BG = "#101010"

TITLE_FONT = ("Segoe UI", 30, "bold")
SUB_FONT = ("Segoe UI", 18)
BTN_FONT = ("Segoe UI", 20, "bold")
PIECE_FONT_NAME = "Segoe UI Symbol"

PIECE_GLYPH = {
    chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝",
    chess.ROOK: "♜", chess.QUEEN: "♛", chess.KING: "♚",
}
PROMO_BY_NUMBER = {1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("kiosk")


class OperatorQuit(Exception):
    """Raised anywhere in the worker to tear down and exit the kiosk."""


class NewGameRequest(Exception):
    """Raised in the worker to abandon the current game and reset."""


# --------------------------------------------------------------------------- #
# Kiosk application
# --------------------------------------------------------------------------- #
class KioskApp:
    def __init__(self) -> None:
        self.human_color = not ROBOT_COLOR
        self.human_white = (self.human_color == chess.WHITE)

        # Hardware handles (created by the worker during init).
        self.vision = None
        self.motion = None
        self.engine = None
        self.corner_cfg = None
        self.camera_id = None
        self.robot_port = None
        self.last_threshold = OCCUPANCY_THRESHOLD
        self._camera_fail_count = 0

        # Worker <-> UI plumbing.
        self.ui_q: "queue.Queue" = queue.Queue()       # worker -> UI callables
        self.action_q: "queue.Queue" = queue.Queue()   # UI -> worker (confirm/number)
        self._quit_evt = threading.Event()             # operator quit
        self._newgame_evt = threading.Event()          # operator/public new game

        self._cur_fen = chess.Board().fen()
        self._cur_last = None
        self._fullscreen = True

        self._build_ui()

    # ----- UI construction ------------------------------------------------- #
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Chess Robot")
        self.root.configure(bg=BG)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        # Closing the window is disabled; operators use Ctrl+Shift+Q.
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)

        # Top status banner.
        status = tk.Frame(self.root, bg=PANEL_BG)
        status.pack(side="top", fill="x", padx=24, pady=(18, 6))
        self.title_var = tk.StringVar(value="Starting up...")
        self.sub_var = tk.StringVar(value="Please wait")
        tk.Label(status, textvariable=self.title_var, bg=PANEL_BG, fg=TITLE_FG,
                 font=TITLE_FONT).pack(anchor="center")
        tk.Label(status, textvariable=self.sub_var, bg=PANEL_BG, fg=SUB_FG,
                 font=SUB_FONT, wraplength=1100, justify="center").pack(anchor="center")

        # Action buttons (rebuilt per prompt).
        self.action_frame = tk.Frame(self.root, bg=PANEL_BG)
        self.action_frame.pack(side="bottom", fill="x", pady=(6, 24))

        # Board canvas (fills the middle, redraws on resize).
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_board(self._cur_fen, self._cur_last))

        # Full-screen overlay for setup/errors (hidden until needed).
        self.overlay = tk.Frame(self.root, bg=OVERLAY_BG)
        self.ov_title = tk.StringVar()
        self.ov_msg = tk.StringVar()
        tk.Label(self.overlay, textvariable=self.ov_title, bg=OVERLAY_BG,
                 fg=TITLE_FG, font=TITLE_FONT).pack(pady=(140, 18))
        tk.Label(self.overlay, textvariable=self.ov_msg, bg=OVERLAY_BG, fg=SUB_FG,
                 font=SUB_FONT, wraplength=1000, justify="center").pack(pady=10)
        self.ov_buttons = tk.Frame(self.overlay, bg=OVERLAY_BG)
        self.ov_buttons.pack(pady=30)

        # Key bindings.
        self.root.bind("<space>", lambda e: self._push("confirm"))
        self.root.bind("<Return>", lambda e: self._push("confirm"))
        for n in range(1, 10):
            self.root.bind(str(n), lambda e, num=n: self._push("number", num))
        self.root.bind("<Control-Shift-Q>", lambda e: self._push("quit"))
        self.root.bind("<Control-Shift-N>", lambda e: self._push("newgame"))
        self.root.bind("<F11>", lambda e: self._toggle_fullscreen())

        self._draw_board(self._cur_fen, self._cur_last)

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    # ----- thread-safe UI helpers (called from the worker) ----------------- #
    def _post(self, fn) -> None:
        self.ui_q.put(fn)

    def _drain_ui(self) -> None:
        try:
            while True:
                fn = self.ui_q.get_nowait()
                try:
                    fn()
                except Exception:
                    log.exception("UI callback failed")
        except queue.Empty:
            pass
        self.root.after(UI_POLL_MS, self._drain_ui)

    def _status(self, title: str, sub: str = "") -> None:
        def apply():
            self.title_var.set(title)
            self.sub_var.set(sub)
        self._post(apply)

    def _render(self, board: chess.Board) -> None:
        fen = board.fen()
        last = board.peek().uci() if board.move_stack else None
        self._post(lambda: self._draw_board(fen, last))

    def _actions(self, specs: list) -> None:
        """specs: list of (label, kind, value). Built on the UI thread."""
        self._post(lambda: self._build_actions(specs))

    def _build_actions(self, specs: list) -> None:
        for child in self.action_frame.winfo_children():
            child.destroy()
        inner = tk.Frame(self.action_frame, bg=PANEL_BG)
        inner.pack(anchor="center")
        for label, kind, value in specs:
            tk.Button(inner, text=label, font=BTN_FONT, bg=BTN_BG, fg=BTN_FG,
                      activebackground=BTN_ACTIVE, activeforeground=BTN_FG,
                      relief="flat", padx=26, pady=14, borderwidth=0,
                      command=lambda k=kind, v=value: self._push(k, v)
                      ).pack(side="left", padx=10)

    def _overlay_show(self, title: str, msg: str, specs: list) -> None:
        def apply():
            self.ov_title.set(title)
            self.ov_msg.set(msg)
            for child in self.ov_buttons.winfo_children():
                child.destroy()
            for label, kind, value in specs:
                tk.Button(self.ov_buttons, text=label, font=BTN_FONT, bg=BTN_BG,
                          fg=BTN_FG, activebackground=BTN_ACTIVE,
                          activeforeground=BTN_FG, relief="flat", padx=26, pady=14,
                          borderwidth=0,
                          command=lambda k=kind, v=value: self._push(k, v)
                          ).pack(side="left", padx=10)
            self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.overlay.lift()
        self._post(apply)

    def _overlay_hide(self) -> None:
        self._post(self.overlay.place_forget)

    # ----- board drawing --------------------------------------------------- #
    def _draw_board(self, fen: str, last_uci) -> None:
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 50:
            return
        self._cur_fen, self._cur_last = fen, last_uci

        margin = 28
        size = min(w, h) - 2 * margin
        sq = size / 8.0
        x0 = (w - size) / 2.0
        y0 = (h - size) / 2.0

        try:
            board = chess.Board(fen)
        except ValueError:
            board = chess.Board()

        hl = set()
        if last_uci:
            try:
                mv = chess.Move.from_uci(last_uci)
                hl = {mv.from_square, mv.to_square}
            except ValueError:
                hl = set()

        piece_font = (PIECE_FONT_NAME, max(10, int(sq * 0.72)))
        label_font = ("Segoe UI", max(8, int(sq * 0.16)))

        for square in chess.SQUARES:
            f = chess.square_file(square)
            r = chess.square_rank(square)
            col = f if self.human_white else 7 - f
            row = (7 - r) if self.human_white else r
            x = x0 + col * sq
            y = y0 + row * sq
            is_light = (f + r) % 2 == 1
            if square in hl:
                fill = LIGHT_HL if is_light else DARK_HL
            else:
                fill = LIGHT_SQ if is_light else DARK_SQ
            c.create_rectangle(x, y, x + sq, y + sq, fill=fill, outline=fill)

            # Edge coordinate labels.
            if row == 7:
                c.create_text(x + sq - 4, y + sq - 4, anchor="se",
                              text="abcdefgh"[f], fill=LABEL_FG, font=label_font)
            if col == 0:
                c.create_text(x + 4, y + 2, anchor="nw",
                              text=str(r + 1), fill=LABEL_FG, font=label_font)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            f = chess.square_file(square)
            r = chess.square_rank(square)
            col = f if self.human_white else 7 - f
            row = (7 - r) if self.human_white else r
            cx = x0 + col * sq + sq / 2.0
            cy = y0 + row * sq + sq / 2.0
            glyph = PIECE_GLYPH[piece.piece_type]
            if piece.color == chess.WHITE:
                fill, outline = WHITE_PIECE, BLACK_PIECE
            else:
                fill, outline = BLACK_PIECE, WHITE_PIECE
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2),
                           (-1, -1), (1, 1), (-1, 1), (1, -1)):
                c.create_text(cx + dx, cy + dy, text=glyph, fill=outline, font=piece_font)
            c.create_text(cx, cy, text=glyph, fill=fill, font=piece_font)

    # ----- actions / waiting ----------------------------------------------- #
    def _push(self, kind: str, value=None) -> None:
        if kind == "quit":
            self._quit_evt.set()
        elif kind == "newgame":
            self._newgame_evt.set()
        else:
            self.action_q.put((kind, value))

    def _drain_actions(self) -> None:
        try:
            while True:
                self.action_q.get_nowait()
        except queue.Empty:
            pass

    def _await(self, allowed, timeout=None, on_tick=None):
        """Block the worker until an allowed action arrives.

        `allowed` is a subset of {"confirm", "number", "newgame"}. Operator
        quit (event) always raises OperatorQuit. Returns (kind, value);
        ("timeout", None) if `timeout` seconds elapse.
        """
        self._drain_actions()
        deadline = time.time() + timeout if timeout else None
        while True:
            if self._quit_evt.is_set():
                raise OperatorQuit
            if "newgame" in allowed and self._newgame_evt.is_set():
                self._newgame_evt.clear()
                return ("newgame", None)
            try:
                kind, value = self.action_q.get(timeout=0.3)
            except queue.Empty:
                if deadline:
                    remaining = max(0, int(round(deadline - time.time())))
                    if on_tick:
                        self._post(lambda r=remaining: on_tick(r))
                    if time.time() >= deadline:
                        return ("timeout", None)
                continue
            if kind in allowed:
                return (kind, value)
            # Unexpected action for this prompt: ignore.

    # ----- hardware init & recovery ---------------------------------------- #
    def _retry(self, fn, label: str):
        """Run fn(); on failure show a retry overlay and loop. Returns fn()."""
        while True:
            try:
                result = fn()
                self._overlay_hide()
                return result
            except OperatorQuit:
                raise
            except Exception as e:
                log.exception("%s", label)
                self._overlay_show(
                    "Setup problem",
                    f"{label}.\n\n{type(e).__name__}: {e}\n\n"
                    "An attendant can resolve this, then press Retry.",
                    [("Retry", "confirm", None), ("Quit", "quit", None)])
                self._await({"confirm"})
                self._overlay_hide()

    def _make_motion(self) -> RobotMotion:
        if self.motion is not None:
            try:
                self.motion.disconnect()
            except Exception:
                pass
        motion = RobotMotion(port=self.robot_port)
        motion.enable_torque()
        return motion

    def _detect_corners(self) -> None:
        img = self.vision.capture_image()
        corners = find_corners(self.corner_cfg, img)
        self.vision.board_corners = corners
        log.info("Board corners detected.")

    def _make_vision(self) -> ChessVision:
        if self.vision is not None:
            try:
                self.vision.release()
            except Exception:
                pass
        vision = ChessVision(camera_id=self.camera_id, robot_color=ROBOT_COLOR)
        vision.occupancy_threshold = self.last_threshold
        return vision

    def _init_hardware(self) -> None:
        self._status("Starting up...", "Loading configuration")
        cfg_path = HERE / DEVICE_CONFIG_FILE
        if not cfg_path.exists():
            self._overlay_show(
                "Configuration missing",
                f"{DEVICE_CONFIG_FILE} not found.\n"
                "Run find_port.py and calibration.py on this machine first.",
                [("Retry", "confirm", None), ("Quit", "quit", None)])
            self._await({"confirm"})
            raise RuntimeError(f"{DEVICE_CONFIG_FILE} missing")
        with open(cfg_path) as f:
            dcfg = json.load(f)
        self.camera_id = dcfg["camera_id"]
        self.robot_port = dcfg["robot_port"]

        self._status("Starting up...", "Loading the vision model")
        self.corner_cfg = CN.load_yaml_with_base("config://corner_detection.yaml")
        self.vision = self._retry(self._make_vision, "Camera / vision init failed")

        self._status("Starting up...", "Connecting to the robot arm")
        self.motion = self._retry(self._make_motion, "Robot arm connection failed")

        self._status("Starting up...", "Starting the chess engine")
        self.engine = self._retry(
            lambda: chess.engine.SimpleEngine.popen_uci(ENGINE_PATH),
            "Chess engine failed to start")

        self._status("Starting up...", "Homing the robot")
        self._park()

        self._status("Starting up...", "Finding the board")
        self._retry(self._detect_corners, "Could not find the board")

    def _park(self) -> None:
        try:
            self.motion.go_to_init()
        except Exception as e:
            log.warning("go_to_init failed: %s", e)

    def _recover_robot(self) -> None:
        self._status("Please wait", "Reconnecting to the robot arm...")
        self.motion = self._retry(self._make_motion, "Robot arm reconnection failed")
        self._park()

    def _recover_engine(self) -> None:
        self._status("Please wait", "Restarting the chess engine...")
        try:
            self.engine.quit()
        except Exception:
            pass
        self.engine = self._retry(
            lambda: chess.engine.SimpleEngine.popen_uci(ENGINE_PATH),
            "Chess engine restart failed")

    def _recover_vision(self) -> None:
        self._status("Please wait", "Recovering the camera...")
        self.vision = self._retry(self._make_vision, "Camera recovery failed")
        self._retry(self._detect_corners, "Could not find the board")
        self._camera_fail_count = 0

    def _write_stop(self) -> None:
        try:
            STOP_SENTINEL.write_text(time.strftime("%Y-%m-%d %H:%M:%S"),
                                     encoding="utf-8")
        except Exception:
            log.exception("could not write stop sentinel")

    def _cleanup(self) -> None:
        for closer in (
            lambda: self.engine.quit() if self.engine else None,
            lambda: self.motion.disconnect() if self.motion else None,
            lambda: self.vision.release() if self.vision else None,
        ):
            try:
                closer()
            except Exception:
                pass

    # ----- engine / robot moves with recovery ------------------------------ #
    def _engine_play(self, board: chess.Board) -> chess.Move:
        attempts = 0
        while True:
            try:
                result = self.engine.play(board, chess.engine.Limit(time=ENGINE_THINK_TIME_S))
                if result.move is None:
                    raise RuntimeError("engine returned no move")
                return result.move
            except OperatorQuit:
                raise
            except Exception as e:
                attempts += 1
                log.exception("engine.play failed (attempt %d)", attempts)
                if attempts >= 4:
                    raise
                self._recover_engine()

    def _robot_move(self, uci: str, board: chess.Board) -> None:
        while True:
            try:
                self.motion.move_robot(uci, board)
                return
            except OperatorQuit:
                raise
            except Exception as e:
                log.exception("robot move failed")
                self._overlay_show(
                    "Robot needs attention",
                    f"The arm could not finish its move.\n\n"
                    f"{type(e).__name__}: {e}\n\n"
                    "An attendant: clear any obstruction, then press Retry.",
                    [("Retry", "confirm", None), ("Quit", "quit", None)])
                self._await({"confirm"})
                self._overlay_hide()
                self._recover_robot()

    # ----- detection ------------------------------------------------------- #
    def _detect_move(self, board: chess.Board) -> list:
        self._park()
        seen: dict = {}
        for i in range(1, DETECTION_ITERATIONS + 1):
            self._status("Reading the board...", f"Scan {i} of {DETECTION_ITERATIONS}")
            try:
                occ = self.vision.detect_occupancy()
            except Exception as e:
                log.warning("occupancy read failed: %s", e)
                self._camera_fail_count += 1
                if self._camera_fail_count >= CAMERA_FAILS_BEFORE_RECOVER:
                    self._recover_vision()
                continue
            self._camera_fail_count = 0
            if occ is None:
                continue
            departed, arrived = self.vision.get_occupancy_diff()
            if departed or arrived:
                for m in infer_move_candidates(board, departed, arrived):
                    seen.setdefault(m.uci(), m)
            self.vision.pop_last_occupancy()
        return list(seen.values())

    # ----- per-turn flow --------------------------------------------------- #
    def _maybe_promotion(self, board: chess.Board, move: chess.Move):
        if move.promotion is not None:
            return move
        if board.piece_type_at(move.from_square) != chess.PAWN:
            return move
        dst_rank = chess.square_rank(move.to_square)
        last_rank = (board.turn == chess.WHITE and dst_rank == 7) or \
                    (board.turn == chess.BLACK and dst_rank == 0)
        if not last_rank:
            return move
        self._status("Pawn promotion!", "Which piece did you place on the board?")
        self._actions([("1  Queen", "number", 1), ("2  Rook", "number", 2),
                       ("3  Bishop", "number", 3), ("4  Knight", "number", 4)])
        while True:
            kind, value = self._await({"number", "newgame"})
            if kind == "newgame":
                raise NewGameRequest
            if value in PROMO_BY_NUMBER:
                return chess.Move(move.from_square, move.to_square,
                                  promotion=PROMO_BY_NUMBER[value])

    def _pick_candidate(self, board: chess.Board, candidates: list):
        self._status("Which move did you play?", "Tap the move you made.")
        specs = [(f"{i}.  {board.san(m)}", "number", i)
                 for i, m in enumerate(candidates, 1)]
        specs.append(("None - scan again", "confirm", None))
        self._actions(specs)
        while True:
            kind, value = self._await({"number", "confirm", "newgame"})
            if kind == "newgame":
                raise NewGameRequest
            if kind == "confirm":
                return None
            if isinstance(value, int) and 1 <= value <= len(candidates):
                return candidates[value - 1]

    def _human_turn(self, board: chess.Board) -> None:
        color = "White" if self.human_color == chess.WHITE else "Black"
        while True:
            self._render(board)
            self._status(f"Your move  ({color})",
                         "Make your move on the board, then press SPACE / ENTER.")
            self._actions([("I've moved   (Space)", "confirm", None)])
            kind, _ = self._await({"confirm", "newgame"})
            if kind == "newgame":
                raise NewGameRequest

            self._actions([])
            candidates = self._detect_move(board)

            if not candidates:
                self._render(board)
                self._status("No move detected",
                             "Make sure the pieces match the screen, "
                             "then press SPACE / ENTER to scan again.")
                self._actions([("Scan again   (Space)", "confirm", None)])
                kind, _ = self._await({"confirm", "newgame"})
                if kind == "newgame":
                    raise NewGameRequest
                continue

            if len(candidates) == 1:
                move = self._maybe_promotion(board, candidates[0])
            else:
                choice = self._pick_candidate(board, candidates)
                if choice is None:
                    continue
                move = self._maybe_promotion(board, choice)

            board.push(move)
            log.info("Human move: %s", move.uci())
            self._render(board)
            return

    def _robot_turn(self, board: chess.Board) -> None:
        self._render(board)
        self._status("Robot's turn", "The robot is thinking...")
        self._actions([])
        move = self._engine_play(board)
        san = board.san(move)
        self._status("Robot's turn", f"The robot plays {san}. Moving the piece...")
        self._robot_move(move.uci(), board)
        board.push(move)
        log.info("Robot move: %s", move.uci())
        self._render(board)

    def _prepare_game(self, board: chess.Board, first: bool) -> None:
        self._newgame_evt.clear()
        self._render(board)
        self._overlay_hide()
        if first:
            self._status("Ready to play",
                         "Set the pieces in the starting position, "
                         "then press SPACE / ENTER to begin.")
        else:
            self._status("New game",
                         "Reset every piece to the starting position, "
                         "then press SPACE / ENTER to begin.")
        self._actions([("Start   (Space)", "confirm", None)])
        self._await({"confirm"})

        self.vision.update_occupancy_from_board(board)

        self._status("Calibrating the camera...",
                     "Tuning piece detection to the current lighting.")
        self._actions([])
        self._park()
        try:
            result = self.vision.grid_search_threshold(board)
            self.vision.occupancy_threshold = result["best"]
            self.last_threshold = result["best"]
            log.info("Grid search: threshold=%.3f accuracy=%d/%d",
                     result["best"], result["accuracy"], result["total"])
        except Exception as e:
            log.warning("grid search failed, keeping threshold %.3f: %s",
                        self.vision.occupancy_threshold, e)

    def _game_over(self, board: chess.Board) -> None:
        self._render(board)
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            reason = "Game over"
        else:
            term = outcome.termination.name.replace("_", " ").title()
            if outcome.winner is None:
                reason = f"{term} - draw ({board.result()})"
            else:
                winner = "White" if outcome.winner == chess.WHITE else "Black"
                reason = f"{term} - {winner} wins ({board.result()})"
        self._status("Game over", reason)
        self._actions([("New game   (Space)", "newgame", None)])

        def tick(remaining):
            self.sub_var.set(f"{reason}\nA new game starts in {remaining}s "
                             f"(press Space to start now).")

        self._await({"confirm", "newgame"}, timeout=GAMEOVER_AUTORESET_S, on_tick=tick)

    def _play_loop(self) -> None:
        board = chess.Board()
        self._prepare_game(board, first=True)
        while True:
            try:
                while not board.is_game_over(claim_draw=True):
                    if board.turn == ROBOT_COLOR:
                        self._robot_turn(board)
                    else:
                        self._human_turn(board)
                    self.vision.update_occupancy_from_board(board)
                self._game_over(board)
            except NewGameRequest:
                log.info("New game requested.")
            board = chess.Board()
            self._prepare_game(board, first=False)

    # ----- worker thread --------------------------------------------------- #
    def _worker(self) -> None:
        while True:
            try:
                self._init_hardware()
                self._play_loop()
            except OperatorQuit:
                log.info("Operator quit.")
                self._write_stop()
                self._cleanup()
                self._post(self.root.destroy)
                return
            except Exception as e:
                log.exception("Fatal error in game loop")
                self._park()
                self._actions([])
                self._overlay_show(
                    "Something went wrong",
                    f"{type(e).__name__}: {e}\n\n"
                    "The kiosk will restart. An attendant can press Restart now.",
                    [("Restart", "confirm", None), ("Quit", "quit", None)])
                try:
                    self._await({"confirm"})
                except OperatorQuit:
                    self._write_stop()
                    self._cleanup()
                    self._post(self.root.destroy)
                    return
                self._overlay_hide()
                self._cleanup()
                # Loop back around to a full re-initialisation.

    def run(self) -> None:
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        self.root.after(UI_POLL_MS, self._drain_ui)
        self.root.mainloop()


def main() -> None:
    # Resolve relative paths (calibration.json, device_config.json, recap
    # config:// roots) regardless of how the kiosk is auto-started.
    os.chdir(HERE)
    log.info("Kiosk starting (robot plays %s)",
             "WHITE" if ROBOT_COLOR == chess.WHITE else "BLACK")
    try:
        KioskApp().run()
    except Exception:
        log.exception("Kiosk crashed during startup")
        raise
    log.info("Kiosk exited.")


if __name__ == "__main__":
    main()
