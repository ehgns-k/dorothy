"""
Game controller: the whole exhibition flow as a worker-thread state machine,
independent of the UI. The UI reads `controller.view` (a snapshot with a
version counter) and sends visitor actions through `controller.send()`.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np

import config
import debuglog
import vision as V
from engine import Engine
from robot import RobotError, move_plan

log = logging.getLogger("game")

# highlight kinds understood by the UI
HL_LAST, HL_DETECT, HL_PROBLEM, HL_HELP, HL_CHECK = "last", "detect", "problem", "help", "check"

PREVIEW_COLORS = {HL_DETECT: (255, 170, 40), HL_PROBLEM: (40, 40, 230),
                  HL_HELP: (40, 200, 255), HL_LAST: (60, 220, 120)}


class OperatorQuit(Exception):
    pass


class NewGame(Exception):
    pass


@dataclass(frozen=True)
class Button:
    label: str
    action: str
    value: object = None
    style: str = "primary"     # primary | secondary | danger | choice
    key: str = ""              # keyboard shortcut shown on the button


@dataclass(frozen=True)
class View:
    phase: str = "boot"
    headline: str = "Starting up"
    detail: str = ""
    tone: str = "info"                     # info | good | warn | error | robot
    fen: str = chess.STARTING_FEN
    last_move: Optional[str] = None
    arrow: Optional[Tuple[int, int]] = None
    highlights: Tuple = ()                 # ((square, kind), ...)
    moves_san: Tuple = ()
    eval_pawns: Optional[float] = None
    difficulty: str = ""
    robot_white: bool = True
    buttons: Tuple = ()
    progress: Optional[float] = None       # 0..1 countdown / activity bar
    busy: bool = False                     # animated bar: checking, please wait
    overlay: Optional[Tuple[str, str]] = None
    hand: bool = False
    engine_name: str = ""
    status_line: str = ""
    version: int = 0


class Controller:
    def __init__(self, sim: bool = False):
        self.sim = sim
        self.world = None
        self.camera = None
        self.vision: Optional[V.Vision] = None
        self.arm = None
        self.engine: Optional[Engine] = None
        self.robot_color = config.robot_color()
        self.board = chess.Board()
        self.difficulty_idx = config.DEFAULT_DIFFICULTY
        self._view = View(robot_white=self.robot_color == chess.WHITE)
        self._vlock = threading.Lock()
        self._actions: "queue.Queue" = queue.Queue()
        self._quit = threading.Event()
        self._newgame = threading.Event()
        self.games_played = 0
        self.rec = debuglog.get()          # detailed debug recording (debuglog.py)
        self._last_board_check = 0.0

    # ------------------------------------------------------------------ #
    # view / actions plumbing
    # ------------------------------------------------------------------ #
    @property
    def view(self) -> View:
        with self._vlock:
            return self._view

    def _set(self, **kw) -> None:
        # a new message clears the "busy" animation unless it says otherwise
        if "busy" not in kw and ({"headline", "buttons", "phase"} & kw.keys()):
            kw["busy"] = False
        with self._vlock:
            old = self._view
            self._view = new = replace(old, version=old.version + 1, **kw)
        if self.rec.enabled and (new.headline, new.detail, new.buttons, new.overlay, new.phase) != \
                (old.headline, old.detail, old.buttons, old.overlay, old.phase):
            self.rec.event("screen", phase=new.phase, headline=new.headline, detail=new.detail,
                           buttons=[b.label for b in new.buttons], overlay=new.overlay,
                           fen=new.fen)

    def _board_view(self, highlights: Dict[int, str] = None, arrow=None) -> dict:
        b = self.board
        hl = dict(highlights or {})
        if b.is_check():
            hl.setdefault(b.king(b.turn), HL_CHECK)
        if self.vision is not None:
            self.vision.highlight = {s: PREVIEW_COLORS[k] for s, k in hl.items()
                                     if k in PREVIEW_COLORS}
        san, tmp = [], chess.Board()
        for m in b.move_stack:
            san.append(tmp.san(m))
            tmp.push(m)
        return dict(fen=b.fen(), last_move=b.peek().uci() if b.move_stack else None,
                    highlights=tuple(hl.items()), arrow=arrow, moves_san=tuple(san))

    def send(self, action: str, value=None) -> None:
        self.rec.event("action", action=action, value=value)
        if action == "quit":
            self._quit.set()
        elif action == "newgame":
            self._newgame.set()
        self._actions.put((action, value))

    def _poll(self, timeout: float = 0.0):
        """Return the next visitor action (or None), raising on quit / new game."""
        if self._quit.is_set():
            raise OperatorQuit
        try:
            a = self._actions.get(timeout=timeout) if timeout else self._actions.get_nowait()
        except queue.Empty:
            return None
        if a[0] == "quit":
            raise OperatorQuit
        if a[0] == "newgame":
            self._newgame.clear()
            raise NewGame
        return a

    def _flush(self) -> None:
        while True:
            try:
                self._actions.get_nowait()
            except queue.Empty:
                return

    def _wait_for(self, allowed, timeout: Optional[float] = None, tick=None,
                  flush: bool = True):
        """Wait for one of `allowed`. flush=False keeps presses that arrived
        meanwhile - needed in loops that re-enter this every second, or a key
        pressed at the wrong moment is silently dropped."""
        if flush:
            self._flush()
        deadline = None if timeout is None else time.time() + timeout
        while True:
            a = self._poll(0.1)
            if a and a[0] in allowed:
                return a
            if deadline is not None:
                left = deadline - time.time()
                if tick:
                    tick(max(0.0, left))
                if left <= 0:
                    return ("timeout", None)

    def _sleep(self, s: float) -> None:
        end = time.time() + s
        while time.time() < end:
            self._poll(min(0.1, max(0.01, end - time.time())))

    # ------------------------------------------------------------------ #
    # hardware
    # ------------------------------------------------------------------ #
    def _retry(self, label: str, fn):
        while True:
            try:
                return fn()
            except (OperatorQuit, NewGame):
                raise
            except Exception as e:
                log.exception(label)
                self._set(phase="error", overlay=(label, f"{e}\n\nAn attendant can fix this "
                                                         "and press Retry."),
                          buttons=(Button("Retry", "retry", key="Space"),))
                self._wait_for({"retry"})
                self._set(overlay=None, buttons=())

    def _init_hardware(self) -> None:
        self.rec = debuglog.start()
        self.rec.event("session", sim=self.sim, robot_color=config.ROBOT_COLOR,
                       settings={k: getattr(config, k) for k in dir(config)
                                 if k.isupper() and isinstance(getattr(config, k),
                                                               (int, float, str, bool))})
        self._set(phase="boot", headline="Starting up", detail="Preparing the camera")
        if self.sim:
            from sim import SIM_CORNERS, SimCamera, SimWorld
            from robot import SimArm
            self.world = SimWorld(style=config.SIM_STYLE, lean=config.SIM_STYLE == "real")
            self.camera = SimCamera(self.world).start()
            geo = V.BoardGeometry(SIM_CORNERS)
            self.arm = SimArm(self.world)
        else:
            from robot import RobotArm
            dcfg = self._retry("Configuration missing", config.load_device_config)
            if self.camera is None:
                self.camera = self._retry("Camera not available",
                                          lambda: V.Camera(dcfg.get("camera", dcfg.get("camera_id", 0))).start())
            self._set(detail="Finding the chessboard (this can take a few seconds)...")

            def locate():
                saved = V.BoardGeometry.load()
                g = V.refine_geometry(self.camera, saved, assume_start=True)
                if g is not None and (saved is None or not V.HOME_BG_FILE.exists()):
                    V.save_home_background(self.camera.read()[0], g)
                if g is None:
                    raise RuntimeError("Can't see the chessboard. Make sure it is fully in "
                                       "view with the pieces in their starting position "
                                       "(or run: python setup_board.py).")
                g.save()
                return g
            geo = self._retry("Chessboard not found", locate)
            self._set(detail="Connecting to the robot arm")

            def connect():
                arm = RobotArm(dcfg["robot_port"])
                arm.connect()
                return arm
            self.arm = self._retry("Robot arm not responding", connect)
        if self.vision is None:
            self.vision = V.Vision(self.camera, geo).start()
        else:
            self.vision.geometry = geo
        self.vision.recorder = self.rec if self.rec.enabled else None
        self._set(detail="Starting the chess engine")
        self.engine = Engine(allow_fallback=self.sim)
        self._retry("Chess engine failed to start", self.engine.start)
        self._set(engine_name=self.engine.name)
        self._set(detail="Moving the arm to its rest position")
        self._retry("Robot arm could not reach its rest pose", self.arm.park)
        self._retry("Camera shows no picture",
                    lambda: self.vision.wait_stable(timeout=8, min_s=0.3) or
                    (_ for _ in ()).throw(RuntimeError("no stable camera image")))

    def request_quit(self) -> None:
        """Operator quit: stop whatever the worker is doing as fast as
        possible (arm motion aborts at the next control tick)."""
        self._quit.set()
        self._actions.put(("quit", None))
        if self.arm is not None:
            self.arm.abort = True

    def emergency_limp(self) -> None:
        """Ctrl+C / kill: stop moving and let the arm go limp immediately."""
        self._limp_requested = True
        self._quit.set()
        if self.arm is not None:
            self.arm.limp()

    def shutdown(self) -> None:
        """Normal exit: park, then torque off. After emergency_limp() the
        arm is NOT moved again - it just stays limp."""
        worker = getattr(self, "worker", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        # park only if nothing else can be driving the arm any more
        park = not getattr(self, "_limp_requested", False) and \
            not (worker is not None and worker.is_alive())

        def do_park():
            if park and self.arm:
                self.arm.abort = False
                self.arm.park()
        for fn in (lambda: self.engine and self.engine.quit(),
                   do_park,
                   lambda: self.arm and self.arm.close(),
                   lambda: self.vision and self.vision.stop(),
                   lambda: self.camera and self.camera.stop()):
            try:
                fn()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Worker-thread entry point. Never returns except on operator quit."""
        while True:
            try:
                self._init_hardware()
                while True:
                    try:
                        self._attract()
                        self._play()
                    except NewGame:
                        log.info("new game requested")
            except OperatorQuit:
                # the UI closes and main() runs shutdown() (park + limp)
                log.info("operator quit")
                self._set(phase="exit")
                return
            except Exception as e:
                log.exception("fatal error in game loop")
                self._set(phase="error", overlay=("Something went wrong",
                                                  f"{type(e).__name__}: {e}\n\n"
                                                  "Restarting automatically..."),
                          buttons=(Button("Restart now", "retry", key="Space"),))
                try:
                    self._wait_for({"retry"}, timeout=10)
                except OperatorQuit:
                    self._set(phase="exit")
                    return
                except NewGame:
                    pass
                self._set(overlay=None, buttons=())

    # ------------------------------------------------------------------ #
    def _attract(self) -> None:
        """Start screen: pick a difficulty; check the pieces are set up."""
        self.board = chess.Board()
        if self.sim and self.games_played:
            self.world.reset()
        names = [d["name"] for d in config.DIFFICULTIES]
        self._flush()  # a leftover key from the last game must not start this one
        while True:
            self._periodic_board_check()
            mismatch = self.vision.start_position_mismatch() or []
            hl = {s: HL_PROBLEM for s in mismatch}
            buttons = [Button(n, "difficulty", i,
                              "choice-on" if i == self.difficulty_idx else "choice", str(i + 1))
                       for i, n in enumerate(names)]
            buttons.append(Button("Start game", "start", None, "primary", "Space"))
            detail = ("Choose how strong the robot should be (\u2190 \u2192), then press "
                      "Enter or Start."
                      if not mismatch else
                      "Please set every piece on its starting square - the highlighted "
                      "squares don't look right yet.")
            self._set(phase="attract", overlay=None, headline="Can you beat the robot?",
                      detail=detail, tone="warn" if mismatch else "info",
                      buttons=tuple(buttons), progress=None, eval_pawns=None,
                      difficulty=names[self.difficulty_idx], **self._board_view(hl))
            a = self._wait_for({"difficulty", "start"}, timeout=1.0, flush=False)
            if a[0] == "difficulty":
                self.difficulty_idx = int(a[1])
            elif a[0] == "start":
                break
        self.engine.set_difficulty(self.difficulty_idx)
        self._set(phase="starting", headline="Getting ready...", tone="info", buttons=(),
                  detail="Checking the camera view of the board.")
        if not self.vision.wait_stable(timeout=5):
            log.warning("scene not still at game start")
        if not self.sim:  # the camera / board may have moved since the last game
            self._ensure_board(force=True)
        first = self.vision.start_reference is None
        mismatch = None if first else self.vision.start_position_mismatch()
        if first or not mismatch:
            # pieces known to be in the starting position: learn how far they
            # lean and what empty / white / black squares look like right now
            self.vision.learn_start_position()
        else:
            log.warning("game starts with a doubtful setup on %s - keeping the previous "
                        "colour model", [chess.square_name(s) for s in mismatch])
        if first:
            self.vision.remember_start_position()
        self.vision.set_reference(self.board, trusted=False)
        self._last_board_check = time.time()
        self.games_played += 1
        log.info("game %d start (difficulty %s)", self.games_played, names[self.difficulty_idx])
        self.rec.event("game_start", game=self.games_played,
                       difficulty=names[self.difficulty_idx], setup_mismatch=[
                           chess.square_name(s) for s in (mismatch or [])])

    def _play(self) -> None:
        while not self.board.is_game_over(claim_draw=True):
            if self.board.turn == self.robot_color:
                self._robot_turn()
            else:
                self._human_turn()
            self._update_eval()
        self._game_over()

    def _update_eval(self) -> None:
        try:
            ev = self.engine.evaluate(self.board)
        except Exception:
            ev = None
        self._set(eval_pawns=ev)

    # ------------------------------------------------------------------ #
    # board / camera movement
    # ------------------------------------------------------------------ #
    def _periodic_board_check(self) -> None:
        """Re-detect the board every BOARD_RECHECK_S while nobody is reaching
        in, so a camera that was bumped is re-registered quietly. A failed
        look-up is NOT an alarm here (pieces can hide the grid mid-game); the
        move detector asks for help if the view really stops making sense."""
        v = self.vision
        if time.time() - self._last_board_check < config.BOARD_RECHECK_S or not v.is_stable():
            return
        self._last_board_check = time.time()
        try:
            st = v.board_status(attempts=1)
        except Exception:
            log.exception("periodic board check failed")
            return
        if st["kind"] == "camera_moved":
            v.geometry.save()
        elif st["kind"] != "ok":
            log.info("periodic board check: %s (ignored)", st["kind"])

    def _ensure_board(self, force: bool = False) -> None:
        """Make sure vision is registered on the board and the board is where
        the arm expects it. Camera moves are absorbed silently; a board that
        slid away from the arm makes us ask the visitor to push it back."""
        v = self.vision
        if not force and v.alignment_shift() < config.BOARD_DRIFT_PX:
            return
        saved_view = self.view
        quitting = lambda: self._quit.is_set()  # noqa: E731
        n = config.BOARD_LOST_RETRIES
        # Only ask for help once we are SURE: several failed look-ups in a row
        # (a visitor's arm over the board must not trigger the prompt). After
        # the first miss, show what's going on so it never looks frozen.
        for attempt in range(n):
            if attempt:
                self._set(phase="board", headline="Looking for the chessboard...",
                          tone="info", overlay=None, buttons=(), progress=attempt / n,
                          detail=f"Attempt {attempt + 1} of {n}. Please keep hands away from "
                                 "the board for a moment.")
            v.wait_stable(timeout=3, min_s=0.3, cancel=quitting)
            self._poll()  # honours Quit / New game
            st = v.board_status()
            if st["kind"] in ("ok", "camera_moved"):
                if st["kind"] == "camera_moved" or st.get("offset", 0) > 1.0:
                    v.geometry.save()
                if attempt:
                    with self._vlock:
                        self._view = replace(saved_view, version=self._view.version + 1)
                return
            log.info("board check %d/%d: %s", attempt + 1, n, st["kind"])
            self._sleep(0.7)
        log.warning("board check: %s", st)
        v.show_home = st["kind"] == "board_moved"
        v.show_grid = st["kind"] != "lost"
        checks = n
        try:
            while True:
                status = f"\n\nChecked {checks} times - I keep looking automatically."
                if st["kind"] == "lost":
                    self._set(phase="board", headline="I can't see the chessboard",
                              tone="error", overlay=None, progress=None,
                              detail="Something may be covering the board, or the camera "
                                     "was moved too far. Please make sure the camera sees the "
                                     "whole board (live view on the left)." + status,
                              buttons=(Button("Check again", "recheck", key="Space"),))
                else:
                    off = st.get("offset", 0)
                    self._set(phase="board", headline="The board has moved", tone="warn",
                              overlay=None, progress=None,
                              detail="The robot needs the board where it was set up. Please "
                                     "slide the board so its corners match the yellow outline "
                                     f"in the camera view (off by {off:.0f} px)." + status,
                              buttons=(Button("Check again", "recheck", key="Space"),
                                       Button("Keep it here (attendant)", "rehome",
                                              style="secondary")))
                a = self._wait_for({"recheck", "rehome"}, timeout=1.5, flush=False)
                if a[0] == "rehome" and st["kind"] == "board_moved":
                    frame, _, _ = self.camera.read()
                    v.geometry = V.BoardGeometry(v.geometry.corners)
                    v.geometry.save()
                    V.save_home_background(frame, v.geometry)
                    log.warning("attendant accepted new board position as home")
                    break
                if not v.wait_stable(timeout=2, min_s=0.4, cancel=quitting):
                    self._poll()
                    continue
                self._set(detail=self.view.detail.split("\n\nChecked")[0]
                          + "\n\nChecking now...", progress=0.5)
                st = v.board_status()
                checks += 1
                if st["kind"] in ("ok", "camera_moved"):
                    v.geometry.save()
                    break
                v.show_home = st["kind"] == "board_moved"
                v.show_grid = st["kind"] != "lost"
        finally:
            v.show_home = False
            v.show_grid = True
        log.info("board back in place")
        with self._vlock:
            self._view = replace(saved_view, version=self._view.version + 1)

    # ------------------------------------------------------------------ #
    # robot turn
    # ------------------------------------------------------------------ #
    def _robot_turn(self) -> None:
        b = self.board
        if not self.sim:
            self._ensure_board()
        self._set(phase="robot_think", headline="Robot is thinking...", tone="robot",
                  detail="", buttons=(), progress=None, **self._board_view())
        move = self.engine.best_move(b)
        self.rec.event("engine", fen=b.fen(), move=move.uci(), choice=self.engine.last_choice)
        self._poll()   # honour Quit / New game pressed while it was thinking
        san = b.san(move)
        expected = V.changed_squares(b, move)
        self._set(phase="robot_move", headline=f"Robot plays {san}",
                  detail=self._describe(b, move), tone="robot",
                  **self._board_view({s: HL_DETECT for s in expected},
                                     arrow=(move.from_square, move.to_square)))
        log.info("robot: %s", move.uci())
        self.rec.event("robot_move", move=move.uci(), san=san, fen=b.fen())
        before = b.copy()
        self._execute_robot_move(move)
        b.push(move)
        if move.promotion:
            self._ask_human_swap_promotion(move)
        reply = self._check_robot_move(before, move, san)
        if reply is not None:
            # the visitor answered while the arm was still busy: take it
            # and carry straight on with the robot's next move
            rsan = b.san(reply)
            log.info("human (during robot move): %s", reply.uci())
            self.rec.event("human_move", move=reply.uci(), san=rsan, during_robot_move=True)
            b.push(reply)
            self._set(phase="human_done", headline=f"You played {rsan}", tone="good",
                      detail="Seen while I was moving - my turn again!", buttons=(),
                      progress=None, **self._board_view(
                          {s: HL_DETECT for s in V.changed_squares(before, move) |
                           V.changed_squares(self._pre(b), reply)}))
            self._sleep(1.2)
        self.vision.set_reference(b, trusted=self._verified)
        self._set(**self._board_view())

    @staticmethod
    def _describe(b: chess.Board, move: chess.Move) -> str:
        p = b.piece_at(move.from_square)
        name = chess.piece_name(p.piece_type) if p else "piece"
        frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
        if b.is_castling(move):
            return "Castling - the king and rook swap sides."
        if b.is_en_passant(move):
            return f"Pawn takes en passant on {to}!"
        if b.is_capture(move):
            cap = b.piece_at(move.to_square)
            return f"The {name} on {frm} captures your {chess.piece_name(cap.piece_type)} on {to}."
        if b.gives_check(move):
            return f"{name.capitalize()} {frm} to {to} - check!"
        return f"Moving the {name} from {frm} to {to}."

    def _execute_robot_move(self, move: chess.Move) -> None:
        attempts = 0
        while True:
            try:
                self.arm.execute(self.board, move)
                return
            except RobotError as e:
                attempts += 1
                log.exception("robot error")
                if attempts <= 1:
                    self._set(detail="Reconnecting to the arm...")
                    try:
                        self.arm.close()
                        self.arm.connect()
                        self.arm.park()
                        continue
                    except Exception:
                        log.exception("reconnect failed")
                self._set(overlay=("The robot needs an attendant",
                                   f"{e}\n\nCheck the arm, then press Retry - or move the "
                                   "piece by hand."),
                          buttons=(Button("Retry", "retry", key="Space"),
                                   Button("I'll move it by hand", "byhand", style="secondary")))
                a = self._wait_for({"retry", "byhand"})
                self._set(overlay=None, buttons=())
                if a[0] == "byhand":
                    self._human_finishes_robot_move(move_plan(self.board, move), "")
                    return
                try:
                    self.arm.close()
                    self.arm.connect()
                    self.arm.park()
                except Exception:
                    log.exception("reconnect failed")

    def _human_finishes_robot_move(self, steps, reason: str) -> None:
        lines, hl = [], {}
        for src, dst in steps:
            hl[chess.parse_square(src)] = HL_HELP
            if dst == "capture":
                lines.append(f"take the piece on {src} off the board")
            else:
                lines.append(f"move the piece on {src} to {dst}")
                hl[chess.parse_square(dst)] = HL_HELP
        self._set(phase="help", headline="Could you help me?", tone="warn",
                  detail=(reason + " Please " + ", then ".join(lines) +
                          ", and press Done.").strip(),
                  buttons=(Button("Done", "done", key="Space"),), **self._board_view(hl))
        self._wait_for({"done"})
        self._set(buttons=())

    def _ask_human_swap_promotion(self, move: chess.Move) -> None:
        piece = chess.piece_name(move.promotion)
        self._set(phase="help", headline=f"Promotion to a {piece}!", tone="warn",
                  detail=f"Please replace my pawn on {chess.square_name(move.to_square)} "
                         f"with a {piece}, then press Done.",
                  buttons=(Button("Done", "done", key="Space"),),
                  **self._board_view({move.to_square: HL_HELP}))
        self._wait_for({"done"})
        self._set(buttons=())

    @staticmethod
    def _pre(b: chess.Board) -> chess.Board:
        """The position before the last move of `b`."""
        p = b.copy()
        p.pop()
        return p

    def _look(self, phase: str, fn):
        """Wait for a still scene, then run a vision check (logged)."""
        self.vision.wait_stable(timeout=6, cancel=lambda: self._quit.is_set())
        if self._quit.is_set():
            raise OperatorQuit  # shutting down: don't judge a half-finished scene
        inf = fn()
        if inf is not None:
            self._log_inference(phase, inf, images=True)
        return inf

    def _check_robot_move(self, before: chess.Board, move: chess.Move,
                          san: str) -> Optional[chess.Move]:
        """Once the arm is back at rest: does the board show the robot's move?

        Also recognises a visitor who already played their reply while the
        arm was busy - that reply is returned. If the board looks wrong, the
        visitor is asked to fix it and press OK; a second OK is taken as
        "the board is right" even if the camera still disagrees."""
        expected = V.changed_squares(before, move)
        self._verified = True
        self._set(busy=True, detail="Checking the board...")
        asked = 0
        while True:
            inf = self._look("robot_check", lambda: self.vision.check_robot_move(before, move))
            if inf is not None and inf.status == "move":
                line = inf.lines[0]
                if len(line) == 1:
                    return None
                # the visitor moved too: look once more before believing it
                self._sleep(0.4)
                again = self._look("robot_check", lambda: self.vision.check_robot_move(before, move))
                if again is not None and again.status == "move" and again.lines[0] == line:
                    return self._with_promotion(line[1], [ln[-1] for ln in again.lines])
                continue
            ok, missing, extra = self.vision.verify_expected(expected)
            log.warning("check failed after %s: %s, missing=%s extra=%s", san,
                        inf.status if inf else None,
                        [chess.square_name(s) for s in missing],
                        [chess.square_name(s) for s in extra])
            if asked >= 2:
                log.warning("visitor confirmed the board twice - carrying on")
                self._verified = False
                return None
            hl = {s: HL_PROBLEM for s in missing + extra} or \
                {s: HL_PROBLEM for s in (inf.changed[:6] if inf else [])}
            names = ", ".join(chess.square_name(s) for s in hl) or "the board"
            if asked == 0:
                head = "Is the board right?"
                detail = (f"Something looks off on {names} after {san}. Please make the "
                          "board match the screen, then press OK.")
            else:
                head = "Still looks different"
                detail = (f"I still see a difference on {names}. Please check those squares "
                          "- or press OK if the board matches the screen.")
            self._set(phase="help", headline=head, tone="warn", detail=detail, progress=None,
                      buttons=(Button("OK", "ok", key="Space"),), **self._board_view(hl))
            self._wait_for({"ok"})
            asked += 1
            self._set(phase="checking", headline="Checking the board...", tone="info",
                      detail="Thanks! Please keep your hands away for a moment.",
                      buttons=(), busy=True)

    # ------------------------------------------------------------------ #
    # human turn
    # ------------------------------------------------------------------ #
    def _human_turn(self) -> None:
        b = self.board
        color = "White" if b.turn == chess.WHITE else "Black"
        base = dict(phase="human", tone="info", progress=None)
        idle_since = time.time()
        pending: Optional[chess.Move] = None
        pending_since = 0.0
        rejected: set = set()
        unclear_since: Optional[float] = None
        last_seq, last_infer = -1, 0.0
        last_board_check = 0.0
        inf: Optional[V.Inference] = None
        manual_btn = Button("I've moved", "moved", style="secondary", key="Space")
        prompt = f"Your move ({color})"
        if b.is_check():
            prompt = f"Check! Your move ({color})"
        self._set(headline=prompt, detail="Move a piece - I'm watching the board.",
                  buttons=(manual_btn,), **base, **self._board_view())
        self._flush()

        while True:
            a = self._poll(0.08)
            v = self.vision
            hand = v.hand_in_view()
            if hand:
                idle_since = time.time()

            # -- visitor actions ------------------------------------------ #
            if a and a[0] == "moved":
                self._set(phase="checking", headline="Looking at the board...", tone="info",
                          detail="Please keep your hands away for a moment.", buttons=(),
                          busy=True)
                v.wait_stable(timeout=3, min_s=0.3)
                inf = v.infer(b)
                if inf is not None:
                    self._log_inference("human_button", inf, images=True)
                mv = self._resolve(inf, manual=True)
                if mv is not None:
                    return self._commit(mv)
                self._set(headline=prompt, detail="Move a piece - I'm watching the board.",
                          buttons=(manual_btn,), **base, **self._board_view())
                continue
            if a and a[0] == "reject" and pending is not None:
                rejected.add(pending.uci())
                pending = None
                choice = self._choose(inf.guesses if inf else [], "Which move did you play?",
                                      exclude=rejected)
                if choice is not None:
                    return self._commit(choice)
                self._set(headline=prompt, detail="Move a piece - I'm watching the board.",
                          buttons=(manual_btn,), **base, **self._board_view())
                continue
            if a and a[0] == "confirm" and pending is not None:
                return self._commit(self._with_promotion(pending, inf.moves))

            # -- watch the board ----------------------------------------- #
            if hand:
                if pending is not None or unclear_since is not None:
                    pending, unclear_since = None, None
                    self._set(headline=prompt, detail="Move a piece - I'm watching the board.",
                              buttons=(manual_btn,), **base, **self._board_view())
                self._set(hand=True)
                continue
            self._set(hand=False)
            if not v.is_stable() or not config.AUTO_DETECT:
                continue
            if pending is None and unclear_since is None:
                before_check = self._last_board_check
                self._periodic_board_check()
                if self._last_board_check != before_check:
                    continue     # that took a moment: look at a fresh frame
            if time.time() - idle_since > config.IDLE_RESET_S:
                log.info("idle timeout - abandoning game")
                raise NewGame
            if v.frame_seq == last_seq or time.time() - last_infer < 0.25:
                continue  # inference costs ~0.1 s; 4 Hz is plenty
            last_seq, last_infer = v.frame_seq, time.time()
            inf = v.infer(b)
            if inf is None:
                continue
            self._log_inference("human", inf)

            if inf.status == "move":
                unclear_since = None
                mv = inf.moves[0]
                if mv.uci() in rejected:
                    continue
                if pending is None or pending != mv:
                    pending, pending_since = mv, time.time()
                frac = (time.time() - pending_since) / config.DETECT_CONFIRM_S
                if frac >= 1.0:
                    return self._commit(self._with_promotion(mv, inf.moves))
                self._set(headline=f"You played {b.san(mv)}?",
                          detail="Confirming... press 'Not my move' if that's wrong.",
                          tone="good", progress=frac,
                          buttons=(Button("Yes", "confirm", key="Space"),
                                   Button("Not my move", "reject", style="secondary")),
                          **self._board_view({s: HL_DETECT for s in V.changed_squares(b, mv)},
                                             arrow=(mv.from_square, mv.to_square)))
            elif inf.status == "ambiguous":
                choice = self._choose(inf.moves, "Which move did you play?")
                if choice is not None:
                    return self._commit(choice)
            elif inf.status == "unclear" and not self.sim and len(inf.changed) >= 4 \
                    and time.time() - last_board_check > 5:
                last_board_check = time.time()
                before = v.geometry.corners.copy()
                self._ensure_board(force=True)
                if float(np.abs(v.geometry.corners - before).max()) > 1.5:
                    unclear_since = None   # re-registered: look again with the fixed view
                continue
            elif inf.status == "unclear":
                pending = None
                if unclear_since is None:
                    unclear_since = time.time()
                    self._set(headline=prompt, detail="Hmm, let me look closely...",
                              buttons=(manual_btn,), **base,
                              **self._board_view({s: HL_DETECT for s in inf.changed[:6]}))
                elif time.time() - unclear_since > config.UNEXPLAINED_HINT_S:
                    names = ", ".join(chess.square_name(s) for s in inf.changed[:6])
                    self._set(headline="That doesn't look like a legal move",
                              detail=f"Changes on {names}. Please check those squares - "
                                     "or press 'I've moved' to pick your move.",
                              tone="warn", buttons=(manual_btn,),
                              **self._board_view({s: HL_PROBLEM for s in inf.changed[:6]}))
            else:  # still
                if pending is not None or unclear_since is not None:
                    pending, unclear_since = None, None
                    self._set(headline=prompt, detail="Move a piece - I'm watching the board.",
                              buttons=(manual_btn,), **base, **self._board_view())

    def _resolve(self, inf: Optional[V.Inference], manual: bool) -> Optional[chess.Move]:
        if inf is None:
            return None
        if inf.status == "move":
            return self._with_promotion(inf.moves[0], inf.moves)
        if inf.status == "still":
            self._set(headline="I don't see a move yet", tone="warn",
                      detail="Move one of your pieces, then let go of it.")
            self._sleep(2.0)
            return None
        return self._choose(inf.moves if inf.status == "ambiguous" else inf.guesses,
                            "Which move did you play?")

    def _with_promotion(self, mv: chess.Move, candidates: List[chess.Move]) -> chess.Move:
        promos = [m for m in candidates if m.promotion and
                  m.from_square == mv.from_square and m.to_square == mv.to_square]
        if not promos:
            return mv
        order = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        self._set(phase="promotion", headline="Promotion!", tone="good",
                  detail="Which piece did you put on the board?",
                  buttons=tuple(Button(chess.piece_name(p).capitalize(), "promo", p, "choice",
                                       str(i + 1)) for i, p in enumerate(order)))
        a = self._wait_for({"promo"})
        return chess.Move(mv.from_square, mv.to_square, promotion=int(a[1]))

    def _choose(self, moves: List[chess.Move], title: str,
                exclude=frozenset()) -> Optional[chess.Move]:
        b = self.board
        seen, opts = set(), []
        for m in moves:
            key = (m.from_square, m.to_square)
            if key in seen or m.uci() in exclude:
                continue
            seen.add(key)
            opts.append(m)
        opts = opts[:4]
        buttons = [Button(b.san(chess.Move(m.from_square, m.to_square,
                                           chess.QUEEN if m.promotion else None)),
                          "pick", i, "choice", str(i + 1)) for i, m in enumerate(opts)]
        buttons.append(Button("Look again", "rescan", style="secondary", key="Space"))
        hl = {}
        for m in opts:
            hl[m.from_square] = HL_DETECT
            hl[m.to_square] = HL_DETECT
        self._set(phase="pick", headline=title, tone="warn",
                  detail="Tap your move - or fix the board and press Look again."
                  if opts else "I couldn't work it out. Check the board and press Look again.",
                  buttons=tuple(buttons), progress=None, **self._board_view(hl))
        a = self._wait_for({"pick", "rescan"})
        if a[0] == "rescan":
            return None
        m = opts[int(a[1])]
        return self._with_promotion(m, moves)

    def _commit(self, mv: chess.Move) -> None:
        b = self.board
        san = b.san(mv)
        log.info("human: %s", mv.uci())
        self.rec.event("human_move", move=mv.uci(), san=san, fen=b.fen())
        b.push(mv)
        self._set(phase="human_done", headline=f"You played {san}", tone="good",
                  detail="", buttons=(), progress=None, hand=False, **self._board_view())
        self.vision.wait_stable(timeout=4)
        self.vision.set_reference(b)

    def _log_inference(self, phase: str, inf: "V.Inference", images: bool = False) -> None:
        """Record a recognition result; camera images are attached when asked
        or whenever the verdict changes (so every decision can be replayed)."""
        if not self.rec.enabled:
            return
        r = inf.summary()
        key = (phase, r["status"], tuple(r["lines"][:1]))
        now = time.time()
        if not images and key == getattr(self, "_last_inf_key", None) and \
                now - getattr(self, "_last_inf_t", 0.0) < config.DEBUG_INFER_EVERY_S:
            return
        self._last_inf_t = now
        data = dict(phase=phase, fen=self.board.fen(), result=r)
        if images or key != getattr(self, "_last_inf_key", None):
            v = self.vision
            data["frame"] = self.rec.image(f"{phase}_{r['status']}", v.frame)
            data["board_image"] = self.rec.image(f"{phase}_{r['status']}_board", v.current_board())
        self._last_inf_key = key
        self.rec.event("inference", **data)

    # ------------------------------------------------------------------ #
    def _game_over(self) -> None:
        b = self.board
        out = b.outcome(claim_draw=True)
        if out is None or out.winner is None:
            head = "It's a draw!"
        elif out.winner == self.robot_color:
            head = "The robot wins!"
        else:
            head = "You win! Congratulations!"
        term = out.termination.name.replace("_", " ").lower() if out else ""
        log.info("game over: %s (%s)", b.result(), term)

        def tick(left):
            self._set(progress=1.0 - left / config.GAMEOVER_AUTORESET_S)
        self._set(phase="gameover", headline=head, tone="good",
                  detail=f"{term.capitalize()} - {b.result()}. Please reset the pieces for "
                         "the next player.",
                  buttons=(Button("New game", "again", key="Space"),), **self._board_view())
        self._wait_for({"again"}, timeout=config.GAMEOVER_AUTORESET_S, tick=tick)
