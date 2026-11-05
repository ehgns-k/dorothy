"""
SO-101 arm: smooth motion, torque-sensed grasps and full chess-move execution.

Calibration format (calibration.json):
    "<square>": [floor_pose, lift_pose]   a1..h8 (most are predicted, see arm_model.py)
    "capture":  pose                      captured pieces are released here
    "init":     pose                      rest pose, out of the camera's view
    "gripper":  {"open": ticks, "close_dir": +1|-1}
Each pose is {joint_name: raw_ticks}. Older files with [floor, lift] pairs for
capture/init and [open, closed] for the gripper are still accepted.
"""

import json
import logging
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import chess

import config

log = logging.getLogger("robot")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = "gripper"


class RobotError(RuntimeError):
    """The arm cannot continue (comms lost, not calibrated, ...)."""


# --------------------------------------------------------------------------- #
# Move planning (pure chess logic - shared by the real and simulated arm)
# --------------------------------------------------------------------------- #
def move_plan(board: chess.Board, move: chess.Move) -> List[Tuple[str, str]]:
    """Ordered (src, dst) transports for `move` on the PRE-move board.
    Captured pieces go to "capture" first."""
    name = chess.square_name
    if board.is_castling(move):
        rank = 0 if board.turn == chess.WHITE else 7
        if board.is_kingside_castling(move):
            return [(name(chess.square(4, rank)), name(chess.square(6, rank))),
                    (name(chess.square(7, rank)), name(chess.square(5, rank)))]
        return [(name(chess.square(4, rank)), name(chess.square(2, rank))),
                (name(chess.square(0, rank)), name(chess.square(3, rank)))]
    src, dst = name(move.from_square), name(move.to_square)
    if board.is_en_passant(move):
        ep = name(chess.square(chess.square_file(move.to_square),
                               chess.square_rank(move.from_square)))
        return [(ep, "capture"), (src, dst)]
    if board.is_capture(move):
        return [(dst, "capture"), (src, dst)]
    return [(src, dst)]


def min_jerk(s: float) -> float:
    """Smooth 0->1 profile with zero velocity and acceleration at both ends."""
    s = min(1.0, max(0.0, s))
    return s * s * s * (10 + s * (-15 + 6 * s))


def move_duration(max_delta: float, speed_frac: float = 1.0) -> float:
    v = config.MAX_JOINT_SPEED_TPS * speed_frac
    a = config.MAX_JOINT_ACCEL_TPS2 * speed_frac
    # min-jerk peak velocity = 1.875 d/T, peak acceleration = 5.77 d/T^2
    return max(config.MIN_MOVE_TIME_S, 1.875 * max_delta / v,
               math.sqrt(5.774 * max_delta / a))


# --------------------------------------------------------------------------- #
# Real arm
# --------------------------------------------------------------------------- #
class RobotArm:
    def __init__(self, port: str, calibration_file=config.CALIBRATION_FILE):
        from servo_bus import FeetechBus  # imported lazily so --sim needs no SDK
        self.calibration_file = calibration_file
        self.calibration: Dict = {}
        if calibration_file.exists():
            self.calibration = json.loads(calibration_file.read_text())
            import arm_model
            if arm_model.needs_completion(self.calibration):
                # floor-only / reduced calibration not completed yet: compute once
                self.calibration, model = arm_model.complete(self.calibration)
                if model is not None:
                    calibration_file.write_text(json.dumps(self.calibration, indent=2))
                    log.info("computed all squares from %d taught floor poses",
                             len(self.calibration["_taught"]))
            log.info("calibration: %d entries", len(self.calibration))
        self.bus = FeetechBus(port, config.MOTOR_IDS, config.SERVO_BAUDRATE)
        self.abort = False  # set from another thread to stop motion ASAP

    # ---- lifecycle ------------------------------------------------------ #
    def connect(self) -> None:
        self.bus.connect()
        # Anchor goal = present so enabling torque cannot jerk the arm.
        self.enable_torque_here()
        self._setup_gripper_force()
        # Home first (opens the jaws - drops anything left from an interrupted move).
        self.park()

    def _setup_gripper_force(self) -> None:
        """Work out how hard the gripper may HOLD a piece.

        The SO-101 gripper servo is usually configured (by LeRobot's setup)
        with Overload_Torque = 25 %: if its output stays above that for
        Protection_Time (2 s) it switches its torque OFF - which drops the
        piece in mid-air. So the closing phase may use the full
        GRIPPER_TORQUE_LIMIT for a firm, quick squeeze, but while carrying the
        piece the output is capped just below the servo's own overload
        threshold."""
        self.grip_close_limit = int(config.GRIPPER_TORQUE_LIMIT)
        self.grip_hold_limit = int(config.GRIPPER_TORQUE_LIMIT)
        try:
            from servo_bus import ADDR_OVERLOAD_TORQUE, ADDR_PROTECTION_TIME
            overload = int(self.bus.read_register(GRIPPER, ADDR_OVERLOAD_TORQUE, 1))
            ptime = int(self.bus.read_register(GRIPPER, ADDR_PROTECTION_TIME, 1))
            if 0 < overload < 100:
                self.grip_hold_limit = min(self.grip_hold_limit,
                                           overload * 10 - config.GRIP_OVERLOAD_MARGIN)
            if config.GRIPPER_HOLD_TORQUE:
                self.grip_hold_limit = min(self.grip_hold_limit, int(config.GRIPPER_HOLD_TORQUE))
            log.info("gripper: overload protection at %d %% after %.1f s -> closing force %d, "
                     "holding force %d (of 1000)", overload, ptime / 100.0,
                     self.grip_close_limit, self.grip_hold_limit)
        except Exception as e:
            log.warning("could not read the gripper's overload settings (%s) - holding with %d",
                        e, self.grip_hold_limit)
        self._set_grip_force(self.grip_close_limit)

    def _set_grip_force(self, limit: int) -> None:
        try:
            self.bus.set_torque_limit(GRIPPER, limit)
        except Exception as e:
            log.warning("could not set gripper torque limit: %s", e)

    def close(self) -> None:
        """Torque off (the arm goes limp) and release the port."""
        self.limp()
        try:
            self.bus.disconnect(disable_torque=False)
        except Exception:
            pass

    def limp(self) -> None:
        """Stop any motion and switch all torque off - safe from any thread
        (used on Ctrl+C / kill as well as normal exit)."""
        self.abort = True
        try:
            self.bus.go_limp()
        except Exception as e:
            log.warning("could not switch torque off: %s", e)

    def health(self) -> Dict:
        try:
            temps = self.bus.read_temperatures()
            return {"ok": True, "max_temp": max(temps.values()),
                    "voltage": self.bus.read_voltage()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---- primitives ----------------------------------------------------- #
    def positions(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.bus.read_positions().items()}

    def moveto(self, pose: Dict, speed_frac: float = 1.0, settle: bool = False) -> None:
        target = {n: int(round(pose[n])) for n in ARM_JOINTS}
        start = {n: float(v) for n, v in self.bus.read_positions(ARM_JOINTS).items()}
        deltas = {n: target[n] - start[n] for n in ARM_JOINTS}
        max_delta = max(abs(d) for d in deltas.values())
        if max_delta >= 2:
            T = move_duration(max_delta, speed_frac)
            dt = 1.0 / config.MOTION_UPDATE_HZ
            t0 = time.monotonic()
            while True:
                if self.abort:
                    raise RobotError("motion aborted")
                s = (time.monotonic() - t0) / T
                k = min_jerk(s)
                self.bus.sync_write_positions(
                    {n: start[n] + deltas[n] * k for n in ARM_JOINTS})
                if s >= 1.0:
                    break
                # sleep to the next tick boundary (robust against jitter)
                time.sleep(max(0.0, dt - ((time.monotonic() - t0) % dt)))
        if settle:
            self._settle(target)

    def _settle(self, target: Dict[str, int]) -> None:
        deadline = time.monotonic() + config.SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            cur = self.bus.read_positions(ARM_JOINTS)
            err = max(abs(cur[n] - target[n]) for n in ARM_JOINTS)
            if err <= config.SETTLE_TOL_TICKS:
                return
            time.sleep(0.02)
        log.debug("settle timeout (err=%s)", err)

    # ---- gripper -------------------------------------------------------- #
    def _gripper_cfg(self) -> Tuple[int, int]:
        """(open_ticks, close_direction)."""
        g = self.calibration.get(GRIPPER)
        if isinstance(g, dict):
            return int(g["open"]), int(g["close_dir"])
        if isinstance(g, list):  # legacy [open_pose, closed_pose]
            o, c = g[0][GRIPPER], g[1][GRIPPER]
            return int(o), (1 if c > o else -1)
        return 2000, 1

    def _gripper_write(self, value: int) -> None:
        try:
            err = self.bus.write_position(GRIPPER, value)
        except Exception as e:
            err = 0
            if "Overload" not in str(e):
                raise
        if err & 32:  # overload latch: cycle torque and retry once
            log.warning("gripper overload latch - cycling torque")
            self.bus.set_torque(False, [GRIPPER])
            time.sleep(0.15)
            self.bus.set_torque(True, [GRIPPER])
            time.sleep(0.05)
            self.bus.write_position(GRIPPER, value)

    def open_gripper(self, wait: bool = True) -> None:
        self._set_grip_force(getattr(self, "grip_close_limit", config.GRIPPER_TORQUE_LIMIT))
        self._gripper_write(self._gripper_cfg()[0])
        if wait:
            time.sleep(config.GRIPPER_SETTLE_S)

    def _close_until_contact(self) -> int:
        """Close gradually; stop at the first sign of resistance (the servo
        lags its goal or its load jumps). Leaves a light squeeze applied.
        Returns the position where the jaws stalled."""
        opened, d = self._gripper_cfg()
        start, _ = self.bus.read_position_load(GRIPPER)
        limit = opened + d * config.GRIP_MAX_TRAVEL
        t0 = time.monotonic()
        dt = 1.0 / config.MOTION_UPDATE_HZ
        last_pos, still_since = start, None
        pos = start
        contact = None  # position at the FIRST sign of pressure
        while True:
            t = time.monotonic() - t0
            goal = start + d * config.GRIP_CLOSE_SPEED_TPS * t
            goal = min(goal, limit) if d > 0 else max(goal, limit)
            self._gripper_write(int(goal))
            time.sleep(dt)
            pos, load = self.bus.read_position_load(GRIPPER)
            lag = (goal - pos) * d
            if abs(pos - last_pos) > 2:
                still_since = None
            elif still_since is None:
                still_since = time.monotonic()
            last_pos = pos
            if contact is None and t > 0.12 and (lag > config.GRIP_LAG_TICKS / 3 or
                                                 abs(load) > config.GRIP_LOAD_LIMIT / 2):
                contact = pos
            blocked = t > 0.12 and (lag > config.GRIP_LAG_TICKS or
                                    abs(load) > config.GRIP_LOAD_LIMIT)
            stalled = still_since is not None and time.monotonic() - still_since > 0.15 \
                and lag > config.GRIP_LAG_TICKS / 2
            at_limit = goal == limit and still_since is not None
            if blocked or stalled or at_limit or t > 4.0:
                break
        # go a little further than where pressure was first felt, and hold
        if contact is None or (pos - contact) * d < 0:
            contact = pos
        hold = contact + d * config.GRIPPER_SQUEEZE_TICKS
        hold = min(hold, limit) if d > 0 else max(hold, limit)
        self._gripper_write(int(hold))
        # firm squeeze for a moment, then drop to the holding force before the
        # servo's overload protection (2 s above its threshold) can trip
        time.sleep(config.GRIP_SQUEEZE_S)
        self._set_grip_force(getattr(self, "grip_hold_limit", config.GRIPPER_TORQUE_LIMIT))
        return int(pos)

    def grip(self) -> None:
        """Close on the piece and keep holding it. Whether the move worked is
        judged afterwards by the camera, once the arm is back at rest."""
        self._close_until_contact()

    # ---- composite ------------------------------------------------------ #
    def _single(self, key: str) -> Dict:
        """Rest/capture pose (legacy [a, b] pairs: the first one)."""
        v = self.calibration[key]
        return v[0] if isinstance(v, list) else v

    def _pose(self, key: str, idx: int) -> Dict:
        return self.calibration[key][idx]

    def _transit(self, target: Dict) -> None:
        """Move between the board and an off-board pose (rest / capture)
        without sweeping low over the pieces: first take the target's height
        (radial joints) at the current rotation, or rise to the current
        height before swinging - whichever pose is the HIGHER board pose."""
        cur = self.positions()
        if abs(cur["shoulder_pan"] - target["shoulder_pan"]) < config.TRANSIT_PAN_TICKS:
            self.moveto(target)
            return
        via = dict(target)
        via["shoulder_pan"] = cur["shoulder_pan"]
        via["wrist_roll"] = cur["wrist_roll"]
        self.moveto(via, settle=True)   # really up/over before the next leg
        self.moveto(target)

    def _swing_from(self, lift: Dict, target: Dict) -> None:
        """Leave a board lift pose towards an off-board pose: swing at lift
        height first, then go down/in."""
        if abs(lift["shoulder_pan"] - target["shoulder_pan"]) >= config.TRANSIT_PAN_TICKS:
            via = dict(lift)
            via["shoulder_pan"] = target["shoulder_pan"]
            via["wrist_roll"] = target["wrist_roll"]
            self.moveto(via, settle=True)   # really up/over before the next leg
        self.moveto(target)

    def enable_torque_here(self) -> None:
        """Torque on WITHOUT jumping: goals are first set to where the arm is
        (after it was moved by hand the servos still hold their old goals)."""
        self.bus.sync_write_positions(self.bus.read_positions())
        self.bus.set_torque(True)

    def _nearest_lift(self, pan: float) -> Optional[Dict]:
        lifts = [v[1] for k, v in self.calibration.items()
                 if k in chess.SQUARE_NAMES and isinstance(v, list)]
        if not lifts:
            return None
        return min(lifts, key=lambda p: abs(p["shoulder_pan"] - pan))

    def park(self, open_jaws: bool = True) -> None:
        """Go to the rest pose (out of the camera's view) from ANYWHERE:
        if a swing is needed, first rise to lift height where the arm is,
        swing at that height, then lower into rest - never sweeping over
        the pieces. Jaws are opened first (so a piece is never carried to
        rest by accident)."""
        if "init" not in self.calibration:
            log.warning("'init' not calibrated - arm may occlude the camera")
            return
        rest = self._single("init")
        if open_jaws and GRIPPER in self.calibration:
            self.open_gripper(wait=False)
        cur = self.positions()
        if abs(cur["shoulder_pan"] - rest["shoulder_pan"]) >= config.TRANSIT_PAN_TICKS:
            lift = self._nearest_lift(cur["shoulder_pan"])
            if lift is not None:
                up = dict(lift)
                up["shoulder_pan"], up["wrist_roll"] = cur["shoulder_pan"], cur["wrist_roll"]
                self.moveto(up, speed_frac=0.6, settle=True)
                self._swing_from(up, rest)
                return
        self.moveto(rest, speed_frac=0.6)

    def _pick(self, sq: str) -> None:
        floor, lift = self._pose(sq, 0), self._pose(sq, 1)
        self.moveto(lift)
        self.open_gripper()
        self.moveto(floor, speed_frac=config.APPROACH_SPEED_FRAC, settle=True)
        self.grip()
        self.moveto(lift, speed_frac=config.APPROACH_SPEED_FRAC)

    def _place(self, sq: str) -> None:
        floor, lift = self._pose(sq, 0), self._pose(sq, 1)
        self.moveto(lift)
        self.moveto(floor, speed_frac=config.APPROACH_SPEED_FRAC, settle=True)
        self.open_gripper()
        self.moveto(lift, speed_frac=config.APPROACH_SPEED_FRAC)

    def _drop_captured(self, from_sq: str) -> None:
        self._swing_from(self._pose(from_sq, 1), self._single("capture"))
        self.open_gripper()

    def check_calibrated(self, keys) -> None:
        missing = sorted(k for k in keys if k not in self.calibration)
        if missing:
            raise RobotError(f"missing calibration for {missing} - run calibration.py")

    def execute(self, board: chess.Board, move: chess.Move,
                progress: Optional[Callable[[str], None]] = None) -> None:
        plan = move_plan(board, move)
        self.check_calibrated({"init"} | {s for p in plan for s in p})
        self.abort = False
        done: List[Tuple[str, str]] = []
        last_lift = None
        try:
            for src, dst in plan:
                if progress:
                    progress(f"{src}->{dst}")
                if last_lift is None:          # leaving rest / capture
                    self._transit(self._pose(src, 1))
                self._pick(src)
                if dst == "capture":
                    self._drop_captured(src)
                    last_lift = None           # we're off the board now
                else:
                    self._place(dst)
                    last_lift = self._pose(dst, 1)
                done.append((src, dst))
        finally:
            rest = self._single("init")
            if last_lift is not None:
                self._swing_from(last_lift, rest)
            else:
                self.moveto(rest, speed_frac=0.7)


# --------------------------------------------------------------------------- #
# Simulated arm (for --sim): moves pieces on the simulated physical board
# --------------------------------------------------------------------------- #
class SimArm:
    def __init__(self, world, seconds_per_transport: float = 1.2):
        self.world = world
        self.t = seconds_per_transport
        self.abort = False

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def limp(self) -> None:
        self.abort = True

    def health(self) -> Dict:
        return {"ok": True, "max_temp": 35, "voltage": 12.0}

    def park(self) -> None:
        time.sleep(0.1)

    def execute(self, board, move, progress=None) -> None:
        plan = move_plan(board, move)
        done = []
        self.world.set_arm_visible(True)
        try:
            for src, dst in plan:
                if progress:
                    progress(f"{src}->{dst}")
                time.sleep(self.t)
                self.world.transport(src, dst)
                done.append((src, dst))
        finally:
            time.sleep(0.3)
            self.world.set_arm_visible(False)
