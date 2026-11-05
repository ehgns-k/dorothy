"""Robot logic against a simulated servo bus: torque-sensed gripping,
empty-grasp detection, and the rest/capture transit sequence."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess  # noqa: E402

import config  # noqa: E402
import robot  # noqa: E402

config.MOTION_UPDATE_HZ = 200
config.MAX_JOINT_SPEED_TPS = 2500    # quick, but within what the fake servos can follow
config.MAX_JOINT_ACCEL_TPS2 = 20000
config.MIN_MOVE_TIME_S = 0.01
config.GRIPPER_SETTLE_S = 0.01


class FakeBus:
    """Servos follow their goal at up to 3000 ticks/s; the gripper is blocked
    at `block` (a piece, or the jaws touching each other)."""
    OPEN, CLOSE_DIR, JAWS_TOUCH = 2000, 1, 2900

    def __init__(self):
        self.pos = {n: 2048.0 for n in config.MOTOR_IDS}
        self.pos["gripper"] = self.OPEN
        self.goal = dict(self.pos)
        self.t = time.monotonic()
        self.piece = None   # gripper tick where a held piece blocks
        self.log = []
        self.torque = True  # torque off: servos don't follow their goals

    def _step(self):
        now = time.monotonic()
        dt, self.t = now - self.t, now
        if not self.torque:
            return
        for n, g in self.goal.items():
            p = self.pos[n]
            step = max(-3000 * dt, min(3000 * dt, g - p))
            new = p + step
            if n == "gripper":
                block = self.piece if self.piece is not None else self.JAWS_TOUCH
                new = min(new, block)
            self.pos[n] = new

    def read_positions(self, names=None):
        self._step()
        names = names or list(self.pos)
        return {n: int(self.pos[n]) for n in ([names] if isinstance(names, str) else names)}

    def read_position_load(self, name):
        self._step()
        p, g = self.pos[name], self.goal[name]
        load = min(1000, int(abs(g - p) * 12))   # P-controller effort
        return int(p), load

    def sync_write_positions(self, goals):
        if getattr(self, "locked", False):
            raise RuntimeError("arm is limp")
        self._step()
        self.goal.update(goals)
        if "shoulder_pan" in goals:
            self.log.append(("move", {k: int(v) for k, v in goals.items()}))

    def write_position(self, name, ticks):
        self._step()
        self.goal[name] = ticks
        if name == "gripper":
            self.log.append(("grip", int(ticks)))
        return 0

    def set_torque(self, on, names=None):
        self.torque = on

    def go_limp(self):
        self.torque = False
        self.locked = True

    def set_torque_limit(self, name, limit):
        self.limits = getattr(self, "limits", []) + [(name, int(limit))]

    def read_register(self, name, addr, nbytes=1):
        # LeRobot's gripper setup: overload threshold 25 %, protection time 2 s
        return {36: 25, 35: 200}[addr]


def make_arm():
    arm = robot.RobotArm.__new__(robot.RobotArm)
    arm.bus = FakeBus()
    arm.abort = False
    pose = lambda pan, lift: {"shoulder_pan": pan, "shoulder_lift": lift, "elbow_flex": 2048,
                              "wrist_flex": 2048, "wrist_roll": 2048, "gripper": 2000}
    cal = {s: [pose(1500 + 15 * chess.parse_square(s), 1800),
               pose(1500 + 15 * chess.parse_square(s), 2300)] for s in chess.SQUARE_NAMES}
    cal["init"] = pose(3200, 1000)       # rest: far to the side, folded low
    cal["capture"] = pose(800, 2200)
    cal["gripper"] = {"open": FakeBus.OPEN, "close_dir": FakeBus.CLOSE_DIR}
    arm.calibration = cal
    return arm


fails = 0
arm = make_arm()
arm.bus.piece = 2500                    # a piece between the jaws
arm.open_gripper()
arm.grip()
stall = arm.bus.pos["gripper"]
ok = abs(stall - 2500) < 5 and \
    abs(arm.bus.goal["gripper"] - (2500 + config.GRIPPER_SQUEEZE_TICKS)) < 30
print(f"grip: stalled at {stall:.0f}, holding goal {arm.bus.goal['gripper']:.0f} "
      f"({config.GRIPPER_SQUEEZE_TICKS} past contact)", "OK" if ok else "FAIL")
fails += not ok

# holding force: full force to squeeze, then below the servo's overload threshold
arm = make_arm()
arm._setup_gripper_force()
arm.bus.piece = 2500
arm.open_gripper()
arm.grip()
held = arm.bus.limits[-1][1]
arm.open_gripper(wait=False)
reopened = arm.bus.limits[-1][1]
ok = held == 250 - config.GRIP_OVERLOAD_MARGIN and reopened == config.GRIPPER_TORQUE_LIMIT
print(f"grip force: holding {held} (servo trips above 250), opening {reopened}",
      "OK" if ok else "FAIL")
fails += not ok

# move sequence: a capture, starting and ending at rest
arm = make_arm()
arm.bus.piece = 2500
b = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
arm.bus.log.clear()
arm.execute(b, chess.Move.from_uci("e4d5"))
# the jaws may only open at a floor pose (place) or at the capture pose
opened_midair = 0
holding, last_arm = False, None
for kind, g in arm.bus.log:
    if kind == "move":
        last_arm = g
    elif g >= FakeBus.OPEN - 5:          # an "open" command
        at_floor = last_arm is not None and last_arm["shoulder_lift"] == 1800
        at_capture = last_arm is not None and last_arm["shoulder_pan"] == 800
        if holding and not (at_floor or at_capture):
            opened_midair += 1
        holding = False
    else:
        holding = True
print(f"jaws opened while carrying a piece away from floor/capture: {opened_midair}",
      "OK" if opened_midair == 0 else "FAIL")
fails += opened_midair != 0
arm.bus.log = [e for e in arm.bus.log if e[0] == "move"]
pans = [g["shoulder_pan"] for _, g in arm.bus.log]
lifts = [g["shoulder_lift"] for _, g in arm.bus.log]
# whenever pan swings by more than the transit threshold, the arm must be at lift height
low_swings = 0
for i in range(1, len(pans)):
    if abs(pans[i] - pans[i - 1]) > 40 and lifts[i] < 2200 and lifts[i - 1] < 2200:
        low_swings += 1
ends_at_rest = pans[-1] == 3200 and lifts[-1] == 1000
print(f"capture sequence: {len(pans)} setpoints, low swings over the board: {low_swings}, "
      f"ends at rest: {ends_at_rest}", "OK" if not low_swings and ends_at_rest else "FAIL")
fails += bool(low_swings) or not ends_at_rest

# park from an arbitrary pose low over the board: must rise before swinging
arm = make_arm()
arm.bus.pos.update({"shoulder_pan": 1900, "shoulder_lift": 1500, "gripper": 2600})
arm.bus.goal = dict(arm.bus.pos)
arm.bus.log.clear()
arm.park()
pans = [g["shoulder_pan"] for k, g in arm.bus.log if k == "move"]
lifts = [g["shoulder_lift"] for k, g in arm.bus.log if k == "move"]
swing_low = any(abs(pans[i] - pans[i - 1]) > 1 and lifts[i] < 2200 and pans[i] < 3100
                for i in range(1, len(pans)))
at_rest = pans[-1] == 3200 and lifts[-1] == 1000
jaws_open = arm.bus.goal["gripper"] == FakeBus.OPEN
ok = not swing_low and at_rest and jaws_open
print(f"park from over the board: rose before swinging={not swing_low}, at rest={at_rest}, "
      f"jaws opened={jaws_open}", "OK" if ok else "FAIL")
fails += not ok

# torque on after hand-guiding must not jump back to the old goal
arm.bus.set_torque(False)                  # teaching: arm is limp
arm.bus.goal["elbow_flex"] = 1000          # stale goal from before teaching
arm.bus.pos["elbow_flex"] = 2500           # arm was moved by hand
arm.enable_torque_here()
ok = abs(arm.bus.goal["elbow_flex"] - 2500) < 2
print(f"torque on after hand-guiding keeps the arm in place: {ok}", "OK" if ok else "FAIL")
fails += not ok

# limp: torque off, and any further motion command is refused
arm = make_arm()
arm.limp()
try:
    arm.moveto(arm.calibration["e4"][1])
    refused = False
except Exception:
    refused = True
ok = refused and not arm.bus.torque
print(f"limp: torque off={not arm.bus.torque}, later motion refused={refused}", "OK" if ok else "FAIL")
fails += not ok
sys.exit(1 if fails else 0)
