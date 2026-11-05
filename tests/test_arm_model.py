"""Reduced calibration vs. a synthetic SO-101-like arm.

Teach only the anchor squares (with hand-teaching noise and a gripper held at
varying angles), predict the rest, and measure the jaw-tip error in mm with
the true forward kinematics."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess  # noqa: E402
import numpy as np  # noqa: E402

import arm_model as AM  # noqa: E402
import config  # noqa: E402

K = 4096 / (2 * math.pi)
# true arm = SO-101 geometry (URDF) with the configured jaw length
L1, L2, L3 = AM.SO101_UPPER_ARM_MM, AM.SO101_FOREARM_MM, AM.tool_length_mm()
R0, H0 = AM.SO101_SHOULDER_OFFSET_MM, 95.0   # shoulder axis: ahead of the pan axis / above the board
JAW = 9.0                             # jaw tip sideways from the arm plane
SQ = 37.5                             # square size (mm)
BASE = np.array([4.3, -3.0])          # pan axis in squares (like the real setup)
SIGNS = (1.0, -1.0, 1.0)              # servo directions differ per joint
OFF = {"shoulder_pan": 2031, "shoulder_lift": 1987, "elbow_flex": 2110,
       "wrist_flex": 2050, "wrist_roll": 2040}
FLOOR_Z, LIFT_Z = 12.0, 80.0


def ik(x, y, z, pitch_deg):
    dx, dy = (x - BASE[0]) * SQ, (y - BASE[1]) * SQ
    rho = math.hypot(dx, dy)
    reach = math.sqrt(rho ** 2 - JAW ** 2)
    pan = math.atan2(dy, dx) - math.asin(JAW / rho)
    ph = math.radians(pitch_deg)
    wx, wz = reach - R0 - L3 * math.cos(ph), z - H0 - L3 * math.sin(ph)
    c = (wx ** 2 + wz ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    rel = -math.acos(max(-1, min(1, c)))            # elbow up
    th1 = math.atan2(wz, wx) - math.atan2(L2 * math.sin(rel), L1 + L2 * math.cos(rel))
    th2 = th1 + rel
    t = {"shoulder_pan": OFF["shoulder_pan"] + K * pan,
         "shoulder_lift": OFF["shoulder_lift"] + SIGNS[0] * K * th1,
         "elbow_flex": OFF["elbow_flex"] + SIGNS[1] * K * rel,
         "wrist_flex": OFF["wrist_flex"] + SIGNS[2] * K * (ph - th2),
         "wrist_roll": OFF["wrist_roll"] + K * pan * 0.3,
         "gripper": 2200.0}
    return t


def fk(t):
    pan = (t["shoulder_pan"] - OFF["shoulder_pan"]) / K
    th1 = SIGNS[0] * (t["shoulder_lift"] - OFF["shoulder_lift"]) / K
    th2 = th1 + SIGNS[1] * (t["elbow_flex"] - OFF["elbow_flex"]) / K
    th3 = th2 + SIGNS[2] * (t["wrist_flex"] - OFF["wrist_flex"]) / K
    r = R0 + L1 * math.cos(th1) + L2 * math.cos(th2) + L3 * math.cos(th3)
    z = H0 + L1 * math.sin(th1) + L2 * math.sin(th2) + L3 * math.sin(th3)
    # jaw tip: r along the arm plane, JAW to the side
    a = pan + math.atan2(-JAW, r) * -1
    rho = math.hypot(r, JAW)
    return np.array([rho * math.cos(a), rho * math.sin(a), z])


rng = np.random.default_rng(0)


def pitch_for(name, wobble):
    # a teacher holds the gripper roughly 70-100 deg down, not consistently
    # like the real teaching data: steep near the robot, ~-40 deg far away
    x, y = AM.square_xy(name)
    reach = math.hypot((x - BASE[0]) * SQ, (y - BASE[1]) * SQ)
    return -90 + 72 * (reach - 130) / 290 + rng.normal(0, wobble)


def teach(names, noise, wobble):
    """Floor poses only - lifts are computed by the model."""
    return {s: [{j: v + (rng.normal(0, noise) if j != "gripper" else 0)
                 for j, v in ik(*AM.square_xy(s), FLOOR_Z, pitch_for(s, wobble)).items()}]
            for s in names}


def tip_target(name, z):
    x, y = AM.square_xy(name)
    return ik(x, y, z, pitch_for(name, 0.0))


fails = 0
COUNTS = [int(a) for a in sys.argv[1:]] or [3, 6, 12, 32]
cases = [(f"{n} floor poses, square size given", AM.anchor_squares(n), SQ) for n in COUNTS]
cases.append(("32 floor poses, square size estimated", AM.anchor_squares(32), None))
for label, anchors, sq_mm in cases:
    cal = teach(anchors, 4.0, 8.0)
    config.SQUARE_SIZE_MM = sq_mm
    full, model = AM.complete(cal)
    m = full["_model"]
    floor_err, lift_xy, lift_z = [], [], []
    for s in chess.SQUARE_NAMES:
        tip_floor = fk(full[s][0])
        if s not in anchors:
            floor_err.append(np.linalg.norm(tip_floor - fk(tip_target(s, FLOOR_Z))))
        tip_lift = fk(full[s][1])
        # lift: straight above the (actual) floor tip, at the requested height
        lift_xy.append(np.linalg.norm(tip_lift[:2] - tip_floor[:2]))
        want_h = m["lift_lowered"].get(s, m["lift_mm"])
        lift_z.append(abs((tip_lift[2] - tip_floor[2]) - want_h))
    fe = np.array(floor_err or [0.0])   # 64 taught: nothing computed
    print(f"{label:40s} floor tip mean {fe.mean():4.1f} max {fe.max():4.1f} mm | "
          f"lift off-vertical max {max(lift_xy):4.1f} mm, height err max {max(lift_z):4.1f} mm"
          f" | square={m['square_mm']}mm, lift={m['lift_mm']}mm, "
          f"lowered on {len(m['lift_lowered'])} squares")
    # floor poses must be precise; computed lifts only need to be roughly
    # vertical (floor-only data can't pin the shoulder zero exactly).
    # Below 12 squares accuracy is reported, not required.
    if len(anchors) >= 12 and (fe.size and fe.max() > 0.25 * SQ or max(lift_xy) > 14
                               or max(lift_z) > 14):
        fails += 1
sys.exit(1 if fails else 0)
