"""
Teach the arm where the board is.

    python calibration.py            quick: FLOOR pose of 32 squares (every dark square) + capture, rest,
                                     gripper-open. Every other floor pose and ALL lift
                                     poses are computed (arm_model.py)
    python calibration.py --verify   visit every computed square to check it;
                                     press F on a square that's off to teach it
    python calibration.py --teach e4 g7    (re)teach specific squares, then refit
    python calibration.py --full     teach all 64 floor poses by hand
    python calibration.py --check    with a FULL calibration: how well would the
                                     32-square model have predicted it?

While teaching, torque is OFF - move the arm by hand:
    ENTER/SPACE = record pose     S = skip (keep existing)     Ctrl+C = stop
FLOOR pose: jaws open around the piece's base, just above the board.
Lift poses are NOT taught: they are LIFT_HEIGHT_SQUARES x SQUARE_SIZE_MM
(config.py) straight above each floor pose.
"""

import argparse
import json
import signal
import sys
import termios
import time
import tty

import chess

import arm_model
import config
from robot import ARM_JOINTS, RobotArm


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


def prompt(msg: str, allow_skip: bool = False, extra: str = "") -> str:
    hint = "  [ENTER=record" + (", S=skip" if allow_skip else "") + extra + "]"
    sys.stdout.write(msg + hint + ": ")
    sys.stdout.flush()
    while True:
        ch = getch()
        if ch in ("\r", "\n", " "):
            print("ok")
            return "record"
        if allow_skip and ch in "sS":
            print("skipped")
            return "skip"
        if extra and ch.lower() in extra.lower():
            print(ch)
            return ch.lower()


def go_rest(t: "Teacher") -> None:
    """Return to the rest pose (jaws open) once the arm is under power."""
    full, _ = arm_model.complete(t.cal)
    t.arm.calibration = full
    if "init" not in full:
        print("(no rest pose taught yet - the arm stays where it is)")
        return
    print("Returning to the rest pose...")
    t.arm.park()


class Teacher:
    def __init__(self):
        dcfg = config.load_device_config()
        self.arm = RobotArm(dcfg["robot_port"], calibration_file=config.CALIBRATION_FILE)
        self.arm.bus.connect()
        raw = json.loads(config.CALIBRATION_FILE.read_text()) \
            if config.CALIBRATION_FILE.exists() else {}
        self.cal = raw
        self.taught = set(arm_model.taught_squares(raw))

    def save(self) -> None:
        out = dict(self.cal)
        out["_taught"] = sorted(self.taught)
        config.CALIBRATION_FILE.write_text(json.dumps(out, indent=2))

    def pose(self) -> dict:
        return self.arm.positions()

    def teach(self, key: str) -> None:
        """Record a square's FLOOR pose (its lift pose is computed)."""
        a = prompt(f"  {key.upper():>3} FLOOR", allow_skip=key in self.taught)
        if a == "skip":
            return
        self.cal[key] = [self.pose()]
        self.taught.add(key)
        self.save()

    def teach_single(self, key: str, label: str) -> None:
        a = prompt(f"  {label}", allow_skip=key in self.cal)
        if a == "skip":
            return
        self.cal[key] = self.pose()
        self.save()

    def teach_gripper(self) -> None:
        """Only OPEN is recorded; a small squeeze tells us which way closes.
        Contact is sensed at run time, so no 'closed' position is needed."""
        a = prompt("  GRIPPER fully OPEN", allow_skip="gripper" in self.cal)
        if a == "skip":
            return
        opened = self.arm.bus.read_positions(["gripper"])["gripper"]
        while True:
            prompt("  now squeeze the jaws a little CLOSED (any amount)")
            now = self.arm.bus.read_positions(["gripper"])["gripper"]
            if abs(now - opened) >= 20:
                break
            print("    (hardly moved - squeeze a bit more)")
        self.cal["gripper"] = {"open": int(opened), "close_dir": 1 if now > opened else -1}
        self.save()

    def torque(self, on: bool) -> None:
        try:
            if on:
                self.arm.enable_torque_here()   # no jump back to stale goals
            else:
                self.arm.bus.set_torque(False)
        except Exception as e:
            print(f"  (torque {'on' if on else 'off'} failed: {e})")

    def fit(self) -> None:
        full, model = arm_model.complete({**self.cal, "_taught": sorted(self.taught)})
        if model is None:
            print(f"\n{len(self.taught)} squares taught - need at least "
                  f"{arm_model.MIN_ANCHORS} for the model.")
            return
        self.cal = full
        self.save()
        m = full["_model"]
        print(f"\nModel fitted from {len(self.taught)} floor poses "
              f"(robot base at {m['base_squares']} squares, square {m['square_mm']} mm).")
        print(f"Cross-validated jaw-tip error: ~{m['cv_tip_mm']} mm "
              "(below ~3 mm is good; above ~6 mm -> check with --verify and re-teach).")
        print(f"Lift height: {m['lift_mm']} mm above the board.")
        if m["lift_lowered"]:
            low = ", ".join(f"{k} {v:.0f}mm" for k, v in sorted(m["lift_lowered"].items()))
            print(f"  out of reach at that height, lowered: {low}")
        print("Run  python calibration.py --verify  to check it on the real board.")


def run_quick(t: Teacher, squares, specials: bool = True) -> None:
    print(__doc__.split("While teaching")[1])
    t.torque(False)
    try:
        print("Squares:")
        for s in squares:
            t.teach(s)
        if specials:
            print("Special poses:")
            t.teach_single("capture", "CAPTURE: where taken pieces are dropped (jaws above the tray)")
            t.teach_single("init", "REST: parked, completely out of the camera's view")
            t.teach_gripper()
    finally:
        t.torque(True)
    t.fit()
    go_rest(t)


def run_verify(t: Teacher) -> None:
    """Hover above each square; operator confirms or marks it for re-teaching."""
    full, _ = arm_model.complete(t.cal)
    t.arm.calibration = full
    t.torque(True)
    t.arm.open_gripper()   # jaws open before going near any piece
    bad = []
    print("\nThe arm visits every predicted square. Watch the jaws:")
    print("  ENTER = looks right   F = off, teach it   Q = stop\n")
    order = [s for s in chess.SQUARE_NAMES if s not in t.taught]
    for s in order:
        t.arm.moveto(full[s][1])
        t.arm.moveto(full[s][0], speed_frac=config.APPROACH_SPEED_FRAC, settle=True)
        a = prompt(f"  {s.upper()}", extra=", F=fix, Q=quit")
        t.arm.moveto(full[s][1], speed_frac=config.APPROACH_SPEED_FRAC)
        if a == "f":
            bad.append(s)
        elif a == "q":
            break
    go_rest(t)
    if bad:
        print(f"\nRe-teaching {', '.join(bad)}:")
        run_quick(t, bad, specials=False)


def run_check(t: Teacher) -> None:
    errs = arm_model.leave_one_out(t.cal, arm_model.ANCHORS)
    if not errs:
        print("Need a full calibration (all 64 squares) for --check.")
        return
    worst = sorted(errs.items(), key=lambda kv: -max(kv[1].values()))
    print(f"\nPredicting {len(errs)} squares from the {len(arm_model.ANCHORS)} anchors {arm_model.ANCHORS}:")
    for j in ARM_JOINTS:
        v = [e[j] for e in errs.values()]
        print(f"  {j:14s} mean {sum(v) / len(v):5.1f}  max {max(v):5.1f} ticks")
    print("  worst squares:", ", ".join(f"{s}({max(e.values()):.0f})" for s, e in worst[:6]))


def _raise_interrupt(signum, _frame):
    raise KeyboardInterrupt


def main() -> None:
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_interrupt)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--teach", nargs="+", metavar="SQUARE")
    args = ap.parse_args()
    t = Teacher()
    start = time.time()
    try:
        if args.check:
            run_check(t)
        elif args.verify:
            run_verify(t)
        elif args.teach:
            run_quick(t, [s.lower() for s in args.teach], specials=False)
        elif args.full:
            run_quick(t, list(chess.SQUARE_NAMES))
        else:
            run_quick(t, arm_model.ANCHORS)
    except KeyboardInterrupt:
        print("\nStopped - everything recorded so far is saved.")
    finally:
        print(f"({int(time.time() - start)} s)  Arm torque off - it is limp now.")
        t.arm.close()   # torque off + release the port, on every exit path


if __name__ == "__main__":
    main()
