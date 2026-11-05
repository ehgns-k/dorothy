"""The "Calibrate camera" button (C) in simulation: works from the start
screen, during the visitor's turn and while the robot is moving (it waits for
the arm), shows what it's doing at every step, reports the result, and
returns to the screen it interrupted. A failed search offers Try again /
Cancel and keeps the old calibration."""
import os
os.environ["DOROTHY_DEBUG_LOG"] = "0"
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import robot  # noqa: E402
from game import Controller  # noqa: E402

ctl = Controller(sim=True)
threading.Thread(target=ctl.run, daemon=True).start()
seen = []


def watch():
    last = None
    while True:
        v = ctl.view
        key = (v.phase, v.cam_cal, v.headline, v.busy)
        if key != last:
            seen.append(key)
            last = key
        time.sleep(0.01)


threading.Thread(target=watch, daemon=True).start()
fails = 0


def wait(pred, timeout=60, what=""):
    end = time.time() + timeout
    while time.time() < end:
        if pred(ctl.view):
            return ctl.view
        time.sleep(0.02)
    raise TimeoutError(f"{what}: at {ctl.view.phase} / {ctl.view.cam_cal}: {ctl.view.headline}")


def check(label, ok):
    global fails
    fails += not ok
    print(f"{label}: {'OK' if ok else 'FAIL'}")


def calibrate_and_return(where):
    before = ctl.view.phase
    n = len(seen)
    ctl.send("calibrate_camera")
    reacted = ctl.view.cam_cal == "pending"
    wait(lambda v: v.cam_cal == "done", 40, "calibrated")
    wait(lambda v: v.cam_cal == "" and v.phase == before, 20, "returned")
    states = [s for s in seen[n:]]
    ok = reacted and any(s[1] == "running" and s[3] for s in states) \
        and any(s[2] == "Camera calibrated" for s in states)
    check(f"{where}: button reacts at once, busy while looking, reports, returns to '{before}'", ok)


wait(lambda v: v.phase == "attract", 90, "start screen")
calibrate_and_return("start screen")

# camera bumped: calibration finds the shift and corrects it
ctl.world.camera_offset = np.float32([12, -7])
time.sleep(0.5)
n = len(seen)
ctl.send("calibrate_camera")
v = wait(lambda v: v.cam_cal == "done", 40, "calibrated after bump")
check(f"camera bumped 14 px -> '{v.detail}'", "shifted by" in v.detail)
wait(lambda v: v.cam_cal == "", 20)

ctl.send("difficulty", 0)
ctl.send("start")
wait(lambda v: v.phase == "human", 90, "human turn")
calibrate_and_return("visitor's turn")

# while the robot moves: queued, then runs once the arm is done
orig = robot.SimArm.execute
robot.SimArm.execute = lambda self, *a, **k: (time.sleep(2.0), orig(self, *a, **k))
mv = next(iter(ctl.board.legal_moves))
ctl.world.human_move(mv.from_square, mv.to_square)
wait(lambda v: v.phase == "robot_move", 30, "robot move")
ctl.send("calibrate_camera")
check("during the robot's move: queued (pending)", ctl.view.cam_cal == "pending")
wait(lambda v: v.cam_cal == "running", 40, "runs after the arm")
wait(lambda v: v.cam_cal == "", 30, "done")
robot.SimArm.execute = orig

# failure: board can't be found -> Try again / Cancel, old calibration kept
wait(lambda v: v.phase == "human", 60, "human turn")
corners = ctl.vision.geometry.corners.copy()
real = ctl.vision.board_status
ctl.vision.board_status = lambda *a, **k: {"kind": "lost"}
ctl.send("calibrate_camera")
v = wait(lambda v: v.cam_cal == "failed", 30, "failure shown")
check(f"failure offers {[b.label for b in v.buttons]}", [b.label for b in v.buttons] == ["Try again", "Cancel"])
ctl.send("cal_retry")
wait(lambda v: v.cam_cal == "running", 10, "retrying")
wait(lambda v: v.cam_cal == "failed", 30, "fails again")
ctl.send("cal_cancel")
v = wait(lambda v: v.cam_cal == "", 10, "cancelled")
ctl.vision.board_status = real
check("cancel returns to the visitor's turn with the old calibration",
      v.phase == "human" and np.allclose(ctl.vision.geometry.corners, corners))

ctl.send("quit")
time.sleep(0.3)
sys.stdout.flush()
os._exit(1 if fails else 0)
