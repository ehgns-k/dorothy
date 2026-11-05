"""Game flow in simulation (no window): the situations a visitor creates.

1. The visitor answers while the robot is still moving: once the arm is back,
   both moves are recognised and the robot carries straight on.
2. The board is wrong after a robot move: ONE "OK" button; pressing it shows
   "Checking the board..." (busy) and the game continues once it's fixed.
3. Still wrong after OK: the prompt says so; a second OK carries on anyway.
"""
import os
os.environ["DOROTHY_DEBUG_LOG"] = "0"
import random  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess  # noqa: E402

import config  # noqa: E402
import robot  # noqa: E402

from game import Controller  # noqa: E402

ctl = Controller(sim=True)
threading.Thread(target=ctl.run, daemon=True).start()
rng = random.Random(4)
fails = 0
seen_phases = []


def watch():
    last = None
    while True:
        v = ctl.view
        key = (v.phase, v.headline, v.busy, tuple(b.label for b in v.buttons))
        if key != last:
            seen_phases.append(key)
            last = key
        time.sleep(0.01)


threading.Thread(target=watch, daemon=True).start()


def wait(pred, timeout=60, what=""):
    end = time.time() + timeout
    while time.time() < end:
        if pred(ctl.view):
            return ctl.view
        time.sleep(0.03)
    raise TimeoutError(f"{what}: stuck at {ctl.view.phase}: {ctl.view.headline}")


def human_plays():
    v = wait(lambda v: v.phase == "human", what="human turn")
    mv = rng.choice(list(ctl.board.legal_moves))
    ctl.world.human_move(mv.from_square, mv.to_square, mv.promotion or chess.QUEEN)
    wait(lambda v: v.phase in ("robot_think", "robot_move"), what="move detected")
    return mv


def check(label, ok):
    global fails
    fails += not ok
    print(f"{label}: {'OK' if ok else 'FAIL'}")


wait(lambda v: v.phase == "attract", 90, "start screen")
ctl.send("difficulty", 0)
time.sleep(0.3)
ctl.send("start")

# --- 1. visitor replies while the arm is still over the board -----------
orig = robot.SimArm.execute
reply = {}


def execute_and_visitor_replies(self, board, move, progress=None):
    orig(self, board, move, progress)
    b = board.copy()
    b.push(move)
    mv = rng.choice(list(b.legal_moves))
    reply["move"] = mv
    self.world.set_arm_visible(True)        # arm still in view while they move
    self.world.human_move(mv.from_square, mv.to_square, mv.promotion or chess.QUEEN)
    time.sleep(0.5)
    self.world.set_arm_visible(False)


wait(lambda v: v.phase == "human", 60, "first human turn")
robot.SimArm.execute = execute_and_visitor_replies   # before the robot's turn starts
n0 = len(ctl.board.move_stack) + 1
human_plays()
wait(lambda v: v.phase == "robot_move", what="robot move")
robot.SimArm.execute = orig
v = wait(lambda v: v.phase in ("human_done", "help", "human"), what="after robot move")
ok = v.phase == "human_done" and "while I was moving" in v.detail
wait(lambda v: len(ctl.board.move_stack) >= n0 + 3 or v.phase == "help", 40, "robot carries on")
stack = ctl.board.move_stack
ok = ok and len(stack) >= n0 + 2 and stack[n0 + 1] == reply["move"]
check(f"reply during robot move ({reply['move'].uci()}) recognised, robot carried on", ok)

# --- 2. board wrong after a robot move; fixed; ONE OK button ----------------
def execute_and_knock(self, board, move, progress=None):
    orig(self, board, move, progress)
    # a piece far from the move gets knocked onto an empty square
    occupied = set(self.world.pieces)
    src = next(s for s in sorted(occupied, reverse=True)
               if s not in (move.from_square, move.to_square)
               and chess.square_distance(s, move.to_square) > 2)
    dst = next(s for s in range(16, 48) if s not in occupied
               and chess.square_distance(s, move.to_square) > 2)
    knocked.update(src=src, dst=dst)
    self.world.transport(chess.square_name(src), chess.square_name(dst))


knocked = {}
robot.SimArm.execute = execute_and_knock     # before the robot's turn starts
human_plays()
v = wait(lambda v: v.phase == "help", 60, "board-wrong prompt")
robot.SimArm.execute = orig
labels = [b.label for b in v.buttons]
check(f"board-wrong prompt has one button {labels}", labels == ["OK"])
ctl.world.transport(chess.square_name(knocked["dst"]), chess.square_name(knocked["src"]))  # fix it
n_seen = len(seen_phases)
ctl.send("ok")
v = wait(lambda v: v.phase == "human", 30, "game continues after the fix")
after = seen_phases[n_seen:]
check("pressing OK shows 'Checking the board...' (busy)",
      any(p[0] == "checking" and p[2] for p in after))

# --- 3. still wrong after OK: told so; second OK carries on -----------------
robot.SimArm.execute = execute_and_knock
mv = rng.choice(list(ctl.board.legal_moves))
ctl.world.human_move(mv.from_square, mv.to_square, mv.promotion or chess.QUEEN)
v = wait(lambda v: v.phase == "help", 60, "board-wrong prompt")
robot.SimArm.execute = orig
ctl.send("ok")                                   # ...without fixing anything
v = wait(lambda v: v.phase == "help" and v.headline == "Still looks different", 30, "still wrong")
check("not fixed: says it still looks different", True)
ctl.send("ok")
v = wait(lambda v: v.phase in ("human", "human_done"), 30, "carries on after the 2nd OK")
check("second OK carries on", True)

ctl.send("quit")
time.sleep(0.5)
print("moves:", " ".join(m.uci() for m in ctl.board.move_stack))
sys.stdout.flush()
os._exit(1 if fails else 0)  # skip interpreter teardown around native threads
