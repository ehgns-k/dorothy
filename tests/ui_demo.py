"""Scripted end-to-end run of the kiosk in simulation mode.

A robot "visitor" plays random legal moves on the simulated board while the
real UI, vision pipeline and game controller run. Screenshots go to the
directory given as the first argument.

    .venv/bin/python tests/ui_demo.py /tmp/shots [plies]
"""

import os
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess  # noqa: E402
from PIL import ImageGrab  # noqa: E402

import kiosk  # noqa: E402
from game import Controller  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "shots")
plies = int(sys.argv[2]) if len(sys.argv) > 2 else 8
out.mkdir(parents=True, exist_ok=True)
kiosk.setup_logging()
ctl = Controller(sim=True)
app = kiosk.Kiosk(ctl, fullscreen=False)
app.root.attributes("-topmost", True)  # keep other windows out of the screenshots
result = {"ok": False, "human_moves": 0}


def shot(name):
    time.sleep(0.4)
    r = app.root
    x, y, w, h = r.winfo_rootx(), r.winfo_rooty(), r.winfo_width(), r.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h), xdisplay=None).save(out / f"{name}.png")


def wait_phase(phases, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if ctl.view.phase in phases:
            return ctl.view
        time.sleep(0.05)
    raise TimeoutError(f"phase {phases} not reached; at {ctl.view.phase}: {ctl.view.headline}")


def driver():
    rng = random.Random(3)
    try:
        wait_phase({"attract"})
        shot("01_attract")
        ctl.send("difficulty", 0)
        time.sleep(1.2)
        ctl.send("start")
        n = 0
        while n < plies:
            v = wait_phase({"human", "gameover"}, timeout=90)
            if v.phase == "gameover":
                break
            if n == 0:
                shot("02_human_turn")
            b = ctl.board.copy()
            mv = rng.choice(list(b.legal_moves))
            ctl.world.human_move(mv.from_square, mv.to_square, mv.promotion or chess.QUEEN)
            if n == 1:
                time.sleep(1.6)
                shot("03_detecting")
            v = wait_phase({"robot_think", "robot_move", "promotion", "gameover"}, timeout=30)
            if v.phase == "promotion":
                ctl.send("promo", chess.QUEEN)
                wait_phase({"robot_think", "robot_move", "gameover"}, timeout=30)
            got = ctl.board.move_stack[-1]
            if (got.from_square, got.to_square) != (mv.from_square, mv.to_square):
                print("MISMATCH: played", mv.uci(), "recorded", got.uci())
                result["mismatch"] = result.get("mismatch", 0) + 1
            n += 1
            if n == 2:
                wait_phase({"robot_move"})
                shot("04_robot_move")
        # an illegal / unclear situation: knock two pieces around
        wait_phase({"human"})
        sqs = [s for s in ctl.world.pieces][:2]
        ctl.world.human_move(sqs[0], chess.E4 if chess.E4 not in ctl.world.pieces else chess.D5)
        time.sleep(6.5)
        shot("05_unclear")
        result["ok"] = True
        result["human_moves"] = n
    except Exception as e:
        print("DRIVER ERROR:", e)
        shot("99_error")
    finally:
        ctl.send("quit")


threading.Thread(target=driver, daemon=True).start()
app.run()
print("RESULT", result, "moves:", " ".join(ctl.board.move_stack and
                                           [m.uci() for m in ctl.board.move_stack]))
sys.stdout.flush()
os._exit(0 if result['ok'] and not result.get('mismatch') else 1)
