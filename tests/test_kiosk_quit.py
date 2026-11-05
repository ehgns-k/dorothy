"""Ctrl+Shift+Q must close the kiosk quickly even while the game is busy
(here: the simulated robot in the middle of a move)."""
import os
os.environ["DOROTHY_DEBUG_LOG"] = "0"   # tests: no debug recording
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
probe = r'''
import sys, threading, time
sys.path.insert(0, %r)
import kiosk, robot
from game import Controller
kiosk.setup_logging()
slow = robot.SimArm.execute
def very_slow(self, *a, **k):          # a robot move that takes "forever"
    time.sleep(30)
robot.SimArm.execute = very_slow
ctl = Controller(sim=True)
app = kiosk.Kiosk(ctl, fullscreen=False)
def driver():
    while ctl.view.phase != "attract": time.sleep(0.1)
    time.sleep(1); ctl.send("start")
    while ctl.view.phase != "robot_move": time.sleep(0.1)
    time.sleep(1)
    t0 = time.time()
    def press():
        app.root.focus_force()
        app.root.event_generate("<Control-Shift-KeyPress-Q>", when="tail")
    app.root.after(0, press)
    print("PRESSED", flush=True)
threading.Thread(target=driver, daemon=True).start()
try:
    app.run()
finally:
    print("CLOSED", flush=True)
    ctl.shutdown()
''' % str(ROOT)
p = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-c", probe], stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, cwd=ROOT)
pressed = None
for line in p.stdout:
    if line.startswith("PRESSED"):
        pressed = time.time()
    if line.startswith("CLOSED"):
        dt = time.time() - pressed if pressed else None
        print(f"window closed {dt:.1f} s after Ctrl+Shift+Q (robot was mid-move)")
        p.kill()
        sys.exit(0 if dt is not None and dt < 6 else 1)
print("FAIL: kiosk never closed")
sys.exit(1)
