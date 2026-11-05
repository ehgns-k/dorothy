"""Keyboard navigation in the kiosk (simulation mode, real key events)."""
import os
os.environ["DOROTHY_DEBUG_LOG"] = "0"   # tests: no debug recording
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kiosk  # noqa: E402
from game import Controller  # noqa: E402

kiosk.setup_logging()
ctl = Controller(sim=True)
app = kiosk.Kiosk(ctl, fullscreen=False)
app.root.attributes("-topmost", True)
res = {}


def key(k):
    done = threading.Event()

    def go():
        app.root.focus_force()
        app.root.event_generate(k, when="tail")
        done.set()
    app.root.after(0, go)
    done.wait(2)
    time.sleep(0.4)


def focused():
    nav = app._nav_buttons()
    return nav[app._focus % len(nav)][0].label if nav else None


def driver():
    try:
        while ctl.view.phase != "attract":
            time.sleep(0.1)
        time.sleep(1.5)
        res["start"] = focused()                       # the selected level
        key("c")                                       # C = calibrate camera
        res["c_label"] = app.cal_btn.cget("text")
        end = time.time() + 30
        states = set()
        while time.time() < end and ctl.view.cam_cal != "done":
            states.add(ctl.view.cam_cal)
            time.sleep(0.02)
        states.add(ctl.view.cam_cal)
        res["c_states"] = sorted(states)
        while ctl.view.cam_cal or ctl.view.phase != "attract":
            time.sleep(0.05)
        time.sleep(0.5)
        key("<Left>"); res["left"] = (focused(), ctl.difficulty_idx)
        key("<Tab>"); res["tab"] = (focused(), ctl.difficulty_idx)
        key("h"); key("h"); key("k"); res["hhk"] = (focused(), ctl.difficulty_idx)
        key("<Return>"); time.sleep(2.5)                # Enter starts the game
        res["phase"] = ctl.view.phase
        res["level"] = ctl.engine.difficulty["name"]
    finally:
        ctl.send("quit")


threading.Thread(target=driver, daemon=True).start()
app.run()
print(res)
ok = (res.get("start") == "Casual"
      and "Calibrat" in res.get("c_label", "")
      and {"running", "done"} <= set(res.get("c_states", []))
      and res.get("left") == ("Beginner", 0)
      and res.get("tab") == ("Casual", 1)
      and res.get("hhk") == ("Club", 2)          # Casual -> Beginner -> Master -> Club
      and res.get("phase") not in ("attract", None)
      and res.get("level") == "Club")
print("OK" if ok else "FAIL")
sys.stdout.flush()
os._exit(0 if ok else 1)  # skip interpreter teardown around native threads
