"""Board / camera movement: re-registration and 'push the board back' logic."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess, numpy as np  # noqa: E402
import vision as V  # noqa: E402
from sim import SIM_CORNERS, SimCamera, SimWorld  # noqa: E402

V.config.BOARD_FIXED_TO_ROBOT = False  # exercise the opt-in "board may move" mode
V.HOME_BG_FILE = Path("/tmp") / f"sim_home_{time.time():.0f}.png" if len(sys.argv) < 2 else Path(sys.argv[1])

def setup(midgame: bool):
    w = SimWorld(seed=1); w._hand_until = 0
    if midgame:  # scatter pieces so the grid is less visible
        b = chess.Board()
        for uci in "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6 g1f3 c8f5 f1c4 e7e6".split():
            mv = chess.Move.from_uci(uci); w.human_move(mv.from_square, mv.to_square)
        w._hand_until = 0
    cam = SimCamera(w).start()
    geo = V.BoardGeometry(SIM_CORNERS)
    v = V.Vision(cam, geo).start(); time.sleep(0.6)
    V.save_home_background(cam.read()[0], geo)
    v.set_reference()
    return w, cam, v

fails = 0
for midgame in (False, True):
    for case, (cam_off, board_off), want in [
            ("still", ((0, 0), (0, 0)), "ok"),
            ("camera bumped", ((14, -9), (0, 0)), "camera_moved"),
            ("board slid", ((0, 0), (-16, 11)), "board_moved"),
            ("both", ((10, 4), (-12, 0)), "board_moved")]:
        w, cam, v = setup(midgame)
        w.camera_offset = np.float32(cam_off); w.board_offset = np.float32(board_off)
        time.sleep(0.5)
        t = time.time(); st = v.board_status(); dt = time.time() - t
        expect_corners = np.float32(SIM_CORNERS) + np.float32(cam_off) + np.float32(board_off)
        err = float(np.abs(v.geometry.corners - expect_corners).max())
        # after re-registration the reference must still explain the board
        inf = v.infer(chess.Board()) if not midgame else None
        ok = st["kind"] == want and err < 3.0 and (inf is None or inf.status == "still")
        fails += not ok
        print(f"{"midgame" if midgame else "start  "} {case:14s} -> {st['kind']:13s} "
              f"corner err {err:4.1f}px  {dt*1000:4.0f} ms  {'OK' if ok else 'FAIL'}")
        v.stop(); cam.stop()
sys.stdout.flush()
os._exit(1 if fails else 0)   # skip teardown: camera threads may still be inside OpenCV
