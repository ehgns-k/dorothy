"""Locate the chessboard in the camera image and save board.json.

Normally NOT needed: the kiosk finds the board by itself at start-up and
re-registers when the camera moves. Use this to check the camera view or to
force a fresh setup.

    python setup_board.py            detect (pieces in the STARTING position),
                                     save, and write board_check.jpg
    python setup_board.py --rotate   turn the a1 corner by 90 deg (if a1 is wrong)
"""

import argparse
import time

import cv2
import numpy as np

import config
import vision as V


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotate", action="store_true")
    args = ap.parse_args()
    cam = V.Camera(config.load_device_config()["camera"]).start()
    try:
        if args.rotate:
            geo = V.BoardGeometry.load()
            geo = V.BoardGeometry(np.roll(geo.corners, -1, axis=0))
        else:
            time.sleep(1.0)
            c = V.find_board(cam, attempts=40)
            if c is None:
                raise SystemExit("Board not found: make sure the whole board is visible, "
                                 "evenly lit, and the pieces are in the starting position.")
            geo = V.BoardGeometry(c)
        geo.save()
        frame, _, _ = cam.read()
        V.save_home_background(frame, geo)
        img = frame.copy()
        for name, sq in (("a1", 0), ("h1", 7), ("a8", 56), ("h8", 63)):
            p = geo.square_polygon(sq).mean(axis=0).astype(int)
            cv2.putText(img, name, tuple(p - [14, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)
        cv2.polylines(img, [geo.corners.astype(np.int32)], True, (0, 255, 255), 2)
        out = config.HERE / "board_check.jpg"
        cv2.imwrite(str(out), img)
        print(f"Saved {config.BOARD_FILE.name}. Check {out.name}: a1 must be White's "
              "left-hand corner. If not, run with --rotate.")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
