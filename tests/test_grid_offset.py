"""Board detector: a grid that latched on one square off is put back.

The chessboard corner detector sees only a 7x7 lattice of inner corners;
with pieces hiding an edge row it can pick a lattice shifted by one square.
`vision.grid_offset` must find the real 8x8 board from the checker pattern.
Every shift by -1/0/+1 square in x and y is tried on the simulated board
(flat and realistic pieces, starting position and an empty board).

    .venv/bin/python tests/test_grid_offset.py
"""
import os
os.environ["DOROTHY_DEBUG_LOG"] = "0"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import vision as V  # noqa: E402
from sim import SIM_CORNERS, SimWorld  # noqa: E402

# grid units -> image (a1, h1, h8, a8 = (0,8), (8,8), (8,0), (0,0))
H = cv2.getPerspectiveTransform(np.float32([[0, 8], [8, 8], [8, 0], [0, 0]]),
                                np.float32(SIM_CORNERS))
fails = 0
for style in ("flat", "real"):
    for empty in (False, True):
        w = SimWorld(seed=3, style=style, lean=style == "real")
        if empty:
            w.pieces.clear()
        frame = w.render()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                Hs = H @ np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
                got = V.grid_offset(frame, Hs)
                ok = got == (-dx, -dy)
                fails += not ok
                if not ok:
                    print(f"  {style} {'empty' if empty else 'start'}: grid shifted "
                          f"({dx},{dy}) -> correction {got}, want {(-dx, -dy)}")
print("OK" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
