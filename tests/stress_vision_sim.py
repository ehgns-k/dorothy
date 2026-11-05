"""Stress test: sensor noise, lighting swings and visitors nudging pieces."""
import random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess, numpy as np  # noqa: E402
import vision  # noqa: E402
from sim import SimWorld, SIM_CORNERS  # noqa: E402

geo = vision.BoardGeometry(SIM_CORNERS)
rng = random.Random(5)
fails = 0
for label, noise, drift, nudge in [("noisy", 8, 0.15, 0), ("nudges", 4, 0.05, 0.3)]:
    err = tot = 0; ins = []; outs = []; dt = 0
    for g in range(4):
        w = SimWorld(noise=noise, drift=drift, seed=g); b = chess.Board()
        snap = lambda: vision.crop_board(geo.warp(w.render()))
        ref = snap()
        while not b.is_game_over() and b.ply() < 70:
            mv = rng.choice(list(b.legal_moves))
            w.human_move(mv.from_square, mv.to_square, mv.promotion or chess.QUEEN); w._hand_until = 0
            w._t0 -= rng.uniform(0, 4)
            if rng.random() < nudge:
                sq = rng.choice([s for s in w.pieces if s not in (mv.from_square, mv.to_square)])
                ox, oy = w.offsets[sq]
                w.offsets[sq] = (ox + rng.choice([-6, 6]), oy + rng.choice([-6, 6])); w._tex = None
            cur = snap(); t0 = time.perf_counter()
            sc = vision.square_changes(ref, cur); inf = vision.infer_move(b, sc); dt += time.perf_counter() - t0
            tot += 1
            S = vision.changed_squares(b, mv)
            ins.append(min(sc[s] for s in S)); outs.append(max(sc[s] for s in chess.SQUARES if s not in S))
            if not (inf.status == "move" and mv in inf.moves): err += 1
            b.push(mv); ref = cur
    fails += err
    print(f"{label:7s} {tot-err}/{tot}  weakest move-square p5={np.percentile(ins,5):.1f}  "
          f"strongest other p95={np.percentile(outs,95):.1f} max={max(outs):.1f}  {1000*dt/tot:.0f} ms")
sys.exit(1 if fails else 0)
