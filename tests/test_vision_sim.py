"""Self-test: play random games on the simulated board and check that the
vision pipeline recovers every move from images alone - the way the kiosk
does it: piece lean and the empty / white / black colour model are learned
from the starting position, and the colour model is retrained after every
move.

Also checks the "visitor already moved while the robot was busy" case: the
robot's move and the visitor's reply are recognised together.

    .venv/bin/python tests/test_vision_sim.py [n_games]
"""

import collections
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess  # noqa: E402

import vision as V  # noqa: E402
from sim import SIM_CORNERS, SimWorld  # noqa: E402

SHAPE = (720, 1280)


class Pipeline:
    def __init__(self, world: SimWorld):
        self.w = world
        self.geo = V.BoardGeometry(SIM_CORNERS)
        img = self.snap()
        k = V.fit_lean(img, chess.Board(), self.geo, SHAPE)[0]
        self.centres = V.square_centres(V.lean_offsets(self.geo, k, SHAPE))
        self.colour = V.ColourModel()
        self.colour.add(V.square_features(V._lab(img), self.centres),
                        V.board_states(chess.Board()), start=True)
        self.ref = img
        self.k = k

    def snap(self):
        self.w._hand_until = 0
        return self.geo.warp(self.w.render())

    def infer(self, board, lines=None):
        a = V.analyze(self.ref, self.snap(), self.centres)
        post = self.colour.posterior(a.features, a.l_shift)
        lines = lines or [(m,) for m in board.legal_moves]
        return V.infer_lines(board, lines, a.scores, post), a

    def accept(self, board, a):
        self.ref = self.snap()
        self.colour.add(a.features, V.board_states(board))


def pick(rng, board):
    moves = list(board.legal_moves)
    # prefer "interesting" moves so captures/castling/ep get covered
    special = [m for m in moves if board.is_capture(m) or board.is_castling(m) or m.promotion]
    return rng.choice(special if special and rng.random() < 0.6 else moves)


def kind(board, mv):
    return ("castle" if board.is_castling(mv) else "en passant" if board.is_en_passant(mv)
            else "capture" if board.is_capture(mv) else "promotion" if mv.promotion else "quiet")


def run(n_games: int = 3, max_plies: int = 70, seed: int = 1) -> int:
    rng = random.Random(seed)
    errors = 0
    for style, lean, jitter in (("flat", False, 5), ("real", True, 12)):
        stats = collections.Counter()
        t_infer, total, ks = 0.0, 0, []
        for g in range(n_games):
            world = SimWorld(seed=seed + g, style=style, lean=lean, place_jitter=jitter)
            p = Pipeline(world)
            ks.append(p.k)
            board = chess.Board()
            while not board.is_game_over() and board.ply() < max_plies:
                pair = board.ply() % 7 == 3 and board.ply() + 2 < max_plies
                mv = pick(rng, board)
                lines = None
                if pair:   # robot moves; the visitor answers before the camera looks
                    nb = board.copy(stack=False)
                    nb.push(mv)
                    if nb.is_game_over():
                        pair = False
                    else:
                        reply = pick(rng, nb)
                        lines = [(mv,)] + [(mv, r) for r in nb.legal_moves]
                for m in ([mv, reply] if pair else [mv]):
                    world.human_move(m.from_square, m.to_square, m.promotion or chess.QUEEN)
                world._t0 -= rng.uniform(0, 3)  # lighting drift between plies
                t0 = time.perf_counter()
                inf, a = p.infer(board, lines)
                t_infer += time.perf_counter() - t0
                total += 1
                want = (mv, reply) if pair else (mv,)
                ok = inf.status == "move" and want in inf.lines
                label = "robot+reply" if pair else kind(board, mv)
                stats[label, ok] += 1
                if not ok:
                    errors += 1
                    print(f"  {style} game {g} ply {board.ply()}: played "
                          f"{' '.join(m.uci() for m in want)} -> {inf.status} "
                          f"{[' '.join(x.uci() for x in ln) for ln in inf.lines[:3]]} "
                          f"changed={[chess.square_name(s) for s in inf.changed]} {inf.note}")
                for m in want:
                    board.push(m)
                p.accept(board, a)
        labels = sorted({k for k, _ in stats})
        print(f"{style:4s} board{' + piece lean' if lean else '':13s}: " +
              ", ".join(f"{k} {stats[k, True]}/{stats[k, True] + stats[k, False]}" for k in labels) +
              f"  ({1000 * t_infer / max(1, total):.0f} ms per inference, learned lean {ks})")
    return errors


if __name__ == "__main__":
    sys.exit(1 if run(int(sys.argv[1]) if len(sys.argv) > 1 else 3) else 0)
