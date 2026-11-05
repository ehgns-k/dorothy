"""Stockfish wrapper with difficulty presets, eval for the UI and
self-healing restarts."""

import logging
import math
import random
import threading
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine

import config

log = logging.getLogger("engine")

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class Engine:
    def __init__(self, path: Optional[str] = None, allow_fallback: bool = False):
        self.path = path
        self.allow_fallback = allow_fallback
        self._eng: Optional[chess.engine.SimpleEngine] = None
        self._lock = threading.Lock()
        self.difficulty = config.DIFFICULTIES[config.DEFAULT_DIFFICULTY]
        self.name = "?"
        self.rng = random.Random()
        self.last_choice: Dict = {}     # why the last move was chosen (debug log)

    def start(self) -> None:
        try:
            path = self.path or config.find_engine()
            self._eng = chess.engine.SimpleEngine.popen_uci(path)
            self.name = self._eng.id.get("name", "Stockfish")
            opts = self._eng.options
            base = {"Threads": config.ENGINE_THREADS, "Hash": config.ENGINE_HASH_MB}
            self._eng.configure({k: v for k, v in base.items() if k in opts})
            log.info("engine: %s", self.name)
        except Exception:
            if not self.allow_fallback:
                raise
            log.warning("Stockfish unavailable - using the built-in toy engine")
            self._eng, self.name = None, "toy engine"

    def quit(self) -> None:
        if self._eng is not None:
            try:
                self._eng.quit()
            except Exception:
                pass
            self._eng = None

    def set_difficulty(self, idx: int) -> None:
        self.difficulty = config.DIFFICULTIES[max(0, min(idx, len(config.DIFFICULTIES) - 1))]

    def _options(self) -> dict:
        d, opts = self.difficulty, self._eng.options
        o = {}
        if "UCI_LimitStrength" in opts:
            o["UCI_LimitStrength"] = d.get("elo") is not None
            if d.get("elo") is not None:
                lo, hi = opts["UCI_Elo"].min, opts["UCI_Elo"].max
                o["UCI_Elo"] = max(lo, min(hi, int(d["elo"])))
        if "Skill Level" in opts:
            o["Skill Level"] = 20 if d.get("skill") is None else int(d["skill"])
        return o

    def _call(self, fn):
        """Run fn(engine); restart the engine once if it died."""
        with self._lock:
            for attempt in range(2):
                try:
                    return fn(self._eng)
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError,
                        BrokenPipeError, OSError) as e:
                    log.warning("engine failure (%s) - restarting", e)
                    self.quit()
                    self.start()
            raise RuntimeError("chess engine keeps failing")

    def best_move(self, board: chess.Board) -> chess.Move:
        if self._eng is None:
            return self._toy_move(board)
        if "depth" in self.difficulty:
            return self._call(lambda eng: self._human_move(eng, board))

        def go(eng):
            eng.configure(self._options())
            res = eng.play(board, chess.engine.Limit(time=self.difficulty["time"]))
            if res.move is None:
                raise chess.engine.EngineError("no move")
            return res.move
        return self._call(go)

    # ------------------------------------------------------------------ #
    # human-like play
    # ------------------------------------------------------------------ #
    def _human_move(self, eng, board: chess.Board) -> chess.Move:
        """Pick a move the way a person of this level might (see the
        DIFFICULTIES comment in config.py)."""
        d, rng = self.difficulty, self.rng
        full = {"UCI_LimitStrength": False} if "UCI_LimitStrength" in eng.options else {}
        if "Skill Level" in eng.options:
            full["Skill Level"] = 20
        eng.configure(full)
        infos = eng.analyse(board, chess.engine.Limit(depth=int(d["depth"])),
                            multipv=int(d.get("multipv", 8)))
        cands: List[Tuple[chess.Move, int]] = []
        for info in infos:
            if info.get("pv"):
                cands.append((info["pv"][0], info["score"].relative.score(mate_score=100000)))
        if not cands:
            raise chess.engine.EngineError("no move")
        best = max(cp for _, cp in cands)

        # a mate in one is usually spotted (not always by beginners)
        mates = [m for m, _ in cands if self._mates(board, m)]
        if mates and rng.random() < d.get("mate_in_one", 1.0):
            return self._chosen(mates[0], "mate in one", cands, best)

        temp = float(d.get("temp_cp", 50))
        max_loss = float(d.get("max_loss_cp", 300))
        blunder = float(d.get("blunder", 0.0))
        if d.get("adapt"):
            # keep the game close: relax when far ahead, focus when behind
            if best > 700:
                temp, max_loss, blunder = temp * 2.2, max_loss * 2.0, blunder * 2.0
            elif best > 300:
                temp, max_loss, blunder = temp * 1.6, max_loss * 1.5, blunder * 1.5
            elif best < -150:
                temp, max_loss, blunder = temp * 0.6, max_loss * 0.6, blunder * 0.4
        if board.ply() < 10:
            temp = max(temp, 30.0)   # openings vary from game to game
        if abs(best) >= 90000:
            temp, blunder = min(temp, 40.0), 0.0   # mates in sight: stay on track

        bonus_scale = float(d.get("human_bonus", 0))
        if rng.random() < blunder:
            # a clearly worse but natural-looking move (hung piece, missed threat)
            pool = [(m, cp) for m, cp in cands if 120 <= best - cp <= 900]
            if pool:
                w = [math.exp(bonus_scale * self._naturalness(board, m) / 60.0) for m, _ in pool]
                m = rng.choices([m for m, _ in pool], weights=w)[0]
                return self._chosen(m, "blunder", cands, best)
        options, weights = [], []
        for m, cp in cands:
            loss = best - cp
            if loss > max_loss:
                continue
            adj = loss - bonus_scale * self._naturalness(board, m)
            options.append(m)
            weights.append(math.exp(-max(-300.0, adj) / temp))
        if not options:
            options, weights = [cands[0][0]], [1.0]
        m = rng.choices(options, weights=weights)[0]
        return self._chosen(m, "normal", cands, best)

    def _chosen(self, move, why, cands, best) -> chess.Move:
        self.last_choice = {"level": self.difficulty["name"], "why": why, "move": move.uci(),
                            "best_cp": best,
                            "candidates": [(m.uci(), cp) for m, cp in cands]}
        log.info("%s move %s (%s; best %+d cp; options %s)", self.difficulty["name"],
                 move.uci(), why, best, " ".join(f"{m.uci()}:{cp:+d}" for m, cp in cands[:6]))
        return move

    @staticmethod
    def _mates(board: chess.Board, m: chess.Move) -> bool:
        board.push(m)
        try:
            return board.is_checkmate()
        finally:
            board.pop()

    @staticmethod
    def _naturalness(board: chess.Board, m: chess.Move) -> float:
        """How 'obvious' a move looks to a person, roughly -1..1."""
        n = 0.0
        piece = board.piece_at(m.from_square)
        last = board.peek() if board.move_stack else None
        if board.is_capture(m):
            n += 0.4
            if last is not None and m.to_square == last.to_square:
                n += 0.5                        # recapture
        if board.gives_check(m):
            n += 0.3
        if board.is_castling(m):
            n += 0.6
        if piece is not None:
            back = 0 if piece.color == chess.WHITE else 7
            fwd = chess.square_rank(m.to_square) - chess.square_rank(m.from_square)
            if piece.color == chess.BLACK:
                fwd = -fwd
            if board.ply() < 20 and piece.piece_type in (chess.KNIGHT, chess.BISHOP) and \
                    chess.square_rank(m.from_square) == back:
                n += 0.4                        # develop the pieces
            if piece.piece_type == chess.KING and not board.is_castling(m) and \
                    board.ply() < 50 and board.queens:
                n -= 0.6                        # people don't walk their king early
            if fwd < 0 and not board.is_capture(m):
                n -= 0.2                        # retreats are harder to see
            if board.ply() < 16 and piece.piece_type == chess.QUEEN:
                n -= 0.3                        # early queen trips
        return n

    def evaluate(self, board: chess.Board) -> Optional[float]:
        """Evaluation in pawns from White's view (+-100 for mate)."""
        if board.is_game_over():
            r = board.result()
            return 100.0 if r == "1-0" else -100.0 if r == "0-1" else 0.0
        if self._eng is None:
            return float(sum(PIECE_VALUE[p.piece_type] * (1 if p.color else -1)
                             for p in board.piece_map().values()))

        def go(eng):
            eng.configure({"UCI_LimitStrength": False} if "UCI_LimitStrength" in eng.options else {})
            info = eng.analyse(board, chess.engine.Limit(time=config.EVAL_TIME_S))
            sc = info["score"].white()
            return 100.0 if sc.is_mate() and sc.mate() > 0 else \
                -100.0 if sc.is_mate() else sc.score() / 100.0
        try:
            return self._call(go)
        except Exception:
            return None

    @staticmethod
    def _toy_move(board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)

        def gain(m):
            v = 0
            if board.is_capture(m):
                cap = board.piece_at(m.to_square)
                v += PIECE_VALUE[cap.piece_type] if cap else 1
            if board.gives_check(m):
                v += 0.5
            if m.promotion:
                v += 8
            return v + random.random() * 0.3
        return max(moves, key=gain)
