"""
Chess robot framework - main game loop.

Robot turn: Stockfish picks a move, RobotMotion executes it; the next-turn
occupancy baseline is set from chess.Board.
Human turn: user moves pieces, presses <ENTER>/<SPACE>; vision captures
occupancy across DETECTION_ITERATIONS reads, diffs against the baseline,
and auto-confirms a unique candidate (or prompts to pick among multiple).

First-time setup:
  python find_port.py     -> writes device_config.json
  python calibration.py   -> writes calibration.json
"""

import json
import msvcrt
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.engine

from module import ChessVision, RobotMotion, DEVICE_CONFIG_FILE


ROBOT_COLOR = chess.WHITE       # chess.WHITE (robot first) or chess.BLACK
ENGINE_PATH = "C:\\Users\\hhung\\CNSCI\\stockfish\\stockfish-windows-x86-64-avx2.exe"
ENGINE_THINK_TIME_S = 0.000005
DETECTION_ITERATIONS = 5        # camera reads per <ENTER>/<SPACE>; unioned

CONFIRM_KEYS = {"\r", "\n", " "}
SPECIAL_KEYS = {"B", "C", "I", "T", "Q"}
SEPARATOR = "=" * 48


class Settings:
    """Runtime-tunable knobs adjusted via in-game hotkeys."""
    def __init__(self):
        self.iterations: int = DETECTION_ITERATIONS


def wait_for_key(allowed: Optional[set] = None) -> str:
    """Block on a single keypress (uppercased for letters)."""
    while True:
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):    # arrow/function-key prefix
            msvcrt.getch()
            continue
        if ch == b"\x03":               # Ctrl-C
            raise KeyboardInterrupt
        try:
            c = ch.decode("utf-8")
        except UnicodeDecodeError:
            continue
        c = c.upper() if c.isalpha() else c
        if allowed is None or c in allowed:
            return c


def infer_move_candidates(board: chess.Board,
                          departed: set, arrived: set) -> list:
    """Return legal chess.Moves compatible with an occupancy diff."""
    legal = list(board.legal_moves)

    if len(departed) == 1 and len(arrived) == 1:
        src = chess.parse_square(next(iter(departed)))
        dst = chess.parse_square(next(iter(arrived)))
        return [m for m in legal if m.from_square == src and m.to_square == dst]

    if len(departed) == 1 and len(arrived) == 0:
        # Capture: dst was already occupied so it doesn't appear in `arrived`.
        src = chess.parse_square(next(iter(departed)))
        return [m for m in legal
                if m.from_square == src and board.is_capture(m)
                and not board.is_en_passant(m)]

    if len(departed) == 2 and len(arrived) == 1:
        dst = chess.parse_square(next(iter(arrived)))
        return [m for m in legal if m.to_square == dst and board.is_en_passant(m)]

    if len(departed) == 2 and len(arrived) == 2:
        cands = [m for m in legal if board.is_castling(m)]
        dep_files = {chess.square_file(chess.parse_square(s)) for s in departed}
        if 7 in dep_files:
            return [m for m in cands if board.is_kingside_castling(m)]
        return [m for m in cands if board.is_queenside_castling(m)]

    return []


def attach_promotion(move: chess.Move, board: chess.Board) -> chess.Move:
    """Prompt for promotion piece if this is a pawn reaching the last rank."""
    if move.promotion is not None:
        return move
    if board.piece_type_at(move.from_square) != chess.PAWN:
        return move
    dst_rank = chess.square_rank(move.to_square)
    if not ((board.turn == chess.WHITE and dst_rank == 7) or
            (board.turn == chess.BLACK and dst_rank == 0)):
        return move

    print("  Pawn promotion! Choose piece:")
    print("    1: Queen    2: Rook    3: Bishop    4: Knight")
    promo_map = {"1": chess.QUEEN, "2": chess.ROOK,
                 "3": chess.BISHOP, "4": chess.KNIGHT}
    k = wait_for_key(set(promo_map))
    return chess.Move(move.from_square, move.to_square, promotion=promo_map[k])


def pick_from_candidates(candidates: list, board: chess.Board) -> Optional[chess.Move]:
    print("\n  Multiple possible moves detected:")
    for i, m in enumerate(candidates, 1):
        print(f"    {i}: {board.san(m)} ({m.uci()})")
    print("  Press a number to select, <R> to retry.")
    allowed = {str(i) for i in range(1, len(candidates) + 1)} | {"R"}
    k = wait_for_key(allowed)
    if k == "R":
        return None
    return candidates[int(k) - 1]


def attempt_human_move(vision: ChessVision, board: chess.Board,
                       iterations: int) -> list:
    """N occupancy reads; return the deduplicated union of detected legal moves."""
    seen: dict = {}
    for i in range(1, iterations + 1):
        occ = vision.detect_occupancy()
        if occ is None:
            print(f"  [{i}/{iterations}] reading rejected (legality)")
            continue

        departed, arrived = vision.get_occupancy_diff()
        if not departed and not arrived:
            print(f"  [{i}/{iterations}] no change vs baseline")
        else:
            candidates = infer_move_candidates(board, departed, arrived)
            if candidates:
                new = [m for m in candidates if m.uci() not in seen]
                for m in new:
                    seen[m.uci()] = m
                tag = "new" if new else "duplicate"
                print(f"  [{i}/{iterations}] {tag}: {[c.uci() for c in candidates]}")
            else:
                print(f"  [{i}/{iterations}] no legal match  "
                      f"emptied={sorted(departed) or '-'}  "
                      f"occupied={sorted(arrived) or '-'}")
        vision.pop_last_occupancy()
    return list(seen.values())


def process_human_move(vision: ChessVision, motion: RobotMotion,
                       board: chess.Board, iterations: int) -> Optional[chess.Move]:
    motion.go_to_init()
    candidates = attempt_human_move(vision, board, iterations)
    if not candidates:
        print(f"  No legal move detected in {iterations} iterations.")
        return None

    if len(candidates) == 1:
        move = attach_promotion(candidates[0], board)
        print(f"\n  Detected move: {board.san(move)} ({move.uci()})")
        return move

    choice = pick_from_candidates(candidates, board)
    if choice is None:
        return None
    return attach_promotion(choice, board)


def print_turn_header(board: chess.Board, include_keys: bool,
                       settings: Optional[Settings] = None,
                       vision: Optional[ChessVision] = None) -> None:
    """Reprint board + move metadata + (human turn) key cheatsheet so they
    stay visible above the next prompt regardless of intervening output."""
    print()
    print(SEPARATOR)
    print(board)
    mover = "WHITE" if board.turn == chess.WHITE else "BLACK"
    print(f"\nMove {board.fullmove_number} - {mover} to move")
    if include_keys:
        it = settings.iterations if settings else DETECTION_ITERATIONS
        th = vision.occupancy_threshold if vision else 0.0
        print(f"Your turn. (iters={it}, threshold={th:.3f})")
        print("  <ENTER>/<SPACE> = move played    <B> = recalibrate board")
        print("  <C> = recalibrate robot          <I> = set iterations")
        print("  <T> = retune threshold           <Q> = quit")


def handle_robot_turn(board: chess.Board, motion: RobotMotion,
                      engine: chess.engine.SimpleEngine) -> None:
    print_turn_header(board, include_keys=False)
    print("Robot is thinking...")
    result = engine.play(board, chess.engine.Limit(time=ENGINE_THINK_TIME_S))
    rmove = result.move
    print(f"Robot plays: {board.san(rmove)} ({rmove.uci()})")
    motion.move_robot(rmove.uci(), board)
    board.push(rmove)


def handle_human_turn(board: chess.Board, vision: ChessVision,
                      motion: RobotMotion, settings: Settings) -> bool:
    """Inner loop for one human move. Return False to quit."""
    while True:
        print_turn_header(board, include_keys=True, settings=settings, vision=vision)
        k = wait_for_key(CONFIRM_KEYS | SPECIAL_KEYS)

        if k in CONFIRM_KEYS:
            move = process_human_move(vision, motion, board, settings.iterations)
            if move is not None:
                board.push(move)
                print(f"  Recorded: {move.uci()}")
                return True
            continue

        if k == "B":
            motion.go_to_init()
            vision.detect_board_corners()
            continue

        if k == "C":
            if input("Calibration takes a long time. Proceed? (Y/n): ").strip().lower() in ("", "y"):
                motion.calibrate()
            continue

        if k == "I":
            prompt_set_iterations(settings)
            continue

        if k == "T":
            run_threshold_grid_search(vision, board)
            continue

        if k == "Q":
            if input("Quit the game? (Y/n): ").strip().lower() in ("", "y"):
                return False


def prompt_set_iterations(settings: Settings) -> None:
    raw = input(f"\n  Iterations (current {settings.iterations}, "
                f"blank = keep, range 1..100): ").strip()
    if not raw:
        return
    try:
        n = int(raw)
    except ValueError:
        print(f"  Invalid: '{raw}' is not an integer")
        return
    if not 1 <= n <= 100:
        print(f"  Invalid: {n} out of range 1..100")
        return
    settings.iterations = n
    print(f"  -> iterations = {n}")


def prompt_set_threshold(vision: ChessVision) -> None:
    raw = input(f"\n  Occupancy threshold (current {vision.occupancy_threshold:.3f}, "
                f"blank = keep, range (0, 1)): ").strip()
    if not raw:
        return
    try:
        v = float(raw)
    except ValueError:
        print(f"  Invalid: '{raw}' is not a number")
        return
    if not 0.0 < v < 1.0:
        print(f"  Invalid: {v} not in open interval (0, 1)")
        return
    vision.occupancy_threshold = v
    print(f"  -> threshold = {v:.3f}")


def run_threshold_grid_search(vision: ChessVision, board: chess.Board) -> None:
    """Grid-search the occupancy threshold against `board`'s expected
    occupancy; offer accept / manual / decline."""
    result = vision.grid_search_threshold(board)
    lo, hi = result["range"]
    range_str = (f"{result['best']:.3f}" if lo == hi
                 else f"{result['best']:.3f}  (tied range {lo:.3f}..{hi:.3f})")
    print(f"\n  Grid search result:")
    print(f"    best threshold = {range_str}")
    print(f"    accuracy       = {result['accuracy']}/{result['total']}")
    print(f"    current        = {vision.occupancy_threshold:.3f}")
    print("  <ENTER>/<SPACE> = accept   <M> = enter manually   <D> = decline")
    k = wait_for_key(CONFIRM_KEYS | {"M", "D"})
    if k in CONFIRM_KEYS:
        vision.occupancy_threshold = result["best"]
        print(f"  -> threshold = {result['best']:.3f}")
    elif k == "M":
        prompt_set_threshold(vision)
    else:
        print("  Declined; threshold unchanged.")


def main() -> None:
    print("=== Chess Robot Framework ===\n")

    if not Path(DEVICE_CONFIG_FILE).exists():
        print(f"'{DEVICE_CONFIG_FILE}' not found. Run: python find_port.py")
        sys.exit(1)
    with open(DEVICE_CONFIG_FILE) as f:
        dcfg = json.load(f)

    print("Loading vision system...")
    vision = ChessVision(camera_id=dcfg["camera_id"], robot_color=ROBOT_COLOR)

    print("Connecting to robot arm...")
    motion = RobotMotion(port=dcfg["robot_port"])
    try:
        motion.enable_torque()
    except Exception as e:
        print(f"\nERROR: could not enable robot torque: {e}")
        print("Power-cycle the robot and try again.")
        motion.disconnect()
        vision.release()
        sys.exit(1)

    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    except FileNotFoundError:
        print(f"\nERROR: Stockfish not found at '{ENGINE_PATH}'.")
        motion.disconnect()
        vision.release()
        sys.exit(1)

    print("\n[Startup] Detecting board corners...")
    motion.go_to_init()
    while not vision.detect_board_corners():
        if input("Retry? (Y/n): ").strip().lower() not in ("", "y"):
            engine.quit()
            motion.disconnect()
            vision.release()
            sys.exit(1)

    board = chess.Board()
    settings = Settings()
    run_threshold_grid_search(vision, board)
    print(f"\nGame on. Robot plays {'WHITE' if ROBOT_COLOR == chess.WHITE else 'BLACK'}.")

    try:
        while not board.is_game_over():
            if board.turn == ROBOT_COLOR:
                handle_robot_turn(board, motion, engine)
            else:
                if not handle_human_turn(board, vision, motion, settings):
                    print("\nGame quit by user.")
                    return
            vision.update_occupancy_from_board(board)

        print("\n" + SEPARATOR)
        print("GAME OVER")
        print(board)
        print(f"Result: {board.result()}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            engine.quit()
        except Exception:
            pass
        motion.disconnect()
        vision.release()


if __name__ == "__main__":
    main()
