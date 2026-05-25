"""
Chess robot framework modules.

ChessVision  - USB camera + chesscog occupancy classifier
RobotMotion  - lerobot SO-101 follower arm control via Feetech STS3215 servos
"""

import sys
import json
import msvcrt
import time
from pathlib import Path
from typing import Optional

# Allow `import chesscog.*` to find the bundled source at ./chesscog/chesscog/
# when chesscog isn't pip-installed (otherwise ./chesscog/ is read as a
# namespace package with no submodules).
_bundled = Path(__file__).resolve().parent / "chesscog"
if (_bundled / "chesscog" / "__init__.py").is_file():
    sys.path.insert(0, str(_bundled))

import chess
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from recap import URI, CfgNode as CN

from chesscog.corner_detection import find_corners
from chesscog.occupancy_classifier import create_dataset as occ_dataset
from chesscog.core import device, DEVICE
from chesscog.core.dataset import build_transforms, Datasets

try:
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
except ImportError as e:
    raise ImportError(
        "lerobot is not installed. Activate cnsci and `pip install lerobot`."
    ) from e


CALIBRATION_FILE = "calibration.json"
DEVICE_CONFIG_FILE = "device_config.json"

# SO-101 motor IDs (set with `lerobot-setup-motors` if not yet 1..6).
SO101_MOTORS = {
    "shoulder_pan":  Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(id=3, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(id=4, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(id=5, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(id=6, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
}

# Fallback gripper values if calibration["gripper"] is missing.
GRIPPER_OPEN_TICKS_FALLBACK = 2000
GRIPPER_CLOSED_TICKS_FALLBACK = 2400

# Default occupancy decision threshold; runtime-adjustable on ChessVision.
OCCUPANCY_THRESHOLD = 0.02

# Software-interpolated joint-space motion: stream Goal_Position at
# MOTION_UPDATE_HZ; the bottleneck joint runs at MAX_JOINT_SPEED_TPS,
# others scale down so all five reach the target on the same step.
MAX_JOINT_SPEED_TPS = 350
MOTION_UPDATE_HZ = 200
MOTION_STEP_DT_S = 1.0 / MOTION_UPDATE_HZ
GRIPPER_SETTLE_S = 0.5


class ChessVision:
    _squares = list(chess.SQUARES)

    def __init__(self, camera_id: int = 0, robot_color: chess.Color = chess.WHITE):
        """
        Args:
            camera_id: cv2 camera index from find_port.py.
            robot_color: side the camera is on. chesscog always crops as if
                white-side; on BLACK we flip the mask 180 deg so chess.SQUARES
                indices match physical squares either way.
        """
        self.robot_color = robot_color
        self.board_corners: Optional[np.ndarray] = None
        self.occupancy_threshold: float = OCCUPANCY_THRESHOLD
        self.occupancy_hist = [self._starting_occupancy()]

        self._corner_cfg = CN.load_yaml_with_base("config://corner_detection.yaml")

        # chesscog's .pt files pickle the full model object, not a state dict.
        occ_path = URI("models://") / "occupancy_classifier"
        self._occ_cfg = CN.load_yaml_with_base(next(iter(occ_path.glob("*.yaml"))))
        self._occ_model = device(torch.load(
            next(iter(occ_path.glob("*.pt"))),
            map_location=DEVICE, weights_only=False))
        self._occ_model.eval()
        self._occ_transforms = build_transforms(self._occ_cfg, mode=Datasets.TEST)
        self._occ_occupied_idx = self._occ_cfg.DATASET.CLASSES.index("occupied")

        self._camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self._camera.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_id}")
        for _ in range(5):
            self._camera.read()

    @staticmethod
    def _starting_occupancy() -> list:
        b = chess.Board()
        return sorted(chess.square_name(sq) for sq in chess.SQUARES if b.piece_at(sq))

    def capture_image(self) -> np.ndarray:
        for _ in range(5):
            self._camera.grab()
        ret, frame = self._camera.read()
        if not ret or frame is None:
            raise RuntimeError("Camera read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect_board_corners(self) -> bool:
        print("Detecting board corners...")
        img = self.capture_image()
        try:
            corners = find_corners(self._corner_cfg, img)
        except Exception as e:
            print(f"  Corner detection failed: {e}")
            return False

        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        pts = corners.astype(int)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
        for pt, lbl, col in zip(pts,
                                ["TL", "TR", "BR", "BL"],
                                [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]):
            cv2.circle(vis, tuple(pt), 12, col, -1)
            cv2.putText(vis, lbl, tuple(pt + np.array([15, 5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        win = "Board Corners - press any key to close"
        cv2.imshow(win, vis)
        while True:
            if cv2.waitKey(50) != -1:
                break
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        cv2.destroyAllWindows()

        if input("Accept these corners? (y/n): ").strip().lower() == "y":
            self.board_corners = corners
            print("  Board corners saved.")
            return True
        print("  Corners rejected.")
        return False

    def _classify_occupancy(self, img: np.ndarray) -> np.ndarray:
        warped = occ_dataset.warp_chessboard_image(img, self.board_corners)
        crops = [occ_dataset.crop_square(warped, sq, chess.WHITE) for sq in self._squares]
        tensors = device(torch.stack(
            [self._occ_transforms(PILImage.fromarray(c)) for c in crops]))
        with torch.no_grad():
            probs = torch.softmax(self._occ_model(tensors), dim=-1)
        mask = (probs[:, self._occ_occupied_idx].cpu().numpy()
                >= self.occupancy_threshold)
        if self.robot_color == chess.BLACK:
            mask = mask[::-1]
        return mask

    def detect_occupancy(self) -> Optional[list]:
        if self.board_corners is None:
            raise RuntimeError("Board corners not set - call detect_board_corners() first.")
        mask = self._classify_occupancy(self.capture_image())
        squares = sorted(chess.square_name(sq)
                         for sq, occ in zip(self._squares, mask) if occ)
        if self.is_occupancy_legal(squares):
            self.occupancy_hist.append(squares)
            self.clean_occupancy_hist()
            return squares
        print(f"  [Vision] Reading rejected: {len(squares)} occupied squares.")
        return None

    def is_occupancy_legal(self, occupancy_list: list) -> bool:
        return 2 <= len(occupancy_list) <= 32

    def clean_occupancy_hist(self) -> None:
        while len(self.occupancy_hist) > 2:
            self.occupancy_hist.pop(0)

    def update_occupancy_from_board(self, board: chess.Board) -> None:
        """Append canonical board occupancy (used after the robot's move)."""
        self.occupancy_hist.append(sorted(
            chess.square_name(sq) for sq in chess.SQUARES if board.piece_at(sq)))
        self.clean_occupancy_hist()

    def get_occupancy_diff(self) -> tuple:
        if len(self.occupancy_hist) < 2:
            return set(), set()
        before, after = set(self.occupancy_hist[-2]), set(self.occupancy_hist[-1])
        return before - after, after - before

    def pop_last_occupancy(self) -> None:
        if len(self.occupancy_hist) > 1:
            self.occupancy_hist.pop()

    def release(self) -> None:
        try:
            self._camera.release()
        except Exception:
            pass

    def __del__(self):
        if hasattr(self, "_camera"):
            self.release()


class RobotMotion:
    SQUARE_KEYS = [f"{f}{r}" for f in "abcdefgh" for r in range(1, 9)]
    SPECIAL_KEYS = ["capture", "init"]
    GRIPPER_KEY = "gripper"
    JOINT_NAMES = list(SO101_MOTORS.keys())
    ARM_JOINTS = [n for n in JOINT_NAMES if n != "gripper"]

    def __init__(self, port: str, calibration_file: str = CALIBRATION_FILE):
        self.calibration_file = calibration_file
        self.calibration: dict = {}
        if Path(calibration_file).exists():
            with open(calibration_file) as f:
                self.calibration = json.load(f)
            print(f"Calibration loaded from {calibration_file} "
                  f"({len(self.calibration)} entries)")

        self._bus = FeetechMotorsBus(port=port, motors=SO101_MOTORS)
        self._bus.connect(handshake=True)
        print(f"Connected to robot at {port}")

        # Anchor Goal_Position = Present_Position so a later enable_torque
        # doesn't jerk the arm toward a stale goal. Callers enable torque.
        current = self._bus.sync_read("Present_Position", normalize=False)
        self._bus.sync_write("Goal_Position", current, normalize=False)

    def disconnect(self) -> None:
        try:
            self._bus.disconnect(disable_torque=True)
        except Exception:
            pass

    def disable_torque(self) -> None:
        self._bus.disable_torque()

    def enable_torque(self) -> None:
        self._bus.enable_torque()

    def retrieve_joint_angles(self) -> dict:
        """All 6 joints (gripper included) as {name: raw_ticks}."""
        return {n: float(v) for n, v in
                self._bus.sync_read("Present_Position", normalize=False).items()}

    def _read_arm_positions(self) -> dict:
        """The 5 arm joints (gripper excluded) as {name: raw_ticks}."""
        return {n: float(v) for n, v in
                self._bus.sync_read("Present_Position",
                                    motors=self.ARM_JOINTS,
                                    normalize=False).items()}

    def moveto(self, pose: dict) -> None:
        """Synchronised linear motion of the 5 arm joints (gripper untouched).
        Streams interpolated Goal_Positions at MOTION_UPDATE_HZ; the
        longest-delta joint runs at MAX_JOINT_SPEED_TPS, others scale down
        so all reach the target on the same step. No trailing settle - the
        next moveto's read picks up wherever the servos actually are.
        """
        target = {n: int(round(pose[n])) for n in self.ARM_JOINTS}
        try:
            current = self._read_arm_positions()
        except ConnectionError as e:
            raise RuntimeError(
                f"Lost serial comms before moveto: {e}\n"
                "Motors may be in error state; power-cycle the robot."
            ) from e

        deltas = {n: target[n] - current[n] for n in self.ARM_JOINTS}
        max_delta = max(abs(d) for d in deltas.values())
        if max_delta < 1:
            return

        n_steps = max(1, int(np.ceil(max_delta / (MAX_JOINT_SPEED_TPS * MOTION_STEP_DT_S))))
        for step in range(1, n_steps + 1):
            frac = step / n_steps
            interp = {n: int(round(current[n] + deltas[n] * frac))
                      for n in self.ARM_JOINTS}
            self._bus.sync_write("Goal_Position", interp, normalize=False)
            time.sleep(MOTION_STEP_DT_S)

    def grip(self) -> None:
        """Close to calibrated 'closed' then relax Goal = Present so the
        motor doesn't sustain stall current (which latches a Feetech
        Overload error after a few seconds)."""
        self._gripper_write(int(round(self._gripper_value(closed=True))))
        time.sleep(GRIPPER_SETTLE_S)
        actual = self._bus.read("Present_Position", "gripper", normalize=False)
        self._gripper_write(int(round(actual)))

    def release_gripper(self) -> None:
        self._gripper_write(int(round(self._gripper_value(closed=False))))
        time.sleep(GRIPPER_SETTLE_S)

    def _gripper_write(self, value: int) -> None:
        """Write gripper Goal_Position; clear an Overload latch once on hit."""
        try:
            self._bus.write("Goal_Position", "gripper", value, normalize=False)
            return
        except RuntimeError as e:
            if "Overload" not in str(e):
                raise
        print("  [gripper] overload latch hit; cycling motor and retrying")
        self._bus.disable_torque(motors="gripper")
        time.sleep(0.2)
        self._bus.enable_torque(motors="gripper")
        time.sleep(0.1)
        self._bus.write("Goal_Position", "gripper", value, normalize=False)

    def go_to_init(self) -> None:
        if "init" not in self.calibration:
            print("  Warning: 'init' not calibrated; arm may occlude camera.")
            return
        self.moveto(self.calibration["init"][0])

    def _gripper_value(self, closed: bool) -> float:
        if self.GRIPPER_KEY in self.calibration:
            return self.calibration[self.GRIPPER_KEY][1 if closed else 0]["gripper"]
        return (GRIPPER_CLOSED_TICKS_FALLBACK if closed
                else GRIPPER_OPEN_TICKS_FALLBACK)

    def calibrate(self) -> None:
        """Walk through floor+lift for every board position, then gripper
        open+closed. Existing entries can be skipped with <S> at their first
        prompt. Saved after every entry."""
        position_keys = self.SQUARE_KEYS + self.SPECIAL_KEYS
        n_existing = sum(k in self.calibration
                         for k in position_keys + [self.GRIPPER_KEY])
        print("\n=== Robot Arm Calibration ===")
        print("Torque is now DISABLED. Move the arm by hand to each pose.")
        print("  <ENTER>/<SPACE> = record this pose")
        print("  <S>             = skip (keep existing; shown only when one exists)")
        if n_existing:
            print(f"  {n_existing} existing entries - those prompts offer <S>.")
        print()

        try:
            self.disable_torque()
        except Exception as e:
            print(f"  Note: disable_torque failed ({e}); assuming torque is off.")
        calibration: dict = {}

        try:
            for i, key in enumerate(position_keys, 1):
                existing = key in self.calibration
                action = self._prompt(
                    f"[{i:>2}/{len(position_keys)}] Move the arm to "
                    f"{key.upper()} FLOOR position",
                    allow_skip=existing)
                if action == "skip":
                    calibration[key] = self.calibration[key]
                    self._save_calibration(calibration)
                    continue
                floor = self.retrieve_joint_angles()
                self._prompt(f"          Move the arm to {key.upper()} LIFT position",
                             allow_skip=False)
                calibration[key] = [floor, self.retrieve_joint_angles()]
                self._save_calibration(calibration)

            print("\n[Gripper] Record OPEN and CLOSED gripper positions.")
            existing_g = self.GRIPPER_KEY in self.calibration
            action = self._prompt("          Open the gripper fully", allow_skip=existing_g)
            if action == "skip":
                calibration[self.GRIPPER_KEY] = self.calibration[self.GRIPPER_KEY]
            else:
                g_open = self.retrieve_joint_angles()
                self._prompt("          Close the gripper fully", allow_skip=False)
                calibration[self.GRIPPER_KEY] = [g_open, self.retrieve_joint_angles()]
            self._save_calibration(calibration)

            self.calibration = calibration
            print(f"\nCalibration complete - saved to {self.calibration_file}")
        finally:
            try:
                self.enable_torque()
            except Exception as e:
                print(f"  Note: enable_torque failed ({e}); power-cycle the robot.")

    @staticmethod
    def _prompt(message: str, *, allow_skip: bool) -> str:
        """Wait for <ENTER>/<SPACE> ('record') or <S> ('skip', if allowed)."""
        suffix = "  [<ENTER>=record, <S>=skip]" if allow_skip else ""
        sys.stdout.write(message + suffix + ": ")
        sys.stdout.flush()
        while True:
            ch = msvcrt.getch()
            if ch == b"\x03":
                raise KeyboardInterrupt
            if ch in (b"\r", b"\n", b" "):
                sys.stdout.write("\n")
                return "record"
            if allow_skip and ch in (b"s", b"S"):
                sys.stdout.write("[skipped]\n")
                return "skip"

    def _save_calibration(self, calibration: dict) -> None:
        with open(self.calibration_file, "w") as f:
            json.dump(calibration, f, indent=2)

    def uci_parse(self, uci: str, board: chess.Board) -> dict:
        """Parse UCI against the *pre-move* board state."""
        move = chess.Move.from_uci(uci)
        return {
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "capture": board.is_capture(move),
            "castling": board.is_castling(move),
            "en_passant": board.is_en_passant(move),
            "promotion": move.promotion,
            "move": move,
        }

    def _check_calibrated(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self.calibration]
        if missing:
            raise RuntimeError(
                f"Missing calibration for: {missing}. Run calibration.py first.")

    def _pickup_from(self, sq: str) -> None:
        cal = self.calibration
        self.moveto(cal[sq][1])
        self.release_gripper()
        self.moveto(cal[sq][0])
        self.grip()
        self.moveto(cal[sq][1])

    def _drop_at(self, sq: str) -> None:
        cal = self.calibration
        self.moveto(cal[sq][1])
        self.moveto(cal[sq][0])
        self.release_gripper()
        self.moveto(cal[sq][1])

    @staticmethod
    def _castling_pairs(board: chess.Board, mv: chess.Move) -> list:
        if board.is_kingside_castling(mv):
            return [("e1", "g1"), ("h1", "f1")] if board.turn == chess.WHITE \
                   else [("e8", "g8"), ("h8", "f8")]
        return [("e1", "c1"), ("a1", "d1")] if board.turn == chess.WHITE \
               else [("e8", "c8"), ("a8", "d8")]

    def _move_plan(self, info: dict, board: chess.Board) -> list:
        """Any move -> ordered list of (src, dst) piece transports.
        Captures/en-passant put the captured piece on "capture" first."""
        src, dst, mv = info["from"], info["to"], info["move"]
        if info["castling"]:
            return self._castling_pairs(board, mv)
        if info["en_passant"]:
            ep_sq = chess.square_name(
                chess.square(chess.square_file(mv.to_square),
                             chess.square_rank(mv.from_square)))
            return [(ep_sq, "capture"), (src, dst)]
        if info["capture"]:
            return [(dst, "capture"), (src, dst)]
        return [(src, dst)]

    def move_robot(self, uci: str, board: chess.Board) -> None:
        """Execute a chess move. Arm starts and ends at init_floor."""
        if not self.calibration:
            raise RuntimeError("Robot not calibrated. Run: python calibration.py")

        info = self.uci_parse(uci, board)
        plan = self._move_plan(info, board)

        required = {"init"}
        for s, d in plan:
            required.add(s); required.add(d)
        self._check_calibrated(*required)

        self.moveto(self.calibration["init"][1])
        for src_sq, dst_sq in plan:
            self._pickup_from(src_sq)
            self._drop_at(dst_sq)
        self.moveto(self.calibration["init"][1])
        self.moveto(self.calibration["init"][0])

        if info["promotion"]:
            piece_letter = chess.piece_symbol(info["promotion"]).upper()
            print(f"  Promotion at {info['to'].upper()} - please replace the pawn "
                  f"with the promoted piece ({piece_letter}).")
            self._prompt("  Press <ENTER>/<SPACE> when done", allow_skip=False)
