"""
Central configuration for the DOROTHY chess robot.

Every tunable lives here. Values can be overridden without editing code by
putting a `settings.json` next to this file, e.g.

    {"ROBOT_COLOR": "black", "MAX_JOINT_SPEED_TPS": 900, "FULLSCREEN": false}
"""

import json
import os
import shutil
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
DEVICE_CONFIG_FILE = HERE / "device_config.json"   # find_port.py
CALIBRATION_FILE = HERE / "calibration.json"       # calibration.py (arm poses)
BOARD_FILE = HERE / "board.json"                   # setup_board.py (camera corners)
SETTINGS_FILE = HERE / "settings.json"             # optional overrides
LOG_DIR = HERE / "logs"
STOP_SENTINEL = HERE / "kiosk.stop"

# --------------------------------------------------------------------------- #
# Game
# --------------------------------------------------------------------------- #
ROBOT_COLOR = "white"            # "white" (robot opens) or "black"

# Engine difficulty presets shown on the start screen.
#
# The three easier levels play like a PERSON of that strength rather than a
# throttled engine (which plays perfect-looking moves and then a random
# blunder). Stockfish looks only `depth` plies ahead - like a person, it
# misses longer tactics - and lists its `multipv` best moves. One of those is
# picked at random, favouring good moves: `temp_cp` is how much a weaker
# move's lower score (in centipawns) is tolerated, `max_loss_cp` caps how bad
# a normal move may be, and with probability `blunder` it plays a clearly
# worse move (a hung piece, a missed threat). Natural choices - recaptures,
# checks, castling, developing pieces - get a `human_bonus` (centipawns).
# `adapt`: when the robot is well ahead it plays more loosely, when it is
# behind it tightens up, so games stay close and fun.
#
# "Master" is plain full-strength Stockfish (time per move in seconds;
# optional "elo" / "skill" use Stockfish's own limiter).
DIFFICULTIES = [
    {"name": "Beginner", "depth": 2, "multipv": 14, "temp_cp": 160, "max_loss_cp": 700,
     "blunder": 0.15, "mate_in_one": 0.5, "human_bonus": 60, "adapt": True},
    {"name": "Casual", "depth": 3, "multipv": 12, "temp_cp": 120, "max_loss_cp": 450,
     "blunder": 0.10, "mate_in_one": 0.85, "human_bonus": 45, "adapt": True},
    {"name": "Club", "depth": 6, "multipv": 8, "temp_cp": 45, "max_loss_cp": 220,
     "blunder": 0.03, "mate_in_one": 1.0, "human_bonus": 30, "adapt": True},
    {"name": "Master", "elo": None, "skill": None, "time": 1.0},
]
DEFAULT_DIFFICULTY = 1
ENGINE_THREADS = 4               # Orin Nano has 6 cores; leave 2 for vision/UI
ENGINE_HASH_MB = 128
EVAL_TIME_S = 0.15               # analysis time for the on-screen eval bar
ENGINE_CANDIDATES = [
    str(HERE / "stockfish" / "stockfish"),
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
]

# --------------------------------------------------------------------------- #
# Camera / vision
# --------------------------------------------------------------------------- #
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"           # MJPG gives full fps at 720p on most UVC cams
CAMERA_AUTOFOCUS = False         # a fixed camera should not hunt focus
CAMERA_RECONNECT_S = 2.0

WARP_SQUARE_PX = 64              # warped board = 8 * this, per side
SQUARE_INNER_FRAC = 0.80         # fraction of the square used for features

# Change detection (per-square colour distance after lighting compensation).
CHANGE_THRESHOLD = 8.5           # min change (Lab units) for a square to count
CHANGE_NOISE_MULT = 4.0          # ...or this x the board-wide noise level
CHANGE_LOGIT_SCALE = 4.0         # softness of the changed / unchanged decision
LIGHTNESS_WEIGHT = 0.7           # down-weight L* (shadows) vs. colour
MAX_ALIGN_SHIFT_PX = 6           # small camera nudges are compensated
NUDGE_TOLERANCE_PX = 8           # pieces pushed this far (warped px) are "unchanged"

# Stability: the scene must be still (no hand in view) before we read it.
MOTION_PIXEL_DIFF = 22           # grey-level change that counts as "moving"
MOTION_FRACTION = 0.004          # fraction of moving pixels => scene not still
STABLE_TIME_S = 0.7              # how long the scene must stay still
VISION_HZ = 15                   # processing rate of the vision thread
DETECT_CONFIRM_S = 1.0           # auto-detect: countdown before committing
AUTO_DETECT = True               # detect the human move without a button press
BOARD_FIXED_TO_ROBOT = True      # exhibition: arm + board fixed, only the camera may move
BOARD_LOST_RETRIES = 4           # failed board look-ups in a row before asking for help
BOARD_HOME_TOL_PX = 7.0          # board this far from where the arm expects it -> ask
BOARD_DRIFT_PX = 2.5             # rectified-view shift that triggers a re-check
UNEXPLAINED_HINT_S = 4.0         # show "check the board" after this long
USE_COLOUR = True                # also judge moves by empty / white / black per square
BOARD_RECHECK_S = 10.0           # re-detect the board (camera bumped?) this often while idle
BOARD_UPDATE_PX = 1.0            # ignore re-detections that moved less than this

# --------------------------------------------------------------------------- #
# Robot arm
# --------------------------------------------------------------------------- #
SERVO_BAUDRATE = 1_000_000
MOTOR_IDS = {
    "shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
    "wrist_flex": 4, "wrist_roll": 5, "gripper": 6,
}

MAX_JOINT_SPEED_TPS = 700        # peak speed of the slowest-to-arrive joint
MAX_JOINT_ACCEL_TPS2 = 2500      # peak acceleration (min-jerk profile)
APPROACH_SPEED_FRAC = 0.55       # lift <-> floor moves run slower for precision
TRANSIT_PAN_TICKS = 120          # bigger pan swings to/from rest+capture: raise first, then swing
MIN_MOVE_TIME_S = 0.25
MOTION_UPDATE_HZ = 100
SETTLE_TOL_TICKS = 18            # floor poses: wait until this close
SETTLE_TIMEOUT_S = 0.8

# Gripper: only the OPEN position is calibrated. Closing is torque-sensed:
# the jaws close gradually and stop as soon as the servo stalls on something
# (it lags behind its goal, or its load reading jumps).
# While CARRYING a piece the force is automatically capped just below the
# servo's own overload threshold (read from the servo; 25 % on a LeRobot-
# configured SO-101 gripper). Above it for 2 s the servo switches itself off
# and drops the piece - so a higher GRIPPER_TORQUE_LIMIT does NOT hold better.
GRIPPER_SETTLE_S = 0.35          # time to let the jaws open
GRIPPER_SQUEEZE_TICKS = 180      # hold this far past the FIRST contact (more = firmer grip)
GRIPPER_TORQUE_LIMIT = 900       # 0..1000 grip force cap (RAM register; reset on power-up)
GRIP_CLOSE_SPEED_TPS = 900       # closing speed before contact
GRIP_MAX_TRAVEL = 1600           # never command further than this from "open"
GRIP_LAG_TICKS = 45              # goal - present beyond this = blocked
GRIP_LOAD_LIMIT = 450            # |Present_Load| (0.1 % units) beyond this = contact
GRIP_SQUEEZE_S = 0.25            # full-force squeeze after contact, then hold force
GRIP_OVERLOAD_MARGIN = 30        # hold this far (0.1 % units) below the servo's overload threshold
GRIPPER_HOLD_TORQUE = None       # optional extra cap on the holding force (0..1000)

# Geometry used by the reduced calibration (arm_model.py)
GRIPPER_TIP_MM = 75.0            # jaw hinge axis -> gripping point, along the gripper (SO-101: 75 mm)
LIFT_HEIGHT_SQUARES = 4.0        # lift poses: jaw tip this many squares above the board
LIFT_MAX_JOINT_TICKS = 900       # safety: a lift never moves a joint more than this (~80 deg)
SQUARE_SIZE_MM = 37.5             # board square edge (mm) - CHANGE THIS to match your board

# --------------------------------------------------------------------------- #
# Kiosk UI
# --------------------------------------------------------------------------- #
FULLSCREEN = True
TITLE = "DOROTHY"
SUBTITLE = "The chess-playing robot arm"
IDLE_RESET_S = 15 * 60           # abandon a game nobody touched for this long
GAMEOVER_AUTORESET_S = 120
CAMERA_PREVIEW_FPS = 12
SIM_STYLE = "real"               # --sim look: "real" (green board, leaning pieces) or "flat"

# --------------------------------------------------------------------------- #
# Debug recording (logs/debug/<session>/ - see debuglog.py)
# --------------------------------------------------------------------------- #
DEBUG_LOG = os.environ.get("DOROTHY_DEBUG_LOG", "1") != "0"   # record every recognition result + images
DEBUG_FEED_FPS = 1.0             # raw camera feed frames per second (0 = off)
DEBUG_INFER_EVERY_S = 0.0        # log inferences at most this often (0 = every one)
DEBUG_LOG_MAX_MB = 20000         # oldest sessions are deleted beyond this


def _apply_overrides() -> None:
    if not SETTINGS_FILE.exists():
        return
    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except Exception as e:  # never let a typo stop the exhibition
        print(f"[config] ignoring unreadable {SETTINGS_FILE.name}: {e}")
        return
    g = globals()
    for k, v in data.items():
        if k in g and k.isupper():
            g[k] = v
        else:
            print(f"[config] unknown setting {k!r} ignored")


_apply_overrides()


def robot_color() -> chess.Color:
    return chess.BLACK if str(ROBOT_COLOR).lower().startswith("b") else chess.WHITE


def find_engine() -> str:
    for c in ENGINE_CANDIDATES:
        if Path(c).is_file():
            return c
    found = shutil.which("stockfish")
    if found:
        return found
    raise FileNotFoundError(
        "Stockfish not found. Run ./setup.sh or put the binary at stockfish/stockfish")


def load_device_config() -> dict:
    if not DEVICE_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"{DEVICE_CONFIG_FILE.name} missing - run: python find_port.py")
    return json.loads(DEVICE_CONFIG_FILE.read_text())
