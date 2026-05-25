"""Stand-alone robot arm calibration. Run: python calibration.py"""

import json
import sys
import time
from pathlib import Path

from module import RobotMotion, DEVICE_CONFIG_FILE


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def main() -> None:
    if not Path(DEVICE_CONFIG_FILE).exists():
        print(f"'{DEVICE_CONFIG_FILE}' not found. Run: python find_port.py")
        sys.exit(1)
    with open(DEVICE_CONFIG_FILE) as f:
        dcfg = json.load(f)

    print("Connecting to robot arm...")
    motion = RobotMotion(port=dcfg["robot_port"])

    start = time.time()
    try:
        motion.calibrate()
    except KeyboardInterrupt:
        print("\nCalibration interrupted. Partial data saved.")
    finally:
        print(f"Elapsed: {_format_elapsed(time.time() - start)}")
        motion.disconnect()


if __name__ == "__main__":
    main()
