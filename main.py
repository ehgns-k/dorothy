"""DOROTHY chess robot - entry point.

    python main.py              exhibition kiosk on the real hardware
    python main.py --sim        same UI with a simulated camera + arm (no hardware)
    python main.py --windowed   don't go full screen

First-time setup: see README.md (setup.sh -> find_port.py -> calibration.py).
"""
import sys

import kiosk

if __name__ == "__main__":
    kiosk.main(sys.argv[1:])
