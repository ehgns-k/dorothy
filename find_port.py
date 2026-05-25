"""Identify the USB serial port of the SO-101 and the USB camera index;
save both to device_config.json. Run: python find_port.py
"""

import json
import time
from pathlib import Path

import cv2
import serial.tools.list_ports

DEVICE_CONFIG_FILE = "device_config.json"


def _list_ports() -> set:
    return {p.device for p in serial.tools.list_ports.comports()}


def find_robot_port() -> str:
    print("\n=== Robot Arm Port Detection ===")
    input("Step 1. Unplug the SO-101 USB; press <ENTER> when done...")
    before = _list_ports()
    input("Step 2. Plug it back in; press <ENTER> when done...")
    time.sleep(1.5)
    new = _list_ports() - before

    if len(new) == 1:
        port = new.pop()
        print(f"  -> Detected robot arm on {port}")
        return port
    if not new:
        print("  No new port detected. Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"    {p.device}: {p.description}")
        return input("  Enter port manually (e.g. COM3): ").strip()
    print(f"  Multiple new ports detected: {sorted(new)}")
    return input("  Enter the port to use: ").strip()


def _preview_camera(idx: int) -> bool:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return False
    print(f"  Previewing camera {idx} - <S> to select, any other key to skip.")
    selected = False
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera {idx}", frame)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord('s'), ord('S')):
                selected = True
                break
            if k != 255:
                break
            if cv2.getWindowProperty(f"Camera {idx}", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return selected


def find_camera_id() -> int:
    print("\n=== Camera Detection ===")
    candidates = []
    for idx in range(6):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  Camera {idx}: OK ({w}x{h})")
                candidates.append(idx)
            cap.release()

    if not candidates:
        return int(input("  No cameras found. Enter index manually: "))
    if len(candidates) == 1:
        print(f"  Only one camera available: index {candidates[0]}")
        return candidates[0]
    for idx in candidates:
        if _preview_camera(idx):
            return idx
    print(f"  No camera selected, defaulting to {candidates[0]}")
    return candidates[0]


def main() -> None:
    config = {}
    if Path(DEVICE_CONFIG_FILE).exists():
        with open(DEVICE_CONFIG_FILE) as f:
            config = json.load(f)
        print(f"Existing config: {config}")

    config["robot_port"] = find_robot_port()
    config["camera_id"] = find_camera_id()

    with open(DEVICE_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved to {DEVICE_CONFIG_FILE}: {config}")


if __name__ == "__main__":
    main()
