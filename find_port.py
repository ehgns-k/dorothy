"""Find the SO-101 serial port and the overhead camera; write
device_config.json. Run: python find_port.py

Stable device names are used (/dev/serial/by-id, /dev/v4l/by-id) so the
config survives reboots and re-plugging into a different USB port.
"""

import glob
import json
import os
import time

import config


def serial_ports() -> set:
    by_id = set(glob.glob("/dev/serial/by-id/*"))
    return by_id or set(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def find_robot_port() -> str:
    print("\n=== Robot arm ===")
    ports = serial_ports()
    if len(ports) == 1:
        port = ports.pop()
        print(f"  Found one serial device: {port}")
        return port
    input("  Unplug the arm's USB cable, then press ENTER...")
    before = serial_ports()
    input("  Plug it back in (a USB-A port), then press ENTER...")
    time.sleep(1.5)
    new = serial_ports() - before
    if len(new) == 1:
        port = new.pop()
        print(f"  -> {port}")
        return port
    print("  Could not identify the arm automatically.")
    print("  Note: the Jetson's USB-C port is device-only - use a USB-A port.")
    print("  Present:", sorted(serial_ports()) or "none")
    return input("  Enter the port manually: ").strip()


def find_camera() -> str:
    import cv2
    print("\n=== Camera ===")
    cams = sorted(p for p in glob.glob("/dev/v4l/by-id/*") if p.endswith("index0"))
    cams = cams or sorted(glob.glob("/dev/video*"))
    good = []
    for c in cams:
        cap = cv2.VideoCapture(c, cv2.CAP_V4L2)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            print(f"  [{len(good)}] {c}  {frame.shape[1]}x{frame.shape[0]}")
            good.append(c)
    if not good:
        raise SystemExit("  No working camera found.")
    if len(good) == 1:
        return good[0]
    i = input(f"  Which camera looks down at the board? [0-{len(good) - 1}]: ").strip()
    return good[int(i or 0)]


def main() -> None:
    cfg = {}
    if config.DEVICE_CONFIG_FILE.exists():
        cfg = json.loads(config.DEVICE_CONFIG_FILE.read_text())
        print(f"Existing config: {cfg}")
    cfg["camera"] = find_camera()
    cfg["robot_port"] = find_robot_port()
    if not os.access(cfg["robot_port"], os.R_OK | os.W_OK):
        print("\n  WARNING: no permission for the serial port. Run once:\n"
              "    sudo usermod -aG dialout $USER     (then log out and back in)")
    config.DEVICE_CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    print(f"\nSaved {config.DEVICE_CONFIG_FILE.name}: {cfg}")


if __name__ == "__main__":
    main()
