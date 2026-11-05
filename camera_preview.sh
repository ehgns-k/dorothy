#!/usr/bin/env bash
# Live webcam preview for board + camera setup (GStreamer, no Python needed).
# Usage: ./camera_preview.sh [device] [width] [height]
#   device defaults to "camera" in device_config.json, then /dev/video0.
# Close the window or press Ctrl+C to quit.
cd "$(dirname "$0")"

DEV="$1"
if [ -z "$DEV" ] && [ -f device_config.json ]; then
  DEV=$(python3 -c 'import json;print(json.load(open("device_config.json")).get("camera",""))' 2>/dev/null)
fi
[ -e "$DEV" ] || DEV=/dev/video0
W=${2:-1280}
H=${3:-720}

echo "Previewing $DEV at ${W}x${H} (Ctrl+C to quit)"

# Try MJPG first (full fps on most UVC cams), fall back to raw YUYV.
gst-launch-1.0 -q v4l2src device="$DEV" \
  ! image/jpeg,width="$W",height="$H" ! jpegdec ! videoconvert \
  ! textoverlay text="$DEV ${W}x${H}" valignment=top halignment=left font-desc="Sans 14" \
  ! autovideosink sync=false \
|| gst-launch-1.0 -q v4l2src device="$DEV" \
  ! video/x-raw,width="$W",height="$H" ! videoconvert \
  ! textoverlay text="$DEV ${W}x${H} (raw)" valignment=top halignment=left font-desc="Sans 14" \
  ! autovideosink sync=false
