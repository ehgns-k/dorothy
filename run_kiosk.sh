#!/usr/bin/env bash
# Exhibition launcher: keeps the kiosk running; relaunches it after a crash.
# Stops only when an operator quits with Ctrl+Shift+Q (kiosk.stop sentinel).
cd "$(dirname "$0")"
mkdir -p logs
rm -f kiosk.stop
# keep the screen awake while the exhibition runs
xset s off -dpms 2>/dev/null || true
# Ctrl+C / kill stops the launcher too (the kiosk itself makes the arm limp)
trap 'echo "$(date "+%F %T") launcher interrupted" >> logs/launcher.log; exit 0' INT TERM HUP
while true; do
  echo "$(date '+%F %T') launching kiosk" >> logs/launcher.log
  .venv/bin/python main.py "$@"
  code=$?
  if [ "$code" -eq 129 ] || [ "$code" -eq 130 ] || [ "$code" -eq 143 ]; then   # hangup / Ctrl+C / kill
    echo "$(date '+%F %T') kiosk stopped by signal ($code)" >> logs/launcher.log
    break
  fi
  if [ -f kiosk.stop ]; then
    rm -f kiosk.stop
    echo "$(date '+%F %T') operator stop" >> logs/launcher.log
    break
  fi
  echo "$(date '+%F %T') kiosk exited - relaunching in 3 s" >> logs/launcher.log
  sleep 3
done
