#!/usr/bin/env bash
# One-time setup on the Jetson (no sudo needed except where noted).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

# --system-site-packages: reuse the system tkinter
[ -d .venv ] || "$UV" venv --system-site-packages .venv
"$UV" pip install -p .venv/bin/python -r requirements.txt

if [ ! -x stockfish/stockfish ]; then
  echo "Downloading Stockfish (linux arm64)..."
  mkdir -p stockfish
  tmp="$(mktemp -d)"
  curl -sL -o "$tmp/sf.tgz" \
    https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-arm64-universal.tar.gz
  tar -xzf "$tmp/sf.tgz" -C "$tmp"
  cp "$tmp"/stockfish/stockfish-linux-arm64-universal stockfish/stockfish
  chmod +x stockfish/stockfish
  rm -rf "$tmp"
fi

if ! id -nG | grep -qw dialout; then
  echo
  echo "!! Your user is not in the 'dialout' group, so the arm's serial port is"
  echo "!! not accessible. Run once, then log out and back in:"
  echo "!!     sudo usermod -aG dialout $USER"
fi
echo
echo "Next: .venv/bin/python find_port.py && .venv/bin/python calibration.py"
