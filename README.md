# DOROTHY - Chess Robot

USB camera + SO-101 follower arm playing chess against you via Stockfish.

## Setup
1. `conda create -n dorothy python=3.10 && conda activate dorothy`
2. `git clone https://github.com/georg-wolflein/chesscog.git`
3. `vim chesscog/chesscog/corner_detection/detect_corners.py`, replace line 207 to `agg = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average")`, save.
4. `cd chesscog && poetry install`
5. `python -m chesscog.occupancy_classifier.download_model`
6. `pip install lerobot feetech-servo-sdk` (plus opencv, python-chess if missing)
7. Download Stockfish, create `stockfish/` folder and put `stockfish.exe` in the created folder. Edit `ENGINE_PATH` in `main.py` to `path/to/stockfish.exe`. ([Download Stockfish](https://stockfishchess.org/download/))
8. `python find_port.py` (writes `device_config.json`)
9. `python calibration.py` (writes `calibration.json`, takes 10~15 minutes of manual work, ~134 prompts)
10. `python main.py`

## Files
- `module.py` - `ChessVison`, `RobotMotion`
- `main.py` - game loop
- `calibration.py` - robot pose recorder
- `find_port.py` - device discovery

