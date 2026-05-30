# DOROTHY - Chess Robot

USB camera + SO-101 follower arm playing chess against you via Stockfish.

## Setup

First, create a virtual environment with Python version 3.10.

```
conda create -n dorothy python=3.10 && conda activate dorothy
```

Clone [Chesscog](https://github.com/georg-wolflein/chesscog.git) directory inside project directory.

```
git clone https://github.com/georg-wolflein/chesscog.git
```

It should look like `dorothy/chesscog/`. Inside the cloned `chesscog/` directory, go to `chesscog/corner_detection/detect_corners.py`, and replace line 207 to `agg = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average")`.

Install Chesscog as Python package. For example, using Poetry, go inside the cloned `chesscog/` directory (`dorothy/chesscog/`) and do `poetry install`.

Then, you should be able to install the pre-trained ResNet model from Chesscog with the following script:

```
python -m chesscog.occupancy_classifier.download_model
```

Install Python packages `lerobot`, `feetech-servo-sdk`, `opencv-python`, `python-chess` if missing any. Download [Stockfish](https://stockfishchess.org/download/), and unzip to `dorothy` directory. Path to the Stockfish executable would look like `dorothy/stockfish/stockfish.exe`. Inside `main.py`, edit `ENGINE_PATH` to the actual path to your Stockfish executable.

Run `find_port.py`. The script will find ports of your SO-101 arm and your camera, and write `device_config.json` file.

Then run `calibration.py`, which will write `calibration.json` file that contains calibration data of your SO-101 arm.

Finally, run `main.py`, and enjoy the chess game against your SO-101 arm!
