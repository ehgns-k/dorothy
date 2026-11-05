# DOROTHY – Chess Robot

Play chess against an SO-101 robot arm. A camera looks straight down at the
board; the robot watches you move, answers with Stockfish and moves its own
pieces. Built for unattended exhibition use on a **Jetson Orin Nano**
(runs on any Linux PC).

> **Before anything else: set your board's square size.**
> The arm's calibration model works in millimetres and assumes squares of
> **37.5 mm** (edge length of one square). If your board is different,
> measure one square and change this line in `config.py`:
>
> ```python
> SQUARE_SIZE_MM = 37.5             # board square edge (mm) - CHANGE THIS to match your board
> ```
>
> A wrong value makes the robot miss squares far from the ones you taught.
> Set it to `None` to have it estimated during calibration instead (slightly
> less accurate). Likewise, `GRIPPER_TIP_MM = 75.0` is the distance from the
> jaw hinge axis to the gripping point, measured along the gripper; change it
> if your gripper differs.

## Quick start

```bash
./setup.sh                          # venv + Python packages + Stockfish (arm64)
sudo usermod -aG dialout $USER      # once: serial-port access (log out/in afterwards)
.venv/bin/python find_port.py       # camera + arm port -> device_config.json
.venv/bin/python calibration.py     # teach 32 floor poses (CALIBRATION_SQUARES) + capture, rest, gripper-open
.venv/bin/python calibration.py --verify   # arm visits every square to check
./run_kiosk.sh                      # exhibition mode (auto-restarts on crash)
```

No hardware at hand? `.venv/bin/python main.py --sim --windowed` runs the full
UI and vision pipeline against a simulated camera and arm; click a piece and
then a square on the screen to "move" it.

To start the kiosk on login: `cp dorothy-kiosk.desktop ~/.config/autostart/`.

### Hardware notes (Jetson)
* Plug the arm's servo adapter into a **USB-A** port. The Orin Nano dev kit's
  USB-C port is device-mode only (for flashing) and cannot see the arm.
* The Jetson kernel includes the `cdc_acm` driver used by CH343-based adapters
  (e.g. the Waveshare bus-servo board, which appears as `/dev/ttyACM0`) but
  **not** `ch341`, so CH340-based adapters won't work without that module.

## Calibration

`calibration.py` switches the motors off so you can move the arm by hand,
then asks for:

1. The **FLOOR pose of `CALIBRATION_SQUARES` squares** (`config.py`, 3 to
   64, default 32; or `calibration.py --squares N` for one run): jaws open
   around the piece's base, just above the board. The squares are spread
   evenly over the board (32 = all the dark squares, 64 = every square);
   calibration prints the list. Squares taught before can be skipped with
   **S**.
2. **Capture**: one pose where captured pieces are dropped (above a tray).
3. **Rest**: one pose, completely out of the camera's view.
4. **Gripper open**, then a small squeeze so it learns which way closes.

That's N + 3 poses in total; no lift poses are taught.

* **Floor poses for the remaining squares** come from a kinematic model that
  uses the SO-101's known dimensions from the official URDF (upper arm
  116 mm, forearm 135 mm, shoulder 30.4 mm ahead of the base axis, jaw hinge
  84.5 mm past the wrist-flex axis) plus `GRIPPER_TIP_MM`. Only the servo
  zero angles, the arm's height above the board and the board's tilt are
  fitted. Because the model knows where the gripping point is, it doesn't
  matter if the gripper was tilted differently on each square. Whatever the
  model can't explain is corrected from the nearest taught squares, and with
  every other square taught, each computed square sits between taught ones.
* **Every lift pose** is the floor pose raised straight up to
  `LIFT_HEIGHT_SQUARES` × `SQUARE_SIZE_MM` above the board (4 × 37.5 mm =
  150 mm). The gripper may tilt on the way up (it turns about the gripping
  point, so the piece doesn't move sideways); the tilt needing the least
  joint motion is used, and no lift may move a joint more than
  `LIFT_MAX_JOINT_TICKS` (~80°). Where the full height is out of reach, the
  lift is lowered just enough, and calibration lists those squares.

In simulation (SO-101 geometry, 37.5 mm squares, ±4-tick teaching noise,
gripper held at varying angles) computed floor poses land within 2.8 mm on
average (5 mm worst; with the former 12 taught squares: 3–4 mm, 6–8 mm
worst), and lifts stay within about 2 mm of vertical. Average error by number
of taught squares (simulated): 3 squares ~7 mm, 4–5 ~5 mm, 6–12 ~4–8 mm,
32 ~3 mm. With fewer than about 8, check the result with `--verify`. On the
real arm the fit independently estimated the squares at 36.8 mm, which
confirms the geometry.

Changing `SQUARE_SIZE_MM`, `LIFT_HEIGHT_SQUARES`, `GRIPPER_TIP_MM` or
`LIFT_MAX_JOINT_TICKS` makes the robot recompute every square the next time
it starts (taught floor poses are kept).

* `calibration.py --verify`: the arm visits every computed square; press
  **F** on one that's off to teach it by hand (it becomes an extra anchor).
* `calibration.py --teach e4 g7`: (re)teach specific squares and refit.
* `calibration.py --full`: teach all 64 floor poses by hand.
* `calibration.py --check`: with a full 64-square file, shows how well the
  32 taught squares would have predicted the others.

Older `calibration.json` files still load: their taught floor poses are
kept, and lift poses are recomputed with the rule above.

## How it works

| Part | File | What it does |
|---|---|---|
| Vision | `vision.py` | After every move the rectified board image is stored. Once nothing is moving in view (no hand or arm), each square's change is measured (colour difference, compensated for lighting drift and slightly nudged pieces) and every *legal* move is scored by how well it explains the change. **Piece lean:** a top-down camera sees tall pieces leaning away from the point below it, so pieces near the edges and corners appear shifted outward; each square is sampled where its piece is actually seen (the amount is learned from the starting position once after each camera calibration - start-up or the button - and then kept). **Piece colour:** every square is also classified as empty / white piece / black piece by a model learned from the start position and retrained after every confirmed move (so it adapts to the lighting, and to cream pieces on white squares); a move must also leave the right colour on every square it touches. Captures, castling, en passant and promotions are recognised, and colour settles cases where two moves would explain the same changes. ~150 ms per check on the Orin CPU; no GPU or PyTorch needed. |
| Board & camera | `vision.py` | The board is found automatically, even with pieces on it, and a1 is identified from the starting position. The corner detector only sees the 7x7 inner corners, and when pieces hide an edge row it can lock onto a grid one square off (one rank cut off, a row of printed margin added - the same problem chesscog solves in its "inferring missing lines" step); every detection is therefore checked against the checker pattern itself: the 8x8 window whose squares alternate light/dark most strongly (sampled at the square corners, where pieces don't reach) is the board, and the grid is moved there. It is found once at start-up; after that nothing about the camera calibration changes by itself - it is only re-detected when someone presses **Calibrate camera** (top right of the screen, or the **C** key). That shows the live camera large while it looks, then reports whether the camera had shifted (and corrects it), or offers Try again / Cancel if the board can't be found (the previous calibration stays in use). It works at any point in a game and returns to the same screen afterwards; pressed during a robot move, it runs as soon as the arm is done. If the camera seems to have moved (many squares change at once), the screen suggests pressing it. |
| Game flow | `game.py` | Moves are detected hands-free: once a legal move is seen with no hand in view it is confirmed after 1 s and the robot replies. "Not my move", a manual picker and "I've moved" remain as fallbacks. After each robot move the arm returns to rest and the camera checks the board. If the visitor already played their reply while the arm was moving, both moves are recognised and the robot carries straight on. If something looks off, the screen highlights the squares and asks the visitor to fix them and press **OK** ("Checking the board..." shows while it looks); if it still looks different it says so, and a second OK carries on. |
| Arm | `robot.py`, `servo_bus.py` | Lightweight direct Feetech driver (no `lerobot`). Smooth minimum-jerk motion, settle-before-grip, torque-sensed gripping (closes until it feels the piece, then holds firmly a little further closed; it never opens until the piece is placed), raise-then-swing travel to rest and capture so it never sweeps over pieces, auto-reconnect. |
| Calibration model | `arm_model.py` | Pan fitted as a rotation about the base; lift/elbow/wrist by the kinematic model above (cross-validated against simpler fits by jaw-tip error); lift poses computed straight above each floor pose. |
| Engine | `engine.py` | Stockfish with four levels. Beginner, Casual and Club play like a *person* of that strength: Stockfish looks only a few moves ahead (so, like a person, it misses deeper tactics) and lists its best candidates; one is picked at random, favouring good moves, natural-looking ones (recaptures, checks, castling, development) and the occasional real mistake. Well ahead, the robot plays more loosely; behind, it tightens up, so games stay close. Master is full-strength Stockfish. Live evaluation, self-restarting. |
| UI | `kiosk.py` | Full-screen 1080p exhibition display: large board with arrows for the robot's move, evaluation bar, move list, live "what the robot sees" panel with highlighted squares, touch-sized buttons, start screen with difficulty choice. |

## Settings

All tunables live in `config.py`. To override them without editing code,
create `settings.json`, e.g.

```json
{"ROBOT_COLOR": "black", "SQUARE_SIZE_MM": 40.0, "MAX_JOINT_SPEED_TPS": 900}
```

Gripper tuning on the real arm: the jaws stop at the first sign of pressure,
squeeze `GRIPPER_SQUEEZE_TICKS` (180) further closed with up to
`GRIPPER_TORQUE_LIMIT` (900 of 1000) for a quarter of a second, then **hold
just below the gripper servo's own overload threshold**. The SO-101 gripper
servo (as LeRobot sets it up) has `Overload_Torque` = 25 %: if its output
stays above that for 2 s it switches itself off, which drops the piece in
mid-air. That, not too little force, is why pieces fell occasionally, and why
a stronger grip setting makes it worse. The threshold is read from the servo
at start-up (the log says e.g. "holding force 220"). If pieces still slip at
that force, the threshold can be raised on the servo itself (EEPROM register
36), but it is there to stop the small gripper motor overheating. If the jaws
stop before touching a piece, raise `GRIP_LAG_TICKS` / `GRIP_LOAD_LIMIT`.

Difficulty levels are `DIFFICULTIES` in `config.py` (search depth,
randomness, mistake rate, ...: see the comment there). Measured as average
loss per move in centipawns against a deeper search (lower = stronger):
Beginner ~160 (27 % of moves lose 2+ pawns), Casual ~75 (10 %), Club ~35
(2 %). The old presets (Stockfish "Skill 0" / "Elo 1400", ~90–100) mixed
near-perfect moves with random blunders, which felt hard to most visitors.

### Kiosk controls
* **Visitors:** touch the buttons, or use the keyboard.
  * Start screen: **arrow keys**, **Tab / Shift+Tab** or **h j k l** cycle
    through the difficulty levels (moving selects), **1–4** pick one
    directly, and **Enter / Space** starts the game.
  * During a game: the same keys move the yellow highlight between the
    buttons, and **Enter / Space** press the highlighted one.
  * **C** (or the button at the top right): calibrate the camera, i.e. find
    the board in the picture again after the camera was moved or bumped.
* **Operator:**
  * **Ctrl+Shift+Q**: quit. Shows "Shutting down", stops the arm's motion
    and closes within a few seconds, whatever the robot is doing. The arm
    then parks and goes limp, and `run_kiosk.sh` stops instead of
    restarting the kiosk.
  * **Ctrl+Shift+N**: new game.
  * **F11**: toggle full screen.
  * **Ctrl+C** in the terminal (or `kill`): stops immediately. The arm goes
    limp where it is, without parking first, and the launcher stops too.
* Logs: `logs/kiosk.log`, `logs/launcher.log`, and the debug recording below.

### Debug recording
Every kiosk run records to `logs/debug/<date-time>/` (`DEBUG_LOG` in
`config.py`):
* `events.jsonl`: one line per event, with a timestamp. It records:
  * every recognition result: change score and colour verdict (with
    confidence) per square, the best candidate moves with their scores, and
    why the winner won or each was rejected;
  * robot moves, with the engine's candidates;
  * board checks and reference images;
  * every message shown on screen and every button pressed.
* `frames/`: the full camera frame plus the rectified board at every decision
  (move seen, check failed, reference taken, ...), linked from `events.jsonl`.
* `feed/`: the raw camera feed, 1 frame per second (`DEBUG_FEED_FPS`).

`.venv/bin/python debuglog.py logs/debug/<session>` prints a readable
timeline. Expect roughly 0.4 GB per hour; the oldest sessions are deleted
beyond `DEBUG_LOG_MAX_MB` (20 GB).

**The arm is left limp (torque off) whenever a program ends**: kiosk or
calibration, normal quit, Ctrl+C, `kill`, or closed terminal. A limp arm
sinks under its own weight, so leave its rest pose somewhere it can settle
safely. Only a hard kill (`kill -9`) or power loss skips this; the servos
then keep their last state until power-cycled.

## Tests
```bash
.venv/bin/python tests/test_vision_sim.py      # random games incl. leaning pieces + colour
.venv/bin/python tests/stress_vision_sim.py    # noise, lighting, nudged pieces
.venv/bin/python tests/test_board_motion.py    # camera/board movement handling
.venv/bin/python tests/test_grid_offset.py     # board detection one square off is corrected
.venv/bin/python tests/test_arm_model.py [N..] # reduced calibration accuracy (for N taught squares)
.venv/bin/python tests/test_robot_fakebus.py   # gripper sensing/holding force + move sequence
.venv/bin/python tests/test_game_flow.py       # reply during robot move, "board wrong" prompt
.venv/bin/python tests/test_camera_calibration.py  # Calibrate camera button: feedback, bumped camera, failure
.venv/bin/python tests/test_kiosk_keys.py      # keyboard navigation
.venv/bin/python tests/test_kiosk_quit.py      # Ctrl+Shift+Q while the robot is busy
.venv/bin/python tests/ui_demo.py shots/ 8     # scripted end-to-end kiosk run
```
