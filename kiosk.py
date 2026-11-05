"""
Exhibition front-end: a full-screen, touch-friendly display for the chess
robot. All game logic lives in game.Controller (worker thread); this file
only draws `controller.view` and forwards button presses / keys.

    python kiosk.py                 # real hardware, full screen
    python kiosk.py --sim           # no hardware: simulated camera + arm
    python kiosk.py --windowed      # not full screen

Visitor keys:   arrows / Tab / h j k l = move between buttons,
                Enter / Space = press the highlighted button, 1-9 = choice buttons
Operator keys:  Ctrl+Shift+Q quit,  Ctrl+Shift+N new game,  F11 full screen
"""

import argparse
import atexit
import logging
import logging.handlers
import math
import os
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import chess
from PIL import Image, ImageDraw, ImageFont, ImageTk

import config
from game import Controller, View

log = logging.getLogger("kiosk")

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
BG = "#0e131c"
PANEL = "#161d29"
PANEL_2 = "#1d2635"
TEXT = "#f2f5fa"
MUTED = "#9aa7bb"
ACCENT = "#4da3ff"
FOCUS = "#ffd166"                 # keyboard-focus ring
TONES = {"info": "#4da3ff", "good": "#3ccf8e", "warn": "#ffb547",
         "error": "#ff5d5d", "robot": "#b58cff"}
BTN = {"primary": ("#4da3ff", "#0b1220", "#7dbcff"),
       "secondary": ("#2a3547", TEXT, "#38465e"),
       "danger": ("#ff5d5d", "#1a0b0b", "#ff8080"),
       "choice": ("#2a3547", TEXT, "#38465e"),
       "choice-on": ("#3ccf8e", "#0b1a12", "#62e0a8")}

LIGHT_SQ = (238, 220, 186)
DARK_SQ = (174, 128, 92)
HL_RGBA = {"last": (255, 225, 80, 110), "detect": (77, 163, 255, 150),
           "problem": (255, 70, 70, 165), "help": (255, 170, 40, 170),
           "check": (255, 40, 40, 150)}
GLYPH = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
         chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
UI_FONT = "Ubuntu"
PIECE_FONT_FILE = "DejaVuSans.ttf"
COORD_FONT_FILE = "DejaVuSans-Bold.ttf"


def _font_family(root) -> str:
    fams = set(tkfont.families(root))
    for f in (UI_FONT, "Noto Sans", "DejaVu Sans"):
        if f in fams:
            return f
    return "TkDefaultFont"


# --------------------------------------------------------------------------- #
# Board renderer (PIL, anti-aliased)
# --------------------------------------------------------------------------- #
class BoardRenderer:
    def __init__(self):
        self._fonts = {}

    def _font(self, file, size):
        key = (file, size)
        if key not in self._fonts:
            try:
                self._fonts[key] = ImageFont.truetype(file, size)
            except OSError:
                self._fonts[key] = ImageFont.load_default()
        return self._fonts[key]

    def render(self, v: View, size: int, white_bottom: bool, selected=None) -> Image.Image:
        sq = size / 8
        img = Image.new("RGB", (size, size), DARK_SQ)
        d = ImageDraw.Draw(img)

        def xy(s):
            f, r = chess.square_file(s), chess.square_rank(s)
            c, row = (f, 7 - r) if white_bottom else (7 - f, r)
            return c * sq, row * sq

        for s in chess.SQUARES:
            x, y = xy(s)
            light = (chess.square_file(s) + chess.square_rank(s)) % 2 == 1
            d.rectangle([x, y, x + sq, y + sq], fill=LIGHT_SQ if light else DARK_SQ)

        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        marks = {}
        if v.last_move:
            mv = chess.Move.from_uci(v.last_move)
            marks[mv.from_square] = marks[mv.to_square] = "last"
        marks.update(dict(v.highlights))
        if selected is not None:
            marks[selected] = "detect"
        for s, kind in marks.items():
            x, y = xy(s)
            col = HL_RGBA.get(kind, HL_RGBA["detect"])
            if kind in ("problem", "help"):
                w = max(4, int(sq * 0.08))
                ld.rectangle([x + w / 2, y + w / 2, x + sq - w / 2, y + sq - w / 2],
                             outline=col[:3] + (255,), width=w)
                ld.rectangle([x, y, x + sq, y + sq], fill=col[:3] + (70,))
            elif kind == "check":
                ld.ellipse([x + sq * 0.05, y + sq * 0.05, x + sq * 0.95, y + sq * 0.95], fill=col)
            else:
                ld.rectangle([x, y, x + sq, y + sq], fill=col)
        img.paste(layer, (0, 0), layer)

        # coordinates
        cf = self._font(COORD_FONT_FILE, max(10, int(sq * 0.15)))
        for i in range(8):
            f = i if white_bottom else 7 - i
            r = 7 - i if white_bottom else i
            light_bottom = (f + (0 if white_bottom else 7)) % 2 == 1
            d.text((i * sq + sq - 4, size - 3), "abcdefgh"[f], font=cf, anchor="rd",
                   fill=DARK_SQ if light_bottom else LIGHT_SQ)
            d.text((4, i * sq + 3), str(r + 1), font=cf, anchor="lt",
                   fill=DARK_SQ if (0 if white_bottom else 7) % 2 == r % 2 else LIGHT_SQ)

        # pieces
        board = chess.Board(v.fen)
        pf = self._font(PIECE_FONT_FILE, int(sq * 0.80))
        stroke = max(2, int(sq * 0.035))
        for s, p in board.piece_map().items():
            x, y = xy(s)
            cx, cy = x + sq / 2, y + sq / 2 + sq * 0.03
            g = GLYPH[p.piece_type]
            if p.color == chess.WHITE:
                d.text((cx + 2, cy + 3), g, font=pf, anchor="mm", fill=(0, 0, 0, 80))
                d.text((cx, cy), g, font=pf, anchor="mm", fill=(250, 250, 248),
                       stroke_width=stroke, stroke_fill=(40, 34, 30))
            else:
                d.text((cx + 2, cy + 3), g, font=pf, anchor="mm", fill=(0, 0, 0, 80))
                d.text((cx, cy), g, font=pf, anchor="mm", fill=(28, 26, 26),
                       stroke_width=stroke, stroke_fill=(235, 230, 220))

        # arrow (robot's move / detected move)
        if v.arrow:
            a = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            ad = ImageDraw.Draw(a)
            (x0, y0), (x1, y1) = xy(v.arrow[0]), xy(v.arrow[1])
            x0, y0, x1, y1 = x0 + sq / 2, y0 + sq / 2, x1 + sq / 2, y1 + sq / 2
            ang = math.atan2(y1 - y0, x1 - x0)
            head = sq * 0.42
            ex, ey = x1 - math.cos(ang) * head * 0.9, y1 - math.sin(ang) * head * 0.9
            col = (181, 140, 255, 200) if v.tone == "robot" else (77, 163, 255, 200)
            ad.line([x0, y0, ex, ey], fill=col, width=int(sq * 0.16))
            pts = [(x1, y1),
                   (x1 - head * math.cos(ang - 0.5), y1 - head * math.sin(ang - 0.5)),
                   (x1 - head * math.cos(ang + 0.5), y1 - head * math.sin(ang + 0.5))]
            ad.polygon(pts, fill=col)
            img.paste(a, (0, 0), a)
        return img


# --------------------------------------------------------------------------- #
# Kiosk window
# --------------------------------------------------------------------------- #
class Kiosk:
    def __init__(self, controller: Controller, fullscreen: bool):
        self.c = controller
        self.root = tk.Tk()
        self.root.title(config.TITLE)
        self.root.configure(bg=BG)
        self.fullscreen = fullscreen
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if fullscreen:
            self.root.attributes("-fullscreen", True)
            self.W, self.H = sw, sh
        else:
            self.W, self.H = min(1600, sw - 80), min(900, sh - 100)
            self.root.geometry(f"{self.W}x{self.H}+40+40")
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)  # visitors can't close it
        self.k = min(self.W / 1920, self.H / 1080)
        self.fam = _font_family(self.root)
        self.renderer = BoardRenderer()
        self.white_bottom = controller.robot_color == chess.BLACK
        self._shown = None
        self._board_key = None
        self._buttons_key = None
        self._overlay_key = None
        self._main_buttons: list = []
        self._ov_buttons: list = []
        self._focus = 0
        self._want_focus = None
        self._board_photo = self._cam_photo = None
        self._selected = None
        self._last_preview = 0.0
        self._build()
        self._bind_keys()

    def F(self, size, weight="normal"):
        return (self.fam, max(8, int(size * self.k)), weight)

    def P(self, px):
        return max(1, int(px * self.k))

    # ---- layout -------------------------------------------------------- #
    def _build(self) -> None:
        P = self.P
        margin = P(48)
        self.board_px = int(min(self.H - 2 * margin, self.W * 0.52))
        left = tk.Frame(self.root, bg=BG)
        left.pack(side="left", fill="y", padx=(margin, P(24)), pady=margin)

        self.eval_canvas = tk.Canvas(left, width=P(26), height=self.board_px, bg=BG,
                                     highlightthickness=0)
        self.eval_canvas.pack(side="left", padx=(0, P(14)))
        self.board_canvas = tk.Canvas(left, width=self.board_px, height=self.board_px,
                                      bg=BG, highlightthickness=0)
        self.board_canvas.pack(side="left")
        self.board_canvas.bind("<Button-1>", self._on_board_click)

        right = tk.Frame(self.root, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(P(24), margin), pady=margin)
        self.right = right
        self.right_w = self.W - self.board_px - P(26) - P(14) - 2 * margin - 2 * P(24)
        wrap = self.right_w - P(90)

        # header
        head = tk.Frame(right, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text=config.TITLE, font=self.F(54, "bold"), fg=TEXT, bg=BG
                 ).pack(side="left")
        # top right: re-find the board in the camera picture (shortcut C)
        colors = BTN["secondary"]
        self.cal_btn = tk.Button(head, text="", font=self.F(17, "bold"), bg=colors[0],
                                 fg=colors[1], activebackground=colors[2],
                                 activeforeground=colors[1], relief="flat", bd=0, takefocus=0,
                                 highlightthickness=0, padx=P(22), pady=P(12), cursor="hand2",
                                 command=self._calibrate_camera)
        self.cal_btn.pack(side="right", anchor="n", pady=(P(10), 0))
        self._cal_state = None
        self._set_cal_button("", "")
        tk.Label(right, text=config.SUBTITLE, font=self.F(22), fg=MUTED, bg=BG, anchor="w"
                 ).pack(fill="x")
        self.info_lbl = tk.Label(right, text="", font=self.F(16), fg=MUTED, bg=BG, anchor="w",
                                 justify="left", wraplength=wrap)
        self.info_lbl.pack(fill="x")

        # status card
        card = tk.Frame(right, bg=PANEL)
        card.pack(fill="x", pady=(P(26), P(18)))
        self.accent = tk.Frame(card, bg=ACCENT, width=P(10))
        self.accent.pack(side="left", fill="y")
        body = tk.Frame(card, bg=PANEL)
        body.pack(side="left", fill="both", expand=True, padx=P(30), pady=P(24))
        self.headline = tk.Label(body, text="", font=self.F(44, "bold"), fg=TEXT, bg=PANEL,
                                 anchor="w", justify="left", wraplength=wrap)
        self.headline.pack(fill="x")
        self.detail = tk.Label(body, text="", font=self.F(23), fg=MUTED, bg=PANEL,
                               anchor="w", justify="left", wraplength=wrap)
        self.detail.pack(fill="x", pady=(P(8), 0))
        self.progress = tk.Canvas(body, height=P(8), bg=PANEL_2, highlightthickness=0)
        self.progress.pack(fill="x", pady=(P(18), 0))
        self.btn_row = tk.Frame(body, bg=PANEL)
        self.btn_row.pack(fill="x", pady=(P(20), 0))

        # bottom: camera + move list
        bottom = tk.Frame(right, bg=BG)
        bottom.pack(fill="both", expand=True)
        cam = tk.Frame(bottom, bg=PANEL)
        cam.pack(side="left", fill="both", expand=True)
        cam.pack_propagate(False)        # the image follows the panel, not the other way
        self.cam_frame = cam
        cam_head = tk.Frame(cam, bg=PANEL)
        cam_head.pack(fill="x", padx=P(20), pady=(P(14), P(6)))
        tk.Label(cam_head, text="WHAT THE ROBOT SEES", font=self.F(14, "bold"), fg=MUTED,
                 bg=PANEL).pack(side="left")
        self.cam_state = tk.Label(cam_head, text="", font=self.F(14, "bold"), fg=TONES["good"],
                                  bg=PANEL)
        self.cam_state.pack(side="right")
        self.cam_lbl = tk.Label(cam, bg=PANEL)
        self.cam_lbl.pack(fill="both", expand=True, padx=P(14), pady=(0, P(14)))
        self.cam_head = cam_head
        self.cam_size = (P(460), P(330))   # updated to the panel's real size

        moves = tk.Frame(bottom, bg=PANEL, width=P(230))
        moves.pack(side="left", fill="y", padx=(P(18), 0))
        moves.pack_propagate(False)
        tk.Label(moves, text="MOVES", font=self.F(14, "bold"), fg=MUTED, bg=PANEL
                 ).pack(anchor="w", padx=P(20), pady=(P(14), P(6)))
        self.moves_lbl = tk.Label(moves, text="", font=(self._mono(), max(8, int(19 * self.k))),
                                  fg=TEXT, bg=PANEL, justify="left", anchor="nw")
        self.moves_lbl.pack(fill="both", expand=True, padx=P(20))

        # overlay for errors / attendant messages
        self.overlay = tk.Frame(self.root, bg="#0a0d13")
        self.ov_title = tk.Label(self.overlay, font=self.F(46, "bold"), fg=TEXT, bg="#0a0d13")
        self.ov_title.pack(pady=(self.H * 0.25, P(20)))
        self.ov_msg = tk.Label(self.overlay, font=self.F(24), fg=MUTED, bg="#0a0d13",
                               wraplength=self.W * 0.6, justify="center")
        self.ov_msg.pack()
        self.ov_btns = tk.Frame(self.overlay, bg="#0a0d13")
        self.ov_btns.pack(pady=P(40))

    def _mono(self):
        fams = set(tkfont.families(self.root))
        for f in ("Ubuntu Mono", "DejaVu Sans Mono"):
            if f in fams:
                return f
        return "TkFixedFont"

    def _bind_keys(self) -> None:
        r = self.root
        r.bind("<space>", lambda e: self._activate_focused())
        r.bind("<Return>", lambda e: self._activate_focused())
        r.bind("<KP_Enter>", lambda e: self._activate_focused())
        for k in ("<Left>", "<Up>", "h", "k", "<ISO_Left_Tab>", "<Shift-Tab>"):
            r.bind(k, lambda e: self._move_focus(-1))
        for k in ("<Right>", "<Down>", "l", "j", "<Tab>"):
            r.bind(k, lambda e: self._move_focus(+1))
        for n in range(1, 10):
            r.bind(str(n), lambda e, n=n: self._key(str(n)))
        # Ctrl+Shift+Q / Ctrl+Shift+N, whatever Caps Lock or the layout says
        r.bind("<Control-KeyPress>", self._control_key)
        r.bind("<F11>", lambda e: self._toggle_fullscreen())
        r.bind("c", lambda e: self._calibrate_camera())
        r.bind("C", lambda e: self._calibrate_camera())

    def _calibrate_camera(self) -> str:
        if not getattr(self, "_quitting", False):
            self.c.send("calibrate_camera")
            self._set_cal_button(self.c.view.cam_cal, self.c.view.phase)  # react at once
        return "break"

    CAL_LABELS = {
        "": ("\u2316  Calibrate camera   C", "secondary"),
        "pending": ("\u23f3  Calibrating soon...", "busy"),
        "pending_arm": ("\u23f3  After the robot's move...", "busy"),
        "running": ("\u23f3  Calibrating...", "busy"),
        "done": ("\u2713  Camera calibrated", "done"),
        "failed": ("\u2715  Board not found", "failed"),
    }
    CAL_COLORS = {"secondary": BTN["secondary"], "busy": ("#3a3320", "#ffd166", "#3a3320"),
                  "done": ("#173a2a", "#3ccf8e", "#173a2a"),
                  "failed": ("#3a1c1c", "#ff7b7b", "#3a1c1c")}

    def _set_cal_button(self, state: str, phase: str) -> None:
        if state == "pending" and phase in ("robot_move", "robot_think"):
            state = "pending_arm"     # the arm must finish first
        if state == self._cal_state:
            return
        self._cal_state = state
        text, style = self.CAL_LABELS.get(state, self.CAL_LABELS[""])
        bg, fg, abg = self.CAL_COLORS[style]
        self.cal_btn.configure(text=text, bg=bg, fg=fg, activebackground=abg,
                               activeforeground=fg,
                               cursor="hand2" if state == "" else "watch")

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _control_key(self, e) -> str:
        if not e.state & 0x1:          # Shift must be held
            return ""
        k = (e.keysym or "").lower()
        if k == "q":
            self._operator_quit()
            return "break"
        if k == "n":
            self.c.send("newgame")
            return "break"
        return ""

    def _operator_quit(self):
        """Ctrl+Shift+Q: always closes within a few seconds, whatever the
        game is busy with (robot move, engine, board search...)."""
        if getattr(self, "_quitting", False):
            return
        self._quitting = True
        try:
            config.STOP_SENTINEL.write_text(time.strftime("%F %T"))
        except Exception:
            pass
        log.info("operator quit requested")
        self.c.request_quit()
        self.ov_title.configure(text="Shutting down")
        self.ov_msg.configure(text="Stopping the robot...")
        for ch in self.ov_btns.winfo_children():
            ch.destroy()
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.overlay.lift()
        deadline = time.time() + 4.0

        def wait_worker():
            w = getattr(self.c, "worker", None)
            if w is None or not w.is_alive() or time.time() > deadline:
                self.root.quit()       # main() then parks (if safe) and goes limp
            else:
                self.root.after(100, wait_worker)
        self.root.after(100, wait_worker)

    def _key(self, key: str) -> None:
        for b in self.c.view.buttons:
            if b.key == key:
                self._press(b)
                return

    # ---- keyboard focus ------------------------------------------------ #
    def _active_buttons(self):
        return self._ov_buttons if self.overlay.winfo_ismapped() else self._main_buttons

    def _nav_buttons(self):
        """Buttons the arrow keys cycle through. On the start screen that is
        ONLY the difficulty choices - Enter/Space always starts the game."""
        btns = self._active_buttons()
        diff = [bw for bw in btns if bw[0].action == "difficulty"]
        return diff or btns

    def _press(self, b) -> None:
        self._want_focus = (b.action, b.value)
        self.c.send(b.action, b.value)

    def _activate_focused(self) -> str:
        nav = self._nav_buttons()
        if nav and nav[0][0].action == "difficulty":
            self._key("Space")             # start screen: Enter = Start game
        elif nav:
            self._press(nav[self._focus % len(nav)][0])
        else:
            self._key("Space")
        return "break"

    def _move_focus(self, step: int) -> str:
        nav = self._nav_buttons()
        if nav:
            self._focus = (self._focus + step) % len(nav)
            b = nav[self._focus][0]
            self._want_focus = (b.action, b.value)
            if b.action == "difficulty":   # moving IS choosing on the start screen
                self.c.send(b.action, b.value)
            self._draw_focus()
        return "break"  # keep Tk's own Tab traversal out of it

    def _restore_focus(self) -> None:
        nav = self._nav_buttons()
        idx = None
        if self._want_focus is not None:
            idx = next((i for i, (b, _) in enumerate(nav)
                        if (b.action, b.value) == self._want_focus), None)
        if idx is None:  # default: the selected choice, else the main (Space) button
            idx = next((i for i, (b, _) in enumerate(nav) if b.style == "choice-on"), None)
        if idx is None:
            idx = next((i for i, (b, _) in enumerate(nav) if b.key == "Space"), 0)
        self._focus = idx
        self._draw_focus()

    def _draw_focus(self) -> None:
        nav = self._nav_buttons()
        focused = nav[self._focus % len(nav)][1] if nav else None
        for lst in (self._main_buttons, self._ov_buttons):
            for b, w in lst:
                on = w is focused
                bg = w.master.cget("bg")
                w.configure(highlightbackground=FOCUS if on else bg,
                            highlightcolor=FOCUS if on else bg)

    # ---- simulation: click the board to move pieces physically -------- #
    def _on_board_click(self, e) -> None:
        if not self.c.sim or self.c.world is None:
            return
        sq = self.board_px / 8
        col, row = int(e.x // sq), int(e.y // sq)
        if not (0 <= col < 8 and 0 <= row < 8):
            return
        f, r = (col, 7 - row) if self.white_bottom else (7 - col, row)
        s = chess.square(f, r)
        if self._selected is None:
            if s in self.c.world.pieces:
                self._selected = s
        else:
            if s != self._selected:
                self.c.world.human_move(self._selected, s)
            self._selected = None
        self._board_key = None

    # ---- refresh loop -------------------------------------------------- #
    def _refresh(self) -> None:
        try:
            v = self.c.view
            if v.phase == "exit":
                self.root.quit()
                return
            if getattr(self, "_quitting", False):
                pass  # keep the "Shutting down" overlay; the camera keeps updating
            elif self._shown is None or v.version != self._shown.version:
                self._apply(v)
                self._shown = v
            now = time.time()
            if self._shown is not None and self._shown.busy:
                self._draw_busy(now)
            if now - self._last_preview >= 1.0 / config.CAMERA_PREVIEW_FPS:
                self._last_preview = now
                self._update_camera(v)
        except Exception:
            log.exception("UI refresh failed")
        self.root.after(40, self._refresh)

    def _apply(self, v: View) -> None:
        tone = TONES.get(v.tone, ACCENT)
        self.accent.configure(bg=tone)
        self.headline.configure(text=v.headline)
        self.detail.configure(text=v.detail)
        self._set_cal_button(v.cam_cal, v.phase)
        info = []
        if v.difficulty:
            info.append(f"Level: {v.difficulty}")
        info.append("Robot plays " + ("White" if v.robot_white else "Black"))
        if self.c.sim:
            info.append("SIMULATION - click pieces on the board to move them")
        self.info_lbl.configure(text="   •   ".join(info))

        # progress bar (busy: animated in _refresh)
        self.progress.delete("all")
        if v.progress is not None and not v.busy:
            w = self.progress.winfo_width() or 400
            self.progress.create_rectangle(0, 0, w * max(0.0, min(1.0, v.progress)),
                                           self.P(8), fill=tone, width=0)

        bkey = (v.fen, v.last_move, v.highlights, v.arrow, v.tone == "robot", self._selected)
        if bkey != self._board_key:
            self._board_key = bkey
            img = self.renderer.render(v, self.board_px, self.white_bottom, self._selected)
            self._board_photo = ImageTk.PhotoImage(img)
            self.board_canvas.delete("all")
            self.board_canvas.create_image(0, 0, image=self._board_photo, anchor="nw")
            self._draw_eval(v.eval_pawns)
            self._draw_moves(v.moves_san)

        if v.buttons != self._buttons_key:
            self._buttons_key = v.buttons
            self._main_buttons = self._make_buttons(self.btn_row, v.buttons, PANEL)

        overlay_key = (v.overlay, v.buttons)
        if v.overlay:
            if overlay_key != self._overlay_key:
                self.ov_title.configure(text=v.overlay[0])
                self.ov_msg.configure(text=v.overlay[1])
                self._ov_buttons = self._make_buttons(self.ov_btns, v.buttons, "#0a0d13")
            self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.overlay.lift()
        else:
            self.overlay.place_forget()
            self._ov_buttons = []
        self._overlay_key = overlay_key
        self.root.update_idletasks()
        self._restore_focus()

    def _draw_busy(self, now: float) -> None:
        """Sliding bar: the robot is checking something - it isn't frozen."""
        c = self.progress
        c.delete("all")
        w = c.winfo_width() or 400
        seg = w * 0.25
        x = ((now * 0.6) % 1.0) * (w + seg) - seg
        c.create_rectangle(max(0, x), 0, min(w, x + seg), self.P(8),
                           fill=TONES.get(self._shown.tone, ACCENT), width=0)

    def _make_buttons(self, parent, buttons, bg) -> list:
        """Build the buttons; returns [(Button, widget)] in reading order."""
        for ch in parent.winfo_children():
            ch.destroy()
        made = []
        # choices (difficulty, candidate moves, promotion) on their own row(s)
        choices = [b for b in buttons if b.style.startswith("choice")]
        others = [b for b in buttons if not b.style.startswith("choice")]
        per_row = 2 if self.right_w < self.P(760) else 4
        rows = [choices[i:i + per_row] for i in range(0, len(choices), per_row)]
        if others:
            rows.append(others)
        for row_buttons in rows:
            row = tk.Frame(parent, bg=bg)
            row.pack(anchor="w")
            made += self._button_row(row, row_buttons)
        return made

    def _button_row(self, parent, buttons) -> list:
        made = []
        for b in buttons:
            colors = BTN.get(b.style, BTN["primary"])
            label = b.label
            if b.key and b.key != "Space" and b.style.startswith("choice"):
                label = f"{b.key}  {b.label}"
            w = tk.Button(parent, text=label, font=self.F(22, "bold"), bg=colors[0],
                          fg=colors[1], activebackground=colors[2],
                          activeforeground=colors[1], relief="flat", bd=0, takefocus=0,
                          highlightthickness=self.P(5), highlightbackground=parent.cget("bg"),
                          highlightcolor=parent.cget("bg"), padx=self.P(30), pady=self.P(16),
                          cursor="hand2", command=lambda b=b: self._press(b))
            w.pack(side="left", padx=(0, self.P(14)), pady=self.P(4))
            made.append((b, w))
        return made

    def _draw_eval(self, ev) -> None:
        c = self.eval_canvas
        c.delete("all")
        h, w = self.board_px, self.P(26)
        c.create_rectangle(0, 0, w, h, fill="#2b2b2b", width=0)
        if ev is None:
            return
        frac = 1 / (1 + math.exp(-ev / 2.5))  # white's share
        if not self.white_bottom:
            frac = 1 - frac
        y = h * (1 - frac)
        c.create_rectangle(0, y, w, h, fill="#f1f1f1", width=0)
        txt = "M" if abs(ev) >= 99 else f"{abs(ev):.1f}"
        top_is_white = not self.white_bottom
        leader_white = ev > 0
        at_bottom = leader_white != top_is_white
        c.create_text(w / 2, h - 6 if at_bottom else 6, text=txt, anchor="s" if at_bottom else "n",
                      font=self.F(11, "bold"), fill="#222" if at_bottom else "#eee")

    def _draw_moves(self, san) -> None:
        rows = [f"{i // 2 + 1:>3}. {san[i]:<8}{san[i + 1] if i + 1 < len(san) else ''}"
                for i in range(0, len(san), 2)]
        self.moves_lbl.configure(text="\n".join(rows[-14:]))

    def _update_camera(self, v: View) -> None:
        vis = self.c.vision
        if vis is None:
            return
        if v.phase == "board":
            # board lost / moved: show the live camera big, where the board was
            big = vis.preview((self.board_px, self.board_px), full=True)
            if big is not None:
                self._big_photo = ImageTk.PhotoImage(Image.fromarray(big))
                self.board_canvas.delete("all")
                self.board_canvas.create_image(self.board_px / 2, self.board_px / 2,
                                               image=self._big_photo)
                self.board_canvas.create_text(self.board_px / 2, self.P(30),
                                              text="LIVE CAMERA", fill=TONES["warn"],
                                              font=self.F(20, "bold"))
                self._board_key = None  # redraw the board once we're back
        fw, fh = self.cam_frame.winfo_width(), self.cam_frame.winfo_height()
        if fw > 50 and fh > 50:   # fill the panel; the frame keeps its own aspect ratio
            self.cam_size = (fw - 2 * self.P(14), fh - self.cam_head.winfo_height() - self.P(34))
        img = vis.preview(self.cam_size, full=True, board=self.c.board)  # whole frame, uncropped
        if img is not None:
            self._cam_photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.cam_lbl.configure(image=self._cam_photo)
        cam = self.c.camera
        if cam is not None and getattr(cam, "error", None):
            self.cam_state.configure(text="● NO SIGNAL", fg=TONES["error"])
        elif v.hand:
            self.cam_state.configure(text="● HAND IN VIEW", fg=TONES["warn"])
        else:
            self.cam_state.configure(text="● LIVE", fg=TONES["good"])

    def run(self) -> None:
        self.c.worker = threading.Thread(target=self.c.run, daemon=True, name="game")
        self.c.worker.start()
        self.root.after(100, self._refresh)
        self.root.mainloop()


def setup_logging() -> None:
    config.LOG_DIR.mkdir(exist_ok=True)
    h = logging.handlers.RotatingFileHandler(config.LOG_DIR / "kiosk.log",
                                             maxBytes=5_000_000, backupCount=5)
    logging.basicConfig(level=logging.INFO, handlers=[h, logging.StreamHandler()],
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="simulate camera and robot")
    ap.add_argument("--windowed", action="store_true", help="don't go full screen")
    args = ap.parse_args(argv)
    os.chdir(config.HERE)
    setup_logging()
    log.info("kiosk starting (sim=%s)", args.sim)
    ctl = Controller(sim=args.sim)
    # Whatever ends the program, the arm ends up limp (torque off).
    atexit.register(lambda: ctl.arm and ctl.arm.limp())
    app = Kiosk(ctl, fullscreen=config.FULLSCREEN and not args.windowed)
    stopped_by = []

    def on_signal(signum, _frame):
        # Ctrl+C, kill, closed terminal: limp right away, no parking move.
        stopped_by.append(signum)
        log.warning("signal %s - arm goes limp, exiting", signal.Signals(signum).name)
        ctl.emergency_limp()
        app.root.quit()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    try:
        app.run()
    finally:
        ctl.shutdown()
    log.info("kiosk exited")
    logging.shutdown()
    # Exit right away: background camera/vision threads may sit inside native
    # code, and tearing the interpreter down around them can abort the process.
    os._exit(128 + stopped_by[0] if stopped_by else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
