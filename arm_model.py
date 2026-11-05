"""
Reduced arm calibration: teach the FLOOR pose of 32 squares (every other one), compute the
other 52 floor poses and ALL lift poses.

  * shoulder_pan is a pure rotation about a vertical axis: its ticks are
    linear in the azimuth from the robot base to the square, which pins down
    the base position, pan zero and the jaws' sideways offset.
  * shoulder_lift / elbow_flex / wrist_flex follow a fitted kinematic model
    of the arm (link lengths, shoulder position, servo zeros) with the known
    wrist->jaw-tip length (config.GRIPPER_TIP_MM) and square size
    (config.SQUARE_SIZE_MM). Taught floor poses must put the jaw tip at the
    square's distance and on one board plane (a slight tilt is fitted).
  * wrist_roll is fitted as a linear function of azimuth.
  * LIFT poses are not taught: each is the floor pose raised straight up by
    config.LIFT_HEIGHT_SQUARES x square size above the board plane, keeping
    the gripper's angle (so a piece comes out vertically). If that is out of
    reach the lift is lowered just enough to be reachable.
  * Residuals at the taught squares are spread to their neighbours so every
    taught square is reproduced exactly.
"""

import math
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np

import config

TICKS_PER_RAD = 4096 / (2 * math.pi)
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
RADIAL = ["shoulder_lift", "elbow_flex", "wrist_flex"]

# Spread over the whole board: corners, edge middles and the inner area.
# every other square (all dark squares, a1 included): 32 taught floor poses.
# Every square then has a taught neighbour, so the residual correction below
# turns the kinematic fit into a local interpolation.
ANCHORS = [chess.square_name(s) for s in chess.SQUARES
           if (chess.square_file(s) + chess.square_rank(s)) % 2 == 0]
MIN_ANCHORS = 8
RBF_LEN = 2.0  # squares: how far an anchor's residual correction reaches


def square_xy(name: str) -> Tuple[float, float]:
    s = chess.parse_square(name)
    return chess.square_file(s) + 0.5, chess.square_rank(s) + 0.5


class _PoseModel:
    """Model of one pose kind (floor or lift)."""

    def __init__(self, samples: Dict[str, Dict[str, float]]):
        self.names = sorted(samples)
        self.xy = np.array([square_xy(n) for n in self.names])
        self.Y = {j: np.array([samples[n][j] for n in self.names], float) for j in samples[self.names[0]]}
        self.extra = [j for j in self.Y if j not in ARM]  # e.g. gripper: keep median
        self._fit_pan()
        self._fit_radial()
        self._fit_residuals()

    # pan: p = p0 + s*K*(atan2(y-by, x-bx) - asin(d/rho))
    def _pan_pred(self, prm, xy, sign):
        bx, by, p0, d = prm
        dx, dy = xy[:, 0] - bx, xy[:, 1] - by
        rho = np.maximum(np.hypot(dx, dy), abs(d) + 1e-3)
        ang = np.arctan2(dy, dx) - np.arcsin(np.clip(d / rho, -1, 1))
        return p0 + sign * TICKS_PER_RAD * ang

    def _fit_pan(self):
        y = self.Y["shoulder_pan"]
        best = None
        for sign in (1.0, -1.0):
            for bx in np.linspace(-6, 14, 21):
                for by in np.linspace(-6, 14, 21):
                    if 0 < bx < 8 and 0 < by < 8:
                        continue  # the base is not on the board
                    prm = np.array([bx, by, 0.0, 0.0])
                    a = self._pan_pred(prm, self.xy, sign)
                    # unwrap the predicted angle around the data
                    prm[2] = np.median(y - a)
                    r = self._pan_pred(prm, self.xy, sign) - y
                    err = float(np.sum(r ** 2))
                    if best is None or err < best[0]:
                        best = (err, prm, sign)
        _, prm, sign = best
        # Levenberg-Marquardt refinement (numeric Jacobian)
        lam = 1e-2
        for _ in range(100):
            r = self._pan_pred(prm, self.xy, sign) - y
            J = np.empty((len(y), 4))
            for k in range(4):
                h = np.zeros(4)
                h[k] = 1e-4 if k != 2 else 1e-2
                J[:, k] = (self._pan_pred(prm + h, self.xy, sign) - y - r) / h[k]
            A = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9)
            step = np.linalg.solve(A, -J.T @ r)
            new = prm + step
            if np.sum((self._pan_pred(new, self.xy, sign) - y) ** 2) < np.sum(r ** 2):
                prm, lam = new, lam * 0.3
            else:
                lam *= 10
            if np.abs(step).max() < 1e-7:
                break
        self.pan_prm, self.pan_sign = prm, sign

    def _polar(self, xy):
        bx, by, _, d = self.pan_prm
        rho = np.hypot(xy[:, 0] - bx, xy[:, 1] - by)
        reach = np.sqrt(np.maximum(rho ** 2 - d ** 2, 0))
        theta = np.arctan2(xy[:, 1] - by, xy[:, 0] - bx)
        return reach, theta

    # ---- radial joints ------------------------------------------------- #
    @staticmethod
    def _ik_angles(prm, reach, branch):
        """Planar 2-link IK: wrist point at (reach, h); returns (q1, q2)."""
        L1, L2, h = prm
        c = np.clip((reach ** 2 + h ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)
        q2 = branch * np.arccos(c)
        q1 = np.arctan2(h, reach) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        return q1, q2

    @staticmethod
    def _ik_fit(reach, lift, elbow, wrist, init=None):
        """Fit (L1, L2, h) + servo zeros/directions; zeros are solved in
        closed form inside the loop (variable projection). None if hopeless.
        `init` (a previous fit) restricts the search to a warm start."""
        best = None
        rmax = float(reach.max())
        combos = [(b, s1, s2) for b in (1.0, -1.0) for s1 in (1.0, -1.0) for s2 in (1.0, -1.0)]
        if init is not None:
            combos = [(init["branch"], init["s1"], init["s2"])]
        for branch, s1, s2 in combos:
            if True:
                if True:
                    def resid(prm):
                        q1, q2 = _PoseModel._ik_angles(prm, reach, branch)
                        a = lift - s1 * TICKS_PER_RAD * q1
                        b = elbow - s2 * TICKS_PER_RAD * q2
                        return np.r_[a - a.mean(), b - b.mean()]
                    starts = [np.array([L1 * rmax, (1.1 - L1) * rmax, h * rmax])
                              for L1 in (0.45, 0.6, 0.75) for h in (-0.4, 0.0, 0.4)]
                    if init is not None:
                        starts = [init["prm"].copy()]
                    for prm in starts:
                        if True:
                            lam = 1e-2
                            r = resid(prm)
                            for _ in range(60):
                                J = np.empty((len(r), 3))
                                for k in range(3):
                                    d = np.zeros(3); d[k] = 1e-5 * rmax
                                    J[:, k] = (resid(prm + d) - r) / d[k]
                                A = J.T @ J
                                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -J.T @ r)
                                new = prm + step
                                rn = resid(new)
                                if rn @ rn < r @ r:
                                    prm, r, lam = new, rn, lam * 0.3
                                else:
                                    lam *= 10
                                if np.abs(step).max() < 1e-8 * rmax:
                                    break
                            err = float(r @ r)
                            if best is None or err < best[0]:
                                best = (err, prm.copy(), branch, s1, s2)
        if best is None or not np.all(np.isfinite(best[1])):
            return None
        _, prm, branch, s1, s2 = best
        q1, q2 = _PoseModel._ik_angles(prm, reach, branch)
        o1 = float(np.mean(lift - s1 * TICKS_PER_RAD * q1))
        o2 = float(np.mean(elbow - s2 * TICKS_PER_RAD * q2))
        # wrist keeps the gripper's pitch: linear in the arm's angle sum
        A = np.c_[np.ones_like(q1), q1 + q2]
        w = np.linalg.lstsq(A, wrist, rcond=None)[0]
        return {"prm": prm, "branch": branch, "s1": s1, "s2": s2, "o1": o1, "o2": o2, "w": w}

    @staticmethod
    def _ik_eval(fit, reach):
        q1, q2 = _PoseModel._ik_angles(fit["prm"], reach, fit["branch"])
        return {"shoulder_lift": fit["o1"] + fit["s1"] * TICKS_PER_RAD * q1,
                "elbow_flex": fit["o2"] + fit["s2"] * TICKS_PER_RAD * q2,
                "wrist_flex": fit["w"][0] + fit["w"][1] * (q1 + q2)}

    @staticmethod
    def _poly_fit(reach, ys):
        n_distinct = len(np.unique(np.round(reach, 2)))
        deg = int(max(1, min(3 if len(reach) >= 8 else 2, n_distinct - 1)))
        mu, sd = reach.mean(), reach.std() + 1e-9
        return {"mu": mu, "sd": sd, "c": {j: np.polyfit((reach - mu) / sd, y, deg) for j, y in ys.items()}}

    @staticmethod
    def _poly_eval(fit, reach):
        z = (reach - fit["mu"]) / fit["sd"]
        return {j: np.polyval(c, z) for j, c in fit["c"].items()}

    def _fit_radial(self):
        reach, theta = self._polar(self.xy)
        ys = {j: self.Y[j] for j in RADIAL if j in self.Y}
        # choose IK vs polynomial by leave-one-out error on the anchors
        def loo(kind):
            errs = []
            for i in range(len(reach)):
                m = np.arange(len(reach)) != i
                sub = {j: y[m] for j, y in ys.items()}
                if kind == "ik":
                    f = self._ik_fit(reach[m], sub["shoulder_lift"], sub["elbow_flex"],
                                     sub["wrist_flex"], init=self._warm)
                    if f is None:
                        return np.inf
                    p = self._ik_eval(f, reach[i:i + 1])
                else:
                    p = self._poly_eval(self._poly_fit(reach[m], sub), reach[i:i + 1])
                errs.append(max(abs(p[j][0] - ys[j][i]) for j in ys))
            return float(np.sqrt(np.mean(np.square(errs))))
        have_ik = all(j in ys for j in RADIAL)
        full_ik = self._ik_fit(reach, ys["shoulder_lift"], ys["elbow_flex"],
                               ys["wrist_flex"]) if have_ik else None
        self._warm = full_ik
        self.cv = {"poly": loo("poly"), "ik": loo("ik") if full_ik is not None else np.inf}
        self.radial_kind = "ik" if self.cv["ik"] <= self.cv["poly"] else "poly"
        self.radial = full_ik if self.radial_kind == "ik" else self._poly_fit(reach, ys)
        if "wrist_roll" in self.Y:
            A = np.c_[np.ones_like(theta), theta]
            self.roll = np.linalg.lstsq(A, self.Y["wrist_roll"], rcond=None)[0]
        self.extra_val = {j: float(np.median(self.Y[j])) for j in self.extra}

    def radial_T(self) -> np.ndarray:
        return np.c_[self.Y["shoulder_lift"], self.Y["elbow_flex"], self.Y["wrist_flex"]]

    def add_kinematic(self, kin: "Kinematics", loo_kins: List, check_z: bool) -> None:
        """Offer the kinematic model and re-judge ALL radial models by where
        their held-out predictions put the jaw tip (mm, through the fitted
        kinematics). Tick errors are misleading: a teacher holds the gripper
        at different angles, so taught ticks vary even when the tip is right."""
        if kin is None or any(k is None for k in loo_kins):
            return
        reach, _ = self._polar(self.xy)
        T = self.radial_T()
        tr, tz, _ = kin.fk(T)  # where the taught tips actually are
        n = len(reach)
        errs = {"poly": [], "ik": [], "fk": []}
        for i in range(n):
            m = np.arange(n) != i
            sub = {j: self.Y[j][m] for j in RADIAL}
            preds = {
                "poly": self._poly_eval(self._poly_fit(reach[m], sub), reach[i:i + 1]),
                "fk": _KinTarget(loo_kins[i], T[m], reach[m], self.xy[m]).eval(
                    reach[i:i + 1], self.xy[i:i + 1]),
            }
            f = self._ik_fit(reach[m], sub["shoulder_lift"], sub["elbow_flex"],
                             sub["wrist_flex"], init=self._warm) if self._warm else None
            if f is not None:
                preds["ik"] = self._ik_eval(f, reach[i:i + 1])
            for name, p in preds.items():
                pr, pz, _ = kin.fk(np.c_[p["shoulder_lift"], p["elbow_flex"], p["wrist_flex"]])
                e = (pr[0] - tr[i]) ** 2 + ((pz[0] - tz[i]) ** 2 if check_z else 0.0)
                errs[name].append(math.sqrt(e))
        self.cv_mm = {k: float(np.sqrt(np.mean(np.square(v)))) if len(v) == n else np.inf
                      for k, v in errs.items()}
        best = min(self.cv_mm, key=self.cv_mm.get)
        self.radial_kind = best
        if best == "fk":
            self.radial = _KinTarget(kin, T, reach, self.xy)
        elif best == "ik":
            self.radial = self._warm
        else:
            self.radial = self._poly_fit(reach, {j: self.Y[j] for j in RADIAL})
        self._fit_residuals()

    def _base_predict(self, xy) -> Dict[str, np.ndarray]:
        reach, theta = self._polar(xy)
        out = {"shoulder_pan": self._pan_pred(self.pan_prm, xy, self.pan_sign)}
        if self.radial_kind == "fk":
            out.update(self.radial.eval(reach, xy))
        else:
            out.update(self._ik_eval(self.radial, reach) if self.radial_kind == "ik"
                       else self._poly_eval(self.radial, reach))
        if "wrist_roll" in self.Y:
            out["wrist_roll"] = self.roll[0] + self.roll[1] * theta
        for j, v in self.extra_val.items():
            out[j] = np.full(len(xy), v)
        return out

    def _fit_residuals(self):
        base = self._base_predict(self.xy)
        self.resid = {j: self.Y[j] - base[j] for j in base if j in self.Y}

    def predict(self, name: str, use_residuals: bool = True) -> Dict[str, float]:
        xy = np.array([square_xy(name)])
        out = {j: float(v[0]) for j, v in self._base_predict(xy).items()}
        if use_residuals:
            d2 = np.sum((self.xy - xy) ** 2, axis=1)
            w = np.exp(-d2 / (2 * RBF_LEN ** 2))
            # weight -> 1 at an anchor; blend so far squares get ~no correction
            for j, r in self.resid.items():
                if j in out:
                    out[j] += float(np.sum(w * r) / max(1.0, np.sum(w)))
        return out


# --------------------------------------------------------------------------- #
# Kinematic model (uses the known wrist->jaw-tip length)
# --------------------------------------------------------------------------- #
# SO-101 geometry in the arm's vertical plane, from the official URDF
# (TheRobotStudio/SO-ARM100, Simulation/SO101/so101_new_calib.urdf).
SO101_SHOULDER_OFFSET_MM = 30.4   # shoulder-lift axis, ahead of the pan axis
SO101_UPPER_ARM_MM = 116.0        # shoulder-lift axis -> elbow axis
SO101_FOREARM_MM = 135.0          # elbow axis -> wrist-flex axis
SO101_WRIST_TO_HINGE_MM = 84.5    # wrist-flex axis -> jaw hinge axis, along the gripper
SO101_TIP_BELOW_AXIS_MM = 7.9     # gripping point sits this far off the gripper's axis


def tool_length_mm() -> float:
    """Wrist-flex axis -> gripping point. config.GRIPPER_TIP_MM is measured
    from the jaw hinge axis to the gripping point (75 mm on the SO-101;
    the URDF's own gripper frame is 74.8 mm past the hinge)."""
    along = SO101_WRIST_TO_HINGE_MM + float(config.GRIPPER_TIP_MM)
    return math.hypot(along, SO101_TIP_BELOW_AXIS_MM)


class Kinematics:
    """Planar FK of lift/elbow/wrist in the arm's vertical plane (mm):

        th1 = s1 (t1 - o1)/K,  th2 = th1 + s2 (t2 - o2)/K,  th3 = th2 + s3 (t3 - o3)/K
        r = r0 + L1 cos th1 + L2 cos th2 + L3 cos th3
        z = h0 + L1 sin th1 + L2 sin th2 + L3 sin th3

    The link lengths r0, L1, L2 and the tool length L3 are the KNOWN SO-101
    dimensions (URDF + the measured jaw length); constant angle offsets
    inside each link are absorbed by the servo zeros. Only the servo zeros
    and directions, the arm's height above the board and the board's tilt
    are fitted - floor poses must put the jaw tip at the square's distance
    and on the board plane."""

    def __init__(self, prm, signs, sq_mm, fit_sq):
        self.prm, self.signs, self.sq_mm, self.fit_sq = prm, signs, sq_mm, fit_sq
        self.L3 = float(prm[9])

    # prm = [L1, L2, r0, h0, o1, o2, o3, tilt_x, tilt_y, L3, (sq_mm)]
    @staticmethod
    def angles(prm, signs, T):
        s1, s2, s3 = signs
        th1 = s1 * (T[:, 0] - prm[4]) / TICKS_PER_RAD
        th2 = th1 + s2 * (T[:, 1] - prm[5]) / TICKS_PER_RAD
        th3 = th2 + s3 * (T[:, 2] - prm[6]) / TICKS_PER_RAD
        return th1, th2, th3

    @classmethod
    def fk_prm(cls, prm, signs, T, L3=None):
        L3 = prm[9]
        th1, th2, th3 = cls.angles(prm, signs, T)
        r = prm[2] + prm[0] * np.cos(th1) + prm[1] * np.cos(th2) + L3 * np.cos(th3)
        z = prm[3] + prm[0] * np.sin(th1) + prm[1] * np.sin(th2) + L3 * np.sin(th3)
        return r, z, th3

    def fk(self, T):
        return self.fk_prm(self.prm, self.signs, T, self.L3)

    @classmethod
    def fit(cls, floor_T, floor_reach, floor_xy,
            sq_mm: Optional[float], init: Optional["Kinematics"] = None) -> Optional["Kinematics"]:
        L3 = tool_length_mm()
        fit_sq = sq_mm is None
        sq0 = 40.0 if fit_sq else float(sq_mm)

        def resid(prm, signs):
            sq = prm[10] if fit_sq else sq0
            rf, zf, _ = cls.fk_prm(prm, signs, floor_T, L3)
            plane = prm[7] * floor_xy[:, 0] + prm[8] * floor_xy[:, 1]
            out = [rf - floor_reach * sq, zf - plane, [prm[7] * 4, prm[8] * 4]]
            if fit_sq:
                out.append([(prm[10] - 40.0) / 25.0])
            return np.concatenate([np.atleast_1d(np.asarray(o, float)) for o in out])

        # fixed SO-101 geometry; only these are optimised:
        # h0, o1, o2, o3, tilt_x, tilt_y (+ square size if not configured)
        free = [3, 4, 5, 6, 7, 8] + ([10] if fit_sq else [])

        def lm(prm, signs, iters=80):
            lam = 1e-2
            r = resid(prm, signs)
            for _ in range(iters):
                J = np.empty((len(r), len(free)))
                for c, k in enumerate(free):
                    d = np.zeros(len(prm))
                    d[k] = 1e-3 if k in (3, 10) else (1e-2 if k < 7 else 1e-5)
                    J[:, c] = (resid(prm + d, signs) - r) / d[k]
                A = J.T @ J
                step_f = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -J.T @ r)
                step = np.zeros(len(prm))
                step[free] = step_f
                new = prm + step
                rn = resid(new, signs)
                if rn @ rn < r @ r:
                    gain = r @ r - rn @ rn
                    prm, r, lam = new, rn, max(lam * 0.3, 1e-9)
                    if gain < 1e-10:
                        break
                else:
                    lam *= 10
                    if lam > 1e10:
                        break
            return prm, float(r @ r)

        med = np.median(floor_T, axis=0)
        if init is not None:
            starts = [(init.prm.copy(), init.signs)]
        else:
            starts = []
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    for s3 in (1.0, -1.0):
                        for a1 in (20, 50, 80):
                            for a2 in (-60, -100, -140):
                                for ph in (-60, -90, -120):
                                    th1 = math.radians(a1)
                                    th2 = th1 + math.radians(a2)
                                    th3 = math.radians(ph)
                                    o1 = med[0] - s1 * TICKS_PER_RAD * th1
                                    o2 = med[1] - s2 * TICKS_PER_RAD * (th2 - th1)
                                    o3 = med[2] - s3 * TICKS_PER_RAD * (th3 - th2)
                                    prm = [SO101_UPPER_ARM_MM, SO101_FOREARM_MM,
                                           SO101_SHOULDER_OFFSET_MM, 100.0, o1, o2, o3,
                                           0.0, 0.0, L3]
                                    if fit_sq:
                                        prm.append(sq0)
                                    starts.append((np.array(prm), (s1, s2, s3)))
            # cheap screening, then polish the most promising starts
            scored = sorted(((float(resid(p, sg) @ resid(p, sg)), i) for i, (p, sg) in enumerate(starts)))
            starts = [starts[i] for _, i in scored[:16]]
        best = None
        for prm, signs in starts:
            prm, err = lm(prm, signs, 60)
            if best is None or err < best[0]:
                best = (err, prm, signs)
        if best is None:
            return None
        prm, err = lm(best[1], best[2], 400)
        if not np.all(np.isfinite(prm)):
            return None
        k = cls(prm, best[2], float(prm[10]) if fit_sq else sq0, fit_sq)
        k.fit_err = err
        return k

    def reachable(self, r_mm, z_mm, pitch, margin: float = 0.98) -> np.ndarray:
        L1, L2, r0, h0 = self.prm[:4]
        wx = r_mm - r0 - self.L3 * np.cos(pitch)
        wz = z_mm - h0 - self.L3 * np.sin(pitch)
        d = np.hypot(wx, wz)
        return (d <= margin * (L1 + L2)) & (d >= abs(L1 - L2) / margin)

    def ik(self, r_mm, z_mm, pitch, branch, th1_ref):
        """Ticks (t1, t2, t3) putting the jaw tip at (r, z) with tool pitch."""
        L1, L2, r0, h0, o1, o2, o3 = self.prm[:7]
        s1, s2, s3 = self.signs
        wx = r_mm - r0 - self.L3 * np.cos(pitch)
        wz = z_mm - h0 - self.L3 * np.sin(pitch)
        c = np.clip((wx ** 2 + wz ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)
        rel = branch * np.arccos(c)
        th1 = np.arctan2(wz, wx) - np.arctan2(L2 * np.sin(rel), L1 + L2 * np.cos(rel))
        th1 = th1 + 2 * np.pi * np.round((th1_ref - th1) / (2 * np.pi))
        th2 = th1 + rel
        return (o1 + s1 * TICKS_PER_RAD * th1,
                o2 + s2 * TICKS_PER_RAD * rel,
                o3 + s3 * TICKS_PER_RAD * (pitch - th2))


class _KinTarget:
    """Per pose kind: where the tip should be (height plane, pitch vs reach)
    for an untaught square, then IK."""

    def __init__(self, kin: Kinematics, T, reach, xy):
        self.kin = kin
        r, z, pitch = kin.fk(T)
        th1, th2, _ = kin.angles(kin.prm, kin.signs, T)
        self.branch = 1.0 if np.median(th2 - th1) >= 0 else -1.0
        self.th1_ref = float(np.median(th1))
        A = np.c_[np.ones(len(xy)), xy]
        self.zc = np.linalg.lstsq(A, z, rcond=None)[0]            # tip height plane
        B = np.c_[np.ones(len(reach)), reach]
        self.pc = np.linalg.lstsq(B, np.unwrap(pitch), rcond=None)[0]  # pitch vs reach

    def eval(self, reach, xy):
        z = self.zc[0] + xy[:, 0] * self.zc[1] + xy[:, 1] * self.zc[2]
        pitch = self.pc[0] + self.pc[1] * reach
        t1, t2, t3 = self.kin.ik(reach * self.kin.sq_mm, z, pitch, self.branch, self.th1_ref)
        return {"shoulder_lift": t1, "elbow_flex": t2, "wrist_flex": t3}


class ArmModel:
    def __init__(self, floors: Dict[str, Dict[str, float]], square_mm: Optional[float] = None):
        if len(floors) < MIN_ANCHORS:
            raise ValueError(f"need at least {MIN_ANCHORS} taught squares, have {len(floors)}")
        self.taught = set(floors)
        self.floor = _PoseModel(floors)
        f = self.floor
        reach, _ = f._polar(f.xy)
        T = f.radial_T()
        sq = square_mm if square_mm is not None else config.SQUARE_SIZE_MM

        def fit(mask, init=None):
            return Kinematics.fit(T[mask], reach[mask], f.xy[mask], sq, init)
        n = len(reach)
        self.kin = fit(np.ones(n, bool))
        if self.kin is None:
            raise ValueError("could not fit the arm geometry - check the taught "
                             "squares (calibration.py --verify) or teach a few more")
        loo = [fit(np.arange(n) != i, init=self.kin) for i in range(n)]
        f.add_kinematic(self.kin, loo, check_z=True)
        # board plane + gripper branch from ALL taught floors
        self.target = _KinTarget(self.kin, T, reach, f.xy)
        self.lift_mm = config.LIFT_HEIGHT_SQUARES * self.kin.sq_mm
        self.lowered: Dict[str, float] = {}

    def floor_pose(self, name: str) -> Dict[str, float]:
        return self.floor.predict(name)

    def lift_pose(self, floor: Dict[str, float], name: str = "") -> Dict[str, float]:
        """The floor pose raised straight up (same spot, same gripper angle)
        to LIFT_HEIGHT_SQUARES x square above the board plane."""
        k, tg = self.kin, self.target
        T = np.array([[floor["shoulder_lift"], floor["elbow_flex"], floor["wrist_flex"]]], float)
        r, z, pitch = k.fk(T)
        x, y = square_xy(name) if name in chess.SQUARE_NAMES else (4.0, 4.0)
        plane = tg.zc[0] + tg.zc[1] * x + tg.zc[2] * y
        # The jaw tip goes straight up; the gripper may tilt on the way (it
        # turns about the gripping point, so the piece doesn't move sideways).
        # Pick the tilt that reaches the height with the least joint motion.
        p0 = float(pitch[0])
        tilts = [p0] + [p0 + math.radians(d) * sgn for d in (10, 20, 30, 40) for sgn in (1, -1)]

        def best_at(h):
            best = None
            for ph in tilts:
                ph_arr = np.array([ph])
                if not k.reachable(r, plane + h, ph_arr)[0]:
                    continue
                t = k.ik(r, plane + h, ph_arr, tg.branch, tg.th1_ref)
                move = max(abs(float(t[i][0]) - T[0, i]) for i in range(3))
                if move <= config.LIFT_MAX_JOINT_TICKS and (best is None or move < best[0]):
                    best = (move, t)
            return best
        h = self.lift_mm
        found = best_at(h)
        while found is None and h > 0.25 * self.lift_mm:
            h -= 2.0
            found = best_at(h)
        if h < self.lift_mm:
            self.lowered[name] = round(float(h), 1)
        if found is None:  # nothing sane: just hold the floor pose's arm config
            t1, t2, t3 = T[:, 0], T[:, 1], T[:, 2]
        else:
            t1, t2, t3 = found[1]
        out = dict(floor)
        out.update({"shoulder_lift": float(t1[0]), "elbow_flex": float(t2[0]),
                    "wrist_flex": float(t3[0])})
        return out

    def base_position(self) -> Tuple[float, float]:
        """Fitted pan axis location in board squares (a1 corner = 0,0)."""
        return float(self.floor.pan_prm[0]), float(self.floor.pan_prm[1])


def _floor_of(entry):
    """A taught square entry -> its floor pose (accepts pose, [floor], [floor, lift])."""
    return entry[0] if isinstance(entry, list) else entry


def taught_squares(calibration: Dict) -> List[str]:
    if "_taught" in calibration:
        return [s for s in calibration["_taught"] if s in calibration]
    return [s for s in chess.SQUARE_NAMES if s in calibration]


def complete(calibration: Dict) -> Tuple[Dict, Optional[ArmModel]]:
    """Return a calibration with every square as [floor, lift] (taught floor
    poses kept exactly; everything else computed) and the model - or the
    input unchanged if too few squares were taught."""
    taught = taught_squares(calibration)
    floors = {s: _floor_of(calibration[s]) for s in taught}
    try:
        model = ArmModel(floors, square_mm=config.SQUARE_SIZE_MM)
    except ValueError:
        return calibration, None
    out = {k: v for k, v in calibration.items() if k not in chess.SQUARE_NAMES}
    for s in chess.SQUARE_NAMES:
        floor = floors[s] if s in floors else model.floor_pose(s)
        out[s] = [floor, model.lift_pose(floor, s)]
    f = model.floor
    out["_taught"] = sorted(taught)
    out.pop("_square_mm", None)  # obsolete: the size lives in config.py
    out["_model"] = {"stamp": _settings_stamp(),
                     "jaw_mm": round(model.kin.L3, 1),
                     "radial": f.radial_kind,
                     "cv_tip_mm": round(getattr(f, "cv_mm", {}).get(f.radial_kind, 0), 1),
                     "square_mm": round(model.kin.sq_mm, 1),
                     "lift_mm": round(model.lift_mm, 1),
                     "lift_lowered": model.lowered,
                     "base_squares": [round(v, 2) for v in model.base_position()]}
    return out, model


MODEL_VERSION = 5  # bump when the computation changes: stored files get recomputed


def _settings_stamp() -> Dict:
    return {"version": MODEL_VERSION, "square_mm": config.SQUARE_SIZE_MM,
            "lift_squares": config.LIFT_HEIGHT_SQUARES,
            "tip_mm": config.GRIPPER_TIP_MM,
            "max_joint": config.LIFT_MAX_JOINT_TICKS}


def needs_completion(calibration: Dict) -> bool:
    """True if squares are missing, or they were computed by an older model /
    with different settings (square size, lift height, ...)."""
    if calibration.get("_model", {}).get("stamp") != _settings_stamp():
        return True
    return any(not (isinstance(calibration.get(s), list) and len(calibration[s]) == 2)
               for s in chess.SQUARE_NAMES)


def leave_one_out(calibration: Dict, anchors: List[str]) -> Dict[str, Dict[str, float]]:
    """Fit on `anchors` only and report the floor-pose prediction error
    (ticks) on every other calibrated square - for checking against a full
    calibration."""
    floors = {k: _floor_of(calibration[k]) for k in anchors if k in calibration}
    model = ArmModel(floors)
    errs = {}
    for s in chess.SQUARE_NAMES:
        if s in calibration and s not in floors:
            p = model.floor_pose(s)
            errs[s] = {j: abs(p[j] - _floor_of(calibration[s])[j]) for j in ARM}
    return errs
