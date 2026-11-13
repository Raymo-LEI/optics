#!/usr/bin/env python3
"""render_run.py - kinematic MuJoCo video of a FINISHED optics-workbench run.

This tool only ILLUSTRATES the poses recorded in a run directory. The workbench's
algorithm_v1 simulation is a pose proxy with no robot in it; the xArm7 (or the floating
suction tool) drawn here is a stand-in that is moved kinematically between the recorded
poses. No physics, no collision checking, not a robot controller, and no hardware is
involved. The run directory is opened strictly read-only.

Usage:
    render_run.py --run-dir <.../runs/<id>> --out demo-output/<id>.mp4 [--gif]
                  [--robot xarm7|gantry] [--speed 1.0] [--exaggerate 10]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation as ScipyRot
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency ({exc}); see tools/visualize_run/README.md for setup")

HERE = Path(__file__).resolve().parent
XARM_XML = HERE / "assets" / "ufactory_xarm7" / "xarm7.xml"

FOOTER = ("Kinematic visualization of the accepted plan. Not physics, not collision-checked, "
          "not a robot controller. physical_verified = false")
FINAL_NO_HW = "No real hardware moved (motions_sent = 0)"

PLATE_T = 0.004          # breadboard plate thickness (parts' feet sit on its top)
TABLE_T = 0.04            # table slab thickness (top surface is z = 0)
PEDESTAL_H = 0.12         # xArm7 pedestal height (Menagerie scene uses the same)
PEDESTAL_R = 0.06
TAB_HALF = (0.008, 0.015, 0.015)   # grip tab half-sizes (x = across the fingers)
PINCH_LOCAL_Z = 0.155     # pinch point on the gripper z axis (middle of the finger pads)
FINGER_BELOW_PINCH = 0.045  # conservative: how far finger tips reach beyond the pinch point


class UnreachedError(RuntimeError):
    pass


# --------------------------------------------------------------------------- helpers
def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:   # read-only, never write into a run dir
        return json.load(fh)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)   # smootherstep


def Rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def T_of(x, y, z, yaw):
    T = np.eye(4)
    T[:3, :3] = Rz(yaw)
    T[:3, 3] = (x, y, z)
    return T


def T_inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def grasp_R(yaw):
    """Top-down gripper orientation: tool z points down, finger-closing axis (gripper y)
    lies along the part's local x axis (the optical axis), so the gripper yaw follows the part."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[-s, c, 0], [c, s, 0], [0, 0, -1.0]]).T  # columns x=(-s,c,0) y=(c,s,0) z=(0,0,-1)


def fmt_mm(m):
    return f"{m * 1000:.2f} mm" if abs(m) < 0.01 else f"{m * 1000:.1f} mm"


def fmt_deg(r):
    return f"{math.degrees(r):.2f} deg"


def load_font(size, bold=False):
    cands = (["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/HelveticaNeue.ttc",
              "/Library/Fonts/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
             if bold else
             ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/HelveticaNeue.ttc",
              "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for c in cands:
        try:
            return ImageFont.truetype(c, size, index=1 if (bold and c.endswith("Helvetica.ttc")) else 0)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


# --------------------------------------------------------------------------- evidence
@dataclass
class Comp:
    oid: str
    label: str
    kind: str
    role: str
    movable: bool
    target: np.ndarray            # x, y, z, yaw  (metres, rad), from layout.json
    params: dict = field(default_factory=dict)


@dataclass
class Attempt:
    n: int
    actions: list                 # plan actions in plan order (dicts)
    before: dict                  # id -> np.array(x,y,z,yaw)
    after: dict
    moved: list
    pos_err: dict                 # evidence: simulation.json position_errors_m
    ang_err: dict                 # evidence: simulation.json angle_errors_rad
    mech_passed: bool
    limits: dict


@dataclass
class Run:
    run_dir: Path
    run_id: str
    request: str
    status: str
    stage: str
    comps: list
    attempts: list
    result: dict
    tests_passed: int | None
    tests_total: int | None
    deferred_n: int
    seeded: dict
    notes: list = field(default_factory=list)

    def comp(self, oid):
        for c in self.comps:
            if c.oid == oid:
                return c
        raise KeyError(oid)


def pose_of(d):
    p = d["position_m"]
    return np.array([p[0], p[1], p[2], float(d["yaw_rad"])], float)


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_run(run_dir) -> Run:
    rd = Path(run_dir).resolve()
    if not (rd / "run.json").is_file():
        raise SystemExit(f"{rd} has no run.json; pass the runs/<id> directory")
    rj = read_json(rd / "run.json")
    layout = read_json(rd / "layout.json")
    exp = {c["object_id"]: c for c in (rj.get("algorithm_experiment") or {}).get("components", [])}
    if not exp:
        exp = {c["object_id"]: c for c in (rj.get("algorithm_design") or {}).get("components", [])}
    comps = []
    for c in layout["components"]:
        tp = c["target_pose"]
        e = exp.get(c["object_id"], {})
        params = {p["name"]: (p["value"], p.get("unit")) for p in e.get("parameters", [])}
        comps.append(Comp(
            oid=c["object_id"], label=e.get("label") or c["object_id"], kind=e.get("kind") or "",
            role=e.get("role") or "", movable=bool(c["movable"]),
            target=np.array([*tp["position_m"], yaw_from_quat(tp["orientation_xyzw"])], float),
            params=params))
    sims = sorted((rd / "team").glob("sim-*"), key=lambda p: int(p.name.split("-")[1]))
    attempts = []
    for sd in sims:
        sim = read_json(sd / "simulation.json")
        plan_path = sd / "plan.json"
        if not plan_path.is_file():
            plan_path = rd / "team" / f"controller-{sd.name.split('-')[1]}" / "plan.json"
        plan = read_json(plan_path)
        attempts.append(Attempt(
            n=int(sim["attempt"]), actions=plan["actions"],
            before={k: pose_of(v) for k, v in sim["before"].items()},
            after={k: pose_of(v) for k, v in sim["after"].items()},
            moved=list(sim["moved_ids"]), pos_err=sim["position_errors_m"],
            ang_err=sim["angle_errors_rad"], mech_passed=bool(sim.get("mechanical_passed")),
            limits=sim.get("limits", {})))
    if not attempts:
        raise SystemExit("run has no team/sim-*/simulation.json: nothing to replay")
    rep = rj.get("algorithm_test_report") or {}
    tests = rep.get("tests") or []
    res = rj.get("result") or (read_json(rd / "result.json") if (rd / "result.json").is_file() else {})
    ver = rj.get("verification") or {}
    st = rj.get("algorithm_state") or {}
    seeded = {"id": st.get("seeded_object_id"),
              "pos_m": (rj.get("algorithm_options") or {}).get("initial_position_error_m"),
              "ang_rad": (rj.get("algorithm_options") or {}).get("initial_angle_error_rad")}
    return Run(
        run_dir=rd, run_id=rj.get("run_id") or rd.name,
        request=(rj.get("request") or read_json(rd / "input.json").get("request", "")),
        status=str(rj.get("status")), stage=str(rj.get("stage")), comps=comps, attempts=attempts,
        result=res,
        tests_passed=(sum(1 for t in tests if t.get("status") == "passed") if tests else None),
        tests_total=(len(tests) if tests else None),
        deferred_n=len(ver.get("deferred_optical_requirements") or rep.get("deferred") or []),
        seeded=seeded)


# --------------------------------------------------------------------------- component look
def comp_class(c: Comp) -> str:
    s = f"{c.kind} {c.oid} {c.label} {c.role}".lower()
    for key, cls in (("breadboard", "support"), ("support", "support"), ("table", "support"),
                     ("beamsplitter", "beamsplitter"), ("beam_splitter", "beamsplitter"),
                     ("splitter", "beamsplitter"), ("mirror", "mirror"), ("lens", "lens"),
                     ("laser", "laser"), ("transparency", "transparency"),
                     ("image_sensor", "detector"), ("detector", "detector"), ("sensor", "detector"),
                     ("camera", "detector"), ("photodiode", "detector")):
        if key in s:
            return cls
    if c.oid.lower().startswith(("bs",)):
        return "beamsplitter"
    if c.oid.lower().startswith(("m", "mirror")) and len(c.oid) <= 3:
        return "mirror"
    if c.oid.lower().startswith("l") and len(c.oid) <= 3:
        return "lens"
    return "generic"


PALETTE = {"laser": (0.80, 0.18, 0.18, 1), "lens": (0.35, 0.78, 0.95, 0.80),
           "mirror": (0.78, 0.82, 0.90, 1), "beamsplitter": (0.98, 0.72, 0.18, 0.88),
           "detector": (0.20, 0.72, 0.38, 1), "transparency": (0.65, 0.40, 0.90, 0.85),
           "generic": (0.62, 0.62, 0.66, 1), "support": (0.80, 0.82, 0.85, 1)}


def comp_style(c: Comp, s: float = 1.0):
    """Return dict: cls, parts [(type, half-size(list), local pos, local euler/quat or None, rgba)],
    h (optic centre height), top (top of optic), tab_z (grip-tab centre height), color."""
    cls = comp_class(c)
    h = float(c.target[2]) if c.target[2] > 0.02 else 0.05
    h *= s
    col = PALETTE[cls]
    parts = []
    if cls == "lens":
        dia = c.params.get("diameter")
        r = (dia[0] * 1e-3 / 2) if (dia and dia[1] == "mm") else 0.0127
        r = max(r, 0.012) * s
        parts.append(("cylinder", [r, 0.005 * s], [0, 0, 0], "x", col))
        half_v = r
    elif cls == "mirror":
        parts.append(("box", [0.0025 * s, 0.025 * s, 0.025 * s], [0, 0, 0], None, col))
        half_v = 0.025 * s
    elif cls == "beamsplitter":
        a = 0.0155 * s
        parts.append(("box", [a, a, a], [0, 0, 0], "z45", (0.62, 0.84, 0.95, 0.45)))
        parts.append(("box", [0.0012 * s, 0.0215 * s, a * 0.98], [0, 0, 0], None, col))
        half_v = a
    elif cls == "laser":
        parts.append(("box", [0.06 * s, 0.025 * s, 0.025 * s], [0, 0, 0], None, col))
        parts.append(("cylinder", [0.011 * s, 0.004 * s], [0.064 * s, 0, 0], "x", (0.1, 0.1, 0.1, 1)))
        half_v = 0.025 * s
    elif cls == "detector":
        parts.append(("box", [0.014 * s, 0.022 * s, 0.022 * s], [0, 0, 0], None, col))
        parts.append(("box", [0.002 * s, 0.014 * s, 0.014 * s], [0.015 * s, 0, 0], None, (0.05, 0.1, 0.1, 1)))
        half_v = 0.022 * s
    elif cls == "transparency":
        parts.append(("box", [0.0025 * s, 0.026 * s, 0.026 * s], [0, 0, 0], None, col))
        parts.append(("box", [0.003 * s, 0.016 * s, 0.016 * s], [0, 0, 0], None, (0.85, 0.75, 0.98, 0.9)))
        half_v = 0.026 * s
    else:
        parts.append(("box", [0.015 * s] * 3, [0, 0, 0], None, col))
        half_v = 0.015 * s
    top = h + half_v
    return dict(cls=cls, parts=parts, h=h, top=top, bottom=h - half_v, tab_z=top + TAB_HALF[2], color=col, movable=c.movable)


# --------------------------------------------------------------------------- story
def exaggerate(pose, target, k, pos_thr=0.02, ang_thr=0.2):
    """Display-only scaling of a SMALL residual offset from its layout target by factor k."""
    if k == 1:
        return pose.copy()
    d = pose[:3] - target[:3]
    dy = wrap(pose[3] - target[3])
    if np.linalg.norm(d[:2]) < pos_thr and abs(d[2]) < pos_thr and abs(dy) < ang_thr:
        out = target.copy()
        out[:3] = target[:3] + k * d
        out[3] = wrap(target[3] + k * dy)
        return out
    return pose.copy()


@dataclass
class Step:
    attempt: int
    index: int                 # position within attempt
    op: str
    obj: str
    p0: np.ndarray             # TRUE (evidence) start pose
    p1: np.ndarray             # TRUE (evidence) end pose
    is_corr: bool
    err_before: tuple | None   # (mm, rad) from evidence, None if part came from the feed row
    err_after: tuple
    tgt_gap: tuple             # true (m, rad) of p1 from target (computed, cross-check)


def build_steps(run: Run):
    steps = []
    notes = []
    for a in run.attempts:
        prev = run.attempts[run.attempts.index(a) - 1] if a is not run.attempts[0] else None
        order = [x["object_id"] for x in a.actions]
        ops = {x["object_id"]: x["operation"] for x in a.actions}
        for mid in a.moved:
            if mid not in order:
                order.append(mid)
                ops[mid] = "move"
                notes.append(f"attempt {a.n}: moved id {mid} has no plan action (shown as 'move')")
        for i, oid in enumerate(order):
            if oid not in a.moved:
                notes.append(f"attempt {a.n}: plan action on {oid} but it is not in moved_ids; not animated")
                continue
            comp = run.comp(oid)
            p0, p1 = a.before[oid], a.after[oid]
            eb = None
            if prev is not None and oid in prev.pos_err:
                eb = (prev.pos_err[oid], prev.ang_err[oid])
            ea = (a.pos_err[oid], a.ang_err[oid])
            gap = (float(np.linalg.norm(p1[:3] - comp.target[:3])), abs(wrap(p1[3] - comp.target[3])))
            if abs(gap[0] - ea[0]) > 1e-6 or abs(gap[1] - ea[1]) > 1e-6:
                notes.append(f"attempt {a.n} {oid}: evidence error ({ea[0]:.6f} m, {ea[1]:.6f} rad) differs "
                             f"from pose-vs-layout ({gap[0]:.6f}, {gap[1]:.6f}); overlay shows the evidence field")
            steps.append(Step(a.n, i, ops[oid], oid, p0, p1, a is not run.attempts[0], eb, ea, gap))
    return steps, notes




# --------------------------------------------------------------------------- IK
class ArmIK:
    """Damped-least-squares IK on the 'pinch' site with a rotation-matrix (rotation-vector)
    orientation error, joint limits, a nullspace posture term, warm start and multi restart."""

    def __init__(self, base_xy):
        spec = mujoco.MjSpec.from_file(str(XARM_XML))
        spec.body("link_base").pos = [float(base_xy[0]), float(base_xy[1]), PEDESTAL_H]
        spec.body("xarm_gripper_base_link").add_site(name="pinch", pos=[0, 0, PINCH_LOCAL_Z], size=[0.006] * 3)
        # table slab so that arm-vs-table penetration also counts as a rejected IK solution
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -0.02], size=[3, 3, 0.02],
                                contype=1, conaffinity=1)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("pinch").id
        jn = [self.m.joint(f"joint{i}").id for i in range(1, 8)]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jn])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in jn])
        lim = np.array([self.m.jnt_range[j] for j in jn])
        self.lo, self.hi = lim[:, 0] + 1e-3, lim[:, 1] - 1e-3
        self.q_home = np.array(self.m.key("home").qpos[:7])
        self.q_ref = np.array([0.0, 0.25, 0.0, 1.0, 0.0, 0.8, 0.0])  # relaxed elbow-up posture
        self.rng = np.random.default_rng(7)

    def collides(self, q):
        """True if the arm intersects itself or the table at q (gripper open). A guard for IK solutions
        (development aid), not a collision checker for the whole scene."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        return self.d.ncon > 0

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def _run(self, pos, R, q0, tol_p, tol_r, iters):
        q = np.clip(np.array(q0, float), self.lo, self.hi)
        J = np.zeros((6, self.m.nv))
        best = None
        for it in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            pc = self.d.site_xpos[self.site]
            Rc = self.d.site_xmat[self.site].reshape(3, 3)
            ep = pos - pc
            er = ScipyRot.from_matrix(R @ Rc.T).as_rotvec()   # world-frame rotation error (axis*angle)
            np_, nr = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if best is None or np_ + nr < best[1] + best[2]:
                best = (q.copy(), np_, nr, it)
            if np_ < tol_p and nr < tol_r:
                return True, q, np_, nr, it
            mujoco.mj_jacSite(self.m, self.d, J[:3], J[3:], self.site)
            Ja = J[:, self.dadr]
            e = np.concatenate([ep, er])
            lam2 = 1e-4 if (np_ > 0.02 or nr > 0.1) else 1e-6
            JJt = Ja @ Ja.T + lam2 * np.eye(6)
            Jp = Ja.T @ np.linalg.inv(JJt)
            dq = Jp @ e + (np.eye(7) - Jp @ Ja) @ (0.15 * (self.q_ref - q))
            mx = np.max(np.abs(dq))
            if mx > 0.25:
                dq *= 0.25 / mx
            q = np.clip(q + dq, self.lo, self.hi)
        return False, best[0], best[1], best[2], best[3]

    def solve(self, pos, R, q0, tol_p=1e-4, tol_r=1e-3, iters=300, restarts=8):
        """Warm start first; if that converges without arm self/table intersection, use it. Otherwise try
        other starts and keep the acceptable converged solution nearest to q0 (joint-space continuity)."""
        ok, q, ep, er, it = self._run(pos, R, q0, tol_p, tol_r, iters)
        if ok and not self.collides(q):
            return ok, q, ep, er, it, 0
        starts = [self.q_ref, self.q_home]
        for _ in range(restarts):
            starts.append(np.clip(self.q_ref + self.rng.normal(0, 0.6, 7), self.lo, self.hi))
        best, fallback = None, (False, q, ep, er, it, 0)
        for si, s in enumerate(starts, 1):
            ok, q, ep, er, it = self._run(pos, R, s, tol_p, tol_r, iters)
            if ok and not self.collides(q):
                dq = float(np.linalg.norm(q - q0))
                if best is None or dq < best[0]:
                    best = (dq, (ok, q, ep, er, it, si))
            elif ep + er < fallback[2] + fallback[3]:
                fallback = (False, q, ep, er, it, si)
        return best[1] if best else fallback


# --------------------------------------------------------------------------- planning
@dataclass
class Seg:
    kind: str                   # hold | move | grip | final
    dur: float
    label: str = ""
    path: np.ndarray | None = None     # (n,7) arm q, or (n,4) gantry x,y,z,yaw
    th0: float = 0.0
    th1: float = 0.0
    attach: str | None = None
    release: tuple | None = None       # (obj, display pose to snap to)
    ov: dict | None = None
    marker: str | None = None          # named frame to remember (at the END of the segment)


def theta_for_gap(gap):
    """Gripper joint angle for a pad-to-pad gap (m). Linear fit of FK-measured pad position:
    pad inner face = 0.0465 - 0.0516*theta - 0.002 (theta in rad, 0 = open, 0.85 = closed)."""
    y = gap / 2 + 0.002
    return float(np.clip((0.0465 - y) / 0.0516, 0.0, 0.85))


class Planner:
    def __init__(self, run, steps, k_exag, robot, base_xy, scale, allow_unreached, speed):
        self.run, self.steps, self.k, self.robot = run, steps, k_exag, robot
        self.s = scale
        self.allow_unreached = allow_unreached
        self.speed = speed
        self.base_xy = base_xy
        self.styles = {c.oid: comp_style(c, scale) for c in run.comps}
        self.ik = ArmIK(base_xy) if robot == "xarm7" else None
        self.ik_log = []
        self.jump_log = []
        self.unreached = []
        tops = [(st["top"] + (2 * TAB_HALF[2] if st["movable"] else 0)) for st in self.styles.values()
                if st["cls"] != "support"]
        self.max_top = max(tops)
        self.z_free = self.max_top + 0.04 + FINGER_BELOW_PINCH      # free travel: fingertips clear everything
        self.lift_carry = self.max_top + 0.03                       # carried part's bottom clears everything
        self.gap = 2 * TAB_HALF[0]
        self.th_close = theta_for_gap(self.gap)
        self.th_open = theta_for_gap(self.gap + 0.030)
        self.q_park = None
        if self.ik is not None:
            # park posture: hovering above the middle of the layout, gripper pointing down
            c = np.mean([c.target[:2] for c in run.comps], axis=0) * scale
            ok, q, *_ = self.ik.solve(np.array([c[0], c[1], self.z_free + 0.08]), grasp_R(0.0), self.ik.q_ref)
            self.q_park = q if ok else self.ik.q_home.copy()

    def disp(self, oid, pose):
        """display pose: exaggerated small residual offset, then uniform scene scale."""
        p = exaggerate(pose, self.run.comp(oid).target, self.k).copy()
        p[:3] *= self.s
        return p

    def target_disp(self, oid):
        p = self.run.comp(oid).target.copy()
        p[:3] *= self.s
        return p

    # ---- waypoint -> robot state
    def solve_wp(self, label, pos, yaw, q_prev):
        if self.ik is None:
            return np.array([pos[0], pos[1], pos[2], yaw], float)
        ok, q, ep, er, it, si = self.ik.solve(np.asarray(pos, float), grasp_R(yaw), q_prev)
        self.ik_log.append(dict(label=label, ok=bool(ok), pos_err_m=float(ep), rot_err_rad=float(er),
                                iters=int(it), start=int(si)))
        if not ok:
            raise UnreachedError(
                f"IK did not converge at waypoint '{label}': target pinch ({pos[0]:.3f}, {pos[1]:.3f}, "
                f"{pos[2]:.3f}) m, gripper yaw {math.degrees(yaw):.1f} deg; best residual {ep * 1000:.2f} mm / "
                f"{math.degrees(er):.2f} deg (arm base at {self.base_xy[0]:.3f}, {self.base_xy[1]:.3f} m, "
                f"scene scale {self.s:g})")
        return q

    def path_between(self, label, pA, pB, state_prev, max_step=0.05, max_dyaw=0.35):
        """Straight Cartesian line (pose = x,y,z,yaw), each sub-waypoint solved with a warm start.
        A joint-space jump above 0.9 rad (IK branch flip) triggers a finer subdivision; if it persists
        the move is reported as unreachable rather than animated as a flip."""
        pA, pB = np.asarray(pA, float), np.asarray(pB, float)
        dist = float(np.linalg.norm(pB[:3] - pA[:3]))
        n0 = max(1, int(math.ceil(max(dist / max_step, abs(pB[3] - pA[3]) / max_dyaw))))
        for mult in (1, 2, 4):
            n = n0 * mult
            mark = len(self.ik_log)
            out, prev, flip = [], state_prev, None
            for i in range(1, n + 1):
                p = pA + (pB - pA) * (i / n)
                q = self.solve_wp(f"{label} [{i}/{n}]", p[:3], p[3], prev)
                jump = float(np.max(np.abs(q - prev))) if self.ik is not None else 0.0
                if jump > 0.9:
                    flip = (f"{label} [{i}/{n}]", jump)
                    break
                out.append((f"{label} [{i}/{n}]", jump))
                prev = q
                out[-1] = (out[-1][0], out[-1][1], q)
            if flip is None:
                self.jump_log.extend((a, b) for a, b, _ in out)
                return np.array([q for _, _, q in out])
            del self.ik_log[mark:]
        raise UnreachedError(f"IK branch flip ({flip[1]:.2f} rad joint jump) at waypoint '{flip[0]}' that "
                             f"finer subdivision did not remove; arm base at ({self.base_xy[0]:.3f}, "
                             f"{self.base_xy[1]:.3f}) m")

    def state_xyz(self, s):
        return self.ik.fk(s)[0] if self.ik is not None else np.asarray(s[:3], float)

    def dur(self, path, prev, slow=False):
        pts = [self.state_xyz(s) for s in np.vstack([np.asarray(prev)[None], path])]
        L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        yaw = 0.0 if self.ik is not None else float(np.sum(np.abs(np.diff(np.r_[prev[3], path[:, 3]]))))
        v = 0.11 if slow else 0.30
        return max(L / v + yaw / 1.2, 0.45) / self.speed

    # ---- the whole story
    def plan(self):
        run = self.run
        tl = []
        ik = self.ik
        cur = {c.oid: self.disp(c.oid, run.attempts[0].before.get(c.oid, c.target)) for c in run.comps}
        self.initial_disp = {k: v.copy() for k, v in cur.items()}
        n_att = len(run.attempts)
        cx = np.mean([c.target[0] for c in run.comps]) * self.s
        cy = np.mean([c.target[1] for c in run.comps]) * self.s
        if ik is not None:
            state = self.q_park.copy()
            pos_prev, gy_prev = ik.fk(state)[0], None
        else:
            state = np.array([cx, cy, self.z_free, 0.0])
            pos_prev, gy_prev = state[:3].copy(), 0.0
        tl.append(Seg("hold", 5.0 / self.speed, "intro", ov=dict(kind="intro"), marker="intro"))
        for st in self.steps:
            ov = self.step_overlay(st, n_att)
            sty = self.styles[st.obj]
            P0 = cur[st.obj].copy()
            P1 = self.disp(st.obj, st.p1)
            if st.is_corr:
                mid = 0.5 * (P0[:2] + P1[:2])
                span = float(np.linalg.norm(P1[:2] - P0[:2]))
                ov["pip"] = dict(look=[float(mid[0]), float(mid[1]), sty["h"] - 0.03], dist=0.28 + 1.6 * span, obj=st.obj)
            lift = 0.04 if st.is_corr else self.lift_carry
            htab = sty["tab_z"]
            zc = htab + lift
            zh = self.z_free
            n_log, n_jump = len(self.ik_log), len(self.jump_log)
            cands = [P0[3], P0[3] + math.pi]          # tab is 180-degree symmetric
            if gy_prev is not None:
                cands = [gy_prev + wrap(c - gy_prev) for c in cands]
                cands.sort(key=lambda c: abs(c - gy_prev))
            done = False
            last_err = None
            for gy0 in cands:
                try:
                    segs, state_end, gy1 = self._plan_step(st, ov, P0, P1, gy0, state, pos_prev, gy_prev, htab, zc, zh)
                    done = True
                    break
                except UnreachedError as exc:
                    last_err = exc
                    del self.ik_log[n_log:]
                    del self.jump_log[n_jump:]
            if not done:
                if not self.allow_unreached:
                    raise last_err
                ov2 = dict(ov)
                ov2["unreached"] = str(last_err)
                self.unreached.append(dict(attempt=st.attempt, obj=st.obj, op=st.op, reason=str(last_err)))
                tl.append(Seg("hold", 3.0 / self.speed, "UNREACHED", ov=ov2))
                continue
            tl.extend(segs)
            state = state_end
            pos_prev = self.state_xyz(state)
            gy_prev = gy1
            cur[st.obj] = P1.copy()
        tl.append(Seg("hold", 1.0 / self.speed, "settle", ov=self.final_overlay(n_att)))
        if ik is not None:
            tl.append(Seg("move", 2.5 / self.speed, "return to park pose", path=np.array([self.q_park]),
                          ov=self.final_overlay(n_att)))
        tl.append(Seg("final", 9.0 / self.speed, "final card", ov=dict(kind="final"), marker="final"))
        self.final_display = {k: v.copy() for k, v in cur.items()}
        return tl

    def _plan_step(self, st, ov, P0, P1, gy0, state, pos_prev, gy_prev, htab, zc, zh):
        lab = f"attempt {st.attempt} {st.op} {st.obj}"
        gy1 = gy0 + wrap(P1[3] - P0[3])
        segs = []
        first = gy_prev is None
        if first:
            qa = self.solve_wp(f"{lab}: above pick", [P0[0], P0[1], zh], gy0, state)
            p1 = np.array([qa])
        else:
            p1 = self.path_between(f"{lab}: to above pick", [*pos_prev[:2], zh, gy_prev],
                                   [P0[0], P0[1], zh, gy0], state)
        banner = Seg("hold", (1.6 if st.is_corr else 0.7) / self.speed, "banner", ov=ov)
        segs.append(banner)
        segs.append(Seg("move", self.dur(p1, state), f"{lab}: to above pick", path=p1, ov=ov))
        s = p1[-1]
        segs.append(Seg("grip", 0.5 / self.speed, "open", th0=0.0 if first else self.th_open, th1=self.th_open, ov=ov))
        p2 = self.path_between(f"{lab}: descend to pick", [P0[0], P0[1], zh, gy0], [P0[0], P0[1], htab, gy0], s)
        segs.append(Seg("move", self.dur(p2, s, True), f"{lab}: descend to pick", path=p2, ov=ov, marker=f"pick:{st.attempt}:{st.obj}"))
        s = p2[-1]
        segs.append(Seg("grip", 0.55 / self.speed, "close", th0=self.th_open, th1=self.th_close, ov=ov))
        p3 = self.path_between(f"{lab}: lift", [P0[0], P0[1], htab, gy0], [P0[0], P0[1], zc, gy0], s)
        segs.append(Seg("move", self.dur(p3, s, True), f"{lab}: lift", path=p3, attach=st.obj, ov=ov))
        s = p3[-1]
        p4 = self.path_between(f"{lab}: carry", [P0[0], P0[1], zc, gy0], [P1[0], P1[1], zc, gy1], s,
                               max_step=0.02 if st.is_corr else 0.05)
        segs.append(Seg("move", self.dur(p4, s, st.is_corr), f"{lab}: carry", path=p4, ov=ov,
                        marker=f"carry:{st.attempt}:{st.obj}"))
        s = p4[-1]
        p5 = self.path_between(f"{lab}: descend to place", [P1[0], P1[1], zc, gy1], [P1[0], P1[1], htab, gy1], s)
        segs.append(Seg("move", self.dur(p5, s, True), f"{lab}: descend to place", path=p5, ov=ov,
                        marker=f"place:{st.attempt}:{st.obj}"))
        s = p5[-1]
        segs.append(Seg("grip", 0.55 / self.speed, "release", th0=self.th_close, th1=self.th_open,
                        release=(st.obj, P1.copy()), ov=ov, marker=f"release:{st.attempt}:{st.obj}"))
        p6 = self.path_between(f"{lab}: retreat", [P1[0], P1[1], htab, gy1], [P1[0], P1[1], zh, gy1], s)
        segs.append(Seg("move", self.dur(p6, s, True), f"{lab}: retreat", path=p6, ov=ov))
        return segs, p6[-1], gy1

    # ---- overlay texts (evidence-driven)
    def final_overlay(self, n_att):
        return dict(kind="step", attempt=n_att, n_att=n_att, title="All recorded attempts shown", err="",
                    exag="", pip=None)

    def step_overlay(self, st, n_att):
        a = st.attempt
        title = f"Attempt {a} of {n_att}: " + (f"place {st.obj}" if st.op == "place" else f"{st.op} {st.obj}")
        ea = st.err_after
        if st.err_before is not None:
            eb = st.err_before
            err = (f"Measured error of {st.obj} (simulation.json): position {fmt_mm(eb[0])} -> {fmt_mm(ea[0])}"
                   f",  angle {fmt_deg(eb[1])} -> {fmt_deg(ea[1])}")
        else:
            err = (f"Part comes from the feed row. Error of {st.obj} after this placement (simulation.json): "
                   f"position {fmt_mm(ea[0])}, angle {fmt_deg(ea[1])}")
        exag = ""
        eb0 = st.err_before or (0.0, 0.0)
        real_m, real_a = max(ea[0], eb0[0]), max(ea[1], eb0[1])
        if self.k != 1 and (real_m > 0 or real_a > 0):
            exag = (f"Offset from target drawn x{self.k:g} for visibility (real: {fmt_mm(real_m)}, "
                    f"{fmt_deg(real_a)}); ghost = layout target")
        return dict(kind="step", attempt=a, n_att=n_att, title=title, err=err, exag=exag, obj=st.obj, pip=None)


# --------------------------------------------------------------------------- reach / base choice
def choose_base(run, steps, robot, scale):
    """Candidate base positions (xy), best first: outside the part cluster preferred, every pick/place point
    in [0.20 m, r] of the base, ranked by the farthest point. Returns (list, info)."""
    pts = np.array([list(st.p0[:2]) for st in steps] + [list(st.p1[:2]) for st in steps]) * scale
    allc = np.array([[c.target[0], c.target[1]] for c in run.comps if comp_class(c) != "support"] +
                    [list(st.p0[:2]) for st in steps] + [list(st.p1[:2]) for st in steps]) * scale
    lo, hi = allc.min(0), allc.max(0)
    if robot != "xarm7":
        return [np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])], dict(method="n/a (gantry)")
    cands = []
    for x in np.arange(lo[0] - 0.5, hi[0] + 0.501, 0.025):
        for y in np.arange(lo[1] - 0.5, hi[1] + 0.501, 0.025):
            p = np.array([x, y])
            if float(np.min(np.linalg.norm(allc - p, axis=1))) < 0.17:
                continue                                  # keep the pedestal clear of every part
            dist = np.linalg.norm(pts - p, axis=1)
            if dist.min() < 0.28:
                continue
            inside = lo[0] - 0.03 < x < hi[0] + 0.03 and lo[1] - 0.03 < y < hi[1] + 0.03
            dmin_all = float(np.min(np.linalg.norm(allc - p, axis=1)))
            # shortest farthest-reach, edge-mounted (outside the part cluster), and not crowding the parts
            score = float(dist.max()) + (0.08 if inside else 0.0) + 0.6 * max(0.0, 0.32 - dmin_all)
            cands.append((score, p, float(dist.max())))
    cands.sort(key=lambda c: c[0])
    return [(c[1], c[2]) for c in cands], dict(method="search", n_candidates=len(cands))


def static_reach_ok(run, steps, base, k, scale):
    """Check every pick/place xy is IK-reachable at grasp, near-grasp and hover heights.
    Returns (ok, label of the first failing waypoint)."""
    pl = Planner(run, steps, k, "xarm7", base, scale, True, 1.0)
    ik, q = pl.ik, pl.ik.q_home.copy()
    for st in steps:
        sty = pl.styles[st.obj]
        for what, pose in (("pick", pl.disp(st.obj, st.p0)), ("place", pl.disp(st.obj, st.p1))):
            for hname, z in (("hover", pl.z_free), ("near", sty["tab_z"] + 0.04), ("grasp", sty["tab_z"])):
                good, res = False, None
                for alt in (0, 1):
                    res = ik.solve(np.array([pose[0], pose[1], z]), grasp_R(pose[3] + alt * math.pi), q,
                                   restarts=2, iters=150)
                    if res[0]:
                        q, good = res[1], True
                        break
                if not good:
                    return False, (f"attempt {st.attempt} {st.op} {st.obj}: {what} {hname} waypoint at "
                                   f"({pose[0]:.3f}, {pose[1]:.3f}, {z:.3f}) m, base ({base[0]:.3f}, {base[1]:.3f}) m, "
                                   f"residual {res[2] * 1000:.1f} mm / {math.degrees(res[3]):.1f} deg")
    return True, ""


def plan_with_reach(run, steps, args, log):
    """Find (base, scale, planner, timeline). Never silently distorts: any scale < 1 is reported."""
    if args.robot != "xarm7":
        pl = Planner(run, steps, args.exaggerate, "gantry", np.array([0.0, 0.0]), 1.0, args.allow_unreached, args.speed)
        return pl, pl.plan(), dict(robot="gantry", base_xy=None, scale=1.0, note="floating suction tool, no arm reach limit")
    if args.force_scale:
        scales = [args.force_scale]
    else:
        scales = [1.0] if args.base else [1.0, 0.9, 0.8, 0.7, 0.6]
    first_fail = None
    fallback = None
    for s in scales:
        if args.base:
            cands, binfo = [(np.array(args.base, float), 0.0)], dict(method="forced via --base")
        else:
            cands, binfo = choose_base(run, steps, "xarm7", s)
        if cands and fallback is None:
            fallback = (cands[0], s, binfo)
        tried = 0
        for base, dmax in cands:
            if tried >= 40:
                break
            tried += 1
            ok, why = static_reach_ok(run, steps, base, args.exaggerate, s)
            if not ok:
                first_fail = first_fail or why
                continue
            try:
                pl = Planner(run, steps, args.exaggerate, "xarm7", base, s, args.allow_unreached, args.speed)
                tl = pl.plan()
            except UnreachedError as exc:
                first_fail = first_fail or str(exc)
                log(f"  base ({base[0]:.3f},{base[1]:.3f}) passed the static check but planning failed: {exc}")
                continue
            info = dict(robot="xarm7", base_xy=[float(base[0]), float(base[1])], scale=s,
                        farthest_pick_place_m=dmax, candidates_tried=tried, method=binfo["method"])
            return pl, tl, info
        log(f"  scale {s:g}: no reachable base position among {tried} candidate(s)")
    if args.allow_unreached and fallback is not None:
        (base, dmax), s, binfo = fallback
        log("  --allow-unreached: planning with the best-ranked base; unreachable moves will be stamped UNREACHED")
        pl = Planner(run, steps, args.exaggerate, "xarm7", base, s, True, args.speed)
        tl = pl.plan()
        return pl, tl, dict(robot="xarm7", base_xy=[float(base[0]), float(base[1])], scale=s,
                            farthest_pick_place_m=dmax, candidates_tried=0, method=binfo["method"] + " (unreached allowed)")
    raise SystemExit("cannot reach every pick/place pose with the xArm7 from any tested base position "
                     f"(scales tried: {', '.join(f'{x:g}' for x in scales)}).\nFirst failing waypoint: "
                     f"{first_fail}\nUse --robot gantry, --base X Y, a smaller --force-scale, or --allow-unreached "
                     "to skip unreachable moves (they are stamped UNREACHED).")


# --------------------------------------------------------------------------- scene
@dataclass
class Scene:
    model: object
    data: object
    robot: str
    part_qadr: dict
    part_body: dict
    arm_qadr: np.ndarray
    grip_qadr: np.ndarray
    tool_qadr: int
    site_id: int
    cup_geom: int
    table_box: tuple
    base_xy: object
    arm_bodies: set
    ik: object


def build_scene(run, pl: Planner, W, H, robot):
    s = pl.s
    spec = mujoco.MjSpec.from_file(str(XARM_XML)) if robot == "xarm7" else mujoco.MjSpec()
    spec.visual.global_.offwidth = max(W, 1280)
    spec.visual.global_.offheight = max(H, 720)
    spec.visual.headlight.ambient = [0.42, 0.42, 0.45]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.rgba.haze = [0.80, 0.86, 0.93, 1]
    wb = spec.worldbody
    GEOM = mujoco.mjtGeom
    # textures / materials
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.55, 0.70, 0.88], rgb2=[0.93, 0.95, 0.98], width=512, height=3072)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.36, 0.40, 0.46], rgb2=[0.33, 0.37, 0.43],
                     markrgb=[0.55, 0.60, 0.68], width=256, height=256)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.50, 0.53, 0.58], rgb2=[0.44, 0.47, 0.52], width=256, height=256)
    mt = spec.add_material(name="m_table", texuniform=True, texrepeat=[10, 10], reflectance=0.05)
    mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_texture(name="plate", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.22, 0.26, 0.31], rgb2=[0.20, 0.24, 0.29],
                     markrgb=[0.50, 0.56, 0.64], width=256, height=256)
    mp = spec.add_material(name="m_plate", texuniform=True, texrepeat=[20, 20], reflectance=0.0)
    mp.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "plate"
    mf = spec.add_material(name="m_floor", texuniform=True, texrepeat=[2, 2], reflectance=0.0)
    mf.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"

    # table sized to the layout (all poses incl. feed row, the arm base) + margin
    pts = [c.target[:2] * s for c in run.comps]
    for a in run.attempts:
        for d in (a.before, a.after):
            pts += [p[:2] * s for p in d.values()]
    if robot == "xarm7":
        pts.append(np.array(pl.base_xy))
    P = np.array(pts)
    lo, hi = P.min(0) - 0.16, P.max(0) + 0.16
    lo = np.minimum(lo, [-0.0, -0.0] if False else lo)
    if hi[0] - lo[0] < 0.9:
        c = (hi[0] + lo[0]) / 2; lo[0], hi[0] = c - 0.45, c + 0.45
    if hi[1] - lo[1] < 0.7:
        c = (hi[1] + lo[1]) / 2; lo[1], hi[1] = c - 0.35, c + 0.35
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    LEG = 0.72
    wb.add_geom(name="table", type=GEOM.mjGEOM_BOX, pos=[ctr[0], ctr[1], -TABLE_T / 2], size=[half[0], half[1], TABLE_T / 2],
                material="m_table", contype=4, conaffinity=1)
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        wb.add_geom(name=f"leg{i}", type=GEOM.mjGEOM_BOX, pos=[ctr[0] + sx * (half[0] - 0.04), ctr[1] + sy * (half[1] - 0.04), -TABLE_T - LEG / 2],
                    size=[0.03, 0.03, LEG / 2], rgba=[0.25, 0.27, 0.30, 1], contype=0, conaffinity=0)
    wb.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, pos=[ctr[0], ctr[1], -TABLE_T - LEG], size=[4, 4, 0.1],
                material="m_floor", contype=0, conaffinity=0)
    # breadboard: a thin plate sized from the other layout components
    for c in run.comps:
        if comp_class(c) == "support":
            others = np.array([o.target[:2] * s for o in run.comps if comp_class(o) != "support"])
            blo, bhi = others.min(0) - 0.07, others.max(0) + 0.07
            cc = c.target[:2] * s
            hx = max(abs(blo[0] - cc[0]), abs(bhi[0] - cc[0]))
            hy = max(abs(blo[1] - cc[1]), abs(bhi[1] - cc[1]))
            wb.add_geom(name=f"plate_{c.oid}", type=GEOM.mjGEOM_BOX, pos=[cc[0], cc[1], PLATE_T / 2], size=[hx, hy, PLATE_T / 2],
                        material="m_plate", contype=0, conaffinity=0)
    # lights
    wb.add_light(name="key", pos=[ctr[0] - 0.4, ctr[1] - 0.8, 1.8], dir=[0.25, 0.45, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.75, 0.75, 0.72], specular=[0.1, 0.1, 0.1], castshadow=True)
    wb.add_light(name="fill", pos=[ctr[0] + 1, ctr[1] + 1, 1.5], dir=[-0.5, -0.5, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.25, 0.25, 0.3], specular=[0, 0, 0], castshadow=False)

    # robot
    arm_bodies = set()
    if robot == "xarm7":
        spec.body("link_base").pos = [float(pl.base_xy[0]), float(pl.base_xy[1]), PEDESTAL_H]
        spec.body("xarm_gripper_base_link").add_site(name="pinch", pos=[0, 0, PINCH_LOCAL_Z], size=[0.006] * 3,
                                                     rgba=[1, 0.2, 0.2, 0])
        wb.add_geom(name="pedestal", type=GEOM.mjGEOM_CYLINDER, pos=[pl.base_xy[0], pl.base_xy[1], PEDESTAL_H / 2],
                    size=[PEDESTAL_R, PEDESTAL_H / 2], rgba=[0.86, 0.87, 0.9, 1], contype=0, conaffinity=0)
    else:
        tool = wb.add_body(name="tool", pos=[0, 0, 0.5])
        tool.add_freejoint()
        tool.add_geom(name="cup", type=GEOM.mjGEOM_CYLINDER, pos=[0, 0, TAB_HALF[2] + 0.004], size=[0.015, 0.004],
                      rgba=[0.9, 0.35, 0.1, 1], contype=0, conaffinity=0)
        tool.add_geom(name="stem", type=GEOM.mjGEOM_CYLINDER, pos=[0, 0, TAB_HALF[2] + 0.35], size=[0.005, 0.34],
                      rgba=[0.15, 0.15, 0.17, 1], contype=0, conaffinity=0)
        tool.add_geom(name="head", type=GEOM.mjGEOM_BOX, pos=[0, 0, TAB_HALF[2] + 0.04], size=[0.016, 0.016, 0.012],
                      rgba=[0.15, 0.15, 0.17, 1], contype=0, conaffinity=0)

    # parts (mounts) + target ghosts for components that get corrected later
    corrected = {st.obj for st in pl.steps if st.is_corr}
    Q = lambda name: {"x": [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0],
                      "z45": [math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8)]}[name]

    def add_optic(body, sty, ghost):
        for i, (typ, size, pos, orient, rgba) in enumerate(sty["parts"]):
            kw = dict(type=GEOM.mjGEOM_BOX if typ == "box" else GEOM.mjGEOM_CYLINDER, size=size,
                      pos=[pos[0], pos[1], sty["h"] + pos[2]], rgba=(0.1, 1.0, 0.25, 0.28) if ghost else rgba)
            if orient:
                kw["quat"] = Q(orient)
            if ghost:
                kw.update(contype=0, conaffinity=0)
            else:
                kw.update(contype=2, conaffinity=1)
            body.add_geom(**kw)

    part_qadr, part_body = {}, {}
    for c in run.comps:
        sty = pl.styles[c.oid]
        if sty["cls"] == "support":
            continue
        p0 = pl.initial_disp[c.oid]
        b = wb.add_body(name=f"p_{c.oid}", pos=[p0[0], p0[1], 0.0])
        b.add_freejoint(name=f"fj_{c.oid}")
        add_optic(b, sty, False)
        b.add_geom(type=GEOM.mjGEOM_CYLINDER, pos=[0, 0, 0.003], size=[0.022 * s, 0.003], rgba=[0.25, 0.27, 0.3, 1],
                   contype=2, conaffinity=1)
        if sty["bottom"] > 0.009:
            hp = (sty["bottom"] - 0.006) / 2
            b.add_geom(type=GEOM.mjGEOM_CYLINDER, pos=[0, 0, 0.006 + hp], size=[0.005, hp], rgba=[0.55, 0.57, 0.62, 1],
                       contype=2, conaffinity=1)
        if c.movable:
            b.add_geom(type=GEOM.mjGEOM_BOX, pos=[0, 0, sty["tab_z"]], size=list(TAB_HALF), rgba=[0.12, 0.12, 0.14, 1],
                       contype=2, conaffinity=1)
        if c.oid in corrected:
            t = pl.target_disp(c.oid)
            g = wb.add_body(name=f"g_{c.oid}", pos=[t[0], t[1], 0.0], quat=mat2quat(Rz(t[3])).tolist())
            add_optic(g, sty, True)
            # flat target marker on the table: disc + a bar along the part's local y (shows the target yaw)
            g.add_geom(type=GEOM.mjGEOM_CYLINDER, pos=[0, 0, PLATE_T + 0.0012], size=[0.034, 0.0012], rgba=[0.1, 1.0, 0.25, 0.7],
                       contype=0, conaffinity=0)
            g.add_geom(type=GEOM.mjGEOM_BOX, pos=[0, 0, PLATE_T + 0.0034], size=[0.0015, 0.045, 0.0008], rgba=[0.0, 0.5, 0.1, 1],
                       contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    for c in run.comps:
        if comp_class(c) != "support":
            jid = model.joint(f"fj_{c.oid}").id
            part_qadr[c.oid] = int(model.jnt_qposadr[jid])
            part_body[c.oid] = model.body(f"p_{c.oid}").id
    sc = Scene(model=model, data=data, robot=robot, part_qadr=part_qadr, part_body=part_body,
               arm_qadr=np.zeros(0, int), grip_qadr=np.zeros(0, int), tool_qadr=-1, site_id=-1, cup_geom=-1,
               table_box=(tuple(lo), tuple(hi)), base_xy=pl.base_xy, arm_bodies=set(), ik=pl.ik)
    if robot == "xarm7":
        sc.arm_qadr = np.array([model.jnt_qposadr[model.joint(f"joint{i}").id] for i in range(1, 8)])
        names = ["left_driver_joint", "left_finger_joint", "left_inner_knuckle_joint",
                 "right_driver_joint", "right_finger_joint", "right_inner_knuckle_joint"]
        sc.grip_qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
        sc.site_id = model.site("pinch").id
        root = model.body("link_base").id
        sc.arm_bodies = {b for b in range(model.nbody) if _is_descendant(model, b, root)}
    else:
        sc.tool_qadr = int(model.jnt_qposadr[model.joint(model.body("tool").jntadr[0]).id])
        sc.cup_geom = model.geom("cup").id
    return sc


def _is_descendant(model, b, root):
    while b > 0:
        if b == root:
            return True
        b = model.body_parentid[b]
    return False


# --------------------------------------------------------------------------- camera + overlay
def cam_basis(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return f, right, up


def project(pt, lookat, dist, az, el, W, H, fovy=45.0):
    f, right, up = cam_basis(az, el)
    pos = np.asarray(lookat) - dist * f
    v = np.asarray(pt) - pos
    z = v @ f
    if z <= 1e-6:
        return None
    foc = (H / 2) / math.tan(math.radians(fovy) / 2)
    return (W / 2 + foc * (v @ right) / z, H / 2 - foc * (v @ up) / z)


def fit_camera(points, az, el, W, H, top=0.16, bottom=0.05, side=0.02):
    pts = np.array(points)
    look = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2, 0.12])
    box = (side * W, top * H, (1 - side) * W, (1 - bottom) * H)
    for _ in range(3):
        d = 0.6
        while d < 8:
            uv = [project(p, look, d, az, el, W, H) for p in pts]
            if all(u is not None for u in uv) and all(box[0] < u[0] < box[2] and box[1] < u[1] < box[3] for u in uv):
                break
            d += 0.02
        uv = np.array([project(p, look, d, az, el, W, H) for p in pts])
        cu = (uv[:, 0].min() + uv[:, 0].max()) / 2 - W / 2
        cv = (uv[:, 1].min() + uv[:, 1].max()) / 2 - (box[1] + box[3]) / 2
        f, right, up = cam_basis(az, el)
        foc = (H / 2) / math.tan(math.radians(45) / 2)
        look = look + right * (cu * d / foc) - up * (cv * d / foc)
    return look, d


class Overlay:
    def __init__(self, W, H, run, pl, info, robot):
        self.W, self.H, self.run, self.pl, self.info, self.robot = W, H, run, pl, info, robot
        sc = H / 720.0
        self.f_small = load_font(int(14 * sc))
        self.f_med = load_font(int(17 * sc))
        self.f_big = load_font(int(28 * sc), True)
        self.f_title = load_font(int(32 * sc), True)
        self.f_lab = load_font(int(15 * sc), True)
        self.f_foot = load_font(int(15 * sc))
        self.cache = {}
        self.rid = run.run_id[:8]

    # -- text helpers
    def text(self, d, xy, s, font, fill=(255, 255, 255, 255), outline=None):
        if outline:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        d.text((xy[0] + dx, xy[1] + dy), s, font=font, fill=outline)
        d.text(xy, s, font=font, fill=fill)

    def wrap_text(self, d, s, font, maxw):
        words, lines, cur = s.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if d.textlength(t, font=font) <= maxw:
                cur = t
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def static_layer(self, ov):
        key = id(ov)
        if key in self.cache:
            return self.cache[key]
        W, H = self.W, self.H
        L = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(L)
        pl, run = self.pl, self.run
        sc = H / 720.0
        foot_h = int(30 * sc)
        d.rectangle([0, H - foot_h, W, H], fill=(15, 18, 24, 235))
        self.text(d, (12, H - foot_h + int(6 * sc)), FOOTER, self.f_foot, (255, 235, 160, 255))
        kind = ov.get("kind")
        if kind in ("step",):
            bar = int(108 * sc)
            d.rectangle([0, 0, W, bar], fill=(15, 18, 24, 215))
            req = run.request if len(run.request) < 90 else run.request[:87] + "..."
            robot = ("xArm7 (stand-in), base at x=%.2f y=%.2f m" % (pl.base_xy[0], pl.base_xy[1])
                     if self.robot == "xarm7" else "floating suction tool (fallback, no arm)")
            self.text(d, (12, int(6 * sc)), f"Run {self.rid}   Request: \"{req}\"", self.f_med, (200, 215, 235, 255))
            right = f"{robot} | scene scale x{pl.s:g}"
            d.text((W - 12 - d.textlength(right, font=self.f_small), int(8 * sc)), right, font=self.f_small,
                   fill=(170, 185, 205, 255))
            self.text(d, (12, int(28 * sc)), ov["title"], self.f_big, (255, 255, 255, 255))
            if ov.get("err"):
                self.text(d, (12, int(66 * sc)), ov["err"], self.f_med, (190, 235, 255, 255))
            if ov.get("exag"):
                self.text(d, (12, int(86 * sc)), ov["exag"], self.f_small, (255, 214, 102, 255))
            # legend
            rows = [c for c in run.comps if comp_class(c) != "support"]
            sup = [c for c in run.comps if comp_class(c) == "support"]
            rows += sup
            lh = int(18 * sc)
            ncol = 2 if getattr(self, "compact_legend", False) else 1
            per = -(-len(rows) // ncol)
            names = [f"{c.oid}  {c.label}" + ("" if c.movable else "  (fixed)") for c in rows]
            colw = [max(d.textlength(n, font=self.f_small) for n in names[j * per:(j + 1) * per]) + 34 for j in range(ncol)]
            lx, ly = 10, H - foot_h - 10 - lh * per - 8
            alpha = 120 if ncol > 1 else 170
            d.rectangle([lx - 4, ly - 4, lx + sum(colw), ly + lh * per + 4], fill=(15, 18, 24, alpha))
            self.legend_rect = (lx - 4, ly - 4, lx + sum(colw), ly + lh * per + 4)
            for i, c in enumerate(rows):
                cx = lx + sum(colw[:i // per])
                cy = ly + (i % per) * lh
                col = tuple(int(255 * v) for v in PALETTE[comp_class(c)][:3]) + (255,)
                d.rectangle([cx + 2, cy + 3, cx + 14, cy + 15], fill=col, outline=(255, 255, 255, 200))
                self.text(d, (cx + 22, cy), names[i], self.f_small, (235, 240, 248, 255))
            if ov.get("unreached"):
                msg = "UNREACHED"
                tw = d.textlength(msg, font=self.f_title)
                d.rectangle([W / 2 - tw / 2 - 20, H / 2 - 40, W / 2 + tw / 2 + 20, H / 2 + 40], fill=(120, 0, 0, 220))
                self.text(d, (W / 2 - tw / 2, H / 2 - 22), msg, self.f_title, (255, 255, 255, 255))
        elif kind == "intro":
            self.card(d, [
                ("Replay of a finished optics-workbench run", "t"),
                (f"Run {run.run_id}", "s"),
                (f"Request: \"{run.request}\"", "m"),
                ("", "m"),
                (f"Attempt 1 places the movable parts from a feed row to their recorded poses.", "m"),
                (self.seed_line(), "m"),
                ("", "m"),
                ("The arm is a stand-in. The workbench simulation (algorithm_v1) is a pose proxy with no robot;", "s"),
                ("this video moves a model arm kinematically between the poses that run recorded.", "s"),
            ])
        elif kind == "final":
            self.card(d, self.final_lines(), anchor="top")
        L = L
        self.cache[key] = L
        return L

    def seed_line(self):
        sd = self.run.seeded
        n = len(self.run.attempts)
        if sd["id"] and sd["pos_m"] is not None:
            return (f"Seeded misplacement on {sd['id']}: {fmt_mm(sd['pos_m'])}, {fmt_deg(sd['ang_rad'])}. "
                    f"Attempts 2-{n} correct it." if n > 1 else f"Seeded misplacement on {sd['id']}.")
        return f"{n} attempt(s) recorded in the run."

    def final_lines(self):
        run = self.run
        last = run.attempts[-1]
        lines = [("Run finished", "t"), (f"Final status (run.json): {run.status}", "m")]
        if run.tests_total:
            lines.append((f"Analytic design tests passed: {run.tests_passed} of {run.tests_total}", "m"))
        for oid in last.moved:
            lim = last.limits
            lines.append((f"Residual error of {oid} after attempt {last.n}: {fmt_mm(last.pos_err[oid])}, "
                          f"{fmt_deg(last.ang_err[oid])}"
                          + (f"   (limits {fmt_mm(lim['position_m'])}, {fmt_deg(lim['angle_rad'])})" if lim else ""), "m"))
        if run.deferred_n:
            lines.append((f"Workflow complete is not 'all requirements verified': {run.deferred_n} requirement(s) "
                          "deferred (need measurement / wave optics).", "s"))
        mot = run.result.get("motions_sent", 0)
        lines += [("", "m"), (f"No real hardware moved (motions_sent = {mot})", "b"),
                  ("physical_verified = false", "b"),
                  ("Kinematic visualization of recorded poses. Not physics, not collision-checked.", "s")]
        return lines

    def card(self, d, lines, anchor="center"):
        W, H = self.W, self.H
        sc = H / 720.0
        fonts = {"t": self.f_title, "b": self.f_big, "m": self.f_med, "s": self.f_small}
        hts = {"t": 44, "b": 38, "m": 30, "s": 24}
        total = sum(hts[k] * sc for _, k in lines) + 40
        pad = 28
        wmax = 0
        for s, k in lines:
            wmax = max(wmax, d.textlength(s, font=fonts[k]))
        x0, x1 = (W - wmax) / 2 - pad, (W + wmax) / 2 + pad
        y0 = (H - total) / 2 if anchor == "center" else 14 * sc
        d.rectangle([x0, y0, x1, y0 + total], fill=(12, 16, 24, 232), outline=(120, 150, 200, 255), width=2)
        y = y0 + 20 * sc
        for s, k in lines:
            col = {"t": (255, 255, 255, 255), "b": (140, 255, 170, 255)}.get(k, (225, 232, 245, 255))
            tw = d.textlength(s, font=fonts[k])
            d.text(((W - tw) / 2, y), s, font=fonts[k], fill=col)
            y += hts[k] * sc

    def compose(self, frame, ov, labels, pip_img, pip_title):
        W, H = self.W, self.H
        img = Image.fromarray(frame).convert("RGBA")
        kind = ov.get("kind")
        if kind == "final":
            dim = Image.new("RGBA", (W, H), (0, 0, 0, 60))
            img = Image.alpha_composite(img, dim)
        img = Image.alpha_composite(img, self.static_layer(ov))
        d = ImageDraw.Draw(img)
        if kind == "step":
            sc_ = H / 720.0
            lr = getattr(self, "legend_rect", None)
            for (u, v, txt) in labels:
                tw_ = d.textlength(txt, font=self.f_lab)
                if v < 108 * sc_ + 2 or v > H - 30 * sc_ - 18 * sc_ or u < 4 or u + tw_ / 2 > W - 4:
                    continue                      # would sit on the header / footer / off screen
                if lr and lr[0] - 20 < u < lr[2] + 20 and lr[1] - 14 < v < lr[3]:
                    continue                      # would sit on the legend
                tw = d.textlength(txt, font=self.f_lab)
                self.text(d, (u - tw / 2, v - 8), txt, self.f_lab, (255, 255, 255, 255), (10, 12, 18, 255))
            if pip_img is not None:
                pw, ph = pip_img.size
                px, py = W - pw - 14, H - int(30 * H / 720) - ph - 12
                d.rectangle([px - 3, py - 3, px + pw + 3, py + ph + 3], fill=(255, 214, 102, 255))
                img.paste(pip_img.convert("RGBA"), (px, py))
                d.rectangle([px, py, px + pw, py + int(22 * H / 720)], fill=(15, 18, 24, 200))
                self.text(d, (px + 8, py + 3), pip_title, self.f_small, (255, 255, 255, 255))
        return np.asarray(img.convert("RGB"))


def true_from_display(run, oid, disp, k, s, pos_thr=0.02, ang_thr=0.2):
    """Inverse of Planner.disp (for check_render): recover the evidence pose from a drawn pose."""
    p = np.array(disp, float)
    p[:3] /= s
    if k == 1:
        return p
    t = run.comp(oid).target
    d = p[:3] - t[:3]
    dy = wrap(p[3] - t[3])
    if np.linalg.norm(d[:2]) / k < pos_thr and abs(d[2]) / k < pos_thr and abs(dy) / k < ang_thr:
        out = t.copy()
        out[:3] = t[:3] + d / k
        out[3] = wrap(t[3] + dy / k)
        return out
    return p


# --------------------------------------------------------------------------- render
def render(run, pl, tl, info, args, log=print):
    W, H, fps = args.width, args.height, args.fps
    robot = args.robot
    sc = build_scene(run, pl, W, H, robot)
    m, d = sc.model, sc.data
    ov_layer = Overlay(W, H, run, pl, info, robot)
    ov_layer.compact_legend = args.camera == "close"
    # ---- camera framing
    pts = []
    lo, hi = sc.table_box
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            pts.append([x, y, 0.0])
    for c in run.comps:
        for st in pl.steps:
            pass
    allp = [p[:2] for p in pl.initial_disp.values()] + [p[:2] for p in pl.final_display.values()]
    for p in allp:
        pts.append([p[0], p[1], 0.0])
        pts.append([p[0], p[1], pl.max_top + 0.15])
        if robot == "gantry":
            pts.append([p[0], p[1], pl.z_free + pl.max_top + 0.15])   # tool head at carry height
    if robot == "xarm7":
        bx, by = pl.base_xy
        pts += [[bx, by, 0.0]]
        ik = pl.ik
        # keep the whole arm in frame: link origins (+ margin for the link geometry) at the park posture
        # and at sampled configurations of every planned move
        samples = [pl.q_park]
        for seg in tl:
            if seg.kind == "move" and seg.path is not None:
                samples += [seg.path[0], seg.path[len(seg.path) // 2], seg.path[-1]]
        for q in samples[::2] if len(samples) > 120 else samples:
            ik.fk(q)
            for b in range(2, ik.m.nbody):
                pts.append(list(ik.d.xpos[b] + np.array([0, 0, 0.05])))
        # arm must stay inside the frame: hover points of every pick/place
    close_cam = args.camera == "close"
    el = args.cam_el if args.cam_el is not None else (-24.0 if close_cam else -38.0)
    if args.cam_az is not None:
        az = args.cam_az
        look, dist = fit_camera(pts, az, el, W, H)
    elif robot == "xarm7":
        cc = np.mean([p[:2] for p in allp], axis=0)
        v = cc - np.array(pl.base_xy)
        v /= (np.linalg.norm(v) + 1e-9)
        a0 = math.degrees(math.atan2(v[1], v[0])) + 180.0   # camera on the far side, looking toward the arm base
        best = None
        for off in (-65, -50, -35, 35, 50, 65):      # pick the angle that shows the scene largest
            lk, dd = fit_camera(pts, a0 + off, el, W, H)
            if best is None or dd < best[0] - 1e-6:
                best = (dd, a0 + off, lk)
        dist, az, look = best
    else:
        az = 100.0
        look, dist = fit_camera(pts, az, el, W, H)
    wide_view = (np.array(look, float), float(dist), float(az), float(el))
    cam_focus = None
    if close_cam:
        # tight framing: the breadboard (final poses) fills the frame; the camera sits on the same side as the
        # wide shot (so the arm is not between the lens and the parts) but lower, and eases toward the part
        # currently being handled (pick at the feed row, then the placement on the board).
        if args.cam_az is None and robot == "xarm7":
            # a more lateral view than the wide shot (same side), so the arm reaches in from the side/back of
            # the board and the gripper, not the upper arm, is what stands between lens and parts
            sgn = 1.0 if wrap(math.radians(az - a0)) > 0 else -1.0
            az = a0 + sgn * 84.0
        fin = np.array([p[:2] for p in pl.final_display.values()])
        ini = np.array([p[:2] for p in pl.initial_disp.values()])
        board_c = (fin.min(0) + fin.max(0)) / 2
        half = max(np.ptp(fin[:, 0]), np.ptp(fin[:, 1]), 0.18) / 2 + 0.10
        # distance from the vertical field of view: board half-extent should span ~ 60 % of the frame height
        fov_t = math.tan(math.radians(45) / 2)
        dist = float(np.clip(1.8 * half, 0.6, 3.0))
        if args.cam_dist is not None:
            dist = float(args.cam_dist)
        look = np.array([board_c[0], board_c[1], 0.07])
        cam_focus = dict(board=look.copy(), dist0=dist, cur=look + np.array([0.0, 0.0, 0.10]), dist=dist * 1.45)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    pw = int(W * 0.30)
    ph = int(pw * 0.75)
    rend = mujoco.Renderer(m, H, W)
    rend_pip = mujoco.Renderer(m, ph, pw)
    cam_pip = mujoco.MjvCamera()
    cam_pip.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_pip.azimuth = 35.0       # oblique: the part face and its offset from the target stay visible
    cam_pip.elevation = -27

    # ---- state
    part_T = {}
    for oid, p in pl.initial_disp.items():
        if oid in sc.part_qadr:
            part_T[oid] = T_of(p[0], p[1], 0.0, p[3])
    held = {"obj": None, "off": None}
    state = {"arm": pl.q_park.copy() if pl.ik is not None else None, "theta": 0.0}
    if robot == "gantry":
        cx = np.mean([p[0] for p in pl.initial_disp.values()])
        cy = np.mean([p[1] for p in pl.initial_disp.values()])
        state["arm"] = np.array([cx, cy, pl.z_free, 0.0])
    cup_base = m.geom_rgba[sc.cup_geom].copy() if sc.cup_geom >= 0 else None
    grasp_gaps = []
    contact = dict(arm_vs_part_frames=0, arm_vs_table_frames=0, arm_self_frames=0, examples=[], self_examples=[])
    grip_bodies = set()
    if robot == "xarm7":
        gb = m.body("xarm_gripper_base_link").id
        grip_bodies = {b for b in range(m.nbody) if _is_descendant(m, b, gb)} | {gb, m.body("link7").id}

    expect = {"obj": None}

    def put_part(oid, T):
        a = sc.part_qadr[oid]
        d.qpos[a:a + 3] = T[:3, 3]
        d.qpos[a + 3:a + 7] = mat2quat(T[:3, :3])

    def apply(arm_state, theta):
        if robot == "xarm7":
            d.qpos[sc.arm_qadr] = arm_state
            d.qpos[sc.grip_qadr] = theta
        else:
            a = sc.tool_qadr
            d.qpos[a:a + 3] = arm_state[:3]
            d.qpos[a + 3:a + 7] = mat2quat(Rz(arm_state[3]))
        for oid, T in part_T.items():
            if oid != held["obj"]:
                put_part(oid, T)
        mujoco.mj_kinematics(m, d)
        if held["obj"] is not None:
            if robot == "xarm7":
                T_site = np.eye(4)
                T_site[:3, :3] = d.site_xmat[sc.site_id].reshape(3, 3)
                T_site[:3, 3] = d.site_xpos[sc.site_id]
            else:
                T_site = np.eye(4)
                T_site[:3, :3] = grasp_R(arm_state[3])
                T_site[:3, 3] = arm_state[:3]
            T = T_site @ held["off"]
            part_T[held["obj"]] = T
            put_part(held["obj"], T)
        mujoco.mj_forward(m, d)

    def site_T(arm_state):
        if robot == "xarm7":
            d.qpos[sc.arm_qadr] = arm_state
            mujoco.mj_kinematics(m, d)
            T = np.eye(4)
            T[:3, :3] = d.site_xmat[sc.site_id].reshape(3, 3)
            T[:3, 3] = d.site_xpos[sc.site_id]
            return T
        T = np.eye(4)
        T[:3, :3] = grasp_R(arm_state[3])
        T[:3, 3] = arm_state[:3]
        return T

    def qa_contacts(fi):
        names = None
        flagged = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            a1, a2 = b1 in sc.arm_bodies, b2 in sc.arm_bodies
            if a1 and a2:
                flagged.add("arm_self")
                if len(contact["self_examples"]) < 6:
                    contact["self_examples"].append(dict(frame=fi, bodies=[m.body(b1).name, m.body(b2).name]))
            elif a1 or a2:
                ab, ob = (b1, b2) if a1 else (b2, b1)
                if ob == 0:
                    g = c.geom2 if a1 else c.geom1
                    if m.geom(g).name == "table":
                        flagged.add("arm_vs_table")
                else:
                    oid = next((k for k, v in sc.part_body.items() if v == ob), None)
                    if oid is not None and oid == expect["obj"] and ab in grip_bodies:
                        continue              # the intended grasp contact (the part being picked / carried)
                    flagged.add("arm_vs_part")
                    if len(contact["examples"]) < 12:
                        contact["examples"].append(dict(frame=fi, part=oid, arm_body=m.body(ab).name))
        for f in flagged:
            contact[f + "_frames"] += 1

    # ---- render loop
    writer = imageio.get_writer(str(args.out), fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                pixelformat="yuv420p", ffmpeg_log_level="error")
    gif_frames = []
    markers = {}
    fi = 0
    t0 = time.time()
    first_frame = None
    for seg in tl:
        nfr = max(1, int(round(seg.dur * fps)))
        if seg.attach:
            T_site = site_T(state["arm"])
            held["obj"] = seg.attach
            held["off"] = T_inv(T_site) @ part_T[seg.attach]
            tab_w = part_T[seg.attach] @ np.array([0, 0, pl.styles[seg.attach]["tab_z"], 1.0])
            grasp_gaps.append(dict(obj=seg.attach, label=seg.label, gap_m=float(np.linalg.norm(T_site[:3, 3] - tab_w[:3]))))
            if cup_base is not None:
                m.geom_rgba[sc.cup_geom] = [0.15, 0.85, 0.3, 1]
        if seg.release:
            oid, P1 = seg.release
            part_T[oid] = T_of(P1[0], P1[1], 0.0, P1[3])      # exact evidence pose (display scaling only)
            held["obj"], held["off"] = None, None
            if cup_base is not None:
                m.geom_rgba[sc.cup_geom] = cup_base
        chain = None
        if seg.kind == "move" and seg.path is not None:
            chain = np.vstack([state["arm"][None], seg.path])
            if robot == "xarm7":
                step_len = np.linalg.norm(np.diff(chain, axis=0), axis=1)
            else:
                step_len = np.linalg.norm(np.diff(chain[:, :3], axis=0), axis=1) + 0.1 * np.abs(np.diff(chain[:, 3]))
            cum = np.r_[0.0, np.cumsum(step_len)]
        for f in range(nfr):
            u = (f + 1) / nfr
            e = smooth(u)
            if chain is not None and cum[-1] > 1e-9:
                sdist = e * cum[-1]
                j = min(int(np.searchsorted(cum, sdist, side="right")) - 1, len(chain) - 2)
                j = max(j, 0)
                w = (sdist - cum[j]) / max(cum[j + 1] - cum[j], 1e-12)
                arm_now = chain[j] + (chain[j + 1] - chain[j]) * w
            elif chain is not None:
                arm_now = chain[-1]
            else:
                arm_now = state["arm"]
            th = seg.th0 + (seg.th1 - seg.th0) * e if seg.kind == "grip" else state["theta"]
            apply(arm_now, th)
            qa_contacts(fi)
            ov = seg.ov or {}
            if cam_focus is not None:
                tgt = cam_focus["board"].copy()
                if ov.get("kind") == "step" and ov.get("obj") in sc.part_body:
                    pp_ = d.xpos[sc.part_body[ov["obj"]]]
                    # aim between the board centre and the handled part (part dominates), part height for z
                    zf = 0.07 + 0.7 * max(pp_[2] - 0.07, 0.0)        # follow a lifted part, but gently
                    tgt = 0.30 * cam_focus["board"] + 0.70 * np.array([pp_[0], pp_[1], zf])
                # critically-damped smoothing (time constant ~0.7 s): no cuts, no fast moves
                card_ = ov.get("kind") in ("intro", "final")
                if card_:
                    tgt = tgt + np.array([0.0, 0.0, 0.10])      # board sits below the text card
                dist_t = cam_focus["dist0"] * (1.45 if card_ else 1.0)
                a_ = 1.0 - math.exp(-1.0 / (0.7 * fps))
                cam_focus["dist"] = cam_focus.get("dist", cam_focus["dist0"])
                cam_focus["dist"] += (dist_t - cam_focus["dist"]) * a_
                dist = cam_focus["dist"]
                cam.distance = dist
                cam_focus["cur"] = cam_focus["cur"] + (tgt - cam_focus["cur"]) * a_
                look = cam_focus["cur"]
                cam.lookat[:] = look + np.array([0.0, 0.0, 0.04])   # frame sits a little below the header bar
            rend.update_scene(d, camera=cam)
            frame = rend.render().copy()
            expect["obj"] = ov.get("obj")
            labels = []
            if ov.get("kind") == "step":
                for oid, bid in sc.part_body.items():
                    sty = pl.styles[oid]
                    pos = d.xpos[bid]
                    uv = project([pos[0], pos[1], pos[2] + sty["tab_z"] + 0.04], cam.lookat, dist, az, el, W, H)
                    if uv and oid == ov.get("obj"):
                        # the part being handled: label beside it (not above), so the gripper / part stay clear
                        uv2 = project([pos[0], pos[1], pos[2] + sty["h"]], cam.lookat, dist, az, el, W, H)
                        labels.append([(uv2 or uv)[0] + 46, (uv2 or uv)[1] - 4, oid])
                    elif uv:
                        labels.append([uv[0], uv[1], oid])
                labels.sort(key=lambda l: l[1])
                for i, l in enumerate(labels):          # nudge overlapping labels apart
                    for o in labels[:i]:
                        if abs(l[0] - o[0]) < 50 and abs(l[1] - o[1]) < 17:
                            l[1] = o[1] + 17
            pip_img = None
            pip_title = ""
            if ov.get("pip"):
                pp = ov["pip"]
                cam_pip.lookat[:] = pp["look"]
                cam_pip.distance = pp["dist"]
                rend_pip.update_scene(d, camera=cam_pip)
                pip_img = Image.fromarray(rend_pip.render().copy())
                pip_title = f"Close-up: {pp['obj']}  (offset x{pl.k:g})" if pl.k != 1 else f"Close-up: {pp['obj']}"
            out = ov_layer.compose(frame, ov, labels, pip_img, pip_title)
            writer.append_data(out)
            if args.gif and fi % max(1, int(round(fps / 8))) == 0:
                gif_frames.append(Image.fromarray(out).resize((480, int(480 * H / W)), Image.LANCZOS))
            if first_frame is None:
                first_frame = out
            fi += 1
        if chain is not None:
            state["arm"] = seg.path[-1].copy()
        if seg.kind == "grip":
            state["theta"] = seg.th1
        if seg.marker:
            markers[seg.marker] = fi - 1
        if seg.kind == "hold" and seg.label == "banner" and seg.ov and seg.ov.get("pip"):
            markers.setdefault(f"closeup:{seg.ov['attempt']}:{seg.ov.get('obj')}", fi - 1)
    writer.close()
    if args.gif:
        gif_path = Path(args.out).with_suffix(".gif")
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=int(1000 / 8), loop=0,
                           optimize=True)
    # drawn part poses straight from MuJoCo state
    final = {}
    for oid, a in sc.part_qadr.items():
        q = d.qpos[a:a + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[3:7])
        R = R.reshape(3, 3)
        final[oid] = np.array([q[0], q[1], q[2], math.atan2(R[1, 0], R[0, 0])])
    log(f"  rendered {fi} frames in {time.time() - t0:.0f}s")
    return dict(frames=fi, fps=fps, duration_s=fi / fps, markers=markers, final_drawn=final, contact=contact, grasp_gaps=grasp_gaps,
                camera=dict(az=az, el=el, dist=float(dist), lookat=[float(x) for x in look]), scene=sc)


# --------------------------------------------------------------------------- pipeline / CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Kinematic MuJoCo video of a finished optics-workbench run "
                                 "(illustration of recorded poses; not physics, not a controller).")
    ap.add_argument("--run-dir", required=True, help="path to runs/<run_id> (read-only)")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--gif", action="store_true", help="also write a small GIF next to the mp4")
    ap.add_argument("--robot", choices=["xarm7", "gantry"], default="xarm7")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (>1 = faster, shorter video)")
    ap.add_argument("--exaggerate", type=float, default=10.0,
                    help="display-only factor on small offsets from the layout target (1 = true scale)")
    ap.add_argument("--allow-unreached", action="store_true",
                    help="skip moves whose IK does not converge and stamp UNREACHED (default: fail)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--base", type=float, nargs=2, metavar=("X", "Y"), help="force the arm base position (m)")
    ap.add_argument("--force-scale", type=float, default=None, help="force a uniform scene scale (<1 shrinks layout)")
    ap.add_argument("--cam-az", type=float, default=None)
    ap.add_argument("--cam-el", type=float, default=None,
                    help="camera elevation in degrees (default: -38 for --camera wide, -24 for --camera close)")
    ap.add_argument("--cam-dist", type=float, default=None,
                    help="camera distance in m (only with --camera close; default: derived from the board size)")
    ap.add_argument("--camera", choices=["close", "wide"], default="close",
                    help="close: tight three-quarter view of the breadboard that eases toward the component being "
                         "handled (arm base/shoulder may be out of frame); wide: the whole arm and layout in frame")
    ap.add_argument("--allow-unfinished", action="store_true")
    a = ap.parse_args(argv)
    a.out = Path(a.out)
    return a


def produce(args, log=print):
    run = load_run(args.run_dir)
    if run.stage != "done" and not args.allow_unfinished:
        raise SystemExit(f"run {run.run_id} is not finished (stage={run.stage}, status={run.status}); "
                         "pass --allow-unfinished to visualize it anyway")
    steps, notes = build_steps(run)
    run.notes = notes
    for n in notes:
        log("  note: " + n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log(f"run {run.run_id}: {len(run.comps)} components, {len(run.attempts)} attempts, {len(steps)} animated steps")
    pl, tl, info = plan_with_reach(run, steps, args, log)
    if info["robot"] == "xarm7":
        log(f"  base at ({info['base_xy'][0]:.3f}, {info['base_xy'][1]:.3f}) m, scene scale {info['scale']:g}, "
            f"{len(pl.ik_log)} IK waypoints")
    res = render(run, pl, tl, info, args, log)
    ik_ok = all(e["ok"] for e in pl.ik_log)
    stats = dict(
        n_waypoints=len(pl.ik_log), all_converged=ik_ok,
        max_pos_err_m=max([e["pos_err_m"] for e in pl.ik_log], default=0.0),
        max_rot_err_rad=max([e["rot_err_rad"] for e in pl.ik_log], default=0.0),
        mean_iters=float(np.mean([e["iters"] for e in pl.ik_log])) if pl.ik_log else 0.0,
        max_joint_step_rad=max([j[1] for j in pl.jump_log], default=0.0),
        largest_joint_steps=sorted(pl.jump_log, key=lambda j: -j[1])[:3])
    report = dict(
        run_id=run.run_id, robot=args.robot, out=str(args.out), frames=res["frames"], fps=res["fps"],
        duration_s=res["duration_s"], reach=info, ik=stats, unreached=pl.unreached, exaggeration=pl.k,
        grasp_gap_max_m=max([g["gap_m"] for g in res["grasp_gaps"]], default=0.0),
        markers=res["markers"], camera=res["camera"], qa_contacts_dev_diagnostic=res["contact"],
        notes=run.notes, honesty=FOOTER)
    rp = args.out.with_suffix(".report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    log(f"  wrote {args.out} ({res['duration_s']:.1f} s, {args.out.stat().st_size / 1e6:.1f} MB) and {rp.name}")
    return dict(run=run, steps=steps, planner=pl, timeline=tl, info=info, result=res, report=report)


def main(argv=None):
    args = parse_args(argv)
    produce(args)


if __name__ == "__main__":
    main()
