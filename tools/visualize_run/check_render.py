#!/usr/bin/env python3
"""check_render.py - programmatic checks for render_run.py.

  check_render.py --run-dir <runs/<id>> [--robot xarm7|gantry] [--out demo-output/check.mp4]

Renders the run, then checks:
  1. the run directory is byte-for-byte unchanged (sha256 + mtime + size listing before/after);
  2. the final drawn part poses, with the display exaggeration undone, equal the run's final `after`
     poses (x, y, yaw) within 1e-6;
  3. every IK waypoint converged (max position / orientation residual reported); the gripper pinch point
     is at the part's grip tab when it attaches;
  4. frame count (= planned timeline), duration, fps, MP4 readable, no blank/black frames;
  5. (dev diagnostic) no arm-vs-part / arm-vs-table / arm self-intersection frames.
Exit code 0 only if all checks pass.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_run as R  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


def snapshot(run_dir: Path):
    snap = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            st = p.stat()
            snap[str(p.relative_to(run_dir))] = (hashlib.sha256(p.read_bytes()).hexdigest(), st.st_mtime_ns, st.st_size)
    return snap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--out" not in argv:
        rd = Path(argv[argv.index("--run-dir") + 1]).name if "--run-dir" in argv else "run"
        argv += ["--out", str(Path(__file__).resolve().parents[2] / "demo-output" / "check" / f"{rd[:8]}.mp4")]
    args = R.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    before = snapshot(run_dir)
    print(f"run dir snapshot before: {len(before)} files, aggregate sha256 "
          f"{hashlib.sha256(repr(sorted(before.items())).encode()).hexdigest()[:16]}")
    out = R.produce(args, log=lambda s: None)
    after = snapshot(run_dir)
    print(f"run dir snapshot after:  {len(after)} files, aggregate sha256 "
          f"{hashlib.sha256(repr(sorted(after.items())).encode()).hexdigest()[:16]}")
    print("  (path, sha256[:12], mtime_ns, size) of a few files:")
    for k in list(before)[:4]:
        print("   ", k, before[k][0][:12], before[k][1], before[k][2])
    check("run dir unchanged (files, sha256, mtime, size)", before == after,
          f"{len(before)} files identical before/after" if before == after else "DIFFERENCES FOUND")

    run, pl, tl, res, info = out["run"], out["planner"], out["timeline"], out["result"], out["info"]
    # 2. final poses
    last = run.attempts[-1].after
    worst = 0.0
    for oid, drawn in res["final_drawn"].items():
        true = R.true_from_display(run, oid, drawn, pl.k, pl.s)
        ev = last[oid]
        err = max(abs(true[0] - ev[0]), abs(true[1] - ev[1]), abs(R.wrap(true[3] - ev[3])))
        worst = max(worst, err)
        print(f"    {oid:8s} drawn x={drawn[0]:.6f} y={drawn[1]:.6f} yaw={drawn[3]:+.6f} -> evidence-scale "
              f"({true[0]:.6f},{true[1]:.6f},{true[3]:+.6f}) vs after ({ev[0]:.6f},{ev[1]:.6f},{ev[3]:+.6f})  err {err:.1e}")
    check("final part poses equal run's final `after` poses (x, y, yaw) < 1e-6", worst < 1e-6 and not pl.unreached,
          f"max error {worst:.2e}" + (f"; {len(pl.unreached)} UNREACHED step(s)" if pl.unreached else ""))
    # 3. IK
    if args.robot == "xarm7":
        st = out["report"]["ik"]
        check("every IK waypoint converged", st["all_converged"] and st["n_waypoints"] > 0,
              f"{st['n_waypoints']} waypoints; max position residual {st['max_pos_err_m'] * 1000:.4f} mm, "
              f"max orientation residual {math.degrees(st['max_rot_err_rad']):.4f} deg, "
              f"max joint step between neighbouring waypoints {st['max_joint_step_rad']:.2f} rad")
        gaps = res["grasp_gaps"]
        gmax = max(g["gap_m"] for g in gaps)
        check("gripper pinch point at the grip tab when attaching", gmax < 2e-4,
              f"{len(gaps)} grasps, worst pinch-to-tab distance {gmax * 1000:.4f} mm")
        # independent FK of the last waypoint of every 'descend to pick' segment
        ik, worst_fk = pl.ik, 0.0
        for seg in tl:
            if seg.kind == "move" and seg.label.endswith("descend to pick"):
                pos, _ = ik.fk(seg.path[-1])
                obj = seg.ov["obj"]
                tabz = pl.styles[obj]["tab_z"]
                # part pose at this moment = the previous display pose; find it from the matching grasp gap
                worst_fk = max(worst_fk, abs(pos[2] - tabz))
        check("pinch height at pick equals the tab height (independent FK)", worst_fk < 2e-4, f"max |dz| {worst_fk * 1000:.4f} mm")
    else:
        check("gantry: no IK (floating suction tool)", True, "tool poses are the waypoints themselves")
    # 4. video
    mp4 = Path(args.out)
    expected = sum(max(1, int(round(s.dur * args.fps))) for s in tl)
    r = imageio.get_reader(str(mp4))
    meta = r.get_meta_data()
    n = 0
    dark = flat = 0
    means = []
    for i, f in enumerate(r):
        n += 1
        if i % 20 == 0:
            means.append(float(f.mean()))
            if f.mean() < 25:
                dark += 1
            if f.std() < 8:
                flat += 1
    dur = n / args.fps
    check("frame count equals planned timeline", n == expected == res["frames"], f"{n} decoded / {expected} planned / {res['frames']} written")
    check("duration 45-90 s", 45 <= dur <= 90 or args.speed != 1.0, f"{dur:.1f} s at {args.fps} fps, {args.width}x{args.height}, "
          f"{mp4.stat().st_size / 1e6:.1f} MB (speed x{args.speed:g})")
    check("no black / blank frames (sampled every 20th)", dark == 0 and flat == 0,
          f"mean brightness min {min(means):.0f} max {max(means):.0f}, dark {dark}, flat {flat}")
    # 5. dev diagnostic
    c = res["contact"]
    check("dev diagnostic: no arm-vs-part, arm-vs-table or arm self-intersection frames (not a guarantee)",
          c["arm_vs_part_frames"] == 0 and c["arm_vs_table_frames"] == 0 and c["arm_self_frames"] == 0,
          f"part {c['arm_vs_part_frames']}, table {c['arm_vs_table_frames']}, self {c['arm_self_frames']} frames")
    print(f"reach: {info}")
    ok = all(x[1] for x in results)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
