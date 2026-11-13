# visualize_run: kinematic MuJoCo video of a finished optics-workbench run

Turns a FINISHED run directory (`.../data-mcp/runs/<run_id>`) into a watchable video of an xArm7
placing the optical components, including the seeded misplacement and its corrections.

## What this is, and is not (read this first)

**Kinematic visualization of the accepted plan. Not physics, not collision-checked, not a robot
controller. `physical_verified = false`.** The workbench's `algorithm_v1` simulation is a pose
proxy with no robot in it. This tool only illustrates the poses that run recorded: a model arm
(or a floating suction tool) is moved kinematically between them. No real hardware, camera or
optical measurement is involved (`motions_sent = 0`). Every frame carries that statement in its
footer. The seeded error (3 mm, 2 degrees) is far too small to see at table scale, so small offsets
from the layout target are drawn x10 (`--exaggerate`, 1 = true scale); the overlay always shows the
real numbers and states the factor. The green ghost / disc on the table is the layout target.

## Setup

```
cd tools/visualize_run
python3.12 -m venv .venv            # any Python 3.10+ that has MuJoCo wheels
.venv/bin/python -m pip install -r requirements.txt
```
(`.venv/` is git-ignored. Headless rendering uses MuJoCo's offscreen renderer; on a Linux server set
`MUJOCO_GL=egl` or `osmesa`.)

## Usage

```
tools/visualize_run/.venv/bin/python tools/visualize_run/render_run.py \
    --run-dir handover-review/sanitized/optics-mechanical-ai-handoff/optics-langgraph/data-mcp/runs/<run_id> \
    --out demo-output/<run_id>.mp4  [--gif] [--robot xarm7|gantry] [--speed 1.0] [--exaggerate 10] [--camera close|wide]
```
Other options: `--allow-unreached` (skip a move whose IK does not converge and stamp UNREACHED; default is
to fail naming the waypoint), `--base X Y`, `--force-scale S`, `--fps 24`, `--width 1280 --height 720`,
`--cam-az/--cam-el/--cam-dist`. A small `<out>.report.json` (IK statistics, base/scale decision, marker frames,
a development contact diagnostic) is written next to the video. The run directory is opened read-only.

`check_render.py` (same arguments) renders and then verifies: run directory unchanged (sha256 + mtime +
size before/after), final drawn part poses equal the run's final `after` poses within 1e-6, all IK
waypoints converged, frame count/duration, no blank frames.

## Camera (`--camera close|wide`, default `close`)

- `close` (default): a tight, lower three-quarter view of the breadboard (elevation -24 degrees, more lateral than
  the wide shot) so the optics are clearly visible. The camera distance is derived from the board size
  (override with `--cam-dist`). The look-at point eases (critically damped, about 0.7 s time constant, no cuts)
  toward the component currently being handled (pick at the feed row, then placement). The arm base and shoulder
  may be out of frame; gripper, wrist and forearm stay visible. The part being handled is labelled beside
  it (not above, where the gripper is); the legend is a translucent two-column block. Display only: poses,
  sizes and every number come from the evidence exactly as before.
- `wide`: the previous framing, the whole arm and layout in frame (elevation -38 degrees by default).
- `--cam-az` / `--cam-el` override the azimuth / elevation in both modes.

## How the story is derived (only from the run's evidence files)

- `layout.json`: component ids, `movable`, target poses; `run.json`: labels/kinds, request, status,
  tests, the seeded object and magnitude; `result.json`: `motions_sent`.
- `team/sim-NN/simulation.json`: `before`/`after` poses, `moved_ids`, `position_errors_m`,
  `angle_errors_rad`; `team/sim-NN/plan.json`: the actions (operation, object, order).
- Attempt 1: each moved part is picked at its `before` pose and placed at its `after` pose, in plan
  order. Every later attempt: the corrective action (translate / align) on the single moved
  component, from that attempt's `before` to `after`, with a close-up inset and the measured error
  before/after from the evidence.

## Robot, reach and honesty details

- xArm7 from MuJoCo Menagerie (`assets/ufactory_xarm7`, BSD-3-Clause wording, see its LICENSE/NOTICE).
- Damped-least-squares IK on a gripper pinch site with a rotation-vector orientation error, joint
  limits, nullspace posture term, warm start, and a convergence check; solutions with arm
  self/table intersection or a joint-branch flip are rejected. Top-down grasp, gripper yaw follows the
  part yaw. Each pick/place has approach, lift and retreat waypoints; moves are joint-space
  interpolations of Cartesian sub-waypoints with smooth easing; the held part is attached to the
  gripper kinematically.
- The arm base is placed outside the part cluster at the position that minimises the farthest
  pick/place distance and is IK-verified. If no base position reaches everything, the whole layout is
  scaled uniformly (scale printed in the overlay; component sizes scale with it); never silently.
- Parts are schematic mounts (disc / plate / cube / box on a post) chosen from the component kind,
  with a small grip tab on movable parts. The recorded `z` is used as the optic height; the feed row
  `z = 0` is not depicted separately. A breadboard is drawn as a thin plate sized from the layout.
- `--robot gantry` renders the same story with a floating suction tool instead of the arm.
- Requirements are pinned in `requirements.txt` (MuJoCo 4.0.1, numpy, scipy, imageio, imageio-ffmpeg, pillow).
