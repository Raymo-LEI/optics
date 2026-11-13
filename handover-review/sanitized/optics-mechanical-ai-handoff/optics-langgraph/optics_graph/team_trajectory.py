"""Local ROSi profiles for the Cartesian MuJoCo workbench."""
import math

MODEL_ID = "phase1-cartesian-v1"
CALIBRATION_ID = "phase1-optical-table-v1"
TOOL_FRAME = "suction_tcp"
STEP_S = .002
GRIP_OFFSET = .050
HOME = [.03, -.12, .16, 0.]
BOUNDS = ((0., .65), (-.18, .55), (.05, .24))
MAX_LINEAR_PER_TICK = .0006
MAX_YAW_PER_TICK = .025
MAX_TICKS = 40_000


def workbench_bounds(scene_id):
    if scene_id.startswith("registered-"):
        return ((0., .85), (-.18, .74), (.05, .24))
    return BOUNDS


def grip_offset(scene_id):
    return .065 if scene_id.startswith("registered-") else GRIP_OFFSET


def plan_limits(scene_id):
    max_ticks = 80_000 if scene_id.startswith("registered-") else MAX_TICKS
    return {"step_s": STEP_S, "max_ticks": max_ticks,
            "max_linear_m_per_tick": MAX_LINEAR_PER_TICK, "max_yaw_rad_per_tick": MAX_YAW_PER_TICK}


def yaw(quaternion):
    return math.remainder(2 * math.atan2(quaternion[2], quaternion[3]), 2 * math.pi)


def build_trajectory(prob, layout, scene, plan_hashes, travel_z):
    waypoints = []
    cursor = HOME
    elapsed = 0

    def append(operation, *, pose=None, ticks=1, object_id=None):
        nonlocal cursor, elapsed
        if pose is not None:
            ticks = max(1, math.ceil(math.dist(cursor[:3], pose[:3]) / MAX_LINEAR_PER_TICK),
                        math.ceil(abs(math.remainder(pose[3] - cursor[3], 2 * math.pi)) / MAX_YAW_PER_TICK))
            cursor = pose
        elapsed += ticks
        waypoints.append({"op": operation, "pose": list(cursor), "end_tick": elapsed, "object_id": object_id})

    if not .15 <= travel_z <= .22:
        raise ValueError("Travel height outside local workcell limits")
    for component in sorted(layout["components"], key=lambda c: c["object_id"]):
        if not component["movable"]:
            continue
        object_id = component["object_id"]
        start = prob["observations"][object_id]
        target = component["target_pose"]
        x, y = start["position_m"][:2]
        tx, ty = target["position_m"][:2]
        approach = scene["objects"][object_id]["height_m"] / 2 + grip_offset(scene["scene_revision"])
        start_yaw = start["yaw_rad"]
        target_yaw = yaw(target["orientation_xyzw"])
        append("move", pose=[x, y, travel_z, start_yaw])
        append("move", pose=[x, y, approach, start_yaw])
        append("attach", object_id=object_id, ticks=12)
        append("move", pose=[x, y, travel_z, start_yaw])
        append("move", pose=[tx, ty, travel_z, target_yaw])
        append("move", pose=[tx, ty, approach + .0015, target_yaw])
        append("hold", ticks=70)
        append("release", object_id=object_id)
        append("move", pose=[tx, ty, travel_z, target_yaw])
        append("hold", ticks=125)
    append("move", pose=HOME)
    return {
        "schema_version": "local-cartesian-rosi/v0",
        "robot_model_id": MODEL_ID,
        "calibration_id": CALIBRATION_ID,
        "tool_frame": TOOL_FRAME,
        "scene_id": scene["scene_revision"],
        "limits": plan_limits(scene["scene_revision"]),
        "initial_sha256": plan_hashes[0],
        "layout_sha256": plan_hashes[1],
        "waypoints": waypoints,
    }


def check_trajectory(plan, object_ids):
    fields = {"schema_version", "robot_model_id", "calibration_id", "tool_frame", "scene_id",
              "initial_sha256", "layout_sha256", "waypoints", "limits"}
    if (set(plan) != fields or plan["schema_version"] != "local-cartesian-rosi/v0"
            or plan["robot_model_id"] != MODEL_ID or plan["calibration_id"] != CALIBRATION_ID
            or plan["tool_frame"] != TOOL_FRAME
            or not isinstance(plan["scene_id"], str) or plan["limits"] != plan_limits(plan["scene_id"])):
        raise ValueError("ROSi metadata or limits rejected")
    bounds = workbench_bounds(plan["scene_id"])
    max_ticks = plan["limits"]["max_ticks"]
    for key in ("initial_sha256", "layout_sha256"):
        if len(plan[key]) != 64 or any(c not in "0123456789abcdef" for c in plan[key]):
            raise ValueError("ROSi input hash rejected")
    rows = plan["waypoints"]
    pattern = ["move", "move", "attach", "move", "move", "move", "hold", "release", "move", "hold"]
    if [r["op"] for r in rows] != pattern * len(object_ids) + ["move"]:
        raise ValueError("ROSi operation sequence rejected")
    cursor = HOME
    tick = 0
    held = None
    delivered = set()
    for row in rows:
        if set(row) != {"op", "pose", "end_tick", "object_id"}:
            raise ValueError("ROSi waypoint fields rejected")
        operation, pose, end, name = (row[k] for k in ("op", "pose", "end_tick", "object_id"))
        if (len(pose) != 4 or not all(type(n) in (int, float) and math.isfinite(n) for n in pose)
                or not all(lo <= n <= hi for n, (lo, hi) in zip(pose, bounds))
                or abs(pose[3]) > math.pi or type(end) is not int or not tick < end <= max_ticks):
            raise ValueError("ROSi coordinate or time limit rejected")
        span = end - tick
        if operation == "move":
            if (name is not None or math.dist(cursor[:3], pose[:3]) > span * MAX_LINEAR_PER_TICK + 1e-9
                    or abs(math.remainder(pose[3] - cursor[3], 2 * math.pi)) > span * MAX_YAW_PER_TICK + 1e-9):
                raise ValueError("ROSi velocity or move target rejected")
        elif pose != cursor:
            raise ValueError("ROSi stationary operation changed tool pose")
        if operation == "attach":
            if held or name not in object_ids or name in delivered:
                raise ValueError("ROSi attach order rejected")
            held = name
        elif operation == "release":
            if not held or name != held:
                raise ValueError("ROSi release order rejected")
            delivered.add(name)
            held = None
        elif operation == "hold" and name is not None:
            raise ValueError("ROSi hold does not take an object")
        cursor, tick = pose, end
    if held or delivered != set(object_ids) or cursor != HOME:
        raise ValueError("ROSi final tool state rejected")
