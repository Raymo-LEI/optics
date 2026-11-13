"""Validate a bounded recording of physical states from the MuJoCo worker."""
import math

from .sim_io import decode_json


def validate_playback(raw, plan, plan_hash, scene, result):
    recording = decode_json(raw.decode("utf-8"))
    if (not isinstance(recording, dict)
            or set(recording) != {"schema_version", "engine", "plan_hash", "step_s", "objects", "frames"}
            or recording["schema_version"] != "mujoco-playback/v1"
            or recording["engine"] != "mujoco" or recording["plan_hash"] != plan_hash
            or recording["step_s"] != plan["limits"]["step_s"]):
        raise ValueError("Playback metadata does not match the simulation")

    objects, frames = recording["objects"], recording["frames"]
    if not isinstance(objects, list) or len(objects) != len(scene["objects"]):
        raise ValueError("Playback object inventory rejected")
    names = sorted(scene["objects"])
    for name, obj in zip(names, objects):
        spec = scene["objects"][name]
        width, depth = spec["footprint_m"]
        if (not isinstance(obj, dict) or type(obj.get("movable")) is not bool
                or obj != {"id": name, "width_m": width, "depth_m": depth,
                           "height_m": spec["height_m"], "movable": spec["movable"]}):
            raise ValueError("Playback object metadata rejected")

    if not isinstance(frames, list) or not 2 <= len(frames) <= 240:
        raise ValueError("Playback frame count rejected")
    final_tick = plan["waypoints"][-1]["end_tick"]
    previous = -1
    waypoint = 0
    for frame in frames:
        if (not isinstance(frame, dict) or set(frame) != {"tick", "waypoint", "tool", "poses"}
                or type(frame["tick"]) is not int or not previous < frame["tick"] <= final_tick
                or type(frame["waypoint"]) is not int or not waypoint <= frame["waypoint"] <= len(plan["waypoints"])
                or not isinstance(frame["poses"], list) or len(frame["poses"]) != len(names)):
            raise ValueError("Playback frame or timing rejected")
        for pose in [frame["tool"], *frame["poses"]]:
            if (not isinstance(pose, list) or len(pose) != 4
                    or not all(type(n) in (int, float) and -4 <= n <= 4 for n in pose)):
                raise ValueError("Playback contains an invalid pose")
        previous, waypoint = frame["tick"], frame["waypoint"]
    if (frames[0]["tick"] != 0 or frames[0]["waypoint"] != 0
            or previous != final_tick or waypoint != len(plan["waypoints"])):
        raise ValueError("Playback does not cover the complete run")

    for name, pose in zip(names, frames[-1]["poses"]):
        xyz = [pose[0], pose[1], pose[2] - scene["objects"][name]["height_m"] / 2]
        if math.dist(xyz, result["poses"][name]["position_m"]) > 1e-6:
            raise ValueError("Playback final pose differs from simulation evidence")
