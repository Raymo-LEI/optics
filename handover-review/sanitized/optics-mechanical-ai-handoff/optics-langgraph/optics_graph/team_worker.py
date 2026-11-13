"""Process boundary for camera measurements and exact ROSi replay in MuJoCo."""
import hashlib
import math
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import cv2
import mujoco
import numpy as np

if __package__:
    from .mujoco_worker import Workcell, make_model, SimulationStop
    from .sim_io import decode_json, read_json, write_json
    from .team_trajectory import CALIBRATION_ID, HOME, check_trajectory, grip_offset
else:
    from mujoco_worker import Workcell, make_model, SimulationStop
    from sim_io import decode_json, read_json, write_json
    from team_trajectory import CALIBRATION_ID, HOME, check_trajectory, grip_offset


CAMERAS = ((.27, .16, 1.), (.33, .20, 1.))
REGISTERED_CAMERAS = ((.40, .24, 1.42), (.45, .29, 1.42))
FOV = 45
SIZE = (640, 480)
REGISTERED_SIZE = (1280, 960)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def sorted_layout(layout):
    return {**layout, "components": sorted(layout["components"], key=lambda c: c["object_id"])}


def observe(scene, layout):
    components = {c["object_id"]: c for c in sorted_layout(layout)["components"]}
    moving = [c for c in components.values() if c["movable"]]
    root = ET.fromstring(make_model(scene, components))
    world = root.find("worldbody")
    camera_names = []
    registered = "registration" in scene
    cameras = REGISTERED_CAMERAS if registered else CAMERAS
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) if registered else None
    if registered:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, params)
    for index, position in enumerate(cameras):
        name = f"camera_{index}"
        camera_names.append(name)
        ET.SubElement(world, "camera", name=name, pos=" ".join(map(str, position)),
                      quat="1 0 0 0", fovy=str(FOV))
    for channel, component in enumerate(moving):
        object_id = component["object_id"]
        body = world.find(f"body[@name='{object_id}']")
        height = scene["objects"][object_id]["height_m"] / 2
        if registered:
            edge = min(.027, min(scene["objects"][object_id]["footprint_m"]) * .72)
            pattern = cv2.aruco.generateImageMarker(dictionary, channel, 6)
            def tile(x, y, half, shade, z):
                ET.SubElement(body, "geom", type="box", pos=f"{x} {y} {z}",
                              size=f"{half} {half} .00005", rgba=f"{shade} {shade} {shade} 1",
                              contype="0", conaffinity="0", mass="0")
            tile(0., 0., edge * 4 / 6, 1, height + .0004)
            for row, col in np.argwhere(pattern == 0):
                tile((col - 2.5)*edge/6, (2.5 - row)*edge/6, edge/12, 0, height + .0005)
            continue
        rgb = [0.02, 0.02, 0.02]
        rgb[channel % 3] = 1
        rgba = " ".join(map(str, [*rgb, 1]))
        # Large center mark, smaller direction mark. Both are optical geometry.
        for marker, (x, size) in enumerate(((0., .006), (.014, .0035))):
            ET.SubElement(body, "geom", name=f"tag_{object_id}_{marker}", type="box", pos=f"{x} 0 {height + .0005}",
                          size=f"{size} {size} .0005", rgba=rgba, contype="0", conaffinity="0", mass="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    width, height = REGISTERED_SIZE if registered else SIZE
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    model.vis.headlight.ambient[:] = [.85, .85, .85]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    focal = height / (2 * math.tan(math.radians(FOV / 2)))
    observed = {c["object_id"]: [] for c in moving}
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index, camera in enumerate(camera_names):
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            frames.append({"camera_id": camera, "rgb_sha256": digest(frame.tobytes()), "size": [width, height]})
            if registered:
                corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
                markers = {} if ids is None else {int(i): points.reshape(4,2) for i, points in zip(ids.ravel(), corners)}
            for channel, component in enumerate(moving):
                object_id = component["object_id"]
                if registered:
                    points = markers.get(channel)
                    if points is None:
                        raise ValueError(f"CAMERA_TAG_UNRESOLVED:{camera}:{object_id}")
                    center, heading = points.mean(axis=0), points[[1,2]].mean(axis=0)
                else:
                    mask = frame[:, :, channel] > 130
                    for other in range(3):
                        if other != channel:
                            mask &= frame[:, :, other] < 85
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                    blobs = sorted([(stat[-1], center) for stat, center in zip(stats[1:], centroids[1:]) if stat[-1] >= 8],
                                   key=lambda blob: -blob[0])
                    if len(blobs) != 2 or blobs[0][0] / blobs[1][0] < 1.4:
                        raise ValueError(f"CAMERA_TAG_UNRESOLVED:{camera}:{object_id}")
                    center, heading = blobs[0][1], blobs[1][1]
                cx, cy, cz = cameras[index]
                depth = cz - scene["objects"][object_id]["height_m"] - (.00055 if registered else .001)
                u, v = center
                direction = heading - center
                observed[object_id].append([cx + (u + .5 - width/2)*depth/focal,
                                             cy - (v + .5 - height/2)*depth/focal,
                                             math.atan2(-direction[1], direction[0])])
    result = {}
    for name, (a, b) in observed.items():
        if math.dist(a[:2], b[:2]) > (.003 if registered else .002) or abs(math.remainder(a[2] - b[2], 2*math.pi)) > .25:
            raise ValueError(f"CAMERA_DISAGREEMENT:{name}")
        result[name] = {"position_m": [float((a[0]+b[0])/2), float((a[1]+b[1])/2), 0.],
                        "yaw_rad": math.atan2(math.sin(a[2])+math.sin(b[2]), math.cos(a[2])+math.cos(b[2]))}
    return {"schema_version": "local-prob/v0", "calibration_id": CALIBRATION_ID,
            "scene_id": scene["scene_revision"], "frame_id": "optical_table",
            "source": "mujoco_aruco_two_view" if registered else "mujoco_rgb_two_view",
            "frames": frames, "observations": result}


def replay(scene, layout, plan, directory, *, playback_plan_hash=None):
    workcell = Workcell(scene, sorted_layout(layout), "nominal", directory, [0, 0])
    object_ids = {c["object_id"] for c in layout["components"] if c["movable"]}
    check_trajectory(plan, object_ids)
    workcell.tick(150)
    cursor, tick = HOME, 0
    completed = 0
    error = None
    frames = []
    sample_budget = max(60, min(180, 237 - len(plan["waypoints"])))
    sample_interval = max(20, math.ceil(plan["waypoints"][-1]["end_tick"] / sample_budget))

    def record(tick, waypoint):
        def pose(body):
            w, x, y, z = body.xquat
            yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
            return [round(float(value), 7) for value in (*body.xpos, yaw)]

        frame = {"tick": tick, "waypoint": waypoint, "tool": pose(workcell.data.body("tool")),
                 "poses": [pose(workcell.data.body(name)) for name in workcell.components]}
        if frames and frames[-1]["tick"] == tick:
            frames[-1] = frame
        else:
            frames.append(frame)

    if playback_plan_hash:
        record(0, 0)
    for row in plan["waypoints"]:
        operation, pose, end, name = (row[key] for key in ("op", "pose", "end_tick", "object_id"))
        duration = end - tick
        try:
            if operation == "attach":
                body = workcell.data.body(name)
                center = [pose[0], pose[1], pose[2] - grip_offset(scene["scene_revision"])]
                if math.dist(center, body.xpos) > .003:
                    raise SimulationStop("TRACKING_PICK_MISMATCH")
                workcell.data.eq_active[workcell.model.equality(f"grip_{name}").id] = 1
            elif operation == "release":
                workcell.data.eq_active[workcell.model.equality(f"grip_{name}").id] = 0
            if operation == "move":
                origin = np.asarray(cursor[:3])
                direction = np.asarray(pose[:3]) - origin
                dyaw = math.remainder(pose[3] - cursor[3], 2 * math.pi)
                for step in range(1, duration + 1):
                    fraction = step / duration
                    workcell.data.mocap_pos[0] = origin + fraction * direction
                    angle = (cursor[3] + fraction * dyaw) / 2
                    workcell.data.mocap_quat[0] = [math.cos(angle), 0, 0, math.sin(angle)]
                    workcell.tick()
                    if playback_plan_hash and (tick + step) % sample_interval == 0:
                        record(tick + step, completed)
            else:
                if playback_plan_hash:
                    for step in range(1, duration + 1):
                        workcell.tick()
                        if (tick + step) % sample_interval == 0:
                            record(tick + step, completed)
                else:
                    workcell.tick(duration)
        except SimulationStop as exc:
            error = {"code": str(exc).split(":")[0], "message": str(exc), "waypoint": completed}
            break
        cursor, tick = pose, end
        completed += 1
        if playback_plan_hash:
            record(tick, completed)
    if playback_plan_hash and not error:
        objects = []
        for name, component in workcell.components.items():
            spec = scene["objects"][name]
            width, depth = spec["footprint_m"]
            objects.append({"id": name, "width_m": width, "depth_m": depth,
                            "height_m": spec["height_m"], "movable": component["movable"]})
        write_json(directory / "playback.json", {"schema_version": "mujoco-playback/v1",
                   "engine": "mujoco", "plan_hash": playback_plan_hash,
                   "step_s": plan["limits"]["step_s"], "objects": objects, "frames": frames}, compact=True)
    return {"status": "failed" if error else "succeeded", "error": error,
            "waypoints_completed": completed, "engine": "mujoco", "sim_version": mujoco.__version__,
            "poses": {name: workcell.measure(name) for name in workcell.components} if not error else {}}


if __name__ == "__main__":
    stage, path = sys.argv[1], Path(sys.argv[2])
    request = read_json(path / "request.json")
    if stage == "vision":
        result = observe(request["scene"], request["layout"])
    elif stage == "sim":
        with (path / "plan.json").open("rb") as stream:
            plan_bytes = stream.read(256_001)
        if len(plan_bytes) > 256_000:
            raise ValueError("ROSi exceeds 256 KB")
        plan_hash = digest(plan_bytes)
        if plan_hash != request["plan_hash"]:
            raise ValueError("ROSi SHA-256 changed before MuJoCo consumed it")
        result = replay(request["scene"], request["layout"], decode_json(plan_bytes.decode("utf-8")), path,
                        playback_plan_hash=plan_hash if request.get("playback") else None)
        result["plan_hash"] = plan_hash
    else:
        raise ValueError("Unsupported worker stage")
    write_json(path / "response.json", result)
