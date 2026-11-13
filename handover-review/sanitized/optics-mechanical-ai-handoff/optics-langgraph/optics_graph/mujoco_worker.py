"""Execute checked Phase 1 plans with a simulated Cartesian suction tool."""
import hashlib
import math
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

if __package__:
    from .sim_io import read_json, write_json
    from .team_trajectory import grip_offset
else:
    from sim_io import read_json, write_json
    from team_trajectory import grip_offset

DT = .002
TRAVEL_Z = .16
GRIP_OFFSET = .050  # tool origin above component center
HOME = (.03, -.12, TRAVEL_Z)
FEED = ((.13, -.12), (.31, -.12), (.49, -.12))
PROFILES = ("nominal", "transient", "persistent")


class SimulationStop(RuntimeError):
    pass


def values(vector):
    return " ".join(str(float(x)) for x in vector)


def make_model(scene, components):
    root = ET.Element("mujoco", model="phase1_cartesian")
    ET.SubElement(root, "option", timestep=str(DT), integrator="implicitfast",
                  gravity="0 0 -9.81")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="support", type="plane", size="1 1 .1",
                  friction="1 .01 .001")
    tool = ET.SubElement(world, "body", name="tool", mocap="true", pos=values(HOME))
    ET.SubElement(tool, "geom", name="tool_geom", type="box", size=".010 .010 .006")
    equality = ET.SubElement(root, "equality")
    supply = iter(FEED)
    attachment_offset = grip_offset(scene["scene_revision"])
    for component in components.values():
        object_id = component["object_id"]
        spec = scene["objects"][object_id]
        pose = component["target_pose"]
        x, y = (spec["feed_xy_m"] if "feed_xy_m" in spec else next(supply)) if component["movable"] else pose["position_m"][:2]
        w, h = spec["footprint_m"]
        height = spec["height_m"]
        qx, qy, qz, qw = pose["orientation_xyzw"]
        body = ET.SubElement(world, "body", name=object_id, pos=values((x, y, height / 2)),
                             quat=values((1, 0, 0, 0) if component["movable"] else (qw, qx, qy, qz)))
        if component["movable"]:
            ET.SubElement(body, "freejoint")
            ET.SubElement(equality, "weld", name=f"grip_{object_id}", body1="tool", body2=object_id,
                          active="false", relpose=f"0 0 {-attachment_offset} 1 0 0 0",
                          solref=".015 1")
        ET.SubElement(body, "geom", name=f"geom_{object_id}", type="box",
                      size=values((w / 2, h / 2, height / 2)), mass=".15",
                      friction="1 .01 .001")
    return ET.tostring(root, encoding="unicode")


class Workcell:
    def __init__(self, scene, layout, scenario, path, offset):
        self.scene = scene
        self.components = {item["object_id"]: item for item in layout["components"]}
        self.scenario = scenario
        self.offset = offset
        self.held = None
        self.observations = {}
        self.recoveries = []
        self.placement_errors = []
        self.drifted = set()
        self.observation_fault_used = False
        self.contacts = 0
        self.deadline = math.inf
        xml = make_model(scene, self.components)
        (path / "scene.xml").write_text(xml, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.support = self.model.geom("support").id
        self.tool = self.model.geom("tool_geom").id
        self.geom_ids = {self.model.geom(f"geom_{i}").id for i in self.components}

    def tick(self, count=1):
        for _ in range(count):
            previous = self.data.time
            if previous + DT > self.deadline + 1e-9:
                raise SimulationStop("ACTION_SIM_TIME_EXCEEDED")
            mujoco.mj_step(self.model, self.data)
            if (not math.isclose(self.data.time, previous + DT, rel_tol=0, abs_tol=1e-10)
                    or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()):
                raise SimulationStop("PHYSICS_NONFINITE")
            for contact in self.data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                # Only component contacts with the support surface are expected.
                if self.support in (a, b) and (set((a, b)) - {self.support}) <= self.geom_ids:
                    if contact.dist < -.002:
                        raise SimulationStop("SUPPORT_PENETRATION")
                    self.contacts += 1
                else:
                    raise SimulationStop(f"UNEXPECTED_CONTACT:{a},{b}")

    def move(self, xyz, quat=None):
        start = self.data.mocap_pos[0].copy()
        target = np.asarray(xyz)
        if not all(lo <= x <= hi for x, (lo, hi) in zip(target, ((0, .65), (-.18, .55), (.05, .24)))):
            raise SimulationStop("TOOL_WORKSPACE_EXCEEDED")
        if quat is not None:
            # Rotation occurs only at travel height before attaching a component.
            self.data.mocap_quat[0] = quat
        count = max(1, math.ceil(np.linalg.norm(target - start) / .0006))
        for fraction in range(1, count + 1):
            self.data.mocap_pos[0] = start + (target - start) * fraction / count
            self.tick()

    def observe(self, object_id):
        for attempt in range(2):
            if self.scenario == "transient" and not self.observation_fault_used:
                self.observation_fault_used = True
                self.recoveries.append({"layer": "observation", "object_id": object_id,
                                        "reason": "injected_missing_measurement", "attempt": 1})
                self.tick(10)
                continue
            body = self.data.body(object_id)
            return (body.xpos.copy(), body.xquat.copy())
        raise SimulationStop("OBSERVATION_UNAVAILABLE")

    def pick(self, object_id, observed):
        if self.held is not None:
            raise SimulationStop("TOOL_ALREADY_HOLDING")
        position, quat = observed
        self.move((*position[:2], TRAVEL_Z), quat)
        self.move((*position[:2], position[2] + GRIP_OFFSET))
        self.tick(12)
        current = self.data.body(object_id).xpos
        if np.linalg.norm(current - position) > .0015:
            raise SimulationStop("PICK_POSE_CHANGED")
        self.data.eq_active[self.model.equality(f"grip_{object_id}").id] = 1
        self.held = object_id
        self.move((*position[:2], TRAVEL_Z))
        if self.data.body(object_id).xpos[2] < .08:
            raise SimulationStop("LIFT_FAILED")

    def place(self, object_id, pose):
        if self.held != object_id:
            raise SimulationStop("HELD_OBJECT_MISMATCH")
        x, y, z = pose["position_m"]
        x += self.offset[0]
        y += self.offset[1]
        height = self.scene["objects"][object_id]["height_m"]
        qx, qy, qz, qw = pose["orientation_xyzw"]
        if self.scenario == "persistent" or (self.scenario == "transient" and not self.drifted):
            x += .016  # explicit simulated placement fault
            self.drifted.add(object_id)
        self.move((x, y, TRAVEL_Z))
        # Interpolate rotation while the component is in free space.
        a = self.data.mocap_quat[0].copy()
        b = np.array([qw, qx, qy, qz])
        if np.dot(a, b) < 0:
            b = -b
        for index in range(1, 101):
            q = a + (b - a) * index / 100
            self.data.mocap_quat[0] = q / np.linalg.norm(q)
            self.tick()
        self.move((x, y, z + height / 2 + GRIP_OFFSET + .0015))
        self.tick(70)
        self.data.eq_active[self.model.equality(f"grip_{object_id}").id] = 0
        self.held = None
        self.move((x, y, TRAVEL_Z))
        self.tick(125)

    def measure(self, object_id):
        position, q = self.observe(object_id)
        height = self.scene["objects"][object_id]["height_m"]
        return {"frame_id": "optical_table",
                "position_m": [*map(float, position[:2]), float(position[2] - height / 2)],
                "orientation_xyzw": [*map(float, q[1:]), float(q[0])]}

    def verify(self, args):
        object_id, target, tol = args["object_id"], args["target_pose"], args["tolerance"]
        for attempt in range(2):
            actual = self.measure(object_id)
            distance = math.dist(actual["position_m"], target["position_m"])
            a, b = actual["orientation_xyzw"], target["orientation_xyzw"]
            angle = 2 * math.acos(min(1., abs(float(np.dot(a, b))) / (math.hypot(*a) * math.hypot(*b))))
            measurement = {"object_id": object_id, "position_error_m": distance, "angle_error_rad": angle,
                           "attempts": attempt + 1}
            self.placement_errors.append({**measurement, "signed_xy_m": [
                actual["position_m"][i] - target["position_m"][i] for i in range(2)
            ]})
            if distance <= tol["position_m"] and angle <= tol["angle_rad"]:
                return measurement
            if attempt == 0 and distance <= .025 and angle <= .15:
                self.recoveries.append({"layer": "placement", "object_id": object_id,
                                        "reason": "placement_out_of_tolerance", **measurement})
                self.pick(object_id, self.observe(object_id))
                self.place(object_id, target)
            else:
                raise SimulationStop(f"PLACEMENT_OUT_OF_TOLERANCE: {distance * 1000:.3f} mm, {math.degrees(angle):.3f} deg")

    def run(self, plan):
        steps, verifications = [], []
        error = None
        for action in plan["actions"]:
            primitive, args = action["primitive"], action["args"]
            object_id = args.get("object_id")
            start = float(self.data.time)
            self.deadline = start + action["timeout_s"]
            try:
                if primitive == "initialize":
                    self.tick(150)
                elif primitive == "observe_scene":
                    self.observations[action["action_id"]] = {
                        i: self.observe(i) for i in args["object_ids"]
                    }
                elif primitive == "pick":
                    self.pick(object_id, self.observations[args["observation_ref"]][object_id])
                elif primitive == "place":
                    self.place(object_id, args["target_pose"])
                elif primitive == "verify_placement":
                    verifications.append(self.verify(args))
                elif primitive == "home":
                    self.move(HOME)
            except SimulationStop as exc:
                error = {"code": str(exc).split(":")[0], "message": str(exc),
                         "action_id": action["action_id"]}
            steps.append({"action_id": action["action_id"], "primitive": primitive,
                          "status": "failed" if error else "succeeded", "end_s": float(self.data.time)})
            if error:
                break
        return {"status": "failed" if error else "succeeded", "error": error, "steps": steps,
                "verifications": verifications, "recoveries": self.recoveries,
                "placement_errors": self.placement_errors,
                "simulation_time_s": float(self.data.time), "support_contacts": self.contacts,
                "final_poses": {i: self.measure(i) for i in self.components} if not error else {}}


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    path = directory / "job.json"
    job = read_json(path)
    if (type(job) is not dict or set(job) != {"version", "run_id", "layout", "plan", "scene",
            "requirements", "scenario", "worker_sha256", "command_offset_xy_m"} or job["version"] != 2
            or job["scenario"] not in PROFILES
            or job["worker_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError("Unsupported simulation job contract")
    offset = job["command_offset_xy_m"]
    if len(offset) != 2 or not all(type(v) in (float, int) and math.isfinite(v) for v in offset) or math.hypot(*offset) > .025:
        raise ValueError("Command offset outside simulation limit")
    workcell = Workcell(job["scene"], job["layout"], job["scenario"], directory, offset)
    result = workcell.run(job["plan"])
    result.update(version=2, job_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  scene_xml_sha256=hashlib.sha256((directory / "scene.xml").read_bytes()).hexdigest(),
                  engine_version=mujoco.__version__)
    write_json(directory / "evidence.json", result)
