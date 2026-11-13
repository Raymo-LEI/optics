"""Run and independently verify the local Phase 1 physics simulation."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import stat
import subprocess
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import load_scene, validate_layout, validate_plan
from .settings import PROJECT
from .sim_io import read_json, write_json

WORKER = Path(__file__).with_name("mujoco_worker.py")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Pose(Record):
    frame_id: Literal["optical_table"]
    position_m: list[float]
    orientation_xyzw: list[float]


class Step(Record):
    action_id: str
    primitive: str
    status: Literal["succeeded", "failed"]
    end_s: float


class Measurement(Record):
    object_id: str
    position_error_m: float
    angle_error_rad: float
    attempts: Literal[1, 2]


class SignedMeasurement(Measurement):
    signed_xy_m: list[float] = Field(min_length=2, max_length=2)


class Recovery(Record):
    layer: Literal["observation", "placement"]
    object_id: str
    reason: Literal["injected_missing_measurement", "placement_out_of_tolerance"]
    attempt: Literal[1] | None = None
    position_error_m: float | None = None
    angle_error_rad: float | None = None
    attempts: Literal[1] | None = None


class Failure(Record):
    code: str
    message: str
    action_id: str


class Evidence(Record):
    version: Literal[2]
    job_sha256: str
    scene_xml_sha256: str
    engine_version: str
    status: Literal["succeeded", "failed"]
    error: Failure | None
    steps: list[Step]
    verifications: list[Measurement]
    placement_errors: list[SignedMeasurement]
    recoveries: list[Recovery]
    simulation_time_s: float
    support_contacts: int
    final_poses: dict[str, Pose]


def file_digest(path, limit=256_000):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Simulation evidence must be a regular file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Simulation evidence exceeds the size limit")
    return hashlib.sha256(raw).hexdigest()


def verify_evidence(directory, job):
    evidence = Evidence.model_validate(read_json(directory / "evidence.json"))
    if (read_json(directory / "job.json") != job
            or evidence.job_sha256 != file_digest(directory / "job.json")
            or evidence.scene_xml_sha256 != file_digest(directory / "scene.xml")):
        raise ValueError("Simulation evidence does not match the input or the generated scene file")
    steps = evidence.steps
    actions = job["plan"]["actions"]
    observed_ids = {a["args"]["object_id"] for a in actions if a["primitive"] == "verify_placement"}
    if (len(evidence.placement_errors) > 6
            or any(m.object_id not in observed_ids or not 0 <= m.angle_error_rad <= math.pi
                   or math.hypot(*m.signed_xy_m) > m.position_error_m + 1e-8
                   for m in evidence.placement_errors)
            or len({(m.object_id, m.attempts) for m in evidence.placement_errors}) != len(evidence.placement_errors)):
        raise ValueError("The simulation error observation does not match this round's placement-check action")
    elapsed = [step.end_s - start for step, start in zip(steps, [0, *(s.end_s for s in steps)])]
    if (not steps or len(steps) > len(actions)
            or any((s.action_id, s.primitive) != (a["action_id"], a["primitive"])
                   for s, a in zip(steps, actions))
            or any(s.status != "succeeded" for s in steps[:-1])
            or not 0 <= steps[0].end_s <= evidence.simulation_time_s
            or any(not 0 <= b.end_s - a.end_s for a, b in zip(steps, steps[1:]))
            or any(duration > action["timeout_s"] + 1e-9 for duration, action in zip(elapsed, actions))
            or steps[-1].end_s != evidence.simulation_time_s
            or evidence.support_contacts < 0):
        raise ValueError("The simulated action trajectory does not match the current plan")
    if evidence.status == "failed":
        if (evidence.error is None or steps[-1].status != "failed"
                or evidence.error.action_id != steps[-1].action_id or evidence.final_poses):
            raise ValueError("The simulation failure record is inconsistent")
        return evidence, None
    if evidence.error or len(steps) != len(actions) or steps[-1].status != "succeeded":
        raise ValueError("The simulated action evidence is incomplete")

    layout = deepcopy(job["layout"])
    components = layout["components"]
    expected = {item["object_id"] for item in components}
    movable = {item["object_id"] for item in components if item["movable"]}
    if (set(evidence.final_poses) != expected or len(evidence.verifications) != len(movable)
            or {m.object_id for m in evidence.verifications} != movable):
        raise ValueError("The placement evidence does not cover the scene components")
    placements = [r.object_id for r in evidence.recoveries if r.layer == "placement"]
    if (len(set(placements)) != len(placements)
            or any(r.object_id not in movable for r in evidence.recoveries)
            or sum(r.layer == "observation" for r in evidence.recoveries) > 1
            or any(not 0 <= m.position_error_m <= layout["tolerance"]["position_m"]
                   or not 0 <= m.angle_error_rad <= layout["tolerance"]["angle_rad"]
                   or m.attempts != (2 if m.object_id in placements else 1)
                   for m in evidence.verifications)):
        raise ValueError("The simulation correction count or measurements do not follow the contract")
    for item in components:
        pose = evidence.final_poses[item["object_id"]]
        xyz, q = pose.position_m, pose.orientation_xyzw
        target = item["target_pose"]
        if len(xyz) != 3 or len(q) != 4 or not math.isclose(math.hypot(*q), 1, abs_tol=1e-6):
            raise ValueError("Placement-evidence coordinates or quaternion are invalid")
        tol = layout["tolerance"]
        target_q = target["orientation_xyzw"]
        cos_half = abs(sum(x*y for x, y in zip(q, target_q))) / (math.hypot(*q) * math.hypot(*target_q))
        angle = 2 * math.acos(min(1., cos_half))
        distance = math.dist(xyz, target["position_m"])
        if distance > tol["position_m"] or angle > tol["angle_rad"]:
            raise ValueError(f"Final placement is out of tolerance: {item['object_id']}")
        if abs(xyz[2]) > .0005 or math.hypot(*q[:2]) > .0025:
            raise ValueError(f"Component left the optical support plane: {item['object_id']}")
        if item["movable"]:
            norm = math.hypot(*q[2:])
            item["target_pose"] = {"frame_id": pose.frame_id,
                                   "position_m": [xyz[0], xyz[1], 0],
                                   "orientation_xyzw": [0, 0, q[2]/norm, q[3]/norm]}
        elif distance > 1e-6 or angle > 1e-6:
            raise ValueError(f"A fixed component moved: {item['object_id']}")
    return evidence, validate_layout(layout, job["scene"], job["requirements"])


def run_local_sim(state, directory):
    job = {"version": 2, "run_id": state["run_id"], "layout": state["layout"],
           "plan": state["plan"], "scene": state["scene"],
           "requirements": state["intent"]["requirements"],
           "scenario": state.get("sim_scenario", "nominal"), "worker_sha256": file_digest(WORKER),
           "command_offset_xy_m": state.get("command_offset_xy_m", [0, 0])}
    plan_report = validate_plan(job["plan"], job["layout"], state["validation"])
    layout_report = validate_layout(job["layout"], job["scene"], job["requirements"])
    result = {"status": "failed", "physical_execution": False, "evidence_kind": "local_mujoco",
              "plan_sha256": plan_report["plan_sha256"], "layout_sha256": plan_report["layout_sha256"],
              "artifact_directory": directory.name}
    if not (plan_report["passed"] and layout_report["passed"]
            and job["scene"] == load_scene() and not job["scene"]["obstacles"]
            and job["scene"]["support_height_m"] == 0
            and layout_report["validation_id"] == state["validation"]["validation_id"]):
        return {**result, "error": {"code": "SIM_INPUT_INVALID", "message": "The simulation input failed independent review",
                                   "issues": layout_report["issues"] + plan_report["issues"]}}
    try:
        directory.mkdir(parents=True)
    except FileExistsError:
        if not (directory / "evidence.json").is_file():
            return {**result, "error": {"code": "SIM_INCOMPLETE", "message": "An interrupted simulation record was found. Create a new run"}}
    else:
        write_json(directory / "job.json", job)
        python = PROJECT / ".venv-mujoco" / "bin" / "python"
        try:
            child = subprocess.run([str(python), "-E", "-s", str(WORKER), str(directory.resolve())],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, timeout=40, check=False,
                                   env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {**result, "error": {"code": "SIM_ENGINE_UNAVAILABLE", "message": type(exc).__name__}}
        if child.returncode != 0:
            lines = child.stderr.decode(errors="replace").strip().splitlines()
            diagnostic = lines[-1][:300] if lines else "No diagnosis"
            return {**result, "error": {"code": "SIM_ENGINE_FAILED",
                                       "message": f"MuJoCo exit code {child.returncode}: {diagnostic}"}}

    try:
        evidence, optical_report = verify_evidence(directory, job)
    except (ValueError, OSError) as exc:
        return {**result, "error": {"code": "SIM_EVIDENCE_INVALID", "message": str(exc)[:500]}}
    result.update(evidence.model_dump())
    result["planar_optical_recheck"] = optical_report
    if optical_report is not None and not optical_report["passed"]:
        result.update(status="failed", error={"code": "SIM_OPTICAL_RECHECK_FAILED",
                                             "message": "The in-plane optical path after placement failed review",
                                             "issues": optical_report["issues"]})
    return result
