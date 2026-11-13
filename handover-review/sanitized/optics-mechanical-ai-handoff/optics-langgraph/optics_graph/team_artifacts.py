"""ROSi and evidence byte contracts used at Controller, Output and L1."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED, ZIP_STORED
import zlib

from .contracts import validate_layout
from .experiments import is_registered, validate_registered_layout
from .sim_io import decode_json, read_bytes
from .team_trajectory import check_trajectory

LIMIT = 256_000
MEMBERS = {"plan.json", "manifest.json", "result.json"}


def encode(value, *, limit=LIMIT):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    if limit is not None and len(data) > limit:
        raise ValueError("JSON exceeds 256 KB")
    return data


def read_journal_artifact(path, *, limit):
    if path.name in {
        "completion.json", "input.json", "event.json", "decision.json",
        "prompt.json", "prompt.txt", "agent-prompt.txt", "test-agent-prompt.txt",
        "test-model-response.json",
    } or path.name.startswith(("model-", "feedback-")):
        limit = None
    return read_bytes(path, limit=limit)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def make_archive(plan_data, result, *, corrupt_hash=False):
    try:
        return _make_archive(plan_data, result, corrupt_hash=corrupt_hash)
    except (UnicodeError, KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError("Invalid simulator output") from exc


def _make_archive(plan_data, result, *, corrupt_hash=False):
    plan = decode_json(plan_data.decode("utf-8"))
    plan_hash = digest(plan_data)
    if result["plan_hash"] != plan_hash:
        raise ValueError("Simulator read a different ROSi")
    result_data = encode(result)
    manifest = {k: plan[k] for k in ("scene_id", "robot_model_id", "calibration_id", "tool_frame")}
    manifest.update(version=1, engine=result["engine"], sim_version=result["sim_version"],
                    status=result["status"], completed_at=datetime.now(timezone.utc).isoformat(),
                    plan_hash="0"*64 if corrupt_hash else plan_hash, result_hash=digest(result_data))
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("plan.json", plan_data)
        archive.writestr("manifest.json", encode(manifest))
        archive.writestr("result.json", result_data)
    return output.getvalue()


def read_archive(blob, plan_data):
    try:
        return _read_archive(blob, plan_data)
    except (BadZipFile, EOFError, UnicodeError, KeyError, TypeError, IndexError,
            OverflowError, AttributeError, NotImplementedError, RuntimeError, zlib.error) as exc:
        raise ValueError("Invalid simulation evidence") from exc


def _read_archive(blob, plan_data):
    if not isinstance(blob, bytes) or len(blob) > 3 * LIMIT + 4096:
        raise ValueError("ZIP exceeds size limit")
    if not isinstance(plan_data, bytes) or len(plan_data) > LIMIT:
        raise ValueError("ROSi exceeds size limit")
    with ZipFile(BytesIO(blob)) as archive:
        entries = archive.infolist()
        if len(entries) != len(MEMBERS) or {i.filename for i in entries} != MEMBERS:
            raise ValueError("ZIP member names or count rejected")
        if any(i.file_size > LIMIT or i.flag_bits & 1 or i.compress_type not in (ZIP_STORED, ZIP_DEFLATED) for i in entries):
            raise ValueError("ZIP member limit or encryption rejected")
        saved = archive.read("plan.json")
        manifest = decode_json(archive.read("manifest.json").decode("utf-8"))
        raw_result = archive.read("result.json")
    result = decode_json(raw_result.decode("utf-8"))
    plan = decode_json(plan_data.decode("utf-8"))
    plan_hash = digest(plan_data)
    required = {"version", "scene_id", "robot_model_id", "calibration_id", "tool_frame",
                "engine", "sim_version", "status", "completed_at", "plan_hash", "result_hash"}
    result_fields = {"status", "error", "waypoints_completed", "engine", "sim_version", "poses", "plan_hash"}
    if (not isinstance(manifest, dict) or not isinstance(result, dict) or not isinstance(plan, dict)
            or set(result) != result_fields or not isinstance(result["poses"], dict)
            or type(result["waypoints_completed"]) is not int
            or not isinstance(result["sim_version"], str) or not result["sim_version"]
            or not isinstance(plan.get("waypoints"), list)
            or set(manifest) != required or type(manifest["version"]) is not int or manifest["version"] != 1
            or saved != plan_data or manifest["plan_hash"] != plan_hash
            or result["plan_hash"] != plan_hash or manifest["result_hash"] != digest(raw_result)
            or any(manifest[k] != plan[k] for k in ("scene_id", "robot_model_id", "calibration_id", "tool_frame"))
            or any(manifest[k] != result[k] for k in ("engine", "sim_version", "status"))
            or manifest["engine"] != "mujoco" or manifest["status"] != "succeeded"
            or result["error"] is not None or result["waypoints_completed"] != len(plan["waypoints"])):
        raise ValueError("ZIP evidence does not match this verified ROSi")
    if not isinstance(manifest["completed_at"], str):
        raise ValueError("Simulation evidence timestamp rejected")
    completed = datetime.fromisoformat(manifest["completed_at"])
    if completed.tzinfo is None or completed.utcoffset() is None:
        raise ValueError("Simulation evidence timestamp must include timezone")
    age = datetime.now(timezone.utc) - completed
    if not 0 <= age.total_seconds() <= 86400:
        raise ValueError("Simulation evidence expired or from the future")
    return manifest, result


def evaluate_poses(result, layout, scene, requirements):
    try:
        return _evaluate_poses(result, layout, scene, requirements)
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError("Measured pose format rejected") from exc


def _evaluate_poses(result, layout, scene, requirements):
    candidate = deepcopy(layout)
    expected = {c["object_id"] for c in candidate["components"]}
    if not isinstance(result, dict) or not isinstance(result.get("poses"), dict) or set(result["poses"]) != expected:
        raise ValueError("Simulation result does not cover scene inventory")
    measured = []
    for item in candidate["components"]:
        object_id = item["object_id"]
        pose, target = result["poses"][object_id], item["target_pose"]
        if not isinstance(pose, dict) or set(pose) != {"frame_id", "position_m", "orientation_xyzw"}:
            raise ValueError("Measured pose format rejected")
        xyz, q = pose["position_m"], pose["orientation_xyzw"]
        if (not isinstance(xyz, list) or not isinstance(q, list)
                or len(xyz) != 3 or len(q) != 4
                or not all(type(n) in (int, float) and math.isfinite(n) for n in xyz + q)):
            raise ValueError("Measured pose format rejected")
        if (pose["frame_id"] != "optical_table" or len(xyz) != 3 or len(q) != 4
                or not math.isclose(math.hypot(*q), 1, abs_tol=1e-6)):
            raise ValueError("Measured pose format rejected")
        cosine = abs(sum(a*b for a, b in zip(q, target["orientation_xyzw"]))) / math.hypot(*q)
        angle = 2 * math.acos(min(1., cosine))
        position = math.dist(xyz, target["position_m"])
        tolerance = layout["tolerance"] if item["movable"] else {"position_m": 1e-6, "angle_rad": 1e-6}
        if (position > tolerance["position_m"] or angle > tolerance["angle_rad"]
                or abs(xyz[2]) > .0005 or math.hypot(*q[:2]) > .0025):
            raise ValueError(f"Measured pose is out of tolerance: {object_id}")
        measured.append({"object_id":object_id,"position_error_m":position,"angle_error_rad":angle})
        if item["movable"]:
            norm = math.hypot(*q[2:])
            item["target_pose"] = {"frame_id":"optical_table", "position_m":[*xyz[:2], 0],
                                   "orientation_xyzw":[0, 0, q[2]/norm, q[3]/norm]}
    validate = validate_registered_layout if is_registered(scene) else validate_layout
    return validate(candidate, scene, requirements), measured


def accept_l1(plan_data, artifact, receipt, object_ids):
    """L1 consumer has no arm SDK and accepts only the verified artifact and ROSi."""
    if (not isinstance(receipt, dict) or digest(plan_data) != receipt.get("plan_hash") or digest(artifact) != receipt.get("artifact_sha256")):
        raise ValueError("L1 rejected bytes changed after Output")
    manifest, _ = read_archive(artifact, plan_data)
    try:
        check_trajectory(decode_json(plan_data.decode("utf-8")), object_ids)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("Invalid ROSi trajectory") from exc
    return {"mode":"local_l1_dry_run","plan_hash":digest(plan_data),
            "artifact_sha256":digest(artifact),"robot_model_id":manifest["robot_model_id"],
            "calibration_id":manifest["calibration_id"],"motions_sent":0}
