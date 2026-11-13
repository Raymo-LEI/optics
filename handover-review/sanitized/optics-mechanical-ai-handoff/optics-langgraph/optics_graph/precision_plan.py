"""The local xArm7 closed-loop plan and its evidence envelope."""
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from .legacy import EXPECTED_NAMES
from .sim_io import decode_json, read_bytes
from .team_artifacts import LIMIT, digest, encode

SCHEMA = "local-xarm7-rosi/v1"


def file_hash(path):
    from hashlib import file_digest
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checkpoint_files(source):
    names = ("result.json", "world_model.mjb", "trajectory.npz")
    return {name: file_hash(source / name) for name in names}


def read_checkpoint(root, record):
    if not isinstance(record, dict) or set(record) != {"run_id", "directory", "files"}:
        raise ValueError("Checkpoint reference rejected")
    run_id, directory = record["run_id"], record["directory"]
    if not re.fullmatch(r"[0-9a-f]{32}", run_id) or not re.fullmatch(
            r"team/precision_(?:sim|repair)-[0-9]+/simulation", directory):
        raise ValueError("Checkpoint path rejected")
    source = (root / "runs" / run_id / directory).resolve()
    if not source.is_relative_to(root.resolve()) or checkpoint_files(source) != record["files"]:
        raise ValueError("Checkpoint files changed")
    return source


def validate_plan(plan, layout_hash, observation_hash):
    if not isinstance(plan, dict) or set(plan) != {
        "schema_version", "scene_id", "target_layout_sha256", "observed_layout_sha256",
        "operation", "object_ids", "checkpoint", "repair_action", "source_layout_sha256",
    } or plan["schema_version"] != SCHEMA or plan["scene_id"] != "michelson_full":
        raise ValueError("Unsupported xArm7 ROSi")
    if plan["target_layout_sha256"] != layout_hash or plan["observed_layout_sha256"] != observation_hash:
        raise ValueError("ROSi input binding mismatch")
    for key in ("target_layout_sha256", "observed_layout_sha256", "source_layout_sha256"):
        if not isinstance(plan[key], str) or not re.fullmatch(r"[0-9a-f]{64}", plan[key]):
            raise ValueError("ROSi input hash format rejected")
    objects = plan["object_ids"]
    if not isinstance(objects, list) or any(not isinstance(name, str) for name in objects):
        raise ValueError("ROSi object list rejected")
    operation, action = plan["operation"], plan["repair_action"]
    if operation == "assemble":
        valid = objects == sorted(EXPECTED_NAMES) and plan["checkpoint"] is None and action is None
    elif operation == "repair_part":
        valid = (len(objects) == 1 and objects[0] in EXPECTED_NAMES and plan["checkpoint"] is not None
                 and action in {"reseat_drifted_parts", "guard_camera_margin", "replan_packed_lift",
                                "refine_repair_seating", "replan_supported_turn_lift", "replan_corridor_lift",
                                "replan_visual_compensation"})
    elif operation == "repair_optics":
        valid = objects == [] and plan["checkpoint"] is not None and action == "extend_alignment_search"
    else:
        valid = False
    if not valid:
        raise ValueError("ROSi would execute outside the permitted correction scope")


def evidence_zip(raw_plan, result):
    result_data = encode(result)
    manifest = encode({"version": 1, "engine": "mujoco", "plan_hash": digest(raw_plan),
                       "result_hash": digest(result_data)})
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as bundle:
        for name, data in (("plan.json", raw_plan), ("manifest.json", manifest), ("result.json", result_data)):
            bundle.writestr(name, data)
    return output.getvalue()


def read_evidence(blob, raw_plan):
    with ZipFile(BytesIO(blob)) as bundle:
        entries = bundle.infolist()
        names = {"plan.json", "manifest.json", "result.json"}
        if len(entries) != len(names) or {e.filename for e in entries} != names or any(
                e.file_size > LIMIT or e.flag_bits & 1 or e.compress_type != ZIP_DEFLATED for e in entries):
            raise ValueError("Precision artifact member rejected")
        if bundle.read("plan.json") != raw_plan:
            raise ValueError("Precision artifact differs from the executed ROSi")
        raw_result = bundle.read("result.json")
        result = decode_json(raw_result.decode("utf-8"))
        if not isinstance(result, dict) or raw_result != encode(result):
            raise ValueError("Precision artifact result format rejected")
        manifest = decode_json(bundle.read("manifest.json").decode("utf-8"))
    if manifest != {"version": 1, "engine": "mujoco", "plan_hash": digest(raw_plan),
                    "result_hash": digest(raw_result)}:
        raise ValueError("Precision artifact hash mismatch")
    return result


def verify_evidence(blob, raw_plan, result):
    if encode(read_evidence(blob, raw_plan)) != encode(result):
        raise ValueError("Precision artifact differs from the executed result")


def accept_precision_l1(raw_plan, blob, verification):
    """Receive the local xArm7 intent and accepted evidence without commanding hardware."""
    plan_hash, artifact_hash = digest(raw_plan), digest(blob)
    if (not isinstance(verification, dict) or verification.get("plan_hash") != plan_hash
            or verification.get("artifact_sha256") != artifact_hash):
        raise ValueError("L1 rejected precision bytes changed after Output")
    plan = decode_json(raw_plan.decode("utf-8"))
    validate_plan(plan, verification.get("target_layout_sha256"), verification.get("observed_layout_sha256"))
    result = read_evidence(blob, raw_plan)
    checked = result.get("verification") or {}
    if (result.get("success") is not True or result.get("bindings_verified") is not True
            or result.get("plan_hash") != plan_hash or checked.get("accepted") is not True
            or result.get("added_sensor_noise") != {"rgb_std_8bit":0., "optical_std":0.}
            or plan["source_layout_sha256"] != result.get("source_sha256")
            or any(not isinstance(result.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", result[key])
                   or result[key] != verification.get(key) for key in ("source_sha256", "trajectory_sha256"))
            or any(checked.get(key, {}).get("passed") is not True or checked[key] != verification.get(key)
                   for key in ("mechanical", "optical"))):
        raise ValueError("L1 precision evidence did not pass bound independent acceptance")
    if plan["operation"] != "assemble":
        recovery = result.get("recovery") or {}
        objects = plan["object_ids"]
        if (recovery.get("retained_scene") is not True
                or recovery.get("source_report_sha256") != plan["checkpoint"]["files"]["result.json"]
                or recovery.get("target_object_ids") != objects or recovery.get("commanded_object_ids") != objects
                or set(recovery.get("untouched_object_ids", [])) != EXPECTED_NAMES-set(objects)):
            raise ValueError("L1 precision repair scope does not match the accepted ROSi")
    return {"mode":"local_precision_l1_dry_run", "plan_hash":plan_hash, "artifact_sha256":artifact_hash,
            "trajectory_sha256":result["trajectory_sha256"], "schema_version":SCHEMA,
            "object_ids":plan["object_ids"], "motions_sent":0}
