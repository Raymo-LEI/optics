"""Reference workers for the local file interface, with no hardware transport."""
from pathlib import Path
import sys

from .sim_io import decode_json, read_bytes, write_json
from .team_services import BLOB_LIMIT, JOB_VERSION, digest, job_identity
from .team_trajectory import build_trajectory


def run(path):
    # Keep vision and simulation workers usable in the separate MuJoCo environment.
    request_path = path / "service-request.json"
    raw = read_bytes(request_path)
    job = decode_json(raw.decode("utf-8"))
    if job["version"] != JOB_VERSION:
        raise ValueError("Unknown service protocol")
    stage = job["stage"]
    expected_files = {"sim": {"plan.json"}, "execution": {"plan.json", "artifact.zip"}}.get(stage, set())
    if set(job["files"]) != expected_files:
        raise ValueError("Unexpected service attachments")
    files = {}
    for name in expected_files:
        files[name] = read_bytes(path / name, limit=BLOB_LIMIT)
        if digest(files[name]) != job["files"][name]:
            raise ValueError("Service attachment hash or size mismatch")
    payload = job["payload"]
    if stage == "vision":
        from .team_worker import observe
        result = observe(payload["scene"], payload["layout"])
    elif stage == "controller":
        plan = build_trajectory(payload["prob"], payload["layout"], payload["scene"],
                                payload["hashes"], payload["travel_z_m"])
        # The controller owns the file bytes. The graph validates them without reserializing.
        plan_path = path / "candidate-plan.json"
        write_json(plan_path, plan)
        result = {"plan_sha256": digest(read_bytes(plan_path))}
    elif stage == "sim":
        from .team_worker import replay
        plan = files["plan.json"]
        if digest(plan) != payload["plan_hash"]:
            raise ValueError("Simulation plan hash mismatch")
        result = replay(payload["scene"], payload["layout"], decode_json(plan.decode("utf-8")), path,
                        playback_plan_hash=digest(plan) if payload.get("playback") else None)
        result["plan_hash"] = digest(plan)
    elif stage == "execution":
        from .precision_plan import SCHEMA, accept_precision_l1
        from .team_artifacts import accept_l1
        if payload.get("plan_schema") == SCHEMA:
            result = accept_precision_l1(files["plan.json"], files["artifact.zip"], payload["verification"])
        else:
            result = accept_l1(files["plan.json"], files["artifact.zip"], payload["verification"],
                               set(payload["object_ids"]))
    else:
        raise ValueError("Unknown service stage")
    write_json(path / "service-response.json", {**job_identity(job, raw), "result":result})


if __name__ == "__main__":
    run(Path(sys.argv[1]))
