"""Vision-only observation records for the registered full Michelson scene."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from .legacy import ARCHIVE, EXPECTED_NAMES, LAYOUT, legacy_manifest
from .sim_io import decode_json, read_bytes


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _scene_hashes(project):
    return {
        "source_layout_sha256": _sha(project / LAYOUT),
        "calibration_sha256": _sha(project / ARCHIVE / "calibration.json"),
    }


def checkpoint_mechanical_outlier(project: Path, source: Path, object_id: str) -> dict:
    """Re-evaluate a bound trajectory in the independent MuJoCo evaluator."""
    command = [str(project/".venv-mujoco/bin/python"), "-m", "simulation.evaluate_vision_michelson",
               "--output-dir", str(source), "--skip-optics"]
    completed = subprocess.run(command, cwd=project/"robot-arm-control", capture_output=True,
                               text=True, timeout=90)
    if completed.returncode != 1:
        raise ValueError("Independent mechanical diagnostic did not complete")
    checked = decode_json(completed.stdout)
    if not all(checked[key]["passed"] is True for key in ("bindings", "recovery_binding", "sequence")):
        raise ValueError("Independent mechanical diagnostic rejected the checkpoint")
    rows = checked["mechanical"]["objects"]
    if set(rows) != EXPECTED_NAMES or object_id not in EXPECTED_NAMES:
        raise ValueError("Independent mechanical diagnostic returned a different inventory")
    row = rows[object_id]
    position, rotation = row["position_error_m"], row["rotation_error_rad"]
    if (row["passed"] is not False or not all(math.isfinite(value) and value >= 0. for value in (position, rotation))
            or (position <= .001 and rotation <= math.radians(1.))):
        raise ValueError("Independent mechanical diagnostic did not confirm this part outside tolerance")
    return {"position_mm":position*1000, "rotation_deg":math.degrees(rotation)}


def _valid_transform(matrix):
    if (not isinstance(matrix, list) or len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
            or any(type(value) not in (int, float) or not math.isfinite(value) for row in matrix for value in row)
            or matrix[-1] != [0, 0, 0, 1]):
        return False
    a, b, c = [row[:3] for row in matrix[:3]]
    cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    return (abs(math.hypot(*a) - 1) < 1e-5 and abs(math.hypot(*b) - 1) < 1e-5
            and abs(sum(x * y for x, y in zip(a, b))) < 1e-5 and math.dist(cross, c) < 1e-5)


def observe_michelson(project: Path, output: Path) -> dict:
    """Capture the seed-0 stock scene at time zero in the MuJoCo interpreter."""
    project = Path(project).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "initial-observations.json"
    images = {camera: output / f"{camera}.png" for camera in ("global_a", "global_b")}
    if any(path.exists() for path in (report_path, *images.values())):
        raise FileExistsError("Michelson observation output already contains capture artifacts")
    scene_hashes = _scene_hashes(project)
    command = [str(project / ".venv-mujoco/bin/python"), "-m", "simulation.michelson_observe", str(output)]
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    worker = subprocess.run(command, cwd=project / "robot-arm-control", env=env,
                            capture_output=True, text=True, timeout=120)
    if worker.returncode != 0:
        raise RuntimeError(f"MICHELSON_RGB_CAPTURE_FAILED: {worker.stderr.strip()[-1400:]}")
    report = _read(report_path)
    if (scene_hashes != _scene_hashes(project)
            or any(report.get(key) != value for key, value in scene_hashes.items())
            or report.get("scene_id") != "michelson_full" or report.get("stage") != "initial"
            or report.get("frame") != "world"):
        raise ValueError("Michelson capture scene binding mismatch")
    if any(report["images"][camera] != {
        "file": path.name, "sha256": _sha(path), "size_px": [3840, 2160]
    } for camera, path in images.items()):
        raise ValueError("Michelson capture image binding mismatch")
    observations = report["observations"]
    missing = report["missing_object_ids"]
    if (set(observations) & set(missing) or set(observations) | set(missing) != EXPECTED_NAMES
            or report["status"] != ("incomplete" if missing else "observed")
            or any(not _valid_transform(row["T_world_object"]) for row in observations.values())):
        raise ValueError("Invalid Michelson image observation coverage or pose")
    return report


def observed_checkpoint(project: Path, source_dir: Path, *, expected_result_sha256: str | None = None) -> dict:
    """Read a full or partial pre-optical visual sweep from a trusted local runner output.

    Uses image estimates and their logged view windows. The source must be a
    trusted local run. Failed runs without evaluator bindings additionally
    require a caller-pinned result digest.
    """
    project = Path(project).resolve()
    source_dir = Path(source_dir).resolve()
    result_path = source_dir / "result.json"
    evidence = result_path.read_bytes()
    result_sha = hashlib.sha256(evidence).hexdigest()
    if expected_result_sha256 is not None and result_sha != expected_result_sha256:
        raise ValueError("Michelson result does not match the trusted run digest")
    result = json.loads(evidence)
    bindings = result.get("verification", {}).get("bindings", {})
    if expected_result_sha256 is None and bindings.get("passed") is not True:
        raise ValueError("A checkpoint without evaluator bindings requires a trusted result digest")
    checked_bindings = bindings if bindings.get("passed") is True else {}
    scene_hashes = _scene_hashes(project)
    if (any(result.get(key) != value or checked_bindings.get(key, value) != value
            for key, value in scene_hashes.items())
            or result.get("public_scene", {}).get("target_frame", {}).get("source_sha256")
            != scene_hashes["source_layout_sha256"]):
        raise ValueError("Michelson checkpoint scene or calibration binding mismatch")
    for filename, key in (("world_model.mjb", "world_model_sha256"), ("trajectory.npz", "trajectory_sha256")):
        actual = _sha(source_dir / filename)
        if result.get(key) != actual or checked_bindings.get(key, actual) != actual:
            raise ValueError(f"Michelson checkpoint artifact binding mismatch: {filename}")

    windows_path = source_dir / "image_observations.json"
    windows_data = windows_path.read_bytes()
    windows_sha = hashlib.sha256(windows_data).hexdigest()
    windows = json.loads(windows_data)
    frames = {
        (window["object"], view["camera_id"], view["view_id"], view["exposure_mid_time_s"])
        for window in windows if "frames" in window
        for view in window["frames"]
    }
    final = result.get("final_visual_checks", {})
    if not final or set(final) - EXPECTED_NAMES:
        raise ValueError("Michelson checkpoint has no recognized final visual sweep")

    observations = {}
    for name, row in final.items():
        pose = row["image_pose"]
        views = pose.get("views", [])
        if (pose.get("valid") is not True or not _valid_transform(pose.get("T_world_object"))
                or len({(view["camera_id"], view["view_id"]) for view in views}) < 3
                or any((name, view["camera_id"], view["view_id"], view["latest_time_s"]) not in frames
                       for view in views)):
            raise ValueError(f"Michelson checkpoint lacks matching calibrated image windows: {name}")
        observations[name] = {
            "T_world_object": pose["T_world_object"],
            "provenance": {
                "source": "calibrated_simulator_rgb_multiview",
                "capture_time_s": row["time_s"], "views": views,
                "reprojection_rms_px": pose["reprojection_rms_px"],
                "result_sha256": result_sha, "image_observations_sha256": windows_sha,
            },
        }
    missing = sorted(EXPECTED_NAMES - observations.keys())
    return {
        "schema_version": "1.0", "scene_id": "michelson_full", "frame": "world",
        "stage": "mechanical_checkpoint", "source": "calibrated_simulator_rgb_multiview",
        **scene_hashes, "status": "incomplete" if missing else "observed",
        "observations": observations, "missing_object_ids": missing,
        "result_sha256": result_sha, "image_observations_sha256": windows_sha,
    }


def read_checkpoint_assessment(project, output, source_dir, expected_files, *, close_pairs=(),
                               near_object=None, trusted_report_sha256=None):
    """Validate a fresh observation-only pass before its findings can authorize a repair."""
    scene = _scene_hashes(project)
    report_raw = read_bytes(output / "result.json", limit=2_000_000)
    report_sha256 = hashlib.sha256(report_raw).hexdigest()
    if trusted_report_sha256 is not None and report_sha256 != trusted_report_sha256:
        raise ValueError("Previously verified re-observation report changed")
    report = decode_json(report_raw.decode("utf-8"))
    state_hash = _sha(source_dir / "integration_state.npz")
    old = _read(source_dir / "result.json")
    code_dir = project / "robot-arm-control/simulation"
    source = {**scene, "source_dir":str(source_dir.resolve()),
              "source_report_sha256":expected_files["result.json"],
              "source_model_sha256":expected_files["world_model.mjb"],
              "source_trajectory_sha256":expected_files["trajectory.npz"],
              "source_integration_state_sha256":state_hash,
              "checkpoint_mode":"complete_mujoco_integration_state"}
    if trusted_report_sha256 is None:
        source.update(observer_source_sha256=_sha(code_dir/"michelson_reobserve.py"),
                      camera_executor_source_sha256=_sha(code_dir/"vision_michelson_assembly.py"))
    audit = report["offline_motion_audit"]
    if (report["status"] not in {"completed", "partial"} or report.get("error")
            or report.get("added_sensor_noise") != {"rgb_std_8bit":0., "optical_std":0.}
            or report["scope"] != {"observation_only":True, "commanded_object_ids":[], "truth_used_for_control":False}
            or any(report["source"].get(k) != value for k, value in source.items())
            or not .060 <= report["source"]["gripper_pad_clearance_m"]
            or old.get("integration_state_sha256") != state_hash
            or any(_sha(source_dir/name) != value for name, value in expected_files.items())
            or not 0. <= audit["max_translation_m"] <= .00001
            or not 0. <= audit["max_rotation_deg"] <= .01
            or report["observation_windows_file"] != "observation_windows.json"):
        raise ValueError("Re-observation source, completion or motion scope rejected")
    window_raw = read_bytes(output / "observation_windows.json", limit=32_000_000)
    windows_hash = hashlib.sha256(window_raw).hexdigest()
    windows = decode_json(window_raw.decode("utf-8"))
    observations = report["observations"]
    if not isinstance(windows, list) or set(observations) != EXPECTED_NAMES or windows_hash != report["observation_windows_sha256"]:
        raise ValueError("Re-observation window or object inventory rejected")
    if report.get("near_object") != near_object or (near_object is not None and (
        near_object not in EXPECTED_NAMES
        or {name for name, row in observations.items() if row.get("status") == "observed"} != {near_object}
        or any(window.get("object") not in (None, near_object) for window in windows)
    )):
        raise ValueError("Re-observation range or target scope rejected")
    close = {}
    if close_pairs:
        close = {f"{target}:{other}":{"wrist_views":0, "motion_authorized":False}
                 for target,other in close_pairs}
        if set(close) != set(report.get("paired_exposures", {})):
            raise ValueError("Unexpected paired observation scope")
        exposures = set()
        for window in windows:
            for row in window.get("paired_marker_observations", ()):
                target,other = row["object_ids"]
                key = f"{target}:{other}"
                if key not in close or window.get("object") != target:
                    raise ValueError("Unrequested paired observation")
                crop = row.get("rgb_crop")
                if not crop:
                    continue
                frame = row["frame_id"]
                expected = f"pair-{target}-{other}-{frame}.png"
                if (type(frame) is not int or frame < 0 or row["camera_id"] != "wrist_rgbd"
                        or not window.get("view_id", "").startswith("pair-close-")
                        or crop["file"] != expected or (key,frame) in exposures
                        or set(row["corners_by_object"]) != {target,other}
                        or not all(row["corners_by_object"].values())):
                    raise ValueError("Paired RGB exposure identity rejected")
                pixels = read_bytes(output/expected, limit=4_000_000)
                if not pixels.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(pixels).hexdigest() != crop["sha256"]:
                    raise ValueError("Paired RGB crop changed")
                exposures.add((key,frame))
                close[key]["wrist_views"] += 1
    manifest = {row["id"]:row for row in legacy_manifest(project)["targets"]}
    poses, findings, excluded = {}, {}, {}
    for name, row in observations.items():
        if row.get("status") == "unobserved":
            excluded[name] = row.get("reason", "unobserved")
            continue
        start, end = row["window_start_index"], row["window_end_index"]
        if (row.get("status") != "observed" or row.get("valid") is not True
                or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(windows)):
            raise ValueError(f"Re-observation window range rejected for {name}")
        selected = windows[start:end]
        window_bytes = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8")
        pose = row["image_pose"]
        transform = pose["T_world_object"]
        views = pose.get("views", [])
        frames = {(v["camera_id"], v["view_id"], v["exposure_mid_time_s"])
                  for window in selected if window.get("object") == name for v in window.get("frames", [])}
        if (hashlib.sha256(window_bytes).hexdigest() != row["windows_sha256"] or pose.get("valid") is not True
                or not _valid_transform(transform) or len({(v["camera_id"], v["view_id"]) for v in views}) < 3
                or any((v["camera_id"], v["view_id"], v["latest_time_s"]) not in frames for v in views)):
            raise ValueError(f"Re-observation calibrated pose rejected for {name}")
        if len({v["camera_id"] for v in views}) < 2:
            excluded[name] = "insufficient_physical_cameras_for_repair"
            continue
        xyz = [r[3] for r in transform[:3]]
        position = math.dist(xyz, manifest[name]["world_position_m"])
        yaw = math.radians(manifest[name]["world_yaw_deg"])
        trace = (math.cos(yaw)*(transform[0][0]+transform[1][1])
                 + math.sin(yaw)*(transform[1][0]-transform[0][1])+transform[2][2])
        angle = math.degrees(math.acos(max(-1., min(1., (trace-1.)/2))))
        passed = position <= .001 and angle <= 1.
        if (not math.isclose(position, row["position_error_m"], abs_tol=1e-8)
                or not math.isclose(angle, row["rotation_error_deg"], abs_tol=1e-5)
                or row["within_1mm_1deg"] is not passed):
            raise ValueError(f"Re-observation residual disagrees with the fixed design for {name}")
        poses[name] = {"T_world_object":transform, "provenance":{"capture_time_s":row["capture_time_s"],
                        "views":views, "reprojection_rms_px":pose["reprojection_rms_px"],
                        "windows_sha256":row["windows_sha256"]}}
        findings[name] = {"position_m":position, "rotation_deg":angle, "passed":passed}
    return {"schema_version":"1.0", "scene_id":"michelson_full", "frame":"world", **scene,
            "source":"calibrated_simulator_rgb_multiview", "observations":poses,
            "missing_object_ids":sorted(EXPECTED_NAMES-poses.keys()), "findings":findings, "excluded":excluded,
            "result_sha256":report_sha256, "image_observations_sha256":windows_hash,
            "close_pairs":close, **({"near_object":near_object} if near_object else {})}
