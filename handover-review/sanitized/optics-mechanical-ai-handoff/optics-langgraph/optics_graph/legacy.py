"""Adapter for the fixed, historical 15-component Michelson simulation.

This adapter never accepts a Phase 1 plan as input. The legacy runner owns its
fixed design and the result must be bound to that exact design.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import re
import signal
import subprocess
import time

PROFILE = 'legacy_vision_michelson_15'
LAYOUT = Path('Private_Project_Materials/Research/Closed_Model_Full_Experiment_20261027/answer_first.json')
ARCHIVE = Path('Private_Project_Materials/Research/Vision_Michelson_Assembly')
EXPECTED_NAMES = {'LASER', 'ISO', 'SH0', 'L25', 'L50', 'BS50', 'S_REF', 'M_FIXED', 'S_MEAS', 'M_PZT', 'BS10', 'ND2', 'CAM', 'ND1', 'PD'}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def legacy_manifest(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    path = root / LAYOUT
    document = _read(path)
    components = document['components']
    if len(components) != 15 or {c['id'] for c in components} != EXPECTED_NAMES:
        raise ValueError('Legacy design must contain the original 15 components exactly once')
    targets = []
    for component in components:
        position = [float(component[k]) / 1000 for k in ('x_mm', 'y_mm', 'z_mm')]
        angle = float(component.get('normal_deg', component.get('axis_deg', 0.0)))
        if not all(math.isfinite(x) for x in (*position, angle)):
            raise ValueError('Non-finite legacy target')
        targets.append({'id': component['id'], 'layout_position_m': position,
                        'world_position_m': [p + offset for p, offset in zip(position, (-.15, -.10, .70))],
                        'world_yaw_deg': angle})
    historic = _read(root / ARCHIVE / 'result.json')
    names = historic.get('source_sha256', {})
    sources = {name: _sha(root / 'robot-arm-control/simulation' / name) for name in names
               if Path(name).name == name and (root / 'robot-arm-control/simulation' / name).is_file()}
    if len(sources) != len(names) or not sources:
        raise ValueError('Legacy source files missing or invalid')
    return {'profile': PROFILE, 'source_path': str(path), 'source_sha256': _sha(path),
            'targets': targets, 'target_frame': 'world', 'layout_to_world_translation_m': [-.15, -.10, .70],
            'code_sha256': sources, 'calibration_sha256': _sha(root / ARCHIVE / 'calibration.json'), 'evidence_refs': [str(root / ARCHIVE / f) for f in
                ('README.md', 'result.json', 'independent_evaluation.json')]}


def read_legacy_evidence(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    path = root / ARCHIVE / 'result.json'
    report = _read(path)
    accepted = (report.get('status') == 'completed' and report.get('success') is True
                and report.get('verification', {}).get('accepted') is True)
    return {'profile': PROFILE, 'mode': 'historical_evidence', 'historical': True,
            'status': 'historical_accepted' if accepted else 'historical_not_accepted',
            'success': accepted, 'seed': report.get('seed'),
            'source_sha256': report.get('source_layout_sha256'),
            'simulation_s': report.get('simulation_s'), 'wall_s': report.get('wall_s'),
            'evidence_refs': [str(path), str(root / ARCHIVE / 'independent_evaluation.json'), str(root / ARCHIVE / 'README.md')],
            'scope': 'Previously recorded simulation; not a new run or a hardware result.'}


def _base(mode: str) -> dict:
    return {'profile': PROFILE, 'mode': mode, 'historical': False, 'status': 'failed',
            'success': False, 'evidence_refs': []}


def precision_progress(output: Path) -> dict:
    path = output / "result.json"
    try:
        progress = _precision_progress(path, path.stat().st_mtime_ns)
    except (OSError, ValueError):
        return {}
    if progress["phase"] != "assembly":
        return progress
    for log in (output.parent / "resume-worker.log", output / "run.log"):
        try:
            with log.open("rb") as stream:
                stream.seek(max(0, os.fstat(stream.fileno()).st_size - 16384))
                text = stream.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        rows = re.findall(r"^\s*(\d+\.\d+)s  (\w+): ([a-z ]+)$", text, re.MULTILINE)
        if rows:
            seconds, object_id, phase = rows[-1]
            rehearsal = ((progress["supported_turn"] or progress["corridor_lift"]) and
                         "MuJoCo rehearsal passed" not in text)
            return {**progress, "motion_phase":{"object_id":object_id, "phase":phase,
                    "simulation_s":float(seconds), "rehearsal":rehearsal}}
    return progress


@lru_cache(maxsize=32)
def _precision_progress(path: Path, mtime_ns: int) -> dict:
    report = _read(path)
    if report.get("camera_pickup"):
        placements = [{"name":name, "passed":item.get("success",False),
                       "attempts":[{"position_mm":1000*item["placement_xy_error_m"],
                                    "rotation_deg":math.degrees(item["direction_error_rad"])}]}
                      for name,item in report.get("objects",{}).items()]
        return {"phase":"completed" if report.get("success") else "failed",
                "placements":placements,"completed":sum(row["passed"] for row in placements),
                "required":4,"failure":report.get("controller_failure"),"measurement":"mechanical"}
    objects = report.get("objects", {})
    placements = []
    for name, item in objects.items():
        attempts = [{"position_mm":1000 * a["visual_position_residual_m"],
                     "rotation_deg":math.degrees(a["visual_rotation_residual_rad"])}
                    for a in item.get("attempts", []) if "visual_position_residual_m" in a]
        placements.append({"name":name, "attempts":attempts, "passed":item.get("visual_quality_passed", False)})
    phase = report.get("status")
    if phase == "running":
        phase = "final_camera_check" if "final_visual_checks" in report else "assembly"
    transfer_plans = [
        {"label":event["label"], "rejected":len(event.get("rejected_candidates", [])),
         "accepted":event.get("accepted", event["event"] == "held_transfer_plan"),
         "motion":"parking" if event["event"] == "post_release_park_plan" else "transfer",
         "waypoints":len(event.get("waypoints_rad", []))}
        for event in report.get("robot_image_events", [])
        if event.get("event") in ("held_transfer_plan", "held_transfer_plan_failed", "post_release_park_plan")
    ]
    contact = next((event.get("contacts", [])[0]
                    for event in reversed(report.get("robot_image_events", []))
                    if event.get("event") in {"transport_contact_rejected", "supported_turn_lift_qualification"}
                    and event.get("contacts")), None)
    final_checks = report.get("final_visual_checks")
    completed = (sum(item.get("passed") is True for item in final_checks.values())
                 if isinstance(final_checks, dict) else sum(row["passed"] for row in placements))
    return {"phase":phase, "placements":placements, "completed":completed,
            "required":len(EXPECTED_NAMES), "failure":report.get("failure"),
            "measurement":"final_visual" if isinstance(final_checks, dict) else "placement",
            "optical_feedback":report.get("optical_feedback", {}).get("status"),
            "noise":report.get("added_sensor_noise"), "transfer_plans":transfer_plans,
            "protective_contact":contact,
            "supported_turn":"replan_supported_turn_lift" in report.get("recovery_actions", []),
            "corridor_lift":"replan_corridor_lift" in report.get("recovery_actions", [])}


def render_precision_replay(project_root: Path, output: Path, *, experiment="michelson_full") -> str:
    """Render the accepted run's saved MuJoCo states without running control again."""
    if experiment == "mach_zehnder":
        args = [str(project_root / "simulation/record_mzi.py"), "--contact-run", str(output)]
    else:
        args = [str(project_root / "simulation/record_vision_michelson.py"), "--output-dir", str(output),
                "--speed", "45", "--fps", "12", "--view", "overview"]
    subprocess.run([str(project_root / ".venv-mujoco/bin/python"), *args],
                   cwd=project_root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   timeout=600, check=True)
    return _sha(output / "assembly_overview.mp4")


def run_mzi_contact(project_root: Path, output: Path, source_sha256: str) -> dict:
    layout = project_root/"Private_Project_Materials/Research/MZI_Autonomous/01_Optical_Layouts/layout.json"
    if _sha(layout) != source_sha256:
        raise ValueError("MZI layout differs from the registered design")
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"}
    with (output/"run.log").open("wb") as log:
        mechanical = subprocess.run([str(project_root/".venv-mujoco/bin/python"), "-m", "simulation.mzi_assembly",
                                    "--layout", str(layout), "--output", str(output), "--camera-pickup"],
                                    cwd=project_root/"robot-arm-control", env=env, timeout=600,
                                    stdout=log, stderr=subprocess.STDOUT)
        if not (output/"result.json").is_file():
            return {"success":False,"failure":"RGB localization or mechanical execution did not produce a completion record"}
        optical = subprocess.run([str(project_root/".venv-lightpipes/bin/python"), "-m", "optics_graph.mzi_optical_check",
                                  "--contact", str(output)], cwd=project_root/"optics-langgraph",
                                 env=env, timeout=180, stdout=log, stderr=subprocess.STDOUT)
    run = _read(output/"result.json")
    report_path = output/"mzi-contact-optical-report.json"
    wave = _read(report_path) if report_path.is_file() else {}
    audit = wave.get("provenance") or {}
    bindings = bool(audit.get("pass") and (audit.get("pickup_rgb_audit") or {}).get("pass")
                    and audit.get("source_target_sha256") == source_sha256
                    and audit.get("evidence_sha256") == _sha(output/"result.json"))
    accepted = bool(bindings and mechanical.returncode == optical.returncode == 0 and run.get("success")
                    and wave.get("passed"))
    return {"success":accepted,"bindings_verified":bindings, "added_sensor_noise":{"rgb_std_8bit":0.,"optical_std":0.},
            "source_sha256":source_sha256,"trajectory_sha256":run.get("trajectory_sha256"),
            "verification":{"accepted":accepted,"mechanical":{"passed":bool(run.get("success")), "objects":run["objects"]},
                            "optical":{**wave, "passed":wave.get("passed",False)},
                            "failure_reasons":[name for name,passed in wave.get("checks",{}).items() if not passed]},
            "failure":None if accepted else run.get("controller_failure") or wave.get("failure") or "MZI mechanical and optical acceptance did not pass"}


def run_legacy(project_root: Path, output_dir: Path, timeout_s: int = 3600, *, added_noise=True,
               recovery_actions=(), guard_parts=(), resume_dir=None, repair_part=None, repair_stage=None,
               repair_observation=None) -> dict:
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    if timeout_s <= 0:
        raise ValueError('timeout_s must be positive')
    # Only a checkpoint repair may read an existing result. It must never repeat its motion.
    adopted = output.exists() and resume_dir is not None
    if output.exists() and not adopted:
        raise FileExistsError(output)
    if resume_dir is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    result = _base('simulation_run')
    log = output / 'run.log'
    worker_log = log if resume_dir is None else output.parent / 'resume-worker.log'
    result['evidence_refs'] = [str(log)]
    try:
        before = legacy_manifest(root)
        command = [str(root / '.venv-mujoco/bin/python'), '-u', '-m', 'simulation.vision_michelson_assembly',
                   '--output', str(output), '--seed', '0']
        if not added_noise:
            command.append('--no-added-noise')
        if resume_dir is not None:
            if bool(repair_part) == bool(repair_stage):
                raise ValueError("Checkpoint continuation requires exactly one repair scope")
            command.extend(('--resume-dir', str(resume_dir)))
            command.extend(('--repair-part', repair_part) if repair_part else ('--repair-stage', repair_stage))
        for action in recovery_actions:
            command.extend(('--recovery-action', action))
        for part in guard_parts:
            command.extend(('--guard-part', part))
        observation_hash = _sha(Path(repair_observation)) if repair_observation is not None else None
        if repair_observation is not None:
            command.extend(('--repair-observation', str(repair_observation)))
        env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        if adopted:
            deadline = time.monotonic() + min(timeout_s, 1800)
            while True:
                try:
                    report = _read(output / "result.json")
                except (OSError, ValueError):
                    report = {}
                if (report.get("status") in {"completed", "completed_with_quality_failure", "failed"}
                        and report.get("trajectory_sha256") and report.get("world_model_sha256")):
                    break
                if not worker_log.is_file() or time.monotonic() >= deadline:
                    break
                time.sleep(2)
            evidence = report.get("recovery") or {}
            target_ids = [repair_part] if repair_part else []
            if (report.get("status") not in {"completed", "completed_with_quality_failure", "failed"}
                    or not report.get("trajectory_sha256") or not report.get("world_model_sha256")
                    or evidence.get("retained_scene") is not True
                    or evidence.get("source_report_sha256") != _sha(Path(resume_dir) / "result.json")
                    or evidence.get("target_object_ids") != target_ids):
                raise ValueError("Previous repair worker did not publish a complete, bound result; will not repeat motion")
            exit_code = None
            result["adopted_completed_worker"] = True
        else:
            with worker_log.open('x' if resume_dir else 'w', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=root / 'robot-arm-control', env=env,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    exit_code = process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    result.update(status='timed_out', failure='Timed out; process stopped, completion not established.')
                    return result
        if resume_dir is not None and worker_log.is_file():
            output.mkdir(parents=True, exist_ok=True)
            worker_log.replace(log)
        result['exit_code'] = exit_code
        path = output / 'result.json'
        if not path.is_file():
            result['failure'] = 'Process did not produce a fresh result.json.'
            return result
        result['evidence_refs'].append(str(path))
        report = _read(path)
        after = legacy_manifest(root)
        source_hash = before['source_sha256']
        bindings_ok = (after['source_sha256'] == source_hash
                       and before['code_sha256'] == after['code_sha256']
                       and before['calibration_sha256'] == after['calibration_sha256']
                       and report.get('calibration_sha256') == before['calibration_sha256']
                       and report.get('source_sha256') == before['code_sha256']
                       and report.get('recovery_actions', []) == list(recovery_actions)
                       and report.get('guard_parts', []) == list(guard_parts)
                       and report.get('source_layout_sha256') == source_hash
                       and report.get('public_scene', {}).get('target_frame', {}).get('source_sha256') == source_hash)
        if observation_hash is not None:
            bindings_ok = (bindings_ok and report.get('repair_observation_sha256') == observation_hash
                           and _sha(Path(repair_observation)) == observation_hash)
        for filename, key in [('trajectory.npz', 'trajectory_sha256'), ('world_model.mjb', 'world_model_sha256')]:
            artifact = output / filename
            bindings_ok = bindings_ok and artifact.is_file() and _sha(artifact) == report.get(key)
            if artifact.is_file():
                result['evidence_refs'].append(str(artifact))
        for filename in ('independent_evaluation.json', 'optical_feedback.json', 'receiving_alignment.json'):
            if (output / filename).is_file():
                result['evidence_refs'].append(str(output / filename))
        accepted = ((exit_code == 0 or adopted) and bindings_ok and report.get('status') == 'completed'
                    and report.get('success') is True and report.get('verification', {}).get('accepted') is True)
        result.update(status='completed' if accepted else 'failed', success=accepted,
                      source_sha256=source_hash, bindings_verified=bool(bindings_ok),
                      reported_status=report.get('status'), verification=report.get('verification'),
                      recovery_actions=list(recovery_actions),
                      added_sensor_noise=report.get('added_sensor_noise'),
                      simulation_s=report.get('simulation_s'), wall_s=report.get('wall_s'),
                      trajectory_sha256=report.get('trajectory_sha256'),
                      recovery=report.get('recovery'),
                      failure=None if accepted else report.get('failure') or 'Simulation acceptance failed or the evidence source check did not pass.')
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result['failure'] = f'{type(exc).__name__}: {exc}'
    return result


def probe_legacy(project_root: Path) -> dict:
    """Check imports and render one initial legacy camera frame in a child process."""
    root = Path(project_root).resolve()
    result = _base('dependency_probe')
    commands = [([str(root / '.venv-mujoco/bin/python'), '-c',
                  'import mujoco, mujoco_menagerie, cv2, numpy, scipy; print("MuJoCo dependencies import OK")'], root / 'robot-arm-control'),
                ([str(root / '.venv-lightpipes/bin/python'), '-c',
                  'import LightPipes; print("LightPipes import OK")'], root / 'robot-arm-control'),
                ([str(root / '.venv-mujoco/bin/python'), '-c',
                  'import mujoco; from simulation.vision_michelson_scene import make_scene; '
                  'm,d,_=make_scene(0); mujoco.mj_forward(m,d); '
                  'r=mujoco.Renderer(m,width=640,height=360); r.update_scene(d,camera="global_a"); '
                  'im=r.render(); print("initial global_a RGB",im.shape); r.close()'], root / 'robot-arm-control')]
    checks = []
    try:
        for command, cwd in commands:
            run = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
            checks.append({'interpreter': command[0], 'exit_code': run.returncode,
                           'stdout': run.stdout.strip(), 'stderr': run.stderr.strip()[-1000:]})
        passed = all(check['exit_code'] == 0 for check in checks)
        result.update(status='passed' if passed else 'failed', success=passed, checks=checks,
                      scope='Imports and initial global_a camera frame only; no localization, control, optics or hardware tested.')
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update(checks=checks, failure=f'{type(exc).__name__}: {exc}',
                      scope='Imports and initial global_a camera frame only; no localization, control, optics or hardware tested.')
    return result
