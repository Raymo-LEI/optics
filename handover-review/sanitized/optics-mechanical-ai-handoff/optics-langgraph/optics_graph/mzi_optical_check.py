"""Evaluate the registered MZI on post-control MuJoCo poses with LightPipes."""
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from .sim_io import read_json, write_json
from .settings import PROJECT

SOLVER = PROJECT / "Private_Project_Materials/Research/MZI_Autonomous/01_Optical_Layouts/validate_mzi.py"


def load_solver():
    spec = spec_from_file_location("mzi_lightpipes_validator", SOLVER)
    solver = module_from_spec(spec)
    spec.loader.exec_module(solver)
    return solver


def evaluate(workdir):
    input_path = workdir / "mzi-optical-input.json"
    request = read_json(input_path)
    source = SOLVER.with_name("layout.json").read_bytes()
    if sha256(source).hexdigest() != request["source_sha256"]:
        raise ValueError("Registered Mach–Zehnder design source changed")
    document = json.loads(source)
    objects = request["objects"]
    actual = request["actual_poses"]
    if set(actual) != set(objects) or set(actual) != {c["id"] for c in document["components"]}:
        raise ValueError("MuJoCo poses do not cover the optical inventory")

    for component in document["components"]:
        name = component["id"]
        reference = objects[name]["reference_pose"]
        measured = actual[name]
        # Unshift the workbench frame, then rotate the nominal optic-to-body
        # lever arm using the measured mount orientation.
        nominal = Rotation.from_quat(reference["orientation_xyzw"])
        measured_rotation = Rotation.from_quat(measured["orientation_xyzw"])
        correction = measured_rotation * nominal.inv()
        displacement = np.asarray(measured["position_m"]) - reference["position_m"]
        optical_height = component["optical_center_m"][2] - objects[name]["height_m"] / 2
        lever = np.array([0., 0., optical_height])
        component["optical_center_m"] = (
            np.asarray(component["optical_center_m"]) + displacement + correction.apply(lever) - lever
        ).tolist()
        component["direction_xyz"] = correction.apply(component["direction_xyz"]).tolist()

    solver = load_solver()
    write_json(workdir / "mzi-actual-optics.json", document)
    report = {"source_sha256":request["source_sha256"], "plan_sha256":request["plan_sha256"],
              "simulation_result_sha256":request["simulation_result_sha256"],
              "input_sha256":sha256(input_path.read_bytes()).hexdigest(),
              "solver_sha256":sha256(SOLVER.read_bytes()).hexdigest(),
              "acceptance_sha256":sha256(SOLVER.with_name("acceptance.json").read_bytes()).hexdigest(),
              "actual_sha256":sha256((workdir/"mzi-actual-optics.json").read_bytes()).hexdigest(),
              "model":"LightPipes", "scope":"post_control_mujoco_poses", "passed":False}
    try:
        result = solver.evaluate(document, include_scan=False)
        report.update(passed=bool(result["pass"]), checks=result["checks"],
                      wave=result["wave"], geometry=result["geometry"])
    except ValueError as exc:
        report["failure"] = str(exc)
    write_json(workdir / "mzi-optical-report.json", report)


def evaluate_contact(workdir):
    """Independently reload the contact run's measured optical pose and camera records."""
    actual = workdir / "actual_layout.json"
    audit = subprocess.run([str(PROJECT/".venv-mujoco/bin/python"), str(SOLVER.with_name("check_actual_state.py")),
                            "--actual", str(actual), "--metadata", str(actual.with_suffix(".meta.json"))],
                           capture_output=True, text=True, timeout=60)
    proof = json.loads(audit.stdout)
    report = {"model":"LightPipes","scope":"independent_fk_and_pickup_rgb","provenance":proof,"passed":False,
              "solver_sha256":sha256(SOLVER.read_bytes()).hexdigest(),
              "acceptance_sha256":sha256(SOLVER.with_name("acceptance.json").read_bytes()).hexdigest()}
    if audit.returncode != 0 or not proof.get("pass") or not (proof.get("pickup_rgb_audit") or {}).get("pass"):
        report["failure"] = proof.get("error", "RGB or saved-state verification did not pass")
    else:
        solver = load_solver()
        try:
            evaluated = solver.evaluate(solver.parse_layout(actual, actual=True), include_scan=False)
            report.update(passed=bool(evaluated["pass"] and proof["mechanical_success"]),
                          checks=evaluated["checks"],wave=evaluated["wave"],geometry=evaluated["geometry"])
        except ValueError as exc:
            report["failure"] = str(exc)
    write_json(workdir / "mzi-contact-optical-report.json", report)
    return report["passed"]


if __name__ == "__main__":
    if sys.argv[1] == "--contact":
        raise SystemExit(0 if evaluate_contact(Path(sys.argv[2])) else 1)
    evaluate(Path(sys.argv[1]))
