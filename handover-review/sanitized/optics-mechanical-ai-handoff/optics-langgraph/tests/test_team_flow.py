import json

import pytest

from optics_graph.contracts import load_scene
from optics_graph.runtime import RunInput, Runtime
from optics_graph.settings import PROJECT
from optics_graph.team_artifacts import digest


pytestmark = pytest.mark.skipif(not (PROJECT / ".venv-mujoco/bin/python").exists(),
                                reason="MuJoCo environment unavailable")


def prob0():
    scene = load_scene()
    names = sorted(name for name, obj in scene["objects"].items() if obj["movable"])
    return {"schema_version":"local-prob/v0","calibration_id":"phase1-optical-table-v1",
            "scene_id":scene["scene_revision"],"frame_id":"optical_table","source":"provided","frames":[],
            "observations":{name:{"position_m":[x,-.12,0.],"yaw_rad":0.}
                            for name,x in zip(names,(.13,.31,.49))}}


@pytest.mark.parametrize("fault, expected", [
    ("prompt", ("validator","prompt")),
    ("act", ("validator","act")),
    ("controller", ("controller","controller")),
    ("sim", ("sim","sim")),
    ("output", ("output","sim")),
])
def test_flowchart_feedback_and_same_file(tmp_path, fault, expected):
    runtime = Runtime(tmp_path)
    state = runtime.execute(runtime.create(RunInput(mode="github_local",team_entry="nl",
                                                    prob0=prob0(),team_fault=fault)))
    assert state["status"] == "team_l1_passed"
    assert [(r["from"],r["to"]) for r in state["retry_log"]] == [expected]
    root = runtime.run_dir(state["run_id"])
    raw = (root / state["plan_path"]).read_bytes()
    sim_raw = (root / state["artifact_path"]).with_name("plan.json").read_bytes()
    assert raw == sim_raw
    assert state["result"]["plan_hash"] == state["verification"]["plan_hash"] == digest(raw)
    assert state["result"]["motions_sent"] == 0


def test_execution_rejection_is_terminal(tmp_path):
    runtime = Runtime(tmp_path)
    state = runtime.execute(runtime.create(RunInput(mode="github_local",team_entry="nl",
                                                    prob0=prob0(),team_fault="execution")))
    assert state["status"] == "failed"
    assert state["error"]["code"] == "EXECUTION_REJECTED"
    assert state["attempts"]["execution"] == 1
    assert not state.get("retry_log")
    assert not (runtime.run_dir(state["run_id"]) / "team/execution-01/execution_receipt.json").exists()


def test_restarted_approval_rechecks_original_bytes(tmp_path):
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(mode="github_local",team_entry="nl",prob0=prob0(),require_review=True))
    state = runtime.execute(run_id)
    assert state["status"] == "awaiting_review"
    assert "execution" not in state["attempts"]
    resumed = Runtime(tmp_path).execute(run_id,resume={"approved":True,"plan_sha256":state["plan_sha256"]})
    assert resumed["status"] == "team_l1_passed"
    assert resumed["approval"]["plan_hash"] == resumed["result"]["plan_hash"]


def test_mode_sim_never_reaches_execution(tmp_path):
    runtime = Runtime(tmp_path)
    state = runtime.execute(runtime.create(RunInput(mode="github_sim",team_entry="nl",prob0=prob0())))
    assert state["status"] == "team_sim_passed"
    assert "execution" not in state["attempts"]
