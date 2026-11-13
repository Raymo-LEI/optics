import json
from pathlib import Path

import pytest

from optics_graph.local_sim import run_local_sim
from optics_graph.runtime import RunInput, Runtime
from optics_graph.settings import PROJECT
from optics_graph.sim_io import read_json
from optics_graph.gateway import CandidateFormatError, ModelGateway


@pytest.mark.parametrize("payload", [b'{"v":1,"v":2}', b'{"v":1e999}', b'\xff', b" " * 256_001])
def test_sim_json_rejects_invalid_files(tmp_path, payload):
    path = tmp_path / "job.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_json(path)


@pytest.mark.skipif(not (PROJECT / ".venv-mujoco/bin/python").exists(), reason="MuJoCo environment unavailable")
@pytest.mark.parametrize("scenario, outcome", [
    ("nominal", "local_sim_passed"),
    ("transient", "local_sim_passed"),
    ("persistent", "failed"),
])
def test_local_simulation_and_recovery(tmp_path, scenario, outcome):
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(mode="local_sim", sim_scenario=scenario))
    state = runtime.execute(run_id)
    assert state["status"] == outcome
    evidence = state["result"]
    if scenario == "persistent":
        assert evidence["error"]["code"] == "PLACEMENT_OUT_OF_TOLERANCE"
        assert len(evidence["steps"]) == 5
        assert len(evidence["recoveries"]) == 1
        return
    assert len(evidence["steps"]) == 14
    assert evidence["planar_optical_recheck"]["passed"]
    assert len(evidence["verifications"]) == 3
    if scenario == "transient":
        assert [row["layer"] for row in evidence["recoveries"]] == ["observation", "placement"]
    else:
        assert not evidence["recoveries"]
        path = runtime.run_dir(run_id) / "local-simulation"
        assert run_local_sim(state, path)["status"] == "succeeded"
        evidence_file = path / "evidence.json"
        tampered = json.loads(evidence_file.read_text())
        tampered["job_sha256"] = "0" * 64
        evidence_file.write_text(json.dumps(tampered))
        assert run_local_sim(state, path)["error"]["code"] == "SIM_EVIDENCE_INVALID"


def test_model_candidate_format_error_uses_existing_retry_budget(tmp_path):
    class FaultyGateway(ModelGateway):
        def __init__(self, provider, run_dir):
            super().__init__("replay", run_dir)

        def layout(self, request, scene, intent, feedback, attempt):
            if attempt == 1:
                raise CandidateFormatError("broken layout JSON")
            assert feedback[0]["code"] == "MODEL_INVALID_JSON"
            return super().layout(request, scene, intent, feedback, attempt)

        def plan(self, layout, validation, feedback, attempt):
            if attempt == 1:
                raise CandidateFormatError("broken plan JSON")
            assert feedback[0]["code"] == "MODEL_INVALID_JSON"
            return super().plan(layout, validation, feedback, attempt)

    runtime = Runtime(tmp_path, gateway_factory=FaultyGateway)
    state = runtime.execute(runtime.create(RunInput(mode="design", provider="openai")))
    assert state["status"] == "design_ready"
    assert state["layout_attempts"] == state["plan_attempts"] == 2


def test_invalid_persisted_input_stops_before_simulator(tmp_path):
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(mode="local_sim"))
    path = runtime.run_dir(run_id)
    (path / "input.json").write_text('{"mode":"local_sim","mode":"design"}')
    result = runtime.execute(run_id)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "RUN_INPUT_INVALID"
    assert not (path / "local-simulation").exists()


@pytest.mark.skipif(not (PROJECT / ".venv-mujoco/bin/python").exists(), reason="MuJoCo environment unavailable")
@pytest.mark.parametrize("sign, expected", [(-1, "local_sim_passed"), (1, "failed")])
def test_system_diagnoses_and_revalidates_persistent_offset(tmp_path, sign, expected):
    class DiagnosticGateway(ModelGateway):
        def __init__(self, provider, run_dir):
            super().__init__("replay", run_dir)

        def diagnose_simulation(self, evidence, attempt):
            assert attempt == 1
            assert evidence["allowed_routes"] == ["calibration", "stop"]
            measurements = evidence["placement_errors"][-2:]
            return {
                "route": "calibration",
                "diagnosis": "Both observations show a bias in the same direction",
                "delta_xy_m": [sign * sum(m["signed_xy_m"][i] for m in measurements) / 2 for i in range(2)],
            }

    runtime = Runtime(tmp_path, gateway_factory=DiagnosticGateway)
    run_id = runtime.create(RunInput(mode="local_sim", sim_scenario="persistent", provider="openai"))
    state = runtime.execute(run_id)
    assert state["status"] == expected
    assert state["sim_history"][0]["error"]["code"] == "PLACEMENT_OUT_OF_TOLERANCE"
    if sign == -1:
        assert state["sim_attempts"] == 2
        assert len(state["repair_history"]) == 1
        assert state["result"]["planar_optical_recheck"]["passed"]
        assert state["result"]["layout_sha256"] == state["validation"]["layout_sha256"]
        assert (runtime.run_dir(run_id) / "local-simulation-02/evidence.json").is_file()
    else:
        assert state["sim_attempts"] == 1
        assert "SIM_REPAIR_INVALID" in state["error"]["message"]


@pytest.mark.skipif(not (PROJECT / ".venv-mujoco/bin/python").exists(), reason="MuJoCo environment unavailable")
@pytest.mark.parametrize("route,code", [("layout","SIM_OPTICAL_RECHECK_FAILED"), ("plan","UNEXPECTED_CONTACT")])
def test_diagnosis_can_send_validated_revision_back_to_simulation(tmp_path, monkeypatch, route, code):
    from optics_graph import graph

    class DiagnosticGateway(ModelGateway):
        def __init__(self, provider, run_dir):
            super().__init__("replay", run_dir)

        def diagnose_simulation(self, evidence, attempt):
            assert route in evidence["allowed_routes"]
            return {"route": route, "diagnosis": "Return to the corresponding node", "delta_xy_m": None}

        def layout(self, request, scene, intent, feedback, attempt):
            result = super().layout(request, scene, intent, feedback, attempt)
            if route == "layout" and attempt == 2:
                assert feedback[0]["code"] == code
                mirror = next(c for c in result["components"] if c["role"] == "mirror_arm_x")
                mirror["target_pose"]["position_m"][0] += .005
            return result

        def plan(self, layout, validation, feedback, attempt):
            result = super().plan(layout, validation, feedback, attempt)
            if route == "plan" and attempt == 2:
                assert feedback[0]["code"] == code
                actions = result["actions"]
                result["actions"] = actions[:1] + actions[5:9] + actions[1:5] + actions[9:]
                previous = []
                for action in result["actions"]:
                    action["depends_on"] = previous
                    previous = [action["action_id"]]
            return result

    def simulator(state, directory):
        if not state.get("sim_attempts"):
            return {"status": "failed", "error": {"code": code, "action_id": "a04"}}
        return run_local_sim(state, directory)

    monkeypatch.setattr(graph, "run_local_sim", simulator)
    runtime = Runtime(tmp_path, gateway_factory=DiagnosticGateway)
    state = runtime.execute(runtime.create(RunInput(mode="local_sim", provider="openai")))
    assert state["status"] == "local_sim_passed"
    assert state["sim_attempts"] == 2
    assert state["repair_history"][0]["route"] == route
