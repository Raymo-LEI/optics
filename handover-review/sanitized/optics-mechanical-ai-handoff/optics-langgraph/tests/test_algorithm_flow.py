"""Exercise the canonical graph using only the local synthetic table."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from math import degrees
from threading import Event
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from optics_graph import team_flow
from optics_graph.algorithm_sim import AlgorithmServices
from optics_graph.runtime import RunInput, Runtime
from optics_graph.server import create_app
from optics_graph.sim_io import read_bytes, read_json
from optics_graph.team_artifacts import decode_json, digest, encode


MICHELSON = "Assemble a 15-component complete Michelson interferometer."


def run_input(request=MICHELSON, *, options=None, **kwargs):
    return RunInput(request=request, profile="auto", mode="github_local",
                    simulation_backend="algorithm_v1", provider="replay",
                    algorithm_options={"delay_s": 0.0, **(options or {})}, **kwargs)


def run(tmp_path, **kwargs):
    runtime = Runtime(tmp_path)
    run_id = runtime.create(run_input(**kwargs))
    return runtime.execute(run_id), runtime.run_dir(run_id)


def routes(state):
    return [(item["from"], item["to"]) for item in state.get("retry_log", [])]


def test_full_michelson_uses_canonical_graph_and_same_handoff_bytes(tmp_path):
    with Runtime(tmp_path).graph("github_local", algorithm=True) as graph:
        assert set(graph.get_graph().nodes) == {
            "__start__", "input", "vision", "prompt", "act", "validator",
            "controller", "sim", "output", "feedback_router", "mode",
            "safety_gate", "execution", "done", "failed", "rejected", "__end__",
        }

    state, directory = run(tmp_path)
    assert state["status"] == "algorithm_passed"
    assert state["selected_experiment"] == "michelson_full"
    assert [event["stage"] for event in state["events"]] == [
        "input", "vision", "prompt", "act", "validator", "controller", "sim", "output",
        "feedback_router", "controller", "sim", "output", "feedback_router",
        "act", "validator", "controller", "sim", "output", "mode", "execution", "done",
    ]
    assert routes(state) == [("output", "controller"), ("output", "act")]
    assert [item["code"] for item in state["retry_log"]] == [
        "ALGORITHM_POSITION_MISMATCH", "ALGORITHM_OPTICAL_MISMATCH",
    ]
    decisions = state["algorithm_decisions"]
    assert [item["action"] for item in decisions] == [
        "correct_position", "correct_position", "align_single_object", "apply_alignment",
    ]
    assert all(item["source"] == "rule_policy_v1" and item["target_object_ids"] == ["L25"]
               for item in decisions)

    world = state["algorithm_state"]
    history = state["algorithm_history"]
    assert [item["revision"] for item in history] == [1, 2, 3]
    assert len(world["initial"]) == 15 and world["revision"] == 3
    current = world["initial"]
    for item in history:
        assert item["before"] == current
        assert item["moved_ids"] == ["L25"]
        assert {key for key in current if current[key] != item["after"][key]} == {"L25"}
        assert item["limits"] == pytest.approx({"position_m": .001, "angle_rad": .00872664626})
        current = item["after"]
    assert current == world["current"]
    assert [item["position_errors_m"]["L25"] for item in history] == pytest.approx(
        [.0015, .00075, .00075])
    assert [degrees(item["angle_errors_rad"]["L25"]) for item in history] == pytest.approx(
        [2, 2, .4])
    assert [(item["mechanical_passed"], item["optical_passed"]) for item in history] == [
        (False, False), (True, False), (True, True),
    ]

    plan = (directory / state["plan_path"]).read_bytes()
    blob = (directory / state["artifact_path"]).read_bytes()
    with ZipFile(BytesIO(blob)) as archive:
        assert archive.read("plan.json") == plan
        assert decode_json(archive.read("result.json").decode()) == history[-1]
    assert (directory / "team/sim-03/plan.json").read_bytes() == plan
    assert (directory / "team/execution-01/plan.json").read_bytes() == plan
    assert state["result"]["plan_hash"] == state["verification"]["plan_hash"] == digest(plan)
    assert state["result"]["artifact_sha256"] == digest(blob)
    assert state["result"]["status"] == "accepted_dry_run"
    assert state["result"]["motions_sent"] == 0
    assert state["result"]["physical_verified"] is False


@pytest.mark.parametrize(("fault", "route"), [
    ("vision", ("vision", "input")), ("prompt", ("validator", "prompt")),
    ("act", ("validator", "act")), ("controller", ("controller", "controller")),
    ("sim", ("sim", "sim")), ("output", ("output", "sim")),
])
def test_recoverable_fault_routes_from_actual_failure(tmp_path, fault, route):
    state, _ = run(tmp_path, team_fault=fault)
    assert state["status"] == "algorithm_passed"
    assert routes(state) == [route, ("output", "controller"), ("output", "act")]
    assert any(event["stage"] == route[0] and event.get("outcome") == "failed"
               for event in state["events"])
    assert [item["revision"] for item in state["algorithm_history"]] == [1, 2, 3]
    assert state["attempts"]["execution"] == 1


def test_execution_fault_does_not_produce_receipt(tmp_path):
    state, directory = run(tmp_path, team_fault="execution")
    assert state["status"] == "failed"
    assert state["error"]["stage"] == "execution"
    assert state["error"]["code"] == "ALGORITHM_HANDOFF_REJECTED"
    assert state["attempts"]["execution"] == 1
    assert state["result"] is None
    assert not list((directory / "team").glob("execution-*/execution_receipt.json"))


def test_matrix_fault_and_slow_response_cannot_exceed_total_retry_budget(tmp_path):
    state, _ = run(tmp_path, team_fault="matrix", options={"angle_response": .4})
    assert state["status"] == "failed"
    assert routes(state)[:2] == [("vision", "input"), ("validator", "prompt")]
    assert ("sim", "sim") in routes(state)
    terminal = state["retry_log"][-1]
    assert terminal["to"] == "stop"
    assert terminal["total_retries"] == terminal["total_limit"]
    assert "execution" not in state["attempts"] and state["result"] is None


@pytest.mark.parametrize(("option", "target"), [
    ("position_response", "controller"), ("angle_response", "act"),
])
def test_zero_response_stops_at_retry_budget(tmp_path, option, target):
    state, _ = run(tmp_path, options={option: 0.0})
    assert state["status"] == "failed"
    assert state["retry_log"][-1]["reason"] == "target_retry_limit"
    assert routes(state)[-1] == ("output", "stop")
    key = state["retry_log"][-1]["retry_key"]  # placement repairs count per part, e.g. "controller:L25"
    assert key.startswith(target + ":") and target not in state["retry_counts"]
    assert state["retry_counts"][key] == state["retry_log"][-1]["retry_limit"] == 2  # the part still stops at 2
    assert "execution" not in state["attempts"]
    assert state["result"] is None


def test_registered_mach_zehnder(tmp_path):
    state, _ = run(tmp_path, request="Assemble a 7-component Mach-Zehnder interferometer.")
    assert state["status"] == "algorithm_passed"
    assert state["selected_experiment"] == "mach_zehnder"
    assert len(state["algorithm_state"]["current"]) == 7


def test_registered_assembly_is_bound_to_its_offline_reference(tmp_path):
    request_text = "On the optical table, assemble a Michelson interferometer."
    state, directory = run(tmp_path, request=request_text)
    assert state["status"] == "algorithm_passed"
    spec = state["algorithm_experiment"]
    assert spec["request"] == request_text
    assert spec["component_count"] == len(state["algorithm_state"]["current"]) == 15
    assert spec["source"] == "registered_reference"
    assert spec["assumptions"] and spec["pending_requirements"]
    assert spec["design_verified"] is state["verification"]["design_verified"] is False
    assert routes(state) == [("output", "controller"), ("output", "act")]
    assert all(len(item["moved_ids"]) == 1 for item in state["algorithm_history"])
    plan_bytes = (directory / state["plan_path"]).read_bytes()
    plan = decode_json(plan_bytes.decode())
    assert plan["experiment_sha256"] == state["verification"]["experiment_sha256"]
    assert plan["experiment_sha256"] == spec["contract_sha256"]
    input_dir = directory / "team/input-01"
    experiment_bytes = (input_dir / "experiment.json").read_bytes()
    assert decode_json(experiment_bytes.decode()) == spec
    completion = decode_json((input_dir / "completion.json").read_text())
    assert completion["files"]["experiment.json"] == digest(experiment_bytes)
    with ZipFile(directory / state["artifact_path"]) as archive:
        assert archive.read("plan.json") == plan_bytes


@pytest.mark.parametrize("request_text", [
    "Assemble a Fabry-Perot interferometer.",
    "Assemble a custom 27-component polarization experiment.",
    "Assemble a polarization experiment with two polarizers and three wave plates.",
    "Build a telescope with two lenses and a camera.",
])
def test_novel_requests_require_agent_instead_of_offline_fixture(tmp_path, request_text):
    state, directory = run(tmp_path, request=request_text)
    assert state["status"] == "failed"
    assert state["error"]["code"] == "EXPERIMENT_DEFINITION_INVALID"
    assert state["attempts"] == {"input": 1}
    assert not list((directory / "team").glob("controller-*/plan.json"))


def test_structured_experiment_preserves_fixed_components_and_targets(tmp_path):
    fixed = {"object_id": "REF-1", "label": "Reference", "position_m": [.2, .3, .04],
             "movable": False}
    adjustable = {"object_id": "new.part-7", "label": "Adjustable component",
                  "position_m": [.43, .28, .05], "yaw_rad": .2}
    request_text = encode({"title": "Arbitrary bench", "components": [fixed, adjustable]}).decode()
    state, _ = run(tmp_path, request=request_text)
    assert state["status"] == "algorithm_passed"
    assert state["algorithm_experiment"]["source"] == "user_definition"
    for component in (fixed, adjustable):
        target = next(item for item in state["layout"]["components"]
                      if item["object_id"] == component["object_id"])
        assert target["target_pose"]["position_m"] == component["position_m"]
    world = state["algorithm_state"]
    assert world["initial"]["REF-1"] == world["current"]["REF-1"]
    assert world["current"]["REF-1"]["position_m"] == fixed["position_m"]
    assert all(item["moved_ids"] == ["new.part-7"] for item in state["algorithm_history"])
    assert world["current"]["new.part-7"]["yaw_rad"] == pytest.approx(.2, abs=.009)


@pytest.mark.parametrize("request_text", [
    '{"title":"duplicate","components":[{"object_id":"A","position_m":[0,0,0]},'
    '{"object_id":"A","position_m":[1,0,0]}]}',
    '{"title":"nonfinite","components":[{"object_id":"A","position_m":[NaN,0,0]}]}',
    "Assemble a custom 4-component experiment with two polarizers and three wave plates.",
])
def test_invalid_experiment_definition_stops_at_input(tmp_path, request_text):
    state, _ = run(tmp_path, request=request_text)
    assert state["status"] == "failed"
    assert state["attempts"] == {"input": 1}
    assert state["error"]["code"] == "EXPERIMENT_DEFINITION_INVALID"


@pytest.mark.parametrize("substitute_other_run", [False, True])
def test_changed_experiment_cannot_reach_simulation_or_handoff(
        tmp_path, monkeypatch, substitute_other_run):
    other = None
    if substitute_other_run:
        other, _ = run(tmp_path, request="Assemble a 7-component Mach-Zehnder interferometer.")
        assert other["status"] == "algorithm_passed"
    simulate = AlgorithmServices.sim

    def with_changed_experiment(self, state, path, attempt):
        changed = (other["algorithm_experiment"] if other else
                   {**state["algorithm_experiment"], "title": "Changed title"})
        return simulate(self, {**state, "algorithm_experiment": changed}, path, attempt)

    monkeypatch.setattr(AlgorithmServices, "sim", with_changed_experiment)
    state, _ = run(tmp_path)
    assert state["status"] == "failed"
    assert state["error"]["stage"] == "sim"
    assert state["algorithm_state"]["revision"] == 0
    assert "execution" not in state["attempts"]


@pytest.mark.parametrize(("mutation", "message"), [
    ("hash", "recorded hash"), ("scope", "single-object authorization"),
])
def test_sim_rejects_changed_plan_hash_or_scope(tmp_path, monkeypatch, mutation, message):
    sim = AlgorithmServices.sim

    def simulate(self, state, path, attempt):
        plan_path = self.root / state["plan_path"]
        raw = plan_path.read_bytes()
        if mutation == "hash":
            plan_path.write_bytes(raw + b" ")
        else:
            changed = decode_json(raw.decode())
            changed["allowed_object_ids"] = []
            raw = encode(changed)
            plan_path.write_bytes(raw)
            state = {**state, "plan_sha256": digest(raw)}
        return sim(self, state, path, attempt)

    monkeypatch.setattr(AlgorithmServices, "sim", simulate)
    state, directory = run(tmp_path)
    assert state["status"] == "failed"
    assert state["error"]["stage"] == "sim"
    assert message in state["error"]["message"]
    assert state["algorithm_state"]["revision"] == 0
    assert not state.get("algorithm_history") and "execution" not in state["attempts"]
    assert not list((directory / "team").glob("sim-*/artifact.zip"))


def test_pause_continue_then_stop_at_committed_boundaries(tmp_path, monkeypatch):
    original = AlgorithmServices.sim
    entered, release = [Event(), Event()], [Event(), Event()]

    def simulate(self, state, path, attempt):
        if attempt <= 2:
            entered[attempt - 1].set()
            if not release[attempt - 1].wait(5):
                raise RuntimeError("test did not release simulation")
        return original(self, state, path, attempt)

    monkeypatch.setattr(AlgorithmServices, "sim", simulate)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(run_input())
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(runtime.execute, run_id)
        try:
            assert entered[0].wait(5)
            runtime.control(run_id, "pause")
        finally:
            release[0].set()
        paused = job.result(timeout=5)
        assert (paused["status"], paused["stage"]) == ("paused", "sim")
        assert paused["algorithm_state"]["revision"] == 1
        assert "output" not in paused["attempts"]

        runtime = Runtime(tmp_path)
        job = pool.submit(runtime.execute, run_id, continue_run=True)
        try:
            assert entered[1].wait(5)
            runtime.control(run_id, "stop")
        finally:
            release[1].set()
        stopped = job.result(timeout=5)
        assert (stopped["status"], stopped["stage"]) == ("stopped", "sim")
        assert [item["revision"] for item in stopped["algorithm_history"]] == [1, 2]
        assert stopped["attempts"]["output"] == 1
        assert "execution" not in stopped["attempts"]
        with pytest.raises(ValueError):
            runtime.execute(run_id, continue_run=True)


def test_restart_after_sim_journal_does_not_apply_motion_twice(tmp_path, monkeypatch):
    original = AlgorithmServices.sim
    atomic = team_flow.atomic
    simulated = []

    def simulate(self, state, path, attempt):
        simulated.append(attempt)
        return original(self, state, path, attempt)

    def interrupt(path, data):
        atomic(path, data)
        if path.name == "completion.json" and path.parent.name == "sim-01":
            raise RuntimeError("interrupted after the journal write")

    monkeypatch.setattr(AlgorithmServices, "sim", simulate)
    monkeypatch.setattr(team_flow, "atomic", interrupt)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(run_input())
    assert runtime.execute(run_id)["status"] == "interrupted"
    assert simulated == [1]
    monkeypatch.setattr(team_flow, "atomic", atomic)

    resumed = Runtime(tmp_path).execute(run_id, continue_run=True)
    assert resumed["status"] == "algorithm_passed"
    assert simulated == [1, 2, 3]
    assert [item["revision"] for item in resumed["algorithm_history"]] == [1, 2, 3]
    assert resumed["algorithm_history"][0]["position_errors_m"]["L25"] == pytest.approx(.0015)


def test_api_exposes_only_bound_algorithm_artifacts(tmp_path):
    state, directory = run(tmp_path)
    run_id = state["run_id"]
    with TestClient(create_app(Runtime(tmp_path))) as client:
        graph = client.get('/api/graph?team=true&algorithm=true').json()['mermaid']
        assert 'precision_sim' not in graph and 'feedback_router' in graph
        assert client.get('/api/runs').json()[0]['simulation_backend'] == 'algorithm_v1'
        prefix = f'/api/runs/{run_id}/algorithm/artifacts/'
        assert client.get(prefix+'prompt/1/prompt.txt').status_code == 200
        assert client.get(prefix+'sim/3/artifact.zip').content == (directory/state['artifact_path']).read_bytes()
        experiment = directory / "team/input-01/experiment.json"
        assert client.get(prefix+"input/1/experiment.json").content == experiment.read_bytes()
        experiment.write_bytes(experiment.read_bytes() + b" ")
        assert client.get(prefix+"input/1/experiment.json").status_code == 409
        assert client.get(prefix+'output/1/verification.json').json()['checks']['mechanical_proxy'] is False
        assert client.get(prefix+'config/1/local.json').status_code == 404
        assert client.get(prefix+'sim/99/event.json').status_code == 404
        assert client.post(f'/api/runs/{run_id}/continue',json={},headers={'X-Optics-Local':'1'}).status_code == 409
        (directory/'team/controller-03/plan.json').write_bytes(b'{}')
        assert client.get(prefix+'controller/3/plan.json').status_code == 409


def test_nominal_simulation_and_invalid_options(tmp_path):
    state, _ = run(tmp_path, options={'initial_position_error_m':0.0,'initial_angle_error_rad':0.0})
    assert state['status'] == 'algorithm_passed'
    assert not state.get('retry_log')
    assert state['algorithm_history'][0]['moved_ids'] == []
    assert state['algorithm_state']['initial'] == state['algorithm_state']['current']
    for options in ({'position_response':float('nan')},{'initial_position_error_m':True},{'unknown':1}):
        with pytest.raises(ValueError):
            run_input(options=options)


def test_changed_queued_backend_cannot_reach_physical_worker(tmp_path, monkeypatch):
    def worker(*args, **kwargs):
        pytest.fail("queued algorithm run reached the physical worker")

    monkeypatch.setattr(team_flow, "worker", worker)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(run_input())
    path = runtime.run_dir(run_id) / "input.json"
    input_data = decode_json(path.read_text())
    input_data["simulation_backend"] = "physical"
    path.write_bytes(encode(input_data))

    state = runtime.execute(run_id)
    assert state["status"] == "failed"
    assert state["stage"] == "input"
    assert state["simulation_backend"] == "algorithm_v1"
    assert not state["events"]
    assert not (runtime.run_dir(run_id) / "team").exists()


@pytest.mark.parametrize("completed", [True, False], ids=["completed-receipt", "unknown-handoff"])
def test_execution_restart_does_not_invoke_handoff_again(tmp_path, monkeypatch, completed):
    original = AlgorithmServices.execution
    atomic = team_flow.atomic
    calls = []
    interrupted_at = "completion.json" if completed else "execution-started.json"

    def execution(self, state, path, attempt):
        calls.append(attempt)
        return original(self, state, path, attempt)

    def interrupt(path, data):
        atomic(path, data)
        if path.parent.name == "execution-01" and path.name == interrupted_at:
            raise RuntimeError("interrupted after durable execution record")

    monkeypatch.setattr(AlgorithmServices, "execution", execution)
    monkeypatch.setattr(team_flow, "atomic", interrupt)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(run_input())
    assert runtime.execute(run_id)["status"] == "interrupted"

    directory = runtime.run_dir(run_id) / "team/execution-01"
    record = directory / interrupted_at
    saved_record = record.read_bytes()
    receipt = directory / "execution_receipt.json"
    saved_receipt = receipt.read_bytes() if completed else None
    assert calls == ([1] if completed else [])
    monkeypatch.setattr(team_flow, "atomic", atomic)

    resumed = Runtime(tmp_path).execute(run_id, continue_run=True)
    assert calls == ([1] if completed else [])
    assert record.read_bytes() == saved_record
    assert resumed["attempts"]["execution"] == 1
    if completed:
        assert resumed["status"] == "algorithm_passed"
        assert receipt.read_bytes() == saved_receipt
        assert resumed["result"] == decode_json(saved_receipt.decode())
    else:
        assert resumed["status"] == "failed"
        assert resumed["error"]["stage"] == "execution"
        assert resumed["result"] is None
        assert not receipt.exists()


@pytest.mark.parametrize("tamper", [False, True])
def test_large_feedback_journal_survives_checkpoint_gap(tmp_path, monkeypatch, tamper):
    invoke_input = AlgorithmServices.input
    atomic = team_flow.atomic
    calls = []
    payload = {"code": "TEST_LARGE_FEEDBACK", "details": {"evidence": "x" * 280_000}}
    model_log = encode({"output_text": "x" * 810_000}, limit=None)

    def large_input(self, state, path, attempt):
        calls.append(attempt)
        update = invoke_input(self, state, path, attempt)
        atomic(path / "model-response.json", model_log)
        return {**update, "feedback": [payload, payload]}

    def interrupt(path, data):
        atomic(path, data)
        if path.name == "completion.json" and path.parent.name == "input-01":
            raise RuntimeError("interrupted after journal write")

    monkeypatch.setattr(AlgorithmServices, "input", large_input)
    monkeypatch.setattr(team_flow, "atomic", interrupt)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(run_input())
    assert runtime.execute(run_id)["status"] == "interrupted"
    assert calls == [1]
    monkeypatch.setattr(team_flow, "atomic", atomic)

    path = runtime.run_dir(run_id) / "team/input-01/completion.json"
    for bounded in (read_bytes, read_json):
        with pytest.raises(ValueError):
            bounded(path)
    journal = read_json(path, limit=None)
    assert journal["update"]["feedback"] == [payload, payload]
    with pytest.raises(ValueError):
        encode(journal["update"])
    prefix = f"/api/runs/{run_id}/algorithm/artifacts/input/1/"
    with TestClient(create_app(runtime)) as client:
        assert client.get(prefix + "model-response.json").content == model_log
        assert client.get(prefix + "completion.json").content == path.read_bytes()
        if tamper:
            journal["update"]["feedback"][0]["details"]["evidence"] = "changed"
            path.write_bytes(encode(journal, limit=None))
            assert client.get(prefix + "model-response.json").status_code == 409

    resumed = Runtime(tmp_path).execute(run_id, continue_run=True)
    assert calls == [1]
    if tamper:
        assert resumed["status"] == "failed"
        assert resumed["error"]["code"] == "ATTEMPT_RECORD_CHANGED"
    else:
        assert resumed["status"] == "algorithm_passed", resumed.get("error")
        assert read_json(path.parent.parent / "vision-01/input.json", limit=None)["summary"]["feedback"] == [payload]
