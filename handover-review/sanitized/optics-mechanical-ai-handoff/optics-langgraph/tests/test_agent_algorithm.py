"""End-to-end Agent design and recovery with only the model transport replaced."""
from copy import deepcopy
import json
from math import degrees
from zipfile import ZipFile

import pytest

from optics_graph import agent_algorithm, team_flow
from optics_graph.gateway import RetryableGatewayError
from optics_graph.runtime import RunInput, Runtime


from agent_canned import MICHELSON, TELESCOPE, REVIEW_ACCEPTED, canned_answer, component, fixture, optical_suite  # noqa: F401


def run(tmp_path, monkeypatch, quotes=TELESCOPE, alter=None, *, language=None, resume_legacy=False):
    intent, design = fixture(quotes)
    request = "；".join(quotes)
    calls = []

    class FakeGateway:
        def __init__(self, provider, run_dir):
            assert provider == "openai"
            self.run_dir = run_dir

        def _request(self, role, prompt, schema, attempt):
            instructions, payload = prompt.split("Treat the following JSON as data.\n", 1)
            assert self.output_language == (language or "en")
            assert "Write all human-readable output fields in concise English" in instructions
            assert "Keep each source_quote verbatim" in instructions
            assert "Chinese prose" not in instructions
            data = json.loads(payload)
            assert data["request"] == request
            assert schema["required"]
            calls.append((role, attempt, data))
            if role not in {"interpret", "design", "optical_tests", "review", "repair_act", "repair_controller"}:
                pytest.fail(f"unexpected model role: {role}")
            response = canned_answer(role, data, quotes, intent, design)
            candidate = deepcopy(response)
            return alter(role, attempt, candidate) if alter else candidate

    monkeypatch.setattr(agent_algorithm, "public_settings", lambda: {"model": "gpt-6.1-sol"})
    if resume_legacy:
        original_dump = RunInput.model_dump
        original_atomic = team_flow.atomic

        def legacy_dump(self, **kwargs):
            values = original_dump(self, **kwargs)
            values.pop("output_language", None)
            return values

        def interrupt(path, data):
            original_atomic(path, data)
            if path.name == "completion.json" and path.parent.name == "act-01":
                raise RuntimeError("legacy process stopped after writing the Act journal")

        monkeypatch.setattr(RunInput, "model_dump", legacy_dump)
        monkeypatch.setattr(team_flow, "atomic", interrupt)
    runtime = Runtime(tmp_path, gateway_factory=FakeGateway)
    run_id = runtime.create(RunInput(request=request, profile="auto", mode="github_local",
                                     simulation_backend="algorithm_v1", provider="openai",
                                     algorithm_options={"delay_s": 0.},
                                     **({"output_language": language} if language else {})))
    state = runtime.execute(run_id)
    if resume_legacy:
        assert state["status"] == "interrupted"
        assert "output_language" not in state
        journal = runtime.run_dir(run_id) / "team/act-01/completion.json"
        original = journal.read_bytes()
        monkeypatch.setattr(RunInput, "model_dump", original_dump)
        monkeypatch.setattr(team_flow, "atomic", original_atomic)
        state = runtime.execute(run_id, continue_run=True)
        assert journal.read_bytes() == original
    return state, runtime.run_dir(run_id), calls, design


def read(path):
    return json.loads(path.read_text())


def test_overloaded_test_agent_waits_then_retries_without_redesign(tmp_path, monkeypatch):
    from optics_graph import team_flow
    waits = []
    monkeypatch.setattr(team_flow.time, "sleep", waits.append)

    def overload(role, attempt, candidate):
        if role == "optical_tests" and attempt == 1:
            raise RetryableGatewayError("provider overloaded", code="MODEL_SERVICE_UNAVAILABLE",
                                        details={"provider_code":"server_is_overloaded", "retry_after_s":7.})
        return candidate

    state, directory, calls, _ = run(tmp_path, monkeypatch, alter=overload)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert waits.count(7.) == 1
    assert [role for role, _, _ in calls].count("design") == 1
    assert state["retry_log"][0]["code"] == "MODEL_SERVICE_UNAVAILABLE"
    assert read(directory/"team/act-02/decision.json")["action"] == "preserve_design_for_test_repair"
    assert read(directory/"team/act-02/event.json")["provider_retry_wait_s"] == 7.


def test_coaxial_lens_and_holder_origins_do_not_claim_a_collision():
    from optics_graph.design_agent import check_design
    intent, design = fixture(TELESCOPE)
    holder = component("HOLDER", "lens_holder", "Objective mount", .2, .15)
    design["components"].append(holder)
    design["connections"].append({"from_id":"HOLDER","to_id":"OBJ","role":"Concentric clamping support"})
    assert check_design(intent,design) == []


@pytest.mark.parametrize("quotes", [TELESCOPE, MICHELSON], ids=["telescope", "michelson"])
def test_agent_design_targets_bind_assembly_and_single_object_repair(tmp_path, monkeypatch, quotes):
    state, directory, calls, design = run(tmp_path, monkeypatch, quotes)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert [role for role, _, _ in calls] == [
        "interpret", "design", "optical_tests", "review", "repair_controller", "repair_controller", "repair_act",
    ]
    intent = calls[1][2]["intent"]
    assert [requirement["source_quote"] for requirement in intent["requirements"]] == list(quotes)
    assert state["algorithm_design"] == calls[3][2]["design"] == design
    assert calls[3][2]["test_suite"] == state["algorithm_test_suite"]
    assert calls[3][2]["test_report"] == state["algorithm_test_report"]
    assert state["algorithm_test_report"]["passed"]
    assert state["layout"] == state["reference_layout"]
    assert {part["object_id"]: part["target_pose"]["position_m"]
            for part in state["layout"]["components"]} == {
                part["object_id"]: part["position_m"] for part in design["components"]}
    assert not calls[0][2]["observation"]["observations"]

    team = directory / "team"
    assert read(team / "act-01/design.json") == design
    assert read(team / "prompt-01/requirements.json") == intent
    assert read(team / "validator-01/design-review.json")["accepted"]
    assert read(team / "validator-01/validation.json")["physical_validated"] is False
    assert read(team / "validator-01/inventory.json")["revision"] == 0

    world = state["algorithm_state"]
    history = state["algorithm_history"]
    movable = [part["object_id"] for part in design["components"] if part["movable"]]
    selected = world["seeded_object_id"]
    assert [entry["revision"] for entry in history] == [1, 2, 3, 4]
    assert [entry["moved_ids"] for entry in history] == [movable, [selected], [selected], [selected]]
    assert [entry["position_errors_m"][selected] for entry in history] == pytest.approx(
        [.003, .0015, .00075, .00075])
    assert [degrees(entry["angle_errors_rad"][selected]) for entry in history] == pytest.approx(
        [2, 2, 2, .4])
    assert [item["to"] for item in state["retry_log"]] == ["controller", "controller", "act"]
    current = world["initial"]
    for i, entry in enumerate(history, 1):
        assert entry["before"] == current
        assert {oid for oid in current if current[oid] != entry["after"][oid]} == set(entry["moved_ids"])
        current = entry["after"]
        plan = read(team / f"controller-{i:02}/plan.json")
        assert plan["phase"] == ("assembly" if i == 1 else "repair")
        assert plan["allowed_object_ids"] == entry["moved_ids"]
        assert plan["target_sha256"] == read(team / "controller-01/plan.json")["target_sha256"]
    assert current == world["current"]
    for part in design["components"]:
        if not part["movable"]:
            oid = part["object_id"]
            assert current[oid] == world["initial"][oid]
            assert current[oid]["position_m"] == part["position_m"]
    for (_, _, data), previous in zip(calls[4:], history):
        assert data["current"] == previous["after"]
        assert data["allowed_ids"] == [selected]

    plan_bytes = (directory / state["plan_path"]).read_bytes()
    with ZipFile(directory / state["artifact_path"]) as artifact:
        assert artifact.read("plan.json") == plan_bytes
        assert json.loads(artifact.read("result.json")) == history[-1]
    assert (team / "execution-01/plan.json").read_bytes() == plan_bytes
    assert state["result"]["motions_sent"] == 0
    assert state["result"]["physical_verified"] is False


def test_rejected_review_routes_to_design_and_keeps_previous_candidate(tmp_path, monkeypatch):
    def alter(role, attempt, candidate):
        if role == "design" and attempt == 1:
            candidate["rationale"] = "The telescope magnification is not explained."
        if role == "review" and attempt == 1:
            return {"accepted": False, "issues": ["The telescope magnification is not explained"],
                    "return_to": "act", "problem_source": "design", "reasoning": "The design notes do not state the magnification."}
        return candidate

    state, directory, calls, design = run(tmp_path, monkeypatch, alter=alter)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert [role for role, _, _ in calls[:7]] == ["interpret", "design", "optical_tests", "review", "design", "optical_tests", "review"]
    assert calls[4][2]["feedback"][-1]["code"] == "AGENT_DESIGN_REJECTED"
    assert state["algorithm_design"] == design
    previous = read(directory / "team/act-01/design.json")
    assert previous["rationale"] == "The telescope magnification is not explained."
    retry_data = calls[4][2]
    assert retry_data["reference"]["previous_candidate"] == previous
    assert retry_data["reference"]["candidate_sha256"] == (
        retry_data["feedback"][-1]["details"]["candidate_sha256"]
    )
    assert not read(directory / "team/validator-01/design-review.json")["accepted"]
    assert not (directory / "team/validator-01/inventory.json").exists()


def test_missing_requirement_coverage_rejects_without_model_review_or_fallback(tmp_path, monkeypatch):
    def alter(role, attempt, candidate):
        if role == "design":
            candidate["requirements_coverage"].pop()
        return candidate

    state, directory, calls, _ = run(tmp_path, monkeypatch, alter=alter)
    assert state["status"] == "failed"
    assert state["error"]["code"] == "AGENT_DESIGN_CONTRACT"
    assert {role for role, _, _ in calls} == {"interpret", "design"}
    assert not state["algorithm_state"]["current"]
    assert not list((directory / "team").glob("controller-*/plan.json"))


@pytest.mark.parametrize("recovers", [True, False])
def test_provider_timeout_retries_only_in_graph(tmp_path, monkeypatch, recovers):
    def alter(role, attempt, candidate):
        if role == "design" and (attempt == 1 or not recovers):
            raise RetryableGatewayError("mock model timeout")
        return candidate

    state, directory, calls, _ = run(tmp_path, monkeypatch, alter=alter)
    assert [attempt for role, attempt, _ in calls if role == "design"][:2] == [1, 2]
    assert state["retry_log"][0]["to"] == "act"
    if recovers:
        assert state["status"] == "algorithm_passed", state.get("error")
    else:
        assert state["status"] == "failed"
        assert state["retry_log"][-1]["to"] == "stop"
        assert not state["algorithm_state"]["current"]
        assert not list((directory / "team").glob("controller-*/plan.json"))
        assert {role for role, _, _ in calls} == {"interpret", "design"}


def test_model_cannot_repair_an_unapproved_neighbor(tmp_path, monkeypatch):
    def alter(role, attempt, candidate):
        if role == "repair_controller":
            candidate["object_id"] = "EYE"
        return candidate

    state, directory, calls, _ = run(tmp_path, monkeypatch, alter=alter)
    assert state["status"] == "failed"
    assert [entry["revision"] for entry in state["algorithm_history"]] == [1]
    assert [data["allowed_ids"] for role, _, data in calls if role == "repair_controller"]
    assert all(data["allowed_ids"] == ["OBJ"] for role, _, data in calls if role == "repair_controller")
    assert state["algorithm_state"]["current"] == state["algorithm_history"][0]["after"]
    assert [path.parent.name for path in (directory / "team").glob("controller-*/plan.json")] == [
        "controller-01",
    ]
    assert "execution" not in state["attempts"]


def test_strict_user_tolerance_cannot_pass_at_default_threshold(tmp_path, monkeypatch):
    quotes = (*TELESCOPE[:2], TELESCOPE[2] + ", position tolerance 0.1 mm")

    def alter(role, attempt, candidate):
        if role == "interpret":
            candidate["position_tolerance_m"] = .0001
        return candidate

    state, _, _, _ = run(tmp_path, monkeypatch, quotes, alter=alter)
    assert state["status"] == "failed"
    assert state["error"]["code"] == "ALGORITHM_POSITION_MISMATCH"
    assert state["retry_log"][-1]["reason"] == "target_retry_limit"
    assert state["algorithm_experiment"]["execution_limits"]["position_m"] == .0001
    assert state["layout"]["tolerance"]["position_m"] == .0001
    last = state["algorithm_history"][-1]
    assert .0001 < last["position_errors_m"]["OBJ"] < .001
    assert last["limits"]["position_m"] == .0001
    assert last["mechanical_passed"] is False
    assert "execution" not in state["attempts"]


@pytest.mark.parametrize(("stage", "code", "rounds"), [
    ("validator", "DESIGN_REVIEW_CHANGED", 3),
    ("output", "DESIGN_REVIEW_MISSING", 1),
])
def test_accepted_review_cannot_be_reused_for_another_contract(tmp_path, monkeypatch, stage, code, rounds):
    services = agent_algorithm.AgentAlgorithmServices
    original = getattr(services, stage)

    def with_wrong_review(self, state, path, attempt):
        if state["algorithm_state"]["revision"] > 0:
            review = state["algorithm_review"]
            assert review["accepted"]
            assert review["experiment_sha256"] == state["algorithm_experiment"]["contract_sha256"]
            state = {**state, "algorithm_review": {**review, "experiment_sha256": "0" * 64}}
        return original(self, state, path, attempt)

    monkeypatch.setattr(services, stage, with_wrong_review)
    state, _, calls, _ = run(tmp_path, monkeypatch)
    assert state["status"] == "failed"
    assert state["error"]["code"] == code
    assert state["algorithm_state"]["revision"] == rounds
    assert [role for role, _, _ in calls].count("review") == 1
    assert "execution" not in state["attempts"]


def test_stale_scope_rejected_before_model_is_called(tmp_path, monkeypatch):
    services = agent_algorithm.AgentAlgorithmServices
    original = services.local_repair

    def with_stale_scope(self, node, state, path, attempt, expected):
        feedback = deepcopy(state["feedback"])
        diagnostic = feedback[-1]
        assert diagnostic["details"]["world_revision"] == state["algorithm_state"]["revision"]
        diagnostic["details"]["world_revision"] -= 1
        return original(self, node, {**state, "feedback": feedback}, path, attempt, expected)

    monkeypatch.setattr(services, "local_repair", with_stale_scope)
    state, _, calls, _ = run(tmp_path, monkeypatch)
    assert state["status"] == "failed"
    assert state["error"]["code"] == "REPAIR_SCOPE_UNPROVEN"
    assert [role for role, _, _ in calls] == ["interpret", "design", "optical_tests", "review"]
    assert state["algorithm_state"]["revision"] == 1
    assert "execution" not in state["attempts"]


def test_failed_analytic_check_returns_measured_difference_to_act(tmp_path, monkeypatch):
    def alter(role, attempt, candidate):
        if role == "design" and attempt == 1:
            candidate["components"][1]["position_m"][0] = .46
        return candidate

    state, directory, calls, design = run(tmp_path, monkeypatch, alter=alter)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert [role for role, _, _ in calls[:7]] == [
        "interpret", "design", "optical_tests", "review", "design", "optical_tests", "review",
    ]
    assert calls[3][2]["test_report"]["passed"] is False
    assert read(directory / "team/validator-01/design-review.json")["accepted"] is True
    failure = calls[4][2]["feedback"][-1]
    assert failure["code"] == "OPTICAL_TEST_FAILED"
    assert state["retry_log"][0]["to"] == "act"
    failed_report = failure["details"]["test_report"]
    assert failed_report == read(directory / "team/validator-01/optical-test-report.json")
    assert failure["details"]["test_suite"] == read(directory / "team/validator-01/optical-test-suite.json")
    separation = next(test for test in failed_report["tests"] if test["id"] == "separation")
    assert separation["status"] == "failed"
    assert separation["actual"] == pytest.approx(.26)
    assert separation["expected"] == .25
    assert not (directory / "team/validator-01/inventory.json").exists()
    assert state["algorithm_design"] == design
    assert state["algorithm_test_report"]["passed"]


@pytest.mark.parametrize(("field", "change"), [
    ("algorithm_test_suite", {"title": "Tampered test suite"}),
    ("algorithm_test_report", {"physical_verified": True}),
])
def test_tampered_test_evidence_is_rejected_before_execution(tmp_path, monkeypatch, field, change):
    services = agent_algorithm.AgentAlgorithmServices
    original = services.validator

    def tamper(self, state, path, attempt):
        if state["algorithm_state"]["revision"] > 0:
            state = {**state, field: {**state[field], **change}}
        return original(self, state, path, attempt)

    monkeypatch.setattr(services, "validator", tamper)
    state, _, calls, _ = run(tmp_path, monkeypatch)
    assert state["status"] == "failed"
    assert state["error"]["code"] == "OPTICAL_TEST_BINDING_CHANGED"
    assert [role for role, _, _ in calls].count("review") == 1
    assert "execution" not in state["attempts"]


def test_truncated_model_output_retries_act_with_original_diagnostic(tmp_path, monkeypatch):
    details = {"reason": "max_output_tokens", "usage": {"output_tokens": 1000}, "response_id": "partial"}

    def alter(role, attempt, candidate):
        if role == "design" and attempt == 1:
            raise RetryableGatewayError("MODEL_OUTPUT_TRUNCATED", code="MODEL_OUTPUT_TRUNCATED", details=details)
        return candidate

    state, _, calls, _ = run(tmp_path, monkeypatch, alter=alter)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert state["retry_log"][0]["to"] == "act"
    assert [(role, attempt) for role, attempt, _ in calls[:3]] == [
        ("interpret", 1), ("design", 1), ("design", 2),
    ]
    feedback = calls[2][2]["feedback"][-1]
    assert feedback["code"] == "MODEL_OUTPUT_TRUNCATED"
    assert feedback["details"] == details


@pytest.mark.parametrize("cause", ["invalid_expression", "review_rejected_tests", "review_rejected_failed_test"])
def test_test_failure_reuses_candidate_and_regenerates_suite(tmp_path, monkeypatch, cause):
    def alter(role, attempt, candidate):
        if cause == "invalid_expression" and role == "optical_tests" and attempt == 1:
            candidate["tests"][-1]["expression"] = "unknown('OBJ','EYE')"
        if cause == "review_rejected_failed_test" and role == "optical_tests" and attempt == 1:
            candidate["tests"][-1]["expected"] = ".26"
        if cause.startswith("review_rejected") and role == "review" and attempt == 1:
            return {"accepted": False, "issues": ["The test basis needs to be completed"],
                    "return_to": "act", "problem_source": "tests", "reasoning": "The check basis is incomplete."}
        return candidate

    state, directory, calls, design = run(tmp_path, monkeypatch, alter=alter)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert state["retry_log"][0]["to"] == "act"
    assert [attempt for role, attempt, _ in calls if role == "design"] == [1]
    test_calls = [(attempt, data) for role, attempt, data in calls if role == "optical_tests"]
    assert [attempt for attempt, _ in test_calls] == [1, 2]
    feedback = test_calls[1][1]["feedback"][-1]
    assert feedback["code"] == "OPTICAL_TEST_INVALID"
    if cause == "invalid_expression":
        assert feedback["details"]["test_report"]["tests"][-1]["status"] == "invalid"
    else:
        assert feedback["details"]["problem_source"] == "tests"
        if cause == "review_rejected_failed_test":
            review_data = next(data for role, attempt, data in calls if role == "review" and attempt == 1)
            report = review_data["test_report"]
            assert not report["passed"]
            assert report["tests"][-1]["actual"] == pytest.approx(.25)
            assert report["tests"][-1]["expected"] == .26
            assert feedback["details"]["test_report"] == report
    team = directory / "team"
    assert read(team / "act-01/design.json") == read(team / "act-02/design.json") == design
    assert read(team / "act-01/target_layout.json") == read(team / "act-02/target_layout.json")
    assert not (team / "validator-01/inventory.json").exists()


@pytest.mark.parametrize("tamper", ["missing_check", "rejected_review", "experiment_hash"])
def test_execution_rejects_tampered_acceptance_evidence(tmp_path, monkeypatch, tamper):
    services = agent_algorithm.AgentAlgorithmServices
    original = services.execution

    def change_evidence(self, state, path, attempt):
        state = deepcopy(state)
        if tamper == "missing_check":
            del state["verification"]["checks"]["analytic_design_tests"]
        elif tamper == "rejected_review":
            state["algorithm_review"]["accepted"] = False
        else:
            state["verification"]["experiment_sha256"] = "0" * 64
        return original(self, state, path, attempt)

    monkeypatch.setattr(services, "execution", change_evidence)
    state, directory, _, _ = run(tmp_path, monkeypatch)
    assert state["status"] == "failed"
    expected = "OPTICAL_TEST_BINDING_CHANGED" if tamper == "rejected_review" else "ALGORITHM_HANDOFF_REJECTED"
    assert state["error"]["code"] == expected
    assert not list((directory / "team").glob("execution-*/execution_receipt.json"))


def test_english_run_prompts_every_agent_role(tmp_path, monkeypatch):
    state, directory, calls, _ = run(tmp_path, monkeypatch, language="en")
    assert state["status"] == "algorithm_passed", state.get("error")
    assert state["output_language"] == "en"
    assert read(directory / "input.json")["output_language"] == "en"
    assert {role for role, _, _ in calls} == {
        "interpret", "design", "optical_tests", "review", "repair_act", "repair_controller",
    }
    requirements = read(directory / "team/prompt-01/requirements.json")["requirements"]
    assert [requirement["source_quote"] for requirement in requirements] == list(TELESCOPE)


def test_legacy_run_replays_original_journal(tmp_path, monkeypatch):
    state, _, calls, _ = run(tmp_path, monkeypatch, resume_legacy=True)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert "output_language" not in state
    assert [attempt for role, attempt, _ in calls if role == "design"] == [1]
