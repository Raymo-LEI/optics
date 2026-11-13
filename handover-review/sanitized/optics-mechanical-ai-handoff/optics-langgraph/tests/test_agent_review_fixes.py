"""Review fixes for the agent-decides feature (H1, H2, M1-M3, L2-L5). Every test here fails on the code before
the fix. Harness helpers and the no-cloud guard are shared with tests/test_agent_decisions_mcp.py."""
import errno
import json
import os
import threading
import zipfile
from pathlib import Path

import pytest

from agent_canned import TELESCOPE, canned_answer
from test_agent_decisions_mcp import (EXPECTED_ROLES, LABEL, at_interpret, call, canned, drive, no_cloud,  # noqa: F401
                                      pending, read, request_text, responses, run_json, servers,  # noqa: F401
                                      start_agent, step, submit, team, wait_idle)
from optics_graph import agent_decisions, algorithm_experiment, design_agent, mcp_evidence, mcp_runs
from optics_graph.agent_decisions import AgentDecisionGateway
from optics_graph.algorithm_sim import ANGLE_LIMIT, POSITION_LIMIT, AlgorithmServices
from optics_graph.runtime import RunInput, Runtime
from optics_graph.team_artifacts import LIMIT, encode


# ---- H1: a stop/pause requested while a node runs must survive DecisionPending --------------------------------
class Hold:
    """Holds the gateway call of one role (once) until released: the node is deterministically 'running'."""

    def __init__(self, monkeypatch, role):
        self.reached, self.release, self.fired = threading.Event(), threading.Event(), False
        real = AgentDecisionGateway._request

        def held(gateway, r, prompt, schema, attempt=1):
            if r == role and not self.fired:
                self.fired = True
                self.reached.set()
                assert self.release.wait(60), "test never released the held node"
            return real(gateway, r, prompt, schema, attempt)
        monkeypatch.setattr(AgentDecisionGateway, "_request", held)

    def wait(self):
        assert self.reached.wait(60), "the node never reached its decision"

    def let_go(self, runs):
        self.release.set()
        wait_idle(runs)


@pytest.mark.parametrize("action", ["stop", "pause"])
@pytest.mark.parametrize("position", ["first_decision", "between_validator_decisions"])
def test_h1_control_requested_while_the_node_runs_wins_over_awaiting_decision(servers, monkeypatch, action, position):
    runs, evidence, data = servers
    run_id = start_agent(runs)["run_id"]
    if position == "first_decision":
        answered = pending(runs, run_id)
        hold = Hold(monkeypatch, "design")
        assert submit(runs, run_id, answered, json.dumps(canned(answered)))["accepted"] is True
        call(runs, "continue_run", {"run_id": run_id})  # design now runs, held at its decision
        node, role = "act", "design"
    else:
        drive(runs, run_id, until_role="optical_tests")
        hold = Hold(monkeypatch, "review")
        answered = pending(runs, run_id)
        assert submit(runs, run_id, answered, json.dumps(canned(answered)))["accepted"] is True
        call(runs, "continue_run", {"run_id": run_id})  # validator re-runs: optical_tests consumed, review held
        node, role = "validator", "review"
    hold.wait()
    assert run_json(data, run_id)["status"] != "awaiting_decision" or action == "stop"
    assert call(runs, f"{action}_run", {"run_id": run_id})["status"] == f"{action}_requested"
    hold.let_go(runs)

    state = run_json(data, run_id)
    ended = "stopped" if action == "stop" else "paused"
    assert state["status"] == ended and state["next_node"] == node and "pending_decision" not in state
    folder = team(data, run_id) / f"{node}-01"
    request_file = folder / f"decision-request-{role}.json"
    assert request_file.is_file()  # the request stays as evidence
    request_id = read(request_file)["request_id"]
    assert pending(runs, run_id) == {"run_id": run_id, "pending": False, "status": ended}
    probe = {"request_id": request_id}
    assert "awaiting_decision" in submit(runs, run_id, probe, "{}", expect_error=True)
    assert not (folder / f"decision-response-{role}.json").exists()
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["journals_ok"] is True

    if action == "stop":
        call(runs, "continue_run", {"run_id": run_id}, expect_error=True)
        wait_idle(runs)
        assert run_json(data, run_id)["status"] == "stopped"
        assert not (folder / f"decision-response-{role}.json").exists()
        return
    # paused: continue re-runs the node, which asks the same question again and lands in awaiting_decision
    assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(runs)
    again = run_json(data, run_id)
    assert again["status"] == "awaiting_decision" and again["pending_decision"]["request_id"] == request_id
    p = pending(runs, run_id)
    assert p["pending"] is True and p["request_id"] == request_id and p["role"] == role
    drive(runs, run_id, evidence=evidence)
    assert run_json(data, run_id)["status"] == "algorithm_passed"


def test_h1_a_stop_before_the_very_first_decision_is_honoured(servers, monkeypatch):
    runs, _, data = servers
    hold = Hold(monkeypatch, "interpret")
    run_id = call(runs, "start_run", {"client_request_id": "h1-first", "request": request_text(TELESCOPE),
                                      "decision_mode": "agent"})["run_id"]
    hold.wait()
    assert call(runs, "stop_run", {"run_id": run_id})["status"] == "stop_requested"
    hold.let_go(runs)
    assert run_json(data, run_id)["status"] == "stopped"
    assert (team(data, run_id) / "prompt-01/decision-request-interpret.json").is_file()


# ---- H2a: tolerances ---------------------------------------------------------------------------------------------
def _quotes(*texts):
    return [{"id": f"r{i}", "text": t, "source_quote": t} for i, t in enumerate(texts)]


def _intent(position=None, angle=None, *texts):
    return {"requirements": _quotes(*(texts or ("Build a thing",))), "position_tolerance_m": position,
            "angle_tolerance_rad": angle}


@pytest.mark.parametrize("position, angle, quotes, ok", [
    (None, None, ("anything",), True),                                   # absent: the defaults apply
    (POSITION_LIMIT, ANGLE_LIMIT, ("anything",), True),                  # equal to the default
    (0.0005, 0.001, ("anything",), True),                                # tighter than default
    (5.0, None, ("anything",), False),                                   # the reproduced loosening
    (None, 3.0, ("anything",), False),
    (5.0, 3.0, ("Place within 5 mm",), False),                           # 5 mm is 0.005 m, not 5.0 m
    (0.005, None, ("Place within 5 mm",), True),                         # unit converted to the field's unit
    (0.005, None, ("Place within 5mm",), True),
    (0.005, None, ("Place within 0.005 m",), True),
    (0.0015, None, ("tolerance 1.5 mm",), True),
    (0.0000025, None, ("tolerance 2.5 um",), True),
    (0.005, None, ("Place within 15 mm",), False),                       # 15 is not 5 (token boundary)
    (5.0, None, ("Use 5 components",), False),                           # a bare number never counts
    (0.005, None, ("Place within 5 mmx",), False),
    (0.005, None, ("Lens f 4 mm", "Place within 5 mm"), True),           # any requirement's quote may carry it
    (0.005, None, ("Place within 4 mm",), False),                        # a different number is not support
    (None, 0.0349066, ("angle within 2 deg",), True),
    (None, 0.0349066, ("angle within 2°",), True),
    (None, 0.001, ("no angle quoted",), True),                           # tighter than default: always allowed
    (None, 0.02, ("angle within 20 mrad",), True),
    (None, 0.02, ("angle within 20 mm",), False),                        # wrong unit family
])
def test_h2a_looser_than_default_tolerances_need_a_quote_that_states_the_value(position, angle, quotes, ok):
    errors = agent_decisions.tolerance_errors("interpret", _intent(position, angle, *quotes))
    if ok:
        assert errors == []
    else:
        assert errors and all("Omit it" in e and "source_quote" in e for e in errors), errors


def test_h2a_the_rule_is_for_interpret_only_and_openai_postcheck_is_unchanged():
    assert agent_decisions.tolerance_errors("design", {"requirements": [], "position_tolerance_m": 5.0}) == []
    # the shared post-validation (what the OpenAI path runs) still accepts what the agent path now refuses
    loose = {"title": "t", "objective": "o", "requirements": _quotes("Build a thing"), "assumptions": [],
             "clarification_questions": [], "position_tolerance_m": 5.0, "angle_tolerance_rad": 3.0}
    design_agent.postcheck("interpret", loose, {"request": "Build a thing"})


@pytest.mark.parametrize("loose", [{"position_tolerance_m": 5.0}, {"angle_tolerance_rad": 3.0},
                                   {"position_tolerance_m": 5.0, "angle_tolerance_rad": 3.0}])
def test_h2a_submit_rejects_a_loosened_tolerance_and_names_the_way_out(servers, loose):
    runs, _, data, run_id, p = at_interpret(servers)
    before = sorted(str(f) for f in team(data, run_id).rglob("*") if f.is_file())
    out = submit(runs, run_id, p, json.dumps({**canned(p), **loose}))
    assert out["accepted"] is False and out["rejections_remaining"] == 2
    text = " ".join(out["errors"])
    assert "Omit it" in text and "source_quote" in text and next(iter(loose)) in text
    assert responses(data, run_id) == []
    added = set(str(f) for f in team(data, run_id).rglob("*") if f.is_file()) - set(before)
    assert {Path(a).name for a in added} == {"decision-rejections-interpret.json"}


def test_h2a_submit_accepts_tighter_default_and_quoted_values(servers):
    runs, _, data = servers
    for i, value in enumerate((0.0005, POSITION_LIMIT, 0.05)):  # 0.05 m = "50 mm", which TELESCOPE quotes
        run_id = start_agent(runs, f"tol-{i}")["run_id"]
        p = pending(runs, run_id)
        out = submit(runs, run_id, p, json.dumps({**canned(p), "position_tolerance_m": value}))
        assert out["accepted"] is True, (value, out)


def test_h2a_a_hand_written_response_with_a_loosened_tolerance_is_refused_at_consume(servers):
    import hashlib
    runs, evidence, data, run_id, p = at_interpret(servers)
    text = json.dumps({**canned(p), "position_tolerance_m": 5.0, "angle_tolerance_rad": 3.0})
    response = {"request_id": p["request_id"], "decision_text": text,
                "decision_sha256": hashlib.sha256(text.encode()).hexdigest(), "decider_label": "hand-written",
                "submitted_at": "2026-01-01T00:00:00+00:00", "backend": "standalone"}
    folder = team(data, run_id) / "prompt-01"
    (folder / "decision-response-interpret.json").write_text(json.dumps(response))
    assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(runs)
    state = run_json(data, run_id)
    assert state["status"] == "failed" and state["error"]["code"] == "DECISION_RESPONSE_INVALID"
    assert not (folder / "model-interpret-1.json").exists() and not (team(data, run_id) / "act-01").exists()


# ---- H2b: visibility in the acceptance summary --------------------------------------------------------------------
LABELS = {"interpret": "alpha", "design": "alpha", "optical_tests": "beta", "review": "alpha",
          "repair_controller": "gamma", "repair_act": "gamma"}


def drive_labelled(runs, run_id, labels, quotes=TELESCOPE, mutate=None):
    for _ in range(40):
        p = pending(runs, run_id)
        if not p["pending"]:
            return
        answer = canned_answer(p["role"], p["data"], quotes)
        if mutate:
            answer = mutate(p, answer)
        assert submit(runs, run_id, p, json.dumps(answer), labels[p["role"]])["accepted"] is True
        call(runs, "continue_run", {"run_id": run_id})
        wait_idle(runs)
    raise AssertionError("did not terminate")


def test_h2b_acceptance_summary_shows_provenance_limits_and_test_tolerances(servers):
    runs, evidence, data = servers
    run_id = start_agent(runs, "h2b")["run_id"]
    drive_labelled(runs, run_id, LABELS)
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert s["provider"] == "agent"
    assert s["provenance"] == "agent-decided; decider identity self-reported and unverified"
    assert s["decider_labels"] == {"interpret": ["alpha"], "design": ["alpha"], "optical_tests": ["beta"],
                                   "review": ["alpha"], "repair_act": ["gamma"], "repair_controller": ["gamma"]}
    assert s["design_and_review_share_a_label"] is True and "alpha" in s["same_label_warning"]
    limits = s["execution_limits"]
    assert limits["defaults"] == {"position_m": POSITION_LIMIT, "angle_rad": ANGLE_LIMIT}
    assert limits["used"] == limits["defaults"] and limits["equals_defaults"] is True
    assert limits["used_source"].startswith("sim-") and limits["declared_by_experiment"] == limits["defaults"]
    tests = {t["id"]: t for t in s["optical_tests"]["tests"]}
    assert set(tests) == {"linked", "objective-focal", "eyepiece-focal", "separation"}
    row = tests["separation"]
    assert (row["expected"], row["absolute_tolerance"], row["status"]) == (0.25, 1e-8, "passed")
    assert row["actual"] == pytest.approx(0.25) and row["tolerance_relative_to_expected"] == pytest.approx(4e-8)
    assert s["optical_tests"]["largest_tolerance_relative_to_expected"]["id"] == "separation"
    assert s["optical_tests"]["source"].startswith("output-")


def test_h2b_different_labels_for_design_and_review_clear_the_flag(servers):
    runs, evidence, data = servers
    run_id = start_agent(runs, "h2b-diff")["run_id"]
    drive_labelled(runs, run_id, {**LABELS, "review": "reviewer-b"})
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert s["design_and_review_share_a_label"] is False and "same_label_warning" not in s


def test_h2b_a_quoted_loose_tolerance_is_visible_as_not_the_default(servers):
    runs, evidence, data = servers
    run_id = start_agent(runs, "h2b-loose")["run_id"]
    drive_labelled(runs, run_id, LABELS, mutate=lambda p, a: {**a, "position_tolerance_m": 0.05}
                   if p["role"] == "interpret" else a)  # 0.05 m is quoted as "50 mm"
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert s["execution_limits"]["used"]["position_m"] == 0.05
    assert s["execution_limits"]["equals_defaults"] is False and s["execution_limits"]["note"] == "looser than defaults"


def test_h2b_fixture_runs_report_their_provider_and_defaults(servers):
    runs, evidence, data = servers
    out = call(runs, "start_run", {"client_request_id": "fx", "request": "Assemble a 15-component complete Michelson interferometer."})
    wait_idle(runs)
    s = call(evidence, "get_acceptance_summary", {"run_id": out["run_id"]})
    assert s["provider"] == "replay" and s["provenance"] is None and s["decider_labels"] is None
    assert s["execution_limits"]["equals_defaults"] is True


def test_h2b_values_come_from_verified_journal_files_only(servers):
    runs, evidence, data = servers
    run_id = start_agent(runs, "h2b-tamper")["run_id"]
    drive_labelled(runs, run_id, LABELS)
    report = sorted((team(data, run_id)).glob("output-*/optical-test-report.json"))[-1]
    suite = json.loads(report.read_text())
    suite["tests"][0]["absolute_tolerance"] = 1000.0
    report.write_text(json.dumps(suite))
    label_file = sorted(team(data, run_id).glob("act-*/decision-response-design.json"))[0]
    label_file.write_text(label_file.read_text().replace(LABELS["design"], "forged"))
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert s["optical_tests"]["tests"] == [] and "forged" not in json.dumps(s["decider_labels"])
    assert any("failed journal verification" in n for n in s["provenance_notes"])
    assert s["evidence_verified"] is False


def test_h2b_a_clean_finished_agent_run_has_no_provenance_notes(servers):
    runs, evidence, data = servers
    run_id = start_agent(runs, "h2b-clean")["run_id"]
    drive_labelled(runs, run_id, LABELS)
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert s["evidence_verified"] is True and call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is True
    assert "provenance_notes" not in s  # absent decision files (they belong to other nodes) are not notes
    assert set(s["decider_labels"]) == set(LABELS)


def test_h2b_a_tampered_listed_decision_file_is_noted_and_its_label_omitted(servers):
    runs, evidence, data = servers
    run_id = start_agent(runs, "h2b-one")["run_id"]
    drive_labelled(runs, run_id, LABELS)
    victim = sorted(team(data, run_id).glob("act-*/decision-response-design.json"))[0]
    victim.write_text(victim.read_text().replace(LABELS["design"], "forged"))
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert len(s["provenance_notes"]) == 1
    assert s["provenance_notes"][0].startswith(f"{victim.parent.name}/decision-response-design.json: not used")
    assert "design" not in s["decider_labels"] and "forged" not in json.dumps(s["decider_labels"])
    assert s["decider_labels"]["review"] == ["alpha"]


# ---- H2: strict xfail documenting the open human decision ---------------------------------------------------------
def _wrong_design_with_absurd_tolerance(runs, evidence, data):
    """The eyepiece is placed 0.75 m from the objective; the agent-authored test expects 0.25 m but allows 1000."""
    def mutate(p, answer):
        if p["role"] == "design":
            answer["components"][1]["position_m"][0] = .95
        if p["role"] == "optical_tests":
            row = next(t for t in answer["tests"] if t["id"] == "separation")
            row["absolute_tolerance"] = 1000.0
        return answer
    run_id = start_agent(runs, "absurd")["run_id"]
    drive_labelled(runs, run_id, LABELS, mutate=mutate)
    return run_id, run_json(data, run_id)


def test_h2_premise_an_absurd_agent_authored_tolerance_currently_passes_a_wrong_design(servers):
    runs, evidence, data = servers
    run_id, state = _wrong_design_with_absurd_tolerance(runs, evidence, data)
    row = next(t for t in call(evidence, "get_acceptance_summary", {"run_id": run_id})["optical_tests"]["tests"]
               if t["id"] == "separation")
    assert (row["expected"], row["actual"], row["absolute_tolerance"], row["status"]) == (0.25, pytest.approx(.75), 1000.0, "passed")
    assert state["status"] == "algorithm_passed"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="open human decision: the optics owner must define test-tolerance bounds. An "
                          "agent-authored test with absolute_tolerance 1000 on expected 0.25 still lets a wrong design pass")
def test_h2_a_wrong_design_must_not_pass_on_an_absurd_test_tolerance(servers):
    runs, evidence, data = servers
    _, state = _wrong_design_with_absurd_tolerance(runs, evidence, data)
    assert state["status"] != "algorithm_passed"


# ---- M1 ----------------------------------------------------------------------------------------------------------
def test_m1_agent_runs_are_labelled_agent_design_and_openai_stays_openai_design(servers):
    runs, evidence, data = servers
    run_id = start_agent(runs, "m1")["run_id"]
    drive(runs, run_id)
    assert read(team(data, run_id) / "act-01/experiment.json")["source"] == "agent_design"
    assert run_json(data, run_id)["algorithm_experiment"]["source"] == "agent_design"
    intent = canned_answer("interpret", None, TELESCOPE)
    design = canned_answer("design", None, TELESCOPE)
    default = algorithm_experiment.package_design("req", intent, design)
    assert default[2]["source"] == "openai_design"
    assert algorithm_experiment.package_design("req", intent, design, "openai")[2] == default[2]
    agent = algorithm_experiment.package_design("req", intent, design, "agent")[2]
    assert agent["source"] == "agent_design" and {**agent, "source": "openai_design"}.keys() == default[2].keys()


def test_m1_the_plan_time_binding_check_fires_for_agent_runs(servers):
    runs, _, data = servers
    run_id = start_agent(runs, "m1-bind")["run_id"]
    drive(runs, run_id)
    state = run_json(data, run_id)
    assert state["algorithm_experiment"]["source"] == "agent_design"
    base = data / "runs" / run_id

    class Reject(Exception):
        def __init__(self, code, message, target=None, details=None):
            super().__init__(message)

    def plan(mutate):
        copy = json.loads(json.dumps(state))
        mutate(copy)
        return AlgorithmServices(base, Reject).plan(copy)

    with pytest.raises(ValueError, match="agent design or the requirements have changed"):
        plan(lambda s: s["layout"]["tolerance"].update(position_m=s["layout"]["tolerance"]["position_m"] * 2))
    with pytest.raises(ValueError, match="agent design or the requirements have changed"):
        plan(lambda s: s["algorithm_design"].update(rationale="changed after packaging"))
    with pytest.raises(ValueError, match="agent design or the requirements have changed"):
        plan(lambda s: s["algorithm_intent"].update(objective="changed after packaging"))


# ---- M2 ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("route", ["/api/runs", "/v1/pipelines/agent-loop/runs"])
def test_m2_http_routes_refuse_to_create_agent_runs(tmp_path, route):
    from fastapi.testclient import TestClient
    from optics_graph import server
    body = {"request": request_text(TELESCOPE), "provider": "agent", "profile": "auto", "mode": "github_local",
            "simulation_backend": "algorithm_v1"}
    with TestClient(server.create_app(Runtime(tmp_path))) as client:
        response = client.post(route, json=body, headers={"x-optics-local": "1"})
        assert 400 <= response.status_code < 500, response.text
        assert "start_run" in response.text and "decision_mode" in response.text
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())


def test_m2_the_mcp_refusal_no_longer_points_at_a_panel_that_cannot_help(servers):
    runs, _, data = servers
    panel = runs._optics_service.runtime.create(RunInput(request=request_text(TELESCOPE), profile="auto", mode="github_local",
                                                         simulation_backend="algorithm_v1", provider="replay"))
    text = call(runs, "continue_run", {"run_id": panel}, expect_error=True)
    assert "not created through start_run" in text and "use the web panel" not in text
    assert "panel-created" in text


# ---- M3 / L1 -----------------------------------------------------------------------------------------------------
def test_m3_pending_decision_exists_exactly_while_awaiting_decision(servers):
    runs, evidence, data = servers
    run_id = start_agent(runs, "m3")["run_id"]
    for _ in range(40):
        state = run_json(data, run_id)
        assert ("pending_decision" in state) == (state["status"] == "awaiting_decision"), state["status"]
        if state["status"] != "awaiting_decision":
            break
        assert "submitted" not in state["pending_decision"]
        step(runs, run_id)
    assert run_json(data, run_id)["status"] == "algorithm_passed"


def test_m3_get_run_computes_submitted_from_the_files(servers):
    runs, evidence, data, run_id, p = at_interpret(servers)
    assert call(evidence, "get_run", {"run_id": run_id})["pending_decision"]["submitted"] is False
    submit(runs, run_id, p, json.dumps(canned(p)))
    shown = call(evidence, "get_run", {"run_id": run_id})["pending_decision"]
    assert shown["submitted"] is True and shown["request_id"] == p["request_id"]


def test_m3_stop_at_awaiting_decision_clears_the_pending_request_and_never_regresses_the_file(servers, monkeypatch):
    runs, _, data, run_id, p = at_interpret(servers)
    runtime = runs._optics_service.runtime
    assert "pending_decision" in run_json(data, run_id)
    call(runs, "stop_run", {"run_id": run_id})
    stopped = run_json(data, run_id)
    assert stopped["status"] == "stopped" and "pending_decision" not in stopped

    # control() must read run.json UNDER the lock: a newer publish that lands while it waits survives
    data2 = servers[2]
    run2 = start_agent(runs, "m3-lock")["run_id"]
    attempting, real = threading.Event(), Runtime._control_lock

    def spying(self, rid):
        attempting.set()
        return real(self, rid)
    monkeypatch.setattr(Runtime, "_control_lock", spying)
    worker = threading.Thread(target=lambda: runtime.control(run2, "stop"))
    with real(runtime, run2):
        worker.start()
        assert attempting.wait(30)
        state = runtime.get(run2)  # what a resumed worker would publish meanwhile
        (data2 / "runs" / run2 / "run.json").write_text(json.dumps({**state, "marker": "newer-publish"}))
    worker.join(30)
    final = run_json(data2, run2)
    assert final["status"] == "stopped" and final.get("marker") == "newer-publish"


def test_m3_pause_at_awaiting_decision_says_what_is_going_on(servers):
    runs, _, data, run_id, p = at_interpret(servers)
    text = call(runs, "pause_run", {"run_id": run_id}, expect_error=True)
    assert "finished" not in text and "waiting for an agent decision" in text and "Nothing is executing" in text


def test_m3_panel_labels_the_waiting_state_and_does_not_show_it_as_an_outcome():
    html = (Path(mcp_runs.__file__).parent / "panel.html").read_text()
    assert 'awaiting_decision:"Waiting for an agent decision"' in html
    assert "'awaiting_review','awaiting_decision','pause_requested'" in html  # not a finished "Run outcome" row
    assert 'agent_design:ui("Agent design")' in html


# ---- L2 ----------------------------------------------------------------------------------------------------------
def test_l2_deleting_the_rejections_file_resets_the_budget_but_never_the_validation(servers):
    """Decision: accepted and documented (see agent_decisions module docstring). The 3-try budget bounds a looping
    client, it is not a security control: someone who can delete files in the journal can already replace anything
    the run reads. Resetting it only buys more tries to submit an answer that must still pass full validation, and
    never touches retry_counts."""
    runs, _, data, run_id, p = at_interpret(servers)
    for _ in range(2):
        submit(runs, run_id, p, '{"title": 1}')
    assert pending(runs, run_id)["rejections_remaining"] == 1
    (team(data, run_id) / "prompt-01/decision-rejections-interpret.json").unlink()
    assert pending(runs, run_id)["rejections_remaining"] == 3
    out = submit(runs, run_id, p, json.dumps({**canned(p), "position_tolerance_m": 5.0}))
    assert out["accepted"] is False  # still fully validated
    assert not run_json(data, run_id).get("retry_counts")


# ---- L3 ----------------------------------------------------------------------------------------------------------
def test_l3_a_stale_model_log_is_never_copied_or_trusted_before_a_decision_is_consumed(servers):
    runs, _, data, run_id, p = at_interpret(servers)
    folder = team(data, run_id) / "prompt-01"
    forged = {"role": "interpret", "attempt": 1, "model": "forged-model", "response_id": "f" * 64,
              "provider": "agent", "decided_by": {"label": "forged", "verified": True}, "content": canned(p)}
    (folder / "model-interpret-1.json").write_text(json.dumps(forged))
    gateway = AgentDecisionGateway("agent", folder)
    with pytest.raises(agent_decisions.DecisionPending):  # no response stored: the gateway raises
        design_agent.interpret(gateway, p["data"]["request"], p["data"]["observation"], p["data"]["feedback"], 1)
    assert not (folder / "model-interpret-1.json").exists()
    assert not (folder / "model-response.json").exists()  # _decide's finally had nothing to copy


# ---- L4 ----------------------------------------------------------------------------------------------------------
def test_l4_oversized_interpret_answers_are_rejected_at_submit_with_the_workflow_limit(servers):
    runs, _, data, run_id, p = at_interpret(servers)
    big = {**canned(p), "assumptions": ["x" * 90_000] * 3}  # about 270 KB: accepted by the 400 kB submit cap
    assert len(json.dumps(big)) < agent_decisions.MAX_DECISION_CHARS
    out = submit(runs, run_id, p, json.dumps(big))
    assert out["accepted"] is False and str(LIMIT) in out["errors"][0] and "too large" in out["errors"][0]
    assert responses(data, run_id) == []
    # an answer comfortably under the limit is still accepted
    ok = {**canned(p), "assumptions": ["x" * 40_000]}
    assert submit(runs, run_id, p, json.dumps(ok))["accepted"] is True


def test_l4_a_design_whose_packaged_experiment_exceeds_the_limit_is_rejected_at_submit(servers):
    runs, _, data = servers
    run_id = start_agent(runs, "l4-design")["run_id"]
    step(runs, run_id)
    p = pending(runs, run_id)
    design = canned(p)
    pad = 0
    while len(encode({**design, "rationale": "r" * pad}, limit=None)) < LIMIT - 300:
        pad += 100
    design["rationale"] = "r" * pad
    assert len(encode(design, limit=None)) <= LIMIT  # the design alone fits: only the packaged artifacts do not
    out = submit(runs, run_id, p, json.dumps(design))
    assert out["accepted"] is False and "too large" in out["errors"][0] and str(LIMIT) in out["errors"][0]
    assert responses(data, run_id) == ["decision-response-interpret.json"]


def test_l4_a_test_suite_whose_report_exceeds_the_limit_is_rejected_at_submit(servers):
    from optics_graph.optical_tests import evaluate_suite
    runs, _, data = servers
    run_id = start_agent(runs, "l4-tests")["run_id"]
    drive(runs, run_id, until_role="optical_tests")
    p = pending(runs, run_id)
    suite = canned(p)
    template = dict(suite["tests"][-1], rationale="x" * 1000)
    n = 0
    while True:
        n += 1
        suite["tests"].append({**template, "id": f"pad-{n}", "title": f"pad {n}"})
        if len(encode(suite, limit=None)) > LIMIT - 5000:
            break
    report = encode(evaluate_suite(suite, p["data"]["intent"], p["data"]["design"]), limit=None)
    assert len(encode(suite, limit=None)) <= LIMIT < len(report)  # only the computed report is too big
    out = submit(runs, run_id, p, json.dumps(suite))
    assert out["accepted"] is False and "report" in out["errors"][0] and str(LIMIT) in out["errors"][0]


# ---- L5 ----------------------------------------------------------------------------------------------------------
def _no_hard_links(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError(errno.EPERM, "hard links are not supported here")
    monkeypatch.setattr(os, "link", refuse)


def test_l5_exclusive_create_falls_back_without_hard_links_and_stays_exclusive(tmp_path, monkeypatch):
    _no_hard_links(monkeypatch)
    target = tmp_path / "decision-response-interpret.json"
    outcomes, start = [], threading.Barrier(8)

    def writer(i):
        start.wait()
        try:
            agent_decisions.exclusive_write(target, f"writer-{i}".encode())
            outcomes.append("won")
        except FileExistsError:
            outcomes.append("lost")
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(outcomes) == ["lost"] * 7 + ["won"] and target.read_bytes().startswith(b"writer-")
    assert [p.name for p in tmp_path.iterdir()] == [target.name]


def test_l5_a_failed_fallback_write_leaves_neither_the_final_name_nor_a_temporary(tmp_path, monkeypatch):
    _no_hard_links(monkeypatch)
    target = tmp_path / "decision-request-design.json"
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "disk")))
    with pytest.raises(OSError):
        agent_decisions.exclusive_write(target, b"{}")
    assert list(tmp_path.iterdir()) == []


def test_l5_temporaries_never_live_in_the_journal_directory(servers, monkeypatch):
    runs, _, data, run_id, p = at_interpret(servers)
    seen = []
    real = os.link
    monkeypatch.setattr(os, "link", lambda src, dst, **kw: (seen.append(Path(src).parent), real(src, dst, **kw))[1])
    assert submit(runs, run_id, p, json.dumps(canned(p)))["accepted"] is True
    assert seen == [data / "runs" / run_id]  # the run directory, never team/<node>-NN
    assert not list((data / "runs" / run_id).rglob(".tmp-decision-*"))


def test_l5_a_whole_run_works_on_a_filesystem_without_hard_links(servers, monkeypatch):
    runs, evidence, data = servers
    _no_hard_links(monkeypatch)
    run_id = start_agent(runs, "nolink")["run_id"]
    drive(runs, run_id, evidence=evidence)
    assert run_json(data, run_id)["status"] == "algorithm_passed"
    assert not list((data / "runs" / run_id).rglob(".tmp-decision-*"))
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is True
