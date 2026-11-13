"""Agent-decided runs behave like the OpenAI stand-in runs beyond the happy path.

Each scenario of tests/test_agent_algorithm.py is run twice with the SAME answer alteration: once through the
OpenAI stand-in (only the model transport replaced) and once in agent mode over the real MCP tool surface with
the scripted client of tests/test_agent_decisions_mcp.py. The routes, error codes, retry counts and the sequence
of decisions must agree. Differences that follow from the agent transport are asserted explicitly, never
ignored. No cloud client can be built and no settings or key file is read.
"""
import json
from copy import deepcopy

import pytest

import test_agent_algorithm as standin
from agent_canned import TELESCOPE
from optics_graph import agent_algorithm, mcp_evidence, mcp_runs
from test_agent_decisions_mcp import call, canned, no_cloud, pending, run_json, start_agent, submit, wait_idle  # noqa: F401


def play(runs, run_id, alter, quotes=TELESCOPE, limit=80):
    """Scripted agent: answer every request with the canned answer passed through `alter(role, node_attempt,
    answer)` (the same hook the stand-in uses). A rejected answer is resubmitted unchanged until the request is
    exhausted, then continue_run lets the graph charge its budget. Returns [(role, node_attempt, data)]."""
    asked = []
    for _ in range(limit):
        p = pending(runs, run_id)
        if not p["pending"]:
            return asked
        asked.append((p["role"], p["node_attempt"], p["data"]))
        answer = canned(p, quotes)
        answer = alter(p["role"], p["node_attempt"], answer) if alter else answer
        text = json.dumps(answer)
        while True:
            out = submit(runs, run_id, p, text)
            if out["accepted"] or out["exhausted"]:
                break
        assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
        wait_idle(runs)
    raise AssertionError("agent loop did not terminate")


def summary(state, asked):
    return {"status": state["status"], "error": (state.get("error") or {}).get("code"),
            "routes": [(r["from"], r["to"], r["code"], r.get("reason")) for r in state["retry_log"]],
            "retry_counts": state["retry_counts"], "roles": [(role, attempt) for role, attempt, _ in asked]}


def both(tmp_path, monkeypatch, alter=None, quotes=TELESCOPE):
    """(stand-in summary, agent summary, stand-in calls, agent calls, agent state, agent run id)."""
    state, _, calls, _ = standin.run(tmp_path / "standin", monkeypatch, quotes, alter)
    left = summary(state, [(r, a, d) for r, a, d in calls])
    data = tmp_path / "agent"
    data.mkdir()
    runs, evidence = mcp_runs.build_server(data), mcp_evidence.build_server(data)
    run_id = start_agent(runs, "parity", quotes)["run_id"]
    asked = play(runs, run_id, alter, quotes)
    final = run_json(data, run_id)
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is True
    return left, summary(final, asked), calls, asked, final, run_id


def test_rejected_review_routes_to_design_like_the_standin(tmp_path, monkeypatch):
    def alter(role, attempt, candidate):
        if role == "design" and attempt == 1:
            candidate["rationale"] = "The telescope magnification is not explained."
        if role == "review" and attempt == 1:
            return {"accepted": False, "issues": ["The telescope magnification is not explained"],
                    "return_to": "act", "problem_source": "design",
                    "reasoning": "The design notes do not state the magnification."}
        return candidate

    left, right, calls, asked, state, _ = both(tmp_path, monkeypatch, alter)
    assert left == right
    assert left["status"] == "algorithm_passed"
    assert [r for r, _ in right["roles"]][:7] == ["interpret", "design", "optical_tests", "review", "design",
                                                  "optical_tests", "review"]
    assert asked[4][2]["feedback"][-1]["code"] == "AGENT_DESIGN_REJECTED"
    for key in ("code", "message", "stage"):  # the hashes inside details bind the provider label, so differ
        assert asked[4][2]["feedback"][-1][key] == calls[4][2]["feedback"][-1][key]
    assert asked[4][2]["feedback"][-1]["details"]["issues"] == calls[4][2]["feedback"][-1]["details"]["issues"]


def test_unapproved_neighbour_repair_fails_like_the_standin(tmp_path, monkeypatch):
    def alter(role, attempt, candidate):
        if role == "repair_controller":
            candidate["object_id"] = "EYE"
        return candidate

    left, right, calls, asked, state, run_id = both(tmp_path, monkeypatch, alter)
    # Same outcome and same repair scope; the transport differs in WHERE the bad answer is refused:
    # the stand-in refuses it inside the node (CandidateFormatError), agent mode already at submit_decision
    # (three rejections exhaust the request, then the graph charges AGENT_DECISION_INVALID).
    assert left["status"] == right["status"] == "failed"
    assert [h["revision"] for h in state["algorithm_history"]] == [1]
    assert all(d["allowed_ids"] == ["OBJ"] for r, _, d in asked if r == "repair_controller")
    assert state["algorithm_state"]["current"] == state["algorithm_history"][0]["after"]
    assert "execution" not in state["attempts"]
    assert not any(role == "repair_act" for role, _ in right["roles"])
    # the retry counters charged are the same ones the stand-in charges
    assert left["retry_counts"] == right["retry_counts"]
    assert left["routes"][-1][:2] == right["routes"][-1][:2] == ("controller", "stop")
    assert left["routes"][-1][3] == right["routes"][-1][3]
    assert left["roles"] == right["roles"]
    # DIFFERENCE (documented, by design): the error code. The stand-in surfaces the exception class name of the
    # in-node refusal, agent mode the explicit request-exhausted code; both are the same CandidateFormatError route.
    assert (left["error"], right["error"]) == ("CandidateFormatError", "AGENT_DECISION_INVALID")
    assert [r[:2] + r[3:] for r in left["routes"]] == [r[:2] + r[3:] for r in right["routes"]]
    assert [r[2] for r in left["routes"]][0] == [r[2] for r in right["routes"]][0] == "ALGORITHM_POSITION_MISMATCH"
    assert {r[2] for r in left["routes"][1:]} == {"CandidateFormatError"}
    assert {r[2] for r in right["routes"][1:]} == {"AGENT_DECISION_INVALID"}


def test_stale_repair_scope_is_refused_like_the_standin(tmp_path, monkeypatch):
    services = agent_algorithm.AgentAlgorithmServices
    original = services.local_repair

    def with_stale_scope(self, node, state, path, attempt, expected):
        feedback = deepcopy(state["feedback"])
        diagnostic = feedback[-1]
        assert diagnostic["details"]["world_revision"] == state["algorithm_state"]["revision"]
        diagnostic["details"]["world_revision"] -= 1
        return original(self, node, {**state, "feedback": feedback}, path, attempt, expected)

    monkeypatch.setattr(services, "local_repair", with_stale_scope)
    left, right, _, asked, state, _ = both(tmp_path, monkeypatch)
    assert left == right
    assert right["status"] == "failed" and right["error"] == "REPAIR_SCOPE_UNPROVEN"
    assert [role for role, _ in right["roles"]] == ["interpret", "design", "optical_tests", "review"]
    assert state["algorithm_state"]["revision"] == 1


def test_failed_analytic_check_returns_to_design_like_the_standin(tmp_path, monkeypatch):
    def alter(role, attempt, candidate):
        if role == "design" and attempt == 1:
            candidate["components"][1]["position_m"][0] = .46
        return candidate

    left, right, calls, asked, state, _ = both(tmp_path, monkeypatch, alter)
    assert left == right and left["status"] == "algorithm_passed"
    assert asked[3][2]["test_report"]["passed"] is False
    failure = asked[4][2]["feedback"][-1]
    assert failure["code"] == "OPTICAL_TEST_FAILED" == calls[4][2]["feedback"][-1]["code"]
    assert state["retry_log"][0]["to"] == "act"


@pytest.mark.parametrize("cause", ["invalid_expression", "review_rejected_tests", "review_rejected_failed_test"])
def test_test_failure_regenerates_the_suite_like_the_standin(tmp_path, monkeypatch, cause):
    def alter(role, attempt, candidate):
        if cause == "invalid_expression" and role == "optical_tests" and attempt == 1:
            candidate["tests"][-1]["expression"] = "unknown('OBJ','EYE')"
        if cause == "review_rejected_failed_test" and role == "optical_tests" and attempt == 1:
            candidate["tests"][-1]["expected"] = ".26"
        if cause.startswith("review_rejected") and role == "review" and attempt == 1:
            return {"accepted": False, "issues": ["The test basis needs to be completed"],
                    "return_to": "act", "problem_source": "tests", "reasoning": "The check basis is incomplete."}
        return candidate

    left, right, calls, asked, state, run_id = both(tmp_path, monkeypatch, alter)
    assert left == right and left["status"] == "algorithm_passed"
    assert [a for r, a in right["roles"] if r == "design"] == [1]
    assert [a for r, a in right["roles"] if r == "optical_tests"] == [1, 2]
    feedback = [d for r, a, d in asked if r == "optical_tests"][1]["feedback"][-1]
    assert feedback["code"] == "OPTICAL_TEST_INVALID"
    assert feedback["code"] == [d for r, a, d in calls if r == "optical_tests"][1]["feedback"][-1]["code"]


@pytest.mark.parametrize("scenario", [
    "overloaded test agent (RetryableGatewayError, MODEL_SERVICE_UNAVAILABLE, wait)",
    "provider timeout retries in graph",
    "truncated model output (MODEL_OUTPUT_TRUNCATED)",
])
def test_gateway_failure_scenarios_do_not_exist_in_agent_mode(scenario):
    pytest.skip(f"{scenario}: relies on a model gateway raising RetryableGatewayError (timeouts, overload, "
                "truncation). AgentDecisionGateway never raises it; its only failure routes are rejected "
                "submissions (AGENT_DECISION_INVALID) and stored-file refusals, covered by "
                "tests/test_agent_decisions_mcp.py")
