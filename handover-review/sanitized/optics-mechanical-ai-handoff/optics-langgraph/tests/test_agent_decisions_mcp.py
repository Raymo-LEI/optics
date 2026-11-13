"""Agent-decided runs: the graph pauses at every model decision and an MCP client answers it.

A scripted client plays the agent through the real MCP tool surface (in-memory sessions, as in
tests/test_mcp_wrap.py): get_pending_decision -> canned answer -> submit_decision -> continue_run.
No cloud client can be built and no settings/key file can be read.
"""
import asyncio
import hashlib
import json
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from zipfile import BadZipFile

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from agent_canned import MICHELSON, TELESCOPE, canned_answer
from optics_graph import agent_decisions, gateway, mcp_evidence, mcp_runs, settings, team_flow
from optics_graph.agent_decisions import DecisionPending
from optics_graph.runtime import RunInput, Runtime

LABEL = "scripted-agent"
EXPECTED_ROLES = ["interpret", "design", "optical_tests", "review", "repair_controller", "repair_controller",
                  "repair_act"]


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
    """OpenAI client construction raises, OPENAI_API_KEY is absent, and any settings read is recorded."""
    import openai
    seen = SimpleNamespace(constructed=[], settings_reads=[])

    def boom(*args, **kwargs):
        seen.constructed.append(1)
        raise AssertionError("cloud client constructed: an agent-decided run must never call a model")

    def keyless():
        seen.settings_reads.append(1)
        return {"model": "none", "api_key": "", "robot_endpoint": ""}

    monkeypatch.setattr(openai, "OpenAI", boom)
    monkeypatch.setattr(gateway, "OpenAI", boom)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("optics_graph") and hasattr(module, "read_settings"):
            monkeypatch.setattr(module, "read_settings", keyless)
    monkeypatch.setattr(settings, "read_settings", keyless)
    return seen


# ---- harness ------------------------------------------------------------------------------------------------
def call(server, name, args=None, *, expect_error=False):
    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            return await session.call_tool(name, args or {})
    result = asyncio.run(go())
    if expect_error:
        assert result.isError, f"{name}{args} should have been refused: {result.content[0].text if result.content else ''}"
        return result.content[0].text
    assert not result.isError, result.content[0].text
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


def list_tools(server):
    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            return (await session.list_tools()).tools
    return asyncio.run(go())


def wait_idle(runs, timeout=90):
    service = runs._optics_service
    end = time.time() + timeout
    while time.time() < end:
        with service.lock:
            if not service.active:
                return
        time.sleep(.02)
    raise AssertionError("worker did not finish")


def run_json(data, run_id):
    return json.loads((Path(data) / "runs" / run_id / "run.json").read_text())


def team(data, run_id):
    return Path(data) / "runs" / run_id / "team"


def read(path):
    return json.loads(Path(path).read_text())


def request_text(quotes):
    return "；".join(quotes)


@pytest.fixture
def servers(tmp_path):
    return mcp_runs.build_server(tmp_path), mcp_evidence.build_server(tmp_path), tmp_path


def start_agent(runs, cid="agent-1", quotes=TELESCOPE, **extra):
    out = call(runs, "start_run", {"client_request_id": cid, "request": request_text(quotes),
                                   "decision_mode": "agent", **extra})
    wait_idle(runs)
    return out


def pending(runs, run_id):
    return call(runs, "get_pending_decision", {"run_id": run_id})


def submit(runs, run_id, p, text, label=LABEL, *, expect_error=False):
    return call(runs, "submit_decision", {"run_id": run_id, "request_id": p["request_id"],
                                          "decision_json": text, "decider_label": label}, expect_error=expect_error)


def canned(p, quotes=TELESCOPE):
    return canned_answer(p["role"], p["data"], quotes)


def step(runs, run_id, quotes=TELESCOPE, *, mutate=None):
    """Answer the pending request and resume. Returns the request that was answered."""
    p = pending(runs, run_id)
    assert p["pending"] is True, p
    answer = canned(p, quotes)
    if mutate:
        answer = mutate(p, answer)
    out = submit(runs, run_id, p, json.dumps(answer))
    assert out["accepted"] is True and out["idempotent_replay"] is False, out
    assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(runs)
    return p


def drive(runs, run_id, quotes=TELESCOPE, *, evidence=None, until_role=None, mutate=None, limit=40):
    """Play the agent until the run stops asking (or until `until_role` is pending). Returns the answered requests."""
    stops = []
    for _ in range(limit):
        state = pending(runs, run_id)
        if not state["pending"]:
            return stops
        if evidence is not None:
            verdict = call(evidence, "verify_run_artifacts", {"run_id": run_id})
            assert verdict["ok"] is True, verdict["issues"]
        if state["role"] == until_role:
            return stops
        stops.append(step(runs, run_id, quotes, mutate=mutate))
    raise AssertionError("agent loop did not terminate")


def freeze(data, run_id):
    state = run_json(data, run_id)
    return {"status": state["status"], "retry_counts": state.get("retry_counts"), "retry_log": state.get("retry_log"),
            "pending": state.get("pending_decision"),
            "files": sorted(str(p.relative_to(team(data, run_id))) for p in team(data, run_id).rglob("*") if p.is_file())}


def responses(data, run_id):
    return sorted(p.name for p in team(data, run_id).rglob("decision-response-*.json"))


# ---- acceptance ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("quotes", [TELESCOPE, MICHELSON], ids=["telescope", "michelson"])
def test_agent_run_end_to_end_makes_no_model_call(servers, no_cloud, quotes):
    runs, evidence, data = servers
    started = start_agent(runs, "e2e", quotes)
    run_id = started["run_id"]
    assert started["source"].startswith("agent-decided run") and "unverified" in started["source"]
    first = run_json(data, run_id)
    assert first["provider"] == "agent" and first["status"] == "awaiting_decision"
    assert first["pending_decision"]["role"] == "interpret" and "submitted" not in first["pending_decision"]
    assert call(evidence, "get_run", {"run_id": run_id})["pending_decision"] == {**first["pending_decision"],
                                                                                 "submitted": False}

    stops = drive(runs, run_id, quotes, evidence=evidence)
    state = run_json(data, run_id)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert [s["role"] for s in stops] == EXPECTED_ROLES
    assert len(stops) == 7  # exactly one awaiting_decision stop per decision, none more
    assert [r["to"] for r in state["retry_log"]] == ["controller", "controller", "act"]
    part = state["retry_log"][0]["retry_key"].split(":", 1)[1]  # the part the simulator misplaced
    assert state["retry_counts"] == {f"controller:{part}": 2, f"act:{part}": 1}  # placement repairs count per part
    assert "pending_decision" not in state
    assert state["result"]["motions_sent"] == 0 and state["result"]["physical_verified"] is False
    assert state["agent_model"] == "external-agent"
    assert [(s["node"], s["node_attempt"]) for s in stops] == [
        ("prompt", 1), ("act", 1), ("validator", 1), ("validator", 1), ("controller", 2), ("controller", 3), ("act", 2)]

    folder = data / "runs" / run_id
    plan_bytes = (folder / state["plan_path"]).read_bytes()
    with zipfile.ZipFile(folder / state["artifact_path"]) as archive:
        assert archive.read("plan.json") == plan_bytes
    assert (folder / "team/execution-01/plan.json").read_bytes() == plan_bytes

    # provenance: an agent run never says openai anywhere in its directory (the design source is agent_design)
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        blobs = [path.read_bytes()]
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                blobs = [archive.read(n) for n in archive.namelist()]
        for blob in blobs:
            assert "openai" not in blob.decode("utf-8", "replace").lower(), path
    assert read(folder / "team/act-01/experiment.json")["source"] == "agent_design"
    sources = {}
    for path in sorted((folder / "team").glob("*/decision.json")):
        decision = read(path)
        sources[path.parent.name] = decision["source"]
        assert decision["model"] == "external-agent"
        if decision["source"] == "agent":
            assert decision["decided_by"] == {"label": LABEL, "verified": False}
            assert decision["response_id"] in {s["request_id"] for s in stops}
    assert sources["prompt-01"] == sources["act-01"] == sources["validator-01"] == "agent"
    assert sources["controller-02"] == sources["controller-03"] == sources["act-02"] == "agent"
    assert sources["controller-01"] == "validated_design_compiler"
    assert read(folder / "team/validator-01/validation.json")["source"] == "agent_and_contract"
    assert read(folder / "team/input-01/request.json")["design_backend"] == "agent"

    # the bindings that used to be openai-only are on for agent runs
    verification = state["verification"]
    assert verification["checks"]["conceptual_design"] is True
    assert verification["checks"]["analytic_design_tests"] is True
    assert verification["test_suite_sha256"] == state["algorithm_review"]["test_suite_sha256"]
    assert read(folder / "team/output-04/verification.json") == verification
    assert (folder / "team/output-04/optical-test-report.json").exists()
    assert (folder / "team/validator-01/design-review.json").exists()
    assert set(state["result"]["checks"]) >= {"conceptual_design", "analytic_design_tests"}

    verdict = call(evidence, "verify_run_artifacts", {"run_id": run_id})
    assert verdict["ok"] is True and verdict["plan_bytes_identical"] and verdict["archive_ok"] and verdict["receipt_bound"]
    assert call(evidence, "get_execution_status", {"run_id": run_id})["execution_status"] == "accepted"
    artifact = call(evidence, "get_stage_artifact", {"run_id": run_id, "stage": "prompt", "attempt": 1,
                                                     "filename": "decision-request-interpret.json"})
    request_record = json.loads(artifact["content"])  # journal-verified read of a decision file
    assert request_record["role"] == "interpret" and request_record["request_id"] == stops[0]["request_id"]
    assert artifact["sha256"] == hashlib.sha256((folder / "team/prompt-01/decision-request-interpret.json").read_bytes()).hexdigest()
    assert no_cloud.constructed == [] and no_cloud.settings_reads == []


def test_standalone_and_fixture_runs_are_unchanged(servers):
    runs, evidence, data = servers
    out = call(runs, "start_run", {"client_request_id": "fx", "request": "Assemble a 15-component complete Michelson interferometer."})
    wait_idle(runs)
    assert out["source"] == "offline rule fixture, not a model-designed experiment"
    state = run_json(data, out["run_id"])
    assert state["provider"] == "replay" and state["status"] == "algorithm_passed"
    # decision_mode "fixture" stays out of the idempotency hash: the same body repeats idempotently
    again = call(runs, "start_run", {"client_request_id": "fx", "request": "Assemble a 15-component complete Michelson interferometer.",
                                     "decision_mode": "fixture"})
    assert again["run_id"] == out["run_id"] and again["idempotent_replay"] is True
    # but an agent run with the same id is a different body
    call(runs, "start_run", {"client_request_id": "fx", "request": "Assemble a 15-component complete Michelson interferometer.",
                             "decision_mode": "agent"}, expect_error=True)


# ---- negative tests -----------------------------------------------------------------------------------------
def at_interpret(servers, cid="neg"):
    runs, evidence, data = servers
    run_id = start_agent(runs, cid)["run_id"]
    p = pending(runs, run_id)
    assert p["pending"] and p["role"] == "interpret" and p["rejections_remaining"] == 3 and p["submitted"] is False
    return runs, evidence, data, run_id, p


def assert_untouched_except_rejection(data, run_id, before, role="interpret"):
    after = freeze(data, run_id)
    assert not [f for f in set(after["files"]) - set(before["files"]) if "decision-response" in f]
    assert after["status"] == before["status"] == "awaiting_decision"
    assert after["retry_counts"] == before["retry_counts"] and after["retry_log"] == before["retry_log"]
    assert after["pending"] == before["pending"]
    node, attempt = before["pending"]["node"], before["pending"]["attempt"]
    assert set(after["files"]) - set(before["files"]) <= {f"{node}-{attempt:02d}/decision-rejections-{role}.json"}
    assert set(before["files"]) <= set(after["files"])


# 1
@pytest.mark.parametrize("breakage", ["missing_field", "extra_field", "wrong_type"])
def test_01_schema_invalid_answers_are_rejected_and_recorded(servers, breakage):
    runs, _, data, run_id, p = at_interpret(servers)
    answer = canned(p)
    if breakage == "missing_field":
        del answer["objective"]
    elif breakage == "extra_field":
        answer["note"] = "extra"
    else:
        answer["title"] = 5
    before = freeze(data, run_id)
    out = submit(runs, run_id, p, json.dumps(answer))
    assert out["accepted"] is False and out["errors"] and out["rejections_remaining"] == 2 and out["exhausted"] is False
    joined = " ".join(out["errors"])
    assert {"missing_field": "objective", "extra_field": "note", "wrong_type": "title"}[breakage] in joined
    assert_untouched_except_rejection(data, run_id, before)
    recorded = read(team(data, run_id) / "prompt-01/decision-rejections-interpret.json")
    assert len(recorded) == 1 and recorded[0]["errors"] == out["errors"] and recorded[0]["decider_label"] == LABEL
    assert len(recorded[0]["decision_sha256"]) == 64
    assert pending(runs, run_id)["rejections_remaining"] == 2


# 2
@pytest.mark.parametrize("kind", ["duplicate_key", "nan", "infinity"])
def test_02_duplicate_keys_and_non_finite_numbers_are_rejected(servers, kind):
    runs, _, data, run_id, p = at_interpret(servers)
    answer = canned(p)
    if kind == "duplicate_key":
        text = json.dumps(answer)[:-1] + ', "title": "second title"}'
    else:
        answer["position_tolerance_m"] = float("nan") if kind == "nan" else float("inf")
        text = json.dumps(answer)
    before = freeze(data, run_id)
    out = submit(runs, run_id, p, text)
    assert out["accepted"] is False and "not valid strict JSON" in out["errors"][0]
    assert_untouched_except_rejection(data, run_id, before)


# 3
def test_03_repair_object_outside_allowed_ids_is_rejected_at_submit(servers):
    runs, _, data = servers
    run_id = start_agent(runs)["run_id"]
    drive(runs, run_id, until_role="repair_controller")
    p = pending(runs, run_id)
    assert p["role"] == "repair_controller" and p["data"]["allowed_ids"] == ["OBJ"]
    bad = {"operation": "translate", "object_id": "EYE", "reason": "Move the neighbour instead."}
    before = freeze(data, run_id)
    out = submit(runs, run_id, p, json.dumps(bad))
    assert out["accepted"] is False and "authorization" in out["errors"][0]
    assert freeze(data, run_id)["retry_counts"] == before["retry_counts"]
    assert_untouched_except_rejection(data, run_id, before, "repair_controller")


# 4
def test_04_wrong_repair_operation_is_accepted_then_charged_exactly_like_the_openai_path(servers, tmp_path_factory,
                                                                                         monkeypatch):
    from test_agent_algorithm import run as standin_run
    runs, evidence, data = servers
    run_id = start_agent(runs)["run_id"]
    drive(runs, run_id, until_role="repair_controller")
    before = freeze(data, run_id)
    p = pending(runs, run_id)
    wrong = {"operation": "align", "object_id": "OBJ", "reason": "Rotate it (wrong class for a position error)."}
    out = submit(runs, run_id, p, json.dumps(wrong))
    assert out["accepted"] is True  # schema-valid and authorized: only the graph's own check can refuse it
    assert freeze(data, run_id)["retry_counts"] == before["retry_counts"]  # submit costs nothing
    assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(runs)
    after = run_json(data, run_id)
    assert after["status"] == "awaiting_decision"
    failed = read(team(data, run_id) / "controller-02/decision.json")
    assert failed["source"] == "agent" and failed["changes"]["proposal"] == wrong | {"object_id": "OBJ"}
    assert read(team(data, run_id) / "controller-02/event.json")["error"]["code"] == "AGENT_RECOVERY_INVALID"
    assert after["retry_log"][-1]["code"] == "AGENT_RECOVERY_INVALID"
    assert after["retry_log"][-1]["from"] == after["retry_log"][-1]["to"] == "controller"
    # an invalid decider answer charges the node's own budget, not the part's placement-repair budget
    assert after["retry_counts"] == {**before["retry_counts"], "controller": before["retry_counts"].get("controller", 0) + 1}
    assert pending(runs, run_id)["node_attempt"] == 3 and pending(runs, run_id)["request_id"] != p["request_id"]
    drive(runs, run_id, evidence=evidence)
    final = run_json(data, run_id)
    # the wasted answer is charged to the Controller node's own budget; OBJ keeps its placement repairs and recovers
    assert final["status"] == "algorithm_passed", final.get("error")
    assert final["retry_counts"]["controller"] == 1 and final["retry_counts"]["controller:OBJ"] == 2

    # the OpenAI stand-in charges the same budget for the same wrong answer
    swapped = []

    def alter(role, attempt, candidate):
        if role == "repair_controller" and not swapped:
            swapped.append(1)
            candidate["operation"] = "align"
        return candidate
    standin, _, _, _ = standin_run(tmp_path_factory.mktemp("standin"), monkeypatch, TELESCOPE, alter)
    assert standin["status"] == final["status"] and standin.get("error") == final.get("error")
    keys = ("from", "to", "code", "reason", "retry_limit", "total_limit", "retry_counts", "total_retries", "attempt")
    assert [{k: r[k] for k in keys} for r in final["retry_log"]] == [{k: r[k] for k in keys} for r in standin["retry_log"]]
    assert final["retry_counts"] == standin["retry_counts"]


# 5
def test_05_stale_request_id_is_refused(servers):
    runs, _, data = servers
    run_id = start_agent(runs)["run_id"]
    old = step(runs, run_id)
    assert old["role"] == "interpret"
    current = pending(runs, run_id)
    assert current["role"] == "design" and current["request_id"] != old["request_id"]
    before = freeze(data, run_id)
    message = submit(runs, run_id, old, json.dumps(canned(current)), expect_error=True)
    assert "DECISION_STALE" in message
    assert freeze(data, run_id) == before and responses(data, run_id) == ["decision-response-interpret.json"]
    # a request whose attempt already holds a completion record is stale as well
    (team(data, run_id) / "act-01/completion.json").write_text("{}")
    assert "DECISION_STALE" in submit(runs, run_id, current, json.dumps(canned(current)), expect_error=True)
    assert responses(data, run_id) == ["decision-response-interpret.json"]


# 6
def test_06_double_submit_is_idempotent_or_refused(servers):
    runs, _, data, run_id, p = at_interpret(servers)
    text = json.dumps(canned(p))
    first = submit(runs, run_id, p, text)
    assert first["accepted"] is True and first["idempotent_replay"] is False
    path = team(data, run_id) / "prompt-01/decision-response-interpret.json"
    stored = path.read_bytes()
    state = run_json(data, run_id)
    again = submit(runs, run_id, p, text, "another-label")
    assert again["accepted"] is True and again["idempotent_replay"] is True
    different = canned(p)
    different["title"] = "A different answer"
    assert "DECISION_ALREADY_SUBMITTED" in submit(runs, run_id, p, json.dumps(different), expect_error=True)
    assert path.read_bytes() == stored and responses(data, run_id) == [path.name]
    # submit never schedules anything: still paused, nothing active, run.json untouched
    assert not runs._optics_service.active
    assert run_json(data, run_id) == state and state["status"] == "awaiting_decision"
    assert read(path)["decider_label"] == LABEL and read(path)["backend"] == "standalone"
    assert read(path)["decision_sha256"] == hashlib.sha256(text.encode()).hexdigest()


# 7
def test_07_submit_is_refused_for_queued_finished_fixture_and_panel_runs(servers, monkeypatch):
    runs, _, data = servers
    fake = {"request_id": "a" * 64}
    # finished
    done = start_agent(runs, "done")["run_id"]
    drive(runs, done)
    assert run_json(data, done)["status"] == "algorithm_passed"
    assert "awaiting_decision" in submit(runs, done, fake, "{}", expect_error=True)
    assert pending(runs, done) == {"run_id": done, "pending": False, "status": "algorithm_passed"}
    # fixture run
    fixture = call(runs, "start_run", {"client_request_id": "fix", "request": "Assemble a 15-component complete Michelson interferometer."})["run_id"]
    wait_idle(runs)
    assert "not an agent-decided run" in submit(runs, fixture, fake, "{}", expect_error=True)
    assert "not an agent-decided run" in call(runs, "get_pending_decision", {"run_id": fixture}, expect_error=True)
    # panel-created (never in the idempotency index), even with provider agent
    panel = runs._optics_service.runtime.create(RunInput(request=request_text(TELESCOPE), profile="auto", mode="github_local",
                                                         simulation_backend="algorithm_v1", provider="agent"))
    assert "not created through start_run" in submit(runs, panel, fake, "{}", expect_error=True)
    assert "not created through start_run" in call(runs, "get_pending_decision", {"run_id": panel}, expect_error=True)
    assert run_json(data, panel)["status"] == "queued" and not responses(data, panel)
    # queued: a start_run whose worker never ran
    monkeypatch.setattr(mcp_runs.RunsService, "_submit", lambda self, run_id, continue_run: None)
    queued = call(runs, "start_run", {"client_request_id": "queued", "request": request_text(TELESCOPE),
                                      "decision_mode": "agent"})["run_id"]
    assert run_json(data, queued)["status"] == "queued"
    assert "awaiting_decision" in submit(runs, queued, fake, "{}", expect_error=True)
    assert pending(runs, queued)["pending"] is False
    assert not (data / "runs" / queued / "team").exists() or not responses(data, queued)
    # malformed arguments never reach the run
    for bad in ({"request_id": "xyz"}, {"request_id": "A" * 64}):
        assert "request_id" in call(runs, "submit_decision", {"run_id": done, "decision_json": "{}",
                                                              "decider_label": LABEL, **bad}, expect_error=True)
    for label in ("", "x" * 129, "bad;label", "bad\nlabel"):
        assert "decider_label" in submit(runs, done, fake, "{}", label, expect_error=True)
    assert "decision_json" in call(runs, "submit_decision", {"run_id": done, "request_id": "a" * 64,
                                                              "decision_json": "x" * 400_001, "decider_label": LABEL},
                                   expect_error=True)


# 8
def test_08_the_review_gate_stays_human_only(servers):
    runs, evidence, data = servers
    service = runs._optics_service
    runtime = service.runtime
    run_id = runtime.create(RunInput(request=request_text(TELESCOPE), profile="auto", mode="github_local",
                                     simulation_backend="algorithm_v1", provider="agent", require_review=True,
                                     algorithm_options={"delay_s": 0.0}))
    index = {"gate-test": {"run_id": run_id, "body_sha256": "0" * 64}}
    (data / mcp_runs.INDEX_NAME).write_text(json.dumps(index))
    service._submit(run_id, False)
    wait_idle(runs)
    last = None
    for _ in range(30):
        state = run_json(data, run_id)
        if state["status"] != "awaiting_decision":
            break
        last = step(runs, run_id)
    state = run_json(data, run_id)
    assert state["status"] == "awaiting_review", state.get("error")
    assert last["role"] == "repair_act"
    # nothing an MCP client can call approves, resumes the interrupt or answers the gate
    assert pending(runs, run_id) == {"run_id": run_id, "pending": False, "status": "awaiting_review"}
    assert "awaiting_decision" in submit(runs, run_id, last, "{}", expect_error=True)
    assert "cannot be resumed" in call(runs, "continue_run", {"run_id": run_id}, expect_error=True)
    wait_idle(runs)
    for params in ({"approved": True}, {"plan_sha256": state["plan_sha256"]}, {"resume": {"approved": True}}):
        call(runs, "continue_run", {"run_id": run_id, **params}, expect_error=True)
        call(runs, "submit_decision", {"run_id": run_id, "request_id": last["request_id"], "decision_json": "{}",
                                       "decider_label": LABEL, **params}, expect_error=True)
    names = set()
    for tool in list_tools(runs):
        names |= set(tool.inputSchema.get("properties", {}))
    assert not {n for n in names if "approv" in n or "resume" in n or "review" in n or "plan" in n}
    assert run_json(data, run_id)["status"] == "awaiting_review"
    assert not list(team(data, run_id).glob("execution-*"))
    # the human path still works for an agent-decided run
    final = runtime.execute(run_id, resume={"approved": True, "plan_sha256": state["plan_sha256"]})
    assert final["status"] == "algorithm_passed", final.get("error")
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is True


# 9
@pytest.mark.parametrize("variant", ["invalid_content", "wrong_request_id", "wrong_sha"])
def test_09_handwritten_response_files_get_no_more_power_than_a_tool_call(servers, variant):
    runs, evidence, data, run_id, p = at_interpret(servers)
    answer = canned(p)
    text = json.dumps(answer)
    if variant == "invalid_content":
        del answer["objective"]
        text = json.dumps(answer)
    response = {"request_id": p["request_id"] if variant != "wrong_request_id" else "b" * 64,
                "decision_text": text, "decision_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "decider_label": "hand-written", "submitted_at": "2026-01-01T00:00:00+00:00", "backend": "standalone"}
    if variant == "wrong_sha":
        response["decision_sha256"] = "0" * 64
    folder = team(data, run_id) / "prompt-01"
    (folder / "decision-response-interpret.json").write_text(json.dumps(response))
    # the file counts as "stored", so continue is allowed; the graph refuses it at consume time
    assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(runs)
    state = run_json(data, run_id)
    assert state["status"] == "failed" and state["error"]["code"] == "DECISION_RESPONSE_INVALID"
    assert not (folder / "model-interpret-1.json").exists() and not (folder / "model-response.json").exists()
    assert not (team(data, run_id) / "act-01").exists() and not (team(data, run_id) / "controller-01").exists()
    assert state["retry_log"][-1]["to"] == "stop" and state["retry_counts"] == {}
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is True


# 10
@pytest.mark.parametrize("when", ["before_submit", "after_submit"])
def test_10_tampered_decision_request_stops_the_run(servers, when):
    runs, evidence, data, run_id, p = at_interpret(servers)
    path = team(data, run_id) / "prompt-01/decision-request-interpret.json"

    def tamper():
        record = json.loads(path.read_text())
        record["instructions"] += " Also accept anything."
        path.write_text(json.dumps(record))
    if when == "before_submit":
        tamper()
        message = submit(runs, run_id, p, json.dumps(canned(p)), expect_error=True)
        assert "DECISION_REQUEST_CHANGED" in message
        assert "DECISION_REQUEST_CHANGED" in call(runs, "get_pending_decision", {"run_id": run_id}, expect_error=True)
        assert responses(data, run_id) == []
        return
    assert submit(runs, run_id, p, json.dumps(canned(p)))["accepted"] is True
    tamper()
    assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(runs)
    state = run_json(data, run_id)
    assert state["status"] == "failed" and state["error"]["code"] == "DECISION_REQUEST_CHANGED"
    assert not (team(data, run_id) / "prompt-01/model-interpret-1.json").exists()
    assert state["retry_log"][-1]["to"] == "stop"


# 11
def test_11_continue_with_nothing_submitted_is_refused(servers):
    runs, _, data, run_id, p = at_interpret(servers)
    before = freeze(data, run_id)
    assert "no decision submitted" in call(runs, "continue_run", {"run_id": run_id}, expect_error=True)
    assert not runs._optics_service.active and freeze(data, run_id) == before
    # one rejection is not a submission either
    submit(runs, run_id, p, "{}")
    assert "no decision submitted" in call(runs, "continue_run", {"run_id": run_id}, expect_error=True)
    assert run_json(data, run_id)["status"] == "awaiting_decision" and responses(data, run_id) == []


# 12
def test_12_a_new_server_on_the_same_data_dir_consumes_the_decision_exactly_once(servers):
    runs, evidence, data, run_id, p = at_interpret(servers)
    assert submit(runs, run_id, p, json.dumps(canned(p)))["accepted"] is True
    stored = (team(data, run_id) / "prompt-01/decision-response-interpret.json").read_bytes()
    request_bytes = (team(data, run_id) / "prompt-01/decision-request-interpret.json").read_bytes()
    # the process "dies" here: the first server and its worker pool are abandoned
    again = mcp_runs.build_server(data)
    assert pending(again, run_id)["submitted"] is True
    assert submit(again, run_id, p, json.dumps(canned(p)))["idempotent_replay"] is True
    assert call(again, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(again)
    folder = team(data, run_id) / "prompt-01"
    assert (folder / "decision-response-interpret.json").read_bytes() == stored
    assert (folder / "decision-request-interpret.json").read_bytes() == request_bytes
    assert len(list(folder.glob("model-interpret-*.json"))) == 1
    assert read(folder / "model-interpret-1.json")["response_id"] == p["request_id"]
    state = run_json(data, run_id)
    assert state["attempts"]["prompt"] == 1 and (folder / "completion.json").exists()
    following = pending(again, run_id)
    assert following["role"] == "design" and following["request_id"] != p["request_id"]
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is True
    # a second continue has nothing to consume
    assert "no decision submitted" in call(again, "continue_run", {"run_id": run_id}, expect_error=True)


# 13
def test_13_crash_after_a_completion_record_replays_without_a_new_request(servers, monkeypatch):
    runs, evidence, data = servers
    run_id = start_agent(runs)["run_id"]
    step(runs, run_id)  # interpret -> design is pending
    original = team_flow.atomic

    def die_after_act_journal(path, payload):
        original(path, payload)
        if path.name == "completion.json" and path.parent.name == "act-01":
            raise RuntimeError("process stopped after writing the Act journal")
    monkeypatch.setattr(team_flow, "atomic", die_after_act_journal)
    p = pending(runs, run_id)
    assert p["role"] == "design"
    assert submit(runs, run_id, p, json.dumps(canned(p)))["accepted"] is True
    call(runs, "continue_run", {"run_id": run_id})
    wait_idle(runs)
    assert run_json(data, run_id)["status"] == "interrupted"
    folder = team(data, run_id) / "act-01"
    journal = (folder / "completion.json").read_bytes()
    listing = sorted(x.name for x in folder.iterdir())
    requests = {x.name: x.read_bytes() for x in folder.glob("decision-*")}
    monkeypatch.setattr(team_flow, "atomic", original)
    assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(runs)
    assert (folder / "completion.json").read_bytes() == journal
    assert sorted(x.name for x in folder.iterdir()) == listing
    assert {x.name: x.read_bytes() for x in folder.glob("decision-*")} == requests
    assert sorted(requests) == ["decision-request-design.json", "decision-response-design.json"]
    assert len(list(folder.glob("model-design-*.json"))) == 1
    following = pending(runs, run_id)
    assert (following["node"], following["role"]) == ("validator", "optical_tests")
    assert run_json(data, run_id)["attempts"]["act"] == 1
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is True


# 14
def test_14_stop_run_while_a_decision_is_pending(servers):
    runs, evidence, data, run_id, p = at_interpret(servers)
    before = freeze(data, run_id)
    assert "waiting for an agent decision" in call(runs, "pause_run", {"run_id": run_id}, expect_error=True)
    assert freeze(data, run_id) == before and not (data / "runs" / run_id / "control.json").exists()
    assert call(runs, "stop_run", {"run_id": run_id})["status"] == "stop_requested"
    assert run_json(data, run_id)["status"] == "stopped"
    assert "awaiting_decision" in submit(runs, run_id, p, json.dumps(canned(p)), expect_error=True)
    assert responses(data, run_id) == []
    assert pending(runs, run_id) == {"run_id": run_id, "pending": False, "status": "stopped"}
    call(runs, "continue_run", {"run_id": run_id}, expect_error=True)
    wait_idle(runs)
    assert run_json(data, run_id)["status"] == "stopped"
    assert not (team(data, run_id) / "act-01").exists()
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["journals_ok"] is True


# 15
def test_15_three_rejections_exhaust_the_request_and_the_graph_charges_the_budget(servers):
    runs, evidence, data, run_id, p = at_interpret(servers)
    for expected_left in (2, 1, 0):
        out = submit(runs, run_id, p, '{"title": 1}')
        assert out["accepted"] is False and out["rejections_remaining"] == expected_left
    assert out["exhausted"] is True
    assert "DECISION_EXHAUSTED" in submit(runs, run_id, p, json.dumps(canned(p)), expect_error=True)
    assert responses(data, run_id) == [] and pending(runs, run_id)["rejections_remaining"] == 0
    assert not run_json(data, run_id).get("retry_counts")
    assert call(runs, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_idle(runs)
    state = run_json(data, run_id)
    assert state["status"] == "awaiting_decision"
    assert state["retry_log"][0]["code"] == "AGENT_DECISION_INVALID"
    assert (state["retry_log"][0]["from"], state["retry_log"][0]["to"], state["retry_log"][0]["reason"]) == (
        "prompt", "prompt", "retry_allowed")
    assert state["retry_counts"] == {"prompt": 1}
    failed = read(team(data, run_id) / "prompt-01/event.json")
    assert failed["outcome"] == "failed" and failed["error"]["code"] == "AGENT_DECISION_INVALID"
    assert len(read(team(data, run_id) / "prompt-01/decision-rejections-interpret.json")) == 3
    retry = pending(runs, run_id)
    assert (retry["node"], retry["node_attempt"], retry["role"]) == ("prompt", 2, "interpret")
    assert retry["request_id"] != p["request_id"] and retry["rejections_remaining"] == 3
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is True
    # the rejected request is stale now
    assert "DECISION_STALE" in submit(runs, run_id, p, json.dumps(canned(p)), expect_error=True)
    drive(runs, run_id, evidence=evidence)
    assert run_json(data, run_id)["status"] == "algorithm_passed"


# 16
def test_16_decision_pending_is_not_caught_by_stage():
    import re
    import inspect
    source = inspect.getsource(team_flow.build_team_graph)
    handler = re.search(r"except \(([A-Za-z, ]+)\) as exc:\n\s+code = getattr\(exc,\"code\"", source)
    assert handler, "stage()'s except tuple moved: update this test"
    import builtins
    caught = tuple(getattr(builtins, n.strip(), None) or getattr(team_flow, n.strip()) for n in handler.group(1).split(","))
    assert {c.__name__ for c in caught} == {"ValueError", "GatewayError", "OSError", "KeyError", "TypeError", "BadZipFile"}
    assert not issubclass(DecisionPending, caught) and not isinstance(DecisionPending("x"), caught)
    assert DecisionPending.__mro__ == (DecisionPending, Exception, BaseException, object)
    assert not issubclass(DecisionPending, (RuntimeError, gateway.GatewayError, BadZipFile))
    assert DecisionPending("abc").args == ("abc",)


# 17
def test_17_resume_after_decision_pending_through_the_real_sqlite_checkpoint(tmp_path):
    import sqlite3
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(request=request_text(TELESCOPE), profile="auto", mode="github_local",
                                     simulation_backend="algorithm_v1", provider="agent",
                                     algorithm_options={"delay_s": 0.0}))
    state = runtime.execute(run_id)
    assert state["status"] == "awaiting_decision" and state["next_node"] == "prompt"
    pending_entry = state["pending_decision"]
    assert pending_entry["node"] == "prompt" and pending_entry["role"] == "interpret"
    # the pending write for the escaped exception was committed to the SQLite checkpoint
    rows = sqlite3.connect(tmp_path / "checkpoints.sqlite").execute(
        "select channel, type, value from writes where channel = '__error__'").fetchall()
    assert len(rows) == 1 and rows[0][1] == "msgpack" and b"DecisionPending" in rows[0][2]
    # a second, independent Runtime (fresh SqliteSaver connection) re-runs the node once the answer is stored
    folder = tmp_path / "runs" / run_id
    record = agent_decisions.read_request(folder, pending_entry)
    text = json.dumps(canned_answer("interpret", record["data"], TELESCOPE))
    out = agent_decisions.submit(folder, pending_entry, pending_entry["request_id"], text, LABEL, "test")
    assert out["accepted"] is True
    resumed = Runtime(tmp_path).execute(run_id, continue_run=True)
    assert resumed["status"] == "awaiting_decision" and resumed["pending_decision"]["role"] == "design"
    assert resumed["attempts"]["prompt"] == 1 and (folder / "team/prompt-01/completion.json").exists()
    assert read(folder / "team/prompt-01/decision.json")["source"] == "agent"
    # and the human review gate is untouched by this machinery: no interrupt is pending here
    with Runtime(tmp_path).graph("github_local", algorithm=True) as graph:
        snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
        assert not any(task.interrupts for task in snapshot.tasks) and snapshot.next == ("act",)
    # a stopped run can be continued only by an explicit decision: stop wins and stays stopped
    Runtime(tmp_path).control(run_id, "stop")
    assert Runtime(tmp_path).get(run_id)["status"] == "stopped"
    with pytest.raises(ValueError):
        Runtime(tmp_path).control(run_id, "pause")


def test_runtime_rejects_agent_provider_for_other_backends():
    with pytest.raises(ValueError):
        RunInput(request="x", provider="agent")  # physical backend
    assert RunInput(request="x", provider="agent", profile="auto", mode="github_local",
                    simulation_backend="algorithm_v1").provider == "agent"


def test_cli_and_panel_provider_surfaces_are_unchanged():
    import inspect
    from optics_graph import cli
    source = inspect.getsource(cli)
    assert '"agent"' not in source and "'agent'" not in source


# ---- supporting properties ----------------------------------------------------------------------------------
def test_an_accepted_submit_writes_exactly_one_file_and_changes_nothing_else(servers):
    runs, _, data, run_id, p = at_interpret(servers)
    before = freeze(data, run_id)
    raw_before = (data / "runs" / run_id / "run.json").read_bytes()
    assert submit(runs, run_id, p, json.dumps(canned(p)))["accepted"] is True
    after = freeze(data, run_id)
    assert set(after["files"]) - set(before["files"]) == {"prompt-01/decision-response-interpret.json"}
    assert set(before["files"]) <= set(after["files"])
    assert (data / "runs" / run_id / "run.json").read_bytes() == raw_before
    assert not runs._optics_service.active and not (data / "runs" / run_id / "control.json").exists()
    assert {k: after[k] for k in ("status", "retry_counts", "retry_log", "pending")} == {
        k: before[k] for k in ("status", "retry_counts", "retry_log", "pending")}


def test_exclusive_create_lets_exactly_one_concurrent_writer_win(tmp_path):
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
    assert sorted(outcomes) == ["lost"] * 7 + ["won"]
    assert target.read_bytes().startswith(b"writer-") and [p.name for p in tmp_path.iterdir()] == [target.name]


def test_orphan_adoption_also_matches_on_provider(servers, monkeypatch):
    runs, _, data = servers
    service = runs._optics_service
    real_marker = mcp_runs.RunsService._write_marker

    def marker_then_die(self, run_id, client_request_id, nonce):
        real_marker(self, run_id, client_request_id, nonce)
        raise OSError("process died right after the marker was written")
    monkeypatch.setattr(mcp_runs.RunsService, "_write_marker", marker_then_die)
    body = {"client_request_id": "orphan-agent", "request": request_text(TELESCOPE), "decision_mode": "agent"}
    call(runs, "start_run", body, expect_error=True)
    monkeypatch.setattr(mcp_runs.RunsService, "_write_marker", real_marker)
    (orphan,) = [p.name for p in (data / "runs").iterdir()]
    assert run_json(data, orphan)["provider"] == "agent"
    index = json.loads((data / mcp_runs.INDEX_NAME).read_text())
    entry = index["orphan-agent"]
    find = lambda provider: service._orphans({}, "orphan-agent", entry["nonce"], request_text(TELESCOPE), "none",
                                              {"delay_s": 0.0}, entry["started_at"], provider)
    assert find("agent")[0] == [orphan] and find("replay")[0] == [] and find("replay")[1] == []
    out = call(runs, "start_run", body)  # the retry adopts the agent orphan and starts it
    wait_idle(runs)
    assert out["run_id"] == orphan and len(list((data / "runs").iterdir())) == 1
    assert run_json(data, orphan)["status"] == "awaiting_decision"


def test_hosted_backend_serves_the_same_agent_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from optics_graph import server
    monkeypatch.setattr(server, "public_settings", lambda: {"model": "none", "key_configured": False})
    headers = {"x-optics-local": "1", "accept": "application/json, text/event-stream", "content-type": "application/json"}

    def tool(client, name, args, expect_error=False):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}
        response = client.post("/mcp", content=json.dumps(body), headers=headers)
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert bool(result.get("isError")) == expect_error, result["content"][0]["text"]
        return result["content"][0]["text"] if expect_error else result["structuredContent"]

    def settle(client, run_id):
        end = time.time() + 90
        while time.time() < end:
            body = client.get(f"/api/runs/{run_id}").json()
            if not body["active"]:
                return body
            time.sleep(.02)
        raise AssertionError("worker did not finish")

    with TestClient(server.create_app(Runtime(tmp_path))) as client:
        run_id = tool(client, "start_run", {"client_request_id": "hosted", "request": request_text(TELESCOPE),
                                            "decision_mode": "agent"})["run_id"]
        body = settle(client, run_id)
        assert body["status"] == "awaiting_decision" and body["display_state"] == "awaiting_decision"
        assert "no decision submitted" in tool(client, "continue_run", {"run_id": run_id}, True)
        roles = []
        for _ in range(20):
            p = tool(client, "get_pending_decision", {"run_id": run_id})
            if not p["pending"]:
                break
            roles.append(p["role"])
            out = tool(client, "submit_decision", {"run_id": run_id, "request_id": p["request_id"],
                                                   "decision_json": json.dumps(canned(p)), "decider_label": LABEL})
            assert out["accepted"] is True
            assert read(tmp_path / "runs" / run_id / f"team/{p['node']}-{p['node_attempt']:02d}"
                        / f"decision-response-{p['role']}.json")["backend"] == "hosted"
            tool(client, "continue_run", {"run_id": run_id})
            body = settle(client, run_id)
        assert roles == EXPECTED_ROLES and body["status"] == "algorithm_passed"
        assert tool(client, "get_worker_error", {"run_id": run_id})["display_state"] == "algorithm_passed"


def test_rejected_test_suites_do_not_use_up_a_misplaced_parts_repairs(servers):
    """Regression for a live run (Michelson, 2026-11-14): two reviews rejected the test suite (validator -> act, the
    Act node's own budget), which then left no Act retry for the seeded part's angle repair, so the run stopped
    with target_retry_limit. Placement repairs of one part now count under "act:<part>" / "controller:<part>";
    every key still counts toward the total of 7 and no limit was raised."""
    runs, evidence, data = servers
    run_id = start_agent(runs)["run_id"]
    rejections = []

    def reject_tests_twice(p, answer):
        if p["role"] == "review" and len(rejections) < 2:
            rejections.append(1)
            return {"accepted": False, "issues": ["Test coverage gap: no test checks component orientation (yaw)."],
                    "return_to": "act", "reasoning": "The suite leaves orientation unchecked.", "problem_source": "tests"}
        return answer
    drive(runs, run_id, evidence=evidence, mutate=reject_tests_twice)
    state = run_json(data, run_id)
    assert len(rejections) == 2
    assert state["status"] == "algorithm_passed", state.get("error")
    assert [r["to"] for r in state["retry_log"]] == ["act", "act", "controller", "controller", "act"]
    assert state["retry_counts"] == {"act": 2, "controller:OBJ": 2, "act:OBJ": 1}
    assert state["retry_log"][-1]["total_retries"] == 5 <= state["retry_log"][-1]["total_limit"] == 7
