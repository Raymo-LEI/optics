"""Tests for the optics-evidence (read-only) and optics-runs (mutating) MCP servers.

Both servers are driven through the MCP SDK's in-memory client session
(mcp.shared.memory.create_connected_server_and_client_session), i.e. real
JSON-RPC tool calls and schema validation, without a subprocess.
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from optics_graph import algorithm_sim, gateway, journal_check, mcp_evidence, mcp_runs, settings, team_flow
from optics_graph.algorithm_sim import AlgorithmServices
from optics_graph.runtime import Runtime

MICHELSON = "Assemble a 15-component complete Michelson interferometer."

EVIDENCE_TOOLS = {"list_runs", "get_run", "get_feedback_log", "get_stage_journal", "get_stage_artifact",
                  "verify_run_artifacts", "get_acceptance_summary", "get_execution_status"}
RUNS_TOOLS = {"start_run", "continue_run", "get_worker_error", "pause_run", "stop_run",
              "get_pending_decision", "submit_decision"}
FORBIDDEN = re.compile(r"provider|approv|review|toleran|retr(y|ies)|return_to|plan|setting|api_?key|secret|token|model",
                       re.I)


def _boom(*args, **kwargs):
    raise AssertionError("cloud client constructed: the MCP wrap must never call a model")


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
    """No OpenAI client can be built and no API key (env or config/local.json) can be read.

    gateway.py does `from openai import OpenAI`, so the name that matters is
    optics_graph.gateway.OpenAI; read_settings is patched in every module that imported it by name,
    so config/local.json is never opened.
    """
    import openai

    monkeypatch.setattr(openai, "OpenAI", _boom)
    monkeypatch.setattr(gateway, "OpenAI", _boom)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def keyless():
        return {"model": "none", "api_key": "", "robot_endpoint": ""}
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("optics_graph") and hasattr(module, "read_settings"):
            monkeypatch.setattr(module, "read_settings", keyless)
    monkeypatch.setattr(settings, "read_settings", keyless)


def test_no_cloud_guard_is_live(monkeypatch):
    """The guard itself works: building gateway's client raises, and no key is readable."""
    with pytest.raises(AssertionError, match="cloud client constructed"):
        gateway.OpenAI(api_key="sk-test", timeout=1, max_retries=0)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-read")
    assert gateway.read_settings()["api_key"] == ""
    assert settings.read_settings()["api_key"] == ""
    import openai
    with pytest.raises(AssertionError):
        openai.OpenAI()


def call(server, name, args=None, *, expect_error=False):
    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            return await session.call_tool(name, args or {})
    result = asyncio.run(go())
    if expect_error:
        assert result.isError, f"{name}{args} should have been rejected"
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


def wait_idle(runs, timeout=60):
    service = runs._optics_service
    end = time.time() + timeout
    while time.time() < end:
        with service.lock:
            if not service.active:
                return
        time.sleep(.05)
    raise AssertionError("worker did not finish")


def run_json(data, run_id):
    return json.loads((Path(data) / "runs" / run_id / "run.json").read_text())


@pytest.fixture
def servers(tmp_path):
    return mcp_runs.build_server(tmp_path), mcp_evidence.build_server(tmp_path), tmp_path


@pytest.fixture
def passed(servers):
    runs, evidence, data = servers
    out = call(runs, "start_run", {"client_request_id": "c1", "request": MICHELSON})
    wait_idle(runs)
    return runs, evidence, data, out["run_id"]


# (a)
def test_a_start_run_reaches_algorithm_passed_with_same_plan(passed):
    runs, evidence, data, run_id = passed
    state = run_json(data, run_id)
    assert state["status"] == "algorithm_passed", state.get("error")
    assert state["provider"] == "replay" and state["simulation_backend"] == "algorithm_v1"
    assert len(state["algorithm_history"]) == 3
    assert all(h["moved_ids"] == ["L25"] for h in state["algorithm_history"])
    folder = data / "runs" / run_id
    assert (folder / state["plan_path"]).read_bytes() == (folder / "team/execution-01/plan.json").read_bytes()
    assert state["result"]["motions_sent"] == 0
    assert call(evidence, "get_run", {"run_id": run_id})["status"] == "algorithm_passed"
    assert [r["run_id"] for r in call(evidence, "list_runs")["runs"]] == [run_id]
    assert call(evidence, "get_execution_status", {"run_id": run_id})["execution_status"] == "accepted"


# (b)
def test_b_feedback_log_preserves_original_failures(passed):
    _, evidence, _, run_id = passed
    log = call(evidence, "get_feedback_log", {"run_id": run_id})
    assert log["routes"] == ["output -> controller", "output -> act"]
    assert log["consistent"], log["issues"]
    first, second = log["decisions"]
    assert first["decision"]["code"] == "ALGORITHM_POSITION_MISMATCH"
    assert first["original_failure"]["code"] == "ALGORITHM_POSITION_MISMATCH"
    assert first["original_failure"]["details"]["target_object_ids"] == ["L25"]
    assert first["original_failure"]["details"]["value"] > first["original_failure"]["details"]["limit"]
    assert second["decision"]["code"] == "ALGORITHM_OPTICAL_MISMATCH"
    assert second["original_failure"]["details"]["value"] > second["original_failure"]["details"]["limit"]
    assert second["original_failure"]["requested_return"] == "act"


# (c)
def test_c_verify_passes_then_fails_on_flipped_zip_byte(passed):
    _, evidence, data, run_id = passed
    ok = call(evidence, "verify_run_artifacts", {"run_id": run_id})
    assert ok["journals_ok"] and ok["plan_bytes_identical"] and ok["archive_ok"] and ok["receipt_bound"]
    assert ok["issues"] == [] and ok["ok"] is True
    zip_path = data / "runs" / run_id / "team/sim-01/artifact.zip"
    raw = bytearray(zip_path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    zip_path.write_bytes(bytes(raw))
    bad = call(evidence, "verify_run_artifacts", {"run_id": run_id})
    assert bad["ok"] is False and bad["journals_ok"] is False
    assert any("sim-01" in i and "artifact.zip" in i for i in bad["issues"])
    journal = call(evidence, "get_stage_journal", {"run_id": run_id, "stage": "sim", "attempt": 1})
    assert journal["verified"] is False
    call(evidence, "get_stage_artifact",
         {"run_id": run_id, "stage": "sim", "attempt": 1, "filename": "artifact.zip"}, expect_error=True)


def test_c_final_archive_tamper_is_caught_by_archive_check(passed):
    _, evidence, data, run_id = passed
    zip_path = data / "runs" / run_id / "team/sim-03/artifact.zip"
    raw = bytearray(zip_path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    zip_path.write_bytes(bytes(raw))
    bad = call(evidence, "verify_run_artifacts", {"run_id": run_id})
    assert bad["ok"] is False and bad["archive_ok"] is False and bad["journals_ok"] is False


@pytest.mark.parametrize("mutation", ["update_sha256", "file_hash", "update_content"])
def test_c_altered_completion_json_fails(passed, mutation):
    _, evidence, data, run_id = passed
    path = data / "runs" / run_id / "team/controller-01/completion.json"
    body = json.loads(path.read_text())
    if mutation == "update_sha256":
        body["update_sha256"] = ("0" if body["update_sha256"][0] != "0" else "1") + body["update_sha256"][1:]
    elif mutation == "file_hash":
        name = sorted(body["files"])[0]
        body["files"][name] = "0" * 64
    else:
        body["update"]["stage"] = "tampered"
    path.write_text(json.dumps(body))
    bad = call(evidence, "verify_run_artifacts", {"run_id": run_id})
    assert bad["ok"] is False and bad["journals_ok"] is False
    assert any("controller-01" in i for i in bad["issues"])


def test_c_tampered_published_plan_is_detected(passed):
    _, evidence, data, run_id = passed
    state = run_json(data, run_id)
    plan = data / "runs" / run_id / state["plan_path"]
    plan.write_bytes(plan.read_bytes() + b" ")
    bad = call(evidence, "verify_run_artifacts", {"run_id": run_id})
    assert bad["ok"] is False and bad["plan_bytes_identical"] is False


# (d)
def test_d_unknown_execution_is_never_retried(servers, monkeypatch):
    runs, evidence, data = servers
    atomic = team_flow.atomic
    calls = []
    original = AlgorithmServices.execution

    def execution(self, state, path, attempt):
        calls.append(attempt)
        return original(self, state, path, attempt)

    def interrupt(path, payload):
        atomic(path, payload)
        if path.parent.name == "execution-01" and path.name == "execution-started.json":
            raise RuntimeError("interrupted after durable execution-started marker")

    monkeypatch.setattr(AlgorithmServices, "execution", execution)
    monkeypatch.setattr(team_flow, "atomic", interrupt)
    run_id = call(runs, "start_run", {"client_request_id": "crash", "request": MICHELSON})["run_id"]
    wait_idle(runs)
    assert run_json(data, run_id)["status"] == "interrupted"
    monkeypatch.setattr(team_flow, "atomic", atomic)

    marker = data / "runs" / run_id / "team/execution-01/execution-started.json"
    saved = marker.read_bytes()
    status = call(evidence, "get_execution_status", {"run_id": run_id})
    assert status["execution_status"] == "unknown" and status["marker_present"] is True

    call(runs, "continue_run", {"run_id": run_id})
    wait_idle(runs)
    final = run_json(data, run_id)
    assert final["status"] == "failed"
    assert final["error"]["code"] == "ALGORITHM_EXECUTION_UNKNOWN"
    assert calls == [] and marker.read_bytes() == saved
    assert not (marker.parent / "execution_receipt.json").exists()
    assert call(evidence, "get_execution_status", {"run_id": run_id})["execution_status"] == "unknown"
    # the failed run cannot be resumed again either
    call(runs, "continue_run", {"run_id": run_id}, expect_error=True)
    wait_idle(runs)
    assert calls == []


def test_d_not_started_and_rejected_statuses(servers):
    runs, evidence, data = servers
    r1 = call(runs, "start_run", {"client_request_id": "x1", "request": MICHELSON, "team_fault": "execution"})["run_id"]
    wait_idle(runs)
    rejected = call(evidence, "get_execution_status", {"run_id": r1})
    assert rejected["execution_status"] == "rejected"
    assert rejected["error"]["code"] == "ALGORITHM_HANDOFF_REJECTED"
    r2 = call(runs, "start_run", {"client_request_id": "x2", "request": MICHELSON,
                                 "algorithm_options": {"position_response": 0.0}})["run_id"]
    wait_idle(runs)
    assert run_json(data, r2)["status"] == "failed"
    assert call(evidence, "get_execution_status", {"run_id": r2})["execution_status"] == "not_started"
    acc = call(evidence, "get_acceptance_summary", {"run_id": r2})
    assert acc["real_failures"] and "not complete" in acc["workflow_status"]


# (e)
def test_e_idempotent_start_run(servers):
    runs, evidence, data = servers
    a = call(runs, "start_run", {"client_request_id": "same", "request": MICHELSON})
    wait_idle(runs)
    b = call(runs, "start_run", {"client_request_id": "same", "request": MICHELSON})
    assert a["run_id"] == b["run_id"] and b["idempotent_replay"] is True
    assert len(list((data / "runs").iterdir())) == 1
    call(runs, "start_run", {"client_request_id": "same", "request": MICHELSON + " "}, expect_error=True)
    call(runs, "start_run", {"client_request_id": "same", "request": MICHELSON, "team_fault": "sim"},
         expect_error=True)
    assert len(list((data / "runs").iterdir())) == 1
    assert run_json(data, a["run_id"])["status"] == "algorithm_passed"


def test_e_continue_rejected_while_active_and_options_allowlist(servers, monkeypatch):
    runs, _, data = servers
    for bad in ({"initial_position_error_m": 0.0}, {"tolerance": 1}, {"position_response": 2}):
        call(runs, "start_run", {"client_request_id": "bad", "request": MICHELSON, "algorithm_options": bad},
             expect_error=True)
    assert not (data / "runs").exists() or not list((data / "runs").iterdir())
    import threading
    entered, release = threading.Event(), threading.Event()
    original = AlgorithmServices.sim

    def slow_sim(self, state, path, attempt):
        entered.set()
        assert release.wait(10)
        return original(self, state, path, attempt)
    monkeypatch.setattr(AlgorithmServices, "sim", slow_sim)
    run_id = call(runs, "start_run", {"client_request_id": "ok", "request": MICHELSON})["run_id"]
    assert entered.wait(10)
    try:
        call(runs, "continue_run", {"run_id": run_id}, expect_error=True)  # active in this process
    finally:
        release.set()
    wait_idle(runs)
    assert run_json(data, run_id)["status"] == "algorithm_passed"
    call(runs, "continue_run", {"run_id": run_id}, expect_error=True)  # finished: nothing to resume
    call(runs, "continue_run", {"run_id": "0" * 32}, expect_error=True)


def test_runs_server_refuses_panel_data_dir():
    from optics_graph.settings import DATA
    with pytest.raises(ValueError):
        mcp_runs.build_server(DATA)


# (f)
def test_f_tool_surface(servers):
    runs, evidence, _ = servers
    for server, expected in ((evidence, EVIDENCE_TOOLS), (runs, RUNS_TOOLS)):
        tools = list_tools(server)
        assert {t.name for t in tools} == expected

        def names(schema):
            found = set()
            if isinstance(schema, dict):
                for key, value in schema.items():
                    if key == "properties" and isinstance(value, dict):
                        found |= set(value)
                    found |= names(value)
            elif isinstance(schema, list):
                for item in schema:
                    found |= names(item)
            return found
        for tool in tools:
            params = names(tool.inputSchema)
            assert not [p for p in params if FORBIDDEN.search(p)], (tool.name, params)
    expected_params = {
        "list_runs": {"limit"}, "get_run": {"run_id"}, "get_feedback_log": {"run_id"},
        "get_stage_journal": {"run_id", "stage", "attempt"},
        "get_stage_artifact": {"run_id", "stage", "attempt", "filename"},
        "verify_run_artifacts": {"run_id"}, "get_acceptance_summary": {"run_id"},
        "get_execution_status": {"run_id"},
        "start_run": {"client_request_id", "request", "team_fault", "algorithm_options", "decision_mode"},
        "continue_run": {"run_id"}, "get_worker_error": {"run_id"}, "pause_run": {"run_id"},
        "stop_run": {"run_id"}, "get_pending_decision": {"run_id"},
        "submit_decision": {"run_id", "request_id", "decision_json", "decider_label"}}
    assert set(expected_params) == EVIDENCE_TOOLS | RUNS_TOOLS
    for server in (evidence, runs):
        for tool in list_tools(server):
            assert set(tool.inputSchema.get("properties", {})) == expected_params[tool.name], tool.name


# (g)
def test_g_acceptance_summary_is_not_all_verified(passed):
    _, evidence, _, run_id = passed
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert s["raw_status"] == "algorithm_passed"
    assert "workflow complete" in s["workflow_status"]
    assert "NOT 'all requirements verified'" in s["workflow_status"]
    assert s["physical_verified"] is False
    assert s["core_unaccepted"] and s["deferred_physical"] and s["real_failures"] == []
    assert any("not been verified" in c for c in s["core_unaccepted"])


# (h)
@pytest.mark.parametrize("run_id", ["../x", "0" * 31, "0" * 33, "A" * 32, "g" * 32, "", "0" * 32 + "/..", "%2e%2e"])
def test_h_malformed_run_id(servers, run_id):
    _, evidence, _ = servers
    for tool in sorted(EVIDENCE_TOOLS - {"list_runs"}):
        args = {"run_id": run_id}
        if tool in ("get_stage_journal", "get_stage_artifact"):
            args.update(stage="sim", attempt=1)
        if tool == "get_stage_artifact":
            args["filename"] = "artifact.zip"
        call(evidence, tool, args, expect_error=True)


def test_h_bad_stage_attempt_filename(passed):
    _, evidence, _, run_id = passed
    base = {"run_id": run_id, "stage": "sim", "attempt": 1, "filename": "artifact.zip"}
    call(evidence, "get_stage_artifact", base)  # control: the legitimate call works
    for patch in ({"stage": "../sim"}, {"stage": "sim/.."}, {"stage": "feedback"}, {"attempt": 0}, {"attempt": 101},
                  {"attempt": -1}, {"filename": "../../run.json"}, {"filename": "run.json"},
                  {"filename": "plan.json.bak"}, {"filename": "/etc/passwd"}, {"filename": "..\\plan.json"},
                  {"filename": "sub/plan.json"}):
        call(evidence, "get_stage_artifact", {**base, **patch}, expect_error=True)
    call(evidence, "get_stage_journal", {"run_id": run_id, "stage": "../..", "attempt": 1}, expect_error=True)
    call(evidence, "get_stage_journal", {"run_id": run_id, "stage": "sim", "attempt": 99}, expect_error=True)


def test_h_symlinked_artifact_is_refused(passed):
    _, evidence, data, run_id = passed
    target = data / "outside.json"
    target.write_text("{}")
    link = data / "runs" / run_id / "team/sim-01/simulation.json"
    link.unlink()
    link.symlink_to(target)
    call(evidence, "get_stage_artifact",
         {"run_id": run_id, "stage": "sim", "attempt": 1, "filename": "simulation.json"}, expect_error=True)
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is False


def snapshot_tree(data):
    """Files (mtime, size, sha256) AND directories (mtime, entry names): any create/delete/rewrite shows up."""
    snap = {}
    for p in sorted(Path(data).rglob("*")):
        if p.is_symlink():
            snap[str(p)] = ("link", os.readlink(p))
        elif p.is_dir():
            snap[str(p)] = ("dir", p.stat().st_mtime_ns, tuple(sorted(c.name for c in p.iterdir())))
        else:
            snap[str(p)] = ("file", p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    return snap


def exercise_every_evidence_tool(evidence, run_id, artifact_ok=True):
    call(evidence, "list_runs")
    for tool in ("get_run", "get_feedback_log", "verify_run_artifacts", "get_acceptance_summary",
                 "get_execution_status"):
        call(evidence, tool, {"run_id": run_id})
    base = {"run_id": run_id, "stage": "sim", "attempt": 1}
    call(evidence, "get_stage_journal", {**base, "stage": "execution"})
    call(evidence, "get_stage_artifact", {**base, "filename": "artifact.zip"}, expect_error=not artifact_ok)
    call(evidence, "get_stage_artifact", {**base, "stage": "output", "attempt": 1, "filename": "verification.json"},
         )
    # error paths
    call(evidence, "get_stage_artifact", {**base, "filename": "run.json"}, expect_error=True)
    call(evidence, "get_stage_artifact", {**base, "attempt": 99, "filename": "artifact.zip"}, expect_error=True)
    call(evidence, "get_stage_journal", {**base, "stage": "../.."}, expect_error=True)
    call(evidence, "get_run", {"run_id": "0" * 32}, expect_error=True)


def test_evidence_server_never_writes(passed):
    """Run every evidence tool (success and error paths) and assert files AND directories are unchanged."""
    _, evidence, data, run_id = passed
    before = snapshot_tree(data)
    exercise_every_evidence_tool(evidence, run_id)
    assert snapshot_tree(data) == before


def test_evidence_server_never_writes_on_tampered_runs(passed):
    _, evidence, data, run_id = passed
    base = data / "runs" / run_id
    raw = bytearray((base / "team/sim-01/artifact.zip").read_bytes())
    raw[len(raw) // 2] ^= 1
    (base / "team/sim-01/artifact.zip").write_bytes(bytes(raw))
    shutil.rmtree(base / "team/vision-01")
    (base / "team/feedback-01.json").unlink()
    (base / "team/stray").mkdir()
    (base / "team/act-02/simulation.json").symlink_to(data)
    before = snapshot_tree(data)
    exercise_every_evidence_tool(evidence, run_id, artifact_ok=False)
    assert call(evidence, "verify_run_artifacts", {"run_id": run_id})["ok"] is False
    assert snapshot_tree(data) == before


@pytest.mark.parametrize("module", [mcp_evidence, journal_check], ids=["mcp_evidence", "journal_check"])
def test_evidence_module_has_no_runtime_or_write_paths(module):
    import ast
    tree = ast.parse(Path(module.__file__).read_text())
    imported = set()
    attrs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            attrs.add(node.func.id)
    assert not {"runtime", "Runtime", "sqlite3", "fcntl", "shutil", "tempfile"} & {i.split(".")[-1] for i in imported}
    assert not {"write_text", "write_bytes", "mkdir", "unlink", "rmdir", "replace", "rename", "open", "atomic",
                "rmtree", "flock", "symlink_to", "touch"} & attrs
    if module is journal_check:  # the neutral module must not import mcp at all
        assert not [i for i in imported if i == "mcp" or i.startswith("mcp.")]


# ---------------------------------------------------------------------------
# Review fixes. Helpers
# ---------------------------------------------------------------------------
def edit_run(data, run_id, mutate):
    path = Path(data) / "runs" / run_id / "run.json"
    state = json.loads(path.read_text())
    mutate(state)
    path.write_text(json.dumps(state))


def verify(evidence, run_id):
    return call(evidence, "verify_run_artifacts", {"run_id": run_id})


def started(runs, cid, **extra):
    out = call(runs, "start_run", {"client_request_id": cid, "request": MICHELSON, **extra})
    wait_idle(runs)
    return out["run_id"]


@pytest.fixture
def failed(servers):
    """A legitimately failed run: the position never converges, so the router stops at the retry budget."""
    runs, evidence, data = servers
    run_id = started(runs, "failed1", algorithm_options={"position_response": 0.0})
    assert run_json(data, run_id)["status"] == "failed"
    return runs, evidence, data, run_id


def crash_after_receipt(monkeypatch, runs, data, cid="crash-receipt"):
    """The executor writes plan.json and the receipt, then the process dies before completion.json."""
    real = algorithm_sim.atomic

    def dying(path, payload):
        real(path, payload)
        if path.parent.name == "execution-01" and path.name == "execution_receipt.json":
            raise RuntimeError("process died after the receipt was written")
    monkeypatch.setattr(algorithm_sim, "atomic", dying)
    run_id = call(runs, "start_run", {"client_request_id": cid, "request": MICHELSON})["run_id"]
    wait_idle(runs)
    monkeypatch.setattr(algorithm_sim, "atomic", real)
    return run_id


# ---- clean runs keep verifying ----------------------------------------------
@pytest.mark.parametrize("fault", list(mcp_runs.FAULTS))
def test_clean_and_legitimately_failed_runs_verify_ok(servers, fault):
    runs, evidence, data = servers
    run_id = started(runs, "f-" + fault, team_fault=fault)
    status = run_json(data, run_id)["status"]
    assert status in ("algorithm_passed", "failed"), status
    assert run_json(data, run_id)["provider"] == "replay"
    v = verify(evidence, run_id)
    assert v["ok"] is True and v["issues"] == [], (fault, status, v["issues"])
    summary = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert ("workflow complete" in summary["workflow_status"]) == (status == "algorithm_passed")
    if status == "failed":
        assert "not complete" in summary["workflow_status"] and summary["real_failures"]


def test_failed_run_verifies_as_consistent_but_failed(failed):
    _, evidence, _, run_id = failed
    v = verify(evidence, run_id)
    assert v["ok"] is True, v["issues"]
    assert v["expected_final"] == "failed" and v["receipt_bound"] is None
    assert call(evidence, "get_run", {"run_id": run_id})["error"]["code"] == "ALGORITHM_POSITION_MISMATCH"
    assert call(evidence, "get_feedback_log", {"run_id": run_id})["consistent"] is True


# ---- H1: missing / extra journal directories -------------------------------------
@pytest.mark.parametrize("victim", ["execution-01", "vision-01", "sim-02", "done-01", "controller-03", "input-01",
                                    "mode-01"])
def test_h1_deleted_journal_directory_fails_verification(passed, victim):
    _, evidence, data, run_id = passed
    assert verify(evidence, run_id)["ok"] is True
    shutil.rmtree(data / "runs" / run_id / "team" / victim)
    bad = verify(evidence, run_id)
    assert bad["ok"] is False
    assert any(victim.split("-")[0] in i and ("missing" in i or "no " in i or "journal" in i) for i in bad["issues"]), \
        bad["issues"]


def test_h1_deleting_the_execution_journal_does_not_pass_and_is_not_accepted(passed):
    _, evidence, data, run_id = passed
    shutil.rmtree(data / "runs" / run_id / "team/execution-01")
    bad = verify(evidence, run_id)
    assert bad["ok"] is False
    assert bad["plan_bytes_identical"] is not True and bad["receipt_bound"] is not True
    assert call(evidence, "get_execution_status", {"run_id": run_id})["execution_status"] != "accepted"


def test_h1_extra_journal_directory_fails(passed):
    _, evidence, data, run_id = passed
    team = data / "runs" / run_id / "team"
    shutil.copytree(team / "vision-01", team / "vision-02")
    bad = verify(evidence, run_id)
    assert bad["ok"] is False and any("vision" in i for i in bad["issues"]), bad["issues"]


def test_h1_whole_team_directory_removed(passed):
    _, evidence, data, run_id = passed
    shutil.rmtree(data / "runs" / run_id / "team")
    assert verify(evidence, run_id)["ok"] is False


def test_h1_passed_requires_bound_receipt_and_no_none_results(passed):
    _, evidence, data, run_id = passed
    (data / "runs" / run_id / "team/execution-01/execution_receipt.json").unlink()
    bad = verify(evidence, run_id)
    assert bad["ok"] is False and bad["receipt_bound"] is not True


# ---- H2: run.json tampering ------------------------------------------------------
def test_h2a_erased_failure_history_is_detected_and_recoverable_from_journals(passed):
    _, evidence, data, run_id = passed
    base = data / "runs" / run_id

    def wipe(s):
        s["retry_log"], s["feedback"], s["retry_counts"] = [], [], {}
    edit_run(data, run_id, wipe)
    for p in (base / "team").glob("feedback-*.json"):
        p.unlink()
    assert verify(evidence, run_id)["ok"] is False
    log = call(evidence, "get_feedback_log", {"run_id": run_id})
    assert log["consistent"] is False
    codes = [e["error"]["code"] for e in log["failed_events"]]
    assert codes == ["ALGORITHM_POSITION_MISMATCH", "ALGORITHM_OPTICAL_MISMATCH"]
    assert any("output-01" in i for i in log["issues"])


def test_h2b_edited_feedback_message_is_detected_and_journal_original_is_reported(passed):
    _, evidence, data, run_id = passed
    original = run_json(data, run_id)["feedback"][0]["message"]
    edit_run(data, run_id, lambda s: s["feedback"][0].update(message="all good, nothing happened"))
    bad = verify(evidence, run_id)
    assert bad["ok"] is False and any("feedback" in i for i in bad["issues"])
    log = call(evidence, "get_feedback_log", {"run_id": run_id})
    assert log["consistent"] is False
    first = log["decisions"][0]
    assert first["original_failure"]["message"] == original
    assert first["run_json_failure_agrees"] is False


def test_h2c_failed_run_relabelled_as_passed_is_detected(failed):
    _, evidence, data, run_id = failed
    edit_run(data, run_id, lambda s: s.update(status="algorithm_passed", error=None))
    bad = verify(evidence, run_id)
    assert bad["ok"] is False
    summary = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert "workflow complete" not in summary["workflow_status"]
    assert summary["evidence_verified"] is False and summary["evidence_issues"]


def test_h2_other_run_json_edits_are_detected(passed):
    _, evidence, data, run_id = passed
    for mutate in (lambda s: s["attempts"].update(controller=2), lambda s: s["retry_counts"].update(act=0),
                   lambda s: s.update(plan_sha256="0" * 64), lambda s: s["retry_log"].pop(),
                   lambda s: s.update(status="failed")):
        original = (data / "runs" / run_id / "run.json").read_text()
        edit_run(data, run_id, mutate)
        assert verify(evidence, run_id)["ok"] is False
        (data / "runs" / run_id / "run.json").write_text(original)
    assert verify(evidence, run_id)["ok"] is True


# ---- H3: execution status never more favourable than the evidence --------------------
def test_h3_crash_after_receipt_before_completion_is_unknown_not_accepted(servers, monkeypatch):
    runs, evidence, data = servers
    run_id = crash_after_receipt(monkeypatch, runs, data)
    assert run_json(data, run_id)["status"] == "interrupted"
    exec_dir = data / "runs" / run_id / "team/execution-01"
    assert (exec_dir / "execution_receipt.json").exists() and not (exec_dir / "completion.json").exists()
    interrupted = call(evidence, "get_execution_status", {"run_id": run_id})
    assert interrupted["execution_status"] == "unknown" and interrupted["marker_present"] is True
    assert interrupted["marker"]["plan_sha256"]
    assert verify(evidence, run_id)["ok"] is True  # an interrupted run is a consistent, incomplete chain
    call(runs, "continue_run", {"run_id": run_id})
    wait_idle(runs)
    final = run_json(data, run_id)
    assert final["status"] == "failed" and final["error"]["code"] == "ALGORITHM_EXECUTION_UNKNOWN"
    assert (exec_dir / "execution_receipt.json").exists()
    status = call(evidence, "get_execution_status", {"run_id": run_id})
    assert status["execution_status"] == "unknown", status
    v = verify(evidence, run_id)
    assert v["ok"] is True, v["issues"]
    assert v["receipt_bound"] is None  # a stray receipt is not evidence of acceptance
    summary = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert "not complete" in summary["workflow_status"]


@pytest.mark.parametrize("tamper", ["motions_sent", "plan_hash", "artifact_sha256", "extra_field"])
def test_h3_edited_receipt_is_not_accepted(passed, tamper):
    _, evidence, data, run_id = passed
    path = data / "runs" / run_id / "team/execution-01/execution_receipt.json"
    receipt = json.loads(path.read_text())
    if tamper == "motions_sent":
        receipt["motions_sent"] = 5
    elif tamper == "plan_hash":
        receipt["plan_hash"] = "0" * 64
    elif tamper == "artifact_sha256":
        receipt["artifact_sha256"] = "1" * 64
    else:
        receipt["note"] = "edited"
    path.write_text(json.dumps(receipt))
    status = call(evidence, "get_execution_status", {"run_id": run_id})
    assert status["execution_status"] == "unknown" and status["reason"]
    assert verify(evidence, run_id)["ok"] is False


@pytest.mark.parametrize("mutation", ["result", "plan_sha256", "artifact_sha256", "unknown_code", "no_dir"])
def test_h3_unbound_or_flagged_runs_are_not_accepted(passed, mutation):
    _, evidence, data, run_id = passed
    if mutation == "result":
        edit_run(data, run_id, lambda s: s["result"].update(motions_sent=5))
    elif mutation == "plan_sha256":
        edit_run(data, run_id, lambda s: s.update(plan_sha256="2" * 64))
    elif mutation == "artifact_sha256":
        edit_run(data, run_id, lambda s: s.update(artifact_sha256="3" * 64))
    elif mutation == "unknown_code":
        edit_run(data, run_id, lambda s: s.update(error={"stage": "execution", "code": "ALGORITHM_EXECUTION_UNKNOWN",
                                                         "message": "x"}))
    else:
        shutil.rmtree(data / "runs" / run_id / "team/execution-01")
    assert call(evidence, "get_execution_status", {"run_id": run_id})["execution_status"] == "unknown"


def test_h3_journal_error_unknown_wins_over_files(servers, monkeypatch):
    """Even with every file present and hashes matching, an UNKNOWN code in the journal means unknown."""
    runs, evidence, data = servers
    run_id = crash_after_receipt(monkeypatch, runs, data, "crash2")
    call(runs, "continue_run", {"run_id": run_id})
    wait_idle(runs)
    edit_run(data, run_id, lambda s: s.update(error=None, status="algorithm_passed"))
    assert call(evidence, "get_execution_status", {"run_id": run_id})["execution_status"] == "unknown"


def test_h3_rejected_requires_a_verified_journal(servers):
    runs, evidence, data = servers
    run_id = started(runs, "rej", team_fault="execution")
    assert call(evidence, "get_execution_status", {"run_id": run_id})["execution_status"] == "rejected"
    (data / "runs" / run_id / "team/execution-01/event.json").write_text("{}")
    assert call(evidence, "get_execution_status", {"run_id": run_id})["execution_status"] == "unknown"


# ---- M3: run-root files ---------------------------------------------------------------
def test_m3_input_and_result_json_are_checked(passed):
    _, evidence, data, run_id = passed
    base = data / "runs" / run_id
    original_input = (base / "input.json").read_bytes()
    (base / "input.json").write_bytes(original_input + b" ")
    bad = verify(evidence, run_id)
    assert bad["ok"] is False and any("input.json" in i for i in bad["issues"])
    (base / "input.json").write_bytes(original_input)
    assert verify(evidence, run_id)["ok"] is True
    result = json.loads((base / "result.json").read_text())
    result["motions_sent"] = 9
    (base / "result.json").write_text(json.dumps(result))
    bad = verify(evidence, run_id)
    assert bad["ok"] is False and any("result.json" in i for i in bad["issues"])


def test_m3_run_root_plan_json_legitimately_differs_and_is_not_compared(passed):
    _, evidence, data, run_id = passed
    base = data / "runs" / run_id
    assert (base / "plan.json").read_bytes() != (base / "team/controller-03/plan.json").read_bytes()
    assert verify(evidence, run_id)["ok"] is True


# ---- M4: acceptance summary -------------------------------------------------------------
def test_m4_summary_includes_verdict_and_says_rerouted_not_recovered(passed):
    _, evidence, data, run_id = passed
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert s["evidence_verified"] is True and "workflow complete" in s["workflow_status"]
    assert "recovered_by_router" not in s
    assert [(r["from"], r["to"]) for r in s["rerouted_by_router"]] == [("output", "controller"), ("output", "act")]


def test_m4_failed_run_reroutes_are_not_called_recovered(failed):
    _, evidence, _, run_id = failed
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert "recovered_by_router" not in s and s["rerouted_by_router"]
    assert s["raw_status"] == "failed" and "not complete" in s["workflow_status"]


def test_m4_unverified_passed_run_is_not_workflow_complete(passed):
    _, evidence, data, run_id = passed
    shutil.rmtree(data / "runs" / run_id / "team/done-01")
    s = call(evidence, "get_acceptance_summary", {"run_id": run_id})
    assert s["raw_status"] == "algorithm_passed" and "workflow complete" not in s["workflow_status"]
    assert s["evidence_verified"] is False


def test_m4_pending_requirements_are_a_union(passed):
    _, evidence, data, run_id = passed
    edit_run(data, run_id, lambda s: (s["verification"].update(pending_requirements=["from-verification"]),
                                      s["algorithm_experiment"].update(pending_requirements=["from-experiment"])))
    core = call(evidence, "get_acceptance_summary", {"run_id": run_id})["core_unaccepted"]
    assert "from-verification" in core and "from-experiment" in core


# ---- L1 / L2 / L3 -------------------------------------------------------------------------
def test_l1_oversized_run_json_is_refused_not_slurped(passed):
    _, evidence, data, run_id = passed
    path = data / "runs" / run_id / "run.json"
    state = json.loads(path.read_text())
    state["padding"] = "x" * 9_000_000
    path.write_text(json.dumps(state))
    assert "exceeds" in call(evidence, "get_run", {"run_id": run_id}, expect_error=True)
    bad = verify(evidence, run_id)
    assert bad["ok"] is False and any("run.json" in i for i in bad["issues"])


def test_l1_oversized_feedback_file_is_reported(passed):
    _, evidence, data, run_id = passed
    path = data / "runs" / run_id / "team/feedback-01.json"
    path.write_text(json.dumps({"pad": "x" * 1_100_000}))
    assert verify(evidence, run_id)["ok"] is False
    assert call(evidence, "get_feedback_log", {"run_id": run_id})["consistent"] is False


@pytest.mark.parametrize("bad_path", ["team/sim-03/../sim-03/artifact.zip", "../x/artifact.zip",
                                      "team/sim-3/artifact.zip", "/etc/passwd", 5])
def test_l2_artifact_path_must_match_strict_pattern(passed, bad_path):
    _, evidence, data, run_id = passed
    edit_run(data, run_id, lambda s: s.update(artifact_path=bad_path))
    bad = verify(evidence, run_id)
    assert bad["ok"] is False and any("artifact_path is not team/sim-NN/artifact.zip" in i for i in bad["issues"])


def test_l3_symlinked_journal_directory_is_refused(passed):
    _, evidence, data, run_id = passed
    team = data / "runs" / run_id / "team"
    elsewhere = data / "elsewhere"
    shutil.copytree(team / "sim-02", elsewhere)
    shutil.rmtree(team / "sim-02")
    (team / "sim-02").symlink_to(elsewhere, target_is_directory=True)
    bad = verify(evidence, run_id)
    assert bad["ok"] is False and any("symlink" in i for i in bad["issues"]), bad["issues"]
    call(evidence, "get_stage_journal", {"run_id": run_id, "stage": "sim", "attempt": 2}, expect_error=True)


def test_l3_attempts_sort_numerically(tmp_path):
    team = tmp_path / "team"
    for n in (100, 99, 9, 10):
        (team / f"execution-{n:02d}").mkdir(parents=True)
    store = mcp_evidence.Store(tmp_path)
    assert [p.name for p in store.execution_dirs(tmp_path)] == [
        "execution-09", "execution-10", "execution-99", "execution-100"]


# ---- M1: panel data dir ---------------------------------------------------------------------
def test_m1_refuses_panel_dir_aliases_inside_and_containing(tmp_path, monkeypatch):
    panel = tmp_path / "panel"
    (panel / "inner").mkdir(parents=True)
    monkeypatch.setattr(mcp_runs, "DATA", panel)
    for bad in (panel, panel / "inner", panel / "new-child", tmp_path, tmp_path / "panel" / ".." / "panel"):
        with pytest.raises(ValueError, match="panel"):
            mcp_runs.build_server(bad)
    link = tmp_path / "alias"
    link.symlink_to(panel, target_is_directory=True)
    with pytest.raises(ValueError, match="panel"):
        mcp_runs.build_server(link)
    upper = tmp_path / "PANEL"
    if upper.exists():  # case-insensitive filesystem (macOS default)
        with pytest.raises(ValueError, match="panel"):
            mcp_runs.build_server(upper)
    mcp_runs.build_server(tmp_path / "isolated")  # a genuinely separate directory is fine


# ---- L4 / L5 -------------------------------------------------------------------------------------
def test_l4_worker_exception_is_surfaced(servers, monkeypatch):
    runs, _, data = servers

    def explode(self, run_id, resume=None, continue_run=False):
        raise ValueError("worker blew up")
    monkeypatch.setattr(Runtime, "execute", explode)
    run_id = started(runs, "boom")
    assert run_json(data, run_id)["status"] == "queued"
    err = call(runs, "get_worker_error", {"run_id": run_id})
    assert err["worker_error"]["type"] == "ValueError" and "worker blew up" in err["worker_error"]["message"]
    assert err["active"] is False
    assert err["status"] == "queued" and err["display_state"] == "queued"  # stored status, not an invented label
    call(runs, "get_worker_error", {"run_id": "0" * 32}, expect_error=True)
    monkeypatch.undo()
    healthy = started(runs, "fine")
    ok = call(runs, "get_worker_error", {"run_id": healthy})
    assert ok["worker_error"] is None and ok["status"] == "algorithm_passed"
    assert ok["display_state"] == "algorithm_passed" and ok["active"] is False


def test_l4_standalone_display_state_comes_from_its_own_active_set(servers, monkeypatch):
    runs, _, data = servers
    entered, release = threading.Event(), threading.Event()
    original = AlgorithmServices.sim

    def slow_sim(self, state, path, attempt):
        entered.set()
        assert release.wait(20)
        return original(self, state, path, attempt)
    monkeypatch.setattr(AlgorithmServices, "sim", slow_sim)
    run_id = call(runs, "start_run", {"client_request_id": "disp", "request": MICHELSON})["run_id"]
    assert entered.wait(20)
    try:
        err = call(runs, "get_worker_error", {"run_id": run_id})
        assert err["active"] is True and err["display_state"] == "executing" and err["status"] == "running"
        assert call(runs, "pause_run", {"run_id": run_id})["status"] == "pause_requested"
        err = call(runs, "get_worker_error", {"run_id": run_id})
        assert err["control_request"] == "pause" and err["display_state"] == "pause_requested"
    finally:
        release.set()
    wait_idle(runs)
    done = call(runs, "get_worker_error", {"run_id": run_id})
    assert done["active"] is False and done["status"] == "paused" and done["display_state"] == "paused"


# ---- H1 / L2 (standalone backend) ------------------------------------------------------------------
def panel_made_run(runs, data, **overrides):
    """A run created the way the web panel creates one: straight through Runtime.create, never via start_run."""
    from optics_graph.runtime import RunInput
    fields = dict(request=MICHELSON, profile="auto", mode="github_local", simulation_backend="algorithm_v1",
                  provider="replay", algorithm_options={"delay_s": 0.0})
    fields.update(overrides)
    return runs._optics_service.runtime.create(RunInput(**fields))


def test_h1_standalone_refuses_runs_not_created_by_start_run_or_not_offline(servers):
    runs, _, data = servers
    service = runs._optics_service
    unindexed = panel_made_run(runs, data)
    openai_run = panel_made_run(runs, data, provider="openai")
    for run_id in (unindexed, openai_run):
        for name in ("continue_run", "pause_run", "stop_run", "get_worker_error"):
            assert "refused" in call(runs, name, {"run_id": run_id}, expect_error=True), (name, run_id)
        assert not (data / "runs" / run_id / "control.json").exists()
        assert run_json(data, run_id)["status"] == "queued" and not service.active
    # indexed but the stored run is no longer offline: refused as well
    mine = started(runs, "mine")
    edit_run(data, mine, lambda s: s.update(provider="openai", status="interrupted"))
    assert "refused" in call(runs, "continue_run", {"run_id": mine}, expect_error=True)
    edit_run(data, mine, lambda s: s.update(provider="replay", status="paused"))
    assert call(runs, "continue_run", {"run_id": mine})["status"] == "resume_submitted"
    wait_idle(runs)


def test_l2_every_runs_tool_rejects_unknown_arguments_in_both_schema_and_call(servers):
    runs, _, data = servers
    message = call(runs, "start_run", {"client_request_id": "x", "request": MICHELSON, "provider": "openai"},
                   expect_error=True)
    assert "provider" in message and "Extra inputs are not permitted" in message
    assert not (data / "runs").exists() or not list((data / "runs").iterdir())
    for name in ("continue_run", "pause_run", "stop_run", "get_worker_error"):
        assert "Extra inputs" in call(runs, name, {"run_id": "0" * 32, "extra": 1}, expect_error=True), name
    for tool in list_tools(runs):
        assert tool.inputSchema.get("additionalProperties") is False, tool.name


def crash_after_marker(monkeypatch):
    """The process dies right after the run dir and its mcp_origin.json tag exist, before the index is finished."""
    real_marker = mcp_runs.RunsService._write_marker

    def marker_then_die(self, run_id, client_request_id, nonce):
        real_marker(self, run_id, client_request_id, nonce)
        raise OSError("process died right after the marker was written")
    monkeypatch.setattr(mcp_runs.RunsService, "_write_marker", marker_then_die)
    return real_marker


def test_l5_crash_between_create_and_index_write_is_adopted_on_retry(servers, monkeypatch):
    """L6: a tagged run (marker with this request's client_request_id + nonce) is adopted."""
    runs, evidence, data = servers
    real_marker = crash_after_marker(monkeypatch)
    call(runs, "start_run", {"client_request_id": "orphan", "request": MICHELSON}, expect_error=True)
    monkeypatch.setattr(mcp_runs.RunsService, "_write_marker", real_marker)
    assert len(list((data / "runs").iterdir())) == 1
    index = json.loads((data / mcp_runs.INDEX_NAME).read_text())
    assert index["orphan"]["pending"] is True and index["orphan"]["nonce"]
    only = next((data / "runs").iterdir())
    assert json.loads((only / mcp_runs.MARKER_NAME).read_text()) == {
        "client_request_id": "orphan", "nonce": index["orphan"]["nonce"]}
    out = call(runs, "start_run", {"client_request_id": "orphan", "request": MICHELSON})
    wait_idle(runs)
    assert len(list((data / "runs").iterdir())) == 1, "retry created a second run"
    assert (data / "runs" / out["run_id"]).is_dir()
    assert run_json(data, out["run_id"])["status"] == "algorithm_passed"
    again = call(runs, "start_run", {"client_request_id": "orphan", "request": MICHELSON})
    assert again["run_id"] == out["run_id"] and again["idempotent_replay"] is True
    assert call(evidence, "verify_run_artifacts", {"run_id": out["run_id"]})["ok"] is True  # marker is harmless


def test_l6_crash_between_create_and_marker_is_not_adopted_and_is_reported(servers, monkeypatch):
    runs, _, data = servers

    def die(self, run_id, client_request_id, nonce):
        raise OSError("process died between create and the marker")
    monkeypatch.setattr(mcp_runs.RunsService, "_write_marker", die)
    call(runs, "start_run", {"client_request_id": "amb", "request": MICHELSON}, expect_error=True)
    monkeypatch.undo()
    (untagged,) = list((data / "runs").iterdir())
    message = call(runs, "start_run", {"client_request_id": "amb", "request": MICHELSON}, expect_error=True)
    assert "not adopting" in message and untagged.name in message and mcp_runs.MARKER_NAME in message
    assert len(list((data / "runs").iterdir())) == 1  # no second run, and the ambiguous one was not started
    assert run_json(data, untagged.name)["status"] == "queued"
    assert not runs._optics_service.active


def test_l6_a_panel_created_queued_run_that_matches_is_never_adopted(servers, monkeypatch):
    """Crash before Runtime.create leaves a pending index entry; a panel-created queued run with identical
    fields then appears. The old code adopted (and executed) it."""
    runs, _, data = servers
    service = runs._optics_service

    def die_before_create(self, request):
        raise OSError("process died before the run existed")
    monkeypatch.setattr(Runtime, "create", die_before_create)
    call(runs, "start_run", {"client_request_id": "pend", "request": MICHELSON}, expect_error=True)
    monkeypatch.undo()
    assert not (data / "runs").exists() or not list((data / "runs").iterdir())
    from optics_graph.runtime import RunInput
    panel_run = service.runtime.create(RunInput(
        request=MICHELSON, profile="auto", mode="github_local", simulation_backend="algorithm_v1",
        provider="replay", team_fault="none", algorithm_options={"delay_s": 0.0}))
    message = call(runs, "start_run", {"client_request_id": "pend", "request": MICHELSON}, expect_error=True)
    assert "not adopting" in message and panel_run in message
    assert run_json(data, panel_run)["status"] == "queued" and not service.active
    assert not (data / "runs" / panel_run / mcp_runs.MARKER_NAME).exists()


def test_l6_a_run_tagged_for_another_request_is_not_adopted(servers, monkeypatch):
    runs, _, data = servers
    real_marker = crash_after_marker(monkeypatch)
    call(runs, "start_run", {"client_request_id": "first", "request": MICHELSON}, expect_error=True)
    monkeypatch.setattr(mcp_runs.RunsService, "_write_marker", real_marker)
    # a second client crashed before create with the same request body: it must not take the first one's run
    index = json.loads((data / mcp_runs.INDEX_NAME).read_text())
    index["second"] = {**index["first"], "nonce": "f" * 32}
    (data / mcp_runs.INDEX_NAME).write_text(json.dumps(index))
    out = call(runs, "start_run", {"client_request_id": "second", "request": MICHELSON})
    wait_idle(runs)
    first_run = next(p.name for p in (data / "runs").iterdir()
                     if json.loads((p / mcp_runs.MARKER_NAME).read_text())["client_request_id"] == "first")
    assert out["run_id"] != first_run and len(list((data / "runs").iterdir())) == 2


def test_l5_pending_entry_with_different_body_is_refused(servers, monkeypatch):
    runs, _, data = servers
    real_create = Runtime.create
    monkeypatch.setattr(Runtime, "create", lambda self, r: (real_create(self, r), (_ for _ in ()).throw(OSError("x")))[0])
    call(runs, "start_run", {"client_request_id": "p", "request": MICHELSON}, expect_error=True)
    monkeypatch.setattr(Runtime, "create", real_create)
    call(runs, "start_run", {"client_request_id": "p", "request": MICHELSON + "!"}, expect_error=True)


def test_l5_index_is_fsynced_and_corrupt_index_names_the_file(servers, monkeypatch):
    runs, _, data = servers
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(mcp_runs.os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    started(runs, "durable")
    assert len(synced) >= 2  # the index file and its directory
    (data / mcp_runs.INDEX_NAME).write_text("{not json")
    message = call(runs, "start_run", {"client_request_id": "next", "request": MICHELSON}, expect_error=True)
    assert mcp_runs.INDEX_NAME in message and "corrupt" in message
    (data / mcp_runs.INDEX_NAME).write_text("[1, 2]")
    assert mcp_runs.INDEX_NAME in call(runs, "start_run", {"client_request_id": "n2", "request": MICHELSON},
                                       expect_error=True)


def test_runs_started_through_the_server_use_replay_provider(passed):
    _, _, data, run_id = passed
    state = run_json(data, run_id)
    assert state["provider"] == "replay" and state["simulation_backend"] == "algorithm_v1"
    assert json.loads((data / "runs" / run_id / "input.json").read_text())["provider"] == "replay"
