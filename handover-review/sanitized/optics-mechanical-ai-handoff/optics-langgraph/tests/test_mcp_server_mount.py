"""optics-runs mounted at /mcp inside the web server (Task A), the shared journal check (Task B) and the
run display-state fix (Task C). The HTTP app is driven in-process with FastAPI's TestClient (used as a
context manager so the lifespan starts the MCP session manager); no port is opened and no model is called.
"""
import asyncio
import builtins
import fcntl
import importlib
import logging
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from optics_graph import algorithm_sim, gateway, journal_check, mcp_evidence, mcp_runs, server, settings
from optics_graph.algorithm_sim import AlgorithmServices
from optics_graph.runtime import Runtime, RunInput
from optics_graph.team_artifacts import digest, encode

MICHELSON = "Assemble a 15-component complete Michelson interferometer."
LOCAL = {"x-optics-local": "1"}
MCP_HEADERS = {**LOCAL, "accept": "application/json, text/event-stream", "content-type": "application/json"}


def _boom(*args, **kwargs):
    raise AssertionError("cloud client constructed: the MCP wrap must never call a model")


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
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
    monkeypatch.setattr(server, "public_settings", lambda: {"model": "none", "key_configured": False})


@pytest.fixture
def app_env(tmp_path):
    runtime = Runtime(tmp_path)
    app = server.create_app(runtime)
    with TestClient(app) as client:
        yield client, runtime, tmp_path


def rpc(client, method, params=None, *, headers=MCP_HEADERS):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    return client.post("/mcp", content=json.dumps(body), headers=headers)


def tool(client, name, args=None, *, expect_error=False):
    response = rpc(client, "tools/call", {"name": name, "arguments": args or {}})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    if expect_error:
        assert result["isError"], f"{name}{args} should have been rejected"
        return result["content"][0]["text"]
    assert not result.get("isError"), result["content"][0]["text"]
    return result["structuredContent"]


def wait_for(predicate, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise AssertionError("condition not reached")


def api_run(client, run_id):
    return client.get(f"/api/runs/{run_id}").json()


def wait_finished(client, run_id):
    return wait_for(lambda: (lambda r: r if not r["active"] else None)(api_run(client, run_id)))


# ---- Task A: reachability and the local-only guard -------------------------------------------------
def test_mcp_reachable_with_header_and_refused_without(app_env):
    client, _, _ = app_env
    listed = rpc(client, "tools/list")
    assert listed.status_code == 200
    assert {t["name"] for t in listed.json()["result"]["tools"]} == {
        "start_run", "continue_run", "get_worker_error", "pause_run", "stop_run",
        "get_pending_decision", "submit_decision"}
    plain = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
    assert rpc(client, "tools/list", headers=plain).status_code == 403
    assert rpc(client, "tools/list", headers={**MCP_HEADERS, "origin": "https://evil.example"}).status_code == 403
    assert client.post("/mcp", content="{}", headers={**plain, "x-optics-local": "0"}).status_code == 403
    assert client.get("/healthz").status_code == 200


def test_m3_every_method_on_mcp_needs_the_local_header_and_get_is_405(app_env):
    """Regression: GET /mcp with Accept: text/event-stream and no x-optics-local header used to return 200 and
    hold an SSE stream open forever (the old test only passed because it sent no Accept header)."""
    client, _, _ = app_env
    sse = {"accept": "text/event-stream"}
    assert client.get("/mcp", headers=sse).status_code == 403
    assert client.get("/mcp").status_code == 403
    assert client.get("/mcp", headers={**sse, "x-optics-local": "0"}).status_code == 403
    assert client.get("/mcp", headers={**sse, **LOCAL, "origin": "https://evil.example"}).status_code == 403
    for method in ("GET", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS"):
        assert client.request(method, "/mcp", headers=sse).status_code == 403, method
        with_header = client.request(method, "/mcp", headers={**sse, **LOCAL})
        assert with_header.status_code == 405 and with_header.headers["allow"] == "POST", method
    assert client.get("/mcp/", headers=sse, follow_redirects=False).status_code in (307, 403)
    assert client.get("/mcp/", headers={**sse, **LOCAL}, follow_redirects=True).status_code in (404, 405)


def test_m3_real_streamablehttp_client_round_trip_needs_no_get_stream(tmp_path):
    """A real MCP streamable-HTTP client (initialize, list_tools, call_tool) works against the stateless JSON
    endpoint although GET /mcp is 405: in stateless mode the client never opens a server-initiated stream."""
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    app = server.create_app(Runtime(tmp_path))
    methods = []

    @app.middleware("http")
    async def spy(request, call_next):
        methods.append(request.method)
        return await call_next(request)

    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers=headers, timeout=timeout,
                                 auth=auth, base_url="http://127.0.0.1:8766")

    async def go():
        async with app.router.lifespan_context(app):
            with anyio_timeout(30):
                async with streamablehttp_client("http://127.0.0.1:8766/mcp", headers=LOCAL,
                                                 httpx_client_factory=factory) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        names = {t.name for t in (await session.list_tools()).tools}
                        refused = await session.call_tool("get_worker_error", {"run_id": "0" * 32})
                        return names, refused.isError
    names, refused = asyncio.run(go())
    assert names == {"start_run", "continue_run", "get_worker_error", "pause_run", "stop_run",
                     "get_pending_decision", "submit_decision"}
    assert refused is True
    assert "POST" in methods and "GET" not in methods


def anyio_timeout(seconds):
    import anyio
    return anyio.fail_after(seconds)


def asgi_request(app, method, headers, body=b""):
    """Drive the ASGI app directly (no lifespan) so headers like a missing Content-Length can be controlled."""
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": "/mcp", "raw_path": b"/mcp", "query_string": b"", "server": ("127.0.0.1", 8766),
             "client": ("127.0.0.1", 1), "root_path": "",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}
    sent, messages = False, []

    async def receive():
        nonlocal sent
        if sent:
            await asyncio.sleep(3600)
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    async def go():
        await asyncio.wait_for(app(scope, receive, send), 10)
    asyncio.run(go())
    return next(m["status"] for m in messages if m["type"] == "http.response.start")


def test_l1_mcp_post_needs_a_valid_content_length(tmp_path):
    app = server.create_app(Runtime(tmp_path))
    base = {"host": "127.0.0.1:8766", "x-optics-local": "1", "accept": "application/json, text/event-stream",
            "content-type": "application/json"}
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    assert asgi_request(app, "POST", base, body) == 411  # missing Content-Length
    for bad in ("abc", "", "-1", "1_0", "+5", "1.5", " 5"):
        assert asgi_request(app, "POST", {**base, "content-length": bad}, body) == 400, repr(bad)
    chunked = {**base, "transfer-encoding": "chunked"}
    assert asgi_request(app, "POST", chunked, body) == 400
    assert asgi_request(app, "POST", {**chunked, "content-length": "5"}, body) == 400
    assert asgi_request(app, "POST", {**base, "content-length": "30001"}, body) == 413
    assert asgi_request(app, "POST", {**base, "content-length": "999999999999"}, body) == 413


def test_l1_through_the_test_client_chunked_and_garbage_lengths_are_rejected(app_env):
    client, _, _ = app_env
    chunked = client.post("/mcp", content=(chunk for chunk in (b'{"jsonrpc":"2.0",', b'"id":1,"method":"tools/list"}')),
                          headers=MCP_HEADERS)
    assert chunked.status_code in (400, 411)
    assert client.post("/mcp", content=b"{}", headers={**MCP_HEADERS, "content-length": "nope"}).status_code == 400
    assert rpc(client, "tools/list").status_code == 200  # a normal request still works


def test_mcp_rejects_foreign_host_header(app_env):
    client, _, _ = app_env
    assert rpc(client, "tools/list", headers={**MCP_HEADERS, "host": "evil.example"}).status_code in (400, 421)
    assert rpc(client, "tools/list", headers={**MCP_HEADERS, "host": "127.0.0.1:8766"}).status_code == 200
    assert rpc(client, "tools/list", headers={**MCP_HEADERS, "host": "localhost:8766"}).status_code == 200


def test_server_works_and_has_no_mcp_route_when_mcp_is_not_importable(tmp_path, monkeypatch, caplog):
    real_import = builtins.__import__

    def deny(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError(f"No module named {name!r}", name="mcp")
        return real_import(name, *args, **kwargs)
    monkeypatch.delitem(sys.modules, "optics_graph.mcp_runs")
    monkeypatch.setattr(builtins, "__import__", deny)
    with caplog.at_level(logging.WARNING, logger="optics_graph.server"):
        app = server.create_app(Runtime(tmp_path))
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert [r.getMessage() for r in caplog.records if "/mcp is disabled" in r.getMessage()] and len(caplog.records) == 1
    assert "/mcp" not in {getattr(r, "path", None) for r in app.router.routes}
    with TestClient(app) as client:
        assert client.get("/healthz").json()["ok"] is True
        assert client.get("/api/runs").json() == []
        assert client.post("/mcp", content="{}", headers=MCP_HEADERS).status_code == 404
        assert client.post("/api/runs", json={}, headers=LOCAL).status_code in (202, 422)


@pytest.mark.parametrize("error", [ModuleNotFoundError("No module named 'numpy'", name="numpy"),
                                   ModuleNotFoundError("No module named 'optics_graph.mcp_runs'",
                                                       name="optics_graph.mcp_runs"),
                                   ModuleNotFoundError("No module named 'mcpx'", name="mcpx"),
                                   ImportError("cannot import name 'x' from 'mcp.server.fastmcp'", name="mcp.server"),
                                   AttributeError("SDK mismatch")])
def test_l5_only_a_missing_mcp_package_disables_the_endpoint(tmp_path, monkeypatch, error):
    real = importlib.import_module

    def fake(name, package=None):
        if name == ".mcp_runs":
            raise error
        return real(name, package)
    monkeypatch.setattr(server.importlib, "import_module", fake)
    with pytest.raises(type(error)):
        server.create_app(Runtime(tmp_path))


def test_l5_a_missing_mcp_submodule_also_disables_the_endpoint(tmp_path, monkeypatch):
    real = importlib.import_module

    def fake(name, package=None):
        if name == ".mcp_runs":
            raise ModuleNotFoundError("No module named 'mcp.server.fastmcp'", name="mcp.server.fastmcp")
        return real(name, package)
    monkeypatch.setattr(server.importlib, "import_module", fake)
    app = server.create_app(Runtime(tmp_path))
    assert "/mcp" not in {getattr(r, "path", None) for r in app.router.routes}


# ---- M4: lifespan re-entry ---------------------------------------------------------------------------
def test_m4_two_sequential_lifespans_on_one_app_both_serve_mcp(tmp_path):
    """Regression: the second lifespan raised "StreamableHTTPSessionManager .run() can only be called once"."""
    app = server.create_app(Runtime(tmp_path))
    routes_before = [r for r in app.router.routes if getattr(r, "path", None) == "/mcp"]
    assert len(routes_before) == 1
    assert client_post_without_lifespan(app) == 503  # not running outside a lifespan: a clear error, not a crash
    for round_ in range(2):
        with TestClient(app) as client:
            names = {t["name"] for t in rpc(client, "tools/list").json()["result"]["tools"]}
            assert len(names) == 7, round_
            out = tool(client, "start_run", {"client_request_id": f"life-{round_}", "request": MICHELSON,
                                             "algorithm_options": {"delay_s": 0}})
            assert wait_finished(client, out["run_id"])["status"] == "algorithm_passed"
    assert [r for r in app.router.routes if getattr(r, "path", None) == "/mcp"] == routes_before
    assert client_post_without_lifespan(app) == 503


def client_post_without_lifespan(app):
    return TestClient(app).post("/mcp", content="{}", headers=MCP_HEADERS).status_code


def test_m4_presentation_close_runs_even_if_the_lifespan_body_fails(tmp_path, monkeypatch):
    class Presentation:
        closed = []

        def close(self, wait=True):
            self.closed.append(wait)
    presentation = Presentation()

    def broken(*args, **kwargs):
        raise RuntimeError("cannot build the MCP server")
    monkeypatch.setattr(mcp_runs, "build_hosted_server", broken)
    app = server.create_app(Runtime(tmp_path), presentation)
    with pytest.raises(RuntimeError, match="cannot build"):
        with TestClient(app):
            pass
    assert presentation.closed == [False]


def test_m4_presentation_close_runs_when_the_session_manager_cannot_start(tmp_path, monkeypatch):
    class Presentation:
        closed = []

        def close(self, wait=True):
            self.closed.append(wait)
    presentation = Presentation()
    real = mcp_runs.build_hosted_server

    def build(*args, **kwargs):
        mcp = real(*args, **kwargs)
        mcp.streamable_http_app()

        class Exploding:
            def run(self):
                raise RuntimeError("session manager failed")
        mcp._session_manager = Exploding()
        return mcp
    monkeypatch.setattr(mcp_runs, "build_hosted_server", build)
    app = server.create_app(Runtime(tmp_path), presentation)
    with pytest.raises(RuntimeError, match="session manager failed"):
        with TestClient(app):
            pass
    assert presentation.closed == [False]


# ---- Task A: behaviour through the server's own runtime ---------------------------------------------
def test_start_via_mcp_is_active_in_api_reaches_algorithm_passed_and_evidence_verifies(app_env, monkeypatch):
    client, runtime, data = app_env
    entered, release = threading.Event(), threading.Event()
    original = AlgorithmServices.sim

    def slow_sim(self, state, path, attempt):
        entered.set()
        assert release.wait(20)
        return original(self, state, path, attempt)
    monkeypatch.setattr(AlgorithmServices, "sim", slow_sim)
    out = tool(client, "start_run", {"client_request_id": "m1", "request": MICHELSON})
    run_id = out["run_id"]
    assert out["idempotent_replay"] is False and out["source"].startswith("offline rule fixture")
    assert entered.wait(20)
    try:
        row = next(r for r in client.get("/api/runs").json() if r["run_id"] == run_id)
        assert row["active"] is True and row["display_state"] == "executing"
        detail = api_run(client, run_id)
        assert detail["active"] is True and detail["provider"] == "replay"
        assert detail["simulation_backend"] == "algorithm_v1" and detail["display_state"] == "executing"
        err = tool(client, "get_worker_error", {"run_id": run_id})
        assert err["active"] is True and err["worker_error"] is None
        assert err["display_state"] == "executing" and err["status"] == "running"
        # same conflict the HTTP route gives, as a clear tool error
        assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 409
        message = tool(client, "continue_run", {"run_id": run_id}, expect_error=True)
        assert "conflict" in message and "409" in message and "already executing" in message
    finally:
        release.set()
    final = wait_finished(client, run_id)
    assert final["status"] == "algorithm_passed", final.get("error")
    assert final["display_state"] == "algorithm_passed"  # L3: the stored status, not a generic "idle"
    done = tool(client, "get_worker_error", {"run_id": run_id})
    assert done["active"] is False and done["status"] == "algorithm_passed"
    assert done["display_state"] == "algorithm_passed"
    verdict = mcp_evidence.build_server(data)
    from tests.test_mcp_wrap import call as evidence_call  # noqa: PLC0415 - in-memory evidence client
    assert evidence_call(verdict, "verify_run_artifacts", {"run_id": run_id})["ok"] is True
    assert evidence_call(verdict, "get_execution_status", {"run_id": run_id})["execution_status"] == "accepted"
    tool(client, "continue_run", {"run_id": run_id}, expect_error=True)  # finished: nothing to resume


def test_idempotent_start_and_single_runtime_and_worker(tmp_path, monkeypatch):
    created = []
    real_init = Runtime.__init__
    monkeypatch.setattr(Runtime, "__init__", lambda self, *a, **k: (created.append(1), real_init(self, *a, **k))[1])
    pools = []
    real_pool = mcp_runs.ThreadPoolExecutor
    monkeypatch.setattr(mcp_runs, "ThreadPoolExecutor", lambda *a, **k: (pools.append(1), real_pool(*a, **k))[1])
    runtime = Runtime(tmp_path)
    assert created == [1]
    app = server.create_app(runtime)
    assert created == [1] and pools == []  # mounting /mcp built neither a second Runtime nor a worker pool
    with TestClient(app) as client:
        a = tool(client, "start_run", {"client_request_id": "same", "request": MICHELSON})
        wait_finished(client, a["run_id"])
        b = tool(client, "start_run", {"client_request_id": "same", "request": MICHELSON})
        assert b["run_id"] == a["run_id"] and b["idempotent_replay"] is True
        assert len(list((tmp_path / "runs").iterdir())) == 1
        assert (tmp_path / mcp_runs.INDEX_NAME).is_file()  # the index lives in the server's data dir
        tool(client, "start_run", {"client_request_id": "same", "request": MICHELSON + " "}, expect_error=True)
        assert len(list((tmp_path / "runs").iterdir())) == 1
        assert len(client.get("/api/runs").json()) == 1
        names = [t.name for t in threading.enumerate()]
        assert not [n for n in names if n.startswith("optics-runs")], names
        assert created == [1] and pools == []


def test_hosted_mode_keeps_the_offline_only_restrictions(app_env):
    client, _, data = app_env
    for bad in ({"initial_position_error_m": 0.0}, {"tolerance": 1}, {"position_response": 2}, {"provider": "openai"}):
        tool(client, "start_run", {"client_request_id": "bad", "request": MICHELSON, "algorithm_options": bad},
             expect_error=True)
    # L2: an unknown argument is rejected, not silently dropped (FastMCP ignores extras by default)
    message = tool(client, "start_run", {"client_request_id": "p", "request": MICHELSON, "provider": "openai"},
                   expect_error=True)
    assert "provider" in message and "Extra inputs are not permitted" in message
    for name in ("continue_run", "pause_run", "stop_run", "get_worker_error"):
        assert "Extra inputs" in tool(client, name, {"run_id": "0" * 32, "provider": "openai"}, expect_error=True)
    assert not (data / "runs").exists() or not [p for p in (data / "runs").iterdir()
                                                 if json.loads((p / "run.json").read_text())["provider"] != "replay"]
    run_id = tool(client, "start_run", {"client_request_id": "ok", "request": MICHELSON,
                                        "algorithm_options": {"delay_s": 0}})["run_id"]
    wait_finished(client, run_id)
    state = api_run(client, run_id)
    assert state["provider"] == "replay" and state["mode"] == "github_local" and state["profile"] == "auto"


def test_pause_from_http_is_seen_via_mcp_and_pause_from_mcp_is_seen_via_http(app_env):
    client, runtime, data = app_env
    # MCP -> HTTP: pause requested through the tool shows on the panel API while the node is held open
    run_id = tool(client, "start_run", {"client_request_id": "p1", "request": MICHELSON,
                                        "algorithm_options": {"delay_s": 0.6}})["run_id"]
    wait_for(lambda: runtime.live_tasks.get(run_id))
    assert tool(client, "pause_run", {"run_id": run_id})["status"] == "pause_requested"
    seen = api_run(client, run_id)
    assert seen["active"] is True and seen["control_request"] == "pause"
    assert seen["display_state"] == "pause_requested"
    pending = tool(client, "get_worker_error", {"run_id": run_id})
    assert pending["control_request"] == "pause" and pending["display_state"] == "pause_requested"
    paused = wait_finished(client, run_id)
    assert paused["status"] == "paused" and paused["display_state"] == "paused"
    # HTTP -> MCP: resume through MCP, pause through the panel route, observe through MCP
    assert tool(client, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
    wait_for(lambda: runtime.live_tasks.get(run_id))
    assert client.post(f"/api/runs/{run_id}/pause", headers=LOCAL).json() == {"status": "pause_requested"}
    err = tool(client, "get_worker_error", {"run_id": run_id})
    assert err["active"] is True and err["control_request"] == "pause"
    assert wait_finished(client, run_id)["status"] == "paused"
    from tests.test_mcp_wrap import call as evidence_call  # noqa: PLC0415
    assert evidence_call(mcp_evidence.build_server(data), "get_run", {"run_id": run_id})["status"] == "paused"
    assert client.post(f"/api/runs/{run_id}/stop", headers=LOCAL).status_code == 202
    assert api_run(client, run_id)["display_state"] == "stopped"


def test_worker_error_reflects_the_servers_real_active_set(app_env, monkeypatch):
    client, runtime, _ = app_env

    def explode(self, run_id, resume=None, continue_run=False):
        raise ValueError("worker blew up")
    monkeypatch.setattr(Runtime, "execute", explode)
    run_id = tool(client, "start_run", {"client_request_id": "boom", "request": MICHELSON})["run_id"]
    err = wait_for(lambda: (lambda e: e if e["worker_error"] else None)(
        tool(client, "get_worker_error", {"run_id": run_id})))
    assert err["worker_error"]["type"] == "ValueError" and err["active"] is False
    # an HTTP-started (panel) run is refused by the tool (H1) but is active in the shared set: see the panel API
    entered, release = threading.Event(), threading.Event()

    def hold(self, run_id, resume=None, continue_run=False):
        entered.set()
        assert release.wait(20)
    monkeypatch.setattr(Runtime, "execute", hold)
    other = client.post("/api/runs", json={"request": "x"}, headers=LOCAL).json()["run_id"]
    assert entered.wait(20)
    try:
        assert api_run(client, other)["active"] is True
        assert "not created through start_run" in tool(client, "get_worker_error", {"run_id": other},
                                                       expect_error=True)
    finally:
        release.set()
    wait_for(lambda: not api_run(client, other)["active"])


def test_tool_surface_is_identical_in_standalone_and_hosted_mode(tmp_path, app_env):
    import asyncio
    from mcp.shared.memory import create_connected_server_and_client_session

    async def standalone():
        async with create_connected_server_and_client_session(mcp_runs.build_server(tmp_path / "iso")._mcp_server) as s:
            return (await s.list_tools()).tools
    standalone_tools = {t.name: t for t in asyncio.run(standalone())}
    client, _, _ = app_env
    hosted_tools = {t["name"]: t for t in rpc(client, "tools/list").json()["result"]["tools"]}
    expected = {"start_run": {"client_request_id", "request", "team_fault", "algorithm_options", "decision_mode"},
                "continue_run": {"run_id"}, "get_worker_error": {"run_id"}, "pause_run": {"run_id"},
                "stop_run": {"run_id"}, "get_pending_decision": {"run_id"},
                "submit_decision": {"run_id", "request_id", "decision_json", "decider_label"}}
    assert set(standalone_tools) == set(hosted_tools) == set(expected)
    forbidden = re.compile(r"provider|approv|review|toleran|retr(y|ies)|return_to|plan|setting|api_?key|secret|token|model", re.I)
    for name, params in expected.items():
        a = standalone_tools[name].inputSchema
        b = hosted_tools[name]["inputSchema"]
        assert set(a.get("properties", {})) == set(b.get("properties", {})) == params, name
        assert a == b, name
        assert not [p for p in params if forbidden.search(p)]
        assert standalone_tools[name].description == hosted_tools[name]["description"]


# ---- H1: the control tools only act on offline runs that start_run created ---------------------------
def paused_panel_run(client, runtime):
    """A panel-created (not MCP-created) run with a real checkpoint, paused at its first node boundary."""
    run_id = make_run(runtime, delay_s=0.0)
    runtime.control(run_id, "pause")
    assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 202
    assert wait_finished(client, run_id)["status"] == "paused"
    (runtime.run_dir(run_id) / "control.json").unlink()  # the pause that was honoured; leave no pending request
    return run_id


def test_h1_hosted_tools_refuse_a_panel_created_openai_run_and_never_touch_the_gateway(tmp_path, monkeypatch):
    gateway_calls, executed = [], []

    def stub_gateway(*args, **kwargs):
        gateway_calls.append((args, kwargs))
        raise AssertionError("the gateway stub must never be reached from an MCP tool")
    real_execute = Runtime.execute
    monkeypatch.setattr(Runtime, "execute", lambda self, *a, **k: (executed.append(a), real_execute(self, *a, **k))[1])
    runtime = Runtime(tmp_path, gateway_factory=stub_gateway)
    with TestClient(server.create_app(runtime)) as client:
        run_id = paused_panel_run(client, runtime)
        # what a panel-created provider=openai run looks like after the cloud call failed: interrupted
        state = runtime.get(run_id)
        (runtime.run_dir(run_id) / "run.json").write_text(json.dumps({**state, "provider": "openai",
                                                                      "status": "interrupted"}))
        before = (runtime.run_dir(run_id) / "run.json").read_bytes()
        calls_before = len(executed)
        for name, args in (("continue_run", {"run_id": run_id}), ("pause_run", {"run_id": run_id}),
                           ("stop_run", {"run_id": run_id}), ("get_worker_error", {"run_id": run_id})):
            message = tool(client, name, args, expect_error=True)
            assert "refused" in message and "start_run" in message, (name, message)
        time.sleep(.3)  # a worker that wrongly started would have shown up by now
        assert api_run(client, run_id)["active"] is False
        assert len(executed) == calls_before, "an MCP tool scheduled the worker for a panel-created openai run"
        assert gateway_calls == [], "an MCP tool caused a model request"
        assert (runtime.run_dir(run_id) / "run.json").read_bytes() == before
        assert not (runtime.run_dir(run_id) / "control.json").exists()


def test_h1_hosted_tools_refuse_a_panel_created_replay_run_that_is_not_in_the_index(app_env):
    client, runtime, data = app_env
    run_id = paused_panel_run(client, runtime)  # provider replay, algorithm_v1, github_local: only the index differs
    state = runtime.get(run_id)
    assert (state["provider"], state["simulation_backend"], state["mode"]) == ("replay", "algorithm_v1", "github_local")
    for name in ("continue_run", "pause_run", "stop_run", "get_worker_error"):
        assert "not created through start_run" in tool(client, name, {"run_id": run_id}, expect_error=True), name
    assert api_run(client, run_id)["status"] == "paused" and api_run(client, run_id)["active"] is False
    assert not (runtime.run_dir(run_id) / "control.json").exists()
    # a run start_run did create is still fully controllable
    mine = tool(client, "start_run", {"client_request_id": "mine", "request": MICHELSON,
                                      "algorithm_options": {"delay_s": 0.3}})["run_id"]
    wait_for(lambda: runtime.live_tasks.get(mine))
    assert tool(client, "pause_run", {"run_id": mine})["status"] == "pause_requested"
    assert wait_finished(client, mine)["status"] == "paused"
    assert tool(client, "continue_run", {"run_id": mine})["status"] == "resume_submitted"
    assert wait_finished(client, mine)["status"] == "algorithm_passed"


def test_h1_an_mcp_created_run_whose_stored_provider_is_not_replay_is_refused_too(app_env):
    client, runtime, data = app_env
    run_id = tool(client, "start_run", {"client_request_id": "swap", "request": MICHELSON,
                                        "algorithm_options": {"delay_s": 0}})["run_id"]
    wait_finished(client, run_id)
    path = runtime.run_dir(run_id) / "run.json"
    for field, value in (("provider", "openai"), ("simulation_backend", "physical"), ("mode", "mock")):
        original = path.read_text()
        path.write_text(json.dumps({**json.loads(original), field: value, "status": "interrupted"}))
        assert "refused" in tool(client, "continue_run", {"run_id": run_id}, expect_error=True), field
        path.write_text(original)


# ---- Task B: the shared journal check ---------------------------------------------------------------
def make_journal(directory: Path, files: dict[str, bytes]):
    directory.mkdir(parents=True)
    update = {"events": [], "status": "ok"}
    for name, data in files.items():
        (directory / name).write_bytes(data)
    completion = {"version": 1, "input_sha256": "0" * 64, "files": {n: digest(d) for n, d in files.items()},
                  "update": update, "update_sha256": digest(encode(update, limit=None))}
    (directory / "completion.json").write_bytes(encode(completion, limit=None))
    return directory


def test_journal_check_reads_a_good_file_and_is_whitelisted(tmp_path):
    directory = make_journal(tmp_path / "act-01", {"plan.json": b'{"a":1}'})
    assert journal_check.read_verified(directory, "plan.json") == b'{"a":1}'
    assert journal_check.read_verified(directory, "completion.json") == (directory / "completion.json").read_bytes()
    assert journal_check.is_whitelisted("act", 1, "plan.json")
    for bad in (("nope", 1, "plan.json"), ("act", 0, "plan.json"), ("act", 101, "plan.json"),
                ("act", True, "plan.json"), ("act", 1, "secret.json"), ("act", 1, "../plan.json"),
                ("act", 1, "execution-started.json"), ("act", "1", "plan.json")):
        assert not journal_check.is_whitelisted(*bad), bad


def test_journal_check_hash_mismatch_missing_traversal_and_symlink(tmp_path):
    directory = make_journal(tmp_path / "act-01", {"plan.json": b'{"a":1}', "decision.json": b"{}"})
    (directory / "plan.json").write_bytes(b'{"a":2}')
    with pytest.raises(ValueError):
        journal_check.read_verified(directory, "plan.json")
    assert journal_check.read_verified(directory, "decision.json") == b"{}"
    (directory / "design.json").write_bytes(b"{}")
    with pytest.raises(KeyError):  # whitelisted and present but not recorded in completion.json
        journal_check.read_verified(directory, "design.json")
    with pytest.raises(ValueError):
        journal_check.read_verified(directory, "../act-01/plan.json")
    with pytest.raises(ValueError):
        journal_check.read_verified(directory, "plan.json/../decision.json")
    with pytest.raises(FileNotFoundError):
        journal_check.read_verified(tmp_path / "empty-dir", "plan.json")
    # a symlinked artifact (even with a matching hash) is refused
    target = tmp_path / "outside.json"
    target.write_bytes(b"{}")
    (directory / "decision.json").unlink()
    (directory / "decision.json").symlink_to(target)
    with pytest.raises(ValueError):
        journal_check.read_verified(directory, "decision.json")
    # a tampered update fails the update hash
    (directory / "decision.json").unlink()
    completion = json.loads((directory / "completion.json").read_text())
    completion["update"]["status"] = "forged"
    (directory / "completion.json").write_bytes(encode(completion, limit=None))
    with pytest.raises(ValueError):
        journal_check.read_verified(directory, "plan.json")


def test_journal_check_module_is_neutral_and_evidence_uses_it():
    import ast
    for module in (journal_check,):
        tree = ast.parse(Path(module.__file__).read_text())
        imported, attrs = set(), set()
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
        parts = {piece for name in imported for piece in name.split(".")}
        assert not {"mcp", "runtime", "Runtime", "sqlite3", "fcntl", "shutil", "tempfile", "server"} & parts
        assert not {"write_text", "write_bytes", "mkdir", "unlink", "rmdir", "replace", "rename", "open", "atomic",
                    "rmtree", "flock", "symlink_to", "touch"} & attrs
    assert mcp_evidence.FILENAMES is journal_check.FILENAMES and mcp_evidence.STAGES is journal_check.STAGES
    assert "read_verified" in Path(mcp_evidence.__file__).read_text()
    assert "read_verified" in Path(server.__file__).read_text()


def test_http_artifact_route_status_codes_unchanged(app_env):
    client, runtime, _ = app_env
    run_id = tool(client, "start_run", {"client_request_id": "art", "request": MICHELSON,
                                        "algorithm_options": {"delay_s": 0}})["run_id"]
    wait_finished(client, run_id)
    base = f"/api/runs/{run_id}/algorithm/artifacts"
    ok = client.get(f"{base}/controller/1/plan.json")
    assert ok.status_code == 200 and ok.headers["content-type"].startswith("application/json")
    assert client.get(f"{base}/sim/1/artifact.zip").headers["content-type"] == "application/zip"
    assert client.get(f"{base}/controller/1/secret.json").status_code == 404
    assert client.get(f"{base}/nonsense/1/plan.json").status_code == 404
    assert client.get(f"{base}/controller/101/plan.json").status_code == 404
    assert client.get(f"{base}/controller/9/plan.json").status_code == 404  # no journal yet
    plan = runtime.run_dir(run_id) / "team" / "controller-01" / "plan.json"
    plan.write_bytes(plan.read_bytes() + b" ")
    assert client.get(f"{base}/controller/1/plan.json").status_code == 409


# ---- Task C: "active but displayed as stopped/paused" -----------------------------------------------
def make_run(runtime, **options):
    return runtime.create(RunInput(request=MICHELSON, profile="auto", mode="github_local",
                                   simulation_backend="algorithm_v1", provider="replay",
                                   algorithm_options=options or {"delay_s": 0.3}))


def paused_run(client, runtime):
    run_id = make_run(runtime)
    assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 202
    wait_for(lambda: runtime.live_tasks.get(run_id))
    assert client.post(f"/api/runs/{run_id}/pause", headers=LOCAL).status_code == 202
    state = wait_finished(client, run_id)
    assert state["status"] == "paused" and state["active"] is False
    return run_id


def record_worker_exceptions(monkeypatch):
    """The panel keeps worker errors private to create_app (the MCP tool refuses panel-created runs), so wrap
    Runtime.execute to record what the worker really raised."""
    raised = []
    real = Runtime.execute

    def spy(self, run_id, resume=None, continue_run=False, **kwargs):
        try:
            return real(self, run_id, resume, continue_run, **kwargs)
        except Exception as exc:
            raised.append((run_id, type(exc).__name__, str(exc)))
            raise
    monkeypatch.setattr(Runtime, "execute", spy)
    return raised


def test_c_stop_requested_while_a_resume_is_in_flight_is_not_shown_as_plain_stopped(app_env, tmp_path, monkeypatch):
    """Reproduction. A resumed run keeps `status: paused` in run.json until its first node publishes; a stop
    in that window makes Runtime.control rewrite run.json to `stopped` while the worker is already
    active. Before the fix the API returned status=stopped, active=true with nothing to tell the two apart.
    The runtime lock is held (as a CLI process would) so the window is deterministic. The worker has not
    started a node, so the honest label is waiting_for_worker (M1), not resuming/stop_requested (H2)."""
    client, runtime, data = app_env
    raised = record_worker_exceptions(monkeypatch)
    run_id = paused_run(client, runtime)
    with (data / "runtime.lock").open("a") as other_process:
        fcntl.flock(other_process, fcntl.LOCK_EX)
        try:
            assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 202
            resuming = api_run(client, run_id)
            assert resuming["active"] is True and resuming["status"] == "paused"
            assert resuming["display_state"] == "waiting_for_worker"  # not "paused": a worker is queued on it
            assert client.post(f"/api/runs/{run_id}/stop", headers=LOCAL).status_code == 202
            body = api_run(client, run_id)
            listed = next(r for r in client.get("/api/runs").json() if r["run_id"] == run_id)
            assert body["active"] is True and listed["active"] is True
            assert body["status"] == "stopped"  # stored field unchanged (additive fix only)
            assert body["control_request"] == "stop"  # genuinely newer than the waiting continue
            for payload in (body, listed):
                # H2: a stop that the worker may discard must not be claimed while the worker has not started
                assert payload["display_state"] == "waiting_for_worker", payload
                assert not (payload["active"] and payload["display_state"] in ("paused", "stopped"))
        finally:
            fcntl.flock(other_process, fcntl.LOCK_UN)
    final = wait_finished(client, run_id)
    assert final["status"] == "stopped" and final["display_state"] == "stopped"
    assert final["next_node"] is not None or final.get("stage")  # evidence of the earlier work is kept
    # Runtime.control rewrote run.json to "stopped" and the waiting continue honours that request: the run ends
    # stopped and the worker raises nothing (before the control-sequence fix it raised a ValueError here).
    assert raised == []


def control_while_continue_waits(client, runtime, data, action):
    """Shared by the H2 tests. A checkpointed run whose stored status is `running` (what a crash leaves) gets a
    continue that queues on the runtime lock (the worker has not started), then a pause/stop. Returns the run id
    and the API payloads seen inside that window."""
    run_id = make_run(runtime, delay_s=0.05)
    runtime.control(run_id, "pause")  # queued run: first node runs, then it pauses at the boundary
    assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 202
    assert wait_finished(client, run_id)["status"] == "paused"
    path = runtime.run_dir(run_id) / "run.json"
    path.write_text(json.dumps({**runtime.get(run_id), "status": "running"}))
    window = {}
    with (data / "runtime.lock").open("a") as other_process:
        fcntl.flock(other_process, fcntl.LOCK_EX)
        try:
            assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 202
            assert client.post(f"/api/runs/{run_id}/{action}", headers=LOCAL).status_code == 202
            assert (runtime.run_dir(run_id) / "control.json").is_file()  # the request was recorded
            window["body"] = api_run(client, run_id)
            window["listed"] = next(r for r in client.get("/api/runs").json() if r["run_id"] == run_id)
        finally:
            fcntl.flock(other_process, fcntl.LOCK_UN)
    return run_id, window


@pytest.mark.parametrize("action", ["pause", "stop"])
def test_h2_display_stays_waiting_for_worker_while_a_continue_waits_and_shows_the_new_request(app_env, action):
    client, runtime, data = app_env
    run_id, window = control_while_continue_waits(client, runtime, data, action)
    for payload in (window["body"], window["listed"]):
        assert payload["active"] is True and payload["status"] == "running"
        assert payload["display_state"] == "waiting_for_worker", payload
    assert window["body"]["control_request"] == action  # the request is honoured, so it is shown
    wait_finished(client, run_id)


@pytest.mark.parametrize("action,expected", [("pause", "paused"), ("stop", "stopped")])
def test_h2_pause_or_stop_requested_while_a_continue_waits_is_lost(app_env, action, expected):
    client, runtime, data = app_env
    run_id, _ = control_while_continue_waits(client, runtime, data, action)
    final = wait_finished(client, run_id)
    assert final["status"] == expected, (
        f"{action} was accepted (202) and stored in control.json, but the run ended {final['status']!r}")


def test_c_pause_requested_mid_node_is_distinct_from_paused(app_env):
    client, runtime, _ = app_env
    run_id = make_run(runtime, delay_s=0.6)
    client.post(f"/api/runs/{run_id}/continue", headers=LOCAL)
    wait_for(lambda: runtime.live_tasks.get(run_id))
    assert api_run(client, run_id)["display_state"] == "executing"
    assert client.post(f"/api/runs/{run_id}/pause", headers=LOCAL).status_code == 202
    mid = api_run(client, run_id)
    assert (mid["status"], mid["active"], mid["control_request"], mid["display_state"]) == (
        "running", True, "pause", "pause_requested")
    done = wait_finished(client, run_id)
    assert (done["status"], done["active"], done["display_state"]) == ("paused", False, "paused")
    assert "control_request" not in done


def test_c_stop_requested_mid_node_is_distinct_from_stopped(app_env):
    client, runtime, _ = app_env
    run_id = make_run(runtime, delay_s=0.6)
    client.post(f"/api/runs/{run_id}/continue", headers=LOCAL)
    wait_for(lambda: runtime.live_tasks.get(run_id))
    assert client.post(f"/api/runs/{run_id}/stop", headers=LOCAL).status_code == 202
    assert api_run(client, run_id)["display_state"] == "stop_requested"
    done = wait_finished(client, run_id)
    assert (done["status"], done["display_state"]) == ("stopped", "stopped")


def test_c_status_running_with_nothing_executing_here_is_not_executing_here(tmp_path):
    """After a restart (or while another process, e.g. the CLI, runs it) the stored status stays `running` but no
    worker of THIS server owns the run. The label must not assert that nothing runs it anywhere."""
    runtime = Runtime(tmp_path)
    run_id = make_run(runtime)
    queued = runtime.create(RunInput())
    state = runtime.get(run_id)
    (runtime.run_dir(run_id) / "run.json").write_text(json.dumps({**state, "status": "running"}))
    with TestClient(server.create_app(runtime)) as client:  # a fresh server process: empty active set
        body = api_run(client, run_id)
        listed = {r["run_id"]: r for r in client.get("/api/runs").json()}
        assert body["status"] == "running" and body["active"] is False
        assert body["display_state"] == listed[run_id]["display_state"] == "not_executing_here"
        assert listed[queued]["display_state"] == "queued"  # never started: the stored status, not an orphan


def test_c_another_process_executing_the_run_is_not_labelled_orphaned_or_idle(app_env):
    """A second Runtime on the same data dir (the CLI) executes the run while this server's active set is empty."""
    client, runtime, data = app_env
    run_id = make_run(runtime, delay_s=0.4)
    cli = Runtime(data)
    worker = threading.Thread(target=cli.execute, args=(run_id,))
    worker.start()
    try:
        wait_for(lambda: cli.live_tasks.get(run_id))
        body = api_run(client, run_id)
        assert body["status"] == "running" and body["active"] is False
        assert body["display_state"] == "not_executing_here"
        assert body["display_state"] not in ("orphaned", "idle")
    finally:
        worker.join(60)
    assert api_run(client, run_id)["display_state"] == "algorithm_passed"


def test_c_display_state_table():
    f = server.display_state
    assert f("running", True) == "executing" and f("running", True, "pause") == "pause_requested"
    assert f("running", True, "stop") == "stop_requested" and f("stopped", True) == "stop_requested"
    assert f("paused", True) == "resuming" and f("interrupted", True, "pause") == "resuming"
    assert f("paused", False) == "paused" and f("stopped", False) == "stopped"
    assert f("running", False) == "not_executing_here"
    # L3: nothing active -> the stored status passes through instead of collapsing to "idle"
    for status in ("queued", "algorithm_passed", "awaiting_review", "awaiting_decision", "interrupted", "failed",
                   "rejected", "design_ready", "something_new"):
        assert f(status, False) == status
    # M1: active but no node started yet -> waiting_for_worker, whatever control request is pending (H2)
    for status in ("queued", "running", "paused", "interrupted", "stopped", "awaiting_decision"):
        for control in (None, "pause", "stop"):
            assert f(status, True, control, live=False) == "waiting_for_worker", (status, control)
    assert f("queued", True, live=True) == "executing"
    assert f("algorithm_passed", True, live=False) == "executing"  # tail of a finishing run
    assert f("running", True, live=True) == "executing"


def test_c_server_and_standalone_display_state_agree():
    for status in ("queued", "running", "paused", "stopped", "interrupted", "awaiting_review", "awaiting_decision",
                   "failed", "algorithm_passed"):
        for active in (True, False):
            for control in (None, "pause", "stop"):
                for live in (True, False):
                    assert (server.display_state(status, active, control, live=live)
                            == mcp_runs.display_state(status, active, control, live=live))


def test_m1_a_run_queued_behind_the_single_worker_is_waiting_not_executing(app_env, monkeypatch):
    client, runtime, _ = app_env
    entered, release = threading.Event(), threading.Event()
    original = AlgorithmServices.sim

    def slow_sim(self, state, path, attempt):
        entered.set()
        assert release.wait(20)
        return original(self, state, path, attempt)
    monkeypatch.setattr(AlgorithmServices, "sim", slow_sim)
    first, second = make_run(runtime, delay_s=0.0), make_run(runtime, delay_s=0.0)
    assert client.post(f"/api/runs/{first}/continue", headers=LOCAL).status_code == 202
    assert entered.wait(20)
    assert client.post(f"/api/runs/{second}/continue", headers=LOCAL).status_code == 202
    try:
        listed = {r["run_id"]: r for r in client.get("/api/runs").json()}
        assert listed[first]["display_state"] == "executing"
        assert listed[second]["active"] is True and listed[second]["status"] == "queued"
        assert listed[second]["display_state"] == "waiting_for_worker"
        assert api_run(client, second)["display_state"] == "waiting_for_worker"
    finally:
        release.set()
    assert wait_finished(client, first)["status"] == "algorithm_passed"
    done = wait_finished(client, second)
    assert done["status"] == "algorithm_passed" and done["display_state"] == "algorithm_passed"


def test_m1_a_worker_blocked_on_the_runtime_lock_is_waiting(app_env):
    client, runtime, data = app_env
    run_id = make_run(runtime, delay_s=0.0)
    with (data / "runtime.lock").open("a") as other_process:
        fcntl.flock(other_process, fcntl.LOCK_EX)
        try:
            assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 202
            time.sleep(.3)  # the worker thread is now inside Runtime.execute, blocked on flock
            assert api_run(client, run_id)["display_state"] == "waiting_for_worker"
            assert not runtime.live_tasks
        finally:
            fcntl.flock(other_process, fcntl.LOCK_UN)
    assert wait_finished(client, run_id)["status"] == "algorithm_passed"


def test_c_panel_labels_come_from_display_state():
    html = (Path(server.__file__).parent / "panel.html").read_text()
    assert "const stateLabel=r=>statuses[r.display_state] || statuses[r.status] || r.status;" in html
    assert "$('status').textContent=stateLabel(s);" in html
    assert html.count("statuses[r.status] || r.status") == 1  # only inside stateLabel; history rows use stateLabel too
    for label in ("executing:", "resuming:", "waiting_for_worker:", "not_executing_here:"):
        assert label in html
    assert "orphaned" not in html
    assert "Not running (no worker" not in html
    assert "may be running in another process" in html


# ---- control requests vs a waiting continue, through HTTP and MCP -----------------------------------------
def test_http_stale_pause_is_hidden_and_cleared_when_a_continue_waits_then_the_run_completes(app_env):
    client, runtime, data = app_env
    run_id = paused_run(client, runtime)
    assert (runtime.run_dir(run_id) / "control.json").is_file()  # the honoured pause is still on disk
    with (data / "runtime.lock").open("a") as other_process:
        fcntl.flock(other_process, fcntl.LOCK_EX)
        try:
            assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 202
            body = api_run(client, run_id)
            assert body["display_state"] == "waiting_for_worker" and "control_request" not in body  # stale: not shown
        finally:
            fcntl.flock(other_process, fcntl.LOCK_UN)
    assert wait_finished(client, run_id)["status"] == "algorithm_passed"  # pause, continue, nothing


@pytest.mark.parametrize("action,expected", [("pause", "paused"), ("stop", "stopped")])
def test_http_request_while_a_resume_waits_is_honoured_before_any_node(app_env, monkeypatch, action, expected):
    client, runtime, data = app_env
    raised = record_worker_exceptions(monkeypatch)
    run_id = paused_run(client, runtime)
    started = []
    real_set = dict.__setitem__
    runtime.live_tasks = type("Rec", (dict,), {"__setitem__": lambda self, k, v: (started.append(v), real_set(self, k, v))[1]})()
    with (data / "runtime.lock").open("a") as other_process:
        fcntl.flock(other_process, fcntl.LOCK_EX)
        try:
            assert client.post(f"/api/runs/{run_id}/continue", headers=LOCAL).status_code == 202
            assert client.post(f"/api/runs/{run_id}/{action}", headers=LOCAL).status_code == 202
        finally:
            fcntl.flock(other_process, fcntl.LOCK_UN)
    final = wait_finished(client, run_id)
    assert final["status"] == expected and started == [] and raised == []


@pytest.mark.parametrize("action,expected", [("pause", "paused"), ("stop", "stopped")])
def test_mcp_request_while_a_resume_waits_is_honoured_before_any_node(app_env, action, expected):
    client, runtime, data = app_env
    run_id = tool(client, "start_run", {"client_request_id": "w1", "request": MICHELSON,
                                        "algorithm_options": {"delay_s": 0.3}})["run_id"]
    wait_for(lambda: runtime.live_tasks.get(run_id))
    assert tool(client, "pause_run", {"run_id": run_id})["status"] == "pause_requested"
    assert wait_finished(client, run_id)["status"] == "paused"
    started = []
    real_set = dict.__setitem__
    runtime.live_tasks = type("Rec", (dict,), {"__setitem__": lambda self, k, v: (started.append(v), real_set(self, k, v))[1]})()
    with (data / "runtime.lock").open("a") as other_process:
        fcntl.flock(other_process, fcntl.LOCK_EX)
        try:
            assert tool(client, "continue_run", {"run_id": run_id})["status"] == "resume_submitted"
            assert tool(client, f"{action}_run", {"run_id": run_id})["status"].endswith("_requested")
        finally:
            fcntl.flock(other_process, fcntl.LOCK_UN)
    final = wait_finished(client, run_id)
    assert final["status"] == expected and started == []
    assert tool(client, "get_worker_error", {"run_id": run_id})["worker_error"] is None


def test_mcp_standalone_continue_captures_the_sequence_and_honours_a_later_stop(tmp_path):
    """The stdio backend has its own submit: it must pass the sequence too."""
    backend = mcp_runs.StandaloneBackend(tmp_path / "standalone")
    runtime = backend.runtime
    run_id = make_run(runtime, delay_s=0)
    runtime.control(run_id, "pause")
    assert runtime.execute(run_id)["status"] == "paused"
    (runtime.run_dir(run_id) / "run.json").write_text(json.dumps({**runtime.get(run_id), "status": "running"}))
    with (runtime.data / "runtime.lock").open("a") as other_process:
        fcntl.flock(other_process, fcntl.LOCK_EX)
        try:
            backend.submit(run_id, True)
            runtime.control(run_id, "stop")
        finally:
            fcntl.flock(other_process, fcntl.LOCK_UN)
    wait_for(lambda: run_id not in backend.active)
    assert runtime.get(run_id)["status"] == "stopped" and backend.worker_errors == {}
