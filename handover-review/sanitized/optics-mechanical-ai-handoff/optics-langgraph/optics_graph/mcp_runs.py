"""optics-runs: MUTATING MCP server that drives the OFFLINE rule fixture, or an agent-decided run (stdio).

Always: profile="auto", mode="github_local", simulation_backend="algorithm_v1",
provider="replay" (decision_mode "fixture") or provider="agent" (decision_mode "agent": no model call is
made; the in-graph decisions are submitted by the MCP client through get_pending_decision / submit_decision,
which can only store one schema-valid answer for the one pending request). There is no provider parameter,
no review/approve tool, no settings or API-key access, and no way to set retries, return_to, tolerances,
plan bytes or feedback.

The tool logic (RunsService + register_tools) runs against a small backend interface with two
implementations and exposes the identical tool surface in both:
  * StandaloneBackend (stdio, `build_server`): its own Runtime on an explicit, isolated data directory
    (never the web panel's data dir) and its own single worker thread.
  * ServerBackend (`build_hosted_server`, used by server.py at /mcp): the web server's existing Runtime,
    `schedule`, lock, `active` set and worker-error dict. No second Runtime and no second worker thread.

Usage:  python -m optics_graph.mcp_runs --data-dir DIR   (or env OPTICS_MCP_DATA_DIR)
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from . import agent_decisions
from .runtime import RunInput, Runtime
from .settings import DATA
from .sim_io import decode_json, read_json

CONFLICT = "conflict (HTTP 409 semantics): this run is already executing"
CLIENT_ID_RE = re.compile(r"[A-Za-z0-9_.\-]{1,128}")
RUN_ID_RE = re.compile(r"[0-9a-f]{32}")
INDEX_NAME = "mcp_client_requests.json"
AGENT_SOURCE = ("agent-decided run: in-graph decisions are supplied by the MCP client; decider identity is "
                "self-reported and unverified")
MARKER_NAME = "mcp_origin.json"  # written by start_run into the run dir it created: tags the run as MCP-made
# Simulated-plant knobs that are safe: they can only make the fixture converge
# faster/slower (0 = never, which fails honestly at the retry budget). The
# initial_* seeds are excluded (0 would erase the seeded misplacement the
# fixture exists to repair) and anything else is rejected. Tolerances are not
# in AlgorithmOptions at all (they come from the experiment contract).
ALLOWED_OPTIONS = {"delay_s": (0.0, 2.0), "position_response": (0.0, 1.0), "angle_response": (0.0, 1.0)}
# Every team_fault value only INJECTS a failure on a node's first attempt
# (algorithm_sim.injected -> self.fail); none can skip or relax a check.
FAULTS = ("none", "vision", "prompt", "act", "controller", "sim", "output", "execution", "matrix")


class RunsError(ValueError):
    pass


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _clean_options(options: dict | None) -> dict:
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise RunsError("algorithm_options must be an object")
    unknown = sorted(set(options) - set(ALLOWED_OPTIONS))
    if unknown:
        raise RunsError(f"algorithm_options keys not allowed: {unknown}; allowed: {sorted(ALLOWED_OPTIONS)}")
    cleaned = {"delay_s": 0.0}
    for key, value in options.items():
        lo, hi = ALLOWED_OPTIONS[key]
        if type(value) not in (int, float) or isinstance(value, bool) or not lo <= value <= hi:
            raise RunsError(f"{key} must be a number in [{lo}, {hi}]")
        cleaned[key] = float(value)
    return cleaned


def _contains(outer: Path, inner: Path) -> bool:
    """True when inner is outer or lies below it. Uses os.path.samefile on every existing ancestor so a
    case-insensitive filesystem (macOS) or a symlink cannot evade it; falls back to a case-folded
    comparison when neither side exists yet."""
    outer, inner = Path(outer).resolve(), Path(inner).resolve()
    for candidate in (inner, *inner.parents):
        try:
            if candidate.exists() and outer.exists() and os.path.samefile(candidate, outer):
                return True
        except OSError:
            continue
    folded_outer = [part.casefold() for part in outer.parts]
    folded_inner = [part.casefold() for part in inner.parts]
    return folded_inner[:len(folded_outer)] == folded_outer


def display_state(status, active, control=None, *, live=True):
    """Same value set as server.display_state (kept in sync by a test; neither module may import the other).
    `live` is False while an active run has not started a node yet (queued behind the single worker, or the
    worker is blocked on the runtime lock)."""
    if active:
        if not live and status in ("queued", "running", "paused", "interrupted", "stopped", "awaiting_review",
                                   "awaiting_decision"):
            return "waiting_for_worker"
        if status == "stopped" or control == "stop":
            return "stop_requested"
        if status in ("paused", "interrupted", "awaiting_decision"):
            return "resuming"
        return "pause_requested" if control == "pause" else "executing"
    return "not_executing_here" if status == "running" else status


class StandaloneBackend:
    """stdio mode: own Runtime and own worker thread; refuses the web panel's data directory."""

    kind = "standalone"

    def __init__(self, data_dir: Path):
        data = Path(data_dir).resolve()
        panel = Path(DATA)
        if _contains(panel, data) or _contains(data, panel):
            raise RunsError("refusing to use the web panel's data directory (or one inside or containing it); "
                            "use an isolated directory")
        self.runtime = Runtime(data)
        self.data = data
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optics-runs")
        self.lock = threading.Lock()
        self.active: dict[str, Future] = {}
        self.worker_errors: dict[str, dict] = {}  # last worker failure per run, this process only

    def submit(self, run_id: str, continue_run: bool) -> None:
        with self.lock:
            if run_id in self.active:
                raise RunsError(CONFLICT)
            self.worker_errors.pop(run_id, None)
            seq = {"control_seq": self.runtime.control_seq(run_id)} if continue_run else {}
            future = self.pool.submit(self.runtime.execute, run_id, None, continue_run, **seq)
            self.active[run_id] = future

        def done(f, rid=run_id):
            with self.lock:
                self.active.pop(rid, None)
                try:
                    exc = f.exception()
                except Exception as err:  # noqa: BLE001 - cancelled
                    exc = err
                if exc is not None:
                    self.worker_errors[rid] = {"type": type(exc).__name__, "message": str(exc)[:500],
                                               "time": datetime.now(timezone.utc).isoformat()}
        future.add_done_callback(done)

    def display(self, run_id: str, status: str, active: bool, control: str | None) -> str:
        return display_state(status, active, control, live=run_id in self.runtime.live_tasks)


class ServerBackend:
    """Hosted mode: borrows the web server's runtime, lock, `active` set and worker-error dict.

    `submit(run_id, continue_run)` is the server's `schedule`; it must raise RunsError when the run is
    already executing. Nothing here creates a Runtime or a thread.
    """

    kind = "hosted"

    def __init__(self, runtime: Runtime, submit: Callable[[str, bool], None], lock, active, worker_errors: dict,
                 display: Callable[[str, str, bool, str | None], str] | None = None):
        self.runtime = runtime
        self._display = display
        self.data = Path(runtime.data).resolve()
        self.submit = submit
        self.lock = lock
        self.active = active
        self.worker_errors = worker_errors

    def display(self, run_id: str, status: str, active: bool, control: str | None) -> str:
        if self._display is not None:
            return self._display(run_id, status, active, control)
        return display_state(status, active, control, live=run_id in self.runtime.live_tasks)


class RunsService:
    def __init__(self, backend):
        self.backend = backend
        self.runtime = backend.runtime
        self.data = backend.data
        self.lock = backend.lock
        self.active = backend.active
        self.worker_errors = backend.worker_errors

    # ---- idempotency index ------------------------------------------------
    def _index_path(self) -> Path:
        return self.data / INDEX_NAME

    def _read_index(self) -> dict:
        path = self._index_path()
        if not path.exists():
            return {}
        try:
            index = decode_json(path.read_text("utf-8"))
        except (ValueError, OSError) as exc:
            raise RunsError(f"client request index {path} is unreadable or corrupt ({exc}); "
                            "repair or remove that file by hand") from None
        if not isinstance(index, dict) or not all(isinstance(v, dict) for v in index.values()):
            raise RunsError(f"client request index {path} is corrupt: unexpected structure; "
                            "repair or remove that file by hand")
        return index

    def _write_index(self, index: dict) -> None:
        fd, name = tempfile.mkstemp(dir=self.data, prefix=".index-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self._index_path())
            dir_fd = os.open(self.data, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            Path(name).unlink(missing_ok=True)

    def _marker_path(self, run_id: str) -> Path:
        return self.runtime.run_dir(run_id) / MARKER_NAME

    def _write_marker(self, run_id: str, client_request_id: str, nonce: str) -> None:
        directory = self.runtime.run_dir(run_id)
        fd, name = tempfile.mkstemp(dir=directory, prefix=".marker-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump({"client_request_id": client_request_id, "nonce": nonce}, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self._marker_path(run_id))
        finally:
            Path(name).unlink(missing_ok=True)

    def _read_marker(self, run_id: str) -> dict | None:
        try:
            marker = decode_json(self._marker_path(run_id).read_text("utf-8"))
        except (ValueError, OSError):
            return None
        return marker if isinstance(marker, dict) else None

    def _orphans(self, index: dict, client_request_id: str, nonce: Any, request: str, fault: str, options: dict,
                 since: str, provider: str = "replay") -> tuple[list[str], list[str]]:
        """Queued runs left by an earlier start_run that crashed before its index entry was finished.

        Returns (tagged, ambiguous). `tagged` carry this request's marker (client_request_id + nonce) and may be
        adopted. `ambiguous` match every other field but have no marker at all (the crash landed between
        Runtime.create and the marker write, or the run was made by the panel): never adopted."""
        taken = {v.get("run_id") for v in index.values()}
        try:
            started = datetime.fromisoformat(since)
        except (TypeError, ValueError):
            return [], []
        tagged, ambiguous = [], []
        runs = self.data / "runs"
        for p in sorted(runs.iterdir()) if runs.is_dir() else []:
            if not RUN_ID_RE.fullmatch(p.name) or p.name in taken:
                continue
            try:
                st = self.runtime.get(p.name)
                created = datetime.fromisoformat(st["created_at"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if (st.get("status") == "queued" and created >= started and st.get("request") == request
                    and st.get("team_fault") == fault and st.get("provider") == provider
                    and st.get("simulation_backend") == "algorithm_v1"
                    and all((st.get("algorithm_options") or {}).get(k) == v for k, v in options.items())):
                marker = self._read_marker(p.name)
                if marker is None:
                    ambiguous.append(p.name)
                elif (isinstance(nonce, str) and marker.get("client_request_id") == client_request_id
                        and marker.get("nonce") == nonce):
                    tagged.append(p.name)
        return tagged, ambiguous

    # ---- worker -----------------------------------------------------------
    def _submit(self, run_id: str, continue_run: bool) -> None:
        self.backend.submit(run_id, continue_run)

    def _check(self, run_id: Any) -> str:
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise RunsError("run_id must be exactly 32 lowercase hex characters")
        try:
            self.runtime.get(run_id)
        except (OSError, ValueError):
            raise RunsError("run not found") from None
        return run_id

    def _require_mcp_run(self, run_id: Any) -> dict:
        """Only runs this surface may touch: created by start_run (present in the idempotency index) AND offline
        or agent-decided (provider replay or agent, algorithm_v1, github_local). Panel-created runs, including
        provider=openai ones, are refused."""
        self._check(run_id)
        state = self.runtime.get(run_id)
        known = {v.get("run_id") for v in self._read_index().values()}
        if run_id not in known:
            raise RunsError("refused: this run was not created through start_run, so this offline-only tool "
                            "surface will not act on it (panel-created runs are controlled from the web panel's own "
                            "buttons or its HTTP API, not through these tools)")
        if (state.get("provider") not in ("replay", "agent") or state.get("simulation_backend") != "algorithm_v1"
                or state.get("mode") != "github_local"):
            raise RunsError("refused: only offline or agent-decided runs (provider replay or agent, "
                            "simulation_backend algorithm_v1, mode github_local) can be controlled here; "
                            f"this run is provider={state.get('provider')!r}")
        return state

    # ---- operations -------------------------------------------------------
    def start(self, client_request_id: str, request: str, team_fault: str | None,
              algorithm_options: dict | None, decision_mode: str = "fixture") -> dict:
        if not isinstance(client_request_id, str) or not CLIENT_ID_RE.fullmatch(client_request_id):
            raise RunsError("client_request_id must match [A-Za-z0-9_.-]{1,128}")
        if not isinstance(request, str) or not 1 <= len(request) <= 8000:
            raise RunsError("request must be 1-8000 characters")
        fault = team_fault or "none"
        if fault not in FAULTS:
            raise RunsError(f"team_fault must be one of {list(FAULTS)}")
        if decision_mode not in ("fixture", "agent"):
            raise RunsError("decision_mode must be 'fixture' or 'agent'")
        provider = "agent" if decision_mode == "agent" else "replay"
        options = _clean_options(algorithm_options)
        body = {"request": request, "team_fault": fault, "algorithm_options": options}
        if decision_mode != "fixture":  # existing index entries keep their hashes
            body["decision_mode"] = decision_mode
        body_hash = hashlib.sha256(_canon(body)).hexdigest()
        with (self.data / "mcp_index.lock").open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            index = self._read_index()
            known = index.get(client_request_id)
            run_id, created = None, True
            if known is not None:
                if known.get("body_sha256") != body_hash:
                    raise RunsError("client_request_id was already used with a different request body")
                if known.get("pending") is True:
                    # An earlier call crashed between create and the final index write: adopt its run.
                    orphans, ambiguous = self._orphans(index, client_request_id, known.get("nonce"), request,
                                                       fault, options, known.get("started_at", ""), provider)
                    if len(orphans) > 1:
                        raise RunsError(f"client_request_id has several unindexed queued runs {orphans}; "
                                        "resolve by hand")
                    if not orphans and ambiguous:
                        raise RunsError(f"not adopting: queued run(s) {ambiguous} match this request but carry no "
                                        f"{MARKER_NAME} tag, so start_run cannot tell whether an earlier crashed call "
                                        "or the web panel created them. Resolve by hand (stop or ignore that run and "
                                        f"remove this client_request_id from {INDEX_NAME}), then retry")
                    run_id = orphans[0] if orphans else None
                else:
                    run_id = known.get("run_id")
                    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
                        raise RunsError(f"client request index {self._index_path()} entry for this id is corrupt")
                    created = False
            if run_id is None:
                run_input = RunInput(request=request, profile="auto", mode="github_local",
                                     simulation_backend="algorithm_v1", provider=provider,
                                     team_fault=fault, algorithm_options=options)
                nonce = (known or {}).get("nonce") or uuid.uuid4().hex
                index[client_request_id] = {"body_sha256": body_hash, "pending": True, "nonce": nonce,
                                            "started_at": (known or {}).get("started_at")
                                            or datetime.now(timezone.utc).isoformat()}
                self._write_index(index)  # durable BEFORE the run exists, so a retry can adopt it
                run_id = self.runtime.create(run_input)
                self._write_marker(run_id, client_request_id, nonce)  # tag: only tagged runs are ever adopted
            if created:
                index[client_request_id] = {"run_id": run_id, "body_sha256": body_hash}
                self._write_index(index)
        state = self.runtime.get(run_id)
        if state["status"] == "queued":
            try:  # first start, or recover from a crash between create and submit
                self._submit(run_id, continue_run=False)
            except RunsError:
                pass
            state = self.runtime.get(run_id)
        return {"run_id": run_id, "status": state["status"], "idempotent_replay": not created,
                "source": AGENT_SOURCE if decision_mode == "agent"
                else "offline rule fixture, not a model-designed experiment"}

    def cont(self, run_id: str) -> dict:
        state = self._require_mcp_run(run_id)
        status = state["status"]
        if status == "awaiting_decision":
            pending = state.get("pending_decision")
            try:
                waiting = agent_decisions.decision_state(self.runtime.run_dir(run_id), pending)
            except agent_decisions.DecisionError as exc:
                raise RunsError(str(exc)) from None
            if not waiting["submitted"] and not waiting["exhausted"]:
                raise RunsError("no decision submitted: call get_pending_decision and submit_decision for the "
                                "pending request before continue_run")
        elif status not in ("running", "interrupted", "paused"):
            raise RunsError(f"run status {status!r} cannot be resumed "
                            "(only running/interrupted/paused, or awaiting_decision with a submitted decision)")
        self._submit(run_id, continue_run=True)
        return {"run_id": run_id, "status": "resume_submitted"}

    def _agent_run(self, run_id: str) -> dict:
        state = self._require_mcp_run(run_id)
        if state.get("provider") != "agent":
            raise RunsError("refused: this run is not an agent-decided run (start_run with decision_mode 'agent')")
        return state

    def pending_decision(self, run_id: str) -> dict:
        """Read-only: the decision request the run is waiting for."""
        state = self._agent_run(run_id)
        pending = state.get("pending_decision")
        if state["status"] != "awaiting_decision" or not isinstance(pending, dict):
            return {"run_id": run_id, "pending": False, "status": state["status"]}
        directory = self.runtime.run_dir(run_id)
        try:
            record = agent_decisions.read_request(directory, pending)
            waiting = agent_decisions.decision_state(directory, pending)
        except agent_decisions.DecisionError as exc:
            raise RunsError(str(exc)) from None
        return {"pending": True, "status": state["status"], **record, "submitted": waiting["submitted"],
                "rejections_remaining": waiting["rejections_remaining"]}

    def submit_decision(self, run_id: str, request_id: str, decision_json: str, decider_label: str) -> dict:
        """Store ONE answer for the pending request. Never schedules execution, never touches anything but
        decision-response-<role>.json (or the rejections list on a validation failure)."""
        if not isinstance(request_id, str) or not agent_decisions.REQUEST_ID_RE.fullmatch(request_id):
            raise RunsError("request_id must be exactly 64 lowercase hex characters")
        if not isinstance(decision_json, str) or not decision_json.strip():
            raise RunsError("decision_json must be a non-empty JSON string")
        if len(decision_json) > agent_decisions.MAX_DECISION_CHARS:
            raise RunsError(f"decision_json exceeds {agent_decisions.MAX_DECISION_CHARS} characters")
        if not isinstance(decider_label, str) or not agent_decisions.LABEL_RE.fullmatch(decider_label):
            raise RunsError("decider_label must match [A-Za-z0-9_.:/ -]{1,128}")
        self._agent_run(run_id)
        with self.lock:
            if run_id in self.active:
                raise RunsError("refused: a worker is active for this run; wait until it stops at awaiting_decision")
        with self.runtime._control_lock(run_id):  # exclusive vs other submits and stop requests, across processes
            state = self.runtime.get(run_id)
            if state.get("provider") != "agent" or state.get("status") != "awaiting_decision":
                raise RunsError(f"refused: run status is {state.get('status')!r}, not awaiting_decision")
            try:
                return {"run_id": run_id, **agent_decisions.submit(
                    self.runtime.run_dir(run_id), state.get("pending_decision"), request_id, decision_json,
                    decider_label, self.backend.kind)}
            except agent_decisions.DecisionError as exc:
                raise RunsError(str(exc)) from None

    def worker_error(self, run_id: str) -> dict:
        state = self._require_mcp_run(run_id)
        with self.lock:
            error, active = self.worker_errors.get(run_id), run_id in self.active
        control = None
        if active:  # same rule as GET /api/runs/{id}: a pending request only matters while executing
            try:
                control = read_json(self.runtime.run_dir(run_id) / "control.json").get("action")
            except (OSError, ValueError, AttributeError):
                control = None
        return {"run_id": run_id, "worker_error": error, "active": active, "control_request": control,
                "status": state["status"], "display_state": self.backend.display(run_id, state["status"], active, control)}

    def control(self, run_id: str, action: str) -> dict:
        self._require_mcp_run(run_id)
        try:
            self.runtime.control(run_id, action)
        except ValueError as exc:
            raise RunsError(str(exc)) from None
        return {"run_id": run_id, "status": f"{action}_requested"}


def register_tools(mcp: FastMCP, service: RunsService) -> FastMCP:
    """The one tool surface, shared by the stdio server and the web-server-hosted endpoint."""
    async def off_loop(fn, *args):  # tools do blocking file/lock work: keep the event loop free
        return await anyio.to_thread.run_sync(fn, *args)

    @mcp.tool()
    async def start_run(client_request_id: str, request: str,
                        team_fault: Literal["none", "vision", "prompt", "act", "controller", "sim", "output",
                                            "execution", "matrix"] | None = None,
                        algorithm_options: dict[str, float] | None = None,
                        decision_mode: Literal["fixture", "agent"] = "fixture") -> dict[str, Any]:
        """Start a run. decision_mode "fixture" (default): the OFFLINE rule-fixture run, no model call.
        decision_mode "agent": NO model call either; whenever a node needs a model decision the run stops with
        status awaiting_decision and you answer it with get_pending_decision / submit_decision, then
        continue_run. Idempotent on client_request_id.
        algorithm_options allowlist: delay_s [0,2], position_response [0,1], angle_response [0,1]."""
        return await off_loop(service.start, client_request_id, request, team_fault, algorithm_options,
                              decision_mode)

    @mcp.tool()
    async def get_pending_decision(run_id: str) -> dict[str, Any]:
        """Read-only. For an agent-decided run (start_run decision_mode "agent") stopped at awaiting_decision:
        the pending decision request (request_id, node, role, instructions, data, output_schema, ...) plus
        `submitted` and `rejections_remaining`. Otherwise {pending: false, status}. Only acts on agent runs
        created by start_run."""
        return await off_loop(service.pending_decision, run_id)

    @mcp.tool()
    async def submit_decision(run_id: str, request_id: str, decision_json: str,
                              decider_label: str) -> dict[str, Any]:
        """Answer the pending decision request of an agent-decided run. decision_json is a STRING holding one
        JSON object that matches the request's output_schema (duplicate keys and non-finite numbers are
        rejected). It stores that single answer and nothing else: it does not resume the run (call
        continue_run), cannot approve a review, and cannot change retries, routing, tolerances or plans.
        An invalid answer is recorded as a rejection (three rejections exhaust the request). decider_label is
        a self-reported, unverified name for the evidence record."""
        return await off_loop(service.submit_decision, run_id, request_id, decision_json, decider_label)

    @mcp.tool()
    async def continue_run(run_id: str) -> dict[str, Any]:
        """Resume a paused/interrupted run, or an agent-decided run at awaiting_decision once a decision is
        submitted (or its request is exhausted). Rejected (conflict) if the run is already executing."""
        return await off_loop(service.cont, run_id)

    @mcp.tool()
    async def get_worker_error(run_id: str) -> dict[str, Any]:
        """Read-only: the last exception raised by the background worker for this run in THIS server process
        (None if none), whether the run is executing now (`active`), any pending pause/stop request
        (`control_request`), the stored `status` and a `display_state`. A run is finished only when `active` is
        false AND `status` is terminal; stored status can read paused/stopped while a worker is still executing.
        Only acts on runs created by start_run."""
        return await off_loop(service.worker_error, run_id)

    @mcp.tool()
    async def pause_run(run_id: str) -> dict[str, Any]:
        """Request a pause at the next node boundary."""
        return await off_loop(service.control, run_id, "pause")

    @mcp.tool()
    async def stop_run(run_id: str) -> dict[str, Any]:
        """Request a stop at the next node boundary."""
        return await off_loop(service.control, run_id, "stop")

    for registered in mcp._tool_manager.list_tools():  # FastMCP drops unknown arguments by default: reject them
        model = registered.fn_metadata.arg_model
        model.model_config["extra"] = "forbid"
        model.model_rebuild(force=True)
        registered.parameters = model.model_json_schema(by_alias=True)
    return mcp


def build_server(data_dir: Path | str) -> FastMCP:
    """Standalone (stdio) server with its own Runtime and worker thread."""
    service = RunsService(StandaloneBackend(Path(data_dir)))
    mcp = FastMCP("optics-runs")
    mcp._optics_service = service  # test hook: lets tests wait on the worker
    return register_tools(mcp, service)


# DNS-rebinding protection of the MCP SDK, aligned with server.py's TrustedHostMiddleware
# (127.0.0.1 / localhost / testserver) and local_only origin rule. Both layers apply.
HOSTED_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*", "testserver"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://127.0.0.1", "http://localhost",
                     "http://testserver"])


def build_hosted_server(runtime: Runtime, submit: Callable[[str, bool], None], lock, active,
                        worker_errors: dict, display: Callable[[str, str, bool, str | None], str] | None = None
                        ) -> FastMCP:
    """Server-hosted tool surface on the web server's Runtime / schedule / lock / active set (stateless HTTP)."""
    service = RunsService(ServerBackend(runtime, submit, lock, active, worker_errors, display))
    mcp = FastMCP("optics-runs", stateless_http=True, json_response=True, streamable_http_path="/mcp",
                  transport_security=HOSTED_SECURITY)
    mcp._optics_service = service
    return register_tools(mcp, service)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="optics-runs")
    parser.add_argument("--data-dir", default=os.environ.get("OPTICS_MCP_DATA_DIR"))
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("--data-dir (or OPTICS_MCP_DATA_DIR) is required")
    build_server(Path(args.data_dir)).run("stdio")


if __name__ == "__main__":
    main()
