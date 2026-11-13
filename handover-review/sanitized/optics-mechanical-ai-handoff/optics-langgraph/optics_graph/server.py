"""Loopback-only local control panel; one worker owns the workflow runtime."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import math
from typing import Literal
from urllib.parse import urlparse
from pathlib import Path
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.routing import Route
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from .runtime import Runtime, RunInput
from .settings import ROOT, public_settings, save_settings
from .graph import build_graph
from .team_flow import build_team_graph
from .team_artifacts import digest
from .journal_check import is_whitelisted, read_verified
from .legacy import EXPECTED_NAMES, precision_progress
from .sim_io import decode_json, read_bytes, read_json
from .experiments import catalog, match_registered_assembly
from .presentation import PresentationService
from . import cli_decider


class SettingsInput(BaseModel):
    model_config=ConfigDict(extra="forbid")
    model: str
    api_key: SecretStr=SecretStr("")


class ReviewInput(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    approved: bool
    plan_sha256: str


class ExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request: str = Field(min_length=1, max_length=8000)
    output_language: Literal["en"] = "en"


class CliExperimentInput(ExperimentInput):
    decider: Literal["claude", "codex"]


def live_observation_progress(log: Path, requested=EXPECTED_NAMES):
    """Display complete worker log rows while the signed-off observation is being collected."""
    rows = {}
    try:
        lines = read_bytes(log, limit=16_000).split(b"\n")[:-1]
    except (OSError, ValueError):
        lines = []
    for line in lines:
        try:
            row = decode_json(line.decode("utf-8"))
            if not isinstance(row, dict) or set(row) != {
                    "object_id", "position_error_mm", "rotation_error_deg", "within_1mm_1deg"}:
                continue
            name = row["object_id"]
            errors = (row["position_error_mm"], row["rotation_error_deg"])
            if (not isinstance(name, str) or name not in requested
                    or any(type(v) not in (float, int) or v < 0 or (type(v) is float and not math.isfinite(v)) for v in errors)
                    or row["within_1mm_1deg"] is not all(v <= 1. for v in errors)):
                continue
            rows[name] = row
        except (UnicodeDecodeError, ValueError):
            continue
    return {"measured":len(rows), "required":len(requested), "latest":next(reversed(rows.values()), None)}


def display_state(status, active, control=None, *, live=True):
    """Additive, honest label for one run. `active` is only known to this server process; `live` says a node
    has started (runtime.live_tasks has the run). Values:

    waiting_for_worker   active here but no node has started: queued behind the single worker, or the worker
                         is blocked on runtime.lock. A pause/stop requested in this window is honoured before
                         any node runs (runtime.py control sequence) and is shown as `control_request`; the label
                         stays waiting_for_worker because it is what is true until the worker starts.
    executing | pause_requested | stop_requested   (still executing; acts at the next node boundary)
    resuming             stored status is still paused/interrupted but a worker is already running it
    not_executing_here   stored status says running but nothing in this server is executing it; it may be
                         running in another process (e.g. the CLI) or have been interrupted. Not a claim either way.
    otherwise (nothing active)   the stored status itself: queued, paused, stopped, interrupted, awaiting_review,
                         failed, finished statuses, ...
    """
    if active:
        if not live and status in ("queued", "running", "paused", "interrupted", "stopped", "awaiting_review",
                                   "awaiting_decision"):
            return "waiting_for_worker"
        if status == "stopped" or control == "stop":
            return "stop_requested"  # runtime.control rewrote run.json to stopped while a resume was in flight
        if status in ("paused", "interrupted", "awaiting_decision"):
            return "resuming"  # run.json only changes at the first node boundary; control.json may be stale
        return "pause_requested" if control == "pause" else "executing"
    return "not_executing_here" if status == "running" else status


class _McpEndpoint:
    """Stable ASGI endpoint for /mcp. The session manager behind it is rebuilt on every lifespan entry
    (a StreamableHTTPSessionManager can only run() once), so the route never changes."""
    manager = None

    async def __call__(self, scope, receive, send):
        manager = self.manager
        if manager is None:
            await JSONResponse({"detail": "The MCP endpoint is not running"}, status_code=503)(scope, receive, send)
            return
        await manager.handle_request(scope, receive, send)


def create_app(runtime:Runtime|None=None, presentation_service=None):
    runtime=runtime or Runtime()
    presentation = presentation_service or PresentationService()
    mcp_endpoint=_McpEndpoint()
    mcp_build=None  # set below when the optional `mcp` package is importable
    decider=None  # local CLI decider (Claude Code or Codex) for panel agent runs; needs the optional `mcp` package
    @asynccontextmanager
    async def lifespan(app):
        if decider is not None:decider.start()
        try:
            if mcp_build is not None:
                mcp_server=mcp_build()  # fresh FastMCP + session manager for this lifespan
                mcp_server.streamable_http_app()  # creates the session manager
                async with mcp_server.session_manager.run():
                    mcp_endpoint.manager=mcp_server.session_manager
                    try:
                        yield
                    finally:
                        mcp_endpoint.manager=None
            else:
                yield
        finally:
            presentation.close(wait=False)
            if decider is not None:decider.close()
    app=FastAPI(title="Optics LangGraph",version="0.1.0",lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","testserver"])
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="optics")
    active=set()
    lock=threading.Lock()
    worker_errors={}
    continue_after={}  # run_id -> control sequence taken when its continue was accepted
    precision_filenames = {"report":("result.json","application/json"),
                           "observations":("image_observations.json","application/json"),
                           "truth":("independent_evaluation.json","application/json"),
                           "optics":("optical_feedback.json","application/json"),
                           "audit":("acceptance-audit.json","application/json"),
                           "log":("run.log","text/plain"),
                           "camera":("global_a_latest.jpg","image/jpeg"),
                           "camera_b":("global_b_latest.jpg","image/jpeg"),
                           "wrist":("wrist_rgbd_latest.jpg","image/jpeg"),
                           "beam":("beam_camera_final.png","image/png"),
                           "video":("assembly_overview.mp4","video/mp4")}
    mzi_filenames = {"report":precision_filenames["report"], "log":precision_filenames["log"],
                     "camera":("mzi_a_pickup.png","image/png"), "camera_b":("mzi_b_pickup.png","image/png"),
                     "camera_c":("mzi_c_pickup.png","image/png"),
                     "observations":("pickup_observations.json","application/json"),
                     "truth":("actual_layout.json","application/json"),
                     "optics":("mzi-contact-optical-report.json","application/json"),
                     "video":precision_filenames["video"]}

    def artifact_names(state):
        return mzi_filenames if state.get("precision_kind") == "mzi_contact" else precision_filenames

    def precision_directory(state, requested_attempt=None):
        if state.get("selected_experiment") != "michelson_full" and state.get("precision_kind") != "mzi_contact":
            return None
        if state.get("precision_protocol") == "same_table_v1":
            dirs = state.get("precision_attempt_dirs", [])
            live = state.get("live_node")
            live_execution = live in ("precision_sim", "precision_repair")
            current_attempt = len(dirs) + int(live_execution)
            attempt = current_attempt if requested_attempt is None else requested_attempt
            if not 1 <= attempt <= current_attempt:
                return None
            root = runtime.run_dir(state["run_id"])
            if attempt > len(dirs):
                if live == "precision_sim" and state.get("precision_seed"):
                    origin = state["precision_origin"]
                    return runtime.run_dir(origin["run_id"]) / origin["directory"]
                return root / "team" / f"{live}-{state['live_attempt']:02d}" / "simulation"
            if dirs[attempt-1] is None:
                origin = state["precision_origin"]
                return runtime.run_dir(origin["run_id"]) / origin["directory"]
            return root / dirs[attempt-1]
        attempt = state.get("attempts", {}).get("precision_sim")
        if state.get("live_node") == "precision_sim":
            attempt = state["live_attempt"]
        if type(attempt) is int:
            if requested_attempt is not None:
                if not 1 <= requested_attempt <= attempt:
                    return None
                attempt = requested_attempt
            return runtime.run_dir(state["run_id"]) / "team" / f"precision_sim-{attempt:02d}" / "simulation"

    @app.middleware("http")
    async def local_only(request:Request,call_next):
        is_mcp=request.url.path=="/mcp" or request.url.path.startswith("/mcp/")
        if is_mcp or request.method in ("POST","PUT","DELETE","PATCH") or "/presentation" in request.url.path:
            origin=request.headers.get("origin")
            if request.headers.get("x-optics-local")!="1" or (origin and (urlparse(origin).hostname not in ("localhost","127.0.0.1","testserver") or urlparse(origin).netloc!=request.headers.get("host"))):
                return JSONResponse({"detail":"Only local workbench operations are accepted"},status_code=403)
            if is_mcp:  # stateless JSON endpoint: POST only, with an honest Content-Length (no chunked bypass)
                if request.method!="POST":
                    return JSONResponse({"detail":"The MCP endpoint accepts POST only"},status_code=405,headers={"Allow":"POST"})
                length=request.headers.get("content-length")
                if "transfer-encoding" in request.headers:
                    return JSONResponse({"detail":"Chunked MCP requests are not accepted"},status_code=400)
                if length is None:
                    return JSONResponse({"detail":"Content-Length is required"},status_code=411)
                if not (length.isascii() and length.isdigit()):
                    return JSONResponse({"detail":"Content-Length is invalid"},status_code=400)
                if int(length)>30000:
                    return JSONResponse({"detail":"Request is too large"},status_code=413)
            elif int(request.headers.get("content-length","0"))>30000:
                return JSONResponse({"detail":"Request is too large"},status_code=413)
        response=await call_next(request)
        response.headers["Cache-Control"]="no-store"
        response.headers["X-Content-Type-Options"]="nosniff"
        # The vendored Haloflux animation is the only page the panel itself may frame.
        ancestors="'self'" if request.url.path.startswith("/haloflux/") else "'none'"
        response.headers["Content-Security-Policy"]=f"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self'; frame-ancestors {ancestors}; form-action 'self'"
        return response

    def schedule(run_id,**kwargs):
        with lock:
            if run_id in active:raise HTTPException(409,"This run is already executing")
            active.add(run_id)
            worker_errors.pop(run_id,None)
        if kwargs.get("continue_run"):  # pause/stop requests after this point are newer than the continue
            kwargs["control_seq"]=continue_after[run_id]=runtime.control_seq(run_id)
        def work():
            try:runtime.execute(run_id,**kwargs)
            except Exception as exc:
                with lock:worker_errors[run_id]={"type":type(exc).__name__,"message":str(exc)[:500],
                                                 "time":datetime.now(timezone.utc).isoformat()}
                raise
            finally:
                with lock:active.discard(run_id);continue_after.pop(run_id,None)
        pool.submit(work)

    try:  # optional: without the `mcp` package the server is exactly as before, with no /mcp route
        mcp_runs=importlib.import_module(".mcp_runs",__package__)
    except ModuleNotFoundError as exc:
        if not (exc.name=="mcp" or (exc.name or "").startswith("mcp.")):
            raise  # a bug in mcp_runs or a different missing module must not be hidden
        mcp_runs=None
        logging.getLogger(__name__).warning("The `mcp` package is not importable (%s): /mcp is disabled",exc)
    if mcp_runs is not None:
        def mcp_submit(run_id,continue_run):
            try:schedule(run_id,continue_run=continue_run)
            except HTTPException as exc:raise mcp_runs.RunsError(f"conflict (HTTP {exc.status_code}): {exc.detail}") from None
        def mcp_display(run_id,status,is_active,control):
            return display_state(status,is_active,control,live=run_id in runtime.live_tasks)
        # One Runtime, one worker, one lock, one `active` set: the MCP tools borrow this app's.
        mcp_build=lambda:mcp_runs.build_hosted_server(runtime,mcp_submit,lock,active,worker_errors,mcp_display)
        app.router.routes.append(Route("/mcp",endpoint=mcp_endpoint))  # POST only; local_only applies to every method
        # Same RunsService as the /mcp tools: the panel's Codex runs are ordinary agent-decided runs.
        decider=cli_decider.CliDecider(runtime,mcp_runs.RunsService(mcp_runs.ServerBackend(
            runtime,mcp_submit,lock,active,worker_errors,mcp_display)))

    @app.get("/",response_class=HTMLResponse)
    def home():return (ROOT/"optics_graph/panel.html").read_text()

    # "Agent working" overlay: HALOFLUX by Ryder191, PolyForm Noncommercial 1.0.0 (see vendor/haloflux/NOTICE.txt).
    # "" is the optics-modified copy; upstream.html is the unmodified original, served for comparison.
    haloflux_files={"":("optics.html","text/html; charset=utf-8"),"upstream.html":("index.html","text/html; charset=utf-8"),
                    "wp.js":("wp.js","text/javascript; charset=utf-8"),
                    "LICENSE":("LICENSE","text/plain; charset=utf-8")}
    @app.get("/haloflux/{name:path}")
    def haloflux(name:str):
        if name not in haloflux_files:raise HTTPException(404,"Not found")
        filename,media_type=haloflux_files[name]
        return FileResponse(ROOT/"optics_graph/vendor/haloflux"/filename,media_type=media_type)

    @app.get("/healthz")
    def health():return {"ok":True,"version":"0.1.0","host_scope":"loopback","real_motion_enabled":False}

    @app.get("/api/settings")
    def settings():
        found=decider.available() if decider is not None else {}
        return {**public_settings(),"cli_deciders":{name:bool(found.get(name)) for name in cli_decider.BACKENDS}}

    @app.post("/api/settings")
    def settings_write(body:SettingsInput):
        try:return save_settings(body.model,body.api_key.get_secret_value())
        except ValueError:raise HTTPException(400,"The configuration format is invalid") from None

    def pending_control(run_id):
        try:request=read_json(runtime.run_dir(run_id)/"control.json")
        except (OSError,ValueError,AttributeError):return None
        if request.get("seq",0)<=continue_after.get(run_id,-1):return None  # older than the continue now waiting
        return request.get("action")

    @app.get("/api/runs")
    def runs():
        with lock:running=set(active)
        return [{"run_id":r["run_id"],"status":r["status"],"stage":r.get("stage"),"request":r["request"],
                 "profile":r["profile"],"provider":r["provider"],"mode":r["mode"],"created_at":r["created_at"],
                 "simulation_backend":r.get("simulation_backend", "physical"),
                 "active":r["run_id"] in running,
                 "display_state":display_state(r["status"],r["run_id"] in running,
                     pending_control(r["run_id"]) if r["run_id"] in running and r.get("simulation_backend")=="algorithm_v1" else None,
                     live=r["run_id"] in runtime.live_tasks)}
                for r in runtime.list()]

    @app.get("/api/runs/{run_id}")
    def read(run_id:str):
        try:r=runtime.get(run_id)
        except (ValueError,FileNotFoundError):raise HTTPException(404,"That run was not found") from None
        with lock:r["active"]=run_id in active
        if r.get("simulation_backend") == "algorithm_v1":
            control = runtime.run_dir(run_id)/"control.json"
            if control.is_file() and r["active"] and (request := pending_control(run_id)):
                r["control_request"] = request
        r["display_state"] = display_state(r["status"], r["active"], r.get("control_request"),
                                           live=run_id in runtime.live_tasks)
        if decider is not None and (info := decider.info(run_id)):
            r["decider"] = info
        task = runtime.live_tasks.get(run_id)
        if task:
            r["live_node"], r["live_attempt"] = task
            if r.get("precision_protocol") == "same_table_v1" and r["live_node"] in ("precision_sim", "precision_repair"):
                r["live_precision_attempt"] = len(r.get("precision_attempt_dirs", [])) + 1
            if r["active"] and r["live_node"] == "precision_reobserve":
                log = runtime.run_dir(run_id)/"team"/f"precision_reobserve-{r['live_attempt']:02d}"/"vision.log"
                pending = r.get("precision_pending_assessment") or {}
                pairs = pending.get("pairs", [])
                requested = ({pending["near_object"]} if pending.get("near_object") else
                             {target for target, _ in pairs} if pairs else EXPECTED_NAMES)
                r["live_observation"] = live_observation_progress(log, requested)
        output = precision_directory(r)
        if output is not None:
            r["precision_progress"] = precision_progress(output)
            r["precision_artifacts"] = [key for key,(name,_) in artifact_names(r).items()
                                        if (output / name).is_file()]
            r["precision_attempt_artifacts"] = True
            if r.get("precision_protocol") == "same_table_v1" and not r["active"]:
                audit_path = output / "acceptance-audit.json"
                if audit_path.is_file() and not audit_path.is_symlink():
                    try:
                        audit = read_json(audit_path)
                        with (output / "result.json").open("rb") as stream:
                            report_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                        if (audit["report_sha256"] == report_hash
                                and audit["scope"] == "saved_failed_attempt_diagnostic"
                                and audit["bindings"]["passed"] is True):
                            r["precision_attempt_evidence"] = {str(len(r["precision_attempt_dirs"])): {
                                "checks":{k:audit[k]["passed"] for k in ("mechanical","optical","sensor_feedback")},
                                "diagnostic":True,
                                "diagnostic_objects":{name:row for name,row in audit["mechanical"]["objects"].items()
                                    if not row["passed"] or name in (r.get("precision_result", {}).get("recovery") or {}).get("target_object_ids", [])},
                            }}
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
        return r

    @app.get("/api/runs/{run_id}/precision/{artifact}")
    def precision_artifact(run_id: str, artifact: str, attempt: int | None = None):
        state = read(run_id)
        output = precision_directory(state, attempt)
        filenames = artifact_names(state)
        if output is None or artifact not in filenames or (artifact == "video" and output != precision_directory(state)):
            raise HTTPException(404, "This round does not have that evidence")
        name, media_type = filenames[artifact]
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "This round does not have that evidence")
        if artifact == "video":
            expected = (state.get("verification") or {}).get("video_sha256")
            if not expected:
                raise HTTPException(404, "The replay has not been accepted")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise HTTPException(409, "The replay file does not match the accepted record")
        return FileResponse(path, media_type=media_type)

    @app.get("/api/runs/{run_id}/playback")
    def playback(run_id: str):
        state = read(run_id)
        expected = (state.get("verification") or {}).get("playback_sha256")
        attempt = state.get("attempts", {}).get("sim")
        if not expected or expected != state.get("playback_sha256") or type(attempt) is not int:
            raise HTTPException(404, "This round has no accepted simulation replay")
        path = runtime.run_dir(run_id) / "team" / f"sim-{attempt:02d}" / "playback.json"
        try:
            raw = read_bytes(path)
            if digest(raw) != expected:
                raise ValueError("Playback hash changed")
        except (ValueError, OSError):
            raise HTTPException(409, "The replay file does not match this round's acceptance record") from None
        return Response(raw, media_type="application/json")

    @app.post("/v1/pipelines/agent-loop/runs",status_code=202)
    @app.post("/api/runs",status_code=202)
    def start(body:RunInput):
        if body.provider == "agent":
            # An agent-decided run is useless without a client that answers its decision requests, and only
            # the MCP start_run tool registers one. Refuse instead of creating a run nobody can answer.
            raise HTTPException(422, "Agent-decided runs cannot be created here. Create them through the MCP "
                                     "start_run tool with decision_mode \"agent\"")
        run_id=runtime.create(body)
        schedule(run_id)
        return {"run_id":run_id,"status":"accepted"}

    @app.post("/api/experiments/runs", status_code=202)
    def start_experiment(body: ExperimentInput):
        # Agents interpret/design/recover; only execution is simulated.
        return start(RunInput(request=body.request, provider="openai", profile="auto",
                              mode="github_local", simulation_backend="algorithm_v1",
                              output_language=body.output_language))

    @app.post("/api/experiments/cli-runs", status_code=202)
    def start_cli_experiment(body: CliExperimentInput):
        # Agent-decided run, no API key: each decision goes to a fresh local `claude -p` or `codex exec` process
        # signed in with the user's own login (cli_decider.py).
        if decider is None:
            raise HTTPException(503,"Claude Code / Codex CLI runs need the optional `mcp` package (requirements-mcp.txt)")
        try:run_id=decider.start_run(body.request,body.decider)
        except cli_decider.DeciderError as exc:raise HTTPException(503,str(exc)) from None
        except mcp_runs.RunsError as exc:raise HTTPException(400,str(exc)) from None
        return {"run_id":run_id,"status":"accepted","source":mcp_runs.AGENT_SOURCE}

    @app.post("/api/runs/{run_id}/decider/retry", status_code=202)
    def retry_decider(run_id: str):
        if decider is None or not decider.retry(run_id):
            raise HTTPException(409,"This run is not decided by a local CLI decider")
        return {"status":"accepted"}

    @app.post("/api/runs/{run_id}/review",status_code=202)
    def review(run_id:str,body:ReviewInput):
        state=read(run_id)
        if state["status"]!="awaiting_review" or state.get("plan_sha256")!=body.plan_sha256:
            raise HTTPException(409,"The run is not paused, or the plan hash has changed")
        schedule(run_id,resume=body.model_dump())
        return {"status":"accepted","run_id":run_id}

    @app.post("/api/runs/{run_id}/continue",status_code=202)
    def continue_run(run_id:str):
        state=read(run_id)
        if state["status"] not in ("running","interrupted","queued","paused"):
            raise HTTPException(409,"Only a run interrupted by a process stop can be resumed")
        schedule(run_id,continue_run=state["status"]!="queued")
        return {"status":"accepted"}

    @app.post("/api/runs/{run_id}/pause", status_code=202)
    def pause(run_id: str):
        try:runtime.control(run_id, "pause")
        except (ValueError, FileNotFoundError) as exc:raise HTTPException(409,str(exc)) from None
        return {"status":"pause_requested"}

    @app.post("/api/runs/{run_id}/stop", status_code=202)
    def stop(run_id: str):
        try:runtime.control(run_id, "stop")
        except (ValueError, FileNotFoundError) as exc:raise HTTPException(409,str(exc)) from None
        return {"status":"stop_requested"}

    @app.get("/api/runs/{run_id}/algorithm/artifacts/{stage}/{attempt}/{filename}")
    def algorithm_artifact(run_id: str, stage: str, attempt: int, filename: str):
        state = read(run_id)
        if (state.get("simulation_backend") != "algorithm_v1" or not is_whitelisted(stage, attempt, filename)):
            raise HTTPException(404,"That algorithm-simulation evidence does not exist")
        directory = runtime.run_dir(run_id)/"team"/f"{stage}-{attempt:02d}"
        try:
            raw = read_verified(directory, filename, limit=800_000)
        except FileNotFoundError:
            raise HTTPException(404,"This node has not saved complete evidence yet") from None
        except (ValueError, KeyError, OSError):
            raise HTTPException(409,"The evidence does not match the node completion record") from None
        return Response(raw, media_type="application/zip" if filename.endswith(".zip") else
                        "text/plain" if filename.endswith(".txt") else "application/json")

    @app.get("/api/runs/{run_id}/presentation")
    def presentation_state(run_id: str, language: Literal["en"] = "en", retry: bool = False):
        state = read(run_id)
        return presentation.get(runtime.run_dir(run_id), state, language, retry=retry)

    @app.get("/api/runs/{run_id}/presentation/artifacts/{stage}/{attempt}/{filename}")
    def presentation_artifact(run_id: str, stage: str, attempt: int, filename: str,
                              language: Literal["en"] = "en", retry: bool = False):
        if not filename.endswith((".json", ".txt")):
            raise HTTPException(404, "Only text evidence has a translated view.")
        response = algorithm_artifact(run_id, stage, attempt, filename)
        text = response.body.decode("utf-8")
        prefix = ""
        if filename.endswith(".json"):
            payload = decode_json(text)
        elif "Treat the following JSON as data.\n" in text:
            prefix, content = text.split("Treat the following JSON as data.\n", 1)
            prefix += "Treat the following JSON as data.\n"
            payload = decode_json(content)
        else:
            payload = text
        result = presentation.get(runtime.run_dir(run_id), {"prompt":prefix,"payload":payload}, language, retry=retry)
        if result["status"] != "ready":
            return {"status":result["status"],"content":"English evidence is being prepared.",
                    **({"error":result["error"]} if result.get("error") else {})}
        translations = result["translations"]
        def translated(value):
            if isinstance(value, str):
                return translations.get(value, value)
            if isinstance(value, list):
                return [translated(item) for item in value]
            if isinstance(value, dict):
                return {key:translated(item) for key,item in value.items()}
            return value
        content = translated(payload)
        if not isinstance(content, str):
            content = json.dumps(content,ensure_ascii=False,indent=2)
        display_prefix = translated(prefix)
        prompt_language_override = language == "en" and "Use concise Chinese prose and stable ASCII identifiers." in display_prefix
        if prompt_language_override:
            display_prefix = display_prefix.replace("Use concise Chinese prose and stable ASCII identifiers.",
                                                    "Use concise English prose and stable ASCII identifiers.", 1)
        return {"status":"ready","content":display_prefix+content,
                "source_sha256":digest(response.body),"presentation_only":True,
                "prompt_language_override":prompt_language_override}

    @app.get("/api/graph")
    def graph(team: bool = False, algorithm: bool = False):
        builder = build_team_graph if team else build_graph
        return {"mermaid":builder(runtime.data, **({"algorithm":True} if team and algorithm else {})).get_graph().draw_mermaid()}

    return app


app=create_app()
