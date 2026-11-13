from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from .graph import build_graph
from .settings import DATA, DEFAULT_REQUEST
from .sim_io import read_json, read_bytes
from .team_flow import build_team_graph
from .algorithm_sim import AlgorithmOptions
from .agent_decisions import DecisionPending, locate_pending


class PrecisionSeed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    attempt: int = Field(ge=1, le=100)
    observation_run_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class RunInput(BaseModel):
    model_config=ConfigDict(extra="forbid")
    request: str=Field(default=DEFAULT_REQUEST,min_length=1,max_length=8000)
    provider: Literal["replay","openai","agent"]="replay"
    output_language: Literal["en"] = "en"
    profile: Literal["phase1_michelson","legacy_vision_michelson_15","auto"]="phase1_michelson"
    mode: Literal["mock","design","local_sim","remote_sim","real","legacy_evidence","legacy_mujoco","github_local","github_sim"]="mock"
    sim_scenario: Literal["nominal","transient","persistent"]="nominal"
    team_fault: Literal["none","vision","prompt","act","controller","sim","output","execution","matrix"]="none"
    team_entry: Literal["images","nl"]="images"
    team_transport: Literal["direct", "file_v1"] = "direct"
    prob0: dict | None = None
    playback: bool = False
    require_review: bool=False
    fail_action: str | None=None
    precision_seed: PrecisionSeed | None = None
    simulation_backend: Literal["physical", "algorithm_v1"] = "physical"
    algorithm_options: AlgorithmOptions = Field(default_factory=AlgorithmOptions)

    @model_validator(mode="after")
    def profiles(self):
        if self.provider == "agent" and self.simulation_backend != "algorithm_v1":
            raise ValueError("Agent-decided runs require the algorithm_v1 simulation backend")
        if self.simulation_backend == "algorithm_v1" and (
                self.mode not in {"github_sim", "github_local"} or self.profile != "auto"
                or self.precision_seed is not None):
            raise ValueError("Algorithm execution must use the team workflow and the general experiment entry. It cannot read a physical-bench checkpoint")
        if self.precision_seed and (self.profile != "auto" or self.mode not in {"github_sim", "github_local"}):
            raise ValueError("The current bench checkpoint is only for local simulation and handoff checks of a registered experiment")
        if self.profile == "auto" and (self.mode not in ("github_local","github_sim")
                                      or self.team_entry != "images"):
            raise ValueError("Automatic experiment matching requires the team workflow and vision input")
        if self.playback and self.mode not in ("github_local", "github_sim"):
            raise ValueError("Web simulation replay applies to the team workflow")
        if self.team_transport != "direct" and self.mode not in ("github_local", "github_sim"):
            raise ValueError("The file-job interface is only for the team workflow")
        legacy=self.profile=="legacy_vision_michelson_15"
        if legacy != self.mode.startswith("legacy_"):
            raise ValueError("The legacy 15-component mode and Phase 1 mode cannot be mixed")
        if legacy and self.provider!="replay":
            raise ValueError("A legacy case runs only its original fixed layout and cannot accept a new model layout")
        if self.fail_action and self.mode!="mock":
            raise ValueError("Fault injection is limited to protocol tests")
        if self.sim_scenario != "nominal" and self.mode != "local_sim":
            raise ValueError("Simulation fault scenarios support local MuJoCo only")
        if self.team_fault != "none" and self.mode not in ("github_local", "github_sim"):
            raise ValueError("Team-workflow faults are only for the local team workflow")
        if (self.team_entry == "nl") != (self.prob0 is not None):
            raise ValueError("The natural-language entry requires an explicit PROB0")
        if self.team_entry == "nl" and self.mode not in ("github_local", "github_sim"):
            raise ValueError("The PROB0 entry is only for the team workflow")
        if self.require_review and self.mode == "github_sim":
            raise ValueError("Simulation mode ends after Output. Review is only for the L1 handoff")
        return self


def write_json(path: Path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=".write-")
    try:
        with os.fdopen(fd,"w") as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


class Runtime:
    def __init__(self,data_dir:Path=DATA,gateway_factory=None, *, file_services=None):
        self.data=Path(data_dir)
        self.data.mkdir(parents=True,exist_ok=True)
        self.gateway_factory=gateway_factory
        self.file_services = file_services
        self.live_tasks = {}

    def run_dir(self,run_id):
        if not re.fullmatch(r"[a-f0-9]{32}",run_id):raise ValueError("Invalid run_id")
        return self.data/"runs"/run_id

    def create(self,request:RunInput)->str:
        run_id=uuid.uuid4().hex
        path=self.run_dir(run_id)
        path.mkdir(parents=True)
        input_data=request.model_dump()
        write_json(path/"input.json",input_data)
        write_json(path/"run.json",{**input_data,"input_sha256":hashlib.sha256(read_bytes(path/"input.json")).hexdigest(),
                                  "run_id":run_id,"status":"queued","stage":"queued","events":[],"created_at":datetime.now(timezone.utc).isoformat()})
        return run_id

    @contextmanager
    def graph(self, mode=None, *, algorithm=False):
        # Serialize this local workstation, including CLI + browser processes.
        with (self.data/"runtime.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with SqliteSaver.from_conn_string(str(self.data/"checkpoints.sqlite")) as saver:
                kwargs={"gateway_factory":self.gateway_factory} if self.gateway_factory else {}
                builder = build_team_graph if mode in ("github_local", "github_sim") else build_graph
                if builder is build_team_graph and self.file_services is not None:
                    kwargs["file_services"] = self.file_services
                if builder is build_team_graph and algorithm:
                    kwargs["algorithm"] = True
                yield builder(self.data,saver,**kwargs)

    @contextmanager
    def _control_lock(self, run_id):
        with (self.run_dir(run_id)/"control.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def control_seq(self, run_id) -> int:
        """Sequence number of the latest pause/stop request (0: none yet). A caller that accepts a continue takes
        this at that moment and passes it to execute(control_seq=...): requests numbered above it are newer than
        the continue and are honoured, older ones are stale. control-seq.json is never deleted, so it only grows."""
        if not self.run_dir(run_id).is_dir():
            return 0
        with self._control_lock(run_id):
            return self.control_seq_unlocked(run_id)

    def _control_after(self, run_id, after):
        """The control.json request with a sequence above `after` (the one a continue must honour), else None."""
        with self._control_lock(run_id):
            try:request = read_json(self.run_dir(run_id)/"control.json")
            except (OSError, ValueError):return None
            return request if request.get("seq", 0) > after else None

    def control(self, run_id, action):
        if self.get(run_id).get("simulation_backend") != "algorithm_v1" or action not in {"pause", "stop"}:
            raise ValueError("Only the algorithm-architecture build supports pause and stop at node boundaries")
        with self._control_lock(run_id):
            # The status is read UNDER the lock: a worker that stops at awaiting_decision publishes under the same
            # lock, so a stale snapshot can never be written back over a newer run.json.
            state = self.get(run_id)
            waiting = state["status"] == "awaiting_decision"
            if waiting and action == "pause":
                raise ValueError("Nothing is executing: the run is waiting for an agent decision. Submit the "
                                 "decision and continue it, or stop it. Pause applies only while a node executes")
            if state["status"] not in {"queued", "running", "interrupted", "paused"} and not waiting:
                raise ValueError("The run has finished and its status cannot be changed")
            if state["status"] in {"paused", "interrupted"} or waiting:
                state.pop("pending_decision", None)  # the request no longer describes a waiting run
                write_json(self.run_dir(run_id)/"run.json", {**state, "status":"stopped" if action == "stop" else "paused"})
            seq = self.control_seq_unlocked(run_id) + 1
            write_json(self.run_dir(run_id)/"control-seq.json", {"seq":seq})
            write_json(self.run_dir(run_id)/"control.json", {"action":action, "seq":seq})

    def control_seq_unlocked(self, run_id) -> int:
        try:return int(read_json(self.run_dir(run_id)/"control-seq.json")["seq"])
        except (OSError, ValueError, KeyError, TypeError):return 0

    def execute(self,run_id:str,resume:dict|None=None,continue_run=False,control_seq:int|None=None)->dict:
        path=self.run_dir(run_id)
        previous=self.get(run_id)
        config={"configurable":{"thread_id":run_id},"recursion_limit":100 if previous["mode"].startswith("github_") else 50}
        algorithm = previous.get("simulation_backend") == "algorithm_v1"
        with self.graph(previous["mode"], algorithm=algorithm) as graph:
            # Another process may have completed/paused this run while we waited for the lock.
            previous=self.get(run_id)
            snapshot=graph.get_state(config)

            def publish(values, status=None):
                public = {**values, "created_at": previous["created_at"],
                          **({"input_sha256":previous["input_sha256"]} if "input_sha256" in previous else {})}
                if status is not None:
                    public["status"] = status
                for key in ("layout", "plan", "validation", "plan_validation", "intent", "result", "manifest"):
                    if key in values:
                        write_json(path / f"{key}.json", values[key])
                write_json(path / "run.json", public)
                return public

            if resume is not None:
                if previous["status"]!="awaiting_review" or not snapshot.next:
                    raise ValueError("This run is not waiting for review")
                if type(resume.get("approved")) is not bool or resume.get("plan_sha256")!=snapshot.values.get("plan_sha256"):
                    raise ValueError("The review plan hash does not match, or approved is invalid")
                value=Command(resume=resume)
            elif continue_run:
                if control_seq is None:  # no caller captured one: every request so far predates this start
                    control_seq = self.control_seq(run_id)
                newer = algorithm and self._control_after(run_id, control_seq)
                stop_won = bool(newer) and newer["action"] == "stop" and previous["status"] == "stopped"
                if (previous["status"] not in ("running", "interrupted", "paused", "awaiting_decision") and not stop_won) or not snapshot.values:
                    raise ValueError("There is no interrupted computation to resume. Use approve or reject for review")
                # SQLite can have committed a state before the public files are written.
                if any(task.interrupts for task in snapshot.tasks):
                    return publish(snapshot.values, "awaiting_review")
                if not snapshot.next:
                    if snapshot.values.get("status") in (None, "queued", "running", "interrupted"):
                        raise ValueError("The checkpoint has no resumable final state")
                    return publish(snapshot.values)
                value=None
                if algorithm:
                    with self._control_lock(run_id):  # clear only requests older than the continue
                        try:stale = read_json(path/"control.json").get("seq", 0) <= control_seq
                        except (OSError, ValueError):stale = True
                        if stale:(path/"control.json").unlink(missing_ok=True)
            else:
                if snapshot.values or previous["status"]!="queued":
                    raise ValueError("The run has already started and cannot be started again. Check the result or resume it explicitly")
                try:
                    inputs = RunInput.model_validate(read_json(path / "input.json")).model_dump()
                    registered = RunInput.model_validate({key:previous[key] for key in RunInput.model_fields
                                                          if key in previous}).model_dump()
                    if (inputs != registered or previous.get("input_sha256") and
                            hashlib.sha256(read_bytes(path/"input.json")).hexdigest() != previous["input_sha256"]):
                        raise ValueError("The registered input or algorithm backend changed. Execution was refused")
                except (ValueError, OSError) as exc:
                    result = {**previous, "status":"failed", "stage":"input",
                              "error":{"code":"RUN_INPUT_INVALID", "message":str(exc)[:500]}}
                    write_json(path / "run.json", result)
                    return result
                if inputs["mode"] != previous["mode"]:
                    result = {**previous, "status":"failed", "stage":"input",
                              "error":{"code":"RUN_MODE_MISMATCH","message":"The registered run mode cannot be changed"}}
                    write_json(path/"run.json",result)
                    return result
                value={**inputs,"run_id":run_id,"status":"running","events":[],"error":None}
            try:
                controlled_status = None
                while True:
                    if algorithm and value is None and (path/"control.json").exists():
                        action = read_json(path/"control.json")["action"]
                        controlled_status = "paused" if action == "pause" else "stopped"
                        break
                    stream_options = {"interrupt_after":"*"} if algorithm else {}
                    for kind, state in graph.stream(value, config, stream_mode=["values", "tasks"], **stream_options):
                        if kind == "tasks":
                            if "input" in state:
                                payload = state["input"]
                                attempts = payload.get("attempts", {}) if isinstance(payload, dict) else {}
                                self.live_tasks[run_id] = (state["name"], attempts.get(state["name"], 0) + 1)
                            continue
                        if "__interrupt__" in state:
                            continue
                        publish(state)
                    boundary = graph.get_state(config)
                    if not algorithm or not boundary.next or any(t.interrupts for t in boundary.tasks):
                        break
                    value = None
                snapshot=graph.get_state(config)
                status = None
                if snapshot.next:
                    status = "awaiting_review" if any(t.interrupts for t in snapshot.tasks) else "interrupted"
                result = publish({**(snapshot.values or previous),
                                  "next_node":snapshot.next[0] if snapshot.next else None}, controlled_status or status)
            except DecisionPending as exc:
                # The node needs an agent answer: not a failure and not a LangGraph interrupt. The checkpoint
                # still points at the node, so continue re-runs it once the answer is stored.
                snapshot = graph.get_state(config)
                node = snapshot.next[0] if snapshot.next else None
                attempt = (snapshot.values.get("attempts") or {}).get(node, 0) + 1 if node else 0
                pending = locate_pending(path, node, attempt, exc.args[0]) if node else None
                # A stop/pause requested while this node ran is newer than the sequence this execution started
                # with. It must win over awaiting_decision, exactly as it would have at a node boundary (the
                # decision request file stays as evidence; a later continue re-runs the node, which asks again).
                # Checked and published under the control lock so a request cannot slip in between.
                with self._control_lock(run_id):
                    try:requested = read_json(path/"control.json")
                    except (OSError, ValueError):requested = None
                    if not (algorithm and isinstance(requested, dict)
                            and requested.get("seq", 0) > (control_seq or 0)):requested = None
                    if pending is None:
                        result = self.get(run_id)
                        result.pop("pending_decision", None)
                        result.update(status="interrupted", error={"code":"DecisionPending","message":"A decision request could not be located. Check the last node."})
                        write_json(path/"run.json", result)
                    elif requested is not None:
                        result = publish({**snapshot.values, "next_node":node},
                                         "paused" if requested.get("action") == "pause" else "stopped")
                    else:
                        result = publish({**snapshot.values, "next_node":node, "pending_decision":pending}, "awaiting_decision")
            except Exception as exc:
                result=self.get(run_id)
                result.pop("pending_decision", None)
                result.update(status="interrupted",error={"code":type(exc).__name__,"message":"The run was interrupted. Check the last node. Resume does not blindly resend a recorded physical motion."})
                write_json(path/"run.json",result)
            finally:
                self.live_tasks.pop(run_id, None)
        return result

    def get(self,run_id)->dict:
        return json.loads((self.run_dir(run_id)/"run.json").read_text())

    def list(self)->list[dict]:
        return [json.loads(p.read_text()) for p in sorted((self.data/"runs").glob("*/run.json"),key=lambda p:p.stat().st_mtime,reverse=True)][:30]
