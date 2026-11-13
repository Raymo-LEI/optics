"""Executable graph of GitHub Agent Loop Draft v0, with local service adapters."""
from datetime import datetime, timezone
from ast import literal_eval
from copy import deepcopy
from functools import partial
import math
import re
import json
import os
from operator import add
from pathlib import Path
import subprocess
import time
from typing import Literal, Annotated, TypedDict
from zipfile import BadZipFile

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .contracts import load_example, load_scene, validate_layout
from .experiments import (catalog, is_registered, load_experiment, match_registered_assembly, tightened_tolerance,
                          validate_registered_layout)
from .gateway import CandidateFormatError, GatewayError, ModelGateway, RetryableGatewayError
from .legacy import EXPECTED_NAMES, legacy_manifest, render_precision_replay, run_legacy, run_mzi_contact
from .precision_plan import (SCHEMA as PRECISION_PLAN_SCHEMA, accept_precision_l1, checkpoint_files, evidence_zip, file_hash,
                             read_checkpoint, validate_plan, verify_evidence)
from .precision_recovery import AGENT_NODES, PrecisionRepair, repair_evidence
from .settings import PROJECT, ROOT
from .sim_io import decode_json, read_json
from .team_artifacts import (accept_l1, atomic, digest, encode, evaluate_poses, read_journal_artifact,
                             make_archive, read_archive, read_bytes)
from .team_trajectory import CALIBRATION_ID, build_trajectory, check_trajectory, workbench_bounds
from .team_services import BLOB_LIMIT, FileTeamServices
from .team_playback import validate_playback

WORKER = Path(__file__).with_name("team_worker.py")
NEXT = {"vision":"prompt","prompt":"act","act":"validator","validator":"controller",
        "controller":"sim","sim":"output","output":"mode","execution":"done"}
RETURNS = {"input":{"input"}, "vision":{"input"}, "prompt":{"prompt"}, "act":{"act"}, "validator":{"prompt","act"},
           "controller":{"prompt","act","controller"}, "sim":{"prompt","act","controller","sim"},
           "output":{"prompt","act","controller","sim"}, "precision_output":set(AGENT_NODES)}
RETRY_LIMIT = 2
ALGORITHM_PLACEMENT_CODES = ("ALGORITHM_POSITION_MISMATCH", "ALGORITHM_OPTICAL_MISMATCH")
TOTAL_RETRIES = 7
DOWNSTREAM = {
    "input": ("prob", "scene", "intent", "layout", "validation", "plan_path", "plan_sha256", "artifact_path"),
    "prompt": ("intent", "layout", "validation", "plan_path", "plan_sha256", "artifact_path"),
    "act": ("layout", "validation", "plan_path", "plan_sha256", "artifact_path"),
    "controller": ("plan_path", "plan_sha256", "artifact_path"),
    "sim": ("artifact_path",),
}


class StageError(ValueError):
    def __init__(self, code, message, target=None, details=None):
        super().__init__(message)
        self.code = code
        self.target = target
        self.details = details


class TeamState(TypedDict, total=False):
    output_language: Literal["en"]
    simulation_backend: str
    algorithm_options: dict
    algorithm_experiment: dict
    algorithm_intent: dict
    algorithm_design: dict
    algorithm_review: dict
    algorithm_test_suite: dict
    algorithm_test_report: dict
    agent_model: str
    algorithm_state: dict
    algorithm_result: dict
    algorithm_pending: dict | None
    algorithm_history: Annotated[list[dict], add]
    algorithm_decisions: Annotated[list[dict], add]
    run_id: str
    request: str
    provider: str
    profile: str
    selected_experiment: str
    local_registered_request: bool
    reference_layout: dict
    mode: str
    team_fault: str
    team_entry: str
    team_transport: str
    playback: bool
    prob0: dict | None
    require_review: bool
    status: str
    stage: str
    events: Annotated[list[dict], add]
    attempts: dict
    retry_counts: dict
    retry_log: Annotated[list[dict], add]
    error: dict | None
    return_to: str
    feedback: list[dict]
    prob: dict
    scene: dict
    intent: dict
    layout: dict
    validation: dict
    plan: dict
    plan_path: str
    plan_sha256: str
    artifact_path: str
    artifact_sha256: str
    playback_sha256: str | None
    verification: dict | None
    approved: bool
    approval: dict
    result: dict
    precision_result: dict
    precision_kind: str
    precision_actions: list[str]
    precision_guard_parts: list[str]
    precision_repairs: Annotated[list[dict], add]
    precision_protocol: str
    precision_seed: dict | None
    precision_origin: dict
    precision_attempt_dirs: list[str | None]
    precision_pending: dict | None
    precision_prior_repairs: list[dict]
    precision_pending_assessment: dict | None
    precision_assessment: dict | None


def moving_ids(scene):
    return {name for name, obj in scene["objects"].items() if obj["movable"]}


def validate_prob(prob, scene):
    if (not isinstance(prob, dict)
            or set(prob) != {"schema_version","calibration_id","scene_id","frame_id","source","frames","observations"}
            or prob["schema_version"] != "local-prob/v0" or prob["calibration_id"] != CALIBRATION_ID
            or prob["scene_id"] != scene["scene_revision"] or prob["frame_id"] != "optical_table"
            or prob["source"] not in ("mujoco_rgb_two_view", "mujoco_aruco_two_view", "provided")
            or not isinstance(prob["observations"], dict) or set(prob["observations"]) != moving_ids(scene)
            or not isinstance(prob["frames"], list)
            or any(not isinstance(f, dict) or set(f) != {"camera_id","rgb_sha256","size"}
                   or not isinstance(f["rgb_sha256"], str) for f in prob["frames"])
            or [f["camera_id"] for f in prob["frames"]] != ([] if prob["source"] == "provided" else ["camera_0","camera_1"])
            or any(set(f) != {"camera_id","rgb_sha256","size"}
                   or f["size"] != ([1280,960] if prob["source"] == "mujoco_aruco_two_view" else [640,480])
                   or len(f["rgb_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in f["rgb_sha256"])
                   for f in prob["frames"])):
        raise ValueError("PROB0 scene, calibration or cameras rejected")
    (xmin,xmax),(ymin,ymax),_ = workbench_bounds(scene["scene_revision"])
    for obs in prob["observations"].values():
        if (not isinstance(obs, dict) or set(obs) != {"position_m","yaw_rad"}
                or not isinstance(obs["position_m"], list)):
            raise ValueError("PROB0 pose fields rejected")
        xyz, angle = obs["position_m"], obs["yaw_rad"]
        if (set(obs) != {"position_m","yaw_rad"} or len(xyz) != 3
                or not all(type(n) in (float,int) and math.isfinite(n) for n in [*xyz, angle])
                or not xmin <= xyz[0] <= xmax or not ymin <= xyz[1] <= ymax
                or xyz[2] != 0 or abs(angle) > math.pi):
            raise ValueError("PROB0 pose rejected")


def worker(stage, path, request, timeout=40):
    payload = encode(request)
    request_path = path / "request.json"
    if request_path.exists() and read_bytes(request_path) != payload:
        raise ValueError("Worker attempt request changed")
    atomic(request_path, payload)
    answer = path / "response.json"
    if not answer.exists():
        python = PROJECT / ".venv-mujoco/bin/python"
        proc = subprocess.run([str(python), "-E", "-s", str(WORKER), stage, str(path)],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin"}, timeout=timeout, check=False)
        if proc.returncode:
            lines = proc.stderr.decode(errors="replace").strip().splitlines()
            raise ValueError(lines[-1][:250] if lines else f"worker exit {proc.returncode}")
    return read_json(answer)


def build_team_graph(data_dir, checkpointer=None, gateway_factory=ModelGateway, file_services=None, *, algorithm=False):
    services = file_services if file_services is not None else FileTeamServices()

    def use_files(state):
        return state.get("team_transport") == "file_v1"

    def directory(state):
        return data_dir / "runs" / state["run_id"]

    def gateway(state, path):
        # Keep every model candidate with its stage attempt, including Prompt retries.
        return gateway_factory(state["provider"], path)

    def injected(state, name, attempt):
        fault = state.get("team_fault")
        return attempt == 1 and (fault == name or (fault == "matrix" and name != "execution"))

    def receive(state, path, attempt):
        if state["profile"] == "auto":
            options = catalog()
            selected = state.get("selected_experiment")
            matched = match_registered_assembly(state["request"], options)
            if selected is None:
                selected = matched
                if selected is None:
                    if state["provider"] == "replay":
                        raise StageError("EXPERIMENT_UNAVAILABLE", "An offline local run only accepts a registered experiment that matches directly")
                    choice = gateway(state, path).select_experiment(state["request"], options)
                    selected = choice["profile_id"]
                    if selected not in options:
                        raise StageError("EXPERIMENT_UNAVAILABLE", "No matching experiment configuration was found",
                                         details={"reason":choice["reason"], "available":[
                                             f'{spec["title"]} ({spec["count"]} components)' for spec in options.values()]})
            if state.get("precision_seed") and selected != "michelson_full":
                raise StageError("PRECISION_CHECKPOINT_SCOPE", "The saved bench is only for the registered complete Michelson")
            scene, reference = load_experiment(selected)
            return {"scene":scene, "selected_experiment":selected, "reference_layout":reference,
                    "local_registered_request":matched == selected,
                    "precision_kind": "mzi_contact" if selected == "mach_zehnder" else "",
                    **({"precision_protocol":"same_table_v1"} if selected == "michelson_full" else {})}
        scene = load_scene()
        if state["team_entry"] == "images":
            return {"scene":scene}
        prob = state["prob0"]
        validate_prob(prob, scene)
        atomic(path/"prob0.json",encode(prob))
        return {"prob":prob,"scene":scene}

    def vision(state, path, attempt):
        scene = state["scene"]
        if same_table(state):
            from .precision_contract import observe_michelson, observed_checkpoint
            origin = None
            try:
                if state.get("precision_seed"):
                    origin, failed = seed_checkpoint(state)
                    observation_checkpoint = origin
                    if failed_single_repair(failed):
                        observation_checkpoint = pre_repair_checkpoint(read_checkpoint(data_dir, origin).parent, failed)
                    source = read_checkpoint(data_dir, observation_checkpoint)
                    prob = observed_checkpoint(PROJECT, source, expected_result_sha256=observation_checkpoint["files"]["result.json"])
                else:
                    prob = observe_michelson(PROJECT, path / "vision")
                validate_precision_observation(state, prob, partial=origin is not None)
                atomic(path / "prob0.json", encode(prob))
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                raise StageError("PRECISION_OBSERVATION_REJECTED", str(exc), "input") from exc
            if not origin:
                return {"prob":prob}
            previous = read_json(data_dir / "runs" / origin["run_id"] / "run.json", limit=4_000_000)
            inherited = {}
            if previous.get("precision_protocol") == "same_table_v1":
                prior = [*previous.get("precision_prior_repairs", []), *previous.get("precision_repairs", [])]
                counts = dict(previous.get("retry_counts", {}))
                actions = list(previous.get("precision_actions", []))
                pending = previous.get("precision_pending") or {}
                error = previous.get("error") or {}
                if error.get("stage") == "controller" and error.get("code") == "PRECISION_TARGET_UNOBSERVED" and prior:
                    latest = prior[-1]
                    routes = [row for row in previous.get("retry_log", []) if row.get("reason") == "retry_allowed"]
                    controller = data_dir/"runs"/origin["run_id"]/"team"/f"controller-{previous['attempts']['controller']:02d}"
                    journal = read_json(controller/"completion.json", limit=None)
                    route = routes[-1] if routes else {}
                    key = route.get("retry_key", route.get("to"))
                    if (latest.get("action") == pending.get("action") == "replan_packed_lift"
                            and latest.get("target_object_ids") == pending.get("target_object_ids")
                            and latest.get("failed_attempt") == state["precision_seed"]["attempt"]
                            and pending.get("checkpoint") == observation_checkpoint
                            and journal["update"].get("error") == error
                            and journal["update_sha256"] == digest(encode(journal["update"], limit=None))
                            and not (controller/"plan.json").exists() and actions and actions[-1] == latest["action"]
                            and route.get("to") == latest.get("agent") and counts.get(key, 0) > 0):
                        prior.pop()
                        actions.pop()
                        counts[key] -= 1
                for row in prior:
                    agent, parts = row["agent"], row.get("target_object_ids", [])
                    if len(parts) == 1 and counts.get(agent, 0) > 0:
                        counts[agent] -= 1
                        key = f"{agent}:{parts[0]}"
                        counts[key] = counts.get(key, 0) + 1
                    if row.get("failure_code") == "POST_REPAIR_NEAR_REVIEW":
                        name = row.get("observation_object_id")
                        old, new = f"{agent}:{name}:post_repair", f"{agent}:{name}:near_review"
                        layout_used = any(v.get("failure_code") == "POST_REPAIR_LAYOUT_REVIEW"
                                          and v.get("completed_object_id") == name for v in prior)
                        if name in EXPECTED_NAMES and new not in counts and not layout_used and old in counts:
                            counts[new] = counts.pop(old)
                inherited = {"retry_counts":counts, "precision_prior_repairs":prior,
                             "precision_actions":actions}
            return {"prob":prob, "precision_origin":origin, **inherited}
        try:
            payload = {"scene": scene, "layout": state.get("reference_layout") or load_example("layout")}
            prob = services.invoke("vision", path, payload) if use_files(state) else worker("vision", path, payload)
            if injected(state, "vision", attempt):
                prob["frames"] = prob["frames"][:1]
            atomic(path / "prob0.json", encode(prob))
            validate_prob(prob, scene)
            expected_source = "mujoco_aruco_two_view" if is_registered(scene) else "mujoco_rgb_two_view"
            if prob["source"] != expected_source:
                raise ValueError("Vision must return calibrated camera observations")
        except (ValueError, KeyError, TypeError, subprocess.TimeoutExpired, OSError) as exc:
            raise StageError("VISION_INPUT_REJECTED", str(exc), "input") from exc
        return {"prob":prob,"scene":scene}

    def prompt(state, path, attempt):
        # The draft's Prompt record binds user requirements, PROB0 and prior feedback.
        if state.get("local_registered_request"):
            intent = {"experiment":"registered_geometric", "requirements":{}, "unsupported_requirements":[],
                      "summary":state["request"], "source":"registered_request"}
        else:
            intent = gateway(state, path).interpret(state["request"], state["scene"], state.get("feedback", []))
        expected = "registered_geometric" if is_registered(state["scene"]) else "michelson_geometric"
        if intent["experiment"] != expected or intent["unsupported_requirements"]:
            raise StageError("UNSUPPORTED_REQUIREMENTS", "The requirements do not match the selected experiment configuration",
                             details=intent["unsupported_requirements"])
        if injected(state, "prompt", attempt):
            intent = {**intent, "requirements":{"unknown_input_field":True}}
        atomic(path/"prompt.json", encode({"request":state["request"],"prob0":state.get("prob"),
                                          "intent":intent,"feedback":state.get("feedback",[])}, limit=None))
        return {"intent":intent}

    def precise_experiment(state):
        return state.get("simulation_backend") != "algorithm_v1" and state.get("profile") == "auto" and (
            state.get("selected_experiment") == "michelson_full" or state.get("precision_kind") == "mzi_contact")

    def same_table(state):
        return state.get("precision_protocol") == "same_table_v1"

    def seed_checkpoint(state):
        seed = state["precision_seed"]
        run_id, attempt = seed["run_id"], seed["attempt"]
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid source run ID")
        run_root = data_dir / "runs" / run_id
        previous = read_json(run_root / "run.json", limit=4_000_000)
        if previous.get("precision_protocol") == "same_table_v1":
            directories = previous.get("precision_attempt_dirs", [])
            if previous.get("status") != "failed" or attempt != len(directories):
                raise ValueError("Continuation requires the last failed same-table checkpoint")
            relative = directories[-1]
            if not isinstance(relative, str) or not re.fullmatch(
                    r"team/precision_(?:sim|repair)-[0-9]+/simulation", relative):
                raise ValueError("Continuation source is not an executed physical attempt")
            parent = (run_root / relative).parent
        else:
            parent = run_root / "team" / f"precision_sim-{attempt:02d}"
        journal = read_json(parent / "completion.json", limit=None)
        raw = read_bytes(parent / "precision-result.json")
        result = decode_json(raw.decode("utf-8"))
        if (journal["files"]["precision-result.json"] != digest(raw)
                or journal["update_sha256"] != digest(encode(journal["update"], limit=None))
                or journal["update"].get("precision_result") != result
                or result.get("bindings_verified") is not True
                or result.get("success") is not False
                or result.get("source_sha256") != state["scene"]["registration"]["source_sha256"]
                or not (str(result.get("failure")).startswith("RuntimeError: FINAL_VISUAL_LAYOUT_CHECK_FAILED:")
                        or failed_single_repair(result))):
            raise ValueError("Source is not a completed, bound placement-failure checkpoint")
        source = parent / "simulation"
        files = checkpoint_files(source)
        report = json.loads((source / "result.json").read_text())
        if (report["failure"] != result["failure"]
                or report["trajectory_sha256"] != files["trajectory.npz"]
                or report["world_model_sha256"] != files["world_model.mjb"]
                or report["source_layout_sha256"] != result["source_sha256"]):
            raise ValueError("Source physical checkpoint does not match the failed attempt")
        return {"run_id":run_id, "directory":str(source.relative_to(data_dir/"runs"/run_id)),
                "files":files}, result

    def failed_single_repair(result):
        failure = result.get("failure")
        if not isinstance(failure, str) or not failure.startswith("RuntimeError: "):
            return None
        code, _, part = failure.removeprefix("RuntimeError: ").partition(":")
        if part in EXPECTED_NAMES and code in {"PACKED_CARRIER_LIFT_CONTACT", "REPAIR_VISUAL_CHECK_FAILED"}:
            return code, part
        if code == "NO_VISUAL_GRASP_AXIS" and len(part) <= 8000 and result.get("recovery_actions") == ["replan_packed_lift"]:
            scope = (result.get("recovery") or {}).get("target_object_ids", [])
            try:
                screen = literal_eval(part)
            except (ValueError, SyntaxError, TypeError):
                return None
            if (len(scope) == 1 and scope[0] in EXPECTED_NAMES and isinstance(screen, dict)
                    and screen.get("valid") is False and screen.get("reason") == "NO_COLLISION_SCREENED_LIFT"):
                return "PACKED_CARRIER_LIFT_UNPROVEN", scope[0]
        if code == "TRANSPORT" and len(part) <= 200 and result.get("recovery_actions") in (
                ["refine_repair_seating"], ["replan_corridor_lift"]):
            try:
                time_s, phase, contact = literal_eval(part)
            except (ValueError, SyntaxError, TypeError):
                return None
            if (type(time_s) in (int, float) and math.isfinite(time_s) and time_s >= 0
                    and isinstance(phase, str) and contact == "carried module/other module"):
                name, _, step = phase.partition(": ")
                if name in EXPECTED_NAMES and step in {"visual hover", "lower", "release"}:
                    return "PACKED_CARRIER_SEATING_CONTACT", name
        return None

    def pre_repair_checkpoint(parent, result):
        code, part = failed_single_repair(result)
        raw = read_bytes(parent / "plan.json")
        expected = result.get("plan_hash") or result.get("imported_plan_hash")
        if digest(raw) != expected:
            raise ValueError("Single-part failure is not bound to its executed plan")
        plan = decode_json(raw.decode("utf-8"))
        recovery = result.get("recovery") or {}
        checkpoint = plan.get("checkpoint")
        if (plan.get("operation") != "repair_part" or plan.get("object_ids") != [part]
                or recovery.get("target_object_ids") != [part]
                or recovery.get("commanded_object_ids") != [part]
                or recovery.get("retained_scene") is not True
                or (code == "PACKED_CARRIER_SEATING_CONTACT" and plan.get("repair_action") not in {
                    "refine_repair_seating", "replan_corridor_lift"})
                or (code == "PACKED_CARRIER_LIFT_UNPROVEN" and plan.get("repair_action") != "replan_packed_lift")
                or not checkpoint or recovery.get("source_report_sha256") != checkpoint["files"]["result.json"]):
            raise ValueError("No verified pre-repair checkpoint for the named carrier")
        read_checkpoint(data_dir, checkpoint)
        return checkpoint

    def validate_precision_observation(state, prob, *, partial):
        manifest = legacy_manifest(PROJECT)
        observations = prob["observations"]
        if (prob.get("source_layout_sha256") != state["scene"]["registration"]["source_sha256"]
                or prob.get("calibration_sha256") != manifest["calibration_sha256"]
                or prob.get("scene_id") != "michelson_full" or prob.get("frame") != "world"
                or not isinstance(observations, dict) or not set(observations) <= EXPECTED_NAMES
                or not observations or (not partial and set(observations) != EXPECTED_NAMES)):
            raise ValueError("RGB layout source, frame, calibration or inventory rejected")
        for item in observations.values():
            matrix = item["T_world_object"]
            if (len(matrix) != 4 or any(len(row) != 4 for row in matrix)
                    or matrix[3] != [0., 0., 0., 1.]
                    or not all(type(n) in (int, float) and math.isfinite(n) for row in matrix for n in row)):
                raise ValueError("RGB observation pose rejected")

    def rollback_observation(checkpoint, record):
        run_id, relative = record["run_id"], record["directory"]
        if (not re.fullmatch(r"[0-9a-f]{32}", run_id)
                or not re.fullmatch(r"team/precision_repair-[0-9]+/simulation", relative)):
            raise ValueError("Rollback observation source rejected")
        parent = (data_dir/"runs"/run_id/relative).parent
        plan_raw = read_bytes(parent/"plan.json")
        plan = decode_json(plan_raw.decode("utf-8"))
        if digest(plan_raw) != record["plan_sha256"] or plan["checkpoint"] != checkpoint:
            raise ValueError("Rollback observation refers to a different executed plan or checkpoint")
        expected = plan["observed_layout_sha256"]
        # Older attempts retained the exact Controller input in the Controller journal only.
        for candidate in (parent, *sorted(parent.parent.glob("controller-*"))):
            if not (candidate/"observed-layout.json").is_file() or not (candidate/"plan.json").is_file():
                continue
            if digest(read_bytes(candidate/"plan.json")) != record["plan_sha256"]:
                continue
            raw = read_bytes(candidate/"observed-layout.json")
            journal = read_json(candidate/"completion.json", limit=None)
            if (digest(raw) != expected or journal["files"].get("observed-layout.json") != expected
                    or journal["files"].get("plan.json") != record["plan_sha256"]
                    or journal["update_sha256"] != digest(encode(journal["update"], limit=None))):
                raise ValueError("Rollback observation bytes changed")
            return decode_json(raw.decode("utf-8"))
        raise ValueError("No observation file is bound to the failed repair plan")

    def precision_controller(state, path):
        repair = state.get("precision_pending")
        prob = state["prob"]
        checkpoint = None
        if repair:
            checkpoint = repair["checkpoint"]
            source = read_checkpoint(data_dir, checkpoint)
            if repair.get("assessment_sha256"):
                prob = verified_assessment(state, checkpoint)
            elif repair.get("rollback_observation"):
                prob = rollback_observation(checkpoint, repair["rollback_observation"])
            else:
                from .precision_contract import observed_checkpoint
                prob = observed_checkpoint(PROJECT, source, expected_result_sha256=checkpoint["files"]["result.json"])
            validate_precision_observation(state, prob, partial=True)
            if not set(repair["target_object_ids"]) <= set(prob["observations"]):
                raise StageError("PRECISION_TARGET_UNOBSERVED", "The faulty component is not bound to this bench's RGB observation")
            if "placement_feedback" in repair:
                prob = {**prob, "placement_feedback":repair["placement_feedback"]}
        prob_raw = encode(prob)
        layout_hash = digest(encode(state["layout"]))
        observation_hash = digest(prob_raw)
        action = repair["action"] if repair else None
        objects = repair["target_object_ids"] if repair else sorted(EXPECTED_NAMES)
        operation = ("repair_part" if objects else "repair_optics") if repair else "assemble"
        plan = {"schema_version":PRECISION_PLAN_SCHEMA, "scene_id":"michelson_full",
                "source_layout_sha256":state["scene"]["registration"]["source_sha256"],
                "target_layout_sha256":layout_hash, "observed_layout_sha256":observation_hash,
                "operation":operation, "object_ids":objects, "checkpoint":checkpoint, "repair_action":action}
        validate_plan(plan, layout_hash, observation_hash)
        raw = encode(plan)
        atomic(path / "observed-layout.json", prob_raw)
        atomic(path / "plan.json", raw)
        atomic(path / "plan.sha256", (digest(raw)+"\n").encode())
        return {"plan":plan,"plan_path":str((path/"plan.json").relative_to(directory(state))),
                "plan_sha256":digest(raw),"verification":None}

    def planned_precision_sim(state, path, *, repair=False):
        root = directory(state)
        plan_file = root / state["plan_path"]
        raw = read_bytes(plan_file)
        prob_raw = read_bytes(plan_file.with_name("observed-layout.json"))
        if digest(raw) != state["plan_sha256"]:
            raise StageError("ROSI_CHANGED", "Controller ROSi changed before simulation")
        plan = decode_json(raw.decode("utf-8"))
        validate_plan(plan, digest(encode(state["layout"])), digest(prob_raw))
        if plan["source_layout_sha256"] != state["scene"]["registration"]["source_sha256"]:
            raise ValueError("ROSi target source changed")
        if repair != (plan["operation"] != "assemble"):
            raise ValueError("Wrong simulation entry for this ROSi")
        runs = state.get("precision_attempt_dirs", [])
        output = path / "simulation"
        executed_plan = path / "plan.json"
        if executed_plan.exists() and read_bytes(executed_plan) != raw:
            raise StageError("ROSI_CHANGED", "The original-bench correction attempt is bound to a different plan")
        if not executed_plan.exists():
            atomic(executed_plan, raw)
        executed_observation = path / "observed-layout.json"
        if executed_observation.exists() and read_bytes(executed_observation) != prob_raw:
            raise StageError("ROSI_CHANGED", "The original-bench correction attempt is bound to a different camera observation")
        if not executed_observation.exists():
            atomic(executed_observation, prob_raw)
        imported = not repair and state.get("precision_seed")
        if imported:
            origin, result = seed_checkpoint(state)
            if origin != state["precision_origin"]:
                raise ValueError("Imported checkpoint changed after observation")
            relative = None
        else:
            relative = str(output.relative_to(root))
            if repair:
                source = read_checkpoint(data_dir, plan["checkpoint"])
                objects = plan["object_ids"]
                options = {"resume_dir":source}
                if objects:
                    options["repair_part"] = objects[0]
                    actions = []  # The scoped worker reuses the existing one-item placement operation.
                    if plan["repair_action"] in {"replan_packed_lift", "refine_repair_seating",
                                                  "replan_supported_turn_lift", "replan_corridor_lift"}:
                        actions = [plan["repair_action"]]
                    if plan["repair_action"] == "guard_camera_margin":
                        options["guard_parts"] = objects
                        actions = ["guard_camera_margin"]
                    if plan["repair_action"] == "replan_visual_compensation":
                        options["guard_parts"] = objects
                        options["repair_observation"] = executed_observation
                        actions = [plan["repair_action"]]
                else:
                    options["repair_stage"] = "optical"
                    actions = [plan["repair_action"]]
                result = run_legacy(PROJECT, output, timeout_s=7200, added_noise=False,
                                    recovery_actions=actions, **options)
                evidence = result.get("recovery") or {}
                if result.get("success") and (
                    evidence.get("retained_scene") is not True
                    or evidence.get("source_report_sha256") != plan["checkpoint"]["files"]["result.json"]
                    or evidence.get("target_object_ids") != objects
                    or evidence.get("commanded_object_ids") != objects
                    or set(evidence.get("untouched_object_ids", [])) != EXPECTED_NAMES-set(objects)
                ):
                    raise StageError("PRECISION_SCOPE_UNPROVEN", "The single-component correction record does not match the approved target scope")
            else:
                result = run_legacy(PROJECT, output, timeout_s=5400, added_noise=False)
        recorded = {**result, "plan_hash":None if imported else digest(raw)}
        if imported and result.get("plan_hash"):
            recorded["imported_plan_hash"] = result["plan_hash"]
        archive = evidence_zip(raw, recorded)
        atomic(path/"precision-result.json", encode(recorded))
        atomic(path/"artifact.zip", archive)
        return {"precision_result":recorded, "precision_attempt_dirs":[*runs,relative],
                "precision_pending":None, "precision_assessment":None,
                "artifact_path":str((path/"artifact.zip").relative_to(root)),
                "artifact_sha256":digest(archive)}

    def verified_assessment(state, checkpoint):
        assessment = state["precision_assessment"]
        if assessment["checkpoint"] != checkpoint:
            raise ValueError("Read-only assessment does not refer to this repair checkpoint")
        if not re.fullmatch(r"team/precision_reobserve-[0-9]+/assessment\.json", assessment["path"]):
            raise ValueError("Read-only assessment path rejected")
        record = directory(state)/assessment["path"]
        raw = read_bytes(record, limit=BLOB_LIMIT)
        if digest(raw) != assessment["sha256"]:
            raise ValueError("Read-only assessment changed after vision")
        pending = state.get("precision_pending") or {}
        if pending.get("assessment_sha256", assessment["sha256"]) != assessment["sha256"]:
            raise ValueError("Repair proposal refers to a different assessment")
        result = decode_json(raw.decode("utf-8"))
        evidence = record.parent/"vision"
        if assessment.get("observation_run_id"):
            imported = previous_observation(state, assessment["observation_run_id"], checkpoint,
                                            near_object=assessment.get("near_object"),
                                            pair_keys=assessment.get("close_pairs", {}))
            if imported is None:
                raise ValueError("Imported observation record is missing")
            evidence, _ = imported
        if (file_hash(evidence/"result.json") != result["result_sha256"]
                or file_hash(evidence/"observation_windows.json") != result["image_observations_sha256"]):
            raise ValueError("Re-observation raw evidence changed")
        return result

    def previous_observation(state, run_id, checkpoint, *, near_object=None, pair_keys=()):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id) or run_id == state["run_id"]:
            raise ValueError("Observation import run ID rejected")
        root = data_dir/"runs"/run_id
        previous = read_json(root/"run.json", limit=4_000_000)
        seed = state["precision_seed"]
        previous_seed = previous.get("precision_seed") or {}
        error = previous.get("error") or {}
        number = previous.get("attempts", {}).get("precision_reobserve")
        if (previous.get("status") != "failed" or previous.get("precision_origin") != state["precision_origin"]
                or any(previous_seed.get(k) != seed[k] for k in ("run_id", "attempt"))
                or type(number) is not int or number < 1):
            raise ValueError("Observation import was not collected for this failed attempt")
        recorded = next((item["assessment"] for item in reversed(previous.get("events", []))
                         if item.get("stage") == "precision_reobserve" and item.get("assessment")
                         and item["assessment"].get("checkpoint") == checkpoint
                         and item["assessment"].get("near_object") == near_object
                         and set(item["assessment"].get("close_pairs", {})) == set(pair_keys)), None)
        if recorded:
            if not re.fullmatch(r"team/precision_reobserve-[0-9]+/assessment\.json", recorded.get("path", "")):
                raise ValueError("Observation import assessment path rejected")
            parent = (root/recorded["path"]).parent
        elif near_object:
            return None  # This run may have a wide sweep but no observation at the requested range.
        else:
            parent = root/"team"/f"precision_reobserve-{number:02d}"
        journal = read_json(parent/"completion.json", limit=None)
        if (journal["update_sha256"] != digest(encode(journal["update"], limit=None))
                or journal["files"].get("vision.log") != file_hash(parent/"vision.log")):
            raise ValueError("Observation import completion record changed")
        rejected = (error.get("stage") == "precision_reobserve"
                    and error.get("code") == "PRECISION_REOBSERVATION_REJECTED"
                    and journal["update"].get("error") == error)
        if rejected and not recorded:
            return parent/"vision", None
        if (
            not recorded or recorded.get("observation_run_id")
            or journal["update"].get("precision_assessment") != recorded
            or recorded.get("path") != str((parent/"assessment.json").relative_to(root))
            or journal["files"].get("assessment.json") != recorded.get("sha256")
            or file_hash(parent/"assessment.json") != recorded.get("sha256")
        ):
            raise ValueError("Observation import has no verified assessment for this attempt")
        assessed = read_json(parent/"assessment.json", limit=BLOB_LIMIT)
        return parent/"vision", assessed["result_sha256"]

    def precision_reobserve(state, path, attempt):
        from .precision_contract import read_checkpoint_assessment

        pending = state["precision_pending_assessment"]
        checkpoint = pending["checkpoint"]
        source = read_checkpoint(data_dir, checkpoint)
        pairs = pending.get("pairs", [])
        near_object = pending.get("near_object")
        observation_run = ((state.get("precision_seed") or {}).get("observation_run_id")
                           if state["precision_attempt_dirs"] == [None] else None)
        try:
            imported = (previous_observation(state, observation_run, checkpoint, near_object=near_object,
                                             pair_keys=[f"{target}:{neighbour}" for target,neighbour in pairs])
                        if observation_run else None)
            output, trusted_hash = imported or (path/"vision", None)
            if imported is None:
                observation_run = None
            if imported is None and not output.exists():
                command = [str(PROJECT/".venv-mujoco/bin/python"), "-m", "simulation.michelson_reobserve",
                           "--resume-dir", str(source), "--output", str(output)]
                if pairs:
                    command.extend(("--object", pairs[0][0], "--close-views"))
                    for target, neighbour in pairs:
                        command.extend(("--pair", f"{target}:{neighbour}"))
                if near_object:
                    command.extend(("--object", near_object, "--near-views"))
                env = {**os.environ, "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"}
                with (path/"vision.log").open("wb") as log:
                    subprocess.run(command, cwd=PROJECT/"robot-arm-control", env=env,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)
            options = {}
            if pairs:
                options["close_pairs"] = pairs
            if near_object:
                options["near_object"] = near_object
            if trusted_hash is not None:
                options["trusted_report_sha256"] = trusted_hash
            prob = read_checkpoint_assessment(PROJECT, output, source, checkpoint["files"], **options)
            validate_precision_observation(state, prob, partial=True)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            raise StageError("PRECISION_REOBSERVATION_REJECTED", str(exc)) from exc
        raw = encode(prob)
        record = path/"assessment.json"
        atomic(record, raw)
        return {"precision_pending_assessment":None, "precision_assessment":{
            "path":str(record.relative_to(directory(state))), "sha256":digest(raw),
            "checkpoint":checkpoint, "findings":prob["findings"], "excluded":prob.get("excluded", {}),
            "observation_run_id":observation_run, "close_pairs":prob.get("close_pairs", {}),
            **({"near_object":near_object} if near_object else {})}}

    def precision_repair(state, path, attempt):
        return planned_precision_sim(state, path, repair=True)

    def precision_sim(state, path, attempt):
        if same_table(state):
            return planned_precision_sim(state, path)
        if state.get("selected_experiment") == "michelson_full" and state.get("precision_actions"):
            raise StageError("PRECISION_CHECKPOINT_REQUIRED", "The historical workflow has no current-bench contract. A full-bench rearrangement was blocked")
        if state["intent"]["requirements"]:
            raise StageError("PRECISION_DESIGN_CONSTRAINTS", "Precision robot simulation uses the registered design and acceptance thresholds")
        output = path / "simulation"
        if output.exists():
            raise StageError("PRECISION_RUN_EXISTS", "The simulation directory already has a run record. See the original result")
        source = state["scene"]["registration"]["source_sha256"]
        if state.get("precision_kind") == "mzi_contact":
            try:
                result = run_mzi_contact(PROJECT, output, source)
            except subprocess.TimeoutExpired as exc:
                raise StageError("PRECISION_TIMEOUT", "MZI simulation or independent acceptance timed out") from exc
        else:
            manifest = legacy_manifest(PROJECT)
            if manifest["source_sha256"] != source:
                raise StageError("PRECISION_LAYOUT_MISMATCH", "The arm simulation does not match the registered design source")
            atomic(path/"precision-manifest.json", encode(manifest))
            actions = state.get("precision_actions", [])
            if actions:
                kwargs = {"guard_parts":state["precision_guard_parts"]} if state.get("precision_guard_parts") else {}
                result = run_legacy(PROJECT, output, timeout_s=7200, added_noise=False, recovery_actions=actions, **kwargs)
            else:
                result = run_legacy(PROJECT, output, timeout_s=5400, added_noise=False)
        atomic(path/"precision-result.json", encode(result))
        return {"precision_result":result}

    def scoped_repair_evidence(state, result):
        actions = state.get("precision_actions", [])
        prior = [*state.get("precision_prior_repairs", []), *state.get("precision_repairs", [])]
        failure = str(result.get("failure", ""))
        prefix = "RuntimeError: FINAL_VISUAL_LAYOUT_CHECK_FAILED:"
        if same_table(state) and failure.startswith(prefix):
            part = failure.removeprefix(prefix)
            if part not in EXPECTED_NAMES:
                return None, None
            recovery = result.get("recovery") or {}
            repaired = recovery.get("target_object_ids", [])
            attempts = recovery.get("attempts", [])
            if (result.get("bindings_verified") is True and (result.get("plan_hash") or result.get("imported_plan_hash"))
                    and result.get("added_sensor_noise") == {"rgb_std_8bit":0., "optical_std":0.}
                    and recovery.get("retained_scene") is True and len(repaired) == len(attempts) == 1
                    and repaired[0] in EXPECTED_NAMES and part != repaired[0]
                    and recovery.get("commanded_object_ids") == repaired
                    and attempts[0].get("name") == repaired[0]
                    and attempts[0].get("after", {}).get("valid") is True
                    and 0. <= attempts[0].get("position_residual_m", math.inf) <= .001
                    and 0. <= attempts[0].get("rotation_residual_rad", math.inf) <= math.radians(1.)):
                assessment = state.get("precision_assessment")
                if assessment:
                    tried = {name for row in prior for name in row.get("target_object_ids", [])}
                    affected = [(name, row) for name, row in assessment["findings"].items()
                                if name in EXPECTED_NAMES and name not in tried and name not in repaired
                                and row["passed"] is False]
                    if not affected:
                        if assessment.get("near_object"):
                            name = assessment["near_object"]
                            near = assessment["findings"].get(name, {})
                            if (name in EXPECTED_NAMES and name not in tried and name not in repaired
                                    and near.get("passed") is True and .0009 <= near.get("position_m", 0.) <= .001):
                                return "precision_metrology_agent", {"failure_code":"POST_REPAIR_MECHANICAL_REVIEW",
                                    "target_object_ids":[name], "guard_parts":[name],
                                    "allowed_actions":["guard_camera_margin"],
                                    "measured_residual":{"position_mm":round(near["position_m"]*1000, 3)},
                                    "actions":{"guard_camera_margin":"Only when independent mechanical review confirms that this component is out of tolerance, "
                                                                     "localize it with the camera from the same original bench and tighten the single-component correction margin"}}
                            return None, None
                        borderline = [(name, row["position_m"]) for name, row in assessment["findings"].items()
                                      if name in EXPECTED_NAMES and name not in tried and name not in repaired
                                      and row["passed"] is True and .0009 <= row["position_m"] <= .001]
                        if borderline:
                            name, residual = max(borderline, key=lambda item:(item[1],item[0]))
                            return "precision_vision_agent", {"failure_code":"POST_REPAIR_NEAR_REVIEW",
                                "target_object_ids":[name], "allowed_actions":["reobserve_checkpoint"],
                                "measured_residual":{"position_mm":round(residual*1000, 3)},
                                "actions":{"reobserve_checkpoint":"Recheck only this component from a closer wrist-camera view. Grasp permission is zero"}}
                        return None, None
                    name, row = max(affected, key=lambda item:(max(item[1]["position_m"]*1000, item[1]["rotation_deg"]), item[0]))
                    action = "refine_repair_seating" if assessment.get("near_object") == name else "reseat_drifted_parts"
                    return "precision_motion_agent", {"failure_code":"OBSERVED_POST_REPAIR_DRIFT",
                        "target_object_ids":[name], "allowed_actions":[action],
                        "actions":{action:"Keep the single-component placement just completed. Correct only the other component that the latest bench camera shows is out of tolerance and that has not been tried yet"},
                        "measured_residual":{"position_mm":round(row["position_m"]*1000, 3),
                                             "rotation_deg":round(row["rotation_deg"], 3)}}
                return "precision_vision_agent", {"failure_code":"POST_REPAIR_LAYOUT_REVIEW",
                    "target_object_ids":repaired, "allowed_actions":["reobserve_checkpoint"],
                    "actions":{"reobserve_checkpoint":"Recheck the whole bench at a new checkpoint. Keep placed components. Allow only open-gripper camera observation."}}
            if not any(row["action"] == "reseat_drifted_parts" and part in row.get("target_object_ids", [])
                       for row in prior):
                actions = [action for action in actions if action != "reseat_drifted_parts"]
        failed_repair = failed_single_repair(result)
        if same_table(state) and failed_repair:
            code, part = failed_repair
            if (result.get("bindings_verified") is not True
                    or result.get("added_sensor_noise") != {"rgb_std_8bit":0., "optical_std":0.}
                    or not (result.get("plan_hash") or result.get("imported_plan_hash"))):
                return None, None
            if code == "PACKED_CARRIER_LIFT_UNPROVEN":
                assessment = state.get("precision_assessment") or {}
                if assessment:
                    finding = assessment["findings"].get(part, {})
                    pairs = assessment.get("close_pairs", {})
                    action = "replan_supported_turn_lift"
                    if (finding.get("passed") is False and finding.get("rotation_deg", 0.) > 1.
                            and pairs and all(pair.startswith(part+":") and row["wrist_views"] > 0
                                              for pair, row in pairs.items())
                            and not any(row["action"] == action and part in row.get("target_object_ids", [])
                                        for row in prior)):
                        return "precision_motion_agent", {"failure_code":"SUPPORTED_TURN_LIFT_CANDIDATE",
                            "target_object_ids":[part], "allowed_actions":[action], "close_pairs":pairs,
                            "measured_rotation_deg":finding["rotation_deg"],
                            "actions":{action:"For the single component whose angle is out of tolerance, first rehearse the supported turn and lift on a MuJoCo copy that keeps the rest of the bench. "
                                              "Stop if the rehearsal trips contact protection. Only after it passes, rerun the single-component correction and do independent acceptance."}}
                    return None, None
                if any(row.get("failure_code") == code and
                        row.get("blocked_object_id") == part for row in prior):
                    return None, None
                return "precision_vision_agent", {"failure_code":code, "target_object_ids":[part],
                    "allowed_actions":["reobserve_checkpoint"],
                    "actions":{"reobserve_checkpoint":"From the validated open-gripper bench, add a same-frame close view of the target and its neighbors. Authorize camera observation only. The recheck cannot move anything."}}
            if code == "PACKED_CARRIER_SEATING_CONTACT":
                assessment = state.get("precision_assessment")
                if assessment:
                    affected = [(name, row) for name,row in assessment["findings"].items()
                                if name != part and name in EXPECTED_NAMES and row["passed"] is False]
                    if not affected:
                        return None, None
                    name, row = max(affected, key=lambda item:(max(item[1]["position_m"]*1000, item[1]["rotation_deg"]), item[0]))
                    return "precision_motion_agent", {"failure_code":"OBSERVED_INDEPENDENT_POSE_DRIFT",
                        "target_object_ids":[name], "blocked_object_id":part,
                        "allowed_actions":["reseat_drifted_parts"],
                        "actions":{"reseat_drifted_parts":"Correct only the target that the new image shows is out of tolerance, then recheck the whole bench"},
                        "measured_residual":{"position_mm":round(row["position_m"]*1000, 3),
                                             "rotation_deg":round(row["rotation_deg"], 3)}}
                return "precision_vision_agent", {"failure_code":code, "target_object_ids":[part],
                    "allowed_actions":["reobserve_checkpoint"],
                    "actions":{"reobserve_checkpoint":"From the validated open-gripper bench before contact, recheck with photos only and do not move anything. Only a component that a new image independently shows is out of tolerance may enter the next round"}}
            if code == "PACKED_CARRIER_LIFT_CONTACT":
                executed_actions = result.get("recovery_actions", [])
                if "replan_corridor_lift" in executed_actions:
                    return None, None
                if executed_actions == ["refine_repair_seating"]:
                    action = "replan_corridor_lift"
                    if any(row["action"] == action and part in row.get("target_object_ids", []) for row in prior):
                        return None, None
                    return "precision_motion_agent", {"failure_code":"PACKED_CORRIDOR_LIFT_CONTACT",
                        "target_object_ids":[part], "allowed_actions":[action],
                        "actions":{action:"Return to the original bench from before this component's action, and rehearse only a small offset lift generated from the camera direction of the neighbor. "
                                          "Execute only a candidate whose full MuJoCo rehearsal has no contact. Placement uses the original strict check."}}
            if code == "REPAIR_VISUAL_CHECK_FAILED" and result.get("recovery_actions") == ["guard_camera_margin"]:
                attempts = (result.get("recovery") or {}).get("attempts", [])
                if len(attempts) == 1 and attempts[0].get("name") == part and attempts[0].get("after", {}).get("valid") is True:
                    action = "replan_visual_compensation"
                    if not any(row["action"] == action and part in row.get("target_object_ids", []) for row in prior):
                        return "precision_motion_agent", {"failure_code":"VISUAL_COMPENSATION_REQUIRED",
                            "target_object_ids":[part], "allowed_actions":[action],
                            "actions":{action:"Keep the original design. Compute a bounded compensation from the last actual placement-camera residual, and rerun only this component"}}
                return None, None
            if code == "REPAIR_VISUAL_CHECK_FAILED" and result.get("recovery_actions") == ["replan_visual_compensation"]:
                return None, None
            action = "replan_packed_lift" if code == "PACKED_CARRIER_LIFT_CONTACT" else "refine_repair_seating"
            if any(row["action"] == action and part in row.get("target_object_ids", []) for row in prior):
                return None, None
            evidence = {
                "failure_code":code,
                "target_object_ids":[part],
                "allowed_actions":[action],
                "actions":{action: ("Observe again from the validated bench before this component was lifted, and check the single-component departure path. Stop if no path is available"
                                    if action == "replan_packed_lift" else
                                    "Observe again from the validated bench before this component's action, and search for a single-component placement with neighbor clearance inside the original tolerance. Stop if none is available")},
            }
            if action == "refine_repair_seating":
                attempts = (result.get("recovery") or {}).get("attempts", [])
                if len(attempts) != 1 or attempts[0].get("name") != part:
                    return None, None
                row = attempts[0]
                position, angle = row.get("position_residual_m"), row.get("rotation_residual_rad")
                if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (position, angle)):
                    return None, None
                evidence["measured_residual"] = {"position_mm":round(position*1000, 3),
                                                 "rotation_deg":round(math.degrees(angle), 3)}
            return "precision_motion_agent", evidence
        target, evidence = repair_evidence(result, actions)
        if same_table(state) and evidence:
            if failure.startswith(prefix):
                evidence["target_object_ids"] = [failure.removeprefix(prefix)]
            elif len(evidence.get("guard_parts", [])) == 1:
                evidence["target_object_ids"] = evidence["guard_parts"]
            supported = {"reseat_drifted_parts", "guard_camera_margin", "extend_alignment_search"}
            if not set(evidence["allowed_actions"]) <= supported:
                return None, None
        return target, evidence

    def precision_output(state, path, attempt):
        result = state["precision_result"]
        verification = result.get("verification") or {}
        if same_table(state):
            root = directory(state)
            raw = read_bytes(root/state["plan_path"])
            archive = read_bytes(root/state["artifact_path"], limit=BLOB_LIMIT)
            if digest(raw) != state["plan_sha256"] or digest(archive) != state["artifact_sha256"]:
                raise StageError("PRECISION_ARTIFACT_CHANGED", "This round's ROSi or ZIP has changed")
            verify_evidence(archive, raw, result)
            if result.get("success") and result.get("plan_hash") != digest(raw):
                raise StageError("PRECISION_PLAN_UNBOUND", "The result is not bound to this round's ROSi")
        if (not result.get("success") or not result.get("bindings_verified")
                or result.get("added_sensor_noise") != {"rgb_std_8bit":0., "optical_std":0.}
                or not verification.get("mechanical", {}).get("passed")
                or not verification.get("optical", {}).get("passed")
                or not verification.get("accepted")):
            target, evidence = (None, None)
            if state.get("selected_experiment") == "michelson_full":
                target, evidence = scoped_repair_evidence(state, result)
            close = (state.get("precision_assessment") or {}).get("close_pairs", {})
            if not target and close and (failed_single_repair(result) or (None,))[0] == "PACKED_CARRIER_LIFT_UNPROVEN":
                raise StageError("PRECISION_LIFT_UNPROVEN", "Neighbor views were added, but the evidence still does not prove lift clearance",
                                 details={"close_pairs":close, "authorized_object_ids":[]})
            raise StageError("PRECISION_ACCEPTANCE_FAILED", result.get("failure") or "Combined mechanical and optical acceptance did not pass",
                             target=target, details=evidence or verification.get("failure_reasons"))
        report = {"engine":"mujoco", "robot":"xArm7", "scope":"rgb_robot_and_lightpipes",
                  "mechanical":verification["mechanical"], "optical":verification["optical"],
                  "source_sha256":result["source_sha256"], "trajectory_sha256":result["trajectory_sha256"]}
        if state.get("playback"):
            if same_table(state):
                output = directory(state) / state["precision_attempt_dirs"][-1]
            else:
                output = directory(state) / "team" / f'precision_sim-{state["attempts"]["precision_sim"]:02d}' / "simulation"
            try:
                if state.get("precision_kind") == "mzi_contact":
                    report["video_sha256"] = render_precision_replay(PROJECT, output, experiment="mach_zehnder")
                else:
                    report["video_sha256"] = render_precision_replay(PROJECT, output)
            except (subprocess.SubprocessError, OSError):
                report["video_error"] = "The MuJoCo state record was saved, but web video generation failed"
        if same_table(state):
            report.update(plan_hash=state["plan_sha256"], artifact_sha256=state["artifact_sha256"],
                          target_layout_sha256=state["plan"]["target_layout_sha256"],
                          observed_layout_sha256=state["plan"]["observed_layout_sha256"],
                          recovery=result.get("recovery"))
        atomic(path/"verification.json", encode(report))
        return {"verification":report, "result":report}

    def repair_precision(agent, state, path, attempt):
        result = state["precision_result"]
        actions = state.get("precision_actions", [])
        target, evidence = scoped_repair_evidence(state, result)
        if target != agent:
            raise StageError("PRECISION_REPAIR_EVIDENCE_REJECTED", "The fault evidence does not authorize a correction at this node")
        if same_table(state):
            failed_attempt = len(state["precision_attempt_dirs"])
            failed_path = (directory(state)/state["artifact_path"]).with_name("precision-result.json")
        else:
            failed_attempt = state["attempts"]["precision_sim"]
            failed_path = directory(state)/"team"/f"precision_sim-{failed_attempt:02d}"/"precision-result.json"
        raw = read_bytes(failed_path, limit=BLOB_LIMIT)
        if decode_json(raw.decode("utf-8")) != result:
            raise StageError("PRECISION_REPAIR_EVIDENCE_CHANGED", "The failure report does not match the run record")
        atomic(path/"evidence.json", encode({**evidence, "failed_attempt":failed_attempt,
                                             "result_sha256":digest(raw)}))
        public_evidence = {key:value for key,value in evidence.items() if key != "guard_parts"}
        if same_table(state) and evidence["failure_code"] == "FINAL_VISUAL_LAYOUT_CHECK_FAILED":
            part = str(result.get("failure", "")).rsplit(":", 1)[-1]
            if part not in EXPECTED_NAMES:
                raise StageError("PRECISION_TARGET_UNBOUND", "The failure evidence does not identify exactly one registered component")
            public_evidence["target_object_ids"] = [part]
        if not same_table(state):
            raise StageError("PRECISION_CHECKPOINT_REQUIRED", "The historical workflow has no current-bench contract. A full-bench rearrangement was blocked")
        relative = state["precision_attempt_dirs"][-1]
        try:
            checkpoint = ({"run_id":state["run_id"],"directory":relative,
                           "files":checkpoint_files(directory(state)/relative)} if relative else state["precision_origin"])
            source = read_checkpoint(data_dir, checkpoint)
            failed_checkpoint = checkpoint
            rollback = failed_single_repair(result) is not None
            if rollback:
                checkpoint = pre_repair_checkpoint(source.parent, result)
                source = read_checkpoint(data_dir, checkpoint)
            report = read_json(source / "result.json", limit=8_000_000)
            assessment = None
            if evidence["failure_code"] == "POST_REPAIR_NEAR_REVIEW":
                wide = verified_assessment(state, checkpoint)
                candidate = wide["findings"].get(evidence["target_object_ids"][0], {})
                if candidate.get("passed") is not True or not .0009 <= candidate.get("position_m", 0.) <= .001:
                    raise ValueError("Near-view target lacks bound borderline camera evidence")
            if evidence["failure_code"] == "POST_REPAIR_MECHANICAL_REVIEW":
                from .precision_contract import checkpoint_mechanical_outlier
                assessment = verified_assessment(state, checkpoint)
                part = evidence["target_object_ids"][0]
                finding = assessment["findings"].get(part, {})
                if (assessment.get("near_object") != part or part not in assessment["observations"]
                        or finding.get("passed") is not True or not .0009 <= finding.get("position_m", 0.) <= .001):
                    raise ValueError("Mechanical review lacks a bound near-range observation")
                mechanical = checkpoint_mechanical_outlier(PROJECT, source, part)
                read_checkpoint(data_dir, checkpoint)
                atomic(path/"mechanical-review.json", encode({
                    "object_id":part, "checkpoint":checkpoint, "residual":mechanical}))
                public_evidence["independent_mechanical_residual"] = mechanical
            if evidence["failure_code"] in {"OBSERVED_INDEPENDENT_POSE_DRIFT", "SUPPORTED_TURN_LIFT_CANDIDATE",
                                           "OBSERVED_POST_REPAIR_DRIFT"}:
                assessment = verified_assessment(state, checkpoint)
                part = evidence["target_object_ids"][0]
                if (assessment["findings"] != state["precision_assessment"]["findings"]
                        or assessment["findings"].get(part, {}).get("passed") is not False
                        or part not in assessment["observations"]):
                    raise ValueError("Independent drift is not supported by the bound RGB assessment")
                if (evidence["failure_code"] == "SUPPORTED_TURN_LIFT_CANDIDATE"
                        and assessment.get("close_pairs") != evidence["close_pairs"]):
                    raise ValueError("Supported turn is not bound to the close camera assessment")
                if (evidence["failure_code"] == "OBSERVED_POST_REPAIR_DRIFT"
                        and state["precision_assessment"].get("near_object") != assessment.get("near_object")):
                    raise ValueError("Repair proposal differs from the verified camera range")
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            raise StageError("PRECISION_CHECKPOINT_UNBOUND", str(exc)) from exc
        if (report.get("trajectory_sha256") != checkpoint["files"]["trajectory.npz"]
                or report.get("world_model_sha256") != checkpoint["files"]["world_model.mjb"]
                or (not rollback and report.get("failure") not in (None, result.get("failure")))):
            raise StageError("PRECISION_CHECKPOINT_UNBOUND", "The restored scene does not match the failure evidence")
        if set(report.get("objects", {})) != EXPECTED_NAMES or not all(
                obj.get("visual_quality_passed") is True for obj in report["objects"].values()):
            raise StageError("PRECISION_CHECKPOINT_UNSAFE", "The current scene has no complete assembly and safety-handoff record")
        proposed = PrecisionRepair.model_validate(gateway(state, path).repair_precision(public_evidence, attempt)).model_dump()
        action = proposed["action"]
        decision = {**proposed, "agent":agent, "provider":state["provider"], "failed_attempt":failed_attempt,
                    "failure_code":evidence["failure_code"], "result_sha256":digest(raw)}
        if "independent_mechanical_residual" in public_evidence:
            decision["independent_mechanical_residual"] = public_evidence["independent_mechanical_residual"]
        guard_parts = evidence.get("guard_parts", []) if action == "guard_camera_margin" else []
        if guard_parts:
            if not set(guard_parts) <= moving_ids(state["scene"]):
                raise StageError("PRECISION_REPAIR_REJECTED", "The reviewed component is not registered")
            decision["guard_parts"] = guard_parts
        if action == "stop":
            atomic(path/"decision.json", encode(decision))
            raise StageError("PRECISION_AGENT_STOP", proposed["diagnosis"])
        if action not in evidence["allowed_actions"]:
            atomic(path/"decision.json", encode(decision))
            raise StageError("PRECISION_REPAIR_REJECTED", "The node selected a correction that the fault evidence does not support")
        receipt = path / f"model-precision_repair-{attempt}.json"
        if receipt.is_file():
            decision["model"] = read_json(receipt, limit=None)["model"]
        if action == "reobserve_checkpoint":
            decision["target_object_ids"] = []
            if evidence["failure_code"] == "POST_REPAIR_LAYOUT_REVIEW":
                decision["completed_object_id"] = evidence["target_object_ids"][0]
            elif evidence["failure_code"] == "POST_REPAIR_NEAR_REVIEW":
                decision["observation_object_id"] = evidence["target_object_ids"][0]
            else:
                decision["blocked_object_id"] = evidence["target_object_ids"][0]
            pending = {"checkpoint":checkpoint}
            if evidence["failure_code"] == "POST_REPAIR_NEAR_REVIEW":
                pending["near_object"] = decision["observation_object_id"]
            if evidence["failure_code"] == "PACKED_CARRIER_LIFT_UNPROVEN":
                part = decision["blocked_object_id"]
                positions = {row["id"]:row["world_position_m"][:2] for row in legacy_manifest(PROJECT)["targets"]}
                neighbours = sorted((n for n in positions if n != part and math.dist(positions[part], positions[n]) <= .09),
                                    key=lambda n:(math.dist(positions[part], positions[n]),n))
                if not neighbours:
                    raise StageError("PRECISION_OBSERVATION_REJECTED", "No registered neighbor requiring a same-frame recheck was found")
                pending["pairs"] = [[part,n] for n in neighbours]
                decision["observation_pairs"] = pending["pairs"]
            atomic(path/"decision.json", encode(decision))
            return {"precision_pending_assessment":pending, "precision_repairs":[decision]}
        objects = guard_parts
        if action in ("reseat_drifted_parts", "replan_packed_lift", "refine_repair_seating",
                      "replan_supported_turn_lift", "replan_corridor_lift", "replan_visual_compensation"):
            objects = evidence["target_object_ids"]
        if action == "extend_alignment_search":
            objects = []
        elif len(objects) != 1 or objects[0] not in EXPECTED_NAMES:
            raise StageError("PRECISION_TARGET_UNBOUND", "The evidence does not uniquely identify a component that can be corrected safely")
        decision["target_object_ids"] = objects
        pending = {"action":action,"target_object_ids":objects,"checkpoint":checkpoint}
        if assessment:
            pending["assessment_sha256"] = state["precision_assessment"]["sha256"]
            decision["assessment_sha256"] = pending["assessment_sha256"]
        elif rollback and action in {"replan_packed_lift", "refine_repair_seating", "replan_corridor_lift",
                                     "replan_visual_compensation"}:
            pending["rollback_observation"] = {"run_id":failed_checkpoint["run_id"],
                "directory":failed_checkpoint["directory"],
                "plan_sha256":result.get("plan_hash") or result["imported_plan_hash"]}
        if action == "replan_visual_compensation":
            from .precision_contract import _valid_transform
            row = result["recovery"]["attempts"][0]
            command, measured = row.get("command"), row["after"].get("T_world_object")
            if not _valid_transform(command) or not _valid_transform(measured):
                raise StageError("PRECISION_REPAIR_REJECTED", "Compensation requires a valid, bound command and the actual camera placement")
            pending["placement_feedback"] = {"object_id":objects[0], "command":command,
                "measured_pose":measured, "failure_sha256":digest(raw)}
        atomic(path/"decision.json", encode(decision))
        return {"precision_actions":[*actions,action], "precision_repairs":[decision],
                "precision_pending":pending,
                **({"precision_guard_parts":guard_parts} if guard_parts else {}),
                "precision_result":None, "verification":None}

    def act(state, path, attempt):
        feedback = state.get("feedback",[])
        try:
            if is_registered(state["scene"]):
                layout = tightened_tolerance(deepcopy(state["reference_layout"]), state["intent"]["requirements"])
            else:
                layout = gateway(state, path).layout(state["request"], state["scene"], state["intent"],
                                                     [{"initial_prob0":state["prob"]},*feedback], attempt)
        except CandidateFormatError as exc:
            raise StageError("ACT_INVALID_JSON", str(exc), "act") from exc
        if injected(state,"act",attempt):
            layout = {**layout,"scene_revision":"injected-incompatible-revision"}
        atomic(path/"target_layout.json", encode(layout))
        return {"layout":layout}

    def validator(state, path, attempt):
        validate = validate_registered_layout if is_registered(state["scene"]) else validate_layout
        if same_table(state):
            # This registered fixture has intentional zero-gap neighbours. Its frozen target
            # identity is checked here; the xArm7 planner checks actual carried clearances.
            manifest = legacy_manifest(PROJECT)
            passed = (state["layout"] == state["reference_layout"]
                      and not state["intent"]["requirements"]
                      and manifest["source_sha256"] == state["scene"]["registration"]["source_sha256"]
                      and {target["id"] for target in manifest["targets"]} == moving_ids(state["scene"]))
            report = {"passed":passed, "kind":"frozen_xarm7_layout",
                      "issues":[] if passed else [{"code":"V_REGISTERED_ALIGNMENT"}],
                      "source_sha256":manifest["source_sha256"]}
        else:
            report = validate(state["layout"],state["scene"],state["intent"]["requirements"])
        atomic(path/"validation.json",encode(report))
        if not report["passed"]:
            target = "prompt" if any(i["code"] == "V_REQUEST" for i in report["issues"]) else "act"
            raise StageError("VALIDATOR_REJECTED", "Layout or input definitions failed", target, report["issues"])
        return {"validation":report}

    def controller(state, path, attempt):
        if same_table(state):
            try:
                return precision_controller(state, path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                if isinstance(exc, StageError):
                    raise
                raise StageError("PRECISION_CONTROLLER_REJECTED", str(exc), "controller") from exc
        height = .05 if injected(state,"controller",attempt) else min(.16 + .015 * (attempt - 1), .22)
        if is_registered(state["scene"]) and not injected(state,"controller",attempt):
            height = min(.20 + .01 * (attempt - 1), .22)
        hashes = [digest(encode(state[key])) for key in ("prob","layout")]
        try:
            if use_files(state):
                payload = {key: state[key] for key in ("prob", "layout", "scene")}
                reply = services.invoke("controller", path, {**payload, "hashes": hashes, "travel_z_m": height})
                raw = read_bytes(path / "candidate-plan.json")
                if reply != {"plan_sha256": digest(raw)}:
                    raise ValueError("Controller response hash does not match the ROSi file")
                plan = decode_json(raw.decode("utf-8"))
            else:
                plan = build_trajectory(state["prob"], state["layout"], state["scene"], hashes, height)
                raw = encode(plan)
            check_trajectory(plan, moving_ids(state["scene"]))
            if (plan["scene_id"] != state["scene"]["scene_revision"]
                    or [plan["initial_sha256"], plan["layout_sha256"]] != hashes):
                raise ValueError("ROSi does not bind the current scene, PROB0 and layout")
        except (ValueError, OSError, TypeError, KeyError, OverflowError, subprocess.TimeoutExpired) as exc:
            raise StageError("CONTROLLER_REJECTED",str(exc),"controller",{"travel_z_m":height}) from exc
        atomic(path/"plan.json",raw)
        atomic(path/"plan.sha256",(digest(raw)+"\n").encode())
        return {"plan_path":str((path/"plan.json").relative_to(directory(state))),
                "plan_sha256":digest(raw),"verification":None}

    def sim(state, path, attempt):
        raw = read_bytes(directory(state)/state["plan_path"])
        if digest(raw) != state["plan_sha256"]:
            raise StageError("ROSI_CHANGED", "Controller ROSi hash changed")
        atomic(path/"plan.json",raw)
        request = {"scene":state["scene"],"layout":state["layout"],"plan_hash":digest(raw)}
        if state.get("playback"):
            request["playback"] = True
        try:
            timeout = .001 if injected(state, "sim", attempt) else 40
            if use_files(state):
                result = services.invoke("sim", path, request, files=("plan.json",), timeout=timeout)
            else:
                result = worker("sim", path, request, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise StageError("SIM_TIMEOUT", "Simulator worker exceeded its deadline", "sim") from exc
        except (ValueError,OSError) as exc:
            raise StageError("SIM_WORKER_FAILED",str(exc),"sim") from exc
        if not isinstance(result, dict) or result.get("status") not in ("succeeded", "failed"):
            raise StageError("SIM_OUTPUT_INVALID", "Simulator response status rejected", "sim")
        if result["status"] == "failed":
            failure = result.get("error")
            if (not isinstance(failure, dict) or not isinstance(failure.get("code"), str)
                    or not isinstance(failure.get("message"), str)):
                raise StageError("SIM_OUTPUT_INVALID", "Simulator error record rejected", "sim")
            target = "controller" if failure["code"] == "UNEXPECTED_CONTACT" else None
            raise StageError("SIM_REJECTED",failure["message"],target,failure)
        try:
            recording_hash = None
            if state.get("playback"):
                recording = read_bytes(path / "playback.json")
                validate_playback(recording, decode_json(raw.decode("utf-8")), digest(raw), state["scene"], result)
                recording_hash = digest(recording)
            archive = make_archive(raw,result,corrupt_hash=injected(state,"output",state.get("attempts",{}).get("output",0)+1))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise StageError("SIM_OUTPUT_INVALID", str(exc), "sim") from exc
        atomic(path/"artifact.zip",archive)
        return {"artifact_path":str((path/"artifact.zip").relative_to(directory(state))),
                "artifact_sha256":digest(archive), "playback_sha256":recording_hash, "verification":None}

    def output(state, path, attempt):
        root = directory(state)
        raw = read_bytes(root/state["plan_path"])
        if digest(raw) != state["plan_sha256"]:
            raise StageError("ROSI_CHANGED","Plan changed after simulation")
        try:
            artifact = read_bytes(root/state["artifact_path"], limit=BLOB_LIMIT)
            if state.get("artifact_sha256") and digest(artifact) != state["artifact_sha256"]:
                raise ValueError("ZIP bytes changed after simulation")
            manifest,result = read_archive(artifact,raw)
            if state.get("playback_sha256"):
                recording = read_bytes((root / state["artifact_path"]).with_name("playback.json"))
                if digest(recording) != state["playback_sha256"]:
                    raise ValueError("Playback bytes changed after simulation")
        except (ValueError,KeyError,TypeError,BadZipFile,OSError) as exc:
            raise StageError("OUTPUT_ARTIFACT_REJECTED",str(exc),"sim") from exc
        try:
            optics, measurements = evaluate_poses(result,state["layout"],state["scene"],state["intent"]["requirements"])
        except ValueError as exc:
            raise StageError("OUTPUT_POSE_REJECTED",str(exc),"controller") from exc
        if not optics["passed"]:
            raise StageError("OUTPUT_OPTICS_REJECTED","Final optical geometry rejected","act",optics["issues"])
        receipt = {"plan_hash":manifest["plan_hash"],"artifact_sha256":digest(artifact),
                   "engine":manifest["engine"],"sim_version":manifest["sim_version"],
                   "measurements":measurements,"planar_optical_recheck":optics}
        if state.get("selected_experiment") == "mach_zehnder":
            optical_request = {"actual_poses":result["poses"], "objects":state["scene"]["objects"],
                               "plan_sha256":manifest["plan_hash"], "simulation_result_sha256":manifest["result_hash"],
                               "source_sha256":state["scene"]["registration"]["source_sha256"]}
            optical_input = encode(optical_request)
            atomic(path/"mzi-optical-input.json", optical_input)
            try:
                subprocess.run([str(PROJECT/".venv-lightpipes/bin/python"), "-m", "optics_graph.mzi_optical_check",
                                str(path.resolve())], cwd=ROOT, timeout=150, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                wave = read_json(path/"mzi-optical-report.json")
                expected = {"input_sha256":digest(optical_input),
                            "plan_sha256":manifest["plan_hash"], "simulation_result_sha256":manifest["result_hash"],
                            "source_sha256":optical_request["source_sha256"],
                            "actual_sha256":digest(read_bytes(path/"mzi-actual-optics.json"))}
                if any(wave.get(key) != value for key, value in expected.items()):
                    raise ValueError("The LightPipes report is not bound to this round's input, trajectory, or placement record")
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise StageError("OPTICAL_SOLVER_FAILED", "LightPipes did not return an acceptable report", details=str(exc)) from exc
            if wave.get("passed") is not True:
                failed = [name for name, passed in wave.get("checks", {}).items() if not passed]
                raise StageError("MZI_OPTICAL_REJECTED", "The measured placement did not pass Mach-Zehnder optical acceptance",
                                 details={"failed_checks":failed, "failure":wave.get("failure"), "wave":wave.get("wave")})
            receipt["wave_optics"] = wave
        if state.get("playback_sha256"):
            receipt["playback_sha256"] = state["playback_sha256"]
        atomic(path/"verification.json",encode(receipt))
        return {"verification":receipt}

    def execute(state, path, attempt):
        root = directory(state)
        raw = read_bytes(root/state["plan_path"])
        artifact = read_bytes(root/state["artifact_path"],limit=BLOB_LIMIT)
        if injected(state,"execution",attempt):
            raw += b" "
        if state["require_review"] and state.get("approval",{}).get("plan_hash") != digest(raw):
            raise StageError("EXECUTION_APPROVAL_MISMATCH","Approved ROSi does not match L1 input")
        try:
            object_ids = moving_ids(state["scene"])
            payload = {"verification":state["verification"], "object_ids":sorted(object_ids)}
            if same_table(state):
                payload["plan_schema"] = PRECISION_PLAN_SCHEMA
                receipt = accept_precision_l1(raw, artifact, state["verification"])
            else:
                receipt = accept_l1(raw, artifact, state["verification"], object_ids)
            if use_files(state):
                atomic(path / "plan.json", raw)
                atomic(path / "artifact.zip", artifact)
                reply = services.invoke("execution", path, payload, files=("plan.json", "artifact.zip"))
                if encode(reply) != encode(receipt):
                    raise ValueError("Dry-run executor receipt does not match the verified handoff")
        except (ValueError, BadZipFile, OSError, subprocess.TimeoutExpired) as exc:
            raise StageError("EXECUTION_REJECTED",str(exc)) from exc
        atomic(path/"execution_receipt.json",encode(receipt))
        return {"result":receipt}

    def stage(name, operation, state):
        attempt = state.get("attempts",{}).get(name,0)+1
        path = directory(state)/"team"/f"{name}-{attempt:02d}"
        path.mkdir(parents=True,exist_ok=True)
        event = {"stage":name,"attempt":attempt,"idempotency_key":f"{state['run_id']}:{name}:{attempt}"}
        update = {"stage":name,"attempts":{**state.get("attempts",{}),name:attempt}}
        completion = path / "completion.json"
        # A completed local attempt may precede the SQLite checkpoint when a process dies.
        # Replay its recorded update, never silently run it again with changed inputs/files.
        input_hash = digest(encode({k:v for k,v in state.items() if k not in ("events", "status")}, limit=None))
        synthetic = state.get("simulation_backend") == "algorithm_v1"
        reusable = name != "execution" or synthetic
        try:
            if algorithm != synthetic:
                raise StageError("SIMULATION_BACKEND_MISMATCH", "The run backend does not match the compiled workflow. No service was called")
            if completion.exists() and reusable:
                saved = read_json(completion, limit=None)
                if (not isinstance(saved, dict)
                        or set(saved) != {"version", "input_sha256", "files", "update", "update_sha256"}
                        or not isinstance(saved["files"], dict) or not isinstance(saved["update"], dict)
                        or type(saved["version"]) is not int or saved["version"] != 1
                        or digest(encode(saved["update"], limit=None)) != saved["update_sha256"]
                        or saved["input_sha256"] != input_hash
                        or saved["update"].get("stage") != name
                        or saved["update"].get("attempts") != update["attempts"]
                        or not isinstance(saved["update"].get("events"), list)
                        or len(saved["update"]["events"]) != 1
                        or saved["update"]["events"][0]["idempotency_key"] != event["idempotency_key"]
                        or any(Path(file).name != file or digest(read_journal_artifact(path/file, limit=BLOB_LIMIT)) != sha
                               for file,sha in saved["files"].items())):
                    raise StageError("ATTEMPT_RECORD_CHANGED", "Completed attempt input or artifacts changed")
                if name == "execution" and synthetic:
                    from .algorithm_sim import AlgorithmServices
                    AlgorithmServices(directory(state), StageError).verify_archive(state)
                return saved["update"]
            if synthetic and name == "execution":
                marker = path/"execution-started.json"
                if marker.exists():
                    raise StageError("ALGORITHM_EXECUTION_UNKNOWN", "Simulated acceptance started but has no complete receipt. Keep the evidence and stop. Do not retry automatically")
                atomic(marker, encode({"plan_sha256":state["plan_sha256"], "input_sha256":input_hash}))
            if state.get("simulation_backend") == "algorithm_v1":
                from .algorithm_sim import AlgorithmServices
                previous_failure = (state.get("feedback") or [{}])[-1]
                if (previous_failure.get("code") == "MODEL_SERVICE_UNAVAILABLE"
                        and previous_failure.get("requested_return") == name):
                    retry_after = (previous_failure.get("details") or {}).get("retry_after_s", 0)
                    if type(retry_after) in (int, float) and math.isfinite(retry_after) and retry_after > 0:
                        event["provider_retry_wait_s"] = retry_after
                        time.sleep(retry_after)
                time.sleep(state.get("algorithm_options", {}).get("delay_s", .65))
                summary = {"observed_revision":(state.get("algorithm_state") or {}).get("revision"),
                           "current_layout_sha256":digest(encode((state.get("algorithm_state") or {}).get("current"))),
                           "target_layout_sha256":digest(encode(state.get("layout"))),
                           "allowed_object_ids":(state.get("plan") or {}).get("allowed_object_ids", []),
                           "feedback":state.get("feedback", [])[-1:]}
                atomic(path/"input.json", encode({"source":"algorithm_v1", "node":name,
                    "attempt":attempt, "summary":summary, "request":state["request"],
                    "current":(state.get("algorithm_state") or {}).get("current"),
                    "target":state.get("layout"), "plan_sha256":state.get("plan_sha256")}, limit=None))
                event["input_summary"] = summary
                if name in {"mode","done"}:
                    update.update(operation(state,path,attempt))
                else:
                    from .agent_algorithm import AgentAlgorithmServices
                    from .algorithm_sim import MODEL_PROVIDERS
                    adapter = (AgentAlgorithmServices(directory(state),StageError,gateway_factory)
                               if state.get("provider") in MODEL_PROVIDERS else AlgorithmServices(directory(state),StageError))
                    update.update(adapter.invoke(name,state,path,attempt))
                event["output_summary"] = {"source":"algorithm_v1",
                    "world_revision":update.get("algorithm_state", state.get("algorithm_state", {})).get("revision"),
                    "moved_ids":(update.get("algorithm_result") or {}).get("moved_ids", []),
                    "checks":(update.get("verification") or update.get("result") or {}).get("checks"),
                    "plan_sha256":update.get("plan_sha256", state.get("plan_sha256"))}
            else:
                update.update(operation(state,path,attempt))
            event.update(outcome="passed",message=f"{name} passed on attempt {attempt}")
            if name in ("precision_sim", "precision_repair") and same_table(state):
                event["simulation_attempt"] = len(update["precision_attempt_dirs"])
                event["recovery"] = update["precision_result"].get("recovery")
                event["imported_checkpoint"] = update["precision_attempt_dirs"][-1] is None
            if name == "precision_reobserve":
                event["assessment"] = update["precision_assessment"]
            update.update(error=None,return_to="")
        except (ValueError,GatewayError,OSError,KeyError,TypeError,BadZipFile) as exc:
            code = getattr(exc,"code",None) or type(exc).__name__
            target = getattr(exc,"target",None)
            if name in ("input", "prompt", "act", "controller", "validator") and isinstance(exc, (RetryableGatewayError, CandidateFormatError)):
                target = "act" if name == "validator" else name
            error = {"stage":name,"code":str(code),"message":str(exc)[:500],
                     "details":getattr(exc,"details",None),"requested_return":target}
            event.update(outcome="failed",message=error["message"],error=error)
            update.update(error=error,verification=None,result=None,approved=False,approval=None)
            if state.get("simulation_backend") == "algorithm_v1" and name == "validator":
                for field, filename in (("algorithm_test_suite","optical-test-suite.json"),
                                        ("algorithm_test_report","optical-test-report.json")):
                    if (path/filename).exists():
                        update[field] = read_json(path/filename)
        if name == "precision_output" and same_table(state):
            verification = (state.get("precision_result") or {}).get("verification") or {}
            event["simulation_attempt"] = len(state.get("precision_attempt_dirs", []))
            event["checks"] = {key: ("passed" if verification[key]["passed"] else "failed")
                               for key in ("mechanical", "optical") if "passed" in verification.get(key, {})}
        event["time"] = datetime.now(timezone.utc).isoformat()
        if state.get("simulation_backend") == "algorithm_v1":
            if (path/"decision.json").exists():
                event["decision"] = read_json(path/"decision.json", limit=None)
            event["evidence"] = [f.name for f in path.iterdir() if f.is_file() and f.name != "completion.json"]
        # Preserve the original evidence if a resumed attempt's journal is invalid.
        if completion.exists() and reusable:
            return {**update,"events":[event]}
        atomic(path/"event.json",encode(event, limit=None))
        update["events"] = [event]
        files = {file.name:digest(read_journal_artifact(file, limit=BLOB_LIMIT)) for file in path.iterdir()
                 if file.name != "completion.json" and file.is_file()}
        atomic(completion,encode({"version":1,"input_sha256":input_hash,"files":files,
                                  "update":update,"update_sha256":digest(encode(update, limit=None))}, limit=None))
        return update

    def feedback(state):
        error = state["error"]
        target = error["requested_return"]
        counts = dict(state.get("retry_counts",{}))
        details = error.get("details")
        parts = details.get("target_object_ids", []) if isinstance(details, dict) else []
        # Placement repairs of one named part get their own per-node budget (e.g. "act:laser1"), so test-suite
        # rewrites returning to Act cannot use up a misplaced part's repairs. Every key still counts toward
        # TOTAL_RETRIES; no limit is raised.
        placement = (state.get("simulation_backend") == "algorithm_v1" and error["stage"] == "output"
                     and error.get("code") in ALGORITHM_PLACEMENT_CODES)
        retry_key = (f"{target}:{parts[0]}" if target and (same_table(state) or placement) and len(parts) == 1
                     else target)
        close_observation = (retry_key != target and target == "precision_vision_agent"
                             and details.get("failure_code") == "PACKED_CARRIER_LIFT_UNPROVEN")
        if close_observation:
            retry_key += ":close_observation"
        if retry_key != target and details.get("failure_code") == "REPAIR_VISUAL_CHECK_FAILED":
            retry_key += ":seating"
        lift_strategy = ({"SUPPORTED_TURN_LIFT_CANDIDATE":":supported_turn",
                          "PACKED_CORRIDOR_LIFT_CONTACT":":corridor_lift",
                          "VISUAL_COMPENSATION_REQUIRED":":visual_compensation"}.get(details.get("failure_code"))
                         if retry_key != target else None)
        if lift_strategy:
            retry_key += lift_strategy
        review_strategy = ({"POST_REPAIR_LAYOUT_REVIEW":":post_repair", "OBSERVED_POST_REPAIR_DRIFT":":post_repair",
                            "POST_REPAIR_MECHANICAL_REVIEW":":post_repair",
                            "POST_REPAIR_NEAR_REVIEW":":near_review"}.get(details.get("failure_code"))
                           if retry_key != target else None)
        if review_strategy:
            retry_key += review_strategy
        scoped_attempt = close_observation or lift_strategy is not None or review_strategy is not None
        retry_limit = 1 if scoped_attempt else RETRY_LIMIT
        # New bounded evidence permits one use per part without reopening historical retries.
        charged = lambda: sum(value for key,value in counts.items()
                              if not key.endswith((":close_observation", ":supported_turn", ":corridor_lift",
                                                   ":post_repair", ":near_review", ":visual_compensation")))
        reason = ("route_not_allowed" if target not in RETURNS.get(error["stage"],set()) else
                  "target_retry_limit" if counts.get(retry_key,0)>=retry_limit else
                  "total_retry_limit" if not scoped_attempt and charged()>=TOTAL_RETRIES else "retry_allowed")
        allowed = reason == "retry_allowed"
        item = {"from":error["stage"],"to":target if allowed else "stop","code":error["code"],
                "attempt":state["attempts"][error["stage"]],"time":datetime.now(timezone.utc).isoformat(),
                "reason":reason,"retry_limit":retry_limit,"total_limit":TOTAL_RETRIES}
        if allowed:
            counts[retry_key] = counts.get(retry_key,0)+1
        if retry_key != target:
            item["retry_key"] = retry_key
        item["retry_counts"] = counts
        item["total_retries"] = charged()
        atomic(directory(state)/"team"/f"feedback-{len(state.get('retry_log',[]))+1:02d}.json", encode(item, limit=None))
        invalidated = {key:None for key in DOWNSTREAM.get(target, ())} if allowed else {}
        if "artifact_path" in invalidated:
            invalidated["artifact_sha256"] = None
            invalidated["playback_sha256"] = None
        return {**invalidated,"verification":None,"result":None,"approved":False,"approval":None,
                "return_to":target if allowed else "failed","retry_counts":counts,"retry_log":[item],
                "feedback":[*state.get("feedback",[]),error],"stage":"feedback_router",
                "events":[{"stage":"feedback_router","message":f"{error['stage']} → {item['to']}","route":item}]}

    def gate(state):
        decision = interrupt({"kind":"rosi_l1_review","plan_sha256":state["plan_sha256"],
                              "message":"After the acceptance trajectory is checked, the local L1 handoff check is allowed."})
        approved = decision.get("approved") is True and decision.get("plan_sha256")==state["plan_sha256"]
        return {"approved":approved,"stage":"safety_gate",
                "approval":{"plan_hash":state["plan_sha256"],"approved":approved,
                            "time":datetime.now(timezone.utc).isoformat()}}

    def done(state):
        if state.get("simulation_backend") == "algorithm_v1":
            return {"status":"algorithm_passed", "stage":"done",
                    "result":state.get("result") or state["verification"]}
        if precise_experiment(state):
            return {"status":"team_precision_passed","stage":"done"}
        return {"status":"team_l1_passed" if state["mode"]=="github_local" else "team_sim_passed",
                "stage":"done","result":state.get("result") or state["verification"]}

    builder = StateGraph(TeamState)
    operations = {"input":receive,"vision":vision,"prompt":prompt,"act":act,"validator":validator,
                  "controller":controller,"sim":sim,"output":output,"execution":execute,
                  "precision_sim":precision_sim,"precision_repair":precision_repair,"precision_output":precision_output,
                  "precision_reobserve":precision_reobserve}
    operations.update({node:partial(repair_precision,node) for node in AGENT_NODES})
    if algorithm:
        operations = {name:operation for name,operation in operations.items()
                      if name in {"input", *NEXT}}
    for name, operation in operations.items():
        builder.add_node(name,partial(stage,name,operation))
    builder.add_node("feedback_router",feedback)
    builder.add_node("mode", partial(stage,"mode",lambda state,path,attempt:{"stage":"mode"}) if algorithm
                     else lambda state:{"stage":"mode"})
    builder.add_node("safety_gate",gate)
    builder.add_node("done",partial(stage,"done",lambda state,path,attempt:done(state)) if algorithm else done)
    builder.add_node("failed",lambda state:{"stage":"failed","status":"failed"})
    builder.add_node("rejected",lambda state:{"stage":"safety_gate","status":"rejected"})
    if algorithm:
        # The same stage journal and feedback router, with the team's canonical
        # chain. No precision shortcut or physical worker can be reached.
        builder.add_edge(START,"input")
        for origin, success in {"input":"vision", **NEXT}.items():
            failure = "failed" if origin == "execution" else "feedback_router"
            builder.add_conditional_edges(origin, lambda state, yes=success, no=failure: no if state.get("error") else yes,
                                          [success,failure])
        builder.add_conditional_edges("feedback_router", lambda state:state["return_to"],
                                      ["input","prompt","act","controller","sim","failed"])
        builder.add_conditional_edges("mode", lambda state:"done" if state["mode"] == "github_sim" else
                                      "safety_gate" if state["require_review"] else "execution",
                                      ["done","safety_gate","execution"])
        builder.add_conditional_edges("safety_gate", lambda state:"execution" if state["approved"] else "rejected",
                                      ["execution","rejected"])
        for node in ("done","failed","rejected"):
            builder.add_edge(node,END)
        return builder.compile(checkpointer=checkpointer)
    builder.add_edge(START,"input")
    builder.add_conditional_edges("input",lambda state:"feedback_router" if state.get("error") else
                                  "vision" if state["team_entry"]=="images" and (same_table(state) or not precise_experiment(state)) else "prompt",
                                  ["feedback_router","vision","prompt"])
    for origin,success in NEXT.items():
        if origin == "prompt":
            builder.add_conditional_edges(origin, lambda state: "feedback_router" if state.get("error")
                                          else "precision_sim" if precise_experiment(state) and not same_table(state) else "act",
                                          ["feedback_router","precision_sim","act"])
            continue
        if origin == "controller":
            builder.add_conditional_edges(origin, lambda state:
                "feedback_router" if state.get("error") else
                ("precision_repair" if state.get("precision_pending") else "precision_sim") if same_table(state) else "sim",
                ["feedback_router","sim","precision_sim","precision_repair"])
            continue
        failure = "failed" if origin == "execution" else "feedback_router"
        builder.add_conditional_edges(origin,lambda state, yes=success,no=failure: no if state.get("error") else yes,
                                      [success,failure])
    builder.add_conditional_edges("precision_sim", lambda state: "failed" if state.get("error") else "precision_output",
                                  ["failed","precision_output"])
    builder.add_conditional_edges("precision_output", lambda state: "feedback_router" if state.get("error") else
                                  "mode" if same_table(state) and state["mode"] == "github_local" else "done",
                                  ["feedback_router","mode","done"])
    builder.add_conditional_edges("precision_repair", lambda state: "failed" if state.get("error") else "precision_output",
                                  ["failed","precision_output"])
    for node in AGENT_NODES:
        if node == "precision_vision_agent":
            builder.add_conditional_edges(node, lambda state: "failed" if state.get("error") else
                                          "precision_reobserve" if state.get("precision_pending_assessment") else "controller",
                                          ["failed","precision_reobserve","controller"])
        else:
            builder.add_conditional_edges(node, lambda state: "failed" if state.get("error") else "controller",
                                          ["failed","controller"])
    builder.add_conditional_edges("precision_reobserve", lambda state: "failed" if state.get("error") else "precision_output",
                                  ["failed","precision_output"])
    builder.add_conditional_edges("feedback_router",lambda state:state["return_to"],
                                  ["input","prompt","act","controller","sim",*AGENT_NODES,"failed"])
    builder.add_conditional_edges("mode",lambda state:"done" if state["mode"]=="github_sim"
                                  else "safety_gate" if state["require_review"] else "execution",
                                  ["done","safety_gate","execution"])
    builder.add_conditional_edges("safety_gate",lambda state:"execution" if state["approved"] else "rejected",
                                  ["execution","rejected"])
    for node in ("done","failed","rejected"):
        builder.add_edge(node,END)
    return builder.compile(checkpointer=checkpointer)
