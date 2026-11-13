"""The real LangGraph workflow. Backend claims stay distinct from graph success."""
from __future__ import annotations
import json
import math
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from .contracts import load_scene, validate_layout, validate_plan, canonical_hash
from .gateway import CandidateFormatError, ModelGateway, GatewayError
from .robot import execute_plan
from .local_sim import run_local_sim
from .legacy import legacy_manifest, read_legacy_evidence, run_legacy
from .settings import PROJECT, read_settings


class State(TypedDict, total=False):
    run_id: str
    request: str
    provider: str
    profile: str
    mode: str
    sim_scenario: str
    sim_attempts: int
    sim_history: list[dict]
    sim_signatures: list[str]
    command_offset_xy_m: list[float]
    repair_route: str
    repair_history: list[dict]
    require_review: bool
    fail_action: str | None
    status: str
    stage: str
    scene: dict
    intent: dict
    layout: dict
    validation: dict
    plan: dict
    plan_validation: dict
    plan_sha256: str
    layout_attempts: int
    plan_attempts: int
    feedback: list
    format_issue: str | None
    result: dict
    manifest: dict
    approved: bool
    error: dict | None
    events: Annotated[list[dict], add]


def build_graph(data_dir: Path, checkpointer=None, gateway_factory=ModelGateway):
    def event(stage, message, **kwargs):
        return {"stage":stage, "events":[{"stage":stage,"message":message}], **kwargs}

    def guarded(stage, fn):
        def wrapped(state):
            try:
                return fn(state)
            except (GatewayError, ValueError, OSError, KeyError, TypeError) as exc:
                # API exceptions are sanitized inside the gateway. Never dump traceback/config.
                message = str(exc)[:700] if isinstance(exc,(GatewayError,ValueError)) else type(exc).__name__
                return event(stage, message, status="failed", error={"stage":stage,"code":type(exc).__name__,"message":message})
        return wrapped

    def gateway(state):
        return gateway_factory(state["provider"], data_dir / "runs" / state["run_id"])

    def context(state):
        if state["profile"] == "legacy_vision_michelson_15":
            return event("context", "Read the original fixed 15-component design and keep its source and target independent", manifest=legacy_manifest(PROJECT))
        if state["mode"] == "real":
            return event("context", "Hardware is not configured: the example bench is uncalibrated, and the ROSi, driver contract, and local enable have not been accepted", status="blocked",
                         error={"code":"HARDWARE_NOT_CONFIGURED","message":"A formal scene, robot, and calibration, plus the team planner, sim, and arm services and a separate hardware acceptance, are required. This interface cannot start real motion."})
        return event("context", "Load the Phase 1 public example scene. Do not read object ground truth", scene=load_scene())

    def interpret(state):
        intent = gateway(state).interpret(state["request"], state["scene"])
        if intent["experiment"] != "michelson_geometric" or intent["unsupported_requirements"]:
            return event("interpret", "The current scope cannot satisfy the full request", intent=intent,status="failed",
                         error={"code":"UNSUPPORTED_REQUIREMENTS","message":str(intent["unsupported_requirements"] or intent["experiment"])})
        return event("interpret", "Keep the original request and extract checkable geometric constraints", intent=intent)

    def optics(state):
        attempt = state.get("layout_attempts",0)+1
        try:
            layout = gateway(state).layout(state["request"], state["scene"],state["intent"],state.get("feedback",[]),attempt)
        except CandidateFormatError as exc:
            return event("optics_agent", str(exc), layout=None, layout_attempts=attempt, format_issue=str(exc))
        return event("optics_agent", f"Generated target layout {attempt}",layout=layout,layout_attempts=attempt,format_issue=None)

    def layout_check(state):
        report = validate_layout(state["layout"],state["scene"],state["intent"]["requirements"])
        if state.get("format_issue"):
            report["issues"] = [{"code":"MODEL_INVALID_JSON","message":state["format_issue"]}]
        return event("layout_validator", "Layout validation passed" if report["passed"] else "Layout validation failed. The specific residuals were saved",validation=report,feedback=report["issues"])

    def coding(state):
        attempt=state.get("plan_attempts",0)+1
        try:
            plan=gateway(state).plan(state["layout"],state["validation"],state.get("feedback",[]),attempt)
        except CandidateFormatError as exc:
            return event("coding_agent", str(exc), plan=None, plan_attempts=attempt, format_issue=str(exc))
        return event("coding_agent", f"Generated constrained action plan {attempt}",plan=plan,plan_attempts=attempt,format_issue=None)

    def plan_check(state):
        report=validate_plan(state["plan"],state["layout"],state["validation"])
        if state.get("format_issue"):
            report["issues"] = [{"code":"MODEL_INVALID_JSON","message":state["format_issue"]}]
        return event("plan_validator", "Action-plan validation passed" if report["passed"] else "Plan was not accepted",plan_validation=report,
                     plan_sha256=report["plan_sha256"],feedback=report["issues"])

    def simulate(state):
        if state["mode"] == "design":
            result={"status":"design_ready","physical_execution":False,"evidence_kind":"design_only",
                    "layout_sha256":canonical_hash(state["layout"]),"plan_sha256":state["plan_sha256"]}
        elif state["mode"] == "local_sim":
            attempt = state.get("sim_attempts", 0) + 1
            plan = state["plan"]
            order = [a["args"]["object_id"] for a in plan["actions"] if a["primitive"] == "pick"]
            signature = canonical_hash({"poses": sorted(state["layout"]["components"], key=lambda c: c["object_id"]),
                                        "pick_order": order, "offset": state.get("command_offset_xy_m", [0, 0])})
            signatures = state.get("sim_signatures", [])
            if signature in signatures:
                return event("simulate", "The revision did not change pick-and-place order, target pose, or compensation. Stop repeating the simulation", status="failed",
                             error={"code": "SIM_NO_PROGRESS", "message": "The simulation input is equivalent to the previous candidate"})
            name = "local-simulation" if attempt == 1 else f"local-simulation-{attempt:02d}"
            result = run_local_sim(state, data_dir / "runs" / state["run_id"] / name)
            history = state.get("sim_history", []) + [{"attempt": attempt, "status": result["status"],
                       "error": result.get("error"), "artifact_directory": name}]
            return event("simulate", f"MuJoCo simulation {attempt} finished. Checking run evidence", result=result,
                         sim_attempts=attempt, sim_history=history, sim_signatures=signatures + [signature])
        else:
            endpoint=read_settings().get("robot_endpoint") if state["mode"]=="remote_sim" else None
            if state["mode"]=="remote_sim" and not endpoint:
                raise ValueError("ROBOT_ENDPOINT_MISSING: the team simulation Robot Task service is not configured")
            result=execute_plan(state["plan"],state["layout"],state["validation"],state["run_id"],data_dir,
                                backend="remote" if state["mode"]=="remote_sim" else "mock",
                                endpoint=endpoint,fail_action=state.get("fail_action"))
        return event("simulate", "This round's result is available. Continue checking its source and plan hash",result=result)

    def output(state):
        result=state["result"]
        if result["status"] not in ("succeeded","design_ready"):
            if repair_options(state):
                return event("output_validator", "The simulation-failure evidence is diagnosable. Enter automatic correction")
            return event("output_validator", "Execution did not pass. Later actions stay stopped",status="failed",error=result.get("error") or {"code":"EXECUTION_FAILED","message":"Execution failed"})
        if result.get("plan_sha256")!=canonical_hash(state["plan"]) or result.get("layout_sha256")!=canonical_hash(state["layout"]):
            raise ValueError("ARTIFACT_HASH_MISMATCH: the result does not match this round's plan")
        if state["mode"]=="remote_sim":
            expected={c["object_id"] for c in state["layout"]["components"] if c["movable"]}
            verifications=result.get("visual_verifications",[])
            steps=result.get("steps",[])
            if (result.get("runtime_visual_pass") is not True or len(verifications)!=len(expected)
                or {v.get("object_id") for v in verifications}!=expected
                or any(v.get("passed") is not True or v.get("source")!="vision" for v in verifications)
                or [s.get("action_id") for s in steps]!=[a["action_id"] for a in state["plan"]["actions"]]
                or any(s.get("status")!="succeeded" for s in steps)):
                raise ValueError("REMOTE_EVIDENCE_INCOMPLETE: every action's final state and real visual checks for three objects are required")
        return event("output_validator", "Checked that the result, plan, and layout agree")

    def repair_options(state):
        if state["mode"] != "local_sim" or state["provider"] != "openai" or state.get("sim_attempts", 0) >= 3:
            return []
        result = state["result"]
        if result["status"] != "failed":
            return []
        code = (result.get("error") or {}).get("code")
        routes = []
        if code == "PLACEMENT_OUT_OF_TOLERANCE":
            measurements = result.get("placement_errors", [])[-2:]
            if (len(measurements) == 2 and measurements[0]["object_id"] == measurements[1]["object_id"]
                    and [m["attempts"] for m in measurements] == [1, 2]
                    and math.dist(*(m["signed_xy_m"] for m in measurements)) <= .001
                    and all(math.hypot(*m["signed_xy_m"]) >= .001 and m["angle_error_rad"] <= .01 for m in measurements)):
                routes.append("calibration")
        if code in ("SIM_OPTICAL_RECHECK_FAILED", "UNEXPECTED_CONTACT", "TOOL_WORKSPACE_EXCEEDED") and state.get("layout_attempts", 0) < 3:
            routes.append("layout")
        if code in ("UNEXPECTED_CONTACT", "ACTION_SIM_TIME_EXCEEDED") and state.get("plan_attempts", 0) < 3:
            routes.append("plan")
        return routes

    def diagnose(state):
        result = state["result"]
        routes = repair_options(state)
        evidence = {"allowed_routes": routes + ["stop"], "error": result["error"],
                    "placement_errors": result.get("placement_errors", []),
                    "previous_command_offset_xy_m": state.get("command_offset_xy_m", [0, 0]),
                    "layout": state["layout"], "plan": state["plan"],
                    "requirements": state["intent"]["requirements"]}
        decision = gateway(state).diagnose_simulation(evidence, state["sim_attempts"])
        route = decision["route"]
        history = state.get("repair_history", []) + [{**decision, "after_sim_attempt": state["sim_attempts"]}]
        if route not in routes:
            return event("diagnose_simulation", "The diagnosis produced no executable revision. Stop", status="failed",
                         error=result["error"], repair_history=history)
        if route == "calibration":
            delta = decision.get("delta_xy_m")
            readings = evidence["placement_errors"][-2:]
            expected = [-sum(row["signed_xy_m"][i] for row in readings) / 2 for i in range(2)]
            if (not isinstance(delta, list) or len(delta) != 2
                    or not all(type(n) in (int, float) and math.isfinite(n) for n in delta)
                    or math.dist(delta, expected) > .001):
                return event("diagnose_simulation", "The compensation direction or magnitude does not match the measured bias. Stop", status="failed",
                             repair_history=history, error={"code":"SIM_REPAIR_INVALID",
                             "message":"SIM_REPAIR_INVALID: compensation must agree with two measurements in the same direction"})
            offset = [a + b for a, b in zip(evidence["previous_command_offset_xy_m"], delta)]
            if math.hypot(*offset) > .025:
                return event("diagnose_simulation", "Cumulative simulation compensation exceeds 25 mm. Stop", status="failed",
                             repair_history=history, error={"code":"SIM_REPAIR_INVALID",
                             "message":"Cumulative simulation compensation exceeds 25 mm"})
            return event("diagnose_simulation", "The compensation proposal passed the measurement check. Simulate again", repair_history=history,
                         repair_route=route, command_offset_xy_m=offset)
        feedback = [{"code": result["error"]["code"], "diagnosis": decision["diagnosis"],
                     "failed_execution": result["error"], "previous_layout": state["layout"],
                     "previous_plan": state["plan"]}]
        return event("diagnose_simulation", f"Return the simulation evidence to the {'layout' if route == 'layout' else 'plan'} node for revision",
                     repair_history=history, repair_route=route, feedback=feedback)

    def review(state):
        # This node has no pre-interrupt side effects and is safe to replay on resume.
        decision=interrupt({"kind":"plan_review","run_id":state["run_id"],"plan_sha256":state["plan_sha256"],
                            "message":"Review this round's layout and plan. This confirmation only finishes the local workflow and does not enable hardware."})
        if not isinstance(decision,dict) or type(decision.get("approved")) is not bool or decision.get("plan_sha256")!=state["plan_sha256"]:
            raise ValueError("A review must include an approved boolean and the current plan_sha256")
        return event("review", "Review approved" if decision["approved"] else "The user rejected this round's plan",approved=decision["approved"],
                     status="running" if decision["approved"] else "rejected")

    def execution(state):
        # A mock artifact can never authorize a hardware action. Concrete real
        # planner/ROSi/arm contracts are still team open items (Draft v0).
        if state["mode"]=="real":
            raise ValueError("HARDWARE_NOT_CONFIGURED")
        return event("execution_gate", "This round has no hardware motion. Keep the same validated plan")

    def legacy(state):
        if state["mode"]=="legacy_evidence":
            result=read_legacy_evidence(PROJECT)
        else:
            result=run_legacy(PROJECT,data_dir/"runs"/state["run_id"]/"legacy-simulation")
        return event("legacy_backend","Historical evidence was read" if result["historical"] else "The legacy-case simulation process returned",result=result)

    def finalize(state):
        status=state.get("status","running")
        if status not in ("failed","blocked","rejected"):
            if state["profile"]=="legacy_vision_michelson_15":
                status="historical_evidence" if state["result"].get("historical") else ("simulation_passed" if state["result"].get("success") else "failed")
            else:
                status={"mock":"protocol_passed","design":"design_ready","remote_sim":"remote_runtime_passed","local_sim":"local_sim_passed"}[state["mode"]]
        if state.get("feedback") and not state.get("result") and status=="running":
            status="failed"
        return event("finalize", "The run has closed. The success scope is defined by the status and evidence type",status=status)

    def retry_layout(state):
        if state.get("status")=="failed":return "finalize"
        if state["validation"]["passed"]:return "coding_agent"
        return "optics_agent" if state["provider"]=="openai" and state.get("layout_attempts",0)<3 else "exhausted"

    def retry_plan(state):
        if state.get("status")=="failed":return "finalize"
        if state["plan_validation"]["passed"]:return "simulate"
        return "coding_agent" if state["provider"]=="openai" and state.get("plan_attempts",0)<3 else "exhausted"

    def exhausted(state):
        return event("feedback_router","The candidate-revision budget is exhausted. Stop",status="failed",error={"code":"VALIDATION_FAILED","message":"The candidate did not pass. See feedback","issues":state.get("feedback",[])})

    builder=StateGraph(State)
    nodes={"context":context,"interpret":interpret,"optics_agent":optics,"layout_validator":layout_check,
           "coding_agent":coding,"plan_validator":plan_check,"simulate":simulate,"output_validator":output,"diagnose_simulation":diagnose,
           "execution_gate":execution,"legacy_backend":legacy,"exhausted":exhausted,"finalize":finalize}
    for name,fn in nodes.items():builder.add_node(name,guarded(name,fn))
    builder.add_node("review",review)
    builder.add_edge(START,"context")
    builder.add_conditional_edges("context",lambda s:"finalize" if s.get("status") in ("failed","blocked") else "legacy_backend" if s["profile"]=="legacy_vision_michelson_15" else "interpret",["finalize","legacy_backend","interpret"])
    for origin,next_node in [("interpret","optics_agent"),("optics_agent","layout_validator"),("coding_agent","plan_validator"),("simulate","output_validator")]:
        builder.add_conditional_edges(origin,lambda s,n=next_node:"finalize" if s.get("status")=="failed" else n,["finalize",next_node])
    builder.add_conditional_edges("layout_validator",retry_layout,["finalize","coding_agent","optics_agent","exhausted"])
    builder.add_conditional_edges("plan_validator",retry_plan,["finalize","simulate","coding_agent","exhausted"])
    builder.add_conditional_edges("output_validator",lambda s:"finalize" if s.get("status")=="failed" else "diagnose_simulation" if repair_options(s) else "review" if s.get("require_review") else "execution_gate",["finalize","diagnose_simulation","review","execution_gate"])
    builder.add_conditional_edges("diagnose_simulation", lambda s: "finalize" if s.get("status")=="failed" else
                                  {"layout":"optics_agent","plan":"coding_agent","calibration":"simulate"}[s["repair_route"]],
                                  ["finalize","optics_agent","coding_agent","simulate"])
    builder.add_conditional_edges("review",lambda s:"execution_gate" if s.get("approved") else "finalize",["execution_gate","finalize"])
    for name in ("execution_gate","legacy_backend","exhausted"):builder.add_edge(name,"finalize")
    builder.add_edge("finalize",END)
    return builder.compile(checkpointer=checkpointer)
