"""Stateful, explicitly synthetic service adapter for architecture qualification.

No camera, dynamics engine or robot occurs here. AgentAlgorithmServices supplies
real model decisions while this adapter implements execution. The existing team
graph owns routing, retries and journals; this adapter only implements its ports.
"""
from copy import deepcopy
from io import BytesIO
import math
import re
from typing import Literal
from zipfile import ZipFile, ZIP_DEFLATED

from pydantic import BaseModel, ConfigDict, Field

from .algorithm_experiment import DESIGN_SOURCES, build_experiment, verify_experiment, verify_target
from .sim_io import decode_json, read_bytes, read_json
from .team_artifacts import atomic, digest, encode

SOURCE = "algorithm_v1"
SCHEMA = "algorithm-rosi/v1"
POSITION_LIMIT = .001
ANGLE_LIMIT = math.radians(.5)
# Providers whose in-graph decisions come from a model-style answerer (OpenAI, or the MCP client).
MODEL_PROVIDERS = {"openai", "agent"}


class AlgorithmOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    position_response: float = Field(default=.5, ge=0, le=1)
    angle_response: float = Field(default=.8, ge=0, le=1)
    initial_position_error_m: float = Field(default=.003, ge=0, le=.01)
    initial_angle_error_rad: float = Field(default=math.radians(2), ge=0, le=.1)
    delay_s: float = Field(default=.65, ge=0, le=2)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    object_id: str
    operation: Literal["translate", "align", "place"]
    delta_position_m: list[float] = Field(min_length=3, max_length=3)
    delta_yaw_rad: float


class AlgorithmPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal["algorithm-rosi/v1"]
    source: Literal["algorithm_v1"]
    scene_revision: str
    frame_id: Literal["optical_table"]
    position_unit: Literal["m"]
    angle_unit: Literal["rad"]
    world_revision: int = Field(ge=0)
    current_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assumptions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    phase: Literal["repair", "assembly"] = "repair"
    allowed_object_ids: list[str] = Field(max_length=128)
    actions: list[Action] = Field(max_length=128)


def poses(layout):
    return {c["object_id"]: {"position_m":deepcopy(c["target_pose"]["position_m"]),
                              "yaw_rad":2*math.atan2(c["target_pose"]["orientation_xyzw"][2],
                                                    c["target_pose"]["orientation_xyzw"][3])}
            for c in layout["components"]}


def angle_delta(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


def evaluate(current, target, limits=None):
    limits = limits or {"position_m":POSITION_LIMIT,"angle_rad":ANGLE_LIMIT}
    if set(current) != set(target):
        raise ValueError("The simulated bench component set changed")
    for pose in current.values():
        if (set(pose) != {"position_m", "yaw_rad"} or len(pose["position_m"]) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in [*pose["position_m"], pose["yaw_rad"]])):
            raise ValueError("The simulated pose format is invalid")
    position = {k:math.dist(current[k]["position_m"], target[k]["position_m"]) for k in target}
    angle = {k:abs(angle_delta(current[k]["yaw_rad"], target[k]["yaw_rad"])) for k in target}
    return {"position_errors_m":position, "angle_errors_rad":angle,
            "mechanical_passed":all(v <= limits["position_m"] for v in position.values()),
            "optical_passed":all(v <= limits["angle_rad"] for v in angle.values()),
            "limits":deepcopy(limits)}


class AlgorithmServices:
    def __init__(self, root, error_type):
        self.root, self.Error = root, error_type

    def fail(self, code, message, target=None, details=None):
        raise self.Error(code, message, target, details)

    def injected(self, state, node, attempt):
        return attempt == 1 and state.get("team_fault") in (node, "matrix")

    def invoke(self, node, state, path, attempt):
        return getattr(self, node)(state, path, attempt)

    def input(self, state, path, attempt):
        try:
            selected, scene, reference, experiment = build_experiment(state["request"])
        except ValueError as exc:
            self.fail("EXPERIMENT_DEFINITION_INVALID", str(exc))
        prior = state.get("algorithm_experiment")
        if prior and prior != experiment:
            self.fail("EXPERIMENT_DEFINITION_CHANGED", "The experiment definition changed when the input was reread. The previous bench cannot be reused.")
        atomic(path/"experiment.json", encode(experiment))
        world = state.get("algorithm_state")
        assumptions = AlgorithmOptions.model_validate(state.get("algorithm_options", {})).model_dump()
        if world is None:
            current = poses(reference)
            movable = [c["object_id"] for c in reference["components"] if c["movable"]]
            part = "L25" if "L25" in movable else next(iter(movable), None)
            if part:
                current[part]["position_m"][0] += assumptions["initial_position_error_m"]
                current[part]["yaw_rad"] += assumptions["initial_angle_error_rad"]
            world = {"current":current, "initial":deepcopy(current), "revision":0,
                     "source":SOURCE, "seeded_object_id":part}
        return {"selected_experiment":selected, "scene":scene, "reference_layout":reference,
                "algorithm_state":world, "algorithm_options":assumptions,
                "algorithm_experiment":experiment,
                "local_registered_request":experiment["source"] == "registered_reference"}

    def vision(self, state, path, attempt):
        observation = {"schema_version":"algorithm-observation/v1", "source":SOURCE,
                       "frame_id":"optical_table", "world_revision":state["algorithm_state"]["revision"],
                       "observations":deepcopy(state["algorithm_state"]["current"]),
                       "camera_measured":False}
        atomic(path/"prob0.json", encode(observation))
        if self.injected(state, "vision", attempt):
            self.fail("SYNTHETIC_OBSERVATION_MISSING", "Test case: the observation record is missing. Return to input and acquire the simulated observation again.", "input")
        return {"prob":observation}

    def prompt(self, state, path, attempt):
        experiment = state.get("algorithm_experiment")
        if experiment:
            verify_experiment(experiment, state["request"])
        intent = {"experiment":state["selected_experiment"], "summary":state["request"],
                  "requirements":{}, "source":experiment["source"] if experiment else "registered_request",
                  "unsupported_requirements":[],
                  "pending_requirements":experiment["pending_requirements"] if experiment else [],
                  "experiment_sha256":experiment["contract_sha256"] if experiment else None}
        if self.injected(state, "prompt", attempt):
            intent["requirements"] = {"unknown":True}
        atomic(path/"prompt.json", encode(intent))
        atomic(path/"prompt.txt", ("Algorithm-architecture check. Use this experiment definition and rule_policy_v1. Do not generate a physical measurement.\n"
            + "Request: " + state["request"] + "\nCurrent simulated layout: " + encode(state["prob"]).decode()
            + "\nExperiment definition, assumptions, and requirements still to verify: " + encode(experiment).decode()
            + "\nFeedback: " + encode(state.get("feedback", []), limit=None).decode()
            + "\nRules: keep the design target and tolerances, and plan only the component identified by the evidence. Execute and recompute independently only after validation.\n").encode())
        return {"intent":intent}

    def decision(self, node, state, path, attempt, action, part, diagnosis, changes):
        decision = {"node":node, "attempt":attempt, "source":"rule_policy_v1", "action":action,
                    "diagnosis":diagnosis, "target_object_ids":[part] if part else [],
                    "changes":changes, "input_sha256":digest(encode(state["algorithm_state"]["current"]))}
        atomic(path/"decision.json", encode(decision))
        return decision

    def act(self, state, path, attempt):
        layout = deepcopy(state["reference_layout"])
        update = {"layout":layout}
        feedback = state.get("feedback", [])
        if feedback and feedback[-1]["code"] == "ALGORITHM_OPTICAL_MISMATCH":
            part = feedback[-1]["details"]["target_object_ids"][0]
            delta = angle_delta(poses(layout)[part]["yaw_rad"], state["algorithm_state"]["current"][part]["yaw_rad"])
            update["algorithm_pending"] = {"operation":"align", "object_id":part}
            decision = self.decision("act", state, path, attempt, "align_single_object", part,
                "The assumed optical-angle metric is out of tolerance. Keep the design target, propose a yaw change for that component only, then pass it through Validator and Controller.",
                {"delta_yaw_rad":delta, "target_layout_changed":False})
            update["algorithm_decisions"] = [decision]
        if self.injected(state, "act", attempt):
            layout["scene_revision"] = "invalid-test-revision"
        atomic(path/"target_layout.json", encode(layout))
        return update

    def validator(self, state, path, attempt):
        experiment = state.get("algorithm_experiment")
        if experiment:
            verify_experiment(experiment, state["request"])
            if state["intent"].get("experiment_sha256") != experiment["contract_sha256"]:
                self.fail("ALGORITHM_REQUIREMENTS_FORMAT", "The requirements do not match this experiment definition. Return to Prompt.", "prompt")
        if state["intent"]["requirements"]:
            self.fail("ALGORITHM_REQUIREMENTS_FORMAT", "Test case: the requirements contain an undeclared field. Return to Prompt.", "prompt")
        if state["layout"] != state["reference_layout"]:
            self.fail("ALGORITHM_LAYOUT_FORMAT", "The layout revision, components, target, or tolerance does not match. Return to Act.", "act")
        if experiment:
            verify_target(experiment, state["layout"])
        result = {"passed":True, "source":SOURCE, "scope":"algorithm_experiment_contract",
                  "design_verified":False,
                  "pending_requirements":experiment["pending_requirements"] if experiment else [],
                  "physical_validated":False, "issues":[]}
        atomic(path/"validation.json", encode(result))
        return {"validation":result}

    def controller(self, state, path, attempt):
        world, target = state["algorithm_state"], poses(state["layout"])
        current = world["current"]
        pending = state.get("algorithm_pending")
        if pending:
            part, operation = pending["object_id"], pending["operation"]
        else:
            errors = evaluate(current, target)["position_errors_m"]
            part = next((k for k,v in errors.items() if v > POSITION_LIMIT), None)
            operation = "translate"
        actions = []
        if part:
            if not any(c["object_id"] == part and c["movable"] for c in state["layout"]["components"]):
                self.fail("ALGORITHM_FIXED_OBJECT", "The fault target is a fixed component and cannot be moved.")
            actions = [{"object_id":part, "operation":operation,
                        "delta_position_m":[float(t-c) if operation == "translate" else 0.
                                            for t,c in zip(target[part]["position_m"], current[part]["position_m"])],
                        "delta_yaw_rad":angle_delta(target[part]["yaw_rad"], current[part]["yaw_rad"]) if operation == "align" else 0.}]
        plan = {"schema_version":SCHEMA, "source":SOURCE, "scene_revision":state["scene"]["scene_revision"],
                "frame_id":"optical_table", "position_unit":"m", "angle_unit":"rad",
                "world_revision":world["revision"], "current_sha256":digest(encode(current)),
                "target_sha256":digest(encode(state["layout"])),
                "assumptions_sha256":digest(encode(state["algorithm_options"])),
                "experiment_sha256":(state.get("algorithm_experiment") or {}).get("contract_sha256"),
                "allowed_object_ids":[part] if part else [], "actions":actions}
        decision = self.decision("controller", state, path, attempt,
            "correct_position" if part and operation == "translate" else "apply_alignment" if part else "verify_only", part,
            "Plan a single-component action from the difference between this round's simulated observation and the target. Leave the other components and the acceptance tolerance unchanged." if part else "The current position is within tolerance. Run a read-only check.",
            {"actions":actions, "preserved_object_ids":[k for k in current if k != part]})
        if self.injected(state, "controller", attempt):
            self.fail("ALGORITHM_PLAN_FORMAT", "Test case: the plan units are invalid. Return to Controller and recompute.", "controller")
        raw = encode(AlgorithmPlan.model_validate(plan).model_dump())
        atomic(path/"plan.json", raw)
        atomic(path/"plan.sha256", digest(raw).encode())
        return {"plan":plan, "plan_path":str((path/"plan.json").relative_to(self.root)),
                "plan_sha256":digest(raw), "algorithm_pending":None,
                "algorithm_decisions":[decision]}

    def plan(self, state):
        experiment = state.get("algorithm_experiment")
        if experiment:
            verify_experiment(experiment, state["request"])
            verify_target(experiment, state["layout"])
            if experiment["source"] in DESIGN_SOURCES and (
                    experiment["design_sha256"] != digest(encode(state.get("algorithm_design")))
                    or experiment["intent_sha256"] != digest(encode(state.get("algorithm_intent")))
                    or experiment["execution_limits"] != state["layout"]["tolerance"]):
                raise ValueError("The agent design or the requirements have changed")
        if not re.fullmatch(r"team/controller-[0-9]+/plan\.json", state["plan_path"]):
            raise ValueError("The plan did not come from this round's Controller")
        raw = read_bytes(self.root/state["plan_path"])
        if digest(raw) != state["plan_sha256"]:
            raise ValueError("Plan bytes do not match the recorded hash.")
        plan = AlgorithmPlan.model_validate(decode_json(raw.decode())).model_dump()
        movable = [c["object_id"] for c in state["layout"]["components"] if c["movable"]]
        if plan["phase"] == "assembly":
            if (state.get("provider") not in MODEL_PROVIDERS or plan["world_revision"] != 0
                    or plan["allowed_object_ids"] != movable
                    or any(a["operation"] != "place" for a in plan["actions"])):
                raise ValueError("The initial assembly plan scope or bench revision does not match")
        elif (len(plan["allowed_object_ids"]) > 1 or len(plan["actions"]) > 1
              or any(a["operation"] == "place" for a in plan["actions"])):
            raise ValueError("The repair plan exceeds the single-object authorization.")
        if not set(plan["allowed_object_ids"]) <= set(movable):
            raise ValueError("The plan includes a fixed or unknown component")
        if (plan["target_sha256"] != digest(encode(state["layout"]))
                or plan["assumptions_sha256"] != digest(encode(state["algorithm_options"]))
                or plan["experiment_sha256"] != (experiment or {}).get("contract_sha256")
                or plan["scene_revision"] != state["scene"]["scene_revision"]):
            raise ValueError("The target, model assumptions, or scene revision has changed")
        decisions = [d for d in state.get("algorithm_decisions", []) if d["node"] == "controller"]
        if not decisions or plan["allowed_object_ids"] != decisions[-1]["target_object_ids"]:
            raise ValueError("The plan exceeds the node-approved single-object authorization.")
        return plan, raw

    def sim(self, state, path, attempt):
        if self.injected(state, "sim", attempt):
            self.fail("ALGORITHM_SERVICE_TIMEOUT", "Test case: the simulation service timed out. The original plan was not applied.", "sim")
        plan, raw = self.plan(state)
        world = deepcopy(state["algorithm_state"])
        previous = state.get("algorithm_result")
        # Output may request a fresh archive. Reuse the exact completed job,
        # without applying its motion twice to the persistent synthetic table.
        reused = previous is not None and previous["plan_hash"] == digest(raw)
        if reused:
            result = deepcopy(previous)
            if world["current"] != result["after"] or world["revision"] != result["revision"]:
                raise ValueError("The bench for the repeated request no longer matches the existing simulation result")
        else:
            before = deepcopy(world["current"])
            if plan["current_sha256"] != digest(encode(before)) or plan["world_revision"] != world["revision"]:
                raise ValueError("The bench at the start of the simulated plan is stale")
            if [a["object_id"] for a in plan["actions"]] != plan["allowed_object_ids"]:
                raise ValueError("The action exceeds the single-object authorization.")
            target = poses(state["layout"])
            for action in plan["actions"]:
                part = action["object_id"]
                if part not in before:
                    raise ValueError("The plan references an unknown component")
                delta, yaw = action["delta_position_m"], action["delta_yaw_rad"]
                expected = [t-c for t,c in zip(target[part]["position_m"], before[part]["position_m"])]
                if action["operation"] == "place":
                    if (plan["phase"] != "assembly" or any(abs(a-b)>1e-12 for a,b in zip(delta,expected))
                            or abs(yaw-angle_delta(target[part]["yaw_rad"],before[part]["yaw_rad"]))>1e-12):
                        raise ValueError("The initial assembly does not match the agent target")
                    world["current"][part] = {"position_m":[c+d for c,d in zip(before[part]["position_m"],delta)],
                                              "yaw_rad":before[part]["yaw_rad"]+yaw}
                    if part == world["seeded_object_id"]:
                        world["current"][part]["position_m"][0] += state["algorithm_options"]["initial_position_error_m"]
                        world["current"][part]["yaw_rad"] += state["algorithm_options"]["initial_angle_error_rad"]
                    continue
                if action["operation"] == "translate":
                    if any(abs(a-b)>1e-12 for a,b in zip(delta, expected)) or yaw != 0.:
                        raise ValueError("The translation proposal does not match the current error")
                elif any(delta) or abs(yaw-angle_delta(target[part]["yaw_rad"],before[part]["yaw_rad"]))>1e-12:
                    raise ValueError("The yaw proposal does not match the current error")
                world["current"][part]["position_m"] = [c+state["algorithm_options"]["position_response"]*d
                                                        for c,d in zip(before[part]["position_m"], delta)]
                world["current"][part]["yaw_rad"] += state["algorithm_options"]["angle_response"]*yaw
            world["revision"] += 1
            result = {"source":SOURCE, "physical_verified":False, "attempt":attempt, "before":before, "after":world["current"],
                      "moved_ids":[a["object_id"] for a in plan["actions"]],
                      "revision":world["revision"], "plan_hash":digest(raw),
                      "assumptions":state["algorithm_options"],
                      **evaluate(world["current"], target,(state.get("algorithm_experiment") or {}).get("execution_limits"))}
        atomic(path/"plan.json", raw)
        atomic(path/"simulation.json", encode(result))
        payload = encode(result)
        manifest = {"schema_version":"algorithm-artifact/v1", "source":SOURCE,
                    "plan_hash":digest(raw), "result_hash":digest(payload)}
        if self.injected(state, "output", attempt):
            manifest["result_hash"] = "0"*64
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
            for name, blob in (("plan.json", raw), ("manifest.json", encode(manifest)), ("result.json", payload)):
                archive.writestr(name, blob)
        archive_data = buffer.getvalue()
        atomic(path/"artifact.zip", archive_data)
        observation = {"schema_version":"algorithm-observation/v1", "source":SOURCE,
                       "camera_measured":False, "frame_id":"optical_table", "world_revision":world["revision"],
                       "observations":deepcopy(world["current"])}
        return {"algorithm_state":world, "algorithm_result":result, "prob":observation,
                "algorithm_history":[] if reused else [deepcopy(result)],
                "artifact_path":str((path/"artifact.zip").relative_to(self.root)),
                "artifact_sha256":digest(archive_data)}

    def verify_archive(self, state):
        plan, raw = self.plan(state)
        blob = read_bytes(self.root/state["artifact_path"], limit=800_000)
        if digest(blob) != state["artifact_sha256"]:
            raise ValueError("The evidence archive has changed")
        with ZipFile(BytesIO(blob)) as archive:
            entries = archive.infolist()
            if (len(entries) != 3 or {e.filename for e in entries} != {"plan.json", "manifest.json", "result.json"}
                    or any(e.file_size > 256_000 or e.flag_bits & 1 for e in entries)):
                raise ValueError("An evidence ZIP member or its size does not match")
            saved = archive.read("plan.json")
            result_raw = archive.read("result.json")
            manifest = decode_json(archive.read("manifest.json").decode())
        if (saved != raw or manifest != {"schema_version":"algorithm-artifact/v1", "source":SOURCE,
                                         "plan_hash":digest(raw), "result_hash":digest(result_raw)}):
            raise ValueError("The plan or result hash in the evidence does not match")
        result = decode_json(result_raw.decode())
        if (result != state["algorithm_result"] or result["source"] != SOURCE
                or result["physical_verified"] is not False
                or result["after"] != state["algorithm_state"]["current"]
                or result["revision"] != state["algorithm_state"]["revision"]
                or result["plan_hash"] != digest(raw)
                or result["assumptions"] != state["algorithm_options"]
                or result["moved_ids"] != plan["allowed_object_ids"]
                or any(result["before"][k] != result["after"][k]
                       for k in result["before"] if k not in plan["allowed_object_ids"])):
            raise ValueError("The simulated evidence source, status, or single-component scope does not match")
        if (digest(encode(result["before"])) != plan["current_sha256"]
                or result["revision"] != plan["world_revision"] + 1):
            raise ValueError("The simulation result does not match the bench at the start of the plan")
        recomputed = deepcopy(result["before"])
        for action in plan["actions"]:
            pose = recomputed[action["object_id"]]
            if action["operation"] == "place":
                pose["position_m"] = [c+d for c,d in zip(pose["position_m"],action["delta_position_m"])]
                pose["yaw_rad"] += action["delta_yaw_rad"]
                if action["object_id"] == state["algorithm_state"]["seeded_object_id"]:
                    pose["position_m"][0] += state["algorithm_options"]["initial_position_error_m"]
                    pose["yaw_rad"] += state["algorithm_options"]["initial_angle_error_rad"]
                continue
            pose["position_m"] = [c+state["algorithm_options"]["position_response"]*d
                                  for c,d in zip(pose["position_m"], action["delta_position_m"])]
            pose["yaw_rad"] += state["algorithm_options"]["angle_response"]*action["delta_yaw_rad"]
        if recomputed != result["after"]:
            raise ValueError("The simulated bench does not follow this round's command and response rules")
        checks = evaluate(result["after"], poses(state["layout"]),(state.get("algorithm_experiment") or {}).get("execution_limits"))
        if any(result[k] != v for k,v in checks.items()):
            raise ValueError("The independent recomputation does not match the simulation report")
        return result, raw

    def verify_analytic_tests(self, state):
        from .optical_tests import evaluate_suite
        review = state.get("algorithm_review") or {}
        suite, saved = state.get("algorithm_test_suite"), state.get("algorithm_test_report")
        if (review.get("accepted") is not True
                or review.get("experiment_sha256") != (state.get("algorithm_experiment") or {}).get("contract_sha256")
                or not suite or not saved or not saved.get("passed")
                or review.get("test_suite_sha256") != digest(encode(suite))
                or review.get("test_report_sha256") != digest(encode(saved))):
            self.fail("OPTICAL_TEST_BINDING_CHANGED", "The experiment tests and calculation report bound to the design review are missing.")
        actual = evaluate_suite(suite,state["algorithm_intent"],state["algorithm_design"])
        if actual != saved:
            self.fail("OPTICAL_TEST_BINDING_CHANGED", "The independent recomputation of the experiment tests does not match the reviewed report.")
        return actual

    def output(self, state, path, attempt):
        try:
            result, raw = self.verify_archive(state)
        except ValueError as exc:
            self.fail("ALGORITHM_ARTIFACT_REJECTED", str(exc), "sim")
        verification = {"source":SOURCE, "physical_verified":False, "design_verified":False,
                        "experiment_sha256":(state.get("algorithm_experiment") or {}).get("contract_sha256"),
                        "pending_requirements":(state.get("algorithm_experiment") or {}).get("pending_requirements", []),
                        "plan_hash":digest(raw),
                        "artifact_sha256":state["artifact_sha256"],
                        "checks":{"protocol":True, "same_plan":True,
                                  "mechanical_proxy":result["mechanical_passed"],
                                  "optical_proxy":result["optical_passed"]},
                        "mechanical":{"passed":result["mechanical_passed"]},
                        "optical":{"passed":result["optical_passed"]}}
        if state.get("provider") in MODEL_PROVIDERS:
            review = state.get("algorithm_review") or {}
            if (not review.get("accepted")
                    or review.get("experiment_sha256") != state["algorithm_experiment"]["contract_sha256"]):
                self.fail("DESIGN_REVIEW_MISSING", "This agent design has no concept review, so it cannot be released.")
            verification["checks"]["conceptual_design"] = True
            tests = self.verify_analytic_tests(state)
            verification["checks"]["analytic_design_tests"] = tests["passed"]
            verification["deferred_optical_requirements"] = tests["deferred"]
            verification["test_suite_sha256"] = review["test_suite_sha256"]
            atomic(path/"optical-test-report.json",encode(tests))
        atomic(path/"verification.json", encode(verification))
        for field, limit, code, node, label in (
                ("position_errors_m", result["limits"]["position_m"], "ALGORITHM_POSITION_MISMATCH", "controller", "Simulated position error is out of tolerance"),
                ("angle_errors_rad", result["limits"]["angle_rad"], "ALGORITHM_OPTICAL_MISMATCH", "act", "Assumed optical-angle metric is out of tolerance")):
            part = next((k for k,v in result[field].items() if v > limit), None)
            if part:
                self.fail(code, f"{label}: {part}; propose again for that component only.", node,
                          {"target_object_ids":[part], "value":result[field][part], "limit":limit,
                           "world_revision":state["algorithm_state"]["revision"], "source":SOURCE})
        return {"verification":verification}

    def execution(self, state, path, attempt):
        result, raw = self.verify_archive(state)
        required = {"protocol","same_plan","mechanical_proxy","optical_proxy"}
        if state.get("provider") in MODEL_PROVIDERS:
            self.verify_analytic_tests(state)
            required |= {"conceptual_design","analytic_design_tests"}
        verification = state.get("verification") or {}
        if (any(verification.get("checks", {}).get(k) is not True for k in required)
                or not result["mechanical_passed"] or not result["optical_passed"]
                or verification.get("experiment_sha256") != state["algorithm_experiment"]["contract_sha256"]
                or state.get("provider") in MODEL_PROVIDERS and verification.get("test_suite_sha256") != state["algorithm_review"].get("test_suite_sha256")
                or verification.get("plan_hash") != digest(raw)
                or verification.get("artifact_sha256") != state["artifact_sha256"]
                or self.injected(state, "execution", attempt) and state.get("team_fault") != "matrix"):
            self.fail("ALGORITHM_HANDOFF_REJECTED", "The executor dry-run refused the handoff: acceptance or file binding does not match.")
        receipt = {"source":SOURCE, "status":"accepted_dry_run", "motions_sent":0,
                   "physical_verified":False, "plan_hash":digest(raw),
                   "artifact_sha256":state["artifact_sha256"], "checks":verification["checks"]}
        atomic(path/"plan.json", raw)
        atomic(path/"execution_receipt.json", encode(receipt))
        return {"result":receipt}
