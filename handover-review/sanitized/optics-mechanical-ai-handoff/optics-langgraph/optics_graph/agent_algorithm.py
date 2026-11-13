"""Real design/recovery Agents with a synthetic execution adapter."""
from copy import deepcopy
import math

from . import design_agent
from .algorithm_experiment import package_design, reference_for, verify_experiment
from .algorithm_sim import (AlgorithmServices, AlgorithmOptions, AlgorithmPlan, SOURCE, SCHEMA,
                            POSITION_LIMIT, ANGLE_LIMIT, poses, angle_delta, evaluate)
from .experiments import load_experiment
from .agent_decisions import AgentDecisionGateway, EXTERNAL_MODEL
from .gateway import ModelGateway, CandidateFormatError, RetryableGatewayError
from .settings import public_settings
from .sim_io import read_json
from .team_artifacts import atomic, digest, encode
from .optical_tests import evaluate_suite


class AgentAlgorithmServices(AlgorithmServices):
    def __init__(self, root, error_type, gateway_factory=ModelGateway):
        super().__init__(root, error_type)
        self.gateway_factory = gateway_factory

    def gateway(self, state, path):
        if state.get("provider") == "agent":
            # In-graph decisions are supplied by the MCP client. No client, key or settings file is touched.
            gateway = AgentDecisionGateway("agent", path)
            gateway.output_language = state.get("output_language", "en")
            if state.get("agent_model") != EXTERNAL_MODEL:
                self.fail("MODEL_CONFIGURATION_CHANGED", "The agent-decision configuration does not match this run. Start a new run.")
            return gateway
        gateway = self.gateway_factory(state.get("provider", "openai"), path)
        gateway.output_language = state.get("output_language", "en")
        if type(gateway) is ModelGateway and public_settings()["model"] != state["agent_model"]:
            self.fail("MODEL_CONFIGURATION_CHANGED", "The model configuration does not match this run. Start a new run.")
        return gateway

    def record(self, node, state, path, attempt, action, diagnosis, ids=(), changes=None, source=None):
        response = read_json(path/"model-response.json", limit=None) if (path/"model-response.json").exists() else {}
        decision = {"node":node, "attempt":attempt, "source":source or state.get("provider", "openai"),
                    "model":response.get("model", state.get("agent_model")),
                    "response_id":response.get("response_id"), "action":action,
                    "diagnosis":diagnosis, "target_object_ids":list(ids), "changes":changes or {},
                    "input_sha256":digest(encode(state["algorithm_state"]["current"]))}
        if state.get("provider") == "agent" and isinstance(response.get("decided_by"), dict):
            decision["decided_by"] = response["decided_by"]
        atomic(path/"decision.json", encode(decision))
        return decision

    def input(self, state, path, attempt):
        # No target geometry is fabricated before the design Agent runs.
        if state.get("provider") == "agent":
            model = EXTERNAL_MODEL  # no OpenAI settings are read and no model is pinned
        else:
            model = public_settings()["model"]
            if not model.startswith("gpt-6.1-sol"):
                self.fail("REQUESTED_MODEL_MISMATCH", "This workflow uses GPT-6.1 Sol as requested. Configure that model in settings.")
        options = AlgorithmOptions.model_validate(state.get("algorithm_options", {})).model_dump()
        world = state.get("algorithm_state") or {"current":{}, "initial":{}, "revision":0,
                    "source":SOURCE, "seeded_object_id":None, "initial_source":"empty_simulated_workbench"}
        result = {"selected_experiment":"agent_designed", "agent_model":state.get("agent_model", model),
                  "algorithm_options":options, "algorithm_state":world, "local_registered_request":False}
        atomic(path/"request.json", encode({"request":state["request"], "model":result["agent_model"],
                                            "execution_backend":SOURCE, "design_backend":state.get("provider", "openai")}))
        return result

    def prompt(self, state, path, attempt):
        intent = design_agent.interpret(self.gateway(state,path), state["request"], state["prob"],
                                        state.get("feedback", []), attempt)
        atomic(path/"requirements.json", encode(intent))
        atomic(path/"prompt.json", encode(intent))
        if intent["clarification_questions"]:
            self.fail("EXPERIMENT_NEEDS_CLARIFICATION", "；".join(intent["clarification_questions"]),
                      details={"questions":intent["clarification_questions"]})
        if not intent["requirements"]:
            raise CandidateFormatError("Prompt extracted no experiment requirements, so an acceptable design cannot be created.")
        if self.injected(state, "prompt", attempt):
            intent["requirements"] = []
        decision = self.record("prompt", state,path,attempt,"interpret_experiment", intent["objective"],
                               changes={"requirements":intent["requirements"],"assumptions":intent["assumptions"]})
        return {"intent":{"summary":state["request"], "source":state.get("provider", "openai"), "requirements":intent["requirements"]},
                "algorithm_intent":intent, "algorithm_decisions":[decision]}

    def local_repair(self, node, state, path, attempt, expected):
        code = "ALGORITHM_POSITION_MISMATCH" if expected == "translate" else "ALGORITHM_OPTICAL_MISMATCH"
        diagnostic = next((f for f in reversed(state.get("feedback", []))
                           if f.get("code") == code and f.get("stage") == "output"
                           and f.get("requested_return") == node
                           and (f.get("details") or {}).get("world_revision") == state["algorithm_state"]["revision"]), {})
        ids = (diagnostic.get("details") or {}).get("target_object_ids", [])
        if len(ids) != 1 or ids[0] not in state["algorithm_state"]["current"]:
            self.fail("REPAIR_SCOPE_UNPROVEN", "The failure does not identify a single component, so no correction can be issued.")
        result = design_agent.repair(self.gateway(state,path), node, state["request"], state["algorithm_design"],
                                    state["algorithm_state"]["current"], diagnostic, ids, attempt)
        decision = self.record(node,state,path,attempt,result["operation"],result["reason"],
                               [result["object_id"]] if result["object_id"] else [],
                               {"proposal":result,"design_changed":False})
        if result["operation"] == "stop":
            self.fail("AGENT_RECOVERY_STOPPED", result["reason"])
        if result["operation"] != expected or result["object_id"] != ids[0]:
            self.fail("AGENT_RECOVERY_INVALID", "The agent proposal does not match this round's failure class or single-component scope.", node)
        return result, decision

    def act(self, state, path, attempt):
        world = state["algorithm_state"]
        if world["revision"] > 0:
            # After execution starts, this port can only propose the diagnosed
            # local adjustment. It cannot regenerate an easier target layout.
            proposal, decision = self.local_repair("act",state,path,attempt,"align")
            layout = deepcopy(state["reference_layout"])
            atomic(path/"target_layout.json", encode(layout))
            return {"layout":layout, "algorithm_pending":proposal, "algorithm_decisions":[decision]}
        last_failure = (state.get("feedback") or [{}])[-1]
        if ((last_failure.get("code") == "OPTICAL_TEST_INVALID"
                or (last_failure.get("details") or {}).get("failed_agent") in {"optical_tests","review"})
                and state.get("algorithm_design")
                and (last_failure.get("details") or {}).get("candidate_sha256") == digest(encode(state["algorithm_design"]))):
            # A malformed test is not evidence that the optical design must move.
            # Keep the canonical Act -> Validator edge; the test Agent gets its
            # own failed formula and evaluator error in feedback on the next visit.
            layout = deepcopy(state["reference_layout"])
            atomic(path/"target_layout.json",encode(layout))
            atomic(path/"design.json",encode(state["algorithm_design"]))
            decision = self.record("act",state,path,attempt,"preserve_design_for_test_repair",
                                   "The test expression is invalid. Keep the original candidate and return it to the Validator test agent.",
                                   source="validated_design_reuse")
            return {"layout":layout,"algorithm_decisions":[decision]}
        reference = None
        key = reference_for(state["request"])
        if key:
            scene, layout = load_experiment(key)
            reference = {"reference_id":key,"scene":scene,"layout":layout}
        if state.get("algorithm_design"):
            reference = {"previous_candidate":state["algorithm_design"],
                         "candidate_sha256":digest(encode(state["algorithm_design"])),"baseline_reference":reference}
        design = design_agent.design(self.gateway(state,path), state["request"], state["algorithm_intent"],
                                     state["prob"], state.get("feedback", []), reference, attempt)
        atomic(path/"design.json", encode(design))
        # Malformed coordinates cannot be packaged; valid but unsuitable optical
        # designs are assessed by Validator, which owns the return edge.
        try:
            scene, layout, experiment = package_design(state["request"], state["algorithm_intent"], design,
                                                      state.get("provider", "openai"))
        except ValueError as exc:
            raise CandidateFormatError("Act design format does not match the target contract: " + str(exc)[:300]) from exc
        if self.injected(state, "act", attempt):
            layout["scene_revision"] = "invalid-test-revision"
        atomic(path/"experiment.json", encode(experiment))
        atomic(path/"target_layout.json", encode(layout))
        decision = self.record("act",state,path,attempt,"design_experiment",design["rationale"],
                               [c["object_id"] for c in design["components"]],
                               {"connections":design["connections"],"requirements_coverage":design["requirements_coverage"]})
        return {"scene":scene,"layout":layout,"reference_layout":deepcopy(layout),
                "algorithm_design":design,"algorithm_experiment":experiment,"algorithm_pending":None,
                "algorithm_review":None,"algorithm_test_suite":None,"algorithm_test_report":None,
                "algorithm_decisions":[decision]}

    def validator(self, state, path, attempt):
        intent, design = state["algorithm_intent"], state["algorithm_design"]
        issues = design_agent.check_design(intent, design)
        if not intent["requirements"]:
            self.fail("AGENT_REQUIREMENTS_EMPTY", "The requirement definition is empty. Return to Prompt and reread the request.", "prompt")
        if state["layout"]["scene_revision"] != state["scene"]["scene_revision"]:
            issues.append("The layout revision does not match this design")
        if issues:
            atomic(path/"validation.json", encode({"passed":False,"issues":issues,"physical_validated":False}))
            self.fail("AGENT_DESIGN_CONTRACT", "；".join(issues), "act",
                      {"issues":issues,"candidate_sha256":digest(encode(design))})
        experiment = state["algorithm_experiment"]
        verify_experiment(experiment, state["request"])
        if experiment["intent_sha256"] != digest(encode(intent)) or experiment["design_sha256"] != digest(encode(design)):
            self.fail("DESIGN_BINDING_CHANGED", "The design is not bound to this requirement.")
        from .algorithm_experiment import verify_target
        verify_target(experiment,state["layout"])
        review = state.get("algorithm_review")
        suite = state.get("algorithm_test_suite")
        report = state.get("algorithm_test_report")
        if state["algorithm_state"]["revision"] > 0 and (
                not review or not review["accepted"] or review.get("experiment_sha256") != experiment["contract_sha256"]):
            self.fail("DESIGN_REVIEW_CHANGED", "The reused concept review does not match the current design contract.")
        if not (state["algorithm_state"]["revision"] > 0 and review and review["accepted"]):
            try:
                suite = design_agent.generate_tests(self.gateway(state,path),state["request"],intent,design,
                                                    state.get("feedback", []),attempt)
            except CandidateFormatError as exc:
                self.fail("OPTICAL_TEST_INVALID",str(exc),"act",
                          {"candidate_sha256":digest(encode(design)),"test_authoring_error":str(exc)})
            except RetryableGatewayError as exc:
                exc.details = {**(exc.details or {}),"failed_agent":"optical_tests",
                               "candidate_sha256":digest(encode(design))}
                raise
            finally:
                # Preserve both Validator agents' original prompts and responses.
                for source, target in (("agent-prompt.txt","test-agent-prompt.txt"),
                                       ("model-response.json","test-model-response.json")):
                    if (path/source).exists():
                        atomic(path/target,(path/source).read_bytes())
            report = evaluate_suite(suite,intent,design)
            atomic(path/"optical-test-suite.json",encode(suite))
            atomic(path/"optical-test-report.json",encode(report))
            invalid = bool(report.get("error")) or any(t["status"] == "invalid" for t in report["tests"])
            if invalid:
                atomic(path/"validation.json",encode({"passed":False,"scope":"analytic_design_checks",
                                                       "physical_validated":False,"test_report":report}))
                self.fail("OPTICAL_TEST_INVALID",
                          "The experiment test definition is invalid and its basis must be corrected.", "act",
                          {"candidate_sha256":digest(encode(design)),"test_suite":suite,"test_report":report})
            try:
                review = design_agent.review(self.gateway(state,path),state["request"],intent,design,
                                             state.get("feedback", []),attempt,suite,report)
            except (CandidateFormatError,RetryableGatewayError) as exc:
                exc.details = {**(exc.details or {}),"failed_agent":"review",
                               "candidate_sha256":digest(encode(design))}
                raise
            review = {**review,"experiment_sha256":experiment["contract_sha256"],
                      "test_suite_sha256":digest(encode(suite)),"test_report_sha256":digest(encode(report))}
        if (not suite or not report
                or review.get("test_suite_sha256") != digest(encode(suite))
                or review.get("test_report_sha256") != digest(encode(report))
                or evaluate_suite(suite,intent,design) != report):
            self.fail("OPTICAL_TEST_BINDING_CHANGED", "The experiment tests and calculation report are not bound to the reviewed design.")
        atomic(path/"optical-test-suite.json",encode(suite))
        atomic(path/"optical-test-report.json",encode(report))
        atomic(path/"design-review.json",encode(review))
        validation = {"passed":review["accepted"] and report["passed"],"source":state.get("provider", "openai") + "_and_contract", "physical_validated":False,
                      "scope":"conceptual_review_and_analytic_tests", "issues":review["issues"],
                      "analytic_tests_passed":report["passed"],"deferred":report["deferred"],
                      "experiment_sha256":experiment["contract_sha256"]}
        atomic(path/"validation.json",encode(validation))
        if not review["accepted"]:
            self.fail("OPTICAL_TEST_INVALID" if review["problem_source"] == "tests" else "AGENT_DESIGN_REJECTED",
                      "；".join(review["issues"]),review["return_to"],
                      {**review,"candidate_sha256":digest(encode(design)),"test_suite":suite,"test_report":report})
        if not report["passed"]:
            self.fail("OPTICAL_TEST_FAILED", "The experiment-specific analytic tests failed. Return to Act and correct the design.", "act",
                      {"candidate_sha256":digest(encode(design)),"test_suite":suite,"test_report":report,
                       "review":review})
        decision = self.record("validator",state,path,attempt,"generate_and_run_tests",review["reasoning"],
                               changes={"test_count":len(report["tests"]),"deferred":report["deferred"],
                                        "test_suite_sha256":review["test_suite_sha256"]},
                               source=state.get("provider", "openai") if (path/"model-response.json").exists() else "verified_design_reuse")
        update = {"validation":validation,"algorithm_review":review,"algorithm_decisions":[decision],
                  "algorithm_test_suite":suite,"algorithm_test_report":report}
        if not state["algorithm_state"]["current"]:
            targets = poses(state["layout"])
            current = {}
            movable = []
            for i,c in enumerate(design["components"]):
                oid = c["object_id"]
                if c["movable"]:
                    movable.append(oid)
                    current[oid] = {"position_m":[.05 + .06*(i % 16),-.2 - .06*(i // 16),0.],"yaw_rad":0.}
                else:
                    current[oid] = deepcopy(targets[oid])
            world = {"current":current,"initial":deepcopy(current),"revision":0,"source":SOURCE,
                     "initial_source":"simulated_inventory_staging_fixed_fixtures_at_design_positions",
                     "seeded_object_id":"L25" if "L25" in movable else next(iter(movable),None)}
            atomic(path/"inventory.json",encode(world))
            update["algorithm_state"] = world
            update["prob"] = {"source":SOURCE,"camera_measured":False,"observations":current,
                              "frame_id":"optical_table","world_revision":0}
        return update

    def controller(self, state, path, attempt):
        world, target = state["algorithm_state"], poses(state["layout"])
        current = world["current"]
        assembly = world["revision"] == 0
        pending = state.get("algorithm_pending")
        if assembly:
            ids = [c["object_id"] for c in state["layout"]["components"] if c["movable"]]
            operation, source, reason = "place", "validated_design_compiler", "Compile the reviewed agent target into the initial assembly plan. This step does not create or change the design."
        elif pending:
            ids, operation = [pending["object_id"]], pending["operation"]
            source, reason = "validated_agent_proposal", pending["reason"]
        else:
            proposal, _ = self.local_repair("controller",state,path,attempt,"translate")
            ids, operation = [proposal["object_id"]], proposal["operation"]
            source, reason = state.get("provider", "openai"), proposal["reason"]
        movable = {c["object_id"] for c in state["layout"]["components"] if c["movable"]}
        if not set(ids) <= movable or (not assembly and len(ids) != 1):
            self.fail("REPAIR_SCOPE_UNPROVEN", "The plan exceeds the validated movable range.")
        actions = [{"object_id":oid,"operation":operation,
                    "delta_position_m":[float(t-c) if operation in {"translate","place"} else 0.
                                        for t,c in zip(target[oid]["position_m"],current[oid]["position_m"])],
                    "delta_yaw_rad":angle_delta(target[oid]["yaw_rad"],current[oid]["yaw_rad"])
                                    if operation in {"align","place"} else 0.} for oid in ids]
        plan = AlgorithmPlan.model_validate({"schema_version":SCHEMA,"source":SOURCE,
            "scene_revision":state["scene"]["scene_revision"],"frame_id":"optical_table","position_unit":"m","angle_unit":"rad",
            "world_revision":world["revision"],"current_sha256":digest(encode(current)),
            "target_sha256":digest(encode(state["layout"])),"assumptions_sha256":digest(encode(state["algorithm_options"])),
            "experiment_sha256":state["algorithm_experiment"]["contract_sha256"],"phase":"assembly" if assembly else "repair",
            "allowed_object_ids":ids,"actions":actions}).model_dump()
        decision = self.record("controller",state,path,attempt,"initial_assembly" if assembly else operation,reason,ids,
            {"actions":actions,"preserved_object_ids":[k for k in current if k not in ids],"design_changed":False},source)
        if self.injected(state,"controller",attempt):
            self.fail("ALGORITHM_PLAN_FORMAT","Test case: plan format failed. Return to Controller.","controller")
        raw = encode(plan)
        atomic(path/"plan.json",raw)
        atomic(path/"plan.sha256",digest(raw).encode())
        return {"plan":plan,"plan_path":str((path/"plan.json").relative_to(self.root)),"plan_sha256":digest(raw),
                "algorithm_pending":None,"algorithm_decisions":[decision]}
