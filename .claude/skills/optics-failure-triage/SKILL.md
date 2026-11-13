---
name: optics-failure-triage
description: Use when an optics workbench run has failed, stopped, or been rerouted by the feedback router and you need to classify the failure, name the owning node and owner, and cite evidence. Read-only analysis through the optics-evidence MCP server.
---

# Optics failure triage

1. `get_run(run_id)`: record `status`, `stage`, `attempts`, `retry_counts`, `error` (code, stage, requested_return) and `world_revision`.
2. `get_feedback_log(run_id)`: read every router decision and its original failure. The original failures come from the stage journals; if `consistent` is false or `issues` is non-empty, run.json disagrees with the journals: report that and trust the journal text. Keep the original errors verbatim; note whether the final decision was `stop` and its `reason` (route_not_allowed, target_retry_limit, total_retry_limit).
3. `get_stage_journal(run_id, stage, attempt)` for the failing stage and the stage it returned to: confirm `verified` is true, read the event and file hashes. Use `get_stage_artifact` (whitelisted files only) when the details are needed. If `verified` is false, stop and report tampering or a corrupt journal instead of interpreting it. Exception: for an unknown-execution case (`ALGORITHM_EXECUTION_UNKNOWN`, or `get_execution_status` = `unknown`) an incomplete or error-carrying execution journal is expected; do not treat that alone as tampering, switch to `optics-execution-unknown`, and still stop on any other unverified journal.
4. `verify_run_artifacts(run_id)`: include its result and `issues`. A legitimately failed run should verify `ok` (internally consistent) while still being failed; `ok` does not mean the experiment succeeded. A reroute by the feedback router is not a recovery: check the final `status`.
5. Classify and name the owner (INTERFACES_AND_TASKS section G; the optics owner owns targets and optical acceptance, the mechanical and AI owner owns service adaptation, recovery and fault tests, the control group owns low-level motion and hardware constraints):
   - missing observation (vision, SYNTHETIC_OBSERVATION_MISSING): returns to input; owner mechanical and AI.
   - unreachable target: evidence only; the design node (Act) and the optics owner handle it.
   - planning failure (controller, ALGORITHM_PLAN_FORMAT): Controller; mechanical and AI.
   - pre-execution simulation failure (sim, ALGORITHM_SERVICE_TIMEOUT or ALGORITHM_ARTIFACT_REJECTED): sim/service or artifact stage; mechanical and AI, control group for hardware limits.
   - placement out of tolerance (output, ALGORITHM_POSITION_MISMATCH to controller, ALGORITHM_OPTICAL_MISMATCH to act): the single component named in `target_object_ids`; mechanical and AI for the retry rule, optics owner for the angle target.
   - unknown execution (ALGORITHM_EXECUTION_UNKNOWN): switch to `optics-execution-unknown`.
   - model/service (MODEL_SERVICE_UNAVAILABLE, gateway errors): in-graph policy; mechanical and AI. Neither the offline fixture nor an agent-decided run (`provider` agent) ever calls a model.
   - agent-decided run (`provider` agent): AGENT_DECISION_INVALID (three submitted answers for one request were rejected), DECISION_RESPONSE_INVALID or DECISION_REQUEST_CHANGED (stored decision files do not match: treat as tampering or a corrupt journal and report it), AGENT_DESIGN_CONTRACT, AGENT_RECOVERY_INVALID, OPTICAL_TEST_INVALID, OPTICAL_TEST_FAILED (the decider's answer failed the graph's own checks; the decider label in `get_acceptance_summary` is self-reported and unverified). Never fix by editing the decision files.
6. Report: classification, owning node, owner, evidence (run_id, stage-attempt, file sha256, plan_sha256, artifact_sha256) and the original error text.
7. Never propose loosening a tolerance, raising the retry budget quietly, or changing an optical target. Never edit run data. Suggest only what the owner should examine.
