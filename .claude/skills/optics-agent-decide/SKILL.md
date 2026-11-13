---
name: optics-agent-decide
description: Use when asked to run the optics workbench with the agent making the in-graph decisions (interpret, design, optical tests, review, repair) through the optics-runs MCP server, with no OpenAI key and no model call by the server. Orchestrates the whole run; role skills do the answering.
---

# Optics agent-decided run

The server makes no model call. When a node needs a decision it stops the run at `awaiting_decision`. You answer the pending request, then resume. Everything after the answer (validators, plan compile, simulation, output checks, feedback router, retry budgets, the review gate) is unchanged server code. The pending request carries the authoritative instructions, data and output schema, bound by hash. If this skill and the request's instructions conflict, the request wins.

## Loop
1. `start_run(client_request_id, request, decision_mode="agent")`. Use a fresh descriptive `client_request_id`; the same id with a different body is an error. Do not pass `team_fault` or `algorithm_options` unless the user asks. Note `source` in the answer: the run is "agent-decided, decider self-reported and unverified".
2. Poll `get_worker_error(run_id)` until `active` is false; wait a few seconds between polls after `continue_run` (there is no blocking wait). A worker is active while it executes; `submit_decision` is refused while it is. Read `status` (also `get_run` from optics-evidence).
3. If `status` is `awaiting_decision`: `get_pending_decision(run_id)`. Read `role`, `instructions`, `data`, `output_schema`, `request_id`, `rejections_remaining`. Dispatch on `role`:
   - `interpret` -> skill `optics-decide-interpret`
   - `design` -> `optics-decide-design`
   - `optical_tests` -> `optics-decide-tests` (FRESH CONTEXT, see below)
   - `review` -> `optics-decide-review` (FRESH CONTEXT)
   - `repair_act`, `repair_controller` -> `optics-decide-repair`
4. `submit_decision(run_id, request_id, decision_json, decider_label)`. `decision_json` is a JSON STRING holding one object (no prose, no code fence, no duplicate keys, no NaN). `decider_label` format `<host>/<model>:ctx-<main|fresh>`, for example `claude-code/opus:ctx-main` or `codex/gpt:ctx-fresh`. Allowed characters are letters, digits, `_ . : / space -` (so no `;` or `=`), 1 to 128 long. State the host, model and context honestly; it is self-reported and unverified.
5. If the answer has `accepted: false`, read `errors`, fix exactly those, resubmit. `rejections_remaining` counts down from 3 per request. After the third rejection `exhausted` is true and further submits fail with DECISION_EXHAUSTED: call `continue_run` anyway; the graph records AGENT_DECISION_INVALID, charges its retry budget and asks again with a new request. Do not guess blindly: every rejection costs budget.
6. If accepted (`idempotent_replay` true means the identical answer was already stored): `continue_run(run_id)`, then back to step 2. Repeat until the run is not `awaiting_decision`.

## What to expect
- By default the algorithm simulator seeds one misplaced component (a position offset of 0.003 m in x and a yaw offset of 2 degrees; `AlgorithmOptions` defaults in `algorithm_sim.py`; it picks `L25` if movable, else the first movable component). So every run includes repair decisions after the design is accepted. This is the simulated assembly error, not a design fault.
- Each correction removes only part of that error: a `translate` applies 0.5 of the commanded position change (`position_response`), an `align` applies 0.8 of the commanded angle change (`angle_response`). Expect more than one repair on the same component. Position problems come back as `repair_controller`, angle problems as `repair_act`.
- Retry budgets (`team_flow.py`): 2 retries per return node, 7 charged retries in total; past either the router stops the run. A placement repair of one part (Output's position or angle mismatch) counts under its own node-and-part budget, for example `act:laser1`, so a rejected test suite (which charges Act itself) does not use up that part's repairs. Every rejected or invalid answer can charge them.

## Fresh context for `optical_tests` and `review`
These two MUST be decided by a context that sees only the `get_pending_decision` record, never your design reasoning, so the tester and reviewer are independent of the designer. Use a fresh-context subagent if your host supports one (Claude Code: the `optics-independent-reviewer` agent; Codex: the `optics_independent_reviewer` agent), otherwise a separate new session. Give it only the `run_id` and which role is pending. It calls `get_pending_decision` and `submit_decision` itself with `ctx-fresh` in the label and reports the outcome. Use a new helper for each of the two decisions (the reviewer must not be the tester). You then check `get_pending_decision` (`submitted` true, or the helper reported the rejection errors) and call `continue_run`. Do not paste the design, intent or your own opinion into the helper.

## Other statuses
- `interrupted`: call `get_execution_status(run_id)` (optics-evidence) first and follow `optics-run-operator` step 3. `unknown` means stop and use `optics-execution-unknown`; never retry an execution whose result is unknown.
- `awaiting_review`: STOP. Report `plan_sha256` from `get_run`. A human approves in the web panel. No tool can approve and you must not try.
- `paused` or `stopped`: report it; resume only if the user asks (`continue_run` on `paused`).
- `failed`: hand to `optics-failure-triage` with the run_id.
- DECISION_STALE from `submit_decision`: the request you answered is not the current one; call `get_pending_decision` again and answer that.
- DECISION_ALREADY_SUBMITTED: a different answer is already stored for this request; do not try to overwrite it, call `continue_run`.
- DECISION_REQUEST_CHANGED or DECISION_RESPONSE_INVALID: evidence was modified; stop and report, do not work around it.
- `stop_run(run_id)` works at `awaiting_decision`; afterwards submits are refused. `pause_run` is refused there.

## End of run
Call `get_acceptance_summary(run_id)` and `verify_run_artifacts(run_id)` and report honestly:
- "agent-decided; decider labels self-reported and unverified" (`provenance`, `decider_labels`), and any `same_label_warning` (design and review shared a label).
- Workflow complete (`algorithm_passed` with `verify_run_artifacts` ok) is NOT "all requirements verified". List `core_unaccepted` and `deferred_physical`; `physical_verified` is false; no real hardware moved (`motions_sent` 0).
- `execution_limits`: the limits used and whether `equals_defaults`.
- `optical_tests.largest_tolerance_relative_to_expected` (the widest test tolerance relative to its expected value).
- `rerouted_by_router` and the final `raw_status`; a reroute is not a recovery.
- Optional, only for a finished run and only if `tools/visualize_run/render_run.py` and `tools/visualize_run/.venv` exist: render it with `tools/visualize_run/.venv/bin/python tools/visualize_run/render_run.py --run-dir handover-review/sanitized/optics-mechanical-ai-handoff/optics-langgraph/data-mcp/runs/<run_id> --out demo-output/<run_id>.mp4` and tell the user the file path, stating it is a kinematic visualization of the recorded poses, not a physics simulation (read-only on the run; not evidence).

## Hard rules
- Never write or edit any file under a data directory (`data/`, `data-mcp/`) or any run file. Answer only through `submit_decision`.
- Never loosen a tolerance or change an optical target. Omit tolerances (null) to get the defaults unless the user's request states a value with a unit.
- Never author a test whose tolerance is looser than the requirement it checks.
- Never retry an execution with unknown status. Never try to approve a review.
- Never claim real hardware, physical verification, or that every requirement was verified.
