# Optics workbench: agent instructions

Read `handover-review/sanitized/optics-mechanical-ai-handoff/START_HERE.txt` and `INTERFACES_AND_TASKS.txt` first. They are binding.

## What this repo is
A handoff package for an optics LangGraph workbench (`handover-review/sanitized/optics-mechanical-ai-handoff/optics-langgraph`, called PROJ below). A run goes Input/Vision, Prompt, Act, Validator, Controller, Sim, Output, Mode, Execution, Done; failures return to the node the evidence points to, within retry budgets. Everything is a simulation: the executor is a dry run (`motions_sent` 0). No real hardware, camera, robot arm or real optical measurement is involved.

## Setup
```
cd handover-review/sanitized/optics-mechanical-ai-handoff/optics-langgraph
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt -r requirements-mcp.txt
.venv/bin/python -m pip install -e . --no-deps
```
Use `PROJ/.venv/bin/python` for everything. Tests: `cd PROJ && .venv/bin/python -m pytest -q`.

## MCP servers (stdio; `.mcp.json` for Claude Code, `.codex/config.toml` for Codex)
- `optics-evidence`: READ-ONLY. `list_runs`, `get_run`, `get_feedback_log`, `get_stage_journal`, `get_stage_artifact`, `get_execution_status`, `verify_run_artifacts`, `get_acceptance_summary`. Re-verifies journals against hashes.
- `optics-runs`: starts and controls runs. `start_run`, `continue_run`, `pause_run`, `stop_run`, `get_worker_error`, `get_pending_decision`, `submit_decision`. It acts only on runs it created (its own `data-mcp/` directory, never the web panel's `data/`). There is no tool to approve a review, change settings, retries, tolerances or plans.
- Regenerate the Codex config for another machine: `tools/print-codex-config.sh > .codex/config.toml`.

## Four kinds of run
1. Offline fixture: `start_run` default (`decision_mode` "fixture"). Provider `replay`, no model, rule policy. Labelled "offline rule fixture, not a model-designed experiment".
2. Agent-decided: `start_run` with `decision_mode="agent"`. Provider `agent`, no model call by the server; the run stops at `awaiting_decision` and the coding agent answers each request. Labelled "agent-decided run ... decider identity is self-reported and unverified".
3. OpenAI: created only in the web panel (`python -m optics_graph.cli serve --port 8766`), provider `openai`, needs the user's own key. Not controllable through the MCP tools. Do not start these.
4. Panel CLI decider: created only in the web panel with "Decisions by: Claude Code" or "Codex CLI". An agent-decided run (provider `agent`) whose decisions the server hands to the user's own signed-in CLI: one fresh `claude -p` (no tools, `--safe-mode`, no MCP servers; never `--bare`, which needs an API key) or `codex exec` (read-only sandbox, user config ignored) per decision, with no API key in its environment (`optics_graph/cli_decider.py`; role guides in `optics_graph/decider_guides/`, kept equal to the role skills). The review goes to a different model than the other decisions (Claude: sonnet decides, opus reviews; Codex: gpt-6-astra reviews). Label `<host>/<model>:ctx-fresh`, self-reported and unverified. Settings `OPTICS_CLAUDE_BIN`, `OPTICS_CLAUDE_MODEL`, `OPTICS_CLAUDE_REVIEW_MODEL`, `OPTICS_CLAUDE_EFFORT` and the same four `OPTICS_CODEX_*` names. Subscription sign-in is for the user's own local use; anything offered to other people must use API-key auth. The user starts these; do not start them yourself.

## Which skill (`.claude/skills/`, also visible to Codex as `.agents/skills/`)
- `optics-run-operator`: start, monitor, pause, stop, resume, report a fixture run.
- `optics-agent-decide`: run end to end with you making the decisions. It uses `optics-decide-interpret`, `optics-decide-design`, `optics-decide-tests`, `optics-decide-review`, `optics-decide-repair`.
- `optics-failure-triage`: a run failed, stopped or was rerouted.
- `optics-execution-unknown`: `get_execution_status` is `unknown` or ALGORITHM_EXECUTION_UNKNOWN.
- Fresh-context helper for the tests and review decisions: Claude Code agent `optics-independent-reviewer` (`.claude/agents/`), Codex agent `optics_independent_reviewer` (`.codex/agents/`), otherwise a separate new session.

## Hard rules
- Never write or edit files under `data/` or `data-mcp/`, or any run evidence. Do not edit evidence to make a test pass.
- Never loosen a tolerance or change an optical target. Omit tolerances to get defaults unless the request states a value with a unit.
- Never author a test whose tolerance is looser than the requirement it checks. Test expected values come from the request or a stated physical law, never from the candidate design.
- The `optical_tests` and `review` decisions are made in a fresh context that sees only the pending request.
- Never approve a review gate (`awaiting_review` is for a human in the web panel). Never retry an execution with unknown status.
- Never set or read `OPENAI_API_KEY` or `PROJ/config/local.json`. Never call a cloud model yourself (the panel's CLI decider runs call one under the user's own login, started by the user).
- Do not commit run data (`data/`, `data-mcp/`), keys or `config/local.json`.

## What must never be claimed
- No real hardware was used and `physical_verified` is false.
- `algorithm_passed` ("workflow complete") is NOT "all requirements verified". Report `core_unaccepted` and `deferred_physical` from `get_acceptance_summary`, and `verify_run_artifacts` must be ok.
- Agent-decided runs: the decider is self-reported and unverified.
- A skip caused by a missing external environment is not a pass.

## Layout
- `PROJ/optics_graph/`: source (`runtime.py`, `team_flow.py`, `agent_decisions.py`, `mcp_runs.py`, `mcp_evidence.py`).
- `PROJ/tests/`: tests. `tests/test_skills_conformance.py` keeps the skills, this file and the Codex config in line with the tools.
- `START_HERE.txt` section 4 lists the required offline regression subset.
