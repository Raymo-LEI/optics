# Optics workbench

A LangGraph workbench that takes an optical-experiment request through design, test, simulated robot placement and acceptance, and returns failures to the stage the evidence points to. On top of it sits an agent layer: skills and MCP servers so Claude Code or Codex can operate runs, and a kinematic video tool for finished runs.

**Everything here is a simulation.** No camera, robot arm, hardware or real optical measurement is connected. The executor is a dry run (`motions_sent` is 0) and `physical_verified` is always false.

## How a run works

```
Input/Vision → Prompt → Act → Validator → Controller → Sim → Output → Mode → Execution → Done
```

- A run starts from a natural-language request (for example "Build a 3x Kepler beam expander using 50 mm and 150 mm positive lenses separated by 200 mm") or from the registered 15-component Michelson fixture.
- When a stage fails, `feedback_router` sends the run back to the node the evidence points to, within per-node retry budgets. In the demo a position failure returns to Controller and an angle failure to Act. The original failure, the decision, the plan, the simulation evidence and the acceptance result are all kept.
- Every stage writes a hashed journal, and the run can be paused, stopped and resumed.
- `algorithm_passed` means the workflow finished. It does **not** mean every requirement was verified. Check `core_unaccepted` and `deferred_physical` in the acceptance summary.

## Four ways to run it

| Kind | How it starts | Who decides |
| --- | --- | --- |
| Offline fixture | `start_run` (default), or `handoff_demo.py` | A rule policy. No model, no key. Not a model-designed experiment. |
| Agent-decided | `start_run` with `decision_mode="agent"` | The coding agent (Claude Code or Codex) answers each pending decision. Self-reported and unverified. |
| OpenAI | Web panel only | The OpenAI API with your own `OPENAI_API_KEY`. |
| Panel CLI decider | Web panel, "Decisions by: Claude Code" or "Codex CLI" | Your own signed-in `claude` or `codex` CLI, one fresh process per decision. Self-reported and unverified. |

The review decision always goes to a different model or a fresh context than the design decisions.

## Quick start

Needs macOS or Linux (Windows: WSL2) and Python 3.12 (`python3.12` on PATH; with uv: `uv python install 3.12` then `uv venv --python 3.12 .venv`).

```sh
tools/print-codex-config.sh > .codex/config.toml   # from the repo root, once per machine; the tests check it
cd handover-review/sanitized/optics-mechanical-ai-handoff/optics-langgraph
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt -r requirements-mcp.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest -q          # offline, no API key needed; expect no failures
.venv/bin/python handoff_demo.py       # offline fixture closed loop
.venv/bin/python -m optics_graph.cli serve --port 8766   # web panel at http://127.0.0.1:8766
```

The demo should show an L25 position failure returning to Controller, an angle failure returning to Act, three algorithm rounds, the same plan bytes handed off, and `algorithm_passed`.

Cloud-model runs need your own `OPENAI_API_KEY` in the environment or in `config/local.json` (git-ignored). Never commit either.

## Using it with Claude Code or Codex

Open the repo root in the agent. `.mcp.json` (Claude Code) and `.codex/config.toml` (generated above, for Codex) register two stdio MCP servers:

- `optics-evidence` is read-only: list and inspect runs, feedback logs, stage journals and artifacts, check execution status, re-verify journals against their hashes, and read the acceptance summary.
- `optics-runs` starts and controls runs it created itself (in `data-mcp/`, never the panel's `data/`): start, continue, pause, stop, read the pending decision, submit a decision. It cannot approve a review gate or change settings, retries, tolerances or plans.

Skills (in `.claude/skills/`, also visible to Codex through the `.agents/skills` symlink):

| Skill | Use it for |
| --- | --- |
| `optics-run-operator` | Start, monitor, pause, stop and report an offline fixture run |
| `optics-agent-decide` | Run end to end with the agent making the decisions; uses the five role skills `optics-decide-interpret`, `-design`, `-tests`, `-review`, `-repair` |
| `optics-failure-triage` | A run failed, stopped or was rerouted: classify, name the owner, cite evidence |
| `optics-execution-unknown` | Execution status is unknown: query and verify, never retry |

The tests and review decisions are made by a fresh-context helper: the `optics-independent-reviewer` agent (`.claude/agents/`, and `.codex/agents/` for Codex). `AGENTS.md` is the binding operator contract for any agent working here. Regenerate `.codex/config.toml` for another machine or checkout with `tools/print-codex-config.sh`, and set `OPTICS_PROJ` if the workbench is somewhere else. `mcp-panel.example.json` shows how to point a client at the web panel instead.

## Repository layout

| Path | What it is |
| --- | --- |
| `handover-review/sanitized/optics-mechanical-ai-handoff/` | The handoff package. Read `START_HERE.txt` and `INTERFACES_AND_TASKS.txt` first; `OPTIMIZATION_GOALS.txt` is the task brief; `schemas/`, `examples/` and `references/` hold the contracts, sample artifacts and the pinned reference flow. |
| `.../optics-langgraph/optics_graph/` | Workbench source: `runtime.py` (runs, persistence), `team_flow.py` (the graph, routing, retry budgets, journal), `algorithm_sim.py` (plans and the simulated executor), `agent_algorithm.py` and `design_agent.py` (agents), `agent_decisions.py` and `cli_decider.py` (agent and CLI deciders), `mcp_evidence.py` and `mcp_runs.py` (MCP servers), `server.py` and `panel.html` (web panel) |
| `.../optics-langgraph/tests/` | About 500 offline tests. `START_HERE.txt` section 4 lists the required regression subset. |
| `.claude/`, `.codex/`, `.agents/`, `.mcp.json` | Skills, reviewer agents and MCP client configs |
| `tools/visualize_run/` | Renders a finished run as a kinematic MuJoCo video of an xArm7 placing the components. See its README. |
| `tools/print-codex-config.sh` | Writes the machine-local Codex MCP config |
| `AGENTS.md` | Rules and workflow for agents working in this repo |

Suggested reading order for the code is in `INTERFACES_AND_TASKS.txt`, section B.

## Rendering a run

After a run exists under `data-mcp/runs/<run_id>`, `tools/visualize_run` turns it into an MP4 or GIF. The video illustrates recorded poses only; it is not physics, not collision-checked and not a robot controller. Setup and options are in `tools/visualize_run/README.md`. Run data and rendered output are not part of the repository.

## What is and is not done

Works: requirements and design agents, generated analytic tests, independent review, incremental algorithm plans, simulated execution with error injection, layered rollback, same-plan acceptance, and a web record of each run.

Not done and not to be claimed: a calibrated camera, arm reachability or collision checking, a real ROSi or ROS 2 trajectory, an HTTP job service, or a passed real optical measurement. A skip caused by a missing external environment is not a pass. Open work and ownership are in `INTERFACES_AND_TASKS.txt`, sections E to G.

## Rules for contributors

- Do not edit anything under `data/` or `data-mcp/`, or any run evidence, to make a test pass.
- Never loosen a tolerance or change an optical target to make a run pass.
- Never commit API keys, `config/local.json` or run data.
- Do not approve a review gate from an agent; that is for a person in the web panel.

## Third-party pieces

- `optics_graph/vendor/haloflux/` is HALOFLUX by Ryder191 (PolyForm Noncommercial 1.0.0), used as a decorative panel overlay. `optics.html` is a modified copy. See `NOTICE.txt` there.
- `tools/visualize_run/assets/ufactory_xarm7/` is the UFACTORY xArm7 model (BSD-3-Clause wording); its license is in that folder.

See `THIRD_PARTY_NOTICES.md` for details.

## License

PolyForm Noncommercial License 1.0.0, see `LICENSE`. Noncommercial use only: personal, research, teaching and noncommercial organizations. Use inside a company or for paid or client work is not permitted. Bundled third-party pieces keep their own licenses, listed in `THIRD_PARTY_NOTICES.md`.
