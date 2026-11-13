# Optics workbench

A local optics LangGraph workbench (the received handoff package) plus an agent layer on top of it: skills, MCP servers, web panel changes, and a kinematic run visualizer. Everything here is a simulation. No camera, robot, or optical measurement is connected, and `physical_verified` is false.

This repository is runnable from a clone. API keys, `config/local.json`, virtual environments, run records (`data/`, `data-mcp/`, `demo-output/`) and the original handoff ZIP stay on the machine that produced them. See License below before using it outside noncommercial work.

## Quick start

Needs macOS or Linux (Windows: WSL2) and Python 3.12 (`python3.12` on PATH; with uv, `uv python install 3.12` then `uv venv --python 3.12 .venv`).

```sh
tools/print-codex-config.sh > .codex/config.toml   # from the repo root, once per machine; the tests check it
cd handover-review/sanitized/optics-mechanical-ai-handoff/optics-langgraph
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt -r requirements-mcp.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest -q          # offline, no API key needed; expect no failures
.venv/bin/python handoff_demo.py       # offline fixture closed loop
.venv/bin/python -m optics_graph.cli serve --port 8766   # web panel
```

Then read `handover-review/sanitized/optics-mechanical-ai-handoff/START_HERE.txt` and `INTERFACES_AND_TASKS.txt`. Cloud-model runs need your own `OPENAI_API_KEY`; never commit it.

## Layout

| Path | What it is |
| --- | --- |
| `.claude/skills/` | Skills for operating a run and for each decision role |
| `.claude/agents/` | Fresh-context reviewer for Claude Code |
| `.codex/agents/` | The same reviewer for Codex |
| `.agents/skills` | Symlink so Codex can see the Claude skills |
| `.mcp.json` | Claude Code MCP client config. Paths are relative |
| `mcp-panel.example.json` | Example client config for the local web panel |
| `tools/print-codex-config.sh` | Writes a machine-local `.codex/config.toml` |
| `tools/visualize_run/` | Kinematic video of a finished run. Not a robot controller |
| `handover-review/sanitized/optics-mechanical-ai-handoff/` | The handoff package: docs, contracts, schemas, and `optics-langgraph/` (workbench source, MCP servers, decision loop, panel, tests) |

`AGENTS.md` is the operator contract for an agent working in this checkout. `algorithm_passed` means the workflow finished. It does not mean every requirement was verified.

## Codex on another machine

`.codex/config.toml` is not committed. It contains absolute paths. From the repository root:

```sh
tools/print-codex-config.sh > .codex/config.toml
```

Set `OPTICS_PROJ` to the workbench directory if it is not in the default location.

## Third-party pieces

- `optics_graph/vendor/haloflux/` is HALOFLUX by Ryder191, PolyForm Noncommercial 1.0.0. `index.html` and `wp.js` are the upstream wallpaper. `optics.html` is the modified copy used as a decorative panel overlay. See `NOTICE.txt` in that folder. Noncommercial use only.
- `tools/visualize_run/assets/ufactory_xarm7/` is the UFACTORY xArm7 model. Its license is in that folder.

The wallpaper's own control labels are the upstream text. Workbench-facing prose in this repository is English.

## License

PolyForm Noncommercial License 1.0.0, see `LICENSE`. Noncommercial use only: personal, research, teaching and noncommercial organizations. Use inside a company or for paid or client work is not permitted. Bundled third-party pieces keep their own licenses, listed in `THIRD_PARTY_NOTICES.md`.
