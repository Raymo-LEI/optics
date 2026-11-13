# Optics agent layer

Agent skills, MCP servers, the web panel changes, and a kinematic run visualizer for a local optics workbench. Everything here is a simulation. No camera, robot, or optical measurement is connected, and `physical_verified` is false.

The received handoff package is not in this repository. This tree keeps only the files added or edited for the agent layer. API keys, `config/local.json`, virtual environments, and run records (`data/`, `data-mcp/`, `demo-output/`) stay on the machine that produced them.

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
| `handover-review/.../optics-langgraph/` | Only the edited workbench files: MCP servers, decision loop, panel, tests |

`AGENTS.md` is the operator contract for an agent working in a full local checkout. `algorithm_passed` means the workflow finished. It does not mean every requirement was verified.

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
