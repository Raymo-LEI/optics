"""Cheap guards so the agent-facing layer (skills, agents, AGENTS.md, Codex config) cannot drift from the tools.

The layer lives at the repository root, above this package: ROOT/.claude/skills, ROOT/.claude/agents,
ROOT/.codex, ROOT/.agents, ROOT/AGENTS.md. When it is absent (the package tested on its own) every test skips.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from optics_graph import mcp_evidence, mcp_runs

PROJ = Path(__file__).resolve().parents[1]


def _find_root():
    for parent in PROJ.parents[:6]:
        if (parent / ".claude" / "skills").is_dir():
            return parent
    return None


ROOT = _find_root()
pytestmark = pytest.mark.skipif(ROOT is None, reason="no .claude/skills directory above the package: "
                                "the agent-facing layer is not part of this checkout")
SKILLS = ROOT / ".claude" / "skills" if ROOT else None

# Names that look like calls but belong to the optical-test expression grammar, not to MCP tools.
GRAMMAR = {"distance", "angle", "coordinate", "yaw", "parameter", "count", "connected", "path_length",
           "abs", "sqrt", "sin", "cos"}
# Runtime paths that need not exist in a fresh checkout (gitignored data and keys, placeholders).
RUNTIME_PATHS = {"data/", "data-mcp/", "config/local.json", "PROJ", "PROJ/", ".venv/bin/python", ".venv"}
SERVERS = {"optics-evidence", "optics-runs"}


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    data = tmp_path_factory.mktemp("conformance")
    found = {}
    for server in (mcp_evidence.build_server(data), mcp_runs.build_server(data)):
        for registered in server._tool_manager.list_tools():
            found[registered.name] = {"server": server.name, "params": set(registered.parameters.get("properties", {}))}
    return found


def source_text():
    parts = [p.read_text("utf-8") for p in sorted((PROJ / "optics_graph").glob("*.py"))]
    parts.append((PROJ / "tests" / "agent_canned.py").read_text("utf-8"))
    return "\n".join(parts)


def docs():
    """(label, text) for every skill, the agent definitions and AGENTS.md."""
    out = [(f"skill {p.parent.name}", p.read_text("utf-8")) for p in sorted(SKILLS.glob("*/SKILL.md"))]
    out += [(f"agent {p.name}", p.read_text("utf-8")) for p in sorted((ROOT / ".claude" / "agents").glob("*.md"))]
    out += [(f"codex agent {p.name}", p.read_text("utf-8")) for p in sorted((ROOT / ".codex" / "agents").glob("*.toml"))]
    out.append(("AGENTS.md", (ROOT / "AGENTS.md").read_text("utf-8")))
    return out


def backticked(text):
    return re.findall(r"`([^`\n]+)`", text)


def frontmatter(text):
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, "no YAML frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        assert sep and key.strip() and not key.startswith(" "), f"unparseable frontmatter line {line!r}"
        fields[key.strip()] = value.strip()
    return fields


def split_args(text):
    args, depth, current, quote = [], 0, "", None
    for ch in text:
        if quote:
            current += ch
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
            current += ch
        elif ch in "([{":
            depth += 1
            current += ch
        elif ch in ")]}":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    return args


# ---- skills ---------------------------------------------------------------------------------------------------
def test_every_skill_has_valid_frontmatter():
    folders = sorted(p for p in SKILLS.iterdir() if p.is_dir())
    assert folders, "no skills found"
    for folder in folders:
        skill = folder / "SKILL.md"
        assert skill.is_file(), f"{folder.name} has no SKILL.md"
        fields = frontmatter(skill.read_text("utf-8"))
        assert set(fields) == {"name", "description"}, (folder.name, sorted(fields))
        assert fields["name"] == folder.name
        assert fields["description"].strip(), folder.name


def test_expected_skills_exist():
    expected = {"optics-run-operator", "optics-failure-triage", "optics-execution-unknown", "optics-agent-decide",
                "optics-decide-interpret", "optics-decide-design", "optics-decide-tests", "optics-decide-review",
                "optics-decide-repair"}
    assert expected <= {p.name for p in SKILLS.iterdir() if p.is_dir()}


def test_every_role_skill_defers_to_the_request():
    for name in ("optics-decide-interpret", "optics-decide-design", "optics-decide-tests", "optics-decide-review",
                 "optics-decide-repair", "optics-agent-decide"):
        text = (SKILLS / name / "SKILL.md").read_text("utf-8")
        assert "the request wins" in text or "request's instructions conflict, the request wins" in text, name


def test_fresh_context_roles_say_so():
    for name in ("optics-decide-tests", "optics-decide-review"):
        text = (SKILLS / name / "SKILL.md").read_text("utf-8")
        assert "fresh context" in text.lower() and "ctx-fresh" in text, name
    orchestrator = (SKILLS / "optics-agent-decide" / "SKILL.md").read_text("utf-8")
    assert "optics-independent-reviewer" in orchestrator and "optics_independent_reviewer" in orchestrator


def test_mentioned_tools_exist_with_the_cited_parameters(tools):
    tool_like = re.compile(r"(get|submit|start|continue|pause|stop|list|verify)_[a-z_]+")
    for label, text in docs():
        for token in backticked(text):
            call = re.fullmatch(r"([a-z][a-z_]*)\((.*)\)", token)
            name = call.group(1) if call else token
            if name in GRAMMAR:
                continue
            if not (call or tool_like.fullmatch(name)):
                continue
            if name in {"stop_requested", "pause_requested"}:  # display_state values, not tools
                continue
            assert name in tools, f"{label}: `{token}` names a tool that does not exist"
            if call:
                for arg in split_args(call.group(2)):
                    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(=|$)", arg)
                    if match:  # a bare quoted value such as "execution" is not a parameter name
                        assert match.group(1) in tools[name]["params"], (
                            f"{label}: `{token}` cites parameter {match.group(1)!r}; real parameters: "
                            f"{sorted(tools[name]['params'])}")


def test_agent_decision_tool_protocol_is_cited_correctly(tools):
    assert tools["submit_decision"]["params"] == {"run_id", "request_id", "decision_json", "decider_label"}
    assert tools["get_pending_decision"]["params"] == {"run_id"}
    assert {"decision_mode", "client_request_id", "request"} <= tools["start_run"]["params"]
    text = (SKILLS / "optics-agent-decide" / "SKILL.md").read_text("utf-8")
    for needed in ("start_run(client_request_id, request, decision_mode=\"agent\")", "get_pending_decision(run_id)",
                   "submit_decision(run_id, request_id, decision_json, decider_label)", "continue_run(run_id)",
                   "get_worker_error(run_id)", "get_acceptance_summary(run_id)", "verify_run_artifacts(run_id)"):
        assert needed in text, needed


def test_cited_codes_statuses_and_fields_exist_in_the_source(tools):
    source = source_text()
    known = set(tools) | {p for t in tools.values() for p in t["params"]}
    agent_names = {p.stem for p in (ROOT / ".codex" / "agents").glob("*.toml")}
    unresolved = []
    for label, text in docs():
        for code in set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text)):
            if (PROJ.parent / f"{code}.txt").is_file():
                continue  # a binding project document named without its extension
            if code not in {"OPENAI_API_KEY", "OPTICS_MCP_DATA_DIR", "OPTICS_PROJ"} and code not in source:
                unresolved.append((label, code))
        for token in backticked(text):
            if re.search(r"\.(md|txt|json|toml|sh|py)$|/", token) or token in agent_names:
                continue  # files and paths are checked elsewhere; agent names are checked against the agent files
            for part in token.split("."):
                if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", part) and part not in known:
                    if not re.search(rf"\b{re.escape(part)}\b", source):
                        unresolved.append((label, part))
    assert not unresolved, f"cited but not found in optics_graph source: {sorted(set(unresolved))}"


def test_cited_statuses_exist_in_the_source():
    source = source_text()
    for status in ("awaiting_decision", "awaiting_review", "algorithm_passed", "interrupted", "paused", "stopped",
                   "failed", "not_executing_here", "waiting_for_worker"):
        assert f'"{status}"' in source, status
    for code in ("DECISION_STALE", "DECISION_ALREADY_SUBMITTED", "DECISION_EXHAUSTED", "AGENT_DECISION_INVALID",
                 "DECISION_RESPONSE_INVALID", "DECISION_REQUEST_CHANGED"):
        assert f'"{code}' in source, code


def test_decider_label_format_in_the_skills_is_accepted_by_the_server():
    from optics_graph import agent_decisions
    for label in ("claude-code/opus:ctx-main", "codex/gpt:ctx-fresh", "claude-code/opus:ctx-fresh"):
        assert agent_decisions.LABEL_RE.fullmatch(label), label
    text = "\n".join(t for _, t in docs())
    assert "ctx=" not in text and ";ctx" not in text  # ';' and '=' are rejected by the server's label pattern


def test_skill_and_agent_names_referenced_exist():
    skills = {p.name for p in SKILLS.iterdir() if p.is_dir()}
    agents = {p.stem for p in (ROOT / ".claude" / "agents").glob("*.md")} | {
        p.stem for p in (ROOT / ".codex" / "agents").glob("*.toml")}
    for label, text in docs():
        for token in backticked(text):
            if re.fullmatch(r"optics[-_][a-z_-]+", token) and not token.startswith("optics_graph"):
                assert token in skills | agents | SERVERS, f"{label}: `{token}` is not a skill, agent or server"


# ---- agents ---------------------------------------------------------------------------------------------------
def test_claude_agent_definition(tools):
    path = ROOT / ".claude" / "agents" / "optics-independent-reviewer.md"
    fields = frontmatter(path.read_text("utf-8"))
    assert fields["name"] == path.stem and fields["description"].strip()
    listed = [t.strip() for t in fields["tools"].split(",")]
    assert set(listed) == {"mcp__optics-runs__get_pending_decision", "mcp__optics-runs__submit_decision"}
    for item in listed:
        _, server, name = item.split("__", 2)
        assert server in SERVERS and tools[name]["server"] == server


def test_codex_agent_definition():
    tomllib = pytest.importorskip("tomllib")
    path = ROOT / ".codex" / "agents" / "optics_independent_reviewer.toml"
    data = tomllib.loads(path.read_text("utf-8"))
    assert data["name"] == path.stem and data["description"].strip() and data["developer_instructions"].strip()
    claude = (ROOT / ".claude" / "agents" / "optics-independent-reviewer.md").read_text("utf-8").split("---\n", 2)[2]
    assert data["developer_instructions"].strip() == claude.strip()


# ---- AGENTS.md ------------------------------------------------------------------------------------------------
def test_agents_md_size_paths_and_tools(tools):
    path = ROOT / "AGENTS.md"
    text = path.read_text("utf-8")
    assert len(text.encode()) < 32 * 1024 and len(text.splitlines()) < 150
    bases = [ROOT, PROJ, PROJ.parent, PROJ / "optics_graph"]
    missing = []
    for token in backticked(text) + re.findall(r"^[ ]*(?:cd|python3\.12|\.venv)[^\n]*$", text, re.M):
        for word in token.split():
            word = word.rstrip(",.;:")
            if word.startswith("PROJ/"):
                word = word[len("PROJ/"):]
                bases_for = [PROJ]
            else:
                bases_for = bases
            pathlike = ("/" in word or re.search(r"\.(md|json|toml|txt|sh|py)$", word)) and not re.search(r"[<>*]|^http|\.\.\.", word)
            if not pathlike or word in RUNTIME_PATHS or word.startswith("optics_graph.") or "=" in word:
                continue
            if word.startswith("-") or word in {"and/or", "host/model"}:
                continue
            if not any((base / word.rstrip("/")).exists() for base in bases_for):
                missing.append(word)
    assert not missing, f"AGENTS.md mentions paths that do not exist: {sorted(set(missing))}"
    for name in tools:  # every real tool is described
        assert f"`{name}`" in text, f"AGENTS.md does not mention the tool {name}"
    claude = ROOT / "CLAUDE.md"
    if claude.exists():
        assert claude.read_text("utf-8").strip() == "@AGENTS.md"


# ---- cross-agent wiring ---------------------------------------------------------------------------------------
def test_agents_skills_link_resolves_to_the_skills_directory():
    link = ROOT / ".agents" / "skills"
    assert link.is_symlink() and link.resolve() == SKILLS.resolve()
    assert (link / "optics-agent-decide" / "SKILL.md").is_file()


def test_codex_config_parses_and_mirrors_mcp_json():
    tomllib = pytest.importorskip("tomllib")
    import json
    config = tomllib.loads((ROOT / ".codex" / "config.toml").read_text("utf-8"))["mcp_servers"]
    mcp = json.loads((ROOT / ".mcp.json").read_text("utf-8"))["mcpServers"]
    assert set(config) == set(mcp) == SERVERS
    for name, server in config.items():
        command = Path(server["command"])
        assert command.is_absolute() and command.exists(), (
            f"{name}: {command} does not exist on this machine; regenerate with tools/print-codex-config.sh")
        assert server["args"] == mcp[name]["args"]
        module = server["args"][1]
        assert (PROJ / (module.replace(".", "/") + ".py")).is_file()
        data_dir = Path(server["env"]["OPTICS_MCP_DATA_DIR"])
        assert data_dir.is_absolute() and data_dir.parent.is_dir() and data_dir.name == "data-mcp"
        assert data_dir.parent == PROJ.resolve() or data_dir.parent.resolve() == PROJ.resolve()
    assert config["optics-evidence"]["default_tools_approval_mode"] == "approve"
    assert config["optics-runs"]["default_tools_approval_mode"] == "prompt"


def test_print_codex_config_regenerates_the_committed_config():
    tomllib = pytest.importorskip("tomllib")
    script = ROOT / "tools" / "print-codex-config.sh"
    assert script.is_file()
    out = subprocess.run(["sh", str(script)], capture_output=True, text=True, check=True, cwd=str(ROOT)).stdout
    assert tomllib.loads(out) == tomllib.loads((ROOT / ".codex" / "config.toml").read_text("utf-8"))


# ---- panel Codex CLI decider ------------------------------------------------------------------------------------
def test_codex_decider_guides_equal_the_role_skills():
    """cli_decider sends these guides to `codex exec`; they must not drift from the role skills."""
    from optics_graph import cli_decider
    for role, name in cli_decider.ROLE_GUIDES.items():
        skill = (SKILLS / f"optics-decide-{name}" / "SKILL.md").read_text("utf-8").split("---\n", 2)[2].lstrip("\n")
        assert cli_decider.guide_for(role) == skill, f"decider_guides/{name}.md differs from optics-decide-{name}"
