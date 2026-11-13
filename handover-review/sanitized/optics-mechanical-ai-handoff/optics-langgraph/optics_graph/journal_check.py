"""Neutral, read-only check shared by the HTTP artifact route and the evidence MCP server.

Whitelist plus completion.json hash re-verification for one stage-journal file. This module imports
no mcp, Runtime, sqlite or fcntl and performs no writes; it only reads.
"""
from __future__ import annotations

from pathlib import Path

from .sim_io import read_json
from .team_artifacts import digest, encode, read_journal_artifact

STAGES = frozenset({"input", "vision", "prompt", "act", "validator", "controller", "sim",
                    "output", "execution", "mode", "done"})
FILENAMES = frozenset({
    "input.json", "event.json", "completion.json", "decision.json", "plan.json", "artifact.zip",
    "verification.json", "simulation.json", "prob0.json", "target_layout.json", "prompt.json",
    "validation.json", "execution_receipt.json", "prompt.txt", "experiment.json",
    "requirements.json", "design.json", "design-review.json", "model-response.json", "agent-prompt.txt",
    "inventory.json", "request.json", "optical-test-suite.json", "optical-test-report.json",
    "test-agent-prompt.txt", "test-model-response.json",
    *(f"decision-{kind}-{role}.json" for kind in ("request", "response", "rejections")
      for role in ("interpret", "design", "optical_tests", "review", "repair_act", "repair_controller"))})
ARTIFACT_LIMIT = 800_000


def is_whitelisted(stage, attempt, filename) -> bool:
    """True only for a known stage, an integer attempt in 1..100 and a whitelisted file name."""
    return (isinstance(stage, str) and stage in STAGES
            and type(attempt) is int and 1 <= attempt <= 100
            and isinstance(filename, str) and filename in FILENAMES)


def read_verified(directory: Path, filename: str, *, limit: int = ARTIFACT_LIMIT,
                  completion_limit: int | None = None) -> bytes:
    """Return the bytes of directory/filename after re-verifying them against completion.json.

    FileNotFoundError: the node has not saved complete evidence (no completion.json, or no such file).
    ValueError (and KeyError/OSError/TypeError from malformed or symlinked files): the evidence does not
    match the completion record. A name outside the whitelist is refused with ValueError.
    """
    if not isinstance(filename, str) or filename not in FILENAMES:
        raise ValueError("filename is not on the evidence whitelist")
    directory = Path(directory)
    journal = read_json(directory / "completion.json", limit=completion_limit)
    if digest(encode(journal["update"], limit=None)) != journal["update_sha256"]:
        raise ValueError("Journal changed")
    raw = read_journal_artifact(directory / filename, limit=limit)
    if filename != "completion.json" and digest(raw) != journal["files"][filename]:
        raise ValueError("Artifact changed")
    return raw
