"""optics-evidence: READ-ONLY MCP server over the workbench run directories (stdio).

This module never imports Runtime and has no code path that opens a file for
writing, creates a directory, or removes anything below the data directory.
Every JSON read goes through the strict readers in sim_io (duplicate keys and
non-finite numbers are rejected) with an explicit size bound. The journal
whitelist and hash re-verification duplicate the logic of server.py's
algorithm_artifact route.

Trust model: run.json is NOT trusted. The failure history, attempt counts,
final status/error and the execution receipt are rebuilt from the journal
chain (completion.json updates and feedback-NN.json) and compared with
run.json; any disagreement is reported as an issue.

Usage:  python -m optics_graph.mcp_evidence [--data-dir DIR]
        (or env OPTICS_MCP_DATA_DIR; default: settings.DATA)
"""
from __future__ import annotations

import argparse
import base64
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from mcp.server.fastmcp import FastMCP

from .journal_check import ARTIFACT_LIMIT, FILENAMES, STAGES, read_verified
from .settings import DATA
from .sim_io import decode_json, read_bytes, read_json
from .team_artifacts import digest, encode

RUN_ID_RE = re.compile(r"[0-9a-f]{32}")
JOURNAL_DIR_RE = re.compile(r"(input|vision|prompt|act|validator|controller|sim|output|execution|mode|done)-([0-9]{2,3})")
FEEDBACK_RE = re.compile(r"feedback-([0-9]{2,3})\.json")
PLAN_PATH_RE = re.compile(r"team/controller-([0-9]{2,3})/plan\.json")
ARTIFACT_PATH_RE = re.compile(r"team/sim-([0-9]{2,3})/artifact\.zip")
# STAGES / FILENAMES (the same whitelist as server.py's algorithm_artifact route) live in journal_check.
# execution-started.json is deliberately NOT on it; its parsed content is returned by get_execution_status.
JOURNAL_KEYS = {"version", "input_sha256", "files", "update", "update_sha256"}
UNKNOWN_CODE = "ALGORITHM_EXECUTION_UNKNOWN"
TERMINAL = frozenset({"algorithm_passed", "failed"})
# Read bounds (bytes). run.json and completion.json carry whole graph state.
RUN_JSON_LIMIT = 8_000_000
COMPLETION_LIMIT = 8_000_000
SMALL_JSON_LIMIT = 1_000_000
JOURNAL_FILE_LIMIT = 16_000_000


class EvidenceError(ValueError):
    pass


def _parse_time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _name_parts(name: str) -> tuple[str, int]:
    m = JOURNAL_DIR_RE.fullmatch(name)
    return m.group(1), int(m.group(2))


class Store:
    def __init__(self, data_dir: Path):
        self.data = Path(data_dir).resolve()

    # ---- validation -------------------------------------------------------
    @staticmethod
    def check_run_id(run_id: Any) -> str:
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise EvidenceError("run_id must be exactly 32 lowercase hex characters")
        return run_id

    @staticmethod
    def check_stage(stage: Any) -> str:
        if not isinstance(stage, str) or stage not in STAGES:
            raise EvidenceError(f"stage must be one of {sorted(STAGES)}")
        return stage

    @staticmethod
    def check_attempt(attempt: Any) -> int:
        if type(attempt) is not int or not 1 <= attempt <= 100:
            raise EvidenceError("attempt must be an integer from 1 to 100")
        return attempt

    @staticmethod
    def check_filename(filename: Any) -> str:
        if not isinstance(filename, str) or filename not in FILENAMES:
            raise EvidenceError("filename is not on the evidence whitelist")
        return filename

    def run_dir(self, run_id: str) -> Path:
        self.check_run_id(run_id)
        runs = self.data / "runs"
        path = runs / run_id
        if runs.is_symlink() or path.is_symlink() or not path.is_dir():
            raise EvidenceError("run not found")
        if path.resolve().parent != runs.resolve():
            raise EvidenceError("run not found")
        return path

    def journal_dir(self, run_id: str, stage: str, attempt: int) -> Path:
        base = self.run_dir(run_id)
        self.check_stage(stage)
        self.check_attempt(attempt)
        team = base / "team"
        path = team / f"{stage}-{attempt:02d}"
        if team.is_symlink() or path.is_symlink() or not path.is_dir():
            raise EvidenceError("that stage attempt has no journal")
        return path

    def pending_decision(self, run_id: str, state: dict) -> dict | None:
        """run.json's pending_decision plus `submitted`, computed from the journal directory (run.json cannot
        know: a submit writes only the response file). None unless the run is awaiting_decision."""
        pending = state.get("pending_decision")
        if state.get("status") != "awaiting_decision" or not isinstance(pending, dict):
            return None
        out = {k: v for k, v in pending.items() if k != "submitted"}
        out["submitted"] = None  # unknown unless the entry names a real response file
        node, attempt, role = pending.get("node"), pending.get("attempt"), pending.get("role")
        if (isinstance(node, str) and node in STAGES and type(attempt) is int and 1 <= attempt <= 100
                and isinstance(role, str) and f"decision-response-{role}.json" in FILENAMES):
            try:
                path = self.journal_dir(run_id, node, attempt) / f"decision-response-{role}.json"
                out["submitted"] = path.is_file() and not path.is_symlink()
            except EvidenceError:
                pass
        return out

    def run_state(self, run_id: str) -> dict:
        state = read_json(self.run_dir(run_id) / "run.json", limit=RUN_JSON_LIMIT)
        if not isinstance(state, dict):
            raise EvidenceError("run.json is not an object")
        return state

    def team_file(self, base: Path, rel: str, *, limit: int) -> bytes:
        """Read base/rel (already pattern-checked) refusing a symlinked team dir or journal dir."""
        parts = rel.split("/")
        current = base
        for part in parts[:-1]:
            current = current / part
            if current.is_symlink() or not current.is_dir():
                raise ValueError(f"{rel}: directory is missing or a symlink")
        return read_bytes(current / parts[-1], limit=limit)

    # ---- acceptance provenance (journal-verified reads only) --------------------------------------------------
    @staticmethod
    def has_evidence(directory: Path, filename: str) -> bool:
        """True when directory/filename exists on disk (symlinks included, so they fail verification loudly) or
        completion.json lists it. Absence of a file nothing lists is normal (the decision belongs to another node)."""
        if os.path.lexists(directory / filename):
            return True
        try:
            files = read_json(directory / "completion.json", limit=COMPLETION_LIMIT).get("files")
        except (ValueError, OSError, KeyError, TypeError, AttributeError):
            return False
        return isinstance(files, dict) and filename in files

    def verified_json(self, directory: Path, filename: str, notes: list[str]) -> Any:
        """Parsed JSON of directory/filename after re-verification against completion.json, else None (and a note
        when the evidence exists but fails verification). An unfinished attempt has no completion.json: skipped."""
        try:
            return decode_json(read_verified(directory, filename, limit=JOURNAL_FILE_LIMIT,
                                             completion_limit=COMPLETION_LIMIT).decode("utf-8"))
        except FileNotFoundError:
            return None
        except (ValueError, OSError, KeyError, TypeError, UnicodeDecodeError) as exc:
            notes.append(f"{directory.name}/{filename}: not used, failed journal verification ({str(exc)[:120]})")
            return None

    def provenance(self, run_id: str, state: dict) -> dict:
        """Who decided, which limits were used, and how tight the optical tests are. Every value comes from a
        journal file whose hash was re-verified; run.json is used only for the provider name."""
        from .algorithm_sim import ANGLE_LIMIT, POSITION_LIMIT
        notes: list[str] = []
        provider = state.get("provider")
        dirs = self.journal_dirs(self.run_dir(run_id))

        def latest(stage: str, filename: str) -> tuple[str, Any] | None:
            for directory in reversed([d for d in dirs if d.name.startswith(stage + "-")]):
                if self.has_evidence(directory, filename):
                    return directory.name, self.verified_json(directory, filename, notes)
            return None

        out: dict[str, Any] = {"provider": provider, "provenance": None, "decider_labels": None,
                               "design_and_review_share_a_label": None}
        if provider == "agent":
            labels: dict[str, list[str]] = {}
            for role in ("interpret", "design", "optical_tests", "review", "repair_act", "repair_controller"):
                found = set()
                for directory in dirs:
                    name = f"decision-response-{role}.json"
                    if not self.has_evidence(directory, name):
                        continue
                    doc = self.verified_json(directory, name, notes)
                    if isinstance(doc, dict) and isinstance(doc.get("decider_label"), str):
                        found.add(doc["decider_label"])
                if found:
                    labels[role] = sorted(found)
            out["provenance"] = "agent-decided; decider identity self-reported and unverified"
            out["decider_labels"] = labels
            if "design" in labels and "review" in labels:
                shared = sorted(set(labels["design"]) & set(labels["review"]))
                out["design_and_review_share_a_label"] = bool(shared)
                if shared:
                    out["same_label_warning"] = (f"the design and review decisions carry the same self-reported label "
                                                 f"{shared}: the independent review may not be independent")
        # limits: what the simulation actually used, plus what the experiment declared
        used = latest("sim", "simulation.json")
        used_limits = (used[1] or {}).get("limits") if used and isinstance(used[1], dict) else None
        experiment = latest("act", "experiment.json")
        declared = (experiment[1] or {}).get("execution_limits") if experiment and isinstance(experiment[1], dict) else None
        defaults = {"position_m": POSITION_LIMIT, "angle_rad": ANGLE_LIMIT}
        effective = used_limits if isinstance(used_limits, dict) else declared if isinstance(declared, dict) else None
        out["execution_limits"] = {
            "used": used_limits if isinstance(used_limits, dict) else None,
            "used_source": f"{used[0]}/simulation.json" if isinstance(used_limits, dict) else None,
            "declared_by_experiment": declared if isinstance(declared, dict) else None,
            "defaults": defaults,
            "equals_defaults": None if effective is None else effective == defaults,
            "note": ("looser than defaults" if effective is not None and (
                effective.get("position_m", 0) > POSITION_LIMIT or effective.get("angle_rad", 0) > ANGLE_LIMIT)
                else "no limits recorded yet (defaults apply)" if effective is None else None)}
        report = latest("output", "optical-test-report.json") or latest("validator", "optical-test-report.json")
        rows, widest = [], None
        if report and isinstance(report[1], dict):
            for test in report[1].get("tests") or []:
                if not isinstance(test, dict):
                    continue
                expected, tolerance = test.get("expected"), test.get("absolute_tolerance")
                relative = (tolerance / abs(expected) if isinstance(expected, (int, float)) and expected != 0
                            and isinstance(tolerance, (int, float)) else None)
                rows.append({"id": test.get("id"), "comparison": test.get("comparison"), "unit": test.get("unit"),
                             "expected": expected, "actual": test.get("actual"), "absolute_tolerance": tolerance,
                             "tolerance_relative_to_expected": relative, "status": test.get("status")})
                if relative is not None and (widest is None or relative > widest["tolerance_relative_to_expected"]):
                    widest = {"id": test.get("id"), "tolerance_relative_to_expected": relative}
        out["optical_tests"] = {"source": f"{report[0]}/optical-test-report.json" if report else None,
                                "tests": rows, "largest_tolerance_relative_to_expected": widest,
                                "note": "tolerance_relative_to_expected = absolute_tolerance / |expected|; null when "
                                        "expected is 0. Test tolerances are authored by the run's decider; bounds on "
                                        "them are for the optics owner to define."}
        if notes:
            out["provenance_notes"] = notes
        return out

    # ---- journals ---------------------------------------------------------
    def journal(self, directory: Path) -> dict:
        """Strictly re-verify one stage journal. Never raises for tampering."""
        out: dict = {"stage_dir": directory.name, "present": False, "verified": False,
                     "issues": [], "event": None, "update": None, "files": {},
                     "input_sha256": None, "update_sha256": None}
        issues = out["issues"]
        completion = directory / "completion.json"
        event_path = directory / "event.json"
        if event_path.is_file() and not event_path.is_symlink():
            try:
                out["event"] = read_json(event_path, limit=SMALL_JSON_LIMIT)
            except (ValueError, OSError) as exc:
                issues.append(f"event.json unreadable: {exc}")
        if not completion.exists() and not completion.is_symlink():
            issues.append("completion.json missing (attempt incomplete or interrupted)")
            return out
        out["present"] = True
        try:
            saved = read_json(completion, limit=COMPLETION_LIMIT)
            if not isinstance(saved, dict) or set(saved) != JOURNAL_KEYS:
                raise ValueError("completion.json has an unexpected shape")
            if (type(saved["version"]) is not int or saved["version"] != 1
                    or not isinstance(saved["files"], dict) or not isinstance(saved["update"], dict)):
                raise ValueError("completion.json fields have unexpected types")
        except (ValueError, OSError) as exc:
            issues.append(f"completion.json rejected: {exc}")
            return out
        out["update"] = saved["update"]
        out["input_sha256"] = saved["input_sha256"]
        out["update_sha256"] = saved["update_sha256"]
        try:
            if digest(encode(saved["update"], limit=None)) != saved["update_sha256"]:
                issues.append("update_sha256 does not match the recorded update")
        except (ValueError, TypeError) as exc:
            issues.append(f"update cannot be re-encoded: {exc}")
        listed = saved["files"]
        for name, recorded in listed.items():
            if not isinstance(name, str) or Path(name).name != name or name == "completion.json":
                issues.append(f"illegal journal file name {name!r}")
                continue
            try:
                actual = digest(read_bytes(directory / name, limit=JOURNAL_FILE_LIMIT))
            except (ValueError, OSError) as exc:
                out["files"][name] = {"recorded_sha256": recorded, "actual_sha256": None, "ok": False}
                issues.append(f"{name}: cannot be read ({exc})")
                continue
            ok = actual == recorded
            out["files"][name] = {"recorded_sha256": recorded, "actual_sha256": actual, "ok": ok}
            if not ok:
                issues.append(f"{name}: sha256 differs from completion.json")
        try:
            present = {p.name for p in directory.iterdir() if p.name != "completion.json"}
        except OSError as exc:
            present = set()
            issues.append(f"directory unreadable: {exc}")
        for extra in sorted(present - set(listed)):
            issues.append(f"{extra}: present but not recorded in completion.json")
        events = saved["update"].get("events")
        if out["event"] is not None and (not isinstance(events, list) or events != [out["event"]]):
            issues.append("event.json differs from the event recorded in the completion update")
        out["verified"] = not issues
        return out

    def journal_dirs(self, run_dir: Path) -> list[Path]:
        """Real (non-symlink) journal directories, sorted by stage then NUMERIC attempt."""
        team = run_dir / "team"
        if team.is_symlink() or not team.is_dir():
            return []
        found = [p for p in team.iterdir()
                 if JOURNAL_DIR_RE.fullmatch(p.name) and not p.is_symlink() and p.is_dir()]
        return sorted(found, key=lambda p: _name_parts(p.name))

    def execution_dirs(self, run_dir: Path) -> list[Path]:
        return [p for p in self.journal_dirs(run_dir) if p.name.startswith("execution-")]

    # ---- the journal chain ------------------------------------------------
    def chain(self, base: Path, state: dict) -> dict:
        """Rebuild what run.json SHOULD say from the journals and feedback files.

        Returns lists of issues grouped by area, the verified journals, the
        router decisions (from feedback-NN.json) with the original failure
        taken from the failing journal, and the expected final status/error.
        """
        scan: list[str] = []
        attempts_issues: list[str] = []
        history: list[str] = []
        final: list[str] = []
        dirs: dict[str, Path] = {}
        feedback_paths: dict[int, Path] = {}
        team = base / "team"
        if team.is_symlink():
            scan.append("team is a symlink")
        elif team.exists() and not team.is_dir():
            scan.append("team is not a directory")
        elif team.is_dir():
            for p in sorted(team.iterdir()):
                n = p.name
                fm = FEEDBACK_RE.fullmatch(n)
                if p.is_symlink():
                    scan.append(f"team/{n}: symlink refused")
                elif JOURNAL_DIR_RE.fullmatch(n) and p.is_dir():
                    stage, num = _name_parts(n)
                    if n != f"{stage}-{num:02d}":
                        scan.append(f"team/{n}: non-canonical journal name")
                    else:
                        dirs[n] = p
                elif fm and p.is_file():
                    num = int(fm.group(1))
                    if n != f"feedback-{num:02d}.json":
                        scan.append(f"team/{n}: non-canonical feedback name")
                    else:
                        feedback_paths[num] = p
                else:
                    scan.append(f"team/{n}: unexpected entry")
        journals = {name: self.journal(p) for name, p in dirs.items()}

        # --- attempts ------------------------------------------------------
        status = state.get("status")
        terminal = status in TERMINAL
        raw_attempts = state.get("attempts")
        if raw_attempts is not None and not (isinstance(raw_attempts, dict) and all(
                isinstance(k, str) and type(v) is int and v >= 0 for k, v in raw_attempts.items())):
            attempts_issues.append("run.json attempts is not a mapping of stage to count")
            raw_attempts = {}
        raw_attempts = raw_attempts or {}
        chain_attempts: dict[str, int] = {}
        for name, j in journals.items():
            stage, num = _name_parts(name)
            if j["present"]:
                chain_attempts[stage] = max(chain_attempts.get(stage, 0), num)
            if j["verified"]:
                recorded = j["update"].get("attempts")
                if isinstance(recorded, dict):
                    for k, v in recorded.items():
                        if isinstance(k, str) and type(v) is int:
                            chain_attempts[k] = max(chain_attempts.get(k, 0), v)
        allowed_incomplete: set[str] = set()
        by_stage: dict[str, list[int]] = {}
        for name in journals:
            stage, num = _name_parts(name)
            by_stage.setdefault(stage, []).append(num)
        for stage in sorted(set(by_stage) | set(chain_attempts)):
            have = set(by_stage.get(stage, []))
            want = chain_attempts.get(stage, 0)
            missing = [n for n in range(1, want + 1) if n not in have]
            extra = sorted(n for n in have if n > want)
            if extra == [want + 1] and not terminal and not journals[f"{stage}-{want + 1:02d}"]["present"]:
                allowed_incomplete.add(f"{stage}-{want + 1:02d}")
                extra = []
            if missing:
                attempts_issues.append(f"{stage}: journal directories missing for attempts {missing} "
                                       f"(the chain records {want} attempt(s))")
            if extra:
                attempts_issues.append(f"{stage}: unexpected journal directories for attempts {extra} "
                                       f"(the chain records {want} attempt(s))")
        for stage in sorted(set(raw_attempts) | set(chain_attempts)):
            if raw_attempts.get(stage, 0) != chain_attempts.get(stage, 0):
                attempts_issues.append(f"run.json attempts[{stage!r}]={raw_attempts.get(stage, 0)} but the "
                                       f"journal chain records {chain_attempts.get(stage, 0)}")
        journal_issues: list[str] = []
        for name in sorted(journals, key=_name_parts):
            if name in allowed_incomplete:
                continue
            journal_issues.extend(f"{name}: {m}" for m in journals[name]["issues"])

        # --- feedback / failure history -----------------------------------
        items: dict[int, dict] = {}
        for num, p in sorted(feedback_paths.items()):
            try:
                item = read_json(p, limit=SMALL_JSON_LIMIT)
                if not isinstance(item, dict):
                    raise ValueError("not an object")
                items[num] = item
            except (ValueError, OSError) as exc:
                history.append(f"feedback-{num:02d}.json unreadable: {exc}")
        count = max(feedback_paths, default=0)
        gaps = [n for n in range(1, count + 1) if n not in feedback_paths]
        if gaps:
            history.append(f"feedback files missing for {gaps}")
        failing: dict[str, dict] = {}
        for name, j in journals.items():
            if j["verified"] and isinstance(j["update"].get("error"), dict):
                failing[name] = j["update"]["error"]
        decisions: list[dict] = []
        linked: set[str] = set()
        original: dict[int, dict | None] = {}
        for num in range(1, count + 1):
            item = items.get(num)
            err = None
            if item is not None:
                try:
                    jname = f"{item['from']}-{int(item['attempt']):02d}"
                except (KeyError, ValueError, TypeError):
                    jname = None
                if jname is None or jname not in failing:
                    history.append(f"feedback-{num:02d}.json refers to a failing attempt that no verified "
                                   f"journal records ({jname})")
                else:
                    err = failing[jname]
                    linked.add(jname)
                    if err.get("code") != item.get("code") or err.get("stage") != item.get("from"):
                        history.append(f"feedback-{num:02d}.json code/stage differs from the failure in {jname}")
            original[num] = err
            decisions.append({"index": num, "decision": item, "original_failure": err})
        timeline: list[tuple[datetime, int, str, Any]] = []
        for name, j in journals.items():
            if j["verified"]:
                events = j["update"].get("events")
                when = _parse_time((events[0] if isinstance(events, list) and events and
                                    isinstance(events[0], dict) else {}).get("time"))
                if when is not None:
                    timeline.append((when, 0, name, j["update"]))
        for num, item in items.items():
            when = _parse_time(item.get("time"))
            if when is not None:
                timeline.append((when, 1, f"feedback-{num:02d}", item))
        timeline.sort(key=lambda t: (t[0], t[1]))
        for name in sorted(failing):
            stage, _ = _name_parts(name)
            if stage == "execution" or name in linked:
                continue
            if (name == timeline[-1][2] if timeline else False) and not terminal:
                continue  # failure just recorded; the router has not run yet
            history.append(f"{name} recorded a failure that has no feedback-NN.json router decision")
        expected_log = [items[n] for n in range(1, count + 1) if n in items]
        expected_feedback = [original[n] for n in range(1, count + 1) if original.get(n) is not None]
        have_log = state.get("retry_log") or []
        have_feedback = state.get("feedback") or []
        if have_log != expected_log:
            history.append("run.json retry_log differs from the feedback-NN.json files")
        if have_feedback != expected_feedback:
            history.append("run.json feedback differs from the failures recorded in the stage journals")
        expected_counts = (expected_log[-1].get("retry_counts") if expected_log else None) or {}
        if (state.get("retry_counts") or {}) != expected_counts:
            history.append("run.json retry_counts differs from the last feedback-NN.json")

        # --- final state ---------------------------------------------------
        expected: dict[str, Any] = {"kind": "none", "error": None}
        if timeline:
            when, order, name, payload = timeline[-1]
            if order == 1:
                if payload.get("to") == "stop":
                    num = int(name.split("-")[1])
                    expected = {"kind": "failed", "error": original.get(num)}
                else:
                    expected = {"kind": "inflight", "error": None}
            else:
                stage, _ = _name_parts(name)
                if isinstance(payload.get("error"), dict):
                    expected = ({"kind": "failed", "error": payload["error"]} if stage == "execution"
                                else {"kind": "inflight", "error": None})
                elif stage == "done" and payload.get("status") == "algorithm_passed":
                    expected = {"kind": "passed", "error": None}
                else:
                    expected = {"kind": "inflight", "error": None}
        err = state.get("error")
        if expected["kind"] == "passed":
            if status != "algorithm_passed":
                final.append(f"run.json status is {status!r} but the journal chain ends with a completed done stage")
            if err:
                final.append("run.json carries an error but the journal chain ends in success")
        elif expected["kind"] == "failed":
            if status != "failed":
                final.append(f"run.json status is {status!r} but the journal chain ends in a failure")
            if err != expected["error"]:
                final.append("run.json error differs from the failure that ended the journal chain")
        else:
            if status == "algorithm_passed":
                final.append("run.json says algorithm_passed but the journal chain does not end in a completed done stage")
            elif status == "failed" and not (expected["kind"] == "none" and isinstance(err, dict)
                                             and err.get("code") in ("RUN_INPUT_INVALID", "RUN_MODE_MISMATCH")):
                final.append("run.json says failed but the journal chain records no terminating failure")
        return {"dirs": dirs, "journals": journals, "scan": scan, "attempts": attempts_issues,
                "journal_issues": journal_issues, "history": history, "final": final,
                "decisions": decisions, "failing": failing, "expected": expected,
                "chain_attempts": chain_attempts, "timeline": timeline}

    def last_execution(self, chain: dict) -> tuple[str, dict] | None:
        names = sorted((n for n in chain["journals"] if n.startswith("execution-")), key=_name_parts)
        return (names[-1], chain["journals"][names[-1]]) if names else None

    def receipt_problems(self, base: Path, state: dict, directory: Path, j: dict) -> list[str]:
        """Everything that must hold for an execution receipt to count as accepted."""
        problems: list[str] = []
        update = j.get("update") or {}
        if not j["verified"]:
            problems.append("the execution journal does not verify")
        if update.get("error"):
            problems.append("the execution journal records an error")
        recorded = j["files"].get("execution_receipt.json")
        if not recorded or recorded.get("ok") is not True:
            problems.append("execution_receipt.json is not listed in completion.json or its hash differs")
        try:
            receipt = read_json(directory / "execution_receipt.json", limit=SMALL_JSON_LIMIT)
        except (ValueError, OSError) as exc:
            return problems + [f"execution_receipt.json unreadable: {exc}"]
        if not isinstance(receipt, dict):
            return problems + ["execution_receipt.json is not an object"]
        if update.get("result") != receipt:
            problems.append("the journaled completion result differs from execution_receipt.json")
        if state.get("result") != receipt:
            problems.append("run.json result differs from execution_receipt.json")
        if (receipt.get("status") != "accepted_dry_run" or receipt.get("motions_sent") != 0
                or receipt.get("physical_verified") is not False):
            problems.append("the receipt is not an accepted dry run with motions_sent=0 and physical_verified=false")
        try:
            plan_hash = digest(read_bytes(directory / "plan.json", limit=JOURNAL_FILE_LIMIT))
            if not (receipt.get("plan_hash") == plan_hash == state.get("plan_sha256")):
                problems.append("the receipt plan_hash does not match the execution plan.json and run.json plan_sha256")
        except (ValueError, OSError) as exc:
            problems.append(f"execution plan.json unreadable: {exc}")
        artifact_path = state.get("artifact_path")
        try:
            if not isinstance(artifact_path, str) or not ARTIFACT_PATH_RE.fullmatch(artifact_path):
                raise ValueError("run.json artifact_path is not team/sim-NN/artifact.zip")
            zip_hash = digest(self.team_file(base, artifact_path, limit=ARTIFACT_LIMIT))
            if not (receipt.get("artifact_sha256") == zip_hash == state.get("artifact_sha256")):
                problems.append("the receipt artifact_sha256 does not match the archive and run.json artifact_sha256")
        except (ValueError, OSError) as exc:
            problems.append(f"archive unreadable for the receipt binding: {exc}")
        marker = directory / "execution-started.json"
        if marker.is_file() and not marker.is_symlink():
            try:
                started = read_json(marker, limit=SMALL_JSON_LIMIT)
                if not isinstance(started, dict) or started.get("plan_sha256") != state.get("plan_sha256"):
                    problems.append("execution-started.json is not bound to the run's plan_sha256")
            except (ValueError, OSError):
                problems.append("execution-started.json is unreadable")
        return problems


def build_server(data_dir: Path | str) -> FastMCP:
    store = Store(Path(data_dir))
    mcp = FastMCP("optics-evidence")

    @mcp.tool()
    def list_runs(limit: int = 30) -> dict[str, Any]:
        """List recent runs (newest first): run_id, status, stage, mode, provider, simulation_backend, created_at."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise EvidenceError("limit must be an integer from 1 to 100")
        runs_dir = store.data / "runs"
        if not runs_dir.is_dir() or runs_dir.is_symlink():
            return {"runs": [], "skipped": []}
        candidates = []
        for p in runs_dir.iterdir():
            if RUN_ID_RE.fullmatch(p.name) and p.is_dir() and not p.is_symlink() and (p / "run.json").is_file():
                try:
                    candidates.append(((p / "run.json").stat().st_mtime, p))
                except OSError:
                    continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        runs, skipped = [], []
        for _, p in candidates[:limit]:
            try:
                s = read_json(p / "run.json", limit=RUN_JSON_LIMIT)
                runs.append({k: s.get(k) for k in ("run_id", "status", "stage", "mode", "provider",
                                                   "simulation_backend", "created_at")})
            except (ValueError, OSError) as exc:
                skipped.append({"run_id": p.name, "reason": str(exc)[:200]})
        return {"runs": runs, "skipped": skipped}

    @mcp.tool()
    def get_run(run_id: str) -> dict[str, Any]:
        """Curated run.json state (not independently verified; use verify_run_artifacts): status, stage, attempts, retry counts, error, plan hash, artifact hash, world revision."""
        s = store.run_state(run_id)
        world = s.get("algorithm_state") or {}
        verification = s.get("verification") or {}
        return {
            "run_id": run_id, "status": s.get("status"), "stage": s.get("stage"),
            "next_node": s.get("next_node"), "return_to": s.get("return_to") or None,
            "mode": s.get("mode"), "provider": s.get("provider"), "pending_decision": store.pending_decision(run_id, s),
            "simulation_backend": s.get("simulation_backend"), "created_at": s.get("created_at"),
            "attempts": s.get("attempts"), "retry_counts": s.get("retry_counts"),
            "error": s.get("error"), "plan_sha256": s.get("plan_sha256"),
            "artifact_sha256": s.get("artifact_sha256"), "world_revision": world.get("revision"),
            "rounds": len(s.get("algorithm_history") or []),
            "physical_verified": verification.get("physical_verified", False) is True,
            "note": ("algorithm_passed means the algorithm workflow finished (dry run), "
                     "not that every experiment requirement was verified. These fields are copied from "
                     "run.json; call verify_run_artifacts to check them against the journals."),
        }

    @mcp.tool()
    def get_feedback_log(run_id: str) -> dict[str, Any]:
        """Router decisions from team/feedback-NN.json with the original failures taken from the stage journals; any disagreement with run.json is listed in issues."""
        base = store.run_dir(run_id)
        s = store.run_state(run_id)
        c = store.chain(base, s)
        have_feedback = s.get("feedback") or []
        decisions = []
        for d in c["decisions"]:
            i = d["index"]
            decisions.append({**d, "run_json_failure_agrees":
                              (have_feedback[i - 1] == d["original_failure"]) if i - 1 < len(have_feedback) else None})
        failed_events = []
        for name in sorted(c["journals"], key=_name_parts):
            j = c["journals"][name]
            if j["verified"]:
                for e in (j["update"].get("events") or []):
                    if isinstance(e, dict) and e.get("outcome") == "failed":
                        failed_events.append(e)
        issues = [*c["scan"], *c["history"]]
        unrouted = [{"journal": n, "error": e} for n, e in sorted(c["failing"].items())
                    if _name_parts(n)[0] == "execution"]
        decided = [d for d in decisions if d["decision"] is not None]
        return {"run_id": run_id,
                "routes": [f"{d['decision'].get('from')} -> {d['decision'].get('to')}" for d in decided],
                "decisions": decisions, "failed_events": failed_events, "execution_failures": unrouted,
                "retry_counts": (decided[-1]["decision"].get("retry_counts") if decided else {}),
                "source": "journals and feedback files (run.json is only compared)",
                "consistent": not issues, "issues": issues}

    @mcp.tool()
    def get_stage_journal(run_id: str, stage: str, attempt: int) -> dict[str, Any]:
        """Journal of one stage attempt: event, completion update, file hashes, verified (recomputed hashes match)."""
        directory = store.journal_dir(run_id, stage, attempt)
        return {"run_id": run_id, "stage": stage, "attempt": attempt, **store.journal(directory)}

    @mcp.tool()
    def get_stage_artifact(run_id: str, stage: str, attempt: int, filename: str) -> dict[str, Any]:
        """Whitelisted journal file content. Errors if the file does not match completion.json."""
        directory = store.journal_dir(run_id, stage, attempt)
        store.check_filename(filename)
        try:
            raw = read_verified(directory, filename, limit=ARTIFACT_LIMIT, completion_limit=COMPLETION_LIMIT)
        except FileNotFoundError:
            raise EvidenceError("this node has not saved complete evidence yet") from None
        except (ValueError, KeyError, OSError, TypeError):
            raise EvidenceError("the evidence does not match the node completion record") from None
        result: dict[str, Any] = {"run_id": run_id, "stage": stage, "attempt": attempt,
                                  "filename": filename, "sha256": digest(raw), "size": len(raw)}
        if filename.endswith(".zip"):
            try:
                with ZipFile(BytesIO(raw)) as archive:
                    result["members"] = {i.filename: archive.read(i).decode("utf-8", "replace")
                                         for i in archive.infolist() if i.file_size <= 256_000}
            except (BadZipFile, ValueError, KeyError) as exc:
                raise EvidenceError(f"archive unreadable: {exc}") from None
            result["encoding"] = "zip-members-as-text"
        elif len(raw) <= 400_000:
            try:
                result["content"] = raw.decode("utf-8")
                result["encoding"] = "utf-8"
            except UnicodeDecodeError:
                result["content"] = base64.b64encode(raw).decode()
                result["encoding"] = "base64"
        else:
            result["encoding"] = "omitted-too-large"
        return result

    def _execution(run_id: str) -> dict[str, Any]:
        base = store.run_dir(run_id)
        s = store.run_state(run_id)
        run_error = s.get("error") if isinstance(s.get("error"), dict) else None
        dirs = store.execution_dirs(base)
        if run_error and run_error.get("code") == UNKNOWN_CODE:
            attempt = _name_parts(dirs[-1].name)[1] if dirs else None
            marker = bool(dirs) and (dirs[-1] / "execution-started.json").exists()
            return {"run_id": run_id, "attempt": attempt, "marker_present": marker,
                    "execution_status": "unknown", "error": run_error,
                    "reason": "run.json records ALGORITHM_EXECUTION_UNKNOWN. Do NOT retry; hand to a person."}
        if not dirs:
            if s.get("status") == "algorithm_passed" or (s.get("attempts") or {}).get("execution"):
                return {"run_id": run_id, "execution_status": "unknown", "attempt": None,
                        "reason": "run.json claims an execution but its journal directory is missing"}
            return {"run_id": run_id, "execution_status": "not_started", "attempt": None}
        last = dirs[-1]
        attempt = _name_parts(last.name)[1]
        marker, receipt_path = last / "execution-started.json", last / "execution_receipt.json"
        journal = store.journal(last)
        update = journal.get("update") or {}
        base_out = {"run_id": run_id, "attempt": attempt, "marker_present": marker.exists(),
                    "journal_verified": journal["verified"]}
        if marker.is_file() and not marker.is_symlink():
            try:
                base_out["marker"] = read_json(marker, limit=SMALL_JSON_LIMIT)
            except (ValueError, OSError):
                base_out["marker"] = None
        journal_error = update.get("error") if isinstance(update.get("error"), dict) else None
        if journal_error and journal_error.get("code") == UNKNOWN_CODE:
            return {**base_out, "execution_status": "unknown", "error": journal_error,
                    "reason": "the execution journal records ALGORITHM_EXECUTION_UNKNOWN. Do NOT retry; hand to a person."}
        if receipt_path.exists() or receipt_path.is_symlink():
            problems = store.receipt_problems(base, s, last, journal)
            if problems:
                return {**base_out, "execution_status": "unknown", "reason": "; ".join(problems),
                        "note": "a receipt file exists but the evidence does not support 'accepted'. Do NOT retry."}
            receipt = read_json(receipt_path, limit=SMALL_JSON_LIMIT)
            return {**base_out, "execution_status": "accepted", "receipt": receipt,
                    "receipt_sha256": journal["files"]["execution_receipt.json"]["actual_sha256"],
                    "receipt_journaled": True,
                    "note": "accepted_dry_run only: motions_sent is 0 and nothing was physically executed."}
        if journal_error:
            if not journal["verified"] or journal_error.get("stage") != "execution":
                return {**base_out, "execution_status": "unknown", "error": journal_error,
                        "reason": "the execution journal does not verify, so the rejection cannot be trusted"}
            return {**base_out, "execution_status": "rejected", "error": journal_error}
        if marker.exists():
            return {**base_out, "execution_status": "unknown", "error": run_error,
                    "reason": ("An execution-started marker exists with no receipt. The result is unknown. "
                               "Do NOT retry; verify and hand to a person.")}
        if run_error and run_error.get("stage") == "execution":
            return {**base_out, "execution_status": "unknown", "error": run_error,
                    "reason": "run.json records an execution error that no verified journal backs"}
        return {**base_out, "execution_status": "not_started",
                "reason": "execution journal directory exists but no start marker was written"}

    @mcp.tool()
    def get_execution_status(run_id: str) -> dict[str, Any]:
        """One of not_started / accepted (journal-verified receipt bound to plan, archive and run result) / rejected (verified journal error) / unknown (marker or doubt; never more favourable than the evidence). Includes marker_present and the parsed marker."""
        return _execution(run_id)

    def _verify(run_id: str) -> dict[str, Any]:
        issues: list[str] = []
        result: dict[str, Any] = {"run_id": run_id, "journals_ok": False, "plan_bytes_identical": None,
                                  "archive_ok": None, "receipt_bound": None, "issues": issues}
        base = store.run_dir(run_id)
        try:
            state = store.run_state(run_id)
        except (ValueError, OSError) as exc:
            issues.append(f"run.json rejected: {exc}")
            result["ok"] = False
            return result
        status = state.get("status")
        passed = status == "algorithm_passed"
        algorithm = state.get("simulation_backend") == "algorithm_v1"
        chain: dict[str, Any] | None = None
        checked = 0
        journals_ok = True
        try:
            if algorithm:
                chain = store.chain(base, state)
                checked = len(chain["journals"])
                for key in ("scan", "journal_issues", "attempts", "history", "final"):
                    issues.extend(chain[key])
                    if chain[key] and key != "final":
                        journals_ok = False
            else:  # legacy backends: journals and feedback files only
                for directory in store.journal_dirs(base):
                    checked += 1
                    j = store.journal(directory)
                    if not j["verified"]:
                        journals_ok = False
                        issues.extend(f"{directory.name}: {m}" for m in j["issues"])
                team = base / "team"
                if team.is_dir() and not team.is_symlink():
                    for p in team.iterdir():
                        if p.is_symlink() or (not (JOURNAL_DIR_RE.fullmatch(p.name) and p.is_dir())
                                              and not FEEDBACK_RE.fullmatch(p.name)):
                            journals_ok = False
                            issues.append(f"team/{p.name}: unexpected entry")
                for i, item in enumerate(state.get("retry_log") or [], start=1):
                    try:
                        if read_json(base / "team" / f"feedback-{i:02d}.json", limit=SMALL_JSON_LIMIT) != item:
                            raise ValueError("differs from run.json retry_log")
                    except (ValueError, OSError) as exc:
                        journals_ok = False
                        issues.append(f"feedback-{i:02d}.json: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, never raise
            journals_ok = False
            issues.append(f"journal verification error: {exc}")
        if checked == 0 and (passed or (state.get("attempts") or {}) or not algorithm):
            journals_ok = False
            issues.append("no stage journals found")
        result["journals_ok"] = journals_ok
        result["journals_checked"] = checked
        result["chain_checked"] = chain is not None
        if chain is not None:
            result["expected_final"] = chain["expected"]["kind"]
        last = store.last_execution(chain) if chain is not None else None
        exec_update = (last[1]["update"] or {}) if last and last[1]["verified"] else {}
        claims_accept = passed or isinstance(exec_update.get("result"), dict)
        # 2. plan bytes
        try:
            if claims_accept and last is not None:
                plan_path = state.get("plan_path")
                if not isinstance(plan_path, str) or not PLAN_PATH_RE.fullmatch(plan_path):
                    raise ValueError("run.json plan_path is not team/controller-NN/plan.json")
                published = store.team_file(base, plan_path, limit=JOURNAL_FILE_LIMIT)
                handed = read_bytes(store.journal_dir(run_id, "execution", _name_parts(last[0])[1]) / "plan.json",
                                    limit=JOURNAL_FILE_LIMIT)
                same = published == handed
                if not same:
                    issues.append("published plan bytes differ from the execution handoff plan.json")
                if state.get("plan_sha256") != digest(published):
                    same = False
                    issues.append("run.json plan_sha256 does not match the published plan bytes")
                result["plan_bytes_identical"] = same
        except (ValueError, OSError, UnicodeDecodeError, EvidenceError) as exc:
            result["plan_bytes_identical"] = False
            issues.append(f"plan comparison failed: {exc}")
        # 3. archive
        try:
            if algorithm and state.get("artifact_path"):
                path = state.get("artifact_path")
                if not isinstance(path, str) or not ARTIFACT_PATH_RE.fullmatch(path):
                    raise ValueError("run.json artifact_path is not team/sim-NN/artifact.zip")
                if state.get("plan_path") is not None and not (
                        isinstance(state["plan_path"], str) and PLAN_PATH_RE.fullmatch(state["plan_path"])):
                    raise ValueError("run.json plan_path is not team/controller-NN/plan.json")
                from .algorithm_sim import AlgorithmServices

                class _Reject(Exception):
                    def __init__(self, code, message, target=None, details=None):
                        super().__init__(message)

                AlgorithmServices(base, _Reject).verify_archive(state)
                result["archive_ok"] = True
        except Exception as exc:  # noqa: BLE001 - a failed verification is a result
            result["archive_ok"] = False
            issues.append(f"archive verification failed: {str(exc)[:300]}")
        # 4. receipt binding (only when the run claims an accepted execution)
        notes: list[str] = []
        try:
            if claims_accept and last is not None:
                problems = store.receipt_problems(base, state, store.journal_dir(
                    run_id, "execution", _name_parts(last[0])[1]), last[1])
                result["receipt_bound"] = not problems
                issues.extend(f"execution receipt: {p}" for p in problems)
            elif last is not None and (store.journal_dir(run_id, "execution", _name_parts(last[0])[1])
                                       / "execution_receipt.json").exists():
                notes.append("an execution receipt file exists but the journal does not record an accepted "
                             "execution; it is not treated as evidence")
        except (ValueError, OSError, KeyError, AttributeError, TypeError, EvidenceError) as exc:
            result["receipt_bound"] = False
            issues.append(f"receipt binding check failed: {exc}")
        # 5. run-root files
        try:
            input_raw = read_bytes(base / "input.json", limit=SMALL_JSON_LIMIT)
            if state.get("input_sha256") != digest(input_raw):
                issues.append("input.json sha256 does not match run.json input_sha256")
        except (ValueError, OSError) as exc:
            issues.append(f"input.json unreadable: {exc}")
        if isinstance(state.get("result"), dict) or passed:
            try:
                if read_json(base / "result.json", limit=SMALL_JSON_LIMIT) != state.get("result"):
                    issues.append("result.json differs from run.json result")
            except (ValueError, OSError) as exc:
                issues.append(f"result.json unreadable: {exc}")
        # 6. what algorithm_passed requires
        if passed:
            if not algorithm:
                issues.append("algorithm_passed on a backend other than algorithm_v1")
            if chain is not None:
                names = chain["journals"]
                if last is None:
                    issues.append("algorithm_passed but there is no execution journal")
                elif not last[1]["verified"] or last[1]["update"].get("error") or not isinstance(
                        last[1]["update"].get("result"), dict):
                    issues.append("algorithm_passed but the execution journal is not a verified success")
                done = sorted((n for n in names if n.startswith("done-")), key=_name_parts)
                if not done:
                    issues.append("algorithm_passed but there is no done journal")
                else:
                    u = names[done[-1]]["update"] or {}
                    if (not names[done[-1]]["verified"] or u.get("status") != "algorithm_passed" or u.get("error")
                            or (last is not None and u.get("result") != (last[1]["update"] or {}).get("result"))):
                        issues.append("algorithm_passed but the done journal is not a verified success "
                                      "carrying the execution result")
                for stage, keys in (("controller", ("plan_path", "plan_sha256")),
                                    ("sim", ("artifact_path", "artifact_sha256"))):
                    stage_names = sorted((n for n in names if n.startswith(stage + "-")), key=_name_parts)
                    u = (names[stage_names[-1]]["update"] or {}) if stage_names and names[stage_names[-1]]["verified"] else {}
                    for key in keys:
                        if not u or state.get(key) != u.get(key):
                            issues.append(f"run.json {key} differs from the latest {stage} journal")
                outputs = sorted((n for n in names if n.startswith("output-")), key=_name_parts)
                if not outputs or not names[outputs[-1]]["verified"] or state.get("verification") != (
                        names[outputs[-1]]["update"] or {}).get("verification"):
                    issues.append("run.json verification differs from the latest output journal")
            for key in ("plan_bytes_identical", "archive_ok", "receipt_bound"):
                if result[key] is not True:
                    issues.append(f"algorithm_passed but {key} is {result[key]!r}")
        if notes:
            result["notes"] = notes
        result["ok"] = journals_ok and not issues
        return result

    @mcp.tool()
    def verify_run_artifacts(run_id: str) -> dict[str, Any]:
        """Recompute every journal hash, require the journal directories run.json's attempts imply, rebuild failure history/attempts/final status from the journals and compare with run.json, check input.json/result.json, plan bytes, archive and receipt binding. algorithm_passed needs verified execution and done journals and a bound receipt. Failures are returned, never raised. Limit: without an external root of trust this cannot detect a file changed together with its recorded hashes (including a recomputed update_sha256) or a fully consistent forged run."""
        return _verify(run_id)

    @mcp.tool()
    def get_acceptance_summary(run_id: str) -> dict[str, Any]:
        """Layered acceptance: workflow status (only 'workflow complete' when verify_run_artifacts is ok), unaccepted core items, deferred physical items, real failures. algorithm_passed is only 'workflow complete'."""
        s = store.run_state(run_id)
        verdict = _verify(run_id)
        status = s.get("status")
        verification = s.get("verification") or {}
        result = s.get("result") or {}
        pending: list = []
        for item in [*(verification.get("pending_requirements") or []),
                     *((s.get("algorithm_experiment") or {}).get("pending_requirements") or [])]:
            if item not in pending:
                pending.append(item)
        core = list(pending)
        if verification.get("design_verified") is not True:
            core.append("design_verified is false: the experiment design itself has not been verified.")
        for name, value in sorted((verification.get("checks") or {}).items()):
            if value is not True:
                core.append(f"check '{name}' is not passed ({value!r}).")
        deferred = list(verification.get("deferred_optical_requirements") or [])
        deferred.append("Physical execution and measurement: not performed "
                        f"(motions_sent={result.get('motions_sent', 0)}, dry run only).")
        deferred.append("Camera calibration, reachability, collision avoidance and a real trajectory interface: "
                        "not part of this offline workflow.")
        failures = []
        if status in ("failed", "stopped") or (s.get("error") and status != "algorithm_passed"):
            if s.get("error"):
                failures.append(s["error"])
        if status == "algorithm_passed" and verdict["ok"] is True:
            label = ("workflow complete (algorithm_passed): the algorithm workflow finished as a dry run. "
                     "This is NOT 'all requirements verified'.")
        elif status == "algorithm_passed":
            label = ("workflow not complete: run.json says algorithm_passed but the evidence verification failed "
                     "(see evidence_issues)")
        else:
            label = f"workflow not complete (status={status!r})"
        # Router decisions come from the feedback files; a reroute is NOT a recovery.
        rerouted = []
        try:
            for d in store.chain(store.run_dir(run_id), s)["decisions"]:
                item = d["decision"]
                if item is not None and item.get("to") != "stop":
                    rerouted.append({"from": item.get("from"), "to": item.get("to"), "code": item.get("code")})
        except (ValueError, OSError):
            pass
        try:
            extra = store.provenance(run_id, s)
        except (ValueError, OSError) as exc:
            extra = {"provider": s.get("provider"), "provenance_error": str(exc)[:200]}
        return {
            **extra,
            "run_id": run_id, "workflow_status": label, "raw_status": status,
            "evidence_verified": verdict["ok"] is True, "evidence_issues": verdict["issues"],
            "core_unaccepted": core, "deferred_physical": deferred, "real_failures": failures,
            "rerouted_by_router": rerouted,
            "physical_verified": verification.get("physical_verified") is True and result.get("physical_verified") is True,
            "note": ("Which optical metrics are 'core' is defined by the optics owner. Items here are the "
                     "recorded pending/deferred requirements, not a verdict that the experiment works. "
                     "A reroute is not a recovery: check raw_status for the final outcome."),
        }

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="optics-evidence")
    parser.add_argument("--data-dir", default=os.environ.get("OPTICS_MCP_DATA_DIR") or str(DATA))
    args = parser.parse_args(argv)
    build_server(Path(args.data_dir)).run("stdio")


if __name__ == "__main__":
    main()
