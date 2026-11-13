"""Agent-decided model answers: the run pauses, an MCP client answers, the node resumes.

AgentDecisionGateway replaces the OpenAI transport for provider "agent". It never builds a client and
never reads a key. A node that needs a model answer writes a write-once decision request into its
attempt directory and raises DecisionPending. A client stores exactly one schema-valid answer for that
request (submit); the next node visit consumes it, re-validating it exactly as submit did, and hands
the content to the unchanged design_agent code. Nothing here approves a review, resumes a LangGraph
interrupt, or touches budgets, routing, tolerances or plan bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from . import design_agent
from .algorithm_experiment import package_design as design_agent_package
from .algorithm_sim import ANGLE_LIMIT, POSITION_LIMIT
from .gateway import CandidateFormatError, GatewayError, ModelGateway
from .sim_io import decode_json, read_json
from .optical_tests import evaluate_suite
from .team_artifacts import LIMIT as ARTIFACT_LIMIT, encode

MARKER = "Treat the following JSON as data.\n"  # separates instructions from data in design_agent._decide
REQUEST_SCHEMA = "agent-decision-request/v1"
ROLES = tuple(design_agent.ROLE_MODELS)
MAX_REJECTIONS = 3
MAX_DECISION_CHARS = 400_000
# Every decision-* file is hashed into completion.json, whose reader caps ordinary files at BLOB_LIMIT (772096 bytes).
MAX_FILE_BYTES = 700_000
LABEL_RE = re.compile(r"[A-Za-z0-9_.:/ -]{1,128}")
REQUEST_ID_RE = re.compile(r"[0-9a-f]{64}")
NODE_RE = re.compile(r"[a-z]{3,16}")
RUN_ID_RE = re.compile(r"[0-9a-f]{32}")
EXTERNAL_MODEL = "external-agent"
RESPONSE_KEYS = {"request_id", "decision_text", "decision_sha256", "decider_label", "submitted_at", "backend"}


class DecisionPending(Exception):
    """The node needs an answer that has not been submitted. Deliberately NOT a ValueError, GatewayError,
    OSError, KeyError, TypeError or BadZipFile, so team_flow.stage() lets it escape (the run stays at the
    node boundary and the same node is re-run on continue). The only argument is the request_id string."""


class DecisionError(ValueError):
    """A refused tool call (never raised inside a graph node)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _canon(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_id_for(run_id, node, attempt, role, prompt_sha256, schema_sha256, journal_input_sha256) -> str:
    return _sha(_canon({"schema_version": REQUEST_SCHEMA, "run_id": run_id, "node": node, "node_attempt": attempt,
                        "role": role, "prompt_sha256": prompt_sha256, "schema_sha256": schema_sha256,
                        "journal_input_sha256": journal_input_sha256}))


def record_identity(record: dict) -> str:
    return request_id_for(*(record[k] for k in ("run_id", "node", "node_attempt", "role", "prompt_sha256",
                                                "schema_sha256", "journal_input_sha256")))


def split_prompt(prompt: str) -> tuple[str, str]:
    if MARKER in prompt:
        instructions, data = prompt.split(MARKER, 1)
        return instructions, data
    return prompt, ""


def attempt_dir(run_dir: Path, node: str, attempt: int) -> Path:
    return Path(run_dir) / "team" / f"{node}-{attempt:02d}"


def _path(directory: Path, kind: str, role: str) -> Path:
    return directory / f"decision-{kind}-{role}.json"


def _scratch(path: Path) -> Path:
    """Where temporary files for `path` live. Never the journal directory itself (team/<node>-NN): stage()
    hashes every file it finds there into completion.json, so a leftover from a killed process would become
    evidence. The run directory is on the same filesystem, so a hard link or rename from it still works."""
    parent = path.parent
    return parent.parent.parent if parent.parent.name == "team" else parent


def _stage(path: Path, data: bytes) -> str:
    fd, temporary = tempfile.mkstemp(dir=_scratch(path), prefix=".tmp-decision-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return temporary


def exclusive_write(path: Path, data: bytes) -> None:
    """Create path with `data` or raise FileExistsError. Atomic and safe across processes: the content is
    complete before the name appears (hard link), and an existing name is never replaced.

    On a filesystem without hard links (os.link fails with anything other than "exists") it falls back to
    O_CREAT|O_EXCL on the final name plus fsync, which is still exclusive; a write that fails half way removes
    the file this call created. No temporary file is left behind on any path."""
    temporary = _stage(path, data)
    try:
        try:
            os.link(temporary, path)
            return
        except FileExistsError:
            raise
        except OSError:
            pass  # no hard links here: use the exclusive-create fallback below
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)  # FileExistsError propagates
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    finally:
        Path(temporary).unlink(missing_ok=True)


def _replace_write(path: Path, data: bytes) -> None:
    temporary = _stage(path, data)
    try:
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_rejections(directory: Path, role: str) -> list:
    path = _path(directory, "rejections", role)
    if not os.path.lexists(path):
        return []
    try:
        items = read_json(path, limit=None)
    except (ValueError, OSError):
        return [{"errors": ["rejections file unreadable"]}] * MAX_REJECTIONS  # fail closed: treat as exhausted
    if not isinstance(items, list):
        return [{"errors": ["rejections file malformed"]}] * MAX_REJECTIONS
    return items[:MAX_REJECTIONS]


def _format_errors(exc: ValidationError) -> list[str]:
    out = []
    for error in exc.errors()[:20]:
        where = ".".join(map(str, error["loc"])) or "(root)"
        out.append(f"{where}: {error['msg']}"[:300])
    return out


def validate_submission(role, decision_text, request_record):
    """The one validation gate, used by submit AND by the gateway when it consumes a stored answer.

    Returns the parsed document (a dict) on success, otherwise a non-empty list of error strings.
    Same chain as the OpenAI path: strict JSON (duplicate keys and non-finite numbers rejected), the role's
    strict pydantic model (extra fields forbidden), then the role's post-validation."""
    if role not in design_agent.ROLE_MODELS:
        return [f"unknown decision role {role!r}"]
    if not isinstance(decision_text, str) or not decision_text.strip():
        return ["decision is empty"]
    if len(decision_text) > MAX_DECISION_CHARS:
        return [f"decision exceeds {MAX_DECISION_CHARS} characters"]
    try:
        document = decode_json(decision_text)
    except ValueError as exc:
        return [f"decision is not valid strict JSON: {exc}"[:300]]
    if not isinstance(document, dict):
        return ["decision must be one JSON object"]
    try:
        content = design_agent.ROLE_MODELS[role].model_validate(document).model_dump()
    except ValidationError as exc:
        return _format_errors(exc)
    except (ValueError, TypeError) as exc:
        return [f"decision rejected: {exc}"[:300]]
    try:
        design_agent.postcheck(role, content, request_record["data"])
    except CandidateFormatError as exc:
        return [str(exc)[:300]]
    errors = [*tolerance_errors(role, content), *size_errors(role, content, request_record["data"])]
    return errors or document


# ---- agent-only rules on top of the shared post-validation --------------------------------------------------

_NUMBER = r"(?<![\w.])([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?|\.[0-9]+)"
# unit text -> factor to the field's own unit (position_tolerance_m: metres; angle_tolerance_rad: radians)
_LENGTH_UNITS = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "cm": 1e-2, "mm": 1e-3,
                 "um": 1e-6, "\u00b5m": 1e-6, "\u03bcm": 1e-6, "micron": 1e-6, "microns": 1e-6, "nm": 1e-9}
_ANGLE_UNITS = {"rad": 1.0, "radian": 1.0, "radians": 1.0, "mrad": 1e-3, "urad": 1e-6, "\u00b5rad": 1e-6,
                "\u03bcrad": 1e-6, "deg": math.pi / 180, "degree": math.pi / 180, "degrees": math.pi / 180,
                "\u00b0": math.pi / 180}


def _quoted_values(quotes, units) -> list[float]:
    """Every `<number> <unit>` in the quotes, converted to the field's unit. A number without a recognised
    unit is ignored on purpose (a bare "5" in "5 components" must never justify 5 metres)."""
    names = "|".join(sorted(map(re.escape, units), key=len, reverse=True))
    pattern = re.compile(_NUMBER + r"\s*(" + names + r")(?![A-Za-z])")
    return [float(m.group(1)) * units[m.group(2)] for quote in quotes for m in pattern.finditer(quote)]


def tolerance_errors(role, content) -> list[str]:
    """Agent runs only: an `interpret` answer may not loosen the acceptance tolerances.

    position_tolerance_m / angle_tolerance_rad that are null (or absent) fall back to the code's default
    execution limits (algorithm_sim.POSITION_LIMIT, ANGLE_LIMIT; see algorithm_experiment.package_design).
    A value tighter than or equal to the default is always allowed. A LOOSER value is rejected unless the
    user's request supports that exact number: some requirement's source_quote (already required to be a
    verbatim substring of the request) must contain `<number> <unit>` which, converted to the field's unit
    (metres, radians), equals the value within 0.1 percent. Only units are recognised, so a bare number
    never counts. This is strict by design: the quote must state the same value as the field. The check is
    on the number, not on the quote's wording, so it cannot tell a tolerance from another length of the same
    size in the request. OpenAI-provider behaviour is unchanged: this runs only for agent decisions."""
    if role != "interpret":
        return []
    quotes = [r["source_quote"] for r in content["requirements"]]
    errors = []
    for field, default, units, unit_name, example in (
            ("position_tolerance_m", POSITION_LIMIT, _LENGTH_UNITS, "metres", "'5 mm' supports 0.005"),
            ("angle_tolerance_rad", ANGLE_LIMIT, _ANGLE_UNITS, "radians", "'2 deg' supports 0.0349066")):
        value = content[field]
        if value is None or value <= default * (1 + 1e-9):
            continue
        if not any(math.isclose(value, quoted, rel_tol=1e-3) for quoted in _quoted_values(quotes, units)):
            errors.append(
                f"{field}={value} is looser than the default execution limit ({default:.10g} {unit_name}) and no "
                f"requirement source_quote states that value with a unit. Omit it (null) to use the default, or "
                f"quote the request: a source_quote containing the number and unit ({example}) justifies it. "
                f"An agent may not loosen a tolerance")
    return errors


def _bytes(value) -> int:
    return len(encode(value, limit=None))


def size_errors(role, content, data) -> list[str]:
    """Reject at submit what a downstream encoder (team_artifacts.encode, limit LIMIT bytes) would refuse,
    which would otherwise kill the run deterministically: the answer itself, the decision record built from it,
    and for design/optical_tests the artifacts packaged from it (experiment/layout, test report)."""
    def over(label, value):
        size = _bytes(value)
        return [f"{role} answer is too large: {label} would be {size} bytes, over the workflow limit of "
                f"{ARTIFACT_LIMIT} bytes. Shorten it"] if size > ARTIFACT_LIMIT else []
    errors = over("the answer", content)
    if errors:
        return errors
    data = data if isinstance(data, dict) else {}
    decision = {"node": "x" * 16, "attempt": 100, "source": "agent", "model": EXTERNAL_MODEL, "response_id": "0" * 64,
                "action": "design_experiment", "diagnosis": "", "target_object_ids": [], "changes": {},
                "input_sha256": "0" * 64, "decided_by": {"label": "x" * 128, "verified": False}}
    if role == "interpret":
        errors += over("its decision record", {**decision, "diagnosis": content["objective"], "changes": {
            "requirements": content["requirements"], "assumptions": content["assumptions"]}})
    elif role == "design":
        errors += over("its decision record", {**decision, "diagnosis": content["rationale"],
                                               "target_object_ids": [c["object_id"] for c in content["components"]],
                                               "changes": {"connections": content["connections"],
                                                           "requirements_coverage": content["requirements_coverage"]}})
        if not errors and isinstance(data.get("intent"), dict) and isinstance(data.get("request"), str):
            try:
                design_agent_package(data["request"], data["intent"], content)
            except ValueError as exc:
                if str(exc) == "JSON exceeds 256 KB":
                    errors.append(f"design answer is too large: the experiment/target-layout artifacts built from it "
                                  f"would exceed the workflow limit of {ARTIFACT_LIMIT} bytes. Shorten rationale, "
                                  "explanations, assumptions or parameters")
                # any other failure is the graph's own format check (it charges the budget as for OpenAI)
            except (KeyError, TypeError):
                pass
    elif role == "optical_tests" and isinstance(data.get("intent"), dict) and isinstance(data.get("design"), dict):
        try:
            report = evaluate_suite(content, data["intent"], data["design"])
        except Exception:  # noqa: BLE001 - the validator reports a broken suite itself
            report = None
        if report is not None:
            errors += over("the test report computed from it", report)
    elif role == "review":
        errors += over("the stored review", {**content, "experiment_sha256": "0" * 64,
                                              "test_suite_sha256": "0" * 64, "test_report_sha256": "0" * 64})
    return errors


# ---- the gateway --------------------------------------------------------------------------------------

class AgentDecisionGateway(ModelGateway):
    def __init__(self, provider, run_dir, client=None):
        super().__init__("agent", run_dir)

    @staticmethod
    def _locate(run_dir: Path, attempt) -> tuple[str, str]:
        match = re.fullmatch(r"([a-z]{3,16})-([0-9]{2,3})", run_dir.name)
        if (match is None or run_dir.parent.name != "team" or not RUN_ID_RE.fullmatch(run_dir.parent.parent.name)
                or type(attempt) is not int or int(match.group(2)) != attempt):
            raise GatewayError("DECISION_LOCATION_INVALID: the decision must belong to a team/<node>-NN directory "
                               "whose number equals the attempt", code="DECISION_LOCATION_INVALID")
        return match.group(1), run_dir.parent.parent.name

    def _request(self, role: str, prompt: str, schema: dict, attempt: int = 1) -> dict:
        if role not in design_agent.ROLE_MODELS:
            raise GatewayError(f"DECISION_ROLE_UNKNOWN: {role!r} is not an agent-decidable role",
                               code="DECISION_ROLE_UNKNOWN")
        run_dir = Path(self.run_dir)
        # A model log is evidence this gateway writes itself when it consumes a decision. Whatever is on disk
        # before that point was not written by a consumed decision (a stale or hand-placed file), and
        # design_agent._decide would copy it to model-response.json if this call raises. Never trust it.
        (run_dir / f"model-{role}-{attempt}.json").unlink(missing_ok=True)
        node, run_id = self._locate(run_dir, attempt)
        instructions, payload = split_prompt(prompt)
        try:
            data = json.loads(payload) if payload else None
        except ValueError as exc:
            raise GatewayError("DECISION_REQUEST_INVALID: the prompt data is not JSON",
                               code="DECISION_REQUEST_INVALID") from exc
        try:
            input_path = run_dir / "input.json"
            journal_input_sha256 = _sha(input_path.read_bytes() if input_path.is_file() and not input_path.is_symlink()
                                        else b"")
            world_revision = (read_json(input_path, limit=None).get("summary") or {}).get("observed_revision")
        except (ValueError, OSError, AttributeError) as exc:
            raise GatewayError("DECISION_REQUEST_INVALID: the attempt input record is missing or unreadable",
                               code="DECISION_REQUEST_INVALID") from exc
        prompt_sha256 = _sha(prompt.encode("utf-8"))
        schema_sha256 = _sha(_canon(schema))
        request_id = request_id_for(run_id, node, attempt, role, prompt_sha256, schema_sha256, journal_input_sha256)
        fresh = {"schema_version": REQUEST_SCHEMA, "request_id": request_id, "run_id": run_id, "node": node,
                 "node_attempt": attempt, "role": role, "instructions": instructions, "data": data,
                 "prompt_sha256": prompt_sha256, "output_schema": schema, "schema_sha256": schema_sha256,
                 "journal_input_sha256": journal_input_sha256, "world_revision": world_revision}
        request_path = _path(run_dir, "request", role)
        record = self._request_record(request_path, fresh)
        response_path = _path(run_dir, "response", role)
        if not os.path.lexists(response_path):
            rejections = read_rejections(run_dir, role)
            if len(rejections) >= MAX_REJECTIONS:
                raise CandidateFormatError(
                    "AGENT_DECISION_INVALID: three submitted decisions for this request were rejected",
                    code="AGENT_DECISION_INVALID",
                    details={"request_id": request_id, "role": role, "last_errors": rejections[-1].get("errors")})
            raise DecisionPending(request_id)
        return self._consume(run_dir, response_path, record, role, attempt)

    @staticmethod
    def _request_record(path: Path, fresh: dict) -> dict:
        changed = GatewayError("DECISION_REQUEST_CHANGED: the stored decision request does not match this attempt's "
                               "request. Evidence was modified or the inputs changed", code="DECISION_REQUEST_CHANGED")
        if os.path.lexists(path):
            try:
                existing = read_json(path, limit=None)
            except (ValueError, OSError) as exc:
                raise changed from exc
            if not isinstance(existing, dict) or {k: v for k, v in existing.items() if k != "created_at"} != fresh \
                    or not isinstance(existing.get("created_at"), str):
                raise changed
            return existing
        record = {**fresh, "created_at": datetime.now(timezone.utc).isoformat()}
        # Key order is kept (no sort_keys) so a reader can re-derive prompt_sha256 from instructions + data.
        raw = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
        if len(raw) > MAX_FILE_BYTES:
            raise GatewayError("DECISION_REQUEST_TOO_LARGE: the request is too large for the evidence journal",
                               code="DECISION_REQUEST_TOO_LARGE")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            exclusive_write(path, raw)
        except FileExistsError:  # another process created it first: it must equal ours
            return AgentDecisionGateway._request_record(path, fresh)
        return record

    def _consume(self, run_dir: Path, response_path: Path, record: dict, role: str, attempt: int) -> dict:
        def refuse(message, **details):
            return GatewayError(f"DECISION_RESPONSE_INVALID: {message}", code="DECISION_RESPONSE_INVALID",
                                details=details or None)
        try:
            response = read_json(response_path, limit=None)
        except (ValueError, OSError) as exc:
            raise refuse("the stored decision response cannot be read") from exc
        if not isinstance(response, dict) or set(response) != RESPONSE_KEYS:
            raise refuse("the stored decision response has an unexpected shape")
        text = response["decision_text"]
        if (response["request_id"] != record["request_id"] or not isinstance(text, str)
                or response["decision_sha256"] != _sha(text.encode("utf-8"))
                or not isinstance(response["decider_label"], str) or not LABEL_RE.fullmatch(response["decider_label"])
                or not isinstance(response["submitted_at"], str) or not isinstance(response["backend"], str)):
            raise refuse("the stored decision response is not bound to this request")
        content = validate_submission(role, text, record)
        if isinstance(content, list):
            raise refuse("the stored decision fails validation", errors=content)
        log = {"role": role, "attempt": attempt, "model": EXTERNAL_MODEL, "response_id": record["request_id"],
               "usage": {}, "status": "completed", "provider": "agent",
               "decided_by": {"label": response["decider_label"], "verified": False},
               "submitted_at": response["submitted_at"], "content": content}
        (run_dir / f"model-{role}-{attempt}.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
        return content


# ---- helpers shared with the MCP tools and runtime ---------------------------------------------------------

def locate_pending(run_dir: Path, node: str, attempt: int, request_id: str) -> dict | None:
    """The run.json `pending_decision` entry for a DecisionPending raised at team/<node>-<attempt>."""
    directory = attempt_dir(run_dir, node, attempt)
    for role in ROLES:
        try:
            record = read_json(_path(directory, "request", role), limit=None)
        except (ValueError, OSError):
            continue
        if isinstance(record, dict) and record.get("request_id") == request_id:
            return {"request_id": request_id, "node": node, "attempt": attempt, "role": role,
                    "created_at": record.get("created_at")}
    return None


def _check_pending(pending) -> tuple[str, int, str, str]:
    if (not isinstance(pending, dict) or not isinstance(pending.get("node"), str)
            or not NODE_RE.fullmatch(pending["node"]) or type(pending.get("attempt")) is not int
            or not 1 <= pending["attempt"] <= 100 or pending.get("role") not in ROLES
            or not isinstance(pending.get("request_id"), str) or not REQUEST_ID_RE.fullmatch(pending["request_id"])):
        raise DecisionError("DECISION_STALE", "run.json carries no valid pending decision")
    return pending["node"], pending["attempt"], pending["role"], pending["request_id"]


def read_request(run_dir: Path, pending: dict) -> dict:
    """The stored request for the pending entry, self-checked (identity recomputed from its own fields)."""
    node, attempt, role, request_id = _check_pending(pending)
    path = _path(attempt_dir(run_dir, node, attempt), "request", role)
    try:
        record = read_json(path, limit=None)
    except (ValueError, OSError) as exc:
        raise DecisionError("DECISION_REQUEST_CHANGED", "the stored decision request cannot be read") from exc
    try:
        ok = (isinstance(record, dict) and record["request_id"] == request_id and record["node"] == node
              and record["node_attempt"] == attempt and record["role"] == role and record_identity(record) == request_id
              and isinstance(record["data"], dict | type(None)) and isinstance(record["instructions"], str)
              and _sha(_canon(record["output_schema"])) == record["schema_sha256"]
              # the text the model would have seen must still hash to prompt_sha256
              and _sha((record["instructions"] if record["data"] is None else
                        record["instructions"] + MARKER + json.dumps(record["data"], ensure_ascii=False)
                        ).encode("utf-8")) == record["prompt_sha256"])
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        raise DecisionError("DECISION_REQUEST_CHANGED", "the stored decision request does not match its identity")
    return record


def decision_state(run_dir: Path, pending: dict) -> dict:
    node, attempt, role, _ = _check_pending(pending)
    directory = attempt_dir(run_dir, node, attempt)
    rejections = read_rejections(directory, role)
    return {"submitted": os.path.lexists(_path(directory, "response", role)),
            "rejections": len(rejections), "rejections_remaining": max(0, MAX_REJECTIONS - len(rejections)),
            "exhausted": len(rejections) >= MAX_REJECTIONS,
            "completed": (directory / "completion.json").exists()}


def submit(run_dir: Path, pending: dict, request_id: str, decision_text: str, label: str, backend: str) -> dict:
    """Store ONE answer for the currently pending request. The caller holds the run's control lock and has
    already checked the run (agent provider, awaiting_decision, no worker). This writes at most one file:
    decision-response-<role>.json (exclusive create) or, on a validation failure, the rejections list."""
    node, attempt, role, current = _check_pending(pending)
    directory = attempt_dir(run_dir, node, attempt)
    if request_id != current or (directory / "completion.json").exists():
        raise DecisionError("DECISION_STALE", "this request_id is not the currently pending decision request")
    record = read_request(run_dir, pending)
    response_path = _path(directory, "response", role)
    if os.path.lexists(response_path):
        return _already(response_path, request_id, decision_text)
    rejections = read_rejections(directory, role)
    if len(rejections) >= MAX_REJECTIONS:
        raise DecisionError("DECISION_EXHAUSTED", "three decisions were rejected for this request; "
                            "call continue_run so the graph charges the budget and retries the node")
    content = validate_submission(role, decision_text, record)
    response = {"request_id": request_id, "decision_text": decision_text,
                "decision_sha256": _sha(decision_text.encode("utf-8")), "decider_label": label,
                "submitted_at": datetime.now(timezone.utc).isoformat(), "backend": backend}
    raw = encode(response, limit=None)
    if not isinstance(content, list) and len(raw) > MAX_FILE_BYTES:
        content = [f"decision is too large for the evidence journal ({len(raw)} bytes; limit {MAX_FILE_BYTES})"]
    if isinstance(content, list):
        entry = {"submitted_at": response["submitted_at"], "decision_sha256": response["decision_sha256"],
                 "errors": content, "decider_label": label}
        rejections = [*rejections, entry]
        _replace_write(_path(directory, "rejections", role), encode(rejections, limit=None))
        return {"accepted": False, "request_id": request_id, "errors": content,
                "rejections_remaining": max(0, MAX_REJECTIONS - len(rejections)),
                "exhausted": len(rejections) >= MAX_REJECTIONS}
    try:
        exclusive_write(response_path, raw)
    except FileExistsError:
        return _already(response_path, request_id, decision_text)
    return {"accepted": True, "idempotent_replay": False, "request_id": request_id, "stored": True,
            "next": "call continue_run to resume the node"}


def _already(response_path: Path, request_id: str, decision_text: str) -> dict:
    try:
        stored = read_json(response_path, limit=None)
        same = stored["request_id"] == request_id and stored["decision_text"] == decision_text
    except (ValueError, OSError, KeyError, TypeError):
        same = False
    if not same:
        raise DecisionError("DECISION_ALREADY_SUBMITTED", "a different decision is already stored for this request")
    return {"accepted": True, "idempotent_replay": True, "request_id": request_id, "stored": True,
            "next": "call continue_run to resume the node"}
