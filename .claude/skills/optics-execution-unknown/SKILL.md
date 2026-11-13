---
name: optics-execution-unknown
description: Use when get_execution_status reports unknown, or a run failed with ALGORITHM_EXECUTION_UNKNOWN (an execution request was started but no receipt exists). Query, verify, and hand to a person; never retry.
---

# Optics execution unknown

An execution-started marker without a receipt means the result of the hand-off is not known. Executing again could apply the motion twice.

1. Query: `get_execution_status(run_id)`. Confirm the state (`unknown`) and note `attempt`, `marker_present`, the parsed `marker` (its `plan_sha256` and `input_sha256`), the `reason` and any error. `get_execution_status` is the only way to read the marker: `execution-started.json` is deliberately not on the `get_stage_artifact` whitelist (same as the web panel's route), so do not try to fetch it there.
2. Verify: `verify_run_artifacts(run_id)` and `get_stage_journal(run_id, "execution", attempt)`. Report `plan_bytes_identical`, `receipt_bound`, `journals_ok` and the journal `issues`. In this situation an incomplete execution journal (no completion.json, or a completion that records ALGORITHM_EXECUTION_UNKNOWN) is EXPECTED and is not by itself tampering; a stray `execution_receipt.json` is not evidence that the executor accepted the plan. Any other hash mismatch, missing journal directory or run.json disagreement is still a finding to report.
3. Hand to a person: report run_id, `plan_sha256`, artifact hash, the marker content from `get_execution_status`, and that no journal-verified receipt exists. Ask the owner (mechanical and AI) to decide.
4. Do NOT call `continue_run` for this run, start a new run to "redo" it, or alter any file. Stay read-only.
5. In the report, state clearly that the offline fixture (and an agent-decided run, which uses the same dry-run executor) sends no real motion (motions_sent=0 in dry run) but that the same rule applies to a real executor.
