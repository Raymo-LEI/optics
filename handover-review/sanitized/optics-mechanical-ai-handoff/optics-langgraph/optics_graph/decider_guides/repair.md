# Decide: repair (repair_act, repair_controller)

If this skill and the request's `instructions` conflict, the request wins. Source of truth: `get_pending_decision` (`data` with `node`, `request`, `design`, `current`, `diagnosis`, `allowed_ids`).

1. Output only one JSON object: `operation`, `object_id`, `reason`. No prose, no code fence, no other keys.
2. `operation` is one of `translate`, `align`, `verify`, `stop`. There are no magnitudes, coordinates, angles, gains or commands in this answer; the server computes the correction.
3. Rules (a violation is rejected at submit and the id is checked against `data.allowed_ids`):
   - `reason` non-empty and evidence based (cite the `diagnosis`).
   - `object_id` is either null or one of `data.allowed_ids`; never a neighbour, never another id, even if it looks related.
   - `translate` and `align` require exactly one allowed id.
   - `stop` requires `object_id` null.
   - `verify` takes an allowed id or null (read-only retest, only if a repeat reading can resolve it).
4. Choosing the operation: `translate` for a demonstrated position discrepancy (usually role `repair_controller`), `align` for a demonstrated orientation discrepancy (usually role `repair_act`), `stop` when the evidence supports no operation. A schema-valid but wrong operation class is accepted at submit and then fails inside the graph (AGENT_RECOVERY_INVALID), which charges the retry budget, so read `diagnosis` before choosing.
5. The diagnosis names ONE failure class at a time (a position mismatch first, the angle later) even when `data.current` already shows both. Answer only the class this request asks about. The single allowed id in `data.allowed_ids` is authoritative even if the journal's failed event listed more ids.
6. Preserve all design targets and the current poses of other parts.
7. Shape only (do not copy values; take the id from `data.allowed_ids`):
```json
{"operation": "translate", "object_id": "<the one id in allowed_ids>", "reason": "<diagnosis-based reason>"}
```
