# Decide: design

If this skill and the request's `instructions` conflict, the request wins. Source of truth: `get_pending_decision` (`instructions`, `data` with `request`, `intent`, `observation`, `feedback`, `reference`, `output_schema`).

1. Output only one JSON object matching `output_schema`. No other keys, no prose, no code fence.
2. Fields: `title`, `rationale`, `components`, `connections`, `requirements_coverage`, `assumptions`, `unresolved_requirements`. A component has `object_id`, `label`, `kind`, `role`, `position_m` (exactly three numbers x, y, z in metres on the optical_table frame), `yaw_rad`, `movable`, `parameters` (list of `name`, `value`, `unit`).
3. After submit the graph checks the design; a violation fails the node and charges the retry budget (AGENT_DESIGN_CONTRACT), so check before submitting:
   - 1 to 128 components; `object_id` unique, matching `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`; `label`, `kind`, `role` non-empty; every pose number finite.
   - Every connection endpoint names a component in this design, not itself, `role` non-empty, no duplicated (from, to, role). The connections must link ALL components into one connected graph (add a support connection for holders).
   - `requirements_coverage` has exactly one entry per `intent.requirements` id (no unknown id, no duplicate), each with a non-empty `explanation` and non-empty, unique, existing `component_ids`.
   - `unresolved_requirements` makes the design fail ("Not yet satisfied"). Keep it empty unless the request itself demands a fact that cannot be provided before design. Downstream assembly, simulation and tests are not unresolved requirements.
4. Take positions, yaw and parameter values from the request and optical physics (focal distances, arm lengths, reflection angles). Parameter names and units must be exact and usable by tests (a test reads `parameter('ID','exact name','unit')`). Use real components, never placeholders to reach a count. Do not invent measured results; hardware and measurement caveats go in `assumptions` and `rationale`.
5. If `data.reference.previous_candidate` exists, repair the stated feedback issues in that candidate and keep unaffected components. Never change an optical target to hide a failure.
6. `kind` is free text chosen by the designer, and the test author's `count('<kind>')` expressions match it exactly. Use plain conventional names (for example mirror, beamsplitter, thin_lens, detector) and keep them consistent across components. Yaw convention, exactly as the design instructions give it: "Place component centers at target coordinates on the optical_table frame in metres (x, y, z), with yaw about +z in radians." and "Yaw rotates local +x, the optical surface normal or forward optical axis as applicable." (`design_agent.py`). Do not assume any other convention.
7. Write float fields with a decimal point (`0.0` not `0`).
8. Shape only (do not copy values):
```json
{"title": "...", "rationale": "...",
 "components": [{"object_id": "A", "label": "...", "kind": "...", "role": "...",
   "position_m": [0.0, 0.0, 0.0], "yaw_rad": 0.0, "movable": true,
   "parameters": [{"name": "focal_length", "value": 100.0, "unit": "mm"}]}],
 "connections": [{"from_id": "A", "to_id": "B", "role": "..."}],
 "requirements_coverage": [{"requirement_id": "r1", "component_ids": ["A"], "explanation": "..."}],
 "assumptions": ["..."], "unresolved_requirements": []}
```
