# Decide: review (fresh context)

If this skill and the request's `instructions` conflict, the request wins. Run this in a fresh context: use only the `get_pending_decision` record (`data.request`, `intent`, `design`, `feedback`, `test_suite`, `test_report`). Do not read the designer's or test author's reasoning or any run file. Submit with a label ending `:ctx-fresh`.

1. Output only one JSON object matching `output_schema`: `accepted`, `issues`, `return_to`, `reasoning`, `problem_source`. No prose, no code fence.
2. Judge the design against the raw `data.request` yourself: missing or misread requirements, explicit numbers and units, component functions and counts, whether beam topology, distances and yaw are consistent. Audit the tests: reject vacuous tests, expected constants copied from the candidate, missing defining constraints, falsely attributed coverage, unjustified or loose tolerances. Deferred items must stay deferred, never called passed. Simulation and assembly come later; their absence is not a reason to reject.
3. A failed test cannot be overruled. If `test_report` has a failed or invalid test you may not accept. Decide whether the test is wrong (`problem_source` "tests": error in the test's basis or tolerance) or the design is wrong ("design"); say which in `issues`.
4. Consistency rules (a violation is rejected at submit):
   - `accepted` is true if and only if `issues` is empty; `reasoning` is non-empty.
   - `accepted` true needs `problem_source` "none" (use `return_to` "act"). `accepted` false needs `problem_source` one of "interpretation", "design", "tests".
   - `return_to` is "prompt" when `problem_source` is "interpretation", otherwise "act".
5. Never relax a user constraint, tolerance or target to get an acceptance. Never claim simulated or measured validation.
6. Shape only (do not copy values; the content must be your own judgement):
```json
{"accepted": false, "issues": ["<specific, checkable issue>"], "return_to": "act",
 "reasoning": "...", "problem_source": "design"}
```
   An acceptance is `{"accepted": true, "issues": [], "return_to": "act", "reasoning": "...", "problem_source": "none"}`.
