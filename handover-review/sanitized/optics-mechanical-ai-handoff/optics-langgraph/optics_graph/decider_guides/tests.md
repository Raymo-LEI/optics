# Decide: optical tests (fresh context)

If this skill and the request's `instructions` conflict, the request wins; `instructions` includes the exact expression grammar and the JSON schema. Run this in a fresh context: use only the `get_pending_decision` record (`data.request`, `data.intent`, `data.design`, `data.feedback`). Do not read the designer's reasoning or any run file. Submit with a label ending `:ctx-fresh`.

1. Output only one JSON object matching `output_schema`: `title`, `tests`, `deferred`. No prose, no code fence.
2. Each test: `id` (unique), `title`, `requirement_ids`, `expression`, `expected`, `comparison` (`approx`, `le` or `ge`), `absolute_tolerance` (>= 0), `unit`, `rationale`.
3. Expected values come from the request or a stated physical law (write the law in `rationale`). NEVER copy a number from the candidate design to make a test pass; `expression` reads the design, `expected` states what the request or physics requires. A test whose two sides are the same expression, or that references no design data, is invalid.
4. Grammar (see `instructions` for the full text): numbers, `pi`, `+ - * / **` (exponent magnitude at most 8), `abs sqrt sin cos`; data functions take quoted strings only: `distance('A','B')`, `angle('A','B','C')`, `coordinate('A','x')`, `yaw('A')`, `parameter('A','exact name','unit')`, `count('kind')`, `connected('A','B')`, `path_length('A','B',...)`. Use the exact component ids and parameter names from `data.design`. Keep units consistent on both sides and for the tolerance.
5. Coverage: every `data.intent.requirements` id must appear in some test's `requirement_ids` or in `deferred` (with `requirement_id` and `reason`); an unknown id fails. A requirement partly testable may have both. Defer only what needs a physical model or measurement that is not available; do not defer anything computable from the design.
6. Tolerance rule: the tolerance must not be looser than the requirement it checks. Take it from an explicit requirement or from numeric precision; a stated 0.1 mm requirement may not be tested with 1 mm. Write it in full decimal notation (`0.000001`, not `1e-6`). Never widen a tolerance to make a test pass.
7. Server-side checks happen after `continue_run`: an invalid expression, unknown id or coverage mismatch is recorded as OPTICAL_TEST_INVALID and charges the retry budget. A computed mismatch is OPTICAL_TEST_FAILED and returns to design. Verify your formulas before submitting.
8. Shape only (do not copy values):
```json
{"title": "...", "deferred": [{"requirement_id": "r3", "reason": "..."}],
 "tests": [{"id": "t1", "title": "...", "requirement_ids": ["r1"],
   "expression": "distance('A','B')", "expected": "<value from the request or law>",
   "comparison": "approx", "absolute_tolerance": 0.000001, "unit": "m", "rationale": "..."}]}
```
