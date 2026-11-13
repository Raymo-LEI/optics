---
name: optics-decide-tests
description: Use when get_pending_decision shows role "optical_tests" in an agent-decided optics run. Authors the analytic test suite in a FRESH context that sees only the request record. Invoked from optics-agent-decide or by the independent reviewer agent.
---

# Decide: optical tests (fresh context)

If this skill and the request's `instructions` conflict, the request wins; `instructions` includes the exact expression grammar and the JSON schema. Run this in a fresh context: use only the `get_pending_decision` record (`data.request`, `data.intent`, `data.design`, `data.feedback`). Do not read the designer's reasoning or any run file. Submit with a label ending `:ctx-fresh`.

1. Output only one JSON object matching `output_schema`: `title`, `tests`, `deferred`. No prose, no code fence.
2. Each test: `id` (unique), `title`, `requirement_ids`, `expression`, `expected`, `comparison` (`approx`, `le` or `ge`), `absolute_tolerance` (>= 0), `unit`, `rationale`.
3. Expected values come from the request or a stated physical law (write the law in `rationale`). NEVER copy a number from the candidate design to make a test pass; `expression` reads the design, `expected` states what the request or physics requires. A test whose two sides are the same expression, or that references no design data, is invalid.
4. Grammar (see `instructions` for the full text): numbers, `pi`, `+ - * / **` (exponent magnitude at most 8), `abs sqrt sin cos`; data functions take quoted strings only: `distance('A','B')`, `angle('A','B','C')`, `coordinate('A','x')`, `yaw('A')`, `parameter('A','exact name','unit')`, `count('kind')`, `connected('A','B')`, `path_length('A','B',...)`. Use the exact component ids and parameter names from `data.design`. Keep units consistent on both sides and for the tolerance.
5. Coverage: every `data.intent.requirements` id must appear in some test's `requirement_ids` or in `deferred` (with `requirement_id` and `reason`); an unknown id fails. A requirement partly testable may have both. Defer only what needs a physical model or measurement that is not available; do not defer anything computable from the design.
6. Orientation coverage: positions alone do not define a beam path. For every component whose orientation decides where light goes (source, beam splitter, mirror, grating, lens, detector), include at least one test on its `yaw('ID')`. Distance and collinearity tests leave orientation unchecked: a source turned to emit away from the next element passes them all. Reviews reject such a suite, and each rejection charges the Act retry budget that later placement repairs also need. Under the design's yaw convention (yaw turns local +x, the forward optical axis or surface normal), "A points at B" is two tests: `cos(yaw('A'))` approx `(coordinate('B','x')-coordinate('A','x'))/distance('A','B')`, and the same with `sin` and `'y'`. That suits a source toward the next element, or a mirror at normal incidence toward the element it returns light to. Other parts take the condition their physics requires, for example a splitter at 45 degrees to the incoming beam. Compare through `cos` and `sin`, never raw angles, so a full turn cannot hide an error. Derive the expected direction from positions or the request, never from the design's yaw value.
7. Tolerance rule: the tolerance must not be looser than the requirement it checks. Take it from an explicit requirement or from numeric precision; a stated 0.1 mm requirement may not be tested with 1 mm. Write it in full decimal notation (`0.000001`, not `1e-6`). Never widen a tolerance to make a test pass.
8. Server-side checks happen after `continue_run`: an invalid expression, unknown id or coverage mismatch is recorded as OPTICAL_TEST_INVALID and charges the retry budget. A computed mismatch is OPTICAL_TEST_FAILED and returns to design. Verify your formulas before submitting.
9. Shape only (do not copy values):
```json
{"title": "...", "deferred": [{"requirement_id": "r3", "reason": "..."}],
 "tests": [{"id": "t1", "title": "...", "requirement_ids": ["r1"],
   "expression": "distance('A','B')", "expected": "<value from the request or law>",
   "comparison": "approx", "absolute_tolerance": 0.000001, "unit": "m", "rationale": "..."}]}
```
