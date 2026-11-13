# Decide: interpret

If this skill and the request's `instructions` conflict, the request wins. Source of truth: `get_pending_decision` (`instructions`, `data.request`, `output_schema`).

1. Output only one JSON object matching `output_schema`. No other keys, no prose, no code fence. Pass it as the `decision_json` string.
2. Fields: `title`, `objective`, `requirements` (each `id`, `text`, `source_quote`), `assumptions`, `clarification_questions`, `position_tolerance_m`, `angle_tolerance_rad`.
3. What gets rejected:
   - `source_quote` must be a verbatim, contiguous substring of `data.request` (same characters, same language). Copy it, do not paraphrase.
   - Requirement `id`s non-empty and unique; `text`, `title`, `objective` non-empty.
   - A tolerance, if given, must be positive. Tolerances are `null` unless the request states an explicit assembly or placement accuracy. Focal lengths, arm lengths and beam angles are design parameters, not tolerances.
   - Agent rule: a tolerance looser than the default execution limit is rejected unless some requirement `source_quote` contains that exact number with a unit (for example "5 mm" supports 0.005 metres, "2 deg" supports 0.0349066 radians; a bare number never counts). Tighter or equal is fine. Omit it (null) to use the default. Never loosen a tolerance to make something pass.
4. Keep every explicit number, unit and comparison from the request in the requirement text. Record reasonable defaults as `assumptions`. Missing real inventory is an assumption, not a blocker. Real hardware is not involved: assembly and execution refer to a simulated stage.
5. Shape only (do not copy these values; derive everything from this request):
```json
{"title": "...", "objective": "...",
 "requirements": [{"id": "r1", "text": "...", "source_quote": "<exact substring of data.request>"}],
 "assumptions": ["..."], "clarification_questions": [],
 "position_tolerance_m": null, "angle_tolerance_rad": null}
```
