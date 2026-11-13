"""Canned model answers for the optics agent-design flow, shared by the stand-in and agent-decision tests."""
from copy import deepcopy
from math import pi


TELESCOPE = ("Build a Keplerian telescope", "Objective focal length 200 mm, eyepiece focal length 50 mm", "Lens-center separation 250 mm")
MICHELSON = ("Build a Michelson interferometer", "Use a 632.8 nm laser", "Each arm is 300 mm long")


def component(oid, kind, role, x, y, *, yaw=0., movable=True, parameters=()):
    return {"object_id": oid, "label": role, "kind": kind, "role": role,
            "position_m": [x, y, 0.], "yaw_rad": yaw, "movable": movable,
            "parameters": [{"name": k, "value": v, "unit": u} for k, v, u in parameters]}


def fixture(quotes):
    if quotes[0] == TELESCOPE[0]:
        parts = [
            component("OBJ", "positive_lens", "Objective", .2, .15,
                      parameters=[("focal_length", 200., "mm")]),
            component("EYE", "positive_lens", "Eyepiece", .45, .15,
                      parameters=[("focal_length", 50., "mm")]),
        ]
        links = [("OBJ", "EYE", "The objective's converging beam passes through the shared focal plane and enters the eyepiece")]
        rationale = "Confocal positive lenses 250 mm apart form a 4x inverted telescope. The actual magnification must be measured."
        covers = [(["OBJ", "EYE"], rationale)] * 3
    else:
        parts = [
            component("LASER", "laser", "Fixed source", .1, .25, movable=False,
                      parameters=[("wavelength", 632.8, "nm")]),
            component("BS", "beam_splitter", "Split and recombine", .3, .25, yaw=-pi/4,
                      parameters=[("reflectance", .5, "1")]),
            component("MX", "mirror", "Straight-arm retroreflector", .6, .25),
            component("MY", "mirror", "Side-arm retroreflector", .3, .55, yaw=pi/2),
            component("DET", "detector", "Observe interference fringes", .3, .05, yaw=pi/2),
        ]
        links = [("LASER", "BS", "Incident"), ("BS", "MX", "Transmitted arm"), ("BS", "MY", "Reflected arm"),
                 ("MX", "BS", "Straight-arm return"), ("MY", "BS", "Side-arm return"), ("BS", "DET", "Combined output")]
        rationale = "Two orthogonal arms of 300 mm each. The return beams overlap at the splitter. Fringe visibility must be measured."
        covers = [(["LASER", "BS", "MX", "MY", "DET"], rationale),
                  (["LASER"], "Source wavelength 632.8 nm"), (["BS", "MX", "MY"], rationale)]
    requirements = [{"id": f"r{i}", "text": quote, "source_quote": quote}
                    for i, quote in enumerate(quotes)]
    intent = {"title": quotes[0], "objective": quotes[0], "requirements": requirements,
              "assumptions": ["Ideal thin components. Concept design only"], "clarification_questions": [],
              "position_tolerance_m":None,"angle_tolerance_rad":None}
    design = {"title": quotes[0], "rationale": rationale, "components": parts,
              "connections": [{"from_id": a, "to_id": b, "role": role} for a, b, role in links],
              "requirements_coverage": [{"requirement_id": req["id"], "component_ids": ids,
                                         "explanation": why}
                                        for req, (ids, why) in zip(requirements, covers)],
              "assumptions": intent["assumptions"], "unresolved_requirements": []}
    return intent, design


def optical_suite(quotes):
    if quotes[0] == TELESCOPE[0]:
        checks = [
            ("linked", ["r0"], "connected('OBJ','EYE')", "1", "1"),
            ("objective-focal", ["r1"], "parameter('OBJ','focal_length','mm')", "200", "mm"),
            ("eyepiece-focal", ["r1"], "parameter('EYE','focal_length','mm')", "50", "mm"),
            ("separation", ["r2"], "distance('OBJ','EYE')", ".25", "m"),
        ]
    else:
        checks = [
            ("mirrors", ["r0"], "count('mirror')", "2", "1"),
            ("wavelength", ["r1"], "parameter('LASER','wavelength','nm')", "632.8", "nm"),
            ("direct-arm", ["r2"], "distance('BS','MX')", ".3", "m"),
            ("branch-arm", ["r2"], "distance('BS','MY')", ".3", "m"),
        ]
    return {"title": quotes[0], "deferred": [], "tests": [
        {"id": name, "title": name, "requirement_ids": requirements,
         "expression": expr, "expected": expected, "comparison": "approx",
         "absolute_tolerance": 1e-8, "unit": unit, "rationale": "Check geometry or parameters as requested."}
        for name, requirements, expr, expected, unit in checks
    ]}


REVIEW_ACCEPTED = {"accepted": True, "issues": [], "return_to": "act", "problem_source": "none",
                   "reasoning": "The inventory, geometry, and optical path conceptually match the original request."}


def canned_answer(role, data, quotes, intent=None, design=None):
    """The answer a cooperative model gives for one role. `data` is the decision data for that role."""
    if intent is None or design is None:
        intent, design = fixture(quotes)
    if role == "interpret":
        response = intent
    elif role == "design":
        response = design
    elif role == "optical_tests":
        response = optical_suite(quotes)
    elif role == "review":
        response = REVIEW_ACCEPTED
    elif role in {"repair_act", "repair_controller"}:
        assert len(data["allowed_ids"]) == 1
        response = {"operation": "align" if role == "repair_act" else "translate",
                    "object_id": data["allowed_ids"][0], "reason": "Correct this round's pose error."}
    else:
        raise AssertionError(f"unexpected model role: {role}")
    return deepcopy(response)
