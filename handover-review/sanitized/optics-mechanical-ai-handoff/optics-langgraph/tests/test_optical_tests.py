import ast
from copy import deepcopy
import math

import pytest

from optics_graph.optical_tests import evaluate_suite


def component(object_id, kind, xyz, **params):
    return {"object_id": object_id, "kind": kind, "position_m": xyz, "yaw_rad": 0,
            "parameters": [{"name": name, "value": value, "unit": unit}
                           for name, (value, unit) in params.items()]}


def check(test_id, expression, expected, unit="1", tolerance=1e-9):
    return {"id": test_id, "title": test_id, "requirement_ids": [test_id],
            "expression": expression, "expected": expected, "unit": unit,
            "absolute_tolerance": tolerance, "comparison": "approx", "rationale": "Model equation."}


def run(components, tests, *, current=None, deferred=None, connections=None, requirements=None):
    deferred = [] if deferred is None else deferred
    design = {"components": components, "connections": [] if connections is None else connections}
    intent = {"requirements": [{"id": i} for i in (
        [t["id"] for t in tests] + [d["requirement_id"] for d in deferred]
        if requirements is None else requirements
    )]}
    return evaluate_suite({"title": "Experiment", "tests": tests, "deferred": deferred},
                          intent, design, current)


def test_lens_image_plane_and_magnification_use_real_geometry():
    components = [component("O", "object", [0, 0, 0]),
                  component("L", "lens", [0.3, 0, 0], focal_length=(100, "mm")),
                  component("S", "screen", [0.45, 0, 0])]
    tests = [check("focus", "parameter('L','focal_length','m')",
                   "1 / (1 / distance('O','L') + 1 / distance('L','S'))", "m"),
             check("magnification", "distance('L','S') / distance('O','L')", "0.5")]
    report = run(components, tests)
    assert report["passed"]
    assert report["physical_verified"] is False
    assert report["tests"][0]["actual"] == pytest.approx(0.1)
    current = {c["object_id"]: deepcopy(c) for c in components}
    current["S"]["position_m"][0] = 0.46
    assert [t["status"] for t in run(components, tests, current=current)["tests"]] == ["failed", "failed"]
    del current["S"]
    assert "Current pose is missing" in run(components, tests, current=current)["tests"][0]["reason"]


def test_michelson_angle_and_arm_path_are_separate_checks():
    components = [component("X", "splitter", [0, 0, 0]), component("A", "mirror", [0.2, 0, 0]),
                  component("B", "mirror", [0, 0.2, 0])]
    tests = [check("orthogonal", "angle('A','X','B')", "pi / 2", "rad"),
             check("arms", "path_length('X','A','X')", "path_length('X','B','X')", "m"),
             check("topology", "connected('X','A') + connected('X','B')", "count('mirror')")]
    connections = [{"from_id": "X", "to_id": i} for i in ("A", "B")]
    result = run(components, tests, connections=connections)
    assert result["passed"]
    assert result["tests"][0]["actual"] == pytest.approx(math.pi / 2)
    components[2]["position_m"][1] = 0.3
    assert [t["status"] for t in run(components, tests, connections=connections)["tests"]] == [
        "passed", "failed", "passed"]


def test_opa_frequency_conservation_and_unmodeled_conversion():
    components = [component("P", "pump", [0, 0, 0], wavelength=(400, "nm")),
                  component("S", "signal", [0, 0, 0], wavelength=(600, "nm")),
                  component("I", "idler", [0, 0, 0], wavelength=(1.2, "um"))]
    tests = [check("frequencies", "299792458 / parameter('P','wavelength','m')",
                   "299792458 / parameter('S','wavelength','m') + 299792458 / parameter('I','wavelength','m')",
                   "Hz", 1)]
    deferred = [
        {"requirement_id": "efficiency", "reason": "Gain needs a coupled propagation model."},
        {"requirement_id": "efficiency", "reason": "Depletion needs pump power measurements."},
    ]
    result = run(components, tests, deferred=deferred)
    assert result["passed"]
    assert result["deferred"] == deferred
    assert not result["physical_verified"]
    components[2]["parameters"][0]["value"] = 1.3
    assert run(components, tests, deferred=deferred)["tests"][0]["status"] == "failed"


@pytest.mark.parametrize("actual, expected", [
    ("1", "1"), ("1", "2"), ("yaw('L')", "yaw(\"L\")"), ("count('missing')", "0"),
    ("yaw('missing')", "0"), ("__import__('os')", "0"), ("parameter('L','f','kg')", "0"),
    ("parameter('L','f','s')", "0"), ("parameter('L','missing','m')", "0"),
    ("yaw('L') / 0", "0"), ("sqrt(yaw('L') - 1)", "0"),
    ("yaw('L') + 1e999", "0"), ("yaw('L') + 2 ** 100", "0"),
    ("yaw('L') + (1).__class__", "0"), ("yaw('L') + sum([1])", "0"),
])
def test_unsafe_or_unsupported_formulas_are_invalid(actual, expected):
    components = [component("L", "lens", [0, 0, 0], f=(100, "mm"))]
    report = run(components, [check("test", actual, expected)])
    assert not report["passed"]
    assert report["tests"][0]["status"] == "invalid"


def test_invalid_suite_cannot_pass():
    components = [component("L", "lens", [0, 0, 0])]
    test = check("test", "yaw('L')", "0", "rad")
    for tests, requirements in [
        ([], []), ([test, test], ["test"]), ([test], ["test", "untested"]), ([test], ["different"]),
        ([dict(test, absolute_tolerance=-1)], ["test"]), ([dict(test, absolute_tolerance="0")], ["test"]),
    ]:
        report = run(components, tests, requirements=requirements)
        assert not report["passed"]
        assert report["error"]


def test_planar_reflection_vector_can_use_a_long_analytic_expression():
    mirror = component("M", "mirror", [0, 0, 0])
    mirror["yaw_rad"] = math.pi / 4
    components = [component("A", "source", [-1, 0, 0]), mirror,
                  component("B", "screen", [0, -1, 0])]
    axes = ("x", "y")
    incoming = {axis: f"(coordinate('M','{axis}') - coordinate('A','{axis}')) / distance('A','M')"
                for axis in axes}
    outgoing = {axis: f"(coordinate('B','{axis}') - coordinate('M','{axis}')) / distance('M','B')"
                for axis in axes}
    normal = {"x": "cos(yaw('M'))", "y": "sin(yaw('M'))"}
    dot = " + ".join(f"({incoming[axis]}) * {normal[axis]}" for axis in axes)
    residual = " + ".join(
        f"(({outgoing[axis]}) - ({incoming[axis]}) + 2 * ({dot}) * {normal[axis]}) ** 2"
        for axis in axes
    )
    assert len(list(ast.walk(ast.parse(residual, mode="eval")))) > 100
    tests = [check("reflection", residual, "0")]
    assert run(components, tests)["passed"]
    mirror["yaw_rad"] = math.pi / 6
    assert run(components, tests)["tests"][0]["status"] == "failed"


@pytest.mark.parametrize("source_unit", ["", "dimensionless", "unitless"])
def test_energy_and_dimensionless_unit_conversion(source_unit):
    components = [
        component("P", "pump", [0, 0, 0], energy=(2500, "pJ"), transmission=(0.75, source_unit)),
    ]
    tests = [check("pulse_energy", "parameter('P','energy','nJ')", "2.5", "nJ"),
             check("transmission", "parameter('P','transmission','%')", "75", "%")]
    report = run(components, tests)
    assert report["passed"]
    assert [t["actual"] for t in report["tests"]] == pytest.approx([2.5, 75])


def test_suite_can_cover_more_than_64_component_constraints():
    components = [component(f"M{i}", "mirror", [i / 100, 0, 0]) for i in range(65)]
    tests = [check(f"station{i}", f"coordinate('M{i}','x')", str(i / 100), "m") for i in range(65)]
    report = run(components, tests)
    assert report["passed"]
    assert len(report["tests"]) == len(components)


def test_ast_complexity_stays_bounded():
    components = [component("L", "lens", [0, 0, 0])]
    report = run(components, [check("complex", "yaw('L')" + " + 1" * 345, "345")])
    assert not report["passed"]
    assert report["tests"][0]["status"] == "invalid"
    assert "too complex" in report["tests"][0]["reason"]
