"""Bounded arithmetic for experiment-specific analytic checks."""
from __future__ import annotations

import ast
import json
import math
import operator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


_Text = Annotated[str, Field(min_length=1, max_length=1200)]
_Expression = Annotated[str, Field(min_length=1, max_length=2000)]
_Id = Annotated[str, Field(min_length=1, max_length=120)]


class OpticalTest(_Strict):
    id: _Id
    title: _Text
    requirement_ids: Annotated[list[_Id], Field(min_length=1, max_length=32)]
    expression: _Expression
    expected: _Expression
    comparison: Literal["approx", "le", "ge"]
    absolute_tolerance: Annotated[float, Field(ge=0)]
    unit: _Id
    rationale: _Text


class DeferredRequirement(_Strict):
    requirement_id: _Id
    reason: _Text


class TestSuite(_Strict):
    title: _Text
    tests: Annotated[list[OpticalTest], Field(min_length=1)]
    deferred: Annotated[list[DeferredRequirement], Field(max_length=128)]


_UNIT_GROUPS = (
    {"m": 1, "mm": 1e-3, "um": 1e-6, "nm": 1e-9},
    {"rad": 1, "deg": math.pi / 180},
    {"s": 1, "fs": 1e-15, "ps": 1e-12, "ns": 1e-9},
    {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12},
    {"W": 1, "mW": 1e-3},
    {"J": 1, "mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9, "pJ": 1e-12},
    {"1": 1, "%": 1e-2},
)
_UNITS = {name: (group, scale) for group, units in enumerate(_UNIT_GROUPS) for name, scale in units.items()}
_ALIASES = {"µm": "um", "μm": "um", "µJ": "uJ", "μJ": "uJ", "°": "deg",
            "": "1", "dimensionless": "1", "unitless": "1"}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_MATH = {"abs": abs, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos}
_ARITY = {"distance": 2, "angle": 3, "coordinate": 2, "yaw": 1, "parameter": 3,
          "count": 1, "connected": 2}


def _unit(name: str) -> tuple[int, float]:
    normalized = _ALIASES.get(name, name)
    if normalized not in _UNITS:
        raise ValueError(f"Unsupported unit: {name}")
    return _UNITS[normalized]


def _number(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric value")
    return float(value)


def _parse(expression: str) -> ast.expr:
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 1024:
        raise ValueError("Expression is too complex")
    return tree.body


class _Arithmetic:
    def __init__(self, design: dict, current: dict | None):
        self.components = {c["object_id"]: c for c in design["components"]}
        self.current = self.components if current is None else current
        self.connections = {(e["from_id"], e["to_id"]) for e in design["connections"]}
        self.references = 0

    def _pose(self, object_id: str) -> dict:
        self._component(object_id)
        if object_id not in self.current:
            raise ValueError(f"Current pose is missing for {object_id}")
        return self.current[object_id]

    def _position(self, object_id: str) -> tuple[float, ...]:
        xyz = self._pose(object_id)["position_m"]
        if len(xyz) != 3:
            raise ValueError(f"Expected a three-dimensional position for {object_id}")
        return tuple(_number(x) for x in xyz)

    def _component(self, object_id: str) -> dict:
        if object_id not in self.components:
            raise ValueError(f"Unknown component: {object_id}")
        return self.components[object_id]

    def _lookup(self, name: str, args: list[str]) -> float:
        if name == "count":
            count = sum(c["kind"] == args[0] for c in self.components.values())
            if count:
                self.references += 1
            return count
        self.references += 1
        self._component(args[0])
        if name in ("distance", "path_length"):
            points = [self._position(i) for i in args]
            return sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        if name == "angle":
            a, b, c = map(self._position, args)
            u, v = math.dist(a, b), math.dist(c, b)
            if not u or not v:
                raise ValueError("Angle is undefined for a zero-length segment")
            cosine = sum(((x - y) / u) * ((z - y) / v) for x, y, z in zip(a, b, c))
            return math.acos(max(-1, min(1, cosine)))
        if name == "coordinate":
            if args[1] not in ("x", "y", "z"):
                raise ValueError("Coordinate axis must be x, y or z")
            return self._position(args[0])["xyz".index(args[1])]
        if name == "yaw":
            return _number(self._pose(args[0])["yaw_rad"])
        if name == "connected":
            self._component(args[1])
            return float(tuple(args) in self.connections)
        matches = [p for p in self.components[args[0]]["parameters"] if p["name"] == args[1]]
        if len(matches) != 1:
            raise ValueError(f"Expected one parameter named {args[1]} on {args[0]}")
        param = matches[0]
        source_group, source_scale = _unit(param["unit"])
        target_group, target_scale = _unit(args[2])
        if source_group != target_group:
            raise ValueError(f"Cannot convert {param['unit']} to {args[2]}")
        return _number(param["value"]) * (source_scale / target_scale)

    def evaluate(self, node: ast.expr) -> float:
        if isinstance(node, ast.Constant):
            return _number(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = self.evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = self.evaluate(node.left), self.evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent magnitude must be at most 8")
            return _number(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in _MATH and len(node.args) == 1:
                return _number(_MATH[name](self.evaluate(node.args[0])))
            arity_ok = (2 <= len(node.args) <= 24 if name == "path_length"
                        else name in _ARITY and _ARITY[name] == len(node.args))
            if arity_ok and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in node.args):
                return _number(self._lookup(name, [a.value for a in node.args]))
        raise ValueError(f"Unsupported expression or argument: {type(node).__name__}")


_ERRORS = (ValueError, TypeError, KeyError, ArithmeticError, SyntaxError, RecursionError)


def evaluate_suite(suite: dict | TestSuite, intent: dict, design: dict, current: dict | None = None) -> dict:
    """Return deterministic results; malformed suites or tests cannot pass."""
    report = {"passed": False, "tests": [], "deferred": [],
              "scope": "analytic_design_checks", "physical_verified": False}
    try:
        parsed = TestSuite.model_validate(suite)
        report["title"] = parsed.title
        report["deferred"] = [item.model_dump() for item in parsed.deferred]
        requirements = {item["id"] for item in intent["requirements"]}
        ids = [item.id for item in parsed.tests]
        deferred_ids = [item.requirement_id for item in parsed.deferred]
        tested = {rid for item in parsed.tests for rid in item.requirement_ids}
        covered = tested | set(deferred_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate test IDs")
        if covered != requirements:
            raise ValueError(f"Requirement mismatch; missing: {sorted(requirements - covered)}; "
                             f"unknown: {sorted(covered - requirements)}")
        arithmetic = _Arithmetic(design, current)
    except (ValidationError, *_ERRORS) as exc:
        report["error"] = str(exc)
        return report

    for test in parsed.tests:
        result = test.model_dump(exclude={"expected"})
        result.update(expected_expression=test.expected, actual=None, expected=None)
        try:
            if len(set(test.requirement_ids)) != len(test.requirement_ids):
                raise ValueError("Duplicate requirement IDs in test")
            _unit(test.unit)
            left, right = _parse(test.expression), _parse(test.expected)
            if ast.dump(left) == ast.dump(right):
                raise ValueError("Actual and expected use the same expression")
            arithmetic.references = 0
            actual, expected = arithmetic.evaluate(left), arithmetic.evaluate(right)
            if not arithmetic.references:
                raise ValueError("Test must reference real design data")
            delta = _number(actual - expected)
            tolerance = test.absolute_tolerance
            passed = (abs(delta) <= tolerance if test.comparison == "approx"
                      else delta <= tolerance if test.comparison == "le"
                      else delta >= -tolerance)
            result.update(status="passed" if passed else "failed", actual=actual, expected=expected,
                          reason="Within comparison tolerance" if passed else "Outside comparison tolerance")
        except _ERRORS as exc:
            result.update(status="invalid", reason=str(exc))
        report["tests"].append(result)
    report["passed"] = all(item["status"] == "passed" for item in report["tests"])
    return report


def prompt_contract() -> str:
    return (
        "Generate executable analytic tests specific to this experiment. Return only a TestSuite. "
        "Map every intent requirement ID to one or more tests or document why it requires evidence outside this model in deferred. "
        "A requirement may have both tests and a deferral when the tests cover only part of it. "
        "Use actual component IDs and exact numeric parameter names. Each test must reference real design data; "
        "do not copy the same expression to both sides or use a constant-only test. "
        "expression and expected are arithmetic strings: numbers, pi, parentheses, + - * / ** "
        "(exponent magnitude at most 8), abs(x), sqrt(x), sin(x), cos(x). "
        "Data functions accept only quoted strings: distance('A','B') in meters, angle('A','B','C') "
        "at B in radians, coordinate('A','x') in meters (x/y/z), yaw('A') in radians, "
        "parameter('A','exact name','unit'), count('kind'), connected('A','B') for a directed connection (0 or 1), "
        "path_length('A','B','C',...) sums geometric segment lengths in meters. "
        "Choose consistent units on both sides and for absolute_tolerance. "
        "The engine converts parameter values explicitly and does not infer dimensions of arithmetic. "
        f"Supported unit labels: {', '.join(_UNITS)}. "
        "approx checks absolute error <= tolerance; le checks actual <= expected + tolerance; "
        "ge checks actual >= expected - tolerance. Justify each formula and tolerance using the provided intent or design. "
        "Do not represent unmodeled coupled physics or physical validation as passed tests. Schema: "
        + json.dumps(TestSuite.model_json_schema(), ensure_ascii=False)
    )
