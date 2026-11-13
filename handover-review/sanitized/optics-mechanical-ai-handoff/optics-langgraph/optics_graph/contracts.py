"""Draft 2020-12 business contracts and deterministic example-scene validation."""

import hashlib
import json
import math
from copy import deepcopy
from functools import cache
from itertools import combinations
from pathlib import Path

import rfc8785
from jsonschema import Draft202012Validator

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
VALIDATOR_VERSION = "phase1-example-1"
EPS = 1e-9
POSE_EPS = 1e-6
EXAMPLE_COORDINATE_LIMIT = 1000  # Prevent overflow and degenerate arithmetic from out-of-scale inputs.
ROLES = {
    "laser": False,
    "beam_splitter": True,
    "mirror_arm_x": True,
    "mirror_arm_y": True,
    "detector": False,
}
EXAMPLES = {
    "layout": "layout",
    "plan": "action_plan",
    "observation": "observation",
    "result": "execution_result",
}
ACTION_GROUP = ["observe_scene", "pick", "place", "verify_placement"]


def _unique_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"Duplicate JSON key: {key}")
        document[key] = value
    return document


def _load(path):
    raw = path.read_bytes()
    if len(raw) > 1_000_000:
        raise ValueError("Contract file exceeds 1 MB")
    document = json.loads(raw, object_pairs_hook=_unique_keys)
    rfc8785.dumps(document)  # Also rejects non-finite or unrepresentable numbers.
    return document


def load_example(name: str) -> dict:
    return _load(CONTRACTS / "examples" / f"{EXAMPLES[name]}.json")


def load_scene() -> dict:
    return _load(CONTRACTS / "scene.json")


def canonical_hash(document) -> str:
    return hashlib.sha256(rfc8785.dumps(document)).hexdigest()


@cache
def _validator(kind):
    schema = _load(CONTRACTS / "schemas" / "phase1.schema.json")
    Draft202012Validator.check_schema(schema)
    if kind is None:
        return Draft202012Validator(schema)
    schema.pop("oneOf")
    schema["$ref"] = f"#/$defs/{kind}"
    return Draft202012Validator(schema)


def validate_document(document, kind=None) -> dict:
    """Check a business document or the specified schema definition."""
    try:
        encoded = rfc8785.dumps(document)
    except (ValueError, TypeError, RecursionError):
        return {"passed": False, "issues": [
            {"code": "V_SCHEMA", "message": "Document is not representable as canonical JSON"}
        ]}
    if len(encoded) > 1_000_000:
        return {"passed": False, "issues": [
            {"code": "V_SCHEMA", "message": "Document exceeds 1 MB"}
        ]}
    errors = sorted(_validator(kind).iter_errors(document), key=lambda error: error.json_path)
    return {"passed": not errors, "issues": [
        {"code": "V_SCHEMA", "message": error.message, "path": error.json_path}
        for error in errors
    ]}


def _number(value, *, positive=False):
    return (type(value) in (int, float) and abs(value) <= EXAMPLE_COORDINATE_LIMIT
            and (value > 0 if positive else True))


def _values(value, count):
    return isinstance(value, list) and len(value) == count and all(map(_number, value))


def _difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _edges(polygon):
    return zip(polygon, polygon[1:] + polygon[:1])


def _polygon(value):
    if not isinstance(value, list) or len(value) < 3 or not all(_values(p, 2) for p in value):
        return False
    # A convex CCW boundary has every other vertex on the inside of each edge.
    for a, b in _edges(value):
        if any(_cross(_difference(b, a), _difference(p, a)) <= EPS
               for p in value if p is not a and p is not b):
            return False
        if math.dist(a, b) <= EPS:
            return False
    return True


def _pose_is_planar(pose, height):
    x, y, _, _ = pose["orientation_xyzw"]
    return (abs(math.hypot(*pose["orientation_xyzw"]) - 1) <= POSE_EPS
            and math.hypot(x, y) <= POSE_EPS
            and abs(pose["position_m"][2] - height) <= POSE_EPS)


def _pose_matches(a, b):
    return (a["frame_id"] == b["frame_id"]
            and math.dist(a["position_m"], b["position_m"]) <= EPS
            and min(math.dist(a["orientation_xyzw"], b["orientation_xyzw"]),
                    math.hypot(*(x + y for x, y in zip(a["orientation_xyzw"],
                                                     b["orientation_xyzw"])))) <= EPS)


def _yaw(pose):
    _, _, z, w = pose["orientation_xyzw"]
    return 2 * math.atan2(z, w)


def _center_height(pose, obj):
    return pose["position_m"][2] + obj["optical_center_height_m"]


def _rectangle(pose, dimensions):
    x, y = pose["position_m"][:2]
    w, h = (dimension / 2 for dimension in dimensions)
    c, s = math.cos(_yaw(pose)), math.sin(_yaw(pose))
    return [(x + a * c - b * s, y + a * s + b * c)
            for a, b in [(-w, -h), (w, -h), (w, h), (-w, h)]]


def _edge_distance(a, b, point):
    direction = _difference(b, a)
    return _cross(direction, _difference(point, a)) / math.hypot(*direction)


def _within(footprint, boundary, clearance=0):
    return all(_edge_distance(a, b, p) >= clearance - EPS
               for p in footprint for a, b in _edges(boundary))


def _point_segment_distance(point, a, b):
    direction = _difference(b, a)
    t = max(0, min(1, _dot(_difference(point, a), direction) / _dot(direction, direction)))
    return math.dist(point, (a[0] + t * direction[0], a[1] + t * direction[1]))


def _polygon_gap(a, b):
    separated = any(all(_edge_distance(p, q, v) < -EPS for v in other)
                    for own, other in [(a, b), (b, a)] for p, q in _edges(own))
    if not separated:
        return 0.0
    return min(_point_segment_distance(v, p, q)
               for own, other in [(a, b), (b, a)]
               for p, q in _edges(own) for v in other)


def _segment_intersects(start, end, polygon):
    # Clip a finite segment by the inward half-plane of each polygon edge.
    enter, exit = 0.0, 1.0
    for a, b in _edges(polygon):
        d0 = _edge_distance(a, b, start)
        d1 = _edge_distance(a, b, end)
        if d0 < -EPS and d1 < -EPS:
            return False
        if (d0 < -EPS) != (d1 < -EPS):
            t = (-EPS - d0) / (d1 - d0)
            if d0 < -EPS:
                enter = max(enter, t)
            else:
                exit = min(exit, t)
    return enter <= exit


def _scene_errors(scene):
    """Validate fields used by the example solver before doing any geometry."""
    if not isinstance(scene, dict):
        return ["scene"]
    errors = []
    for key in ("scene_revision", "constraint_revision"):
        if not isinstance(scene.get(key), str) or not scene[key]:
            errors.append(key)
    if scene.get("frame_id") != "optical_table":
        errors.append("frame_id")
    if scene.get("calibration_status") != "example_only_unmeasured":
        errors.append("calibration_status")
    for key in ("workspace_polygon_m", "illustrative_reach_polygon_m"):
        if not _polygon(scene.get(key)):
            errors.append(key)
    for key in ("optical_axis_height_m", "support_height_m", "minimum_clearance_m"):
        value = scene.get(key)
        if not _number(value) or (key == "minimum_clearance_m" and value <= 0):
            errors.append(key)
    for key in ("request_tolerance", "placement_tolerance_limit"):
        if not validate_document(scene.get(key), "tolerance")["passed"]:
            errors.append(key)
    objects = scene.get("objects")
    if not isinstance(objects, dict) or len(objects) != len(ROLES):
        errors.append("objects")
    else:
        roles, markers = set(), set()
        for object_id, obj in objects.items():
            if not isinstance(object_id, str) or not object_id or not isinstance(obj, dict):
                errors.append("objects")
                continue
            role, marker = obj.get("role"), obj.get("marker_id")
            if (not isinstance(role, str) or role not in ROLES or role in roles
                    or type(marker) is not int or not 0 <= marker <= 49 or marker in markers
                    or obj.get("movable") is not ROLES[role]):
                errors.append(f"objects.{object_id}.identity")
                continue
            roles.add(role)
            markers.add(marker)
            dims = obj.get("footprint_m")
            if not _values(dims, 2) or min(dims) <= EPS:
                errors.append(f"objects.{object_id}.footprint_m")
            dimensions = ("height_m", "optical_center_height_m", "aperture_radius_m")
            if not all(_number(obj.get(key), positive=True) for key in dimensions):
                errors.append(f"objects.{object_id}.optics")
            elif obj["optical_center_height_m"] > obj["height_m"]:
                errors.append(f"objects.{object_id}.optical_center_height_m")
            if obj["movable"]:
                if obj.get("grasp_profile_id") != "top_grasp_v1":
                    errors.append(f"objects.{object_id}.grasp_profile_id")
            elif not validate_document(obj.get("fixed_pose"), "pose")["passed"]:
                errors.append(f"objects.{object_id}.fixed_pose")
    obstacles = scene.get("obstacles")
    if not isinstance(obstacles, list):
        errors.append("obstacles")
    else:
        ids = set(objects) if isinstance(objects, dict) else set()
        for index, obstacle in enumerate(obstacles):
            if (not isinstance(obstacle, dict)
                    or not isinstance(obstacle.get("obstacle_id"), str)
                    or not obstacle["obstacle_id"] or obstacle["obstacle_id"] in ids
                    or not _polygon(obstacle.get("polygon_m"))
                    or not _number(obstacle.get("height_m"), positive=True)):
                errors.append(f"obstacles.{index}")
            else:
                ids.add(obstacle["obstacle_id"])
    return errors


def _reflect(direction, normal):
    factor = 2 * _dot(direction, normal)
    return (direction[0] - factor * normal[0], direction[1] - factor * normal[1])


def _trace(components, scene, footprints, issue):
    by_role = {c["role"]: c["object_id"] for c in components}
    optical = {}
    for c in components:
        object_id, pose = c["object_id"], c["target_pose"]
        yaw = _yaw(pose)
        optical[object_id] = (pose["position_m"][:2], (math.cos(yaw), math.sin(yaw)))
    laser, splitter, mirror_x, mirror_y, detector = (by_role[role] for role in ROLES)
    height = scene["optical_axis_height_m"]
    occluders = [
        (c["object_id"], footprints[c["object_id"]], c["target_pose"]["position_m"][2],
         scene["objects"][c["object_id"]]["height_m"])
        for c in components
    ]
    occluders += [(o["obstacle_id"], o["polygon_m"], scene["support_height_m"], o["height_m"])
                  for o in scene["obstacles"]]

    def hit(origin, direction, source_id, target_id):
        center, normal = optical[target_id]
        divisor = _dot(direction, normal)
        distance = _dot(_difference(center, origin), normal) / divisor if abs(divisor) > EPS else 0
        point = (origin[0] + distance * direction[0], origin[1] + distance * direction[1])
        offset = math.dist(point, center)
        radius = scene["objects"][target_id]["aperture_radius_m"]
        leg = {"source_id": source_id, "target_id": target_id}
        if distance <= EPS or offset > radius + EPS:
            issue("V_RAY", "Ray missed the forward optical aperture", **leg,
                  expected={"forward": True, "max_aperture_offset_m": radius},
                  actual={"ray_parameter_m": distance, "aperture_offset_m": offset}, unit="m")
            return None
        for occluder_id, polygon, base, obstacle_height in occluders:
            if (occluder_id not in (source_id, target_id)
                    and base - EPS <= height <= base + obstacle_height + EPS
                    and _segment_intersects(origin, point, polygon)):
                issue("V_RAY", "Ray is blocked by an opaque footprint",
                      **leg, occluder_id=occluder_id)
                return None
        return point

    origin, emitted = optical[laser]
    split_point = hit(origin, emitted, laser, splitter)
    if split_point is None:
        return {}
    splitter_normal = optical[splitter][1]
    lengths, detector_points = {}, []
    for arm, mirror, outgoing in [("x", mirror_x, emitted),
                                  ("y", mirror_y, _reflect(emitted, splitter_normal))]:
        mirror_point = hit(split_point, outgoing, splitter, mirror)
        if mirror_point is None:
            continue
        returning = _reflect(outgoing, optical[mirror][1])
        return_point = hit(mirror_point, returning, mirror, splitter)
        if return_point is None:
            continue
        output = _reflect(returning, splitter_normal) if arm == "x" else returning
        detector_point = hit(return_point, output, splitter, detector)
        if detector_point is not None:
            lengths[arm] = math.dist(split_point, mirror_point)
            detector_points.append(detector_point)
    if len(detector_points) == 2:
        delta = math.dist(*detector_points)
        radius = scene["objects"][detector]["aperture_radius_m"]
        if delta > radius + EPS:
            issue("V_RAY", "The returning spots do not overlap within the detector tolerance",
                  object_id=detector, expected=radius, actual=delta, unit="m")
    return lengths


def _requirements_valid(requirements, object_ids):
    if not isinstance(requirements, dict) or requirements.keys() - {
        "object_targets", "arm_lengths_m", "max_position_error_m", "max_angle_error_rad"
    }:
        return False
    targets = requirements.get("object_targets", {})
    lengths = requirements.get("arm_lengths_m", {})
    return (isinstance(targets, dict) and targets.keys() <= object_ids
            and all(validate_document(pose, "pose")["passed"] and _values(pose["position_m"], 3)
                    for pose in targets.values())
            and isinstance(lengths, dict) and lengths.keys() <= {"x", "y"}
            and all(_number(length, positive=True) for length in lengths.values())
            and all(_number(requirements[key], positive=True)
                    for key in ("max_position_error_m", "max_angle_error_rad") if key in requirements))


def _validation_id(report):
    fields = ("layout_sha256", "scene_sha256", "requirements_sha256", "validator_version")
    return "validation-" + canonical_hash({key: report[key] for key in fields})[:24]


def validate_layout(layout: dict, scene: dict | None = None, requirements: dict | None = None) -> dict:
    schema_result = validate_document(layout, "layout")
    report = {
        "passed": False, "issues": schema_result["issues"], "checks": [],
        "validator_version": VALIDATOR_VERSION, "validation_id": None,
        "layout_sha256": None, "scene_sha256": None, "requirements_sha256": None,
        "layout_id": None, "scene_revision": None, "constraint_revision": None,
    }

    def issue(code, message, **details):
        report["issues"].append({"code": code, "message": message, **details})

    def checked(code, **details):
        report["checks"].append({"code": code, "passed": not any(
            entry["code"] == code for entry in report["issues"]
        ), **details})

    if not schema_result["passed"]:
        checked("V_SCHEMA")
        return report

    report.update({key: layout[key] for key in ("layout_id", "scene_revision", "constraint_revision")})
    report["layout_sha256"] = canonical_hash(layout)
    scene = load_scene() if scene is None else scene
    try:
        scene_hash = canonical_hash(scene)
    except (ValueError, TypeError, RecursionError):
        issue("CONSTRAINT_CONFIG_MISSING", "Scene must be representable as canonical JSON")
        return report
    scene_errors = _scene_errors(scene)
    if scene_errors:
        issue("CONSTRAINT_CONFIG_MISSING", "Invalid or missing example scene configuration", fields=scene_errors)
        return report
    report["scene_sha256"] = scene_hash
    for key in ("scene_revision", "constraint_revision"):
        if layout[key] != scene[key]:
            issue("V_SCHEMA", "Layout revision differs from scene configuration", field=key,
                  expected=scene[key], actual=layout[key])
    for key, limit in scene["placement_tolerance_limit"].items():
        if layout["tolerance"][key] > limit + EPS:
            issue("V_SCHEMA", "Placement tolerance exceeds scene limit", field=key,
                  expected=limit, actual=layout["tolerance"][key], unit=key.split("_")[-1])
    components = layout["components"]
    ids = {c["object_id"] for c in components}
    if ids != scene["objects"].keys() or len(ids) != len(components):
        issue("V_COMPONENT_SET", "Exactly the five distinct scene objects are required",
              expected=sorted(scene["objects"]), actual=[c["object_id"] for c in components])
        checked("V_COMPONENT_SET")
        checked("V_SCHEMA")
        return report

    for component in components:
        object_id = component["object_id"]
        obj = scene["objects"][object_id]
        for key in ("role", "movable", "marker_id"):
            if component[key] != obj[key]:
                issue("V_COMPONENT_SET", "Object identity differs from scene inventory",
                      object_id=object_id, field=key, expected=obj[key], actual=component[key])
        if not obj["movable"] and not _pose_matches(component["target_pose"], obj["fixed_pose"]):
            issue("V_COMPONENT_SET", "Fixed optical component pose was changed",
                  object_id=object_id, expected=obj["fixed_pose"], actual=component["target_pose"])
        if not _pose_is_planar(component["target_pose"], scene["support_height_m"]):
            issue("V_SCHEMA", "Target pose requires a unit yaw quaternion and the support-plane height",
                  object_id=object_id, actual=component["target_pose"],
                  expected={"support_height_m": scene["support_height_m"], "pose_epsilon": POSE_EPS})

    checked("V_SCHEMA")
    checked("V_COMPONENT_SET")
    if report["issues"]:
        return report

    if any(not _values(c["target_pose"]["position_m"], 3) for c in components):
        issue("V_WORKSPACE", "Component position exceeds the numerical range of the example scene")
        checked("V_WORKSPACE")
        return report

    footprints = {}
    clearance = scene["minimum_clearance_m"]
    for component in components:
        object_id, pose = component["object_id"], component["target_pose"]
        obj = scene["objects"][object_id]
        footprint = footprints[object_id] = _rectangle(pose, obj["footprint_m"])
        if not _within(footprint, scene["workspace_polygon_m"], clearance):
            issue("V_WORKSPACE", "Rotated footprint and clearance exceed the workspace boundary",
                  object_id=object_id, expected={"clearance_m": clearance}, actual=footprint, unit="m")
        if obj["movable"] and not _within(footprint, scene["illustrative_reach_polygon_m"]):
            issue("V_REACH_PRECHECK", "Footprint exceeds the illustrative reach region", object_id=object_id)
        for obstacle in scene["obstacles"]:
            distance = _polygon_gap(footprint, obstacle["polygon_m"])
            if distance < clearance - EPS:
                issue("V_WORKSPACE", "Footprint is too close to an exclusion region", object_id=object_id,
                      obstacle_id=obstacle["obstacle_id"], expected=clearance, actual=distance, unit="m")
        center_height = _center_height(pose, obj)
        if abs(center_height - scene["optical_axis_height_m"]) > POSE_EPS:
            issue("V_RAY", "Optical center is off the common beam height", object_id=object_id,
                  expected=scene["optical_axis_height_m"], actual=center_height, unit="m")
    for a, b in combinations(footprints, 2):
        distance = _polygon_gap(footprints[a], footprints[b])
        if distance < clearance - EPS:
            issue("V_OVERLAP", "Component footprints do not meet minimum clearance",
                  object_ids=[a, b], expected=clearance, actual=distance, unit="m")
    checked("V_WORKSPACE")
    checked("V_OVERLAP")
    checked("V_REACH_PRECHECK", calibration_status=scene["calibration_status"])
    ray_ready = all(check["passed"] for check in report["checks"]) and not any(
        entry["code"] == "V_RAY" for entry in report["issues"]
    )
    lengths = _trace(components, scene, footprints, issue) if ray_ready else {}
    if ray_ready and len(lengths) != 2:
        issue("V_TOPOLOGY", "Both round trips must connect the laser to the selected detector port",
              expected=["x", "y"], actual=sorted(lengths))
    if ray_ready:
        checked("V_TOPOLOGY")
        checked("V_RAY", arm_lengths_m=lengths)
    elif any(entry["code"] == "V_RAY" for entry in report["issues"]):
        checked("V_RAY")
    report["arm_lengths_m"] = lengths

    requirements = {} if requirements is None else requirements
    if not _requirements_valid(requirements, ids):
        issue("V_REQUEST", "Unsupported or malformed structured request", reason="UNSATISFIABLE_REQUEST")
    else:
        report["requirements_sha256"] = canonical_hash(requirements)
        _check_requirements(requirements, layout, lengths, scene, issue)
    checked("V_REQUEST")
    report["passed"] = not report["issues"]
    if report["passed"]:
        report["validation_id"] = _validation_id(report)
    return report


def _orientation_error(a, b):
    a = [x / math.hypot(*a) for x in a]
    b = [x / math.hypot(*b) for x in b]
    chord = min(math.dist(a, b), math.hypot(*(x + y for x, y in zip(a, b))))
    return 4 * math.asin(min(1, chord / 2))


def _check_requirements(requirements, layout, lengths, scene, issue):
    position_limit = requirements.get("max_position_error_m", scene["request_tolerance"]["position_m"])
    angle_limit = requirements.get("max_angle_error_rad", scene["request_tolerance"]["angle_rad"])
    for request_field, tolerance_field, unit in (
        ("max_position_error_m", "position_m", "m"),
        ("max_angle_error_rad", "angle_rad", "rad"),
    ):
        limit = requirements.get(request_field)
        actual = layout["tolerance"][tolerance_field]
        if limit is not None and actual > limit:
            issue("V_REQUEST", "Placement tolerance exceeds the requested maximum error",
                  reason="UNSATISFIABLE_REQUEST", field=tolerance_field,
                  expected=limit, actual=actual, unit=unit)
    by_id = {component["object_id"]: component["target_pose"] for component in layout["components"]}
    for object_id, requested in requirements.get("object_targets", {}).items():
        if not _pose_is_planar(requested, scene["support_height_m"]):
            issue("V_REQUEST", "Requested pose is incompatible with the support plane",
                  reason="UNSATISFIABLE_REQUEST", object_id=object_id, actual=requested)
            continue
        actual = by_id[object_id]
        errors = (
            ("position", math.dist(requested["position_m"], actual["position_m"]), position_limit, "m"),
            ("angle", _orientation_error(requested["orientation_xyzw"], actual["orientation_xyzw"]),
             angle_limit, "rad"),
        )
        for field, error, limit, unit in errors:
            if error > limit + EPS:
                issue("V_REQUEST", "Target pose does not meet the requested constraint",
                      reason="UNSATISFIABLE_REQUEST", object_id=object_id, field=field,
                      expected=limit, actual=error, unit=unit)
    for arm, requested in requirements.get("arm_lengths_m", {}).items():
        actual = lengths.get(arm)
        if actual is None or abs(actual - requested) > position_limit + EPS:
            issue("V_REQUEST", "Arm length does not meet the requested constraint",
                  reason="UNSATISFIABLE_REQUEST", arm=arm, expected=requested, actual=actual,
                  tolerance=position_limit, unit="m")


def _matches_validation(layout, layout_hash, validation):
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        return False
    fields = ("layout_id", "scene_revision", "constraint_revision")
    if any(validation.get(key) != layout[key] for key in fields):
        return False
    if validation.get("validator_version") != VALIDATOR_VERSION or validation.get("layout_sha256") != layout_hash:
        return False
    for field in ("scene_sha256", "requirements_sha256"):
        value = validation.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return False
    return validation.get("validation_id") == _validation_id(validation)


def validate_plan(plan, layout, validation: dict) -> dict:
    """Match a plan to a server-provided validation record and the validated layout."""
    plan_schema = validate_document(plan, "plan")
    layout_schema = validate_document(layout, "layout")
    report = {
        "passed": False, "issues": plan_schema["issues"] + layout_schema["issues"],
        "plan_sha256": None, "layout_sha256": None,
    }
    if report["issues"]:
        return report
    report["plan_sha256"] = canonical_hash(plan)
    report["layout_sha256"] = layout_hash = canonical_hash(layout)

    def issue(code, message, **details):
        report["issues"].append({"code": code, "message": message, **details})

    if (not _matches_validation(layout, layout_hash, validation)
            or plan["layout_sha256"] != layout_hash or plan["layout_id"] != layout["layout_id"]
            or plan["validation_id"] != validation["validation_id"]):
        issue("PLAN_VALIDATION_MISMATCH", "Plan, layout and validation record do not match")
        return report

    actions = plan["actions"]
    ids = [action["action_id"] for action in actions]
    if len(set(ids)) != len(ids):
        issue("PLAN_ACTION_ID", "Action IDs must be unique")
    for index, action in enumerate(actions):
        expected = [ids[index - 1]] if index else []
        if action["depends_on"] != expected:
            issue("PLAN_DEPENDENCY", "Dependencies must form a strict linear chain",
                  action_id=ids[index], expected=expected, actual=action["depends_on"])

    sequence = ["initialize", *ACTION_GROUP * 3, "home"]
    if [action["primitive"] for action in actions] != sequence:
        issue("PLAN_SEQUENCE", "Expected initialize, three observe/pick/place/verify groups, and home")
        return report

    components = {component["object_id"]: component for component in layout["components"]}
    movable = {key for key, component in components.items() if component["movable"]}
    moved = set()
    observed = {}
    for start in (1, 5, 9):
        observe, pick, place, verify = actions[start:start + 4]
        observe_args = observe["args"]
        observe_objects = observe_args["object_ids"]
        if (len(set(observe_objects)) != len(observe_objects)
                or set(observe_objects) - components.keys()
                or len(set(observe_args["camera_ids"])) != len(observe_args["camera_ids"])):
            issue("PLAN_OBSERVATION_REF", "Observation lists duplicate IDs or unknown scene objects",
                  action_id=observe["action_id"])
        observed[observe["action_id"]] = observe_objects
        object_id = pick["args"]["object_id"]
        if object_id not in movable or object_id in moved:
            issue("PLAN_OBJECT", "Each movable object must be picked exactly once",
                  action_id=pick["action_id"], object_id=object_id)
            continue
        moved.add(object_id)
        if (object_id not in observed.get(pick["args"]["observation_ref"], [])
                or object_id not in observe_objects):
            issue("PLAN_OBSERVATION_REF", "Pick must cite a preceding observation covering the picked object",
                  action_id=pick["action_id"], object_id=object_id)
        for action in (place, verify):
            args = action["args"]
            if args["object_id"] != object_id:
                issue("PLAN_OBJECT", "Pick, place and verify must refer to the same movable object",
                      action_id=action["action_id"], expected=object_id, actual=args["object_id"])
            if not _pose_matches(args["target_pose"], components[object_id]["target_pose"]):
                issue("PLAN_TARGET", "Action target differs from the validated target pose",
                      action_id=action["action_id"], object_id=object_id)
            if any(abs(args["tolerance"][key] - limit) > EPS for key, limit in layout["tolerance"].items()):
                issue("PLAN_TOLERANCE", "Action tolerance differs from the validated tolerance",
                      action_id=action["action_id"], object_id=object_id)
    if moved != movable:
        issue("PLAN_OBJECT", "Plan must cover all movable objects", missing_object_ids=sorted(movable - moved))
    report["passed"] = not report["issues"]
    return report


def build_plan(layout, validation) -> dict:
    """Build a deterministic replay plan for the validated example scene."""
    if not validate_document(layout, "layout")["passed"]:
        raise ValueError("Expected a structurally valid layout")
    layout_hash = canonical_hash(layout)
    if not _matches_validation(layout, layout_hash, validation):
        raise ValueError("Layout has no matching validation record")
    plan = load_example("plan")
    plan.update(
        plan_id=f"plan-{layout_hash[:16]}", layout_id=layout["layout_id"],
        layout_sha256=layout_hash, validation_id=validation["validation_id"],
    )
    by_role = {component["role"]: component for component in layout["components"]}
    for start, role in zip((1, 5, 9), ("beam_splitter", "mirror_arm_x", "mirror_arm_y")):
        component = by_role[role]
        object_id = component["object_id"]
        observe, pick, place, verify = plan["actions"][start:start + 4]
        observe["args"]["object_ids"] = [object_id]
        pick["args"]["object_id"] = object_id
        for action in (place, verify):
            action["args"].update(
                object_id=object_id, target_pose=deepcopy(component["target_pose"]),
                tolerance=deepcopy(layout["tolerance"]),
            )
    return plan
