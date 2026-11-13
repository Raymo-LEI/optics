"""Registered optical layouts on the local Cartesian workbench."""
from itertools import combinations
import math
import re

from .contracts import (CONTRACTS, _load, _orientation_error, _polygon_gap,
                        _pose_is_planar, _rectangle, _requirements_valid, _within,
                        _check_requirements, canonical_hash, load_example, load_scene)

SIMPLE = "phase1_michelson"
REGISTERED_PREFIX = "registered-"
WORKSPACE = [[0., 0.], [.85, 0.], [.85, .74], [0., .74]]


def catalog():
    registry = _load(CONTRACTS / "experiments.json")
    return {SIMPLE: {"title": "Simplified Michelson", "count": 5,
                     "aliases": ["Michelson"]},
            **{key: {"title": item["title"], "count": len(item["parts"]), "aliases": item["aliases"]}
               for key, item in registry.items()}}


def is_registered(scene):
    return isinstance(scene.get("registration"), dict)


def _count_clause(name):
    return rf"(?P<{name}>\d{{1,2}})(?P<{name}_more>\s+or\s+more)?\s*-?\s*components?"


def match_registered_assembly(request, options):
    """Match plain assembly commands locally; send other wording to the intent parser."""
    for key, item in options.items():
        if key == SIMPLE:
            continue
        aliases = "|".join(re.escape(alias) for alias in sorted(item["aliases"], key=len, reverse=True))
        pattern = (
            r"\s*(?:please\s+)?(?:assemble|build|run|test)\s+(?:(?:a|an|one)\s+)?"
            r"(?:complete\s+)?(?:" + _count_clause("before") + r"\s+)?"
            r"(?:complete\s+)?(?:" + aliases + r")(?:\s+interferometer)?"
            r"(?:\s+with\s+" + _count_clause("after") + r")?"
            r"(?:\s+on\s+the\s+optical\s+table)?\s*[.!]?\s*"
        )
        matched = re.fullmatch(pattern, request, re.IGNORECASE)
        if not matched or (matched["before"] and matched["after"]):
            continue
        stated = None
        for prefix in ("before", "after"):
            if matched[prefix]:
                stated = (int(matched[prefix]), bool(matched[prefix + "_more"]))
        if stated:
            n, more = stated
            if not (n < item["count"] < n + 10 if more else n == item["count"]):
                continue
        elif any(set(item["aliases"]) & set(other["aliases"]) for alias_key, other in options.items()
                 if alias_key != key) and "complete" not in request.lower():
            continue
        return key
    return None


def load_experiment(key):
    if key == SIMPLE:
        return load_scene(), load_example("layout")
    item = _load(CONTRACTS / "experiments.json")[key]
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", key):
        raise ValueError("Invalid experiment ID")
    parts = item["parts"]
    if not 1 <= len(parts) <= 16:
        raise ValueError("Registered workbench has sixteen feeder slots")
    layout = {"layout_id": key, "scene_revision": REGISTERED_PREFIX + key + "-" + canonical_hash(item)[:12],
              "constraint_revision": "registered-geometric-v1",
              "tolerance": {"position_m": .002, "angle_rad": .035}, "components": []}
    objects = {}
    for index, part in enumerate(sorted(parts, key=lambda p: p["id"])):
        name = part["id"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", name) or name in objects:
            raise ValueError("Invalid or duplicate registered component ID")
        w, d, h = part["size_m"]
        if not all(type(v) in (int,float) and .018 <= v <= .05 for v in (w,d)) or not .02 <= h <= .08:
            raise ValueError("Registered component exceeds feeder or tool limits")
        yaw = part["yaw_rad"]
        pose = {"frame_id": "optical_table", "position_m": part["position_m"],
                "orientation_xyzw": [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]}
        layout["components"].append({"object_id": name, "movable": True, "target_pose": pose})
        objects[name] = {"footprint_m": [w,d], "height_m": h, "movable": True,
                         "feed_xy_m": [round(.07 + .10*(index % 8), 4), -.14 if index < 8 else -.055],
                         "reference_pose": pose}
    scene = {"scene_revision":layout["scene_revision"], "constraint_revision":layout["constraint_revision"],
             "registration": {"title":item["title"], "source_sha256":item["source_sha256"]},
             "objects":objects, "workspace_polygon_m":WORKSPACE, "minimum_clearance_m":.002,
             "support_height_m":0., "request_tolerance":layout["tolerance"]}
    return scene, layout


def validate_registered_layout(layout, scene, requirements):
    issues = []
    def issue(code, message, **details):
        issues.append({"code":code, "message":message, **details})

    objects = scene["objects"]
    components = layout["components"]
    ids = [c["object_id"] for c in components]
    if (set(ids) != set(objects) or len(ids) != len(objects)
            or any(c["movable"] is not True for c in components)
            or any(layout[field] != scene[field] for field in ("scene_revision", "constraint_revision"))):
        issue("V_COMPONENT_SET", "Layout differs from the registered inventory")
        return {"passed":False, "issues":issues}
    footprints = {}
    for item in components:
        name, pose = item["object_id"], item["target_pose"]
        obj = objects[name]
        reference = obj["reference_pose"]
        if not _pose_is_planar(pose, 0.):
            issue("V_SCHEMA", "Invalid planar pose", object_id=name)
            continue
        if (math.dist(pose["position_m"], reference["position_m"]) > scene["request_tolerance"]["position_m"]
                or _orientation_error(pose["orientation_xyzw"], reference["orientation_xyzw"]) > scene["request_tolerance"]["angle_rad"]):
            issue("V_REGISTERED_ALIGNMENT", "Pose differs from the registered optical design", object_id=name)
        footprint = _rectangle(pose, obj["footprint_m"])
        footprints[name] = footprint
        if not _within(footprint, scene["workspace_polygon_m"], scene["minimum_clearance_m"]):
            issue("V_WORKSPACE", "Component footprint exceeds the workbench", object_id=name)
    for (a, first), (b, second) in combinations(footprints.items(), 2):
        if _polygon_gap(first, second) < scene["minimum_clearance_m"]:
            issue("V_OVERLAP", "Registered footprints lack clearance", object_ids=[a,b])
    if (not _requirements_valid(requirements, set(objects)) or "arm_lengths_m" in requirements):
        issue("V_REQUEST", "Request requires a different optical layout or validation model")
    elif len(footprints) == len(objects):
        _check_requirements(requirements, layout, {}, scene, issue)
    return {"passed":not issues, "issues":issues, "kind":"registered_layout_alignment",
            "source_sha256":scene["registration"]["source_sha256"]}


def tightened_tolerance(layout, requirements):
    for field, key in (("max_position_error_m", "position_m"), ("max_angle_error_rad", "angle_rad")):
        if field in requirements:
            layout["tolerance"][key] = min(layout["tolerance"][key], requirements[field])
    return layout
