import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from optics_graph.team_artifacts import digest, encode, evaluate_poses, make_archive, read_archive


def evidence():
    plan = encode({"scene_id": "scene", "robot_model_id": "robot", "calibration_id": "cal",
                   "tool_frame": "tool", "waypoints": []})
    result = {"status": "succeeded", "error": None, "waypoints_completed": 0,
              "engine": "mujoco", "sim_version": "3.0", "poses": {}, "plan_hash": digest(plan)}
    return plan, result


def rewrite(blob, *, result=None, manifest=None):
    with ZipFile(BytesIO(blob)) as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    if result is not None:
        data["result.json"] = result
        m = json.loads(data["manifest.json"])
        m["result_hash"] = digest(result)
        data["manifest.json"] = encode(m)
    if manifest is not None:
        m = json.loads(data["manifest.json"])
        m.update(manifest)
        data["manifest.json"] = encode(m)
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, value in data.items():
            archive.writestr(name, value)
    return output.getvalue()


def test_valid_archive_and_corrupt_zip():
    plan, result = evidence()
    blob = make_archive(plan, result)
    assert read_archive(blob, plan)[1] == result
    for bad in (b"not zip", blob[:-10], blob + b"x" * 800000):
        with pytest.raises(ValueError):
            read_archive(bad, plan)


@pytest.mark.parametrize("change", [
    {"completed_at": "2026-01-01T00:00:00"}, {"completed_at": None},
    {"version": True}, {"plan_hash": "0" * 64},
])
def test_invalid_manifest(change):
    plan, result = evidence()
    with pytest.raises(ValueError):
        read_archive(rewrite(make_archive(plan, result), manifest=change), plan)


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("error"), lambda r: r.update(extra=1),
    lambda r: r.update(waypoints_completed=False), lambda r: r.update(poses=[]),
    lambda r: r.update(sim_version=None),
])
def test_invalid_result_fields(mutation):
    plan, result = evidence()
    blob = make_archive(plan, result)
    mutation(result)
    with pytest.raises(ValueError):
        read_archive(rewrite(blob, result=encode(result)), plan)


@pytest.mark.parametrize("raw", [b'{"x":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}', b'\xff'])
def test_invalid_json(raw):
    plan, result = evidence()
    with pytest.raises(ValueError):
        read_archive(rewrite(make_archive(plan, result), result=raw), plan)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0", None, 10**400])
def test_pose_values_rejected(value):
    layout = {"components": [{"object_id": "a", "target_pose": {"orientation_xyzw": [0, 0, 0, 1]}, "movable": True}]}
    result = {"poses": {"a": {"frame_id": "optical_table", "position_m": [value, 0, 0], "orientation_xyzw": [0, 0, 0, 1]}}}
    with pytest.raises(ValueError):
        evaluate_poses(result, layout, {}, {})
