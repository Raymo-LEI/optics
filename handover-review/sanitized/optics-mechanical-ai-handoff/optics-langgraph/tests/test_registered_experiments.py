"""Regression coverage for the one-click entry and data-driven workbench."""
from copy import deepcopy
from threading import Event

import pytest
from fastapi.testclient import TestClient

from optics_graph.experiments import (catalog, load_experiment, match_registered_assembly,
                                     validate_registered_layout)
from optics_graph import server
from optics_graph.runtime import Runtime
from optics_graph.server import create_app
from optics_graph.team_trajectory import build_trajectory, check_trajectory


@pytest.mark.parametrize("key_configured", [False, True])
def test_workbench_starts_a_complete_non_injected_run(tmp_path, monkeypatch, key_configured):
    monkeypatch.setattr(server, "public_settings", lambda: {"key_configured": key_configured})
    completed = Event()
    class CaptureRuntime(Runtime):
        def execute(self, run_id, **kwargs):
            completed.set()

    with TestClient(create_app(CaptureRuntime(tmp_path))) as client:
        headers = {"X-Optics-Local":"1"}
        response = client.post("/api/experiments/runs", json={"request":"Assemble a complete Michelson"}, headers=headers)
        assert response.status_code == 202
        assert completed.wait(3)
        state = client.get("/api/runs/" + response.json()["run_id"]).json()
        assert state["profile"] == "auto" and state["mode"] == "github_local"
        assert state["provider"] == "openai"
        assert state["simulation_backend"] == "algorithm_v1"
        assert state["team_fault"] == "none" and not state["playback"] and not state["require_review"]
        rejected = client.post("/api/experiments/runs",
                               json={"request":"Assemble a Michelson","mode":"github_sim"}, headers=headers)
        assert rejected.status_code == 422


def test_plain_request_matches_only_compatible_registered_experiments():
    options = catalog()
    assert match_registered_assembly("Please assemble a complete Michelson interferometer with 10 or more components.", options) == "michelson_full"
    assert match_registered_assembly("Assemble a 7-component MZI.", options) == "mach_zehnder"
    assert match_registered_assembly("Assemble a complete Michelson interferometer with 10 components.", options) is None
    assert match_registered_assembly("Assemble a complete Michelson interferometer and move the mirrors.", options) is None
    assert match_registered_assembly("Please assemble a complete Michelson interferometer on a real optical table.", options) is None


@pytest.mark.parametrize("key,count", [("michelson_full",15),("mach_zehnder",7)])
def test_registered_inventory_controls_validation_and_trajectory(key, count):
    scene, layout = load_experiment(key)
    assert catalog()[key]["count"] == count
    assert validate_registered_layout(layout, scene, {})["passed"]
    observed = {name:{"position_m":[*part["feed_xy_m"],0.], "yaw_rad":0.}
                for name,part in scene["objects"].items()}
    plan = build_trajectory({"observations":observed},layout,scene,["0"*64,"1"*64],.20)
    check_trajectory(plan, set(scene["objects"]))
    assert len(plan["waypoints"]) == count * 10 + 1

    changed = deepcopy(layout)
    changed["components"][0]["target_pose"]["position_m"][0] += .015
    report = validate_registered_layout(changed, scene, {})
    assert not report["passed"] and "V_REGISTERED_ALIGNMENT" in {issue["code"] for issue in report["issues"]}


def test_complete_michelson_requires_independent_optical_acceptance(tmp_path, monkeypatch):
    from optics_graph import precision_contract, team_flow
    from optics_graph.runtime import RunInput

    scene, _ = load_experiment("michelson_full")
    source = scene["registration"]["source_sha256"]
    monkeypatch.setattr(team_flow, "legacy_manifest", lambda _: {
        "source_sha256":source, "calibration_sha256":"c"*64,
        "targets":[{"id":name} for name in scene["objects"]],
    })
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    monkeypatch.setattr(precision_contract, "observe_michelson", lambda *_: {
        "scene_id":"michelson_full", "frame":"world", "source_layout_sha256":source,
        "calibration_sha256":"c"*64,
        "observations":{name:{"T_world_object":identity} for name in scene["objects"]},
    })

    def failed_optical_run(root, output, *, timeout_s, added_noise):
        assert added_noise is False
        return {"success":False, "bindings_verified":True, "failure":None,
                "added_sensor_noise":{"rgb_std_8bit":0., "optical_std":0.},
                "verification":{"accepted":False,"mechanical":{"passed":True},
                                "optical":{"passed":False},"failure_reasons":["CAM_OVERLAP"]}}

    monkeypatch.setattr(team_flow, "run_legacy", failed_optical_run)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(request="Assemble a 15-component complete Michelson interferometer.",
                                    profile="auto",provider="openai",mode="github_local",
                                    team_transport="file_v1",playback=True))
    result = runtime.execute(run_id)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "PRECISION_ACCEPTANCE_FAILED"
    assert result["error"]["details"] == ["CAM_OVERLAP"]
    assert result["attempts"]["controller"] == 1
