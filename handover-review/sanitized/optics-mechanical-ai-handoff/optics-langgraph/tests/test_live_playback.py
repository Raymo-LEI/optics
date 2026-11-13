from threading import Event

import pytest
from fastapi.testclient import TestClient

from optics_graph.gateway import ModelGateway
from optics_graph.runtime import RunInput, Runtime
from optics_graph.server import create_app, live_observation_progress
from optics_graph.settings import PROJECT
from test_team_flow import prob0


pytestmark = pytest.mark.skipif(not (PROJECT / ".venv-mujoco/bin/python").exists(),
                                reason="MuJoCo environment unavailable")


def team_input(**kwargs):
    return RunInput(mode="github_local", team_transport="file_v1",
                    team_entry="nl", prob0=prob0(), **kwargs)


def test_live_observation_counts_only_complete_consistent_rows(tmp_path):
    log = tmp_path/"vision.log"
    log.write_bytes(b'camera message\n'
                    b'{"object_id":"LASER","position_error_mm":2.1,"rotation_error_deg":1.6,"within_1mm_1deg":false}\n'
                    b'{"object_id":"ISO","position_error_mm":0.98,"rotation_error_deg":0.2,"within_1mm_1deg":true}\n'
                    b'{"object_id":"CAM","position_error_mm":2.0,"rotation_error_deg":0.2,"within_1mm_1deg":true}\n'
                    b'{"object_id":"SH0","position_error_mm":0.9,"rotation_error_deg":1.7,"within_1mm_1deg":false}')
    progress = live_observation_progress(log)
    assert progress["measured"] == 2 and progress["required"] == 15
    assert progress["latest"]["object_id"] == "ISO"
    targeted = live_observation_progress(log, {"LASER"})
    assert targeted["measured"] == targeted["required"] == 1
    assert targeted["latest"]["object_id"] == "LASER"


def test_workbench_reports_node_before_it_completes(tmp_path):
    entered, released, finished = Event(), Event(), Event()

    class SlowGateway(ModelGateway):
        def layout(self, *args):
            entered.set()
            if not released.wait(10):
                raise RuntimeError("test did not release the node")
            return super().layout(*args)

    class ObservedRuntime(Runtime):
        def execute(self, *args, **kwargs):
            try:
                return super().execute(*args, **kwargs)
            finally:
                finished.set()

    runtime = ObservedRuntime(tmp_path, SlowGateway)
    with TestClient(create_app(runtime)) as client:
        response = client.post("/api/runs", json=team_input().model_dump(), headers={"X-Optics-Local": "1"})
        assert response.status_code == 202
        path = "/api/runs/" + response.json()["run_id"]
        try:
            assert entered.wait(5)
            # Wait until the task-start event reaches the public API, while Act is still blocked.
            for _ in range(80):
                state = client.get(path).json()
                if state.get("live_node") == "act":
                    break
                released.wait(.02)
            assert state["live_node"] == "act"
            assert state["live_attempt"] == 1
            assert state["stage"] == "prompt"
        finally:
            released.set()
            assert finished.wait(10)
        assert "live_node" not in client.get(path).json()


def test_verified_playback_shows_actual_motion_and_rejects_tampering(tmp_path):
    runtime = Runtime(tmp_path)
    run_id = runtime.create(team_input(playback=True, team_fault="output"))
    state = runtime.execute(run_id)
    assert state["status"] == "team_l1_passed", state.get("error")
    assert state["attempts"]["sim"] == 2
    path = f"/api/runs/{run_id}/playback"
    with TestClient(create_app(runtime)) as client:
        response = client.get(path)
        assert response.status_code == 200
        recording = response.json()
        assert recording["plan_hash"] == state["plan_sha256"]
        assert recording["frames"][-1]["waypoint"] == 31
        assert max(pose["tool"][2] for pose in recording["frames"]) > .10
        assert any(a != b for a, b in zip(recording["frames"][0]["poses"], recording["frames"][-1]["poses"]))
        file = runtime.run_dir(run_id) / "team/sim-02/playback.json"
        file.write_bytes(file.read_bytes() + b" ")
        assert client.get(path).status_code == 409
