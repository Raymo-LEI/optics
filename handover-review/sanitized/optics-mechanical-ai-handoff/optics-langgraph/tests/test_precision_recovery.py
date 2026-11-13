import json

import pytest
from fastapi.testclient import TestClient

from optics_graph import precision_contract, team_flow
from optics_graph.experiments import load_experiment
from optics_graph.gateway import ModelGateway
from optics_graph.legacy import EXPECTED_NAMES
from optics_graph.precision_plan import accept_precision_l1, checkpoint_files, validate_plan, verify_evidence
from optics_graph.precision_recovery import repair_evidence
from optics_graph.runtime import RunInput, Runtime
from optics_graph.team_artifacts import encode
from optics_graph.server import create_app


@pytest.fixture
def observation(monkeypatch):
    scene, _ = load_experiment("michelson_full")
    source = scene["registration"]["source_sha256"]
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    observed = {"schema_version":"1.0", "scene_id":"michelson_full", "frame":"world",
                "source_layout_sha256":source, "calibration_sha256":"f" * 64,
                "observations":{name:{"T_world_object":identity} for name in EXPECTED_NAMES}}
    monkeypatch.setattr(precision_contract, "observe_michelson", lambda *_: observed)
    monkeypatch.setattr(precision_contract, "observed_checkpoint", lambda *_, **kw: observed)
    monkeypatch.setattr(team_flow, "legacy_manifest", lambda _: {
        "source_sha256":source, "calibration_sha256":"f" * 64,
        "targets":[{"id":name} for name in EXPECTED_NAMES],
    })
    return source


def write_fixture(output, failure=None, recovery=None):
    output.mkdir()
    for name in ("world_model.mjb", "trajectory.npz"):
        (output/name).write_bytes(name.encode())
    from hashlib import sha256
    report = {"failure":failure, "objects":{name:{"visual_quality_passed":True} for name in EXPECTED_NAMES},
              "world_model_sha256":sha256((output/"world_model.mjb").read_bytes()).hexdigest(),
              "trajectory_sha256":sha256((output/"trajectory.npz").read_bytes()).hexdigest(),
              "recovery":recovery}
    (output / "result.json").write_bytes(encode(report))
    return report


@pytest.mark.parametrize(("failure", "agent", "action", "part", "guard"), [
    ("RuntimeError: FINAL_VISUAL_LAYOUT_CHECK_FAILED:L25", "precision_motion_agent", "reseat_drifted_parts", "L25", []),
    ("RuntimeError: FIRST_FRINGE_FEEDBACK:invalid", "precision_optics_agent", "extend_alignment_search", None, []),
    (None, "precision_metrology_agent", "guard_camera_margin", "CAM", ["CAM"]),
])
def test_local_agent_recovery_preserves_scene_and_limits_actuation(tmp_path, monkeypatch, observation,
                                                            failure, agent, action, part, guard):
    calls = []
    target = [part] if part else []
    common = {"bindings_verified":True, "added_sensor_noise":{"rgb_std_8bit":0., "optical_std":0.}}

    def physics(_project, output, *, recovery_actions=(), guard_parts=(), resume_dir=None,
                repair_part=None, repair_stage=None, **kwargs):
        calls.append((output, resume_dir, repair_part, repair_stage, list(guard_parts)))
        if not resume_dir:
            report = write_fixture(output, failure)
            if failure:
                return {**common, "success":False, "failure":failure}
            checks = {k:{"passed":True} for k in ("optical", "sequence", "bindings", "sensor_feedback")}
            return {**common, "success":False, "failure":"Simulation acceptance failed or the evidence source check did not pass.", "verification":{
                **checks, "accepted":False, "mechanical":{"passed":False,"objects":{
                    "CAM":{"passed":False}, "PD":{"passed":True}}},
                "failure_reasons":["FINAL_BODY_POSE_OUTSIDE_1MM_1DEG"]}}
        recovery = {"retained_scene":True, "source_report_sha256":checkpoint_files(resume_dir)["result.json"],
                    "target_object_ids":target, "commanded_object_ids":target,
                    "untouched_object_ids":sorted(EXPECTED_NAMES-set(target))}
        report = write_fixture(output, recovery=recovery)
        return {**common, "success":True, "source_sha256":observation, "trajectory_sha256":report["trajectory_sha256"],
                "recovery":recovery,
                "verification":{"accepted":True, "mechanical":{"passed":True}, "optical":{"passed":True}}}

    class Gateway(ModelGateway):
        def repair_precision(self, evidence, attempt):
            assert "guard_parts" not in evidence
            return super().repair_precision(evidence, attempt)

    monkeypatch.setattr(team_flow, "run_legacy", physics)
    runtime = Runtime(tmp_path, Gateway)
    run_id = runtime.create(RunInput(mode="github_sim", profile="auto", provider="replay",
                                    request="Assemble a 15-component complete Michelson interferometer."))
    state = runtime.execute(run_id)
    assert state["status"] == "team_precision_passed", state.get("error")
    assert [item[1:] for item in calls] == [(None, None, None, []),
                                            (calls[0][0], part, None if part else "optical", guard)]
    assert state["attempts"]["precision_sim"] == 1 and state["attempts"]["precision_repair"] == 1
    assert state["attempts"]["controller"] == 2
    assert state["retry_log"][0]["to"] == agent
    assert state["precision_repairs"][0]["action"] == action
    assert state["plan"]["object_ids"] == target
    assert state["verification"]["recovery"]["commanded_object_ids"] == target
    root = runtime.run_dir(run_id)
    plan = (root/state["plan_path"]).read_bytes()
    blob = (root/state["artifact_path"]).read_bytes()
    verify_evidence(blob, plan, state["precision_result"])
    with pytest.raises(ValueError):
        verify_evidence(blob, plan+b" ", state["precision_result"])
    plan_object = json.loads(plan)
    plan_object["object_ids"] = ["L25", "LASER"]
    with pytest.raises(ValueError, match="permitted correction scope"):
        validate_plan(plan_object, plan_object["target_layout_sha256"], plan_object["observed_layout_sha256"])
    with TestClient(create_app(runtime)) as client:
        base = f"/api/runs/{run_id}/precision/report"
        assert client.get(base).json()["recovery"]["target_object_ids"] == target
        assert client.get(base, params={"attempt":1}).json()["failure"] == failure
        assert client.get(base, params={"attempt":3}).status_code == 404


@pytest.mark.parametrize("bound", [True, False])
def test_unsupported_candidate_or_unbound_report_cannot_trigger_motion(tmp_path, monkeypatch, observation, bound):
    calls = []

    def physics(_project, output, **options):
        calls.append(options)
        failure = "RuntimeError: FINAL_VISUAL_LAYOUT_CHECK_FAILED:L25"
        write_fixture(output, failure)
        return {"bindings_verified":bound, "added_sensor_noise":{"rgb_std_8bit":0., "optical_std":0.},
                "success":False, "failure":failure}

    class Gateway(ModelGateway):
        def repair_precision(self, evidence, attempt):
            return {"action":"reacquire_views", "diagnosis":"unsupported for this incident"}

    monkeypatch.setattr(team_flow, "run_legacy", physics)
    runtime = Runtime(tmp_path, Gateway)
    run_id = runtime.create(RunInput(mode="github_sim", profile="auto", provider="replay",
                                    request="Assemble a 15-component complete Michelson interferometer."))
    state = runtime.execute(run_id)
    assert state["status"] == "failed"
    assert len(calls) == 1
    assert state.get("precision_actions") is None
    if bound:
        assert state["error"]["code"] == "PRECISION_REPAIR_REJECTED"
    else:
        assert "precision_motion_agent" not in state["attempts"]


def test_archived_packed_lift_failure_has_exact_recovery():
    evidence = {"bindings_verified": True, "added_sensor_noise": {"rgb_std_8bit": 0., "optical_std": 0.},
                "failure": "RuntimeError: TRANSPORT:(2987.303, 'LASER: lift', 'carried module/other module')"}
    assert repair_evidence(evidence, [])[0] is None
    agent, proposal = repair_evidence(evidence, ["reseat_drifted_parts"])
    assert agent == "precision_motion_agent"
    assert proposal["allowed_actions"] == ["space_packed_carriers"]
    assert repair_evidence(evidence, ["reseat_drifted_parts", "space_packed_carriers"])[0] is None


@pytest.mark.parametrize("after_evidence", [False, True, "near", "corridor", "metrology", "metrology_rejected"])
def test_second_part_gets_its_own_same_table_repair(tmp_path, monkeypatch, observation, after_evidence):
    calls = []
    metrology = after_evidence in {"metrology", "metrology_rejected"}
    near = after_evidence in {"near", "corridor"} or metrology
    corridor = after_evidence == "corridor"
    common = {"bindings_verified":True, "added_sensor_noise":{"rgb_std_8bit":0., "optical_std":0.}}

    def physics(_project, output, *, resume_dir=None, repair_part=None, **kwargs):
        calls.append((output, resume_dir, repair_part))
        executed = list(kwargs.get("recovery_actions", []))
        if repair_part == "LASER" and near:
            assert executed in (["refine_repair_seating"], ["replan_corridor_lift"], ["guard_camera_margin"])
            if metrology:
                assert executed == ["guard_camera_margin"] and kwargs["guard_parts"] == ["LASER"]
        recovery = None
        if repair_part:
            recovery = {"retained_scene":True, "source_report_sha256":checkpoint_files(resume_dir)["result.json"],
                        "target_object_ids":[repair_part], "commanded_object_ids":[repair_part],
                        "untouched_object_ids":sorted(EXPECTED_NAMES-{repair_part})}
            if repair_part == "L25" and after_evidence:
                recovery["attempts"] = [{"name":"L25", "after":{"valid":True},
                                         "position_residual_m":.0007, "rotation_residual_rad":.01}]
        failure = {
            None:"RuntimeError: FINAL_VISUAL_LAYOUT_CHECK_FAILED:L25",
            "L25":"RuntimeError: FINAL_VISUAL_LAYOUT_CHECK_FAILED:LASER",
        }.get(repair_part)
        if corridor and executed == ["refine_repair_seating"]:
            failure = "RuntimeError: PACKED_CARRIER_LIFT_CONTACT:LASER"
        report = write_fixture(output, failure, recovery)
        result = {**common, "success":failure is None, "failure":failure, "source_sha256":observation,
                  "trajectory_sha256":report["trajectory_sha256"], "recovery":recovery,
                  "recovery_actions":executed}
        if not failure:
            result["verification"] = {"accepted":True, "mechanical":{"passed":True}, "optical":{"passed":True}}
        return result

    monkeypatch.setattr(team_flow, "run_legacy", physics)
    if after_evidence:
        from hashlib import sha256
        from pathlib import Path
        real_subprocess = team_flow.subprocess.run

        def observe(command, **kwargs):
            if "simulation.michelson_reobserve" not in command:
                return real_subprocess(command, **kwargs)
            assert Path(command[command.index("--resume-dir")+1]) == calls[1][0]
            output = Path(command[command.index("--output")+1])
            if output.parent.name == "precision_reobserve-02":
                assert near and command[-3:] == ["--object", "LASER", "--near-views"]
            output.mkdir()
            (output/"result.json").write_bytes(b"{}")
            (output/"observation_windows.json").write_bytes(b"[]")

        def verify(project, output, checkpoint, hashes, *, near_object=None):
            assert checkpoint == calls[1][0] and hashes == checkpoint_files(checkpoint)
            close = near_object == "LASER"
            borderline = near and (not close or metrology)
            findings = {"LASER":{"position_m":.00098 if borderline else .002,
                                  "rotation_deg":.1, "passed":borderline}}
            if not close:
                findings["L25"] = {"position_m":.003, "rotation_deg":.2, "passed":False}
            return {**precision_contract.observe_michelson(project, output), "findings":findings,
                    **({"near_object":near_object} if close else {}),
                    "result_sha256":sha256(b"{}").hexdigest(),
                    "image_observations_sha256":sha256(b"[]").hexdigest()}

        monkeypatch.setattr(team_flow.subprocess, "run", observe)
        monkeypatch.setattr(precision_contract, "read_checkpoint_assessment", verify)
        if metrology:
            def mechanical(project, checkpoint, part):
                assert checkpoint == calls[1][0] and part == "LASER"
                if after_evidence == "metrology_rejected":
                    raise ValueError("Independent mechanical diagnostic did not confirm this part outside tolerance")
                return {"position_mm":1.064, "rotation_deg":.25}
            monkeypatch.setattr(precision_contract, "checkpoint_mechanical_outlier", mechanical)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(mode="github_local", profile="auto", provider="replay", team_transport="file_v1",
                                    request="Assemble a 15-component complete Michelson interferometer."))
    state = runtime.execute(run_id)
    if after_evidence == "metrology_rejected":
        assert state["status"] == "failed" and state["error"]["stage"] == "precision_metrology_agent"
        assert [call[2] for call in calls] == [None, "L25"]
        return
    assert state["status"] == "team_precision_passed", state.get("error")
    assert [call[1:] for call in calls] == [(None, None), (calls[0][0], "L25"), (calls[1][0], "LASER")] + (
        [(calls[1][0], "LASER")] if corridor else [])
    assert [item["target_object_ids"] for item in state["precision_repairs"]] == (
        [["L25"], [], [], ["LASER"], ["LASER"]] if corridor else
        [["L25"], [], [], ["LASER"]] if near else
        [["L25"], [], ["LASER"]] if after_evidence else [["L25"], ["LASER"]])
    assert state["attempts"]["precision_sim"] == 1
    assert state["attempts"]["precision_repair"] == (3 if corridor else 2)
    root = runtime.run_dir(run_id)
    if after_evidence:
        assert state["precision_repairs"][1]["completed_object_id"] == "L25"
        if near:
            assert state["precision_repairs"][2]["observation_object_id"] == "LASER"
            assert state["retry_counts"]["precision_vision_agent:LASER:near_review"] == 1
        assert (root/"team/precision_repair-02/observed-layout.json").read_bytes() == (
            root/"team"/f"precision_reobserve-{2 if near else 1:02d}"/"assessment.json").read_bytes()
        if corridor:
            assert state["plan"]["repair_action"] == "replan_corridor_lift"
            assert (root/"team/precision_repair-03/observed-layout.json").read_bytes() == (
                root/"team/precision_repair-02/observed-layout.json").read_bytes()
        if metrology:
            review = json.loads((root/"team/precision_metrology_agent-01/mechanical-review.json").read_bytes())
            assert review["object_id"] == "LASER" and review["residual"]["position_mm"] == 1.064
    execution = root / "team" / "execution-01"
    raw = (root/state["plan_path"]).read_bytes()
    blob = (root/state["artifact_path"]).read_bytes()
    assert (execution/"plan.json").read_bytes() == raw
    assert (execution/"artifact.zip").read_bytes() == blob
    assert json.loads((execution/"service-response.json").read_bytes())["result"] == state["result"]
    assert state["result"]["object_ids"] == ["LASER"] and state["result"]["motions_sent"] == 0
    with pytest.raises(ValueError, match="changed after Output"):
        accept_precision_l1(raw+b" ", blob, state["verification"])


@pytest.mark.parametrize(("fault", "action", "source_bound"), [
    ("PACKED_CARRIER_LIFT_CONTACT", "replan_packed_lift", True),
    ("PACKED_CARRIER_LIFT_CONTACT", "replan_packed_lift", False),
    ("REPAIR_VISUAL_CHECK_FAILED", "refine_repair_seating", True),
])
def test_single_part_fault_replans_from_the_pre_repair_checkpoint(tmp_path, monkeypatch, observation,
                                                                 fault, action, source_bound):
    calls = []
    common = {"bindings_verified":True, "added_sensor_noise":{"rgb_std_8bit":0., "optical_std":0.}}

    def physics(_project, output, *, resume_dir=None, repair_part=None, recovery_actions=(), **kwargs):
        calls.append((output, resume_dir, repair_part, list(recovery_actions)))
        recovery = None
        if not resume_dir:
            failure = "RuntimeError: FINAL_VISUAL_LAYOUT_CHECK_FAILED:LASER"
        else:
            recovery = {"retained_scene":True, "source_report_sha256":checkpoint_files(resume_dir)["result.json"],
                        "target_object_ids":[repair_part], "commanded_object_ids":[repair_part],
                        "untouched_object_ids":sorted(EXPECTED_NAMES-{repair_part})}
            if recovery_actions:
                failure = None
            else:
                failure = f"RuntimeError: {fault}:LASER"
                if fault == "REPAIR_VISUAL_CHECK_FAILED":
                    recovery["attempts"] = [{"name":"LASER", "position_residual_m":.001347,
                                             "rotation_residual_rad":.01}]
                if not source_bound:
                    recovery["source_report_sha256"] = "0" * 64
        report = write_fixture(output, failure, recovery)
        result = {**common, "success":failure is None, "failure":failure, "source_sha256":observation,
                  "trajectory_sha256":report["trajectory_sha256"], "recovery":recovery}
        if not failure:
            result["verification"] = {"accepted":True, "mechanical":{"passed":True}, "optical":{"passed":True}}
        return result

    monkeypatch.setattr(team_flow, "run_legacy", physics)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(mode="github_sim", profile="auto", provider="replay",
                                    request="Assemble a 15-component complete Michelson interferometer."))
    state = runtime.execute(run_id)
    if not source_bound:
        assert state["status"] == "failed"
        assert len(calls) == 2
        return
    assert state["status"] == "team_precision_passed", state.get("error")
    assert [call[1:] for call in calls] == [(None, None, []), (calls[0][0], "LASER", []),
                                           (calls[0][0], "LASER", [action])]
    key = "precision_motion_agent:LASER"
    assert [item["retry_key"] for item in state["retry_log"]] == [key, key+":seating" if action == "refine_repair_seating" else key]
    assert [item["target_object_ids"] for item in state["precision_repairs"]] == [["LASER"], ["LASER"]]


@pytest.mark.parametrize("independent_drift,lift_unproven,turn_candidate",
                         [(True, False, False), (False, False, False), (True, True, False), (True, True, True)])
def test_seating_contact_routes_to_observation_before_any_other_motion(tmp_path, monkeypatch, observation,
                                                                    independent_drift, lift_unproven, turn_candidate):
    calls = []
    source = None
    common = {"bindings_verified":True, "added_sensor_noise":{"rgb_std_8bit":0., "optical_std":0.}}

    def physics(_project, output, *, resume_dir=None, repair_part=None, recovery_actions=(), **kwargs):
        nonlocal source
        calls.append((resume_dir, repair_part, list(recovery_actions)))
        recovery = None
        if not resume_dir:
            source = output
            failure = "RuntimeError: FINAL_VISUAL_LAYOUT_CHECK_FAILED:LASER"
        else:
            recovery = {"retained_scene":True, "source_report_sha256":checkpoint_files(resume_dir)["result.json"],
                        "target_object_ids":[repair_part], "commanded_object_ids":[repair_part],
                        "untouched_object_ids":sorted(EXPECTED_NAMES-{repair_part})}
            if recovery_actions == ["refine_repair_seating"]:
                failure = "RuntimeError: TRANSPORT:(31.1, 'LASER: lower', 'carried module/other module')"
            elif recovery_actions == ["replan_packed_lift"]:
                assert repair_part == "SH0"
                failure = ("RuntimeError: NO_VISUAL_GRASP_AXIS:{'valid': False, 'reason': 'NO_COLLISION_SCREENED_LIFT'}"
                           if lift_unproven else None)
            elif recovery_actions == ["replan_supported_turn_lift"]:
                assert turn_candidate and repair_part == "SH0"
                failure = None
            elif repair_part == "LASER":
                failure = "RuntimeError: REPAIR_VISUAL_CHECK_FAILED:LASER"
                recovery["attempts"] = [{"name":"LASER", "position_residual_m":.0013, "rotation_residual_rad":.01}]
            else:
                assert repair_part == "SH0"
                failure = "RuntimeError: PACKED_CARRIER_LIFT_CONTACT:SH0"
        report = write_fixture(output, failure, recovery)
        result = {**common, "success":failure is None, "failure":failure, "source_sha256":observation,
                  "trajectory_sha256":report["trajectory_sha256"], "recovery":recovery,
                  "recovery_actions":list(recovery_actions)}
        if not failure:
            result["verification"] = {"accepted":True, "mechanical":{"passed":True}, "optical":{"passed":True}}
        return result

    def observe(command, **kwargs):
        from pathlib import Path
        assert Path(command[command.index("--resume-dir")+1]) == source
        if "--close-views" in command:
            assert command[command.index("--object")+1] == "SH0"
            assert [command[i+1] for i,v in enumerate(command) if v == "--pair"] == ["SH0:ISO", "SH0:L25"]
        output = Path(command[command.index("--output")+1])
        output.mkdir()
        (output/"result.json").write_bytes(b"{}")
        (output/"observation_windows.json").write_bytes(b"[]")

    def verify(project, output, checkpoint, hashes, *, close_pairs=()):
        from hashlib import sha256
        assert checkpoint == source and hashes == checkpoint_files(source)
        return {**precision_contract.observe_michelson(project, output),
                "findings":{"LASER":{"position_m":.002, "rotation_deg":1.6, "passed":False},
                            "ISO":{"position_m":.0005, "rotation_deg":.1, "passed":True},
                            "SH0":{"position_m":.0009 if not (close_pairs and not turn_candidate) else .0011,
                                   "rotation_deg":1.7 if independent_drift and (not close_pairs or turn_candidate) else .7,
                                   "passed":not independent_drift}},
                "result_sha256":sha256((output/"result.json").read_bytes()).hexdigest(),
                "image_observations_sha256":sha256((output/"observation_windows.json").read_bytes()).hexdigest(),
                "close_pairs":{f"{a}:{b}":{"wrist_views":5,"motion_authorized":False} for a,b in close_pairs}}

    monkeypatch.setattr(team_flow, "run_legacy", physics)
    monkeypatch.setattr(team_flow.subprocess, "run", observe)
    monkeypatch.setattr(precision_contract, "read_checkpoint_assessment", verify)
    def partial_observation(project, output, **kwargs):
        observed = precision_contract.observe_michelson(project, output)
        return {**observed, "observations":{name:pose for name,pose in observed["observations"].items() if name != "SH0"}}
    monkeypatch.setattr(precision_contract, "observed_checkpoint", partial_observation)
    if lift_unproven:
        manifest = team_flow.legacy_manifest(None)
        x_by_id = {"SH0":0., "ISO":.05, "L25":-.05}
        for i,row in enumerate(manifest["targets"]):
            row["world_position_m"] = [x_by_id.get(row["id"], 1.+i),0.,.8]
        monkeypatch.setattr(team_flow, "legacy_manifest", lambda _: manifest)
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(mode="github_sim", provider="replay", profile="auto",
                                    request="Assemble a 15-component complete Michelson interferometer."))
    state = runtime.execute(run_id)
    assert state["attempts"]["precision_reobserve"] == 1+int(lift_unproven)
    assert calls[:3] == [(None, None, []), (source, "LASER", []), (source, "LASER", ["refine_repair_seating"])]
    assert state["precision_repairs"][2]["target_object_ids"] == []
    if independent_drift:
        if lift_unproven and not turn_candidate:
            assert state["status"] == "failed"
            assert state["error"]["code"] == "PRECISION_LIFT_UNPROVEN"
            assert state["error"]["details"]["authorized_object_ids"] == []
        else:
            assert state["status"] == "team_precision_passed", state.get("error")
        assert calls[3:5] == [(source, "SH0", []), (source, "SH0", ["replan_packed_lift"])]
        assert calls[5:] == ([(source, "SH0", ["replan_supported_turn_lift"])] if turn_candidate else [])
        assert state["precision_repairs"][3]["target_object_ids"] == ["SH0"]
        run = runtime.run_dir(run_id)
        assert (run/"team/precision_repair-04/observed-layout.json").read_bytes() == (
            run/"team/precision_repair-03/observed-layout.json").read_bytes()
        if turn_candidate:
            decision = state["precision_repairs"][-1]
            assert decision["target_object_ids"] == ["SH0"]
            assert decision["agent"] == "precision_motion_agent"
            assert (run/"team/precision_repair-05/observed-layout.json").read_bytes() == (
                run/"team/precision_reobserve-02/assessment.json").read_bytes()
    else:
        assert state["status"] == "failed" and len(calls) == 3
