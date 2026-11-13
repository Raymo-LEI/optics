"""Integration tests for the actual graph, using only fixtures and fake services."""
import json
from pathlib import Path

import pytest

from optics_graph.contracts import build_plan, load_example
from optics_graph.runtime import RunInput, Runtime
from optics_graph.settings import DEFAULT_REQUEST


@pytest.fixture(autouse=True)
def no_private_configuration(monkeypatch):
    # No test reads the user's configuration or contacts a model provider.
    import optics_graph.graph as graph
    import optics_graph.gateway as gateway
    fake = lambda: {'model': 'dummy-model', 'api_key': 'SECRET_FOR_ISOLATION_TEST', 'robot_endpoint': 'http://fake-service'}
    monkeypatch.setattr(graph, 'read_settings', fake)
    monkeypatch.setattr(gateway, 'read_settings', fake)


def start(tmp_path, **kwargs):
    runtime = Runtime(tmp_path)
    run_id = runtime.create(RunInput(**kwargs))
    return runtime, run_id, runtime.execute(run_id)


def test_replay_sqlite_and_repeated_run_rejected(tmp_path):
    runtime, run_id, result = start(tmp_path)
    assert result['status'] == 'protocol_passed'
    assert result['result']['evidence_kind'] == 'mock'
    assert result['result']['runtime_visual_pass'] is False
    assert result['result']['physical_execution'] is False
    assert (tmp_path / 'checkpoints.sqlite').is_file()
    with pytest.raises(ValueError):
        Runtime(tmp_path).execute(run_id)


@pytest.mark.parametrize('approved, expected', [(True, 'protocol_passed'), (False, 'rejected')])
def test_restart_review_approval_and_rejection(tmp_path, approved, expected):
    runtime, run_id, result = start(tmp_path, require_review=True)
    assert result['status'] == 'awaiting_review'
    assert result['plan_sha256']
    with pytest.raises(ValueError):
        Runtime(tmp_path).execute(run_id, resume={'approved': approved, 'plan_sha256': 'wrong'})
    resumed = Runtime(tmp_path).execute(run_id, resume={'approved': approved, 'plan_sha256': result['plan_sha256']})
    assert resumed['status'] == expected
    with pytest.raises(ValueError):
        Runtime(tmp_path).execute(run_id, resume={'approved': approved, 'plan_sha256': result['plan_sha256']})


def test_failure_injection_stops_and_no_home(tmp_path):
    _, _, result = start(tmp_path, fail_action='place')
    assert result['status'] == 'failed'
    steps = result['result']['steps']
    assert any(step['primitive'] == 'place' and step['status'] == 'failed' for step in steps)
    assert next(step for step in steps if step['primitive'] == 'home')['status'] == 'skipped'


def test_real_blocked_and_replay_custom_request_not_generated(tmp_path):
    _, _, real = start(tmp_path, mode='real')
    assert real['status'] == 'blocked'
    assert 'result' not in real
    _, _, custom = start(tmp_path, request='Change it to a different experiment')
    assert custom['status'] == 'failed'
    assert 'REPLAY_REQUEST_MISMATCH' in custom['error']['message']
    assert 'layout' not in custom


class InvalidLayoutGateway:
    seen = []
    def __init__(self, provider, run_dir): pass
    def interpret(self, request, scene):
        return {'experiment': 'michelson_geometric', 'requirements': {}, 'unsupported_requirements': [], 'summary': 'fake', 'source': 'fake'}
    def layout(self, request, scene, intent, feedback, attempt):
        self.seen.append((attempt, feedback))
        layout = load_example('layout')
        layout['components'][0]['target_pose']['position_m'][0] = .12
        return layout
    def plan(self, *args):
        raise AssertionError('Invalid layout must never proceed to a plan')


def test_invalid_model_layout_retries_bounded_then_fails(tmp_path):
    InvalidLayoutGateway.seen = []
    runtime = Runtime(tmp_path, gateway_factory=InvalidLayoutGateway)
    run_id = runtime.create(RunInput(provider='openai'))
    result = runtime.execute(run_id)
    assert result['status'] == 'failed'
    assert result['error']['code'] == 'VALIDATION_FAILED'
    assert result['layout_attempts'] == 3
    assert [a for a, _ in InvalidLayoutGateway.seen] == [1, 2, 3]
    assert InvalidLayoutGateway.seen[1][1]
    assert 'result' not in result


class InvalidPlanGateway(InvalidLayoutGateway):
    seen = []
    def layout(self, request, scene, intent, feedback, attempt):
        return load_example('layout')
    def plan(self, layout, validation, feedback, attempt):
        self.seen.append((attempt, feedback))
        plan = build_plan(layout, validation)
        plan['actions'][-1]['primitive'] = 'pick'
        return plan


def test_invalid_model_plan_retries_bounded_then_fails(tmp_path):
    InvalidPlanGateway.seen = []
    runtime = Runtime(tmp_path, gateway_factory=InvalidPlanGateway)
    run_id = runtime.create(RunInput(provider='openai'))
    result = runtime.execute(run_id)
    assert result['status'] == 'failed'
    assert result['plan_attempts'] == 3
    assert [a for a, _ in InvalidPlanGateway.seen] == [1, 2, 3]
    assert InvalidPlanGateway.seen[1][1]
    assert 'result' not in result


def test_remote_incomplete_result_cannot_be_accepted(tmp_path, monkeypatch):
    import optics_graph.graph as graph
    monkeypatch.setattr(graph, 'execute_plan', lambda *a, **k: {'status': 'succeeded', 'physical_execution': False,
        'evidence_kind': 'remote', 'runtime_visual_pass': False, 'steps': [], 'visual_verifications': []})
    _, _, result = start(tmp_path, mode='remote_sim')
    assert result['status'] != 'remote_runtime_passed'


def test_credentials_not_in_state_or_artifacts(tmp_path):
    _, run_id, result = start(tmp_path)
    assert 'SECRET_FOR_ISOLATION_TEST' not in json.dumps(result)
    for path in (tmp_path / 'runs' / run_id).glob('*.json'):
        assert 'SECRET_FOR_ISOLATION_TEST' not in path.read_text()
    assert b'SECRET_FOR_ISOLATION_TEST' not in (tmp_path / 'checkpoints.sqlite').read_bytes()


def test_mock_does_not_read_private_configuration(tmp_path, monkeypatch):
    import optics_graph.graph as graph
    import optics_graph.gateway as gateway
    def forbidden():
        raise AssertionError('mock/replay must not read settings')
    monkeypatch.setattr(graph, 'read_settings', forbidden)
    monkeypatch.setattr(gateway, 'read_settings', forbidden)
    _, _, result = start(tmp_path)
    assert result['status'] == 'protocol_passed'


def test_failed_backend_cannot_report_success(tmp_path, monkeypatch):
    import optics_graph.graph as graph
    monkeypatch.setattr(graph, 'execute_plan', lambda *a, **k: {'status': 'failed', 'physical_execution': False,
        'evidence_kind': 'mock', 'runtime_visual_pass': False, 'steps': [], 'visual_verifications': [],
        'error': {'code': 'SENSOR_UNAVAILABLE'}})
    _, _, result = start(tmp_path)
    assert result['status'] == 'failed'
    assert result['error']['code'] == 'SENSOR_UNAVAILABLE'
