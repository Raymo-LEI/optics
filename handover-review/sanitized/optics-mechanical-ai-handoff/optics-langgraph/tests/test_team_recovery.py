"""Faults are introduced at boundaries; only graph nodes perform recovery."""
import pytest

from optics_graph import team_flow
from optics_graph.gateway import CandidateFormatError, GatewayError, ModelGateway, RetryableGatewayError
from optics_graph.runtime import Runtime, RunInput
from optics_graph.settings import PROJECT
from test_team_flow import prob0


pytestmark = pytest.mark.skipif(not (PROJECT / '.venv-mujoco/bin/python').exists(),
                                reason='MuJoCo environment unavailable')


def request(**kwargs):
    return RunInput(mode='github_local', team_entry='nl', prob0=prob0(), **kwargs)


@pytest.mark.parametrize('stage', ['prompt', 'act'])
@pytest.mark.parametrize('failure', [RetryableGatewayError, CandidateFormatError])
def test_model_failure_returns_to_own_node(tmp_path, stage, failure):
    calls = []

    class Gateway(ModelGateway):
        def interpret(self, *args):
            calls.append('prompt')
            if stage == 'prompt' and calls.count('prompt') == 1:
                raise failure('transient')
            return super().interpret(*args)

        def layout(self, *args):
            calls.append('act')
            if stage == 'act' and calls.count('act') == 1:
                raise failure('transient')
            return super().layout(*args)

    runtime = Runtime(tmp_path, Gateway)
    state = runtime.execute(runtime.create(request()))
    assert state['status'] == 'team_l1_passed'
    assert [(r['from'], r['to']) for r in state['retry_log']] == [(stage, stage)]
    assert state['retry_log'][0]['reason'] == 'retry_allowed'
    assert calls.count(stage) == 2


@pytest.mark.parametrize('failure, count', [(RetryableGatewayError, 3), (GatewayError, 1)])
def test_model_retry_budget_and_permanent_failure_stop(tmp_path, failure, count):
    class Gateway(ModelGateway):
        def layout(self, *args):
            raise failure('unavailable')

    runtime = Runtime(tmp_path, Gateway)
    state = runtime.execute(runtime.create(request()))
    assert state['status'] == 'failed'
    assert state['attempts']['act'] == count
    assert state['retry_log'][-1]['to'] == 'stop'
    assert state['retry_log'][-1]['reason'] == ('target_retry_limit' if count == 3 else 'route_not_allowed')
    assert 'execution' not in state['attempts']
    assert state['verification'] is None and state['result'] is None
    assert len(list((runtime.run_dir(state['run_id'])/'team').glob('feedback-*.json'))) == count


def interrupt_after_completion(monkeypatch, stage):
    atomic = team_flow.atomic
    fired = []

    def crash(path, data):
        atomic(path, data)
        if path.name == 'completion.json' and path.parent.name == f'{stage}-01' and not fired:
            fired.append(True)
            raise RuntimeError('process interrupted after durable result, before checkpoint')

    monkeypatch.setattr(team_flow, 'atomic', crash)
    return fired


@pytest.mark.parametrize('tamper', [False, True])
def test_completed_attempt_survives_checkpoint_gap(tmp_path, monkeypatch, tamper):
    calls = []

    class Gateway(ModelGateway):
        def layout(self, *args):
            calls.append(self.run_dir)
            return super().layout(*args)

    interrupted = interrupt_after_completion(monkeypatch, 'act')
    runtime = Runtime(tmp_path, Gateway)
    rid = runtime.create(request())
    state = runtime.execute(rid)
    assert state['status'] == 'interrupted' and interrupted
    assert len(calls) == 1
    path = runtime.run_dir(rid)/'team/act-01/target_layout.json'
    original = path.read_bytes()
    if tamper:
        path.write_bytes(original + b' ')
    resumed = Runtime(tmp_path, Gateway).execute(rid, continue_run=True)
    assert len(calls) == 1  # No second model call for a completed local attempt.
    if tamper:
        assert resumed['status'] == 'failed'
        assert resumed['error']['code'] == 'ATTEMPT_RECORD_CHANGED'
        assert path.read_bytes() == original + b' '
        assert 'execution' not in resumed['attempts']
    else:
        assert resumed['status'] == 'team_l1_passed'
        assert resumed['attempts']['act'] == 1
        assert len([e for e in resumed['events'] if e['stage'] == 'act']) == 1
        assert calls[0].name == 'act-01'


def test_return_to_controller_invalidates_previous_passes(tmp_path, monkeypatch):
    evaluate = team_flow.evaluate_poses
    calls = []

    def fail_once(*args):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError('measured placement exceeds tolerance')
        return evaluate(*args)

    monkeypatch.setattr(team_flow, 'evaluate_poses', fail_once)
    runtime = Runtime(tmp_path)
    state = runtime.execute(runtime.create(request()))
    assert state['status'] == 'team_l1_passed'
    assert [(r['from'], r['to']) for r in state['retry_log']] == [('output', 'controller')]
    assert state['attempts']['controller'] == state['attempts']['sim'] == state['attempts']['output'] == 2
    assert state['attempts']['execution'] == 1
    assert 'controller-02' in state['plan_path'] and 'sim-02' in state['artifact_path']
    assert state['verification']['plan_hash'] == state['result']['plan_hash'] == state['plan_sha256']


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(frames=[None]),
    lambda p: p.update(observations=[]),
    lambda p: p['observations'].update({next(iter(p['observations'])): None}),
    lambda p: p['observations'][next(iter(p['observations']))].update(position_m=[True, 0, 0]),
])
def test_bad_prob_stops_at_input(tmp_path, mutate):
    initial = prob0()
    mutate(initial)
    runtime = Runtime(tmp_path)
    state = runtime.execute(runtime.create(RunInput(mode='github_local', team_entry='nl', prob0=initial)))
    assert state['status'] == 'failed'
    assert state['attempts'] == {'input': 1}


def test_total_budget_stops_combined_cross_layer_failures(tmp_path, monkeypatch):
    calls = {'prompt': 0, 'act': 0, 'controller': 0, 'sim': 0}

    class Gateway(ModelGateway):
        def interpret(self, *args):
            calls['prompt'] += 1
            if calls['prompt'] <= 2:
                raise CandidateFormatError('bad interpretation JSON')
            return super().interpret(*args)

        def layout(self, *args):
            calls['act'] += 1
            if calls['act'] <= 2:
                raise CandidateFormatError('bad layout JSON')
            return super().layout(*args)

    build = team_flow.build_trajectory
    worker = team_flow.worker

    def controller(*args):
        calls['controller'] += 1
        if calls['controller'] <= 2:
            raise ValueError('trajectory rejected')
        return build(*args)

    def sim(stage, *args, **kwargs):
        calls['sim'] += 1
        if calls['sim'] == 1:
            raise team_flow.subprocess.TimeoutExpired('sim', 40)
        return worker(stage, *args, **kwargs)

    monkeypatch.setattr(team_flow, 'build_trajectory', controller)
    monkeypatch.setattr(team_flow, 'worker', sim)
    runtime = Runtime(tmp_path, Gateway)
    state = runtime.execute(runtime.create(request(team_fault='output')))
    assert state['status'] == 'failed'
    assert state['retry_counts'] == {'prompt': 2, 'act': 2, 'controller': 2, 'sim': 1}
    assert state['retry_log'][-1]['reason'] == 'total_retry_limit'
    assert state['retry_log'][-1]['total_retries'] == 7
    assert state['retry_log'][-1]['from'] == 'output'
    assert 'execution' not in state['attempts']


def test_sim_contact_replans_before_rerun(tmp_path, monkeypatch):
    worker = team_flow.worker
    calls = []

    def sim(stage, *args, **kwargs):
        calls.append(stage)
        if len(calls) == 1:
            return {'status': 'failed', 'error': {'code': 'UNEXPECTED_CONTACT', 'message': 'collision'}}
        return worker(stage, *args, **kwargs)

    monkeypatch.setattr(team_flow, 'worker', sim)
    runtime = Runtime(tmp_path)
    state = runtime.execute(runtime.create(request()))
    assert state['status'] == 'team_l1_passed'
    assert [(r['from'], r['to']) for r in state['retry_log']] == [('sim', 'controller')]
    assert state['attempts']['controller'] == state['attempts']['sim'] == 2


def test_output_optics_returns_to_act_and_revalidates(tmp_path, monkeypatch):
    evaluate = team_flow.evaluate_poses
    calls = []

    def optics(*args):
        calls.append(True)
        if len(calls) == 1:
            return {'passed': False, 'issues': [{'code': 'OPTICAL_PATH_REJECTED'}]}, []
        return evaluate(*args)

    monkeypatch.setattr(team_flow, 'evaluate_poses', optics)
    runtime = Runtime(tmp_path)
    state = runtime.execute(runtime.create(request()))
    assert state['status'] == 'team_l1_passed'
    assert [(r['from'], r['to']) for r in state['retry_log']] == [('output', 'act')]
    assert all(state['attempts'][stage] == 2 for stage in ('act', 'validator', 'controller', 'sim', 'output'))
    assert state['attempts']['execution'] == 1


@pytest.mark.parametrize('bad', [None, [], {'status': 'unknown'}, {'status': 'failed', 'error': None}, {'status': 'succeeded'}])
def test_malformed_sim_response_is_retried_with_budget(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(team_flow, 'worker', lambda *args, **kwargs: bad)
    runtime = Runtime(tmp_path)
    state = runtime.execute(runtime.create(request()))
    assert state['status'] == 'failed'
    assert state['error']['code'] == 'SIM_OUTPUT_INVALID'
    assert state['attempts']['sim'] == 3
    assert state['retry_log'][-1]['reason'] == 'target_retry_limit'
    assert 'execution' not in state['attempts']
