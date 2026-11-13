"""Exercise the real file transport and its rejection boundaries."""
import json
import sys
from zipfile import ZipFile

import pytest

from optics_graph.runtime import RunInput, Runtime, write_json
from optics_graph.settings import PROJECT
from optics_graph.sim_io import read_json
from optics_graph.team_artifacts import digest
from optics_graph.team_services import FileTeamServices
from test_team_flow import prob0


pytestmark = pytest.mark.skipif(not (PROJECT / '.venv-mujoco/bin/python').exists(),
                                reason='MuJoCo environment unavailable')


def run_input(**kwargs):
    return RunInput(mode='github_local', team_transport='file_v1', team_entry='nl', prob0=prob0(), **kwargs)


def test_file_services_share_exact_controller_bytes_and_recover(tmp_path):
    runtime = Runtime(tmp_path)
    rid = runtime.create(run_input(team_fault='matrix'))
    state = runtime.execute(rid)
    assert state['status'] == 'team_l1_passed'
    assert [(row['from'], row['to']) for row in state['retry_log']] == [
        ('validator', 'act'), ('validator', 'prompt'), ('controller', 'controller'),
        ('sim', 'sim'), ('output', 'sim'),
    ]
    root = runtime.run_dir(rid)
    plan = root / state['plan_path']
    archive = root / state['artifact_path']
    execution = root / 'team/execution-01'
    raw = plan.read_bytes()
    assert raw == plan.with_name('candidate-plan.json').read_bytes()
    assert raw == archive.with_name('plan.json').read_bytes() == (execution / 'plan.json').read_bytes()
    with ZipFile(archive) as evidence:
        assert evidence.read('plan.json') == raw
    job = read_json(execution / 'service-request.json')
    response = read_json(execution / 'service-response.json')
    assert response['request_sha256'] == digest((execution / 'service-request.json').read_bytes())
    assert job['files']['plan.json'] == response['result']['plan_hash'] == digest(raw)
    assert job['files']['artifact.zip'] == response['result']['artifact_sha256'] == digest(archive.read_bytes())
    assert response['result']['motions_sent'] == 0


@pytest.mark.parametrize('field,value', [('unchanged', None), ('job_id', 'another-run:execution-01'), ('motions_sent', 1)])
def test_replaced_executor_must_match_identity_and_receipt(tmp_path, field, value):
    # A real independent replacement process uses the same contract as the reference worker.
    alternate = tmp_path / 'executor.py'
    alternate.write_text('''
import hashlib, json, sys
from pathlib import Path
path = Path(sys.argv[-1])
raw = (path / 'service-request.json').read_bytes()
job = json.loads(raw)
plan = json.loads((path / 'plan.json').read_bytes())
result = {'mode': 'local_l1_dry_run', 'plan_hash': job['files']['plan.json'],
          'artifact_sha256': job['files']['artifact.zip'], 'robot_model_id': plan['robot_model_id'],
          'calibration_id': plan['calibration_id'], 'motions_sent': 0}
response = {key: job[key] for key in ('version', 'job_id', 'stage')}
response.update(result=result, request_sha256=hashlib.sha256(raw).hexdigest())
field, value = sys.argv[1], json.loads(sys.argv[2])
if field == 'job_id':
    response[field] = value
elif field == 'motions_sent':
    result[field] = value
(path / 'service-response.json').write_text(json.dumps(response))
''')
    services = FileTeamServices({'execution': (sys.executable, str(alternate), field, json.dumps(value))})
    runtime = Runtime(tmp_path / 'data', file_services=services)
    state = runtime.execute(runtime.create(run_input()))
    assert state['attempts']['execution'] == 1
    if field == 'unchanged':
        assert state['status'] == 'team_l1_passed'
        assert state['result']['motions_sent'] == 0
    else:
        assert state['status'] == 'failed'
        assert state['error']['code'] == 'EXECUTION_REJECTED'
        expected = 'does not match this request' if field == 'job_id' else 'does not match the verified handoff'
        assert expected in state['error']['message']
        assert state['result'] is None


@pytest.mark.parametrize('review', [False, True])
def test_restore_public_status_from_terminal_or_review_checkpoint(tmp_path, review):
    runtime = Runtime(tmp_path)
    rid = runtime.create(run_input(require_review=review))
    state = runtime.execute(rid)
    assert state['status'] == ('awaiting_review' if review else 'team_l1_passed')
    # Simulate a crash after SQLite committed but before the public status was saved.
    root = runtime.run_dir(rid)
    write_json(root / 'run.json', {**state, 'status': 'running', 'result': None})
    restored = Runtime(tmp_path).execute(rid, continue_run=True)
    assert restored == state
    assert read_json(root / 'run.json') == restored
    if review:
        assert 'execution' not in restored['attempts']
