import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from optics_graph.robot import execute_plan, _remote_operation
from optics_graph.contracts import canonical_hash


def fixture():
    pose = {'frame_id': 'optical_table', 'position_m': [0.3, 0.2, 0], 'orientation_xyzw': [0, 0, 0, 1]}
    tol = {'position_m': .01, 'angle_rad': .05}
    actions = [
        {'action_id': 'init', 'primitive': 'initialize', 'args': {}, 'timeout_s': 5},
        {'action_id': 'see', 'primitive': 'observe_scene', 'args': {'camera_ids': ['camera_overhead_a'], 'object_ids': ['m']}, 'timeout_s': 5},
        {'action_id': 'pick', 'primitive': 'pick', 'args': {'object_id': 'm', 'observation_ref': 'see', 'grasp_profile_id': 'test'}, 'timeout_s': 5},
        {'action_id': 'place', 'primitive': 'place', 'args': {'object_id': 'm', 'target_pose': pose, 'tolerance': tol}, 'timeout_s': 5},
        {'action_id': 'verify', 'primitive': 'verify_placement', 'args': {'object_id': 'm', 'target_pose': pose, 'tolerance': tol}, 'timeout_s': 5},
        {'action_id': 'home', 'primitive': 'home', 'args': {}, 'timeout_s': 5}]
    plan = {'plan_id': 'p', 'validation_id': 'v', 'actions': actions}
    layout = {'scene_revision': 's', 'components': [{'object_id': 'm', 'target_pose': pose}]}
    return plan, layout, {}


class RobotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.plan, self.layout, self.validation = fixture()
        self.validator = patch('optics_graph.robot.validate_plan', return_value={'passed': True})
        self.validator.start()

    def tearDown(self):
        self.validator.stop(); self.temp.cleanup()

    def run_plan(self, **kwargs):
        return execute_plan(self.plan, self.layout, self.validation, 'run1', self.dir, **kwargs)

    def test_mock_protocol_success_never_claims_visual_or_hardware(self):
        result = self.run_plan()
        self.assertEqual(result['status'], 'succeeded')
        self.assertFalse(result['physical_execution'])
        self.assertFalse(result['runtime_visual_pass'])
        self.assertEqual(result['visual_verifications'][0]['source'], 'mock')
        self.assertTrue(result['steps'][1]['observation_ids'][0].startswith('mock-observation-'))
        self.assertEqual(result, self.run_plan())

    def test_failure_stops_and_does_not_home_or_drop_held_object(self):
        result = self.run_plan(fail_action='place')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['held_object_id'], 'm')
        self.assertEqual([s['status'] for s in result['steps'][-3:]], ['failed', 'skipped', 'skipped'])

    def test_same_id_different_body_rejected(self):
        self.assertEqual(self.run_plan()['status'], 'succeeded')
        self.plan['actions'][2]['args']['grasp_profile_id'] = 'different'
        result = self.run_plan()
        self.assertEqual(result['steps'][0]['error'], 'REQUEST_ID_CONFLICT_OR_OUTCOME_UNKNOWN')
        self.assertTrue(all(s['status'] == 'skipped' for s in result['steps'][1:]))

    def test_target_mismatch_stops(self):
        self.plan = copy.deepcopy(self.plan)
        self.plan['actions'][3]['args']['target_pose']['position_m'][0] += .1
        result = self.run_plan()
        self.assertEqual(result['steps'][3]['error'], 'PLACE_STATE_OR_TARGET_MISMATCH')

    def test_real_environment_rejected(self):
        with patch('optics_graph.robot._request', return_value={'environment': 'real', 'scene_revision': 's'}) as call:
            result = self.run_plan(backend='remote', endpoint='http://localhost')
        self.assertEqual(result['error']['code'], 'REMOTE_ENVIRONMENT_OR_SCENE_UNSUPPORTED')
        self.assertEqual(call.call_count, 1)

    def test_remote_success_requires_matching_visual_measurement(self):
        calls = []
        def request(endpoint, method, path, payload=None, timeout=10):
            calls.append((method, path, payload))
            if path == '/v1/capabilities': return {'environment': 'simulation', 'scene_revision': 's'}
            if path == '/v1/plans': return {'plan_receipt_id': 'receipt', 'layout_sha256': canonical_hash(self.layout), 'plan_sha256': canonical_hash(self.plan)}
            if path == '/v1/initialize': return {'operation_id': 'op-init', 'status': 'succeeded', 'run_id': 'run1', 'action_id': 'init'}
            if path == '/v1/observe':
                return {'operation_id': 'op-observe', 'status': 'succeeded', 'run_id': 'run1', 'action_id': 'see', 'output': {'observation_id': 'obs', 'objects': [{'object_id': 'm'}]}}
            command = payload['command']
            if command['primitive'] == 'pick':
                self.assertEqual(command['args']['observation_id'], 'obs')
                self.assertNotIn('observation_ref', command['args'])
            result = {'operation_id': 'op-' + command['primitive'], 'status': 'succeeded', 'run_id': 'run1', 'action_id': payload['action_id']}
            if command['primitive'] == 'verify_placement':
                result['output'] = {'object_id': 'm', 'observation_id': 'verify-obs', 'source': 'vision',
                                    'passed': True, 'position_error_m': .001, 'angle_error_rad': .01}
            return result
        with patch('optics_graph.robot._request', side_effect=request):
            result = self.run_plan(backend='remote', endpoint='http://localhost')
        self.assertEqual(result['status'], 'succeeded')
        self.assertTrue(result['runtime_visual_pass'])
        self.assertEqual(result['layout_sha256'], canonical_hash(self.layout))
        self.assertEqual(result['plan_sha256'], canonical_hash(self.plan))
        self.assertFalse(result['physical_execution'])
        self.assertEqual([p for _, p, _ in calls[:4]], ['/v1/capabilities', '/v1/initialize', '/v1/plans', '/v1/observe'])
        init, registration, observe = calls[1][2], calls[2][2], calls[3][2]
        self.assertEqual(set(init), {'api_version', 'run_id', 'request_id', 'scene_revision'})
        self.assertEqual(set(registration), {'api_version', 'run_id', 'request_id', 'layout', 'plan', 'validation_id'})
        self.assertEqual(set(observe), {'api_version', 'run_id', 'request_id', 'camera_ids', 'object_ids'})
        for method, path, body in calls:
            if path == '/v1/actions':
                self.assertEqual(set(body), {'api_version', 'run_id', 'request_id', 'plan_receipt_id', 'action_id', 'timeout_s', 'command'})
                self.assertEqual(body['plan_receipt_id'], 'receipt')
                self.assertNotIn('endpoint', body)

    def test_remote_unknown_never_reposts(self):
        calls = []
        def request(endpoint, method, path, payload=None, timeout=10):
            calls.append((method, path))
            if path == '/v1/capabilities': return {'environment': 'simulation', 'scene_revision': 's'}
            if path == '/v1/initialize': return {'operation_id': 'init', 'status': 'succeeded', 'run_id': 'run1', 'action_id': 'init'}
            if path == '/v1/plans': return {'plan_receipt_id': 'receipt', 'layout_sha256': canonical_hash(self.layout), 'plan_sha256': canonical_hash(self.plan)}
            raise OSError('failure after send')
        with patch('optics_graph.robot._request', side_effect=request):
            first = self.run_plan(backend='remote', endpoint='http://localhost')
            second = self.run_plan(backend='remote', endpoint='http://localhost')
        self.assertEqual(first['status'], 'failed')
        self.assertEqual(second['status'], 'failed')
        self.assertEqual(calls.count(('POST', '/v1/initialize')), 1)
        self.assertEqual(calls.count(('POST', '/v1/plans')), 1)
        self.assertEqual(calls.count(('POST', '/v1/observe')), 1)

    def test_registration_hash_mismatch_stops_before_observe(self):
        calls = []
        def request(endpoint, method, path, payload=None, timeout=10):
            calls.append(path)
            if path == '/v1/capabilities': return {'environment': 'simulation', 'scene_revision': 's'}
            if path == '/v1/initialize': return {'operation_id': 'init', 'status': 'succeeded', 'run_id': 'run1', 'action_id': 'init'}
            if path == '/v1/plans': return {'plan_receipt_id': 'receipt', 'layout_sha256': 'wrong', 'plan_sha256': canonical_hash(self.plan)}
            raise AssertionError('No later actions allowed')
        with patch('optics_graph.robot._request', side_effect=request):
            result = self.run_plan(backend='remote', endpoint='http://localhost')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['code'], 'REMOTE_PLAN_BINDING_MISMATCH')
        self.assertNotIn('layout_sha256', result)
        self.assertNotIn('/v1/observe', calls)

    def test_operation_foreign_run_id_stops(self):
        calls = []
        def request(endpoint, method, path, payload=None, timeout=10):
            calls.append(path)
            if path == '/v1/capabilities': return {'environment': 'simulation', 'scene_revision': 's'}
            if path == '/v1/initialize': return {'operation_id': 'init', 'status': 'succeeded', 'run_id': 'run1', 'action_id': 'init'}
            if path == '/v1/plans': return {'plan_receipt_id': 'receipt', 'layout_sha256': canonical_hash(self.layout), 'plan_sha256': canonical_hash(self.plan)}
            if path == '/v1/observe': return {'operation_id': 'obs', 'status': 'succeeded', 'run_id': 'another', 'action_id': 'see'}
            raise AssertionError('No later actions allowed')
        with patch('optics_graph.robot._request', side_effect=request):
            result = self.run_plan(backend='remote', endpoint='http://localhost')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['code'], 'REMOTE_OPERATION_BINDING_MISMATCH')
        self.assertNotIn('/v1/actions', calls)

    def test_poll_timeout_requests_cancel_then_checks(self):
        calls = []
        def request(endpoint, method, path, payload=None, timeout=10):
            calls.append((method, path))
            if path == '/v1/actions': return {'operation_id': 'o', 'status': 'running'}
            return {'status': 'running'}
        with patch('optics_graph.robot._request', side_effect=request), patch('optics_graph.robot.time.monotonic', side_effect=[0, 2]):
            outcome = _remote_operation('http://localhost', '/v1/actions', {'request_id': 'r'}, 1)
        self.assertTrue(outcome['outcome_unknown'])
        self.assertEqual(calls, [('POST', '/v1/actions'), ('POST', '/v1/operations/o/cancel'), ('GET', '/v1/operations/o')])

if __name__ == '__main__': unittest.main()
