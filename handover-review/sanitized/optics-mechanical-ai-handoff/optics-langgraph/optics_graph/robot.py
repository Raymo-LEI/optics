"""Phase 1 Robot Task protocol adapter; no robot controller is implemented here."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .contracts import canonical_hash, validate_plan


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _request(endpoint, method, path, payload=None, timeout=10):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = Request(endpoint.rstrip('/') + path, data=body, method=method,
                      headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _operation(db, request_id, payload, execute):
    """Reserve before sending; an unresolved reservation is never automatically resent."""
    digest = _hash(payload)
    db.execute('BEGIN IMMEDIATE')
    row = db.execute('SELECT payload_hash, result FROM operations WHERE request_id=?', (request_id,)).fetchone()
    if row:
        db.commit()
        if row[0] != digest:
            raise ValueError('REQUEST_ID_PAYLOAD_MISMATCH')
        if row[1] is None:
            raise RuntimeError('OPERATION_OUTCOME_UNKNOWN_NO_RETRY')
        saved = json.loads(row[1])
        if saved.get('outcome_unknown') is True:
            raise RuntimeError('OPERATION_OUTCOME_UNKNOWN_NO_RETRY')
        return saved
    db.execute('INSERT INTO operations(request_id,payload_hash,result) VALUES(?,?,NULL)', (request_id, digest))
    db.commit()
    outcome = execute()
    # Keep the operation ID even for uncertain outcomes; never resend those reservations.
    db.execute('UPDATE operations SET result=? WHERE request_id=?', (json.dumps(outcome), request_id))
    db.commit()
    return outcome


def _remote_operation(endpoint, path, payload, timeout_s, expected_action_id=None):
    deadline = time.monotonic() + timeout_s
    operation_id = None
    op_path = None
    try:
        response = _request(endpoint, 'POST', path, payload, timeout=min(timeout_s, 10))
        operation_id = response.get('operation_id')
        if not isinstance(operation_id, str) or not operation_id:
            return {'status': 'failed', 'outcome_unknown': True, 'error': 'REMOTE_OPERATION_ID_MISSING'}
        op_path = '/v1/operations/' + quote(operation_id, safe='')
        while response.get('status') not in ('succeeded', 'failed', 'cancelled'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(.25, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            response = _request(endpoint, 'GET', op_path, timeout=min(remaining, 5))
        if response.get('status') in ('succeeded', 'failed', 'cancelled') and time.monotonic() <= deadline:
            # Local extension: initialize and observe requests carry their action ID in
            # request_id; every terminal operation must echo run_id and action_id.
            # The team service must explicitly implement this correlation contract.
            if (response.get('run_id') != payload.get('run_id') or
                    response.get('action_id') != expected_action_id or
                    response.get('operation_id', operation_id) != operation_id):
                return {'status': 'failed', 'operation_id': operation_id, 'outcome_unknown': True,
                        'error': 'REMOTE_OPERATION_BINDING_MISMATCH'}
            response['operation_id'] = operation_id
            return response
    except Exception:
        if not operation_id:
            return {'status': 'failed', 'outcome_unknown': True, 'error': 'REMOTE_OPERATION_OUTCOME_UNKNOWN'}
    # Once an operation is known, request a bounded cancellation and independently
    # query its state. Do not continue after the deadline even if it later succeeded.
    try:
        cancel = {'api_version': '1.0', 'run_id': payload.get('run_id'), 'request_id': payload['request_id'] + ':cancel'}
        _request(endpoint, 'POST', op_path + '/cancel', cancel, timeout=2)
        response = _request(endpoint, 'GET', op_path, timeout=2)
    except Exception:
        response = {}
    if (response.get('status') not in ('succeeded', 'failed', 'cancelled') or
            response.get('run_id') != payload.get('run_id') or
            response.get('action_id') != expected_action_id or
            response.get('operation_id', operation_id) != operation_id):
        return {'status': 'failed', 'operation_id': operation_id, 'outcome_unknown': True,
                'error': 'TIMEOUT_OR_DISCONNECT_OUTCOME_UNKNOWN'}
    return {'status': 'failed', 'operation_id': operation_id,
            'error': 'TIMEOUT_OR_DISCONNECT_CANCEL_REQUESTED', 'remote_status': response['status']}


def _register_plan(endpoint, payload):
    # The service is responsible for checking the referenced validation receipt.
    try:
        response = _request(endpoint, 'POST', '/v1/plans', payload, timeout=30)
        receipt = response.get('plan_receipt_id')
        if not isinstance(receipt, str) or not receipt:
            return {'status': 'failed', 'error': 'REMOTE_PLAN_RECEIPT_MISSING'}
        layout_hash, plan_hash = response.get('layout_sha256'), response.get('plan_sha256')
        if layout_hash != canonical_hash(payload['layout']) or plan_hash != canonical_hash(payload['plan']):
            return {'status': 'failed', 'error': 'REMOTE_PLAN_BINDING_MISMATCH'}
        return {'status': 'succeeded', 'plan_receipt_id': receipt,
                'layout_sha256': layout_hash, 'plan_sha256': plan_hash}
    except Exception:
        return {'status': 'failed', 'outcome_unknown': True, 'error': 'REGISTRATION_OUTCOME_UNKNOWN'}


def _same_pose(a, b):
    try:
        if a['frame_id'] != b['frame_id']:
            return False
        pos_a, pos_b = a['position_m'], b['position_m']
        qa, qb = a['orientation_xyzw'], b['orientation_xyzw']
        if len(pos_a) != 3 or len(pos_b) != 3 or len(qa) != 4 or len(qb) != 4:
            return False
        return (all(abs(x - y) <= 1e-9 for x, y in zip(pos_a, pos_b)) and
                (all(abs(x - y) <= 1e-9 for x, y in zip(qa, qb)) or
                 all(abs(x + y) <= 1e-9 for x, y in zip(qa, qb))))
    except (KeyError, TypeError):
        return False

def execute_plan(plan: dict, layout: dict, validation: dict, run_id: str, data_dir: Path,
                 backend: str = 'mock', endpoint: str | None = None, fail_action: str | None = None) -> dict:
    result = {'status': 'failed', 'steps': [], 'visual_verifications': [], 'runtime_visual_pass': False,
              'physical_execution': False, 'evidence_kind': 'mock' if backend == 'mock' else 'remote',
              'held_object_id': None, 'held_object_state_known': True, 'error': None}
    verdict = validate_plan(plan, layout, validation)
    if not verdict.get('passed'):
        result['error'] = {'code': 'PLAN_INVALID', 'issues': verdict.get('issues', [])}
        return result
    if backend == 'mock':
        result.update(layout_sha256=canonical_hash(layout), plan_sha256=canonical_hash(plan))
    if not isinstance(run_id, str) or not run_id or backend not in ('mock', 'remote'):
        result['error'] = {'code': 'INVALID_RUN_OR_BACKEND'}
        return result
    if backend == 'remote':
        parsed = urlsplit(endpoint or '')
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            result['error'] = {'code': 'ENDPOINT_CONFIGURATION_INVALID'}
            return result
        try:
            caps = _request(endpoint, 'GET', '/v1/capabilities', timeout=10)
            if caps.get('environment') != 'simulation' or caps.get('scene_revision') != layout.get('scene_revision'):
                result['error'] = {'code': 'REMOTE_ENVIRONMENT_OR_SCENE_UNSUPPORTED'}
                return result
        except Exception:
            result['error'] = {'code': 'REMOTE_CAPABILITIES_UNAVAILABLE'}
            return result
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(Path(data_dir) / 'robot_operations.sqlite3', timeout=10)
    db.execute('CREATE TABLE IF NOT EXISTS operations(request_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, result TEXT)')
    db.commit()
    targets = {item['object_id']: item for item in layout['components']}
    observed, pending, held, initialized, failure = {}, None, None, False, None
    try:
        plan_receipt_id = None
        for action in plan['actions']:
            aid, primitive, args = action['action_id'], action['primitive'], action['args']
            step = {'action_id': aid, 'primitive': primitive, 'status': 'skipped', 'operation_id': None,
                    'observation_ids': [], 'error': None}
            result['steps'].append(step)
            if failure:
                continue
            obj = args.get('object_id')
            error = None
            if primitive == 'initialize':
                if initialized or held is not None or pending is not None: error = 'INVALID_INITIALIZE_STATE'
            elif not initialized:
                error = 'NOT_INITIALIZED'
            elif primitive == 'observe_scene':
                if held is not None or pending is not None: error = 'OBSERVE_IN_INVALID_STATE'
            elif primitive == 'pick':
                ref = observed.get(args.get('observation_ref'))
                if held is not None or pending is not None or not ref or obj not in ref['object_ids']:
                    error = 'PICK_REQUIRES_OBJECT_OBSERVATION'
            elif primitive == 'place':
                if held != obj or obj not in targets or not _same_pose(args.get('target_pose'), targets[obj]['target_pose']):
                    error = 'PLACE_STATE_OR_TARGET_MISMATCH'
            elif primitive == 'verify_placement':
                if pending != obj or held is not None or obj not in targets or not _same_pose(args.get('target_pose'), targets[obj]['target_pose']):
                    error = 'VERIFY_STATE_OR_TARGET_MISMATCH'
            elif primitive == 'home':
                if held is not None or pending is not None: error = 'HOME_WITH_UNFINISHED_ACTION'
            else:
                error = 'UNSUPPORTED_PRIMITIVE'
            if error:
                step.update(status='failed', error=error); failure = error; continue
            request_id = f'{run_id}:action:{aid}'
            payload = {'request_id': request_id, 'run_id': run_id, 'plan_id': plan['plan_id'],
                       'validation_id': plan['validation_id'], 'input_sha256': _hash({'plan': plan, 'layout': layout, 'validation': validation}),
                       'action': action, 'backend': backend,
                       'endpoint': endpoint if backend == 'remote' else None, 'mock_fail_action': fail_action if backend == 'mock' else None}
            if primitive == 'pick':
                payload['observation_id'] = observed[args['observation_ref']]['observation_id']
            if backend == 'mock':
                def call():
                    if aid == fail_action or primitive == fail_action:
                        return {'status': 'failed', 'error': 'INJECTED_MOCK_FAILURE'}
                    response = {'status': 'succeeded', 'operation_id': 'mock-' + request_id}
                    if primitive == 'observe_scene':
                        response['observation_id'] = 'mock-observation-' + request_id
                    if primitive == 'verify_placement':
                        response['verification'] = {'object_id': obj, 'observation_id': 'mock-verification-' + request_id,
                                                    'passed': True, 'source': 'mock', 'simulated': True}
                    return response
            else:
                path = {'initialize': '/v1/initialize', 'observe_scene': '/v1/observe'}.get(primitive, '/v1/actions')
                wire = {'api_version': '1.0', 'run_id': run_id, 'request_id': request_id}
                if primitive == 'initialize':
                    wire['scene_revision'] = layout['scene_revision']
                elif primitive == 'observe_scene':
                    wire.update(camera_ids=args['camera_ids'], object_ids=args['object_ids'])
                else:
                    command_args = dict(args)
                    if primitive == 'pick':
                        command_args.pop('observation_ref')
                        command_args['observation_id'] = payload['observation_id']
                    wire.update(plan_receipt_id=plan_receipt_id, action_id=aid,
                                timeout_s=action['timeout_s'], command={'primitive': primitive, 'args': command_args})
                def call():
                    return _remote_operation(endpoint, path, wire, float(action['timeout_s']), expected_action_id=aid)
            try:
                response = _operation(db, request_id, payload, call)
            except (ValueError, RuntimeError):
                step.update(status='failed', error='REQUEST_ID_CONFLICT_OR_OUTCOME_UNKNOWN')
                if primitive in ('pick', 'place'): result['held_object_state_known'] = False
                failure = step['error']; continue
            step['operation_id'] = response.get('operation_id')
            if response.get('status') != 'succeeded':
                if response.get('outcome_unknown') and primitive in ('pick', 'place'):
                    result['held_object_state_known'] = False
                step.update(status='failed', error=(response.get('error', 'ACTION_FAILED') if backend == 'mock' or response.get('outcome_unknown') else 'REMOTE_ACTION_FAILED'))
                failure = step['error']; continue
            if primitive == 'initialize':
                initialized = True
                if backend == 'remote':
                    registration_id = run_id + ':register_plan'
                    registration = {'api_version': '1.0', 'run_id': run_id, 'request_id': registration_id,
                                    'layout': layout, 'plan': plan, 'validation_id': plan['validation_id']}
                    internal = {'wire': registration, 'endpoint': endpoint, 'validation': validation}
                    try:
                        reply = _operation(db, registration_id, internal, lambda: _register_plan(endpoint, registration))
                        if reply.get('status') != 'succeeded':
                            step.update(status='failed', error=reply.get('error', 'REMOTE_PLAN_NOT_REGISTERED'))
                            failure = step['error']; continue
                        plan_receipt_id = reply['plan_receipt_id']
                        result.update(layout_sha256=reply['layout_sha256'], plan_sha256=reply['plan_sha256'],
                                      plan_receipt_id=plan_receipt_id)
                    except (ValueError, RuntimeError):
                        step.update(status='failed', error='REGISTRATION_ID_CONFLICT_OR_UNKNOWN')
                        failure = step['error']; continue
            elif primitive == 'observe_scene':
                observation = response.get('output', {}) if backend == 'remote' else {}
                oid = observation.get('observation_id') if backend == 'remote' else response.get('observation_id')
                if not isinstance(oid, str) or not oid:
                    step.update(status='failed', error='MISSING_OBSERVATION'); failure=step['error']; continue
                if backend == 'remote':
                    ids = {x.get('object_id') for x in observation.get('objects', [])}
                    if observation.get('observation_id') != oid or not set(args['object_ids']).issubset(ids):
                        step.update(status='failed', error='OBSERVATION_OBJECT_MISSING'); failure=step['error']; continue
                observed[aid] = {'observation_id': oid, 'object_ids': set(args['object_ids'])}
                step['observation_ids'].append(oid)
            elif primitive == 'pick': held = obj
            elif primitive == 'place': held, pending = None, obj
            elif primitive == 'verify_placement':
                verification = response.get('output', {}) if backend == 'remote' else response.get('verification', {})
                if backend == 'remote':
                    pos, ang = verification.get('position_error_m'), verification.get('angle_error_rad')
                    if (verification.get('object_id') != obj or verification.get('source') != 'vision'
                            or not verification.get('observation_id') or verification.get('passed') is not True
                            or type(pos) not in (int, float) or type(ang) not in (int, float)
                            or not math.isfinite(pos) or not math.isfinite(ang) or pos < 0 or ang < 0
                            or pos > args['tolerance']['position_m'] or ang > args['tolerance']['angle_rad']):
                        step.update(status='failed', error='VISUAL_VERIFICATION_FAILED'); failure=step['error']; continue
                result['visual_verifications'].append(verification)
                step['observation_ids'].append(verification['observation_id'])
                pending = None
            step['status'] = 'succeeded'
        result['held_object_id'] = held
        result['status'] = 'failed' if failure else 'succeeded'
        result['error'] = {'code': failure} if failure else None
        result['runtime_visual_pass'] = bool(not failure and backend == 'remote' and result['visual_verifications'])
        return result
    finally:
        db.close()
