import json
from types import SimpleNamespace
import httpx
import pytest
from fastapi.testclient import TestClient
from optics_graph.gateway import ModelGateway, GatewayError, strict_loads, output_schema
from optics_graph.contracts import load_scene, load_example
from optics_graph.settings import DEFAULT_REQUEST
from optics_graph import gateway, settings, server
from optics_graph.runtime import Runtime, RunInput


def test_strict_model_json():
    for text in ['{"x":1,"x":2}','{"x":NaN}','[1,2]','```json\n{}\n```']:
        with pytest.raises(ValueError):strict_loads(text)


def test_large_model_json_preserves_strict_validation():
    value = "a" * 270_000
    assert strict_loads(json.dumps({"content": value})) == {"content": value}
    with pytest.raises(ValueError):
        strict_loads('{"content":' + json.dumps(value) + ',"content":null}')


def test_default_intent_does_not_invent_dimensions(tmp_path):
    g=ModelGateway('openai',tmp_path)
    g._request=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('No extraction needed for exact built-in request'))
    assert g.interpret(DEFAULT_REQUEST,load_scene())['requirements']=={}


def test_custom_intent_does_not_receive_scene_dimensions(tmp_path):
    g=ModelGateway('openai',tmp_path)
    captured=[]
    def request(role,prompt,schema):
        captured.append(prompt)
        return {'experiment':'michelson_geometric','object_targets':[], 'arm_x_m':None,'arm_y_m':None,
                'max_position_error_m':None,'max_angle_error_rad':None,'unsupported_requirements':[], 'summary':'Michelson'}
    g._request=request
    g.interpret('Please assemble a Michelson layout',load_scene())
    assert 'footprint' not in captured[0].split('CATALOG:')[-1]
    assert '0.04' not in captured[0]


def test_gateway_raw_output_saved_without_key(tmp_path,monkeypatch):
    monkeypatch.setattr(gateway,'read_settings',lambda:{'model':'unit-test','api_key':'sk-test-secret'})
    captured={}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status='completed',output_text=json.dumps(load_example('layout')),model='unit-test',id='response-test',usage=None)
    client=SimpleNamespace(responses=SimpleNamespace(create=create))
    g=ModelGateway('openai',tmp_path,client)
    assert g._request('layout','example data',output_schema('layout'))['document_type']=='optical_layout'
    assert captured['store'] is False
    assert 'max_output_tokens' not in captured
    assert captured['text']['format']['strict'] is True
    assert 'sk-test-secret' not in (tmp_path/'model-layout-1.json').read_text()


def test_key_private_and_public_view_redacted(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,'CONFIG',tmp_path/'local.json')
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    assert settings.read_settings()['model'] == 'gpt-6.1-sol'
    settings.save_settings('model-test','sk-test-not-real')
    assert (tmp_path/'local.json').stat().st_mode & 0o777==0o600
    assert settings.public_settings()['key_configured'] is True
    assert 'api_key' not in settings.public_settings()
    settings.save_settings('model-test-two','')
    assert settings.read_settings()['api_key']=='sk-test-not-real'


def test_incomplete_model_response_is_preserved_without_fallback(tmp_path,monkeypatch):
    from optics_graph import design_agent
    monkeypatch.setattr(gateway,'read_settings',lambda:{'model':'gpt-6-luna','api_key':'sk-test-secret'})
    response=SimpleNamespace(status='incomplete',output_text='{"title":"partial',model='gpt-6-luna',
                             id='response-partial',usage=None,incomplete_details=None)
    client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:response))
    g=ModelGateway('openai',tmp_path,client)
    with pytest.raises(GatewayError,match='MODEL_INCOMPLETE'):
        design_agent.interpret(g,'Design a telescope',{},[],1)
    raw=(tmp_path/'model-response.json').read_text()
    assert json.loads(raw)['status']=='incomplete'
    assert json.loads(raw)['output_text']==response.output_text
    assert 'sk-test-secret' not in raw


@pytest.mark.parametrize(
    ("status", "reason", "output_text", "refusal", "error", "code"),
    [
        ("incomplete", "max_output_tokens", '{"title":"partial', None,
         gateway.RetryableGatewayError, "MODEL_OUTPUT_TRUNCATED"),
        ("incomplete", "content_filter", "", None, GatewayError, "MODEL_INCOMPLETE"),
        ("completed", None, "", "Provider refused", GatewayError, "MODEL_REFUSED"),
        ("completed", None, '{"title":"partial', None, gateway.CandidateFormatError, "MODEL_INVALID_JSON"),
    ],
)
def test_gateway_failure_classification_and_log(
    tmp_path, monkeypatch, status, reason, output_text, refusal, error, code,
):
    monkeypatch.setattr(gateway, "read_settings", lambda: {"model": "gpt-6.1-sol", "api_key": "sk-test-secret"})
    usage = {"input_tokens": 5, "output_tokens": 12, "total_tokens": 17}
    response = SimpleNamespace(
        status=status, output_text=output_text, model="gpt-6.1-sol", id="response-test",
        usage=SimpleNamespace(model_dump=lambda: usage),
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[SimpleNamespace(
            type="message", content=[SimpleNamespace(type="refusal", refusal=refusal)],
        )] if refusal else [],
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    g = ModelGateway("openai", tmp_path, client)
    with pytest.raises(error, match=code) as caught:
        g._request("layout", "example data", output_schema("layout"), attempt=2)
    assert type(caught.value) is error
    assert caught.value.code == code
    if code == "MODEL_OUTPUT_TRUNCATED":
        assert caught.value.details == {"reason": reason, "usage": usage, "response_id": response.id}
    raw = (tmp_path / "model-layout-2.json").read_text()
    assert "sk-test-secret" not in raw
    log = json.loads(raw)
    assert log["status"] == status
    assert log["output_text"] == output_text
    assert log["refusals"] == ([refusal] if refusal else [])
    assert log["incomplete_details"] == ({"reason": reason} if reason else None)


def test_local_api_guard_and_run_status(tmp_path,monkeypatch):
    monkeypatch.setattr(server,'public_settings',lambda:{'model':'test','key_configured':False})
    runtime=Runtime(tmp_path)
    app=server.create_app(runtime)
    with TestClient(app) as c:
        assert c.get('/healthz').json()['real_motion_enabled'] is False
        assert 'API key' in c.get('/').text
        assert c.post('/api/runs',json={}).status_code==403
        assert c.post('/api/runs',json={},headers={'X-Optics-Local':'1','Origin':'https://evil.example'}).status_code==403
        settings=c.get('/api/settings').json()
        assert set(settings.pop('cli_deciders'))=={'claude','codex'}  # which local CLI deciders were found
        assert settings=={'model':'test','key_configured':False}
        assert c.get('/api/runs/../../config/local.json').status_code==404
        rid=runtime.create(RunInput())
        state=runtime.execute(rid)
        assert state['status']=='protocol_passed'
        assert c.get('/api/runs/'+rid).json()['status']=='protocol_passed'
        assert c.post('/api/runs/'+rid+'/review',json={'approved':True,'plan_sha256':state['plan_sha256']},headers={'X-Optics-Local':'1'}).status_code==409


def test_remote_hash_not_overwritten(tmp_path,monkeypatch):
    from optics_graph import graph
    monkeypatch.setattr(graph,'read_settings',lambda:{'robot_endpoint':'http://local.invalid'})
    monkeypatch.setattr(graph,'execute_plan',lambda *a,**kw:{'status':'succeeded','plan_sha256':'wrong','layout_sha256':'wrong'})
    r=Runtime(tmp_path);rid=r.create(RunInput(mode='remote_sim'))
    state=r.execute(rid)
    assert state['status']=='failed'
    assert 'ARTIFACT_HASH_MISMATCH' in state['error']['message']


def test_background_gateway_polls_and_resumes_only_matching_job(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "read_settings", lambda: {"model": "gpt-6.1-sol", "api_key": "sk-test-secret"})
    sleeps = []
    monkeypatch.setattr(gateway.time, "sleep", sleeps.append)
    requests = []
    retrieved = []
    response_id = "response-background"
    complete = SimpleNamespace(
        id=response_id, status="completed", output_text='{"ok":true}', model="gpt-6.1-sol", usage=None,
    )

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(id=response_id, status="queued")

    def retrieve(job_id):
        retrieved.append(job_id)
        return SimpleNamespace(id=job_id, status="in_progress") if len(retrieved) == 1 else complete

    client = SimpleNamespace(responses=SimpleNamespace(create=create, retrieve=retrieve))
    schema = output_schema("layout")
    g = ModelGateway("openai", tmp_path, client)
    assert g._request("layout", "example data", schema) == {"ok": True}
    assert retrieved == [response_id, response_id]
    assert len(sleeps) == 2
    assert len(requests) == 1
    assert requests[0]["background"] is True
    assert requests[0]["store"] is False
    assert requests[0]["text"]["format"]["strict"] is True
    assert "max_output_tokens" not in requests[0]
    job = json.loads((tmp_path / "model-job-layout-1.json").read_text())
    assert job["response_id"] == response_id
    assert len(job["request_sha256"]) == 64

    reopened = ModelGateway("openai", tmp_path, client)
    assert reopened._request("layout", "example data", schema) == {"ok": True}
    assert retrieved == [response_id] * 3
    with pytest.raises(GatewayError, match="MODEL_JOB_CHANGED") as caught:
        reopened._request("layout", "different input", schema)
    assert caught.value.code == "MODEL_JOB_CHANGED"
    assert type(caught.value) is GatewayError
    assert retrieved == [response_id] * 3
    assert len(requests) == 1


def test_background_gateway_retrieve_connection_failure_preserves_job(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "read_settings", lambda: {"model": "gpt-6.1-sol", "api_key": "sk-test-secret"})
    monkeypatch.setattr(gateway.time, "sleep", lambda seconds: None)
    calls = []
    response_id = "response-background-interrupted"

    def create(**kwargs):
        calls.append("create")
        return SimpleNamespace(id=response_id, status="in_progress", output_text='{"partial":')

    def retrieve(job_id):
        calls.append(job_id)
        raise httpx.ReadError("background retrieval connection lost")

    def sdk(**kwargs):
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(responses=SimpleNamespace(create=create, retrieve=retrieve))

    monkeypatch.setattr(gateway, "OpenAI", sdk)
    g = ModelGateway("openai", tmp_path)
    with pytest.raises(gateway.RetryableGatewayError, match="MODEL_CONNECTION_FAILED") as caught:
        g._request("layout", "example data", output_schema("layout"), attempt=2)
    assert caught.value.code == "MODEL_CONNECTION_FAILED"
    assert caught.value.details == {"response_id": response_id}
    assert calls == ["create", response_id]
    job = json.loads((tmp_path / "model-job-layout-2.json").read_text())
    assert job["response_id"] == response_id
    raw = (tmp_path / "model-layout-2.json").read_text()
    assert "sk-test-secret" not in raw
    log = json.loads(raw)
    assert log["status"] == "transport_failed"
    assert log["response_id"] == response_id
    assert log["output_text"] == '{"partial":'


@pytest.mark.parametrize(("provider_code", "headers", "seconds", "retryable"), [
    ("server_is_overloaded", {"retry-after": "7", "retry-after-ms": "7000"}, 7, True),
    ("server_error", {"retry-after-ms": "-1", "retry-after": "1.5"}, 1.5, True),
    ("rate_limit_exceeded", {"retry-after": "NaN", "retry-after-ms": "Inf"}, None, True),
    ("invalid_prompt", {"retry-after": "7"}, None, False),
])
def test_background_gateway_classifies_failed_response(tmp_path, monkeypatch, provider_code, headers, seconds, retryable):
    monkeypatch.setattr(gateway, "read_settings", lambda: {"model": "gpt-6.1-sol", "api_key": "sk-test-secret"})
    error = {"code": provider_code, "headers": headers, "message": "provider diagnostic"}
    response = SimpleNamespace(
        id="response-background-failed", status="failed", output_text="", model="gpt-6.1-sol", usage=None,
        error=SimpleNamespace(model_dump=lambda **kwargs: error),
        model_dump=lambda **kwargs: {"status": "failed", "error": error},
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    g = ModelGateway("openai", tmp_path, client)
    expected_type = gateway.RetryableGatewayError if retryable else GatewayError
    with pytest.raises(expected_type, match=provider_code) as caught:
        g._request("layout", "example data", output_schema("layout"))
    assert type(caught.value) is expected_type
    details = {"provider_code": provider_code, "response_id": response.id}
    if retryable:
        assert caught.value.code == "MODEL_SERVICE_UNAVAILABLE"
        details["retry_after_s"] = seconds
    else:
        assert caught.value.code == "MODEL_FAILED"
    assert caught.value.details == details
    log = json.loads((tmp_path / "model-layout-1.json").read_text())
    assert log["status"] == "failed"
    assert log["provider_error"] == log["response"]["error"] == error
