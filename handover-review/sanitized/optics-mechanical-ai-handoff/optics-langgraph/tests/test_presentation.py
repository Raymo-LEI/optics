"""English display cache leaves run evidence untouched."""
import json
from time import monotonic, sleep

from optics_graph.presentation import PresentationService, _numbers, collect_texts


def await_status(service, directory, state, status):
    deadline = monotonic() + 2
    while monotonic() < deadline:
        result = service.get(directory, state)
        if result["status"] == status:
            return result
        sleep(.005)
    raise AssertionError(f"Expected {status}, received {result}")


def test_collect_texts_leaves_english_evidence_alone():
    assert collect_texts({
        "title": "Test", "items": [{"label": "Test", "reason": "Separation 250 mm",
                                  "object_id": "OBJ", "allowed_object_ids": ["EYE"],
                                  "expression": "distance('OBJ','EYE')", "expected": "formula",
                                  "artifact_path": "history/file", "stage": "acceptance"}],
    }) == []
    assert collect_texts("Historical prompt") == []


def test_english_state_does_not_call_the_model(tmp_path):
    calls = []
    state = {"title": "Historical run", "items": [{"label": "Historical run"}], "count": 25}
    original_path = tmp_path / "run.json"
    original = json.dumps(state).encode()
    original_path.write_bytes(original)

    class FakeGateway:
        def __init__(self, provider, directory):
            calls.append(provider)

        def _request(self, role, prompt, schema, attempt):
            raise AssertionError("English evidence must not be sent for translation")

    service = PresentationService(FakeGateway)
    try:
        assert service.get(tmp_path, state) == {"status": "ready", "translations": {}}
        assert service.get(tmp_path, state, "en") == {"status": "ready", "translations": {}}
        assert calls == []
        assert original_path.read_bytes() == original
    finally:
        service.close(wait=True)


def test_number_fingerprint_tracks_numeric_tokens():
    text = "OBJ-250 separation 250 mm, tolerance 1e-6 m, reading 1000"
    assert _numbers(text) == _numbers("gap 250 mm, tolerance 1e-6 m, reading 1000")
    assert _numbers("250") != _numbers("251")
    assert _numbers("1000") != _numbers("1,000")
