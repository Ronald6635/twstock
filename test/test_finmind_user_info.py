import os
from app import app as flask_app


def test_user_info_requires_api_key(monkeypatch):
    # If API key is missing, endpoint must return error (no cache fallback)
    monkeypatch.delenv('FINMIND_API_KEY', raising=False)

    client = flask_app.test_client()
    resp = client.get('/api/finmind/user_info')
    assert resp.status_code == 500
    assert resp.get_json() == {'error': 'FinMind API 金鑰未設定'}


def test_user_info_calls_web_api(monkeypatch):
    # With API key set, ensure the route calls web API and returns real-time values
    monkeypatch.setenv('FINMIND_API_KEY', 'fake-key')

    class DummyResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {'user_count': 42, 'api_request_limit': 1000, 'extra': 'value'}

    monkeypatch.setattr('requests.get', lambda *args, **kwargs: DummyResp())

    client = flask_app.test_client()
    resp = client.get('/api/finmind/user_info')

    assert resp.status_code == 200
    assert resp.get_json() == {'user_count': 42, 'api_request_limit': 1000}
