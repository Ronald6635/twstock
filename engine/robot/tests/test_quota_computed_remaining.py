import os
from pathlib import Path

import pytest

from engine.robot import automated_stock_robot as arb


def test_computed_remaining_skips_fetch_and_writes_parent_log(tmp_path, monkeypatch):
    # prepare minimal repo layout
    repo = tmp_path
    (repo / 'scripts').mkdir(parents=True)
    (repo / 'temp').mkdir(parents=True)
    monkeypatch.chdir(repo)

    # ensure subprocess.run is NOT called (fetch should be skipped before subprocess)
    def _fail_run(*a, **k):
        raise AssertionError("subprocess.run should not be called when quota is exceeded")
    monkeypatch.setattr(arb, 'fetch_finmind_usage', lambda token: {'ok': True, 'status_code': 200, 'data': {'user_count': 602, 'api_request_limit': 600}})
    monkeypatch.setattr('subprocess.run', _fail_run)

    cfg = {
        'api_key': 'FINMIND_API_KEY',
        'start_date': '2020-01-01',
        'end_date': '2020-01-02',
        'stocks': ['2392'],
        'output_dirs': {'data': 'data', 'reports': 'reports', 'logs': 'logs', 'summary': 'summary', 'temp': 'temp'},
        'api_usage_threshold': 50,
        'api_usage_threshold_unit': 'absolute',
        'api_retry_max': 1,
        'stream_child_output': False,
    }

    temp_dir = repo / 'temp'
    # run
    ok = arb.run_batch_fetch('2392', cfg, temp_dir)
    assert ok is False

    log = temp_dir / 'batch_fetch_2392.log'
    assert log.exists()
    txt = log.read_text(encoding='utf-8')
    assert '--- PARENT ---' in txt
    assert 'Computed API remaining' in txt or 'Skipping fetch' in txt
