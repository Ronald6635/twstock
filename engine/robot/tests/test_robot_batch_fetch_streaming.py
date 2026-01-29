import os
import stat
import subprocess
from pathlib import Path

import pytest

from engine.robot import automated_stock_robot as arb


SCRIPT_PATH = Path.cwd() / 'scripts' / 'batch_fetch.py'


def _write_dummy_batch_fetch(tmp_path: Path, stdout_lines: list[str]):
    scripts_dir = tmp_path / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'batch_fetch.py'
    script.write_text('#!/usr/bin/env python\nimport sys, time\n' + '\n'.join([f"print('{l}')" for l in stdout_lines]) + '\n')
    # make executable on POSIX (harmless on Windows)
    try:
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    except Exception:
        pass
    return script


@pytest.mark.parametrize('stream', [True, False])
def test_run_batch_fetch_streaming_and_parent_section(tmp_path, monkeypatch, stream):
    """Ensure run_batch_fetch writes a --- PARENT --- section and includes API usage lines
    in both streaming and capture modes."""
    # prepare dummy repo layout
    repo_root = tmp_path
    (repo_root / 'engine' / 'robot').mkdir(parents=True, exist_ok=True)
    (repo_root / 'scripts').mkdir(parents=True, exist_ok=True)

    # dummy batch_fetch that prints usage lines
    stdout_lines = [
        'FinMind API initialized',
        'Fetching API usage info...',
        'API Usage: user_count=123, api_request_limit=600'
    ]
    script = _write_dummy_batch_fetch(repo_root, stdout_lines)

    # point cwd to tmp repo and monkeypatch Path(__file__).parent.parent.parent resolution
    monkeypatch.chdir(repo_root)

    # minimal config
    cfg = {
        'api_key': 'FINMIND_API_KEY',
        'start_date': '2020-01-01',
        'end_date': '2020-01-02',
        'stocks': ['0000'],
        'output_dirs': {
            'data': 'data',
            'reports': 'reports',
            'logs': 'logs',
            'summary': 'summary',
            'temp': 'temp'
        },
        'api_usage_threshold': 0,  # allow fetch
        'api_retry_max': 1,
        'stream_child_output': stream
    }

    temp_dir = repo_root / 'temp'
    temp_dir.mkdir()

    # monkeypatch fetch_finmind_usage so robot believes token is valid and reports remaining
    monkeypatch.setattr(arb, 'fetch_finmind_usage', lambda token: {'ok': True, 'status_code': 200, 'data': {'user_count': 1, 'api_request_limit': 600, 'api_requests_remaining': 500}})

    # Run
    ok = arb.run_batch_fetch('0000', cfg, temp_dir)
    assert ok is True

    # Check temp log
    fetch_log = temp_dir / 'batch_fetch_0000.log'
    assert fetch_log.exists(), fetch_log
    text = fetch_log.read_text(encoding='utf-8')
    assert '--- PARENT ---' in text
    assert 'API Usage:' in text
    assert '--- STDOUT ---' in text
    # ensure dummy stdout captured
    assert 'FinMind API initialized' in text
