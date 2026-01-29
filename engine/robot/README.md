# Automated Stock Analysis Robot

This directory contains the automated stock analysis robot that integrates data fetching, preprocessing, and technical analysis for Taiwan stocks.

## Directory Structure
- `automated_stock_robot.py`: Main robot script
- `config.json`: Configuration file (JSON with optional JS-style comments supported)
- `data/`: Processed data files
- `reports/`: Individual stock analysis reports
- `logs/`: Execution logs (robot-level)
- `summary/`: Summary reports
- `temp/`: Per-stock temporary files and child-run logs

## Key behaviour changes (quota, logging, streaming)
- The robot queries FinMind's `/v2/user_info` to obtain quota information and logs it so you can see API usage in the terminal and log files.
- `config.json` may contain `//` or `/* */` comments — the loader strips them before parsing.

### What you will see in logs
- Robot (console + `logs/robot_*.log`):
  - `Fetching API usage info...`
  - `API Usage: user_count=550, api_request_limit=600, api_requests_remaining=523`
  - Warnings when quota is low or when a fetch is skipped
- Per-stock temp log: `engine/robot/temp/batch_fetch_<STOCKID>.log`
  - `--- PARENT ---` — robot validation / usage messages (always written when relevant)
  - `--- STDOUT ---` — child process stdout
  - `--- STDERR ---` — child process stderr

Note: when the robot skips a fetch due to quota, a parent-only `batch_fetch_<STOCKID>.log` is written explaining the reason.

### Quota enforcement (precise)
- Config knobs:
  - `api_usage_threshold` (int, default 50)
  - `api_usage_threshold_unit` (`"absolute"` | `"percent"`, default `"absolute"`)
  - `api_retry_max` (int, default 3)
- Semantics:
  - absolute: treat `api_usage_threshold` as remaining-request cutoff (skip when remaining ≤ value).
  - percent: convert configured percent to absolute using provider-reported limit (cutoff = ceil(limit * pct/100)). If the provider does not report a total limit, percent-mode falls back to treating the configured value as absolute and logs a warning.
- Computed fallback:
  - If FinMind does not provide `api_requests_remaining` but returns `user_count` and `api_request_limit`, the robot computes:
    remaining = api_request_limit - user_count
  - Negative remaining is treated as 0 (interpreted as over-quota).
  - If remaining is available and remaining ≤ cutoff, the robot:
    - Logs a WARNING,
    - Writes a `--- PARENT ---` temp log with the computed remaining and skip reason,
    - Skips running `scripts/batch_fetch.py` for that stock (returns False).
  - If `remaining` is None (no usable quota info), the robot allows the fetch by default.

### Retry / backoff
- HTTP 429 responses from FinMind trigger retries with exponential backoff (honours `Retry-After` when provided). Controlled by `api_retry_max`.

### Throttle UI (progress bar)
- When the robot applies an adaptive throttle before a fetch, a TTY-only progress bar is shown in the terminal so operators can see how many seconds remain before the next fetch.
- Example (terminal):

  Throttle 2330: [======>          ]  4.8s remaining

- Controlled by `api_throttle_show_progress` (default: `true`). If stdout is not a TTY (CI or redirected output) the robot falls back to a plain sleep (no progress shown). The implementation preserves a single `time.sleep()` call so unit tests that monkeypatch `time.sleep` continue to work.
- When throttling is applied the robot exports `ROBOT_API_THROTTLE_SECONDS` for downstream visibility.

### Environment variables exported to child process
- When validated, the robot exports to the child process environment:
  - `FINMIND_API_KEY`
  - `FINMIND_API_REQUESTS_REMAINING` (when available / computed)
  - `ROBOT_API_USAGE_CUTOFF` (computed cutoff used by the robot)

### Child-output streaming (`stream_child_output`)
- Config key: `stream_child_output` (bool)
  - false (default): robot captures child stdout/stderr and writes them to `engine/robot/temp/batch_fetch_<STOCKID>.log` after the child exits.
  - true: robot streams child stdout/stderr to the terminal in real time *and* saves the same output to the temp log.
  - Note: streaming output may interleave with robot logs in the terminal.

## Quota examples (provider limit = 600)
- `{"api_usage_threshold": 50, "api_usage_threshold_unit": "absolute"}`  
  → skip when remaining ≤ 50 (used ≥ 550)
- `{"api_usage_threshold": 10, "api_usage_threshold_unit": "percent"}`  
  → cutoff = ceil(600 * 0.10) = 60 → skip when remaining ≤ 60
- If FinMind returns `user_count=602, api_request_limit=600` and no explicit `api_requests_remaining`:
  - robot computes remaining = -2 → treated as 0 → 0 ≤ cutoff → fetch skipped and `--- PARENT ---` log written

## Quick checks / smoke tests
- Run the robot and watch quota logging:
```bash
python automated_stock_robot.py
# look for "Fetching API usage info..." and "API Usage: ..." in terminal
# inspect engine/robot/temp/batch_fetch_<STOCKID>.log for the --- PARENT --- section
```
- Test single-stock run:
  - Edit `engine/robot/config.json` and set `"stocks": ["3518"]`
  - Toggle streaming with `"stream_child_output": true` to see child output immediately
- Simulate low-remaining behavior (example):
  - Temporarily set FINMIND_API_REQUESTS_REMAINING env var (for local testing) and run the robot:
    - PowerShell: $env:FINMIND_API_REQUESTS_REMAINING = "40"; python automated_stock_robot.py

## Troubleshooting
- If a stock shows many `FAILED - 'data'` entries but the parent log shows `user_count > api_request_limit`, the robot now skips fetches in that situation — check `--- PARENT ---` in the temp log to confirm the skip reason.
- Increase verbosity by inspecting `logs/robot_*.log`.
- For CI/tests: run `pytest -q` from the repository root (tests add coverage for quota logic and streaming).

## Configuration (summary)
- `api_usage_threshold`: int (default 50)
- `api_usage_threshold_unit`: "absolute" | "percent" (default "absolute")
- `api_retry_max`: int (default 3)
- `stream_child_output`: bool (default false)
- `api_throttle_enabled`: bool (default false) — enable adaptive throttling between fetches
- `api_throttle_min_seconds`: number (default 1.0) — minimum throttle sleep when quota is healthy
- `api_throttle_max_seconds`: number (default 10.0) — maximum throttle sleep when remaining is near cutoff
- `api_throttle_mode`: string (default "linear") — mapping mode (currently `linear`)
- `api_throttle_show_progress`: bool (default true) — show a terminal progress bar while sleeping (TTY only)

## Contributing / tests
- Add regression tests for quota behaviour (examples included in repository tests).
- Create a branch, add tests for new behavior, and open a pull request against `master`.