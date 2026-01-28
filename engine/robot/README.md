# Automated Stock Analysis Robot

This directory contains the automated stock analysis robot that integrates data fetching, preprocessing, and technical analysis for Taiwan stocks.

## Directory Structure
- `automated_stock_robot.py`: Main robot script
- `config.json`: Configuration file
- `data/`: Processed data files
- `reports/`: Individual stock analysis reports
- `logs/`: Execution logs
- `summary/`: Summary reports
- `temp/`: Temporary files

## Prerequisites
- Python 3.10+ (virtual environment recommended)
- FINMIND_API_KEY set in environment variables
- Install dependencies:
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Setup / Run
1. Edit `config.json` to set desired stocks, dates, and parameters.
2. Run the robot:
```bash
python automated_stock_robot.py
```

## Data layout & expected files
- The robot looks for preprocessed files in `engine/datasets/{company}-{stock_id}/` with filenames like:
  - `preprocessed_{company}-{stock_id}.json`
  - `combined_*{stock_id}*.json` (used to generate preprocessed output)
- As a fallback, it may read cache entries in `cache/{company}-{stock_id}/` such as:
  - `2020-01-01_2026-01-28_finmind_taiwan_stock_price.json`
- For indicators (e.g., SuperTrend) your preprocessed JSON / CSV must contain `high`, `low`, `close` columns and either a datetime index or a `Date` / `timestamp` column.

## Troubleshooting
- "No data found for <id>": create a matching `engine/datasets/{company}-{stock_id}` and add `preprocessed_*{stock_id}*.json` or place price JSON in `cache/{company}-{stock_id}/`.
- "Missing columns" for indicators: ensure `high`, `low`, `close` are present in the preprocessed file.
- To debug FinMind fetch issues, examine `engine/robot/temp/` for raw API dumps and logs in `logs/`.
- Increase verbosity by checking the `logs/` directory or running subprocess commands locally to see stderr/stdout.

## Testing
- Run unit tests with pytest:
```bash
pytest -q
```
- Add small regression tests for indicator computation if modifying `engine/datasets/indicators.py`.

## Output
- Individual reports in `reports/`
- Summary in `summary/`
- Logs in `logs/`

## Contributing
- Create a branch, add tests for new behavior, and open a pull request against `master`.