import csv
import io
from pathlib import Path

import pytest

from engine.robot.reports import analyzer


SAMPLE_NORMAL = """
📊 技術指標專業分析報告:
價格趨勢:
   • 價格趨勢: 上漲 1.2%
📈 訊號統計:
   • 買: 4.0
   • 賣: 1.5
   • 中性: 2
   • 綜合評分: 2.5 (偏多)
⚠️  風險提醒:
   • 注意市場波動
"""

SAMPLE_NO_SCORE = """
📊 技術指標專業分析報告:
價格趨勢:
   • 價格趨勢: 下跌 0.5%
📈 訊號統計:
   • 買: 1
   • 賣: 3
"""

SAMPLE_MULTIPLE_SCORES = """
... summary ...
   • 綜合評分: 1.0 (偏多)
... later ...
   • 綜合評分: 3.5 (偏多)
"""


def write_sample(tmp_path: Path, name: str, content: str, encoding: str = "utf-8") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding=encoding)
    return p


def test_parse_normal_report_returns_expected_fields(tmp_path: Path):
    p = write_sample(tmp_path, "analysis_0055.txt", SAMPLE_NORMAL)
    res = analyzer.parse_report_file(p)
    assert res.composite_score == pytest.approx(2.5)
    assert res.stat_buy_strength == pytest.approx(4.0)
    assert res.stat_sell_strength == pytest.approx(1.5)
    assert "價格趨勢" in res.sections and res.sections["價格趨勢"]


def test_missing_composite_score_lenient_and_strict(tmp_path: Path):
    p = write_sample(tmp_path, "analysis_1521.txt", SAMPLE_NO_SCORE)
    res = analyzer.parse_report_file(p, strict=False)
    assert res.composite_score is None

    with pytest.raises(analyzer.ReportParseError):
        analyzer.parse_report_file(p, strict=True)


def test_ambiguous_multiple_scores_logged(tmp_path: Path, caplog):
    p = write_sample(tmp_path, "analysis_1321.txt", SAMPLE_MULTIPLE_SCORES)
    caplog.clear()
    res = analyzer.parse_report_file(p)
    assert res.composite_score == pytest.approx(3.5)
    assert any("multiple_composite_scores_found" in d for d in res.diagnostics)


def test_encoding_fallback_big5_file(tmp_path: Path, caplog):
    # write Big5 encoded file and ensure parser reads it
    content = "價格趨勢:\n   • 價格趨勢: 上漲 0.1%\n   • 綜合評分: 1.0 (偏多)\n"
    p = tmp_path / "analysis_big5-1234.txt"
    p.write_bytes(content.encode("big5"))
    res = analyzer.parse_report_file(p)
    assert res.composite_score == pytest.approx(1.0)


def test_collect_reports_to_csv_sorting_and_header(tmp_path: Path):
    # create three reports with different composite_score and ensure CSV sorted desc
    write_sample(tmp_path, "analysis_A-1.txt", "   • 綜合評分: 1.0\n")
    write_sample(tmp_path, "analysis_B-2.txt", "   • 綜合評分: 3.0\n")
    write_sample(tmp_path, "analysis_C-3.txt", "   • 綜合評分: 2.0\n")

    out = tmp_path / "out.csv"
    n = analyzer.export_reports_to_csv(tmp_path, out)
    assert n == 3
    with out.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        assert rdr.fieldnames == analyzer.CSV_COLUMNS
        rows = list(rdr)
        # order should be 3.0, 2.0, 1.0
        scores = [float(r["composite_score"]) for r in rows]
        assert scores == [3.0, 2.0, 1.0]


def test_enrich_csv_with_stockinfo_fills_names(tmp_path: Path):
    # prepare CSV with empty stock_name and a small stockinfo JSON
    csv_p = tmp_path / "sample.csv"
    csv_p.write_text("stock_name,stock_id,composite_score\n,0055,2.5\n,2330,3.5\nUNKNOWN,9999,1.0\n", encoding="utf-8")
    json_p = tmp_path / "stockinfo.json"
    json_p.write_text('[{"stock_id":"0055","stock_name":"元大台灣50"},{"stock_id":"2330","stock_name":"台積電"}]', encoding="utf-8")

    summary = analyzer.enrich_csv_with_stockinfo(csv_p, json_p, out_csv=tmp_path / "out.csv")
    assert summary["total"] == 3
    assert summary["filled"] == 2
    assert summary["remaining"] == 1

    out = (tmp_path / "out.csv").read_text(encoding="utf-8")
    assert "元大台灣50" in out
    assert "台積電" in out
    assert "UNKNOWN" in out


def test_export_reports_auto_fills_when_stockinfo_present(tmp_path: Path):
    # create reports and a local docs/tw_stock_info.json; export should auto-fill by default
    reports = tmp_path / "reports"
    docs = tmp_path / "docs"
    reports.mkdir()
    docs.mkdir()
    (reports / "analysis_0055.txt").write_text("   • 綜合評分: 2.5\n", encoding="utf-8")
    (reports / "analysis_2330.txt").write_text("   • 綜合評分: 3.5\n", encoding="utf-8")
    # write docs mapping
    (docs / "tw_stock_info.json").write_text('[{"stock_id":"0055","stock_name":"元大台灣50"},{"stock_id":"2330","stock_name":"台積電"}]', encoding="utf-8")

    out = tmp_path / "out.csv"
    # pass the mapping explicitly to simulate repo docs; auto_fill defaults to True
    n = analyzer.export_reports_to_csv(reports, out, stockinfo_path=docs / "tw_stock_info.json")
    assert n == 2
    with out.open("r", encoding="utf-8") as f:
        rdr = list(csv.DictReader(f))
    assert any(r["stock_name"] == "元大台灣50" for r in rdr)
    assert any(r["stock_name"] == "台積電" for r in rdr)


def test_auto_fill_when_run_from_reports_cwd(tmp_path: Path, monkeypatch):
    # simulate repo root with docs/tw_stock_info.json and running inside reports/
    repo_root = tmp_path / "repo"
    reports = repo_root / "engine" / "robot" / "reports"
    docs = repo_root / "docs"
    reports.mkdir(parents=True)
    docs.mkdir(parents=True)

    # create two report files inside reports/
    (reports / "analysis_0055.txt").write_text("   • 綜合評分: 2.5\n", encoding="utf-8")
    (reports / "analysis_2330.txt").write_text("   • 綜合評分: 3.5\n", encoding="utf-8")
    # create mapping in repo/docs
    (docs / "tw_stock_info.json").write_text('[{"stock_id":"0055","stock_name":"元大台灣50"},{"stock_id":"2330","stock_name":"台積電"}]', encoding="utf-8")

    # change cwd to reports (user will run command there)
    monkeypatch.chdir(reports)

    out = reports / "out.csv"
    # call without explicit stockinfo_path; function should search upwards and find docs/tw_stock_info.json
    n = analyzer.export_reports_to_csv(reports, out)
    assert n == 2
    with out.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any(r["stock_name"] == "元大台灣50" for r in rows)
    assert any(r["stock_name"] == "台積電" for r in rows)



