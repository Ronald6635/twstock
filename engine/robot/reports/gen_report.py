"""Report analyzer and CSV exporter for engine/robot/reports/*.txt

Features
- parse_report_file(path): extract stock_id, stock_name, sections (skip 風險提醒), composite_score and signal statistics
- export_reports_to_csv(dir, out_csv): walk analysis_*.txt, parse and write CSV sorted by composite_score (desc)
- CLI entrypoint

Design goals: robust to encoding (utf-8/utf-8-sig/big5 fallback), tolerant parsing heuristics, return diagnostics for failures.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Public CSV header (columns kept stable)
CSV_COLUMNS = [
    "stock_name",
    "stock_id",
    "trend_reversal_detection",
    "consistency_diagnosis",
    "volume_price_divergence",
    "multi_ma",
    "stoploss_takeprofit_reference",
    "key_observation_levels",
    "event_risk",
    "indicator_analysis_summary",
    "signal_statistics",
    "professional_advice",
    "risk_management_advice",
    "stat_buy_strength",
    "stat_sell_strength",
    "stat_neutral",
    "composite_score",
]

# Encoding fallback order
_ENCODING_FALLBACK: Tuple[str, ...] = ("utf-8", "utf-8-sig", "big5", "latin-1")

# Helpful regexes
_RE_COMPOSITE = re.compile(r"綜合評分\s*[:：]\s*([+\-]?\d+(?:\.\d+)?)")
_RE_FILENAME_ID = re.compile(r"analysis_(?P<id>.+?)\.txt$")
_RE_LOADED = re.compile(r"preprocessed[_-](?P<name>.+?)\.(?:json|csv)")
_RE_SECTION_HEADING = re.compile(r"^\s*(?P<header>[^\n]{1,60}?)\s*[:：]")
_RE_STAT_KV = re.compile(r"(?P<key>買|賣|中性)[^\d\n\-]*?(?P<val>[+\-]?\d+(?:\.\d+)?)")


def _normalize_section_header(header: str) -> str:
    """Normalize section header text.

    This reporter output often prefixes headings with emojis (e.g., "🛡️ ", "⚠️ ").
    We strip leading non-word characters so mapping and filtering are stable.
    """
    header = header.strip()
    header = re.sub(r"^[\s\W]+", "", header)
    return header.strip()


def _normalize_text(s: str) -> str:
    """Basic normalization: full-width → half-width, normalize whitespace, strip BOM-like chars."""
    if not s:
        return s
    # map full-width digits and punctuation to ASCII equivalents (minimal)
    fw = "０１２３４５６７８９：，．－／（）；，"
    bw = "0123456789:,.-/();,"
    trans = str.maketrans({k: v for k, v in zip(fw, bw)})
    s = s.translate(trans)
    # normalize common full-width spaces and control chars
    s = s.replace("\u3000", " ").replace("\u200b", "")
    return s.strip()


@dataclass
class ParseResult:
    stock_id: str
    stock_name: Optional[str]
    sections: Dict[str, str]
    composite_score: Optional[float]
    stat_buy_strength: Optional[float]
    stat_sell_strength: Optional[float]
    stat_neutral: Optional[float]
    diagnostics: List[str]


class ReportParseError(RuntimeError):
    pass


def _open_with_fallback(path: Path, encoding: Optional[str] = None) -> Tuple[str, str]:
    """Open file with encoding fallback; return (text, used_encoding)."""
    encodings: Tuple[str, ...] = (encoding,) if encoding else _ENCODING_FALLBACK

    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, errors="strict") as f:
                return f.read(), enc
        except Exception as exc:  # UnicodeDecodeError or others
            last_exc = exc
            continue
    # last resort: read binary and decode lossy
    with path.open("rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = data.decode("latin-1", errors="replace")
    return text, f"fallback({type(last_exc).__name__})"


def parse_report_file(path: Path, *, encoding: Optional[str] = None, strict: bool = False) -> ParseResult:
    """Parse a single analysis_*.txt report.

    Returns ParseResult with parsed sections (skips 風險提醒) and diagnostics.
    If strict=True, raise ReportParseError on critical failures (missing composite_score).
    """
    diagnostics: List[str] = []
    text, used_enc = _open_with_fallback(path, encoding)
    diagnostics.append(f"encoding={used_enc}")
    text = _normalize_text(text)
    lines = text.splitlines()

    # stock_id from filename (primary)
    m = _RE_FILENAME_ID.search(path.name)
    stock_id = m.group("id") if m else path.stem
    stock_name: Optional[str] = None

    # try to extract stock_name from filename if the filename encodes both
    if "-" in stock_id:
        # e.g., 元大MSCI金融-0055 or 0055-元大...
        parts = stock_id.rsplit("-", 1)
        if parts[0].strip() and parts[1].strip():
            # convention: name-id OR id-name
            if parts[1].isdigit() or re.match(r"\w+\d+", parts[1]):
                stock_name = parts[0].strip()
                stock_id = parts[1].strip()
            else:
                # fallback
                stock_name = parts[1].strip()
                stock_id = parts[0].strip()

    # scan for 'Loaded data from' style lines for name/id
    for L in lines[:30]:
        m2 = _RE_LOADED.search(L)
        if m2:
            candidate = m2.group("name")
            if "-" in candidate:
                nm, sid = candidate.rsplit("-", 1)
                stock_name = stock_name or nm
                stock_id = sid
            else:
                # could be only id or only name
                if candidate.isdigit():
                    stock_id = candidate
                else:
                    stock_name = stock_name or candidate
            diagnostics.append(f"found_preprocessed={candidate}")
            break

    # extract composite score(s)
    composite_matches = []
    for L in lines:
        mm = _RE_COMPOSITE.search(L)
        if mm:
            try:
                composite_matches.append((float(mm.group(1)), L.strip()))
            except Exception:
                diagnostics.append(f"bad_number:{mm.group(1)}")

    composite_score: Optional[float]
    if not composite_matches:
        composite_score = None
        diagnostics.append("no_composite_score_found")
        if strict:
            raise ReportParseError(f"no composite score in {path}")
    else:
        # choose last occurrence as canonical
        composite_score = composite_matches[-1][0]
        if len(composite_matches) > 1:
            diagnostics.append(f"multiple_composite_scores_found:{[c for c,_ in composite_matches]}")

    # Section parsing: collect lines under headings (heading ends with ':' and isn't a bullet)
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for L in lines:
        if not L.strip():
            # blank line separates content but keep current
            continue
        # heading detection
        hm = _RE_SECTION_HEADING.match(L)
        is_emoji_heading = L.lstrip().startswith(("🔀", "🔍", "⚡", "📅", "🛡️", "🎯", "🚨", "📈", "💡"))
        if (hm or is_emoji_heading) and not L.lstrip().startswith("•") and "理由" not in L and "部位規模" not in L and "當前價格" not in L:
            if hm:
                header = hm.group("header").strip()
            else:
                header = L.strip()
            header = header.replace("📈 ", "").replace("🔍 ", "")
            header = _normalize_section_header(header)
            current = header
            sections.setdefault(current, [])
            # Extract content after the colon on the same line
            parts = L.split(':', 1)
            if len(parts) > 1:
                content = parts[1].strip()
                if content:
                    sections[current].append(content)
            continue
        # bullets or content
        if current:
            sections.setdefault(current, []).append(L.strip())

    # filter out 風險提醒
    cleaned_sections: Dict[str, str] = {}
    for k, v in sections.items():
        if "風險提醒" in k or "風險" == k:
            diagnostics.append(f"skipped_section:{k}")
            continue
        if "免責" in k or "免責聲明" in k:
            diagnostics.append(f"skipped_section:{k}")
            continue
        cleaned_sections[k] = "\n".join(v).strip()

    # extract signal stats from a likely section name
    stat_buy = stat_sell = stat_neu = None
    for sec_name in ("📈 訊號統計", "訊號統計", "Signal statistics", "訊號"):
        if sec_name in sections:
            txt = "\n".join(sections[sec_name])
            for sm in _RE_STAT_KV.finditer(txt):
                key = sm.group("key")
                val = float(sm.group("val"))
                if key == "買":
                    stat_buy = val
                elif key == "賣":
                    stat_sell = val
                elif key == "中性":
                    stat_neu = val
            break

    # best-effort mapping of common section names to CSV columns
    mapping = {
        "趨勢反轉偵測": "trend_reversal_detection",
        "一致性診斷": "consistency_diagnosis",
        "量價同步": "volume_price_divergence",
        "量價背離": "volume_price_divergence",
        "量價結構": "volume_price_divergence",
        "多時框均線分析": "multi_ma",
        "成交量": "volume_price_divergence",
        "停損/停利參考": "stoploss_takeprofit_reference",
        "止損/停利參考": "stoploss_takeprofit_reference",
        "停損/停利": "stoploss_takeprofit_reference",
        "止損/停利": "stoploss_takeprofit_reference",
        "關鍵觀察價位": "key_observation_levels",
        "關鍵支撐/阻力": "key_observation_levels",
        "事件風險": "event_risk",
        "基本面紅旗": "event_risk",
        "指標分析摘要": "indicator_analysis_summary",
        "訊號統計": "signal_statistics",
        "📈 訊號統計": "signal_statistics",
        "專業投資建議": "professional_advice",
        "建議": "professional_advice",
        "風險管理建議": "risk_management_advice",
        "風險管理": "risk_management_advice",
    }

    row_sections: Dict[str, str] = {col: "" for col in CSV_COLUMNS}
    row_sections["stock_id"] = stock_id
    row_sections["stock_name"] = stock_name or ""

    for sec, text in cleaned_sections.items():
        for pattern, col in mapping.items():
            if pattern in sec or pattern in text[:40]:
                # Allow combining risk_management_advice if multiple sources
                if col == "risk_management_advice" and row_sections[col]:
                    if text and text not in row_sections[col]:
                        row_sections[col] = row_sections[col] + "\n" + text
                else:
                    row_sections[col] = text
                break
        else:
            # unmatched - append to indicator_analysis_summary
            if len(text) < 200:
                if row_sections["indicator_analysis_summary"]:
                    row_sections["indicator_analysis_summary"] += "\n" + f"{sec}: {text}"
                else:
                    row_sections["indicator_analysis_summary"] = f"{sec}: {text}"

    result = ParseResult(
        stock_id=stock_id,
        stock_name=stock_name,
        sections=row_sections,
        composite_score=composite_score,
        stat_buy_strength=stat_buy,
        stat_sell_strength=stat_sell,
        stat_neutral=stat_neu,
        diagnostics=diagnostics,
    )
    return result


def export_reports_to_csv(
    reports_dir: Path | str,
    out_csv: Path | str,
    *,
    sort_by: str = "composite_score",
    encoding: Optional[str] = None,
    strict: bool = False,
    stockinfo_path: Optional[Path | str] = None,
    auto_fill: bool = True,
) -> int:
    """Scan reports_dir for analysis_*.txt, parse and write CSV.

    By default this function will try to auto-fill `stock_name` using
    `docs/tw_stock_info.json` if it exists (auto_fill=True). To override,
    pass `stockinfo_path` or set `auto_fill=False`.

    Returns number of rows written.
    """
    reports_dir = Path(reports_dir)
    out_csv = Path(out_csv)
    rows: List[Dict[str, Any]] = []
    diagnostics_all: Dict[str, List[str]] = {}

    for p in sorted(reports_dir.glob("analysis_*.txt")):
        try:
            pr = parse_report_file(p, encoding=encoding, strict=strict)
        except ReportParseError:
            logger.exception("critical parse error: %s", p)
            if strict:
                raise
            continue
        row: Dict[str, Any] = {c: "" for c in CSV_COLUMNS}
        # populate known columns
        row.update(pr.sections)
        row["stock_id"] = pr.stock_id
        row["stock_name"] = pr.stock_name or ""
        # numeric stats
        row["stat_buy_strength"] = pr.stat_buy_strength if pr.stat_buy_strength is not None else ""
        row["stat_sell_strength"] = pr.stat_sell_strength if pr.stat_sell_strength is not None else ""
        row["stat_neutral"] = pr.stat_neutral if pr.stat_neutral is not None else ""
        row["composite_score"] = pr.composite_score if pr.composite_score is not None else ""
        rows.append(row)
        diagnostics_all[p.name] = pr.diagnostics

    # sorting: put rows with numeric composite_score first in descending order
    def _score_key(r: Dict[str, Any]) -> float:
        v = r.get(sort_by, "")
        try:
            return float(v)
        except Exception:
            return float("-inf") if sort_by != "composite_score" else float("-inf")

    rows.sort(key=_score_key, reverse=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # auto-fill stock_name if requested and mapping exists
    if auto_fill:
        def _find_upwards(relpath: str, max_levels: int = 6) -> Optional[Path]:
            """Search current dir and parents for relpath; return first existing Path or None."""
            cur = Path.cwd()
            for _ in range(max_levels + 1):
                cand = cur / relpath
                if cand.exists():
                    return cand
                if cur.parent == cur:
                    break
                cur = cur.parent
            return None

        candidate: Optional[Path]
        if stockinfo_path:
            candidate = Path(stockinfo_path)
        else:
            # try common locations: docs/tw_stock_info.json somewhere above cwd
            candidate = _find_upwards("docs/tw_stock_info.json")
            # also accept repo-root-level tw_stock_info.json (legacy)
            if candidate is None:
                candidate = _find_upwards("tw_stock_info.json")

        if candidate and candidate.exists():
            try:
                logger.info("auto-filling stock_name using %s", candidate)
                enrich_csv_with_stockinfo(out_csv, candidate, out_csv=out_csv)
            except Exception:
                logger.exception("auto-fill failed using %s", candidate)
        else:
            logger.debug("no stockinfo mapping found (searched upwards); skipping auto-fill")

    # emit summary diagnostics at INFO
    logger.info("wrote %d rows to %s", len(rows), out_csv)
    # some diagnostics (only warnings/errors) to debug
    small_diag = {k: v for k, v in diagnostics_all.items() if any("no_composite_score" in d or "multiple_composite_scores" in d for d in v)}
    if small_diag:
        logger.warning("reports with warnings: %s", small_diag)

    return len(rows)


def enrich_csv_with_stockinfo(
    csv_path: Path | str,
    stockinfo_path: Path | str,
    *,
    out_csv: Optional[Path | str] = None,
    id_col: str = "stock_id",
    name_col: str = "stock_name",
) -> Dict[str, Any]:
    """Backfill `name_col` in `csv_path` using `stockinfo_path` (JSON array of objects with `stock_id`/`stock_name`).

    Behavior:
    - try exact match, then try trimmed-leading-zeros, then zero-pad to 4 digits
    - do not modify rows that already have a non-empty `name_col`
    - write result to `out_csv` (if None, overwrite `csv_path`)

    Returns a summary dict: {total, filled, remaining, samples_filled}
    """
    csv_path = Path(csv_path)
    stockinfo_path = Path(stockinfo_path)
    out_csv = Path(out_csv) if out_csv is not None else csv_path.with_name(csv_path.stem + ".filled" + csv_path.suffix)

    # load JSON mapping (last seen stock_name wins)
    import json

    mapping: Dict[str, str] = {}
    with stockinfo_path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    for item in data:
        sid = str(item.get("stock_id", "")).strip()
        name = item.get("stock_name") or item.get("name") or ""
        if sid and name:
            mapping[sid.upper()] = name

    def _lookup(sid: str) -> Optional[str]:
        if not sid:
            return None
        sid_s = str(sid).strip()
        cand = sid_s.upper()
        # try exact
        if cand in mapping:
            return mapping[cand]
        # try strip leading zeros
        s_nozero = cand.lstrip("0")
        if s_nozero and s_nozero in mapping:
            return mapping[s_nozero]
        # try zero-pad to 4 (common display)
        if cand.isdigit():
            padded = cand.zfill(4)
            if padded in mapping:
                return mapping[padded]
        # try uppercase (already done), fallback None
        return None

    total = 0
    filled = 0
    remaining = 0
    samples_filled: List[Tuple[str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as rf:
        rdr = list(csv.DictReader(rf))

    for row in rdr:
        total += 1
        cur_name = (row.get(name_col) or "").strip()
        sid = (row.get(id_col) or "").strip()
        if cur_name:
            continue
        candidate = _lookup(sid)
        if candidate:
            row[name_col] = candidate
            filled += 1
            samples_filled.append((sid, candidate))
        else:
            remaining += 1

    # write out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rdr:
            writer.writerow(r)

    summary = {
        "total": total,
        "filled": filled,
        "remaining": remaining,
        "out_csv": str(out_csv),
        "samples_filled": samples_filled[:20],
    }
    logger.info("enrich summary: %s", summary)
    return summary


def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="report-analyzer", description="Export analysis_*.txt to CSV")
    p.add_argument("--reports-dir", required=True, help="path to engine/robot/reports")
    p.add_argument("--out-csv", required=True, help="output CSV path")
    p.add_argument("--encoding", default=None, help="force encoding (optional)")
    p.add_argument("--strict", action="store_true", help="fail on parse errors")
    p.add_argument("--sort-by", default="composite_score")
    # default behavior: auto-fill using docs/tw_stock_info.json when present
    p.add_argument("--fill-names", default=None, help="(optional) explicit path to tw_stock_info.json to fill missing stock_name")
    p.add_argument("--no-fill-names", dest="auto_fill", action="store_false", help="disable automatic filling of stock_name from docs/tw_stock_info.json")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    n = export_reports_to_csv(
        args.reports_dir,
        args.out_csv,
        sort_by=args.sort_by,
        encoding=args.encoding,
        strict=args.strict,
        stockinfo_path=args.fill_names,
        auto_fill=args.auto_fill,
    )
    return 0 if n >= 0 else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
