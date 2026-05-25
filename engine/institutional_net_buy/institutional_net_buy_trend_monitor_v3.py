"""
Institutional Advanced Trend Monitor Module (V3 - Cost Deviation & Intensity Edition)

This script leverages fully pivoted TFRecords containing VWAP, Buy/Sell volumes,
and dynamic Institutional Inventory Costs.
"""

import argparse
from pathlib import Path
import sys
import io
import numpy as np
import pandas as pd
import tensorflow as tf

# 定義各法人的加權權重
DEFAULT_WEIGHTS = {
    "foreign": 1.0,
    "foreign_dealer": 1.0,
    "trust": 1.0,
    "dealer": 0.8,
    "dealer_hedge": 0.5,
}

def _safe_slope(values: pd.Series) -> float:
    """計算序列的線性斜率，過濾無效值。"""
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    x_axis = np.arange(len(clean), dtype=float)
    slope, _ = np.polyfit(x_axis, clean.to_numpy(dtype=float), deg=1)
    return float(slope)

def load_tfrecord_to_dataframe(tfrecord_path: Path) -> pd.DataFrame:
    """讀取包含 VWAP、純買賣與成本特徵的 TFRecord"""
    feature_description = {
        "date": tf.io.FixedLenFeature([], tf.string),
        "stock_id": tf.io.FixedLenFeature([], tf.string),
        "stock_name": tf.io.FixedLenFeature([], tf.string),
        "close": tf.io.FixedLenFeature([], tf.float32),
        "vwap": tf.io.FixedLenFeature([], tf.float32),
    }
    
    # 完整載入 5 大法人的 4 種特徵 (net_buy, buy, sell, cost)
    for prefix in DEFAULT_WEIGHTS.keys():
        feature_description[f"{prefix}_net_buy"] = tf.io.FixedLenFeature([], tf.int64)
        feature_description[f"{prefix}_buy"] = tf.io.FixedLenFeature([], tf.int64)
        feature_description[f"{prefix}_sell"] = tf.io.FixedLenFeature([], tf.int64)
        feature_description[f"{prefix}_cost"] = tf.io.FixedLenFeature([], tf.float32)

    rows = []
    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path), buffer_size=1024 * 1024 * 100)
    for raw_example in raw_dataset:
        parsed = tf.io.parse_single_example(raw_example, feature_description)
        row = {
            "date": parsed["date"].numpy().decode("utf-8"),
            "stock_id": str(parsed["stock_id"].numpy().decode("utf-8")),
            "stock_name": parsed["stock_name"].numpy().decode("utf-8"),
            "close": float(parsed["close"].numpy()),
            "vwap": float(parsed["vwap"].numpy()),
        }
        for prefix in DEFAULT_WEIGHTS.keys():
            row[f"{prefix}_net_buy"] = int(parsed[f"{prefix}_net_buy"].numpy())
            row[f"{prefix}_buy"] = int(parsed[f"{prefix}_buy"].numpy())
            row[f"{prefix}_sell"] = int(parsed[f"{prefix}_sell"].numpy())
            row[f"{prefix}_cost"] = float(parsed[f"{prefix}_cost"].numpy())
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    # === 核心特徵工程 ===
    weighted_net = pd.Series(0.0, index=df.index)
    weighted_vol = pd.Series(0.0, index=df.index)
    
    for prefix, weight in DEFAULT_WEIGHTS.items():
        weighted_net += df[f"{prefix}_net_buy"] * weight
        # 總週轉量 = 純買 + 純賣
        weighted_vol += (df[f"{prefix}_buy"] + df[f"{prefix}_sell"]) * weight
    
    df["total_net_buy"] = weighted_net
    df["total_volume"] = weighted_vol
    # 計算籌碼純度 (Intensity)，加上 1e-5 避免除以零
    df["conviction_intensity"] = weighted_net / (weighted_vol + 1e-5)
    
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return df

def _build_recommendation(
    vwap_trend: str, 
    intensity: float, 
    trust_dev: float, 
    foreign_dev: float, 
    net_buy_positive: bool
) -> str:
    """依據成本乖離與純度，輸出精準的戰術建議"""
    if not net_buy_positive:
        return "法人近期偏空，建議觀望。"
        
    # 策略 1：錯殺撈底 (股價跌破投信或外資成本，但近期仍強力買超)
    if (trust_dev < -0.01 or foreign_dev < -0.01) and intensity > 0.3:
        return "🔥 【強力關注】股價跌破法人防守成本線，且籌碼進貨純度高，隨時醞釀報復性反彈！"
    
    # 策略 2：主升段順風車 (均價上升，成本獲利拉開中，持續買進)
    if vwap_trend == "上升" and intensity > 0.5:
        return "🚀 【主升段確認】均價上升且法人純買進意願極強(當沖少)，建議順勢抱緊。"
        
    # 策略 3：底部吸籌 (均價盤整，成本附近吃貨)
    if vwap_trend == "盤整" and abs(trust_dev) < 0.03 and intensity > 0.2:
        return "⏳ 【底部建倉】法人在成本均線附近溫和吸籌，股價尚未發動，適合耐心佈局。"

    return "法人偏多但訊號一般，列入常規觀察名單。"

def analyze_trends(
    dataframe: pd.DataFrame,
    recent_days: int,
    baseline_days: int,
    min_history_days: int,
    min_positive_ratio: float,
    top_k: int,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    # 🛠️ 經理人關鍵修正：強制排除台灣市場所有 ETF 與主動型基金 (代碼 00 開頭與 A/B 結尾標的)
    dataframe = dataframe[~dataframe['stock_id'].str.startswith('00')]
    dataframe = dataframe[~dataframe['stock_id'].str.contains(r'[A-Za-z]', na=False)]

    candidates = []
    grouped = dataframe.groupby(["stock_id", "stock_name"], sort=False)

    for (stock_id, stock_name), stock_data in grouped:
        stock_data = stock_data.sort_values("date").reset_index(drop=True)
        if len(stock_data) < max(min_history_days, recent_days + 5):
            continue

        recent_slice = stock_data.tail(recent_days)
        baseline_slice = stock_data.iloc[:-recent_days].tail(baseline_days)
        if baseline_slice.empty:
            continue

        recent_total = recent_slice["total_net_buy"].sum()
        recent_positive_ratio = (recent_slice["total_net_buy"] > 0).mean()
        
        # 基礎過濾：近期必須是偏多格局
        if recent_total <= 0 or recent_positive_ratio < min_positive_ratio:
            continue

        # 抓取最新一天的 VWAP 與法人成本
        last_row = recent_slice.iloc[-1]
        current_vwap = last_row["vwap"]
        trust_cost = last_row["trust_cost"]
        foreign_cost = last_row["foreign_cost"]
        
        # 計算成本乖離率 (Deviation = (現價 - 成本) / 成本)
        trust_dev = (current_vwap - trust_cost) / trust_cost if trust_cost > 0 else 0.0
        foreign_dev = (current_vwap - foreign_cost) / foreign_cost if foreign_cost > 0 else 0.0
        
        # 計算近期進貨純度
        recent_intensity_avg = recent_slice["conviction_intensity"].mean()

        # 計算 VWAP 價格動能
        vwap_start = recent_slice.iloc[0]["vwap"]
        vwap_end = current_vwap
        price_change_pct = 0.0 if vwap_start == 0 else ((vwap_end - vwap_start) / vwap_start * 100.0)
        price_trend = "上升" if price_change_pct > 2 else ("下降" if price_change_pct < -2 else "盤整")

        # === 核心評分系統 (總分 100) ===
        signal_strength = 0.0
        signal_strength += min(30.0, recent_positive_ratio * 30.0)         # 連續買超天數 (30%)
        signal_strength += min(40.0, max(0.0, recent_intensity_avg * 60.0)) # 籌碼純度 (40%)
        
        # 【破底翻加分機制】：如果跌破投信或外資成本 1%~5% 區間，給予大幅加分 (錯殺紅利)
        if -0.05 < trust_dev < -0.01:
            signal_strength += 15.0
        if -0.05 < foreign_dev < -0.01:
            signal_strength += 15.0
            
        # 動能加分
        signal_strength += min(15.0, max(0.0, price_change_pct))

        observed = (
            f"VWAP波段變化: {vwap_start:.2f} -> {vwap_end:.2f} ({price_change_pct:+.2f}%) | "
            f"進貨純度: {recent_intensity_avg:.1%} | "
            f"投信成本: {trust_cost:.2f} (乖離 {trust_dev:+.2%}) | "
            f"外資成本: {foreign_cost:.2f} (乖離 {foreign_dev:+.2%})"
        )

        candidates.append({
            "stock_id": stock_id,
            "stock_name": stock_name,
            "signal_strength": round(signal_strength, 2),
            "recent_total_net_buy": round(recent_total, 2),
            "recent_intensity": round(recent_intensity_avg, 3),
            "trust_dev": round(trust_dev, 4),
            "foreign_dev": round(foreign_dev, 4),
            "observed_phenomenon": observed,
            "recommendation_reason": _build_recommendation(
                price_trend, recent_intensity_avg, trust_dev, foreign_dev, recent_total > 0
            ),
        })

    if not candidates:
        return pd.DataFrame()

    return pd.DataFrame(candidates).sort_values("signal_strength", ascending=False).head(top_k).reset_index(drop=True)

def main():
    # 強制 stdout 使用 UTF-8 編碼（支援 emoji 在 Windows）
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="法人成本乖離與純度監控 (V3)")
    parser.add_argument("--tfrecord-path", type=str, required=True, help="輸入 V3 版 TFRecord 檔案")
    parser.add_argument("--recent-days", type=int, default=10, help="近期觀察天數")
    parser.add_argument("--baseline-days", type=int, default=20, help="基準期天數")
    parser.add_argument("--min-history-days", type=int, default=40, help="最少歷史天數")
    parser.add_argument("--min-positive-ratio", type=float, default=0.6, help="偏多天數比例門檻")
    parser.add_argument("--top-k", type=int, default=20, help="輸出前 N 檔")
    parser.add_argument("--output-csv", type=str, default="trend_candidates.csv", help="輸出候選股票清單 CSV 檔案")
    args = parser.parse_args()

    tfrecord_path = Path(args.tfrecord_path)
    if not tfrecord_path.exists():
        print(f"找不到檔案: {tfrecord_path}")
        return

    df = load_tfrecord_to_dataframe(tfrecord_path)
    if df.empty:
        print("TFRecord 無有效資料。")
        return

    result = analyze_trends(
        dataframe=df,
        recent_days=args.recent_days,
        baseline_days=args.baseline_days,
        min_history_days=args.min_history_days,
        min_positive_ratio=args.min_positive_ratio,
        top_k=args.top_k,
    )

    if result.empty:
        print("未找到符合條件的股票。")
        return

    print("\n🎯【法人籌碼純度與成本乖離監控雷達】(依戰術訊號強度排序):")
    for rank, row in enumerate(result.itertuples(index=False), start=1):
        print("=" * 90)
        print(f"[{rank:02d}] {row.stock_name} ({row.stock_id}) | 綜合戰力: {row.signal_strength} 分")
        print(f" 📊 數據透視: {row.observed_phenomenon}")
        print(f" 💡 戰術判定: {row.recommendation_reason}")
    
    # 輸出結果到 CSV 檔案
    output_csv_path = Path(args.output_csv)
    result.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n✅ 輸出 {len(result)} 檔候選股票到: {output_csv_path}")

if __name__ == "__main__":
    main()