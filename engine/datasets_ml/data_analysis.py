import pandas as pd
import json
import os

def analyze_stock_data(json_path: str):
    """
    Load preprocessed JSON data and perform basic financial analysis.
    """
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        data = raw.get('data', [])

    df = pd.DataFrame(data)
    if df.empty:
        print("No data found in JSON.")
        return

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    print(f"--- Analysis for {df['stock_id'].iloc[0] if 'stock_id' in df.columns else 'Unknown'} ---")
    print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total Trading Days: {len(df)}")

    # Price Performance
    if 'close' in df.columns:
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        returns = (end_price - start_price) / start_price * 100
        print(f"Price Performance: {start_price:.2f} -> {end_price:.2f} ({returns:+.2f}%)")

    # Financial Stats (EPS & Gross Profit)
    cols = ['eps', 'gross_profit']
    for col in cols:
        if col in df.columns and not df[col].isnull().all():
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if not vals.empty:
                print(f"{col.upper()} - Mean: {vals.mean():.2f}, Max: {vals.max():.2f}, Min: {vals.min():.2f}")
        else:
            print(f"{col.upper()} data not available.")

    # Correlation Analysis
    if all(c in df.columns for c in ['close', 'eps', 'gross_profit']):
        # Drop rows where financials are null for correlation
        corr_df = df[['close', 'eps', 'gross_profit']].apply(pd.to_numeric, errors='coerce').dropna()
        if not corr_df.empty:
            corr = corr_df.corr()
            print("\nCorrelation Matrix:")
            print(corr)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Test with the 1727 preprocessed data we just generated
    target_json = os.path.join(base_dir, "preprocessed_1727.json")
    analyze_stock_data(target_json)
