# stock_analysis.py
"""
A demonstration of the twstock library for retrieving and analyzing Taiwan stock data.
This script showcases historical data retrieval, basic analysis, real-time data, and buy/sell signals.
"""

import twstock
from datetime import datetime

# === Configuration ===
STOCK_ID = "2330"  # TSMC stock code
START_YEAR, START_MONTH = 2000, 10  # For fetch_from example

# === Helper Function ===
def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


# === 1. Historical Stock Data ===
print_section("1. Historical Stock Data")
stock = twstock.Stock(STOCK_ID)

# Basic stock info
print(f"Stock ID: {stock.sid}")
print(f"Recent Closing Prices: {stock.price[:5]}...")  # Show first 5 for brevity
print(f"Recent High Prices: {stock.high[:5]}...")
print(f"Corresponding Dates: {[d.strftime('%Y-%m-%d') for d in stock.date[:5]]}...")

# Fetch specific historical data
print_section("Fetching Custom Historical Data")
data_2015_july = stock.fetch(2015, 7)
print(f"2015 July Data (first entry): {data_2015_july[0]}")
data_from_2000_oct = stock.fetch_from(START_YEAR, START_MONTH)
print(f"Data from {START_YEAR}-{START_MONTH} (first entry): {data_from_2000_oct[0]}")


# === 2. Basic Stock Analysis ===
print_section("2. Basic Stock Analysis")
five_day_ma = stock.moving_average(stock.price, 5)
print(f"5-Day Moving Average (first 5): {five_day_ma[:5]}")
five_day_volume_ma = stock.moving_average(stock.capacity, 5)
print(f"5-Day Volume MA (first 5): {five_day_volume_ma[:5]}")
bias_ratio = stock.ma_bias_ratio(5, 10)
print(f"5 vs 10-Day Bias Ratio (first 5): {bias_ratio[:5]}")


# === 3. Best Four Point Analysis ===
print_section("3. Best Four Point Buy/Sell Signals")
bfp = twstock.BestFourPoint(stock)
buy_signal = bfp.best_four_point_to_buy()
sell_signal = bfp.best_four_point_to_sell()
combined_signal = bfp.best_four_point()
print(f"Buy Signal: {buy_signal}")
print(f"Sell Signal: {sell_signal}")
print(f"Combined Signal: {combined_signal}")


# === 4. Real-Time Data ===
print_section("4. Real-Time Stock Data")
real_time_data = twstock.realtime.get(STOCK_ID)
if real_time_data["success"]:
    print(f"Latest Price: {real_time_data['realtime']['latest_trade_price']}")
    print(f"Open: {real_time_data['realtime']['open']}")
    print(f"High: {real_time_data['realtime']['high']}")
    print(f"Low: {real_time_data['realtime']['low']}")
    print(f"Best Bid Prices: {real_time_data['realtime']['best_bid_price'][:3]}...")
else:
    print(f"Error: {real_time_data['rtmessage']} (Code: {real_time_data['rtcode']})")

# Multiple stocks example
multi_stocks = twstock.realtime.get([STOCK_ID, "2337", "2409"])
if multi_stocks["success"]:
    for stock_id, data in multi_stocks.items():
        if stock_id != "success":  # Skip the success flag
            print(f"{stock_id} Latest Price: {data['realtime']['latest_trade_price']}")


# === 5. Stock Code Validation ===
print_section("5. Stock Code Validation")
print(f"Is {STOCK_ID} a TWSE stock? {'2330' in twstock.twse}")
print(f"Is 6223 a TWSE stock? {'6223' in twstock.twse}")
print(f"Is {STOCK_ID} a TPEX stock? {'2330' in twstock.tpex}")
print(f"Is 6223 a TPEX stock? {'6223' in twstock.tpex}")
print(f"Is {STOCK_ID} a valid Taiwan stock? {'2330' in twstock.codes}")


# === Notes ===
print_section("Notes")
print("Data order: Oldest first, newest last.")
print("BestFourPoint results depend on the current Stock data.")
print("Real-time data requires an internet connection.")
