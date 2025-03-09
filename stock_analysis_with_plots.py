# stock_analysis_with_plots.py
"""
A demonstration of the twstock library for retrieving and analyzing Taiwan stock data,
with visualizations using matplotlib for historical prices and moving averages.
"""

import twstock
from datetime import datetime
import matplotlib.pyplot as plt

# === Configuration ===
STOCK_ID = "2330"  # TSMC stock code
START_YEAR, START_MONTH = 2000, 10  # For fetch_from example
PLOT_DAYS = 30  # Number of days to plot for readability

# === Helper Functions ===
def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")

def plot_prices(dates, prices, ma=None, title="Stock Prices", ylabel="Price (NTD)"):
    """Plot stock prices with optional moving average."""
    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, label="Closing Price", color="blue", marker="o")
    if ma:
        plt.plot(dates, ma, label="5-Day MA", color="orange", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 1. Historical Stock Data ===
print_section("1. Historical Stock Data")
stock = twstock.Stock(STOCK_ID)

# Basic stock info
print(f"Stock ID: {stock.sid}")
print(f"Recent Closing Prices: {stock.price[:5]}...")  # Show first 5 for brevity
print(f"Recent High Prices: {stock.high[:5]}...")
print(f"Corresponding Dates: {[d.strftime('%Y-%m-%d') for d in stock.date[:5]]}...")

# Plot historical prices
dates = [d.strftime("%Y-%m-%d") for d in stock.date[-PLOT_DAYS:]]
prices = stock.price[-PLOT_DAYS:]
plot_prices(dates, prices, title=f"{STOCK_ID} - Last {PLOT_DAYS} Days Closing Prices")

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

# Plot prices with moving average
ma_dates = dates[4:]  # Align with 5-day MA (first 4 days don't have MA)
ma_prices = five_day_ma[-len(ma_dates):]  # Match length with dates
plot_prices(ma_dates, prices[4:], ma_prices, title=f"{STOCK_ID} - Prices with 5-Day MA")


# === 3. Best Four Point Analysis ===
print_section("3. Best Four Point Buy/Sell Signals")
bfp = twstock.BestFourPoint(stock)
buy_signal = bfp.best_four_point_to_buy()
sell_signal = bfp.best_four_point_to_sell()
combined_signal = bfp.best_four_point()
print(f"Buy Signal: {buy_signal}")
print(f"Sell Signal: {sell_signal}")
print(f"Combined Signal: {combined_signal}")


# === 4. Real-Time Stock Data ===
print_section("4. Real-Time Stock Data")
real_time_data = twstock.realtime.get(STOCK_ID)
if real_time_data["success"]:
    print(f"Latest Price: {real_time_data['realtime']['latest_trade_price']}")
    print(f"Open: {real_time_data['realtime']['open']}")
    print(f"High: {real_time_data['realtime']['high']}")
    print(f"Low: {real_time_data['realtime']['low']}")
    print(f"Best Bid Prices: {real_time_data['realtime']['best_bid_price'][:3]}...")
    
    # Plot real-time data (single point)
    rt_prices = [
        float(real_time_data["realtime"]["open"]),
        float(real_time_data["realtime"]["high"]),
        float(real_time_data["realtime"]["low"]),
        float(real_time_data["realtime"]["latest_trade_price"])
    ]
    rt_labels = ["Open", "High", "Low", "Latest"]
    plt.figure(figsize=(8, 5))
    plt.bar(rt_labels, rt_prices, color=["green", "red", "blue", "purple"])
    plt.title(f"{STOCK_ID} - Real-Time Data ({real_time_data['info']['time']})")
    plt.ylabel("Price (NTD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
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
print(f"Plots show the last {PLOT_DAYS} days for historical data.")
