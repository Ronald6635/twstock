from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static', template_folder='templates')

# In a real application, you would connect to actual stock APIs or databases
# This is a simple mock implementation for demonstration purposes

# Mock stock data
STOCK_INFO = {
    '2330': {'name': '台積電', 'industry': '半導體', 'description': '全球最大的專業積體電路製造服務公司'},
    '2317': {'name': '鴻海', 'industry': '電子零組件', 'description': '全球最大的電子產品代工製造廠商'},
    '0050': {'name': '元大台灣50', 'industry': 'ETF', 'description': '由台灣市值前50大之上市公司股票組成的ETF'},
    '2412': {'name': '中華電', 'industry': '電信服務', 'description': '台灣最大的電信服務供應商'}
}

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to get stock data
@app.route('/api/stock/<stock_id>')
def get_stock_data(stock_id):
    # Check if stock exists in our mock database
    if stock_id not in STOCK_INFO:
        return jsonify({'error': '查無此股票代碼'}), 404
    
    # Get basic stock info
    stock = STOCK_INFO[stock_id]
    
    # Generate mock price data for the chart
    today = datetime.now()
    days = 90  # Generate data for 90 days
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    dates.reverse()  # Oldest date first
    
    # Generate price data with some randomness but a general trend
    np.random.seed(int(stock_id))  # Use stock_id as seed for consistent results
    base_price = np.random.randint(100, 1000)
    
    # Create a general trend with some randomness
    trend = np.cumsum(np.random.normal(0, 1, days)) * 5
    prices = base_price + trend
    prices = np.maximum(prices, 50)  # Ensure no negative prices
    
    # Create chart data
    chart_data = [{'date': date, 'value': round(price, 2)} for date, price in zip(dates, prices)]
    
    # Current price is the last price with a small random change
    current_price = round(prices[-1] * (1 + np.random.uniform(-0.02, 0.02)), 2)
    
    # Price change compared to yesterday
    price_change = round((current_price - prices[-2]) / prices[-2] * 100, 2)
    
    # Mock volume
    volume = np.random.randint(1000, 10000)
    
    response_data = {
        'id': stock_id,
        'name': stock['name'],
        'industry': stock['industry'],
        'description': stock['description'],
        'currentPrice': current_price,
        'priceChange': price_change,
        'volume': volume,
        'chartData': chart_data
    }
    
    return jsonify(response_data)

# You might want to add more routes for other features

if __name__ == '__main__':
    app.run(debug=True)
