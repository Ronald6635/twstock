"""
Taiwan Stock Flask Backend Application
This Flask application provides a backend service for retrieving and analyzing Taiwan stock market data.
It utilizes the twstock library to fetch real-time and historical stock information, performs basic analysis,
and generates visualizations.
The application exposes the following endpoints:
- '/' : Main page that renders the frontend interface
- '/api/stock/<stock_id>' : API endpoint that provides detailed stock information including:
    - Current price, open price, high/low prices
    - Price change percentage
    - Historical price data for charting
    - BestFourPoint analysis for investment recommendation
Features:
- Real-time stock data retrieval
- Historical price tracking (last 30 days)
- Price change calculation
- Stock price visualization
- BestFourPoint analysis for investment decisions
Dependencies:
- Flask: Web framework
- pandas: Data processing
- numpy: Numerical operations
- twstock: Taiwan stock market data
- matplotlib: Data visualization
- io, base64: Image handling
"""
"""
Generate a line chart of a stock's closing prices for the last 30 days.
Parameters:
----------
stock_id : str
        The Taiwan stock ID code (e.g., '2330' for TSMC)
Returns:
-------
str
        Base64 encoded PNG image of the stock price chart
Notes:
-----
The function creates a matplotlib figure showing the closing price
trend for the specified stock over the last 30 days.
"""
"""
Render the home page of the application.
Returns:
-------
str
        Rendered HTML template for the Taiwan stock interface
"""
"""
Retrieve comprehensive stock information for the specified stock ID.
Parameters:
----------
stock_id : str
        The Taiwan stock ID code (e.g., '2330' for TSMC)
Returns:
-------
flask.Response
        JSON response containing detailed stock information including:
        - Basic info (ID, name)
        - Current trading data (price, open, high, low, volume)
        - Price change percentage
        - Historical price data for the last 30 days
        - BestFourPoint investment recommendation
Error Responses:
--------------
500:
        If real-time data cannot be retrieved or any other exception occurs
        during data processing
Notes:
-----
This function fetches both historical and real-time data from the twstock
library and performs calculations to determine price changes and investment
recommendations.
"""

from flask import Flask, render_template, jsonify, request # 導入 Flask 相關模組
import pandas as pd # 導入 pandas 模組，用於數據處理
import numpy as np # 導入 numpy 模組，用於數值計算
import os # 導入 os 模組，用於操作系統相關功能
import json # 導入 json 模組，用於處理 JSON 數據
from datetime import datetime, timedelta # 導入 datetime 模組，用於處理日期和時間
import twstock # 導入 twstock 模組，用於台灣股市數據
import matplotlib
matplotlib.use('Agg') # Set backend to Agg to avoid GUI issues
import matplotlib.pyplot as plt # 導入 matplotlib 模組，用於繪圖
import io # 導入 io 模組，用於處理輸入輸出流
import base64 # 導入 base64 模組，用於 base64 編碼

import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

app = Flask(__name__, static_folder='static', template_folder='templates') # 創建 Flask 應用實例，指定靜態文件夾和模板文件夾

# 產生股票圖表
def generate_stock_chart(stock_id):
    stock = twstock.Stock(stock_id) # 創建股票實例
    stock.fetch_31() # 獲取最近 31 天的股票數據
    # Ensure d.date is a string before converting to datetime
    dates = [d.strftime("%Y-%m-%d") for d in stock.date[-30:]] # Limit to last 30 days for chart
    prices = stock.price[-30:] # 提取股價，限制為最近 30 天

    plt.figure(figsize=(10, 6)) # 創建圖表，設置大小
    plt.plot(dates, prices, label="Closing Price", color="blue", marker="o") # 繪製股價折線圖
    plt.title(f"{stock_id} - Last 30 Days Closing Prices") # 設置圖表標題
    plt.xlabel("Date") # 設置 X 軸標籤
    plt.ylabel("Price (NTD)") # 設置 Y 軸標籤
    plt.legend() # 顯示圖例
    plt.xticks(rotation=45) # 旋轉 X 軸刻度
    plt.grid(True) # 顯示網格
    plt.tight_layout() # 調整佈局

    img_stream = io.BytesIO() # 創建內存 IO 對象
    plt.savefig(img_stream, format='png') # 將圖表保存到內存 IO 對象
    img_stream.seek(0) # 將指針移動到 IO 對象的開頭
    img_base64 = base64.b64encode(img_stream.read()).decode('utf-8') # 將 IO 對象中的數據編碼為 base64 字符串
    plt.close() # 關閉圖表
    return img_base64 # 返回 base64 編碼的圖表數據

# Route for the home page # 首頁路由
@app.route('/')
def home():
    return render_template('taiwan-stock.html') # 渲染首頁模板

# API endpoint to get stock data # API 接口，用於獲取股票數據
@app.route('/api/stock/<stock_id>')
def get_stock_data(stock_id):
    """
    API endpoint to retrieve stock data for a given stock ID.
    
    Args:
        stock_id (str): The stock identifier (e.g., '2330').
    
    Returns:
        dict: JSON response with stock data or error message.
    """
    try:
        stock = twstock.Stock(stock_id)  # Create stock instance
        stock.fetch_31() # 獲取最近 31 天的股票數據

        # Get realtime data # 獲取實時數據
        realtime = twstock.realtime.get(stock_id) # 獲取股票實時數據
        if not realtime.get('success', False):
            return jsonify({'error': f'Failed to fetch realtime data for {stock_id}'}), 500
        
        realtime_data = realtime['realtime']
        print(f'Real Time information of {stock_id}:\n{realtime_data}')
        
        # Safely extract and convert prices (handle '-' or invalid values)
        def safe_float(value):
            """Safely convert string to float, returning None if invalid."""
            try:
                return float(value) if value != '-' else None
            except (ValueError, TypeError):
                return None
        
        def safe_int(value):
            """Safely convert string to int, returning None if invalid."""
            try:
                return int(value) if value != '-' else None
            except (ValueError, TypeError):
                return None
        
        current_price = safe_float(realtime_data.get('latest_trade_price', '-'))
        open_price = safe_float(realtime_data.get('open', '-'))
        high_price = safe_float(realtime_data.get('high', '-'))
        low_price = safe_float(realtime_data.get('low', '-'))
        volume = safe_int(realtime_data.get('accumulate_trade_volume', '-'))
        
        print(f'open_price: type is {type(open_price)} and content is {open_price}')
        print(f'current_price: type is {type(current_price)} and content is {current_price}')
        
        # Calculate price change # 計算價格變動
        previous_price = stock.price[-2] if len(stock.price) > 1 else None
        print(f'previous_price:\ntype is {type(previous_price)} and the content is {previous_price}')
        
        if current_price is not None and previous_price is not None:
            price_change = round(((current_price - previous_price) / previous_price) * 100, 2) # 計算價格變動百分比
        else:
            price_change = None

        dates = [d.strftime("%Y-%m-%d") for d in stock.date[-30:]]  # Limit to last 30 days for chart
        prices = stock.price[-30:]  # 提取股價，限制為最近 30 天

        # Generate chart data in the format expected by the frontend
        chart_data = [{'dates': date, 'prices': price} for date, price in zip(dates, prices)]
        chart_image_base64 = generate_stock_chart(stock_id)  # 產生圖表 base64 數據

        # BestFourPoint analysis # BestFourPoint 分析
        bfp = twstock.BestFourPoint(stock) # 創建 BestFourPoint 實例
        best_four_point = bfp.best_four_point() # 獲取 BestFourPoint 分析結果
        best_four_point_str = best_four_point[1] if best_four_point else None # 提取 BestFourPoint 分析結果字符串

        response_data = {
            'id': stock_id, # 股票代碼
            'name': realtime.get('info', {}).get('name', 'Unknown'), # 股票名稱
            'industry': '',  # Not available in twstock # 行業信息，twstock 中沒有提供
            'description': '',  # Not available in twstock # 描述信息，twstock 中沒有提供
            'currentPrice': current_price, # 最新成交價
            'openPrice': open_price, # 開盤價
            'highPrice': high_price, # 最高價
            'priceChange': price_change, # 價格變動百分比
            'lowPrice': low_price, # 最低價
            'volume': volume, # 成交量
            'chartData': chart_data, # 圖表數據
            # 'chartImage': chart_image_base64, # 圖表 base64 數據
            'bestFourPoint': best_four_point_str # BestFourPoint 分析結果
        }

        return jsonify(response_data) # 返回 JSON 格式的響應數據

    except Exception as e:
        app.logger.error(f'Error fetching data for {stock_id}: {str(e)}')
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500 # 返回錯誤信息
    except TypeError as e:
        if 'Data.__new__()' in str(e):
            app.logger.error(f'Data structure mismatch for {stock_id}: {str(e)}')
            return jsonify({'error': f'Data format issue for {stock_id}. Try updating twstock.'}), 500
        else:
            raise  # Re-raise other TypeErrors

# You might want to add more routes for other features # 你可能需要添加更多路由來實現其他功能

if __name__ == '__main__':
    app.run(debug=True) # 啟動 Flask 應用，開啟調試模式
