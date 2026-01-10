from flask import Blueprint, jsonify
import twstock
import json
import os
from datetime import datetime
from .utils import generate_stock_chart, safe_float, safe_int, save_data_to_cache

bp = Blueprint('stock', __name__)

@bp.route('/api/stock/<stock_id>')
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
        stock.fetch_31()  # Fetch recent data

        # Get realtime data
        realtime = twstock.realtime.get(stock_id)
        if not realtime.get('success', False):
            return jsonify({'error': f'Failed to fetch realtime data for {stock_id}'}), 500
        
        realtime_data = realtime['realtime']
        
        current_price = safe_float(realtime_data.get('latest_trade_price', '-'))
        open_price = safe_float(realtime_data.get('open', '-'))
        high_price = safe_float(realtime_data.get('high', '-'))
        low_price = safe_float(realtime_data.get('low', '-'))
        volume = safe_int(realtime_data.get('accumulate_trade_volume', '-'))
        
        # Calculate price change
        previous_price = stock.price[-2] if len(stock.price) > 1 else None
        
        if current_price is not None and previous_price is not None:
            price_change = round(((current_price - previous_price) / previous_price) * 100, 2)
        else:
            price_change = None

        dates = [d.strftime("%Y-%m-%d") for d in stock.date[-30:]]  # Last 30 days
        prices = stock.price[-30:]

        # BestFourPoint analysis
        bfp = twstock.BestFourPoint(stock)
        best_four_point = bfp.best_four_point()
        best_four_point_str = best_four_point[1] if best_four_point else None

        # Generate chart
        chart_image_base64 = generate_stock_chart(dates, prices, stock_id)

        response_data = {
            'stock_id': stock_id,
            'current_price': current_price,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'volume': volume,
            'price_change': price_change,
            'best_four_point': best_four_point_str,
            'chart_image': chart_image_base64,
            'dates': dates,
            'prices': prices
        }

        # Save to cache (exclude chart_image to avoid large files)
        data_to_save = response_data.copy()
        data_to_save.pop('chart_image', None)
        save_data_to_cache(stock_id, data_to_save, 'stock')

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
