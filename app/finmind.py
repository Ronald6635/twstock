"""
FinMind API Integration Module

This module provides Flask routes for accessing FinMind's Taiwan stock market data.
It supports all Free-tier datasets including technical, fundamental, and other market data.

Key features:
- Comprehensive Taiwan stock data access
- Automatic data caching to datasets
- Chart generation for price data
- Error handling and API key validation

Architecture notes:
- Uses FinMind DataLoader for API access
- Routes follow RESTful conventions
- Data is cached locally for performance
"""

from flask import Blueprint, jsonify, request, render_template
import os
import requests
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import pandas as pd
from .utils import save_data_to_datasets, generate_stock_chart, generate_kline_chart, load_data_from_datasets, generate_plotly_kline_chart
from typing import Dict, List, Any, Optional

bp = Blueprint('finmind', __name__)

# =============================================================================
# TECHNICAL ANALYSIS ROUTES
# =============================================================================

@bp.route('/api/finmind/stock_info')
def get_finmind_stock_info():
    """
    Get Taiwan stock overview information.

    Returns basic information for all listed Taiwan stocks.

    Returns:
        JSON: List of stock information dictionaries

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Return cached if available before requiring API key
        cached = load_data_from_datasets('all', 'finmind_stock_info')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_info()
        response_data = df.to_dict(orient='records')
        save_data_to_datasets('all', response_data, 'finmind_stock_info')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/trading_days')
def get_finmind_trading_days():
    """
    Get Taiwan stock trading calendar.

    Returns all trading days for Taiwan stock market.

    Returns:
        JSON: List of trading day dictionaries

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets('trading_days', 'finmind_trading_days')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_trading_days()
        response_data = df.to_dict(orient='records')
        save_data_to_datasets('trading_days', response_data, 'finmind_trading_days')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/sector_price/<sector>')
def get_finmind_sector_price(sector: str):
    """
    Get Taiwan stock sector price data.

    URL Parameters:
        sector (str): Sector identifier

    Returns:
        JSON: Sector price data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare date range and try cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets(sector, 'finmind_sector_price', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_sector_price(
            sector_id=sector,
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets(sector, response_data, 'finmind_sector_price', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/index')
def get_finmind_index():
    """
    Get Taiwan Weighted Index data.

    Returns daily Taiwan Weighted Index values.

    Returns:
        JSON: Index data with chart

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        cached = load_data_from_datasets('TAIEX', 'finmind_index', start_date, end_date)
        if cached is not None:
            chart_image_base64 = generate_stock_chart([d['date'] for d in cached], [d['price'] for d in cached], 'TAIEX')
            return jsonify({'data': cached, 'chart_image': chart_image_base64})

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_index(
            index_id="TAIEX",
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        chart_image_base64 = generate_stock_chart([d['date'] for d in response_data], [d['price'] for d in response_data], 'TAIEX')
        save_data_to_datasets('TAIEX', response_data, 'finmind_index', start_date, end_date)

        return jsonify({'data': response_data, 'chart_image': chart_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/index_return')
def get_finmind_index_return():
    """
    Get Taiwan and OTC index returns.

    Returns daily returns for weighted and OTC indices.

    Returns:
        JSON: Index return data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets('index_return', 'finmind_index_return', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_index_return(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('index_return', response_data, 'finmind_index_return', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# CHIP ANALYSIS ROUTES
# =============================================================================

@bp.route('/api/finmind/margin_total')
def get_finmind_margin_total():
    """
    Get total market margin purchase and short sale data.

    Returns overall market margin trading statistics.

    Returns:
        JSON: Total margin data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets('total', 'finmind_margin_total', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_margin_purchase_short_sale_total(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('total', response_data, 'finmind_margin_total', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/institutional_total')
def get_finmind_institutional_total():
    """
    Get total institutional investor data for all markets.

    Returns combined institutional investor activity across all Taiwan markets.

    Returns:
        JSON: Total institutional data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets('total', 'finmind_institutional_total', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_institutional_investors_total(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('total', response_data, 'finmind_institutional_total', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/holders/<stock_id>')
def get_finmind_holders(stock_id: str):
    """
    Get foreign investor holdings data.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Foreign holdings data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        cached = load_data_from_datasets(stock_id, 'finmind_holders')
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_holders(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_holders')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/securities_lending/<stock_id>')
def get_finmind_securities_lending(stock_id: str):
    """
    Get securities lending transaction details.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Securities lending data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets(stock_id, 'finmind_securities_lending', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_securities_lending(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_securities_lending', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# FUNDAMENTAL ANALYSIS ROUTES
# =============================================================================

@bp.route('/api/finmind/cash_flow/<stock_id>')
def get_finmind_cash_flow(stock_id: str):
    """
    Get cash flow statement data.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Cash flow data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_cash_flow')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_cash_flows_statement(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_cash_flow')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/income_statement/<stock_id>')
def get_finmind_income_statement(stock_id: str):
    """
    Get income statement data.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Income statement data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_income_statement')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_income_statement(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_income_statement')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/balance_sheet/<stock_id>')
def get_finmind_balance_sheet(stock_id: str):
    """
    Get balance sheet data.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Balance sheet data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_balance_sheet')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_balance_sheet(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_balance_sheet')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/dividend/<stock_id>')
def get_finmind_dividend(stock_id: str):
    """
    Get dividend policy data.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Dividend data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_dividend')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_dividend(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_dividend')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/ex_dividend/<stock_id>')
def get_finmind_ex_dividend(stock_id: str):
    """
    Get ex-dividend and ex-rights results.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Ex-dividend data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_ex_dividend')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_ex_dividend(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_ex_dividend')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/delisted')
def get_finmind_delisted():
    """
    Get delisted Taiwan stocks data.

    Returns information about delisted stocks.

    Returns:
        JSON: Delisted stocks data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets('delisted', 'finmind_delisted')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_delisted()
        response_data = df.to_dict(orient='records')
        save_data_to_datasets('delisted', response_data, 'finmind_delisted')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/split/<stock_id>')
def get_finmind_split(stock_id: str):
    """
    Get stock split reference prices.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Stock split data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_split')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_split(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_split')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/face_value_change/<stock_id>')
def get_finmind_face_value_change(stock_id: str):
    """
    Get face value change reference prices.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: Face value change data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_face_value_change')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_face_value_change(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_face_value_change')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# DERIVATIVES ROUTES
# =============================================================================

@bp.route('/api/finmind/futures_options_daily')
def get_finmind_futures_options_daily():
    """
    Get futures and options daily trading overview.

    Returns daily trading data for futures and options.

    Returns:
        JSON: Futures and options daily data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets('futures_options', 'finmind_futures_options_daily', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.futures_and_options_daily(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('futures_options', response_data, 'finmind_futures_options_daily', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/futures_daily')
def get_finmind_futures_daily():
    """
    Get futures daily trading data.

    Returns daily futures trading information.

    Returns:
        JSON: Futures daily data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets('futures', 'finmind_futures_daily', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.futures_daily(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('futures', response_data, 'finmind_futures_daily', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/options_daily')
def get_finmind_options_daily():
    """
    Get options daily trading data.

    Returns daily options trading information.

    Returns:
        JSON: Options daily data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets('options', 'finmind_options_daily', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.options_daily(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('options', response_data, 'finmind_options_daily', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/futures_institutional')
def get_finmind_futures_institutional():
    """
    Get futures institutional investor data.

    Returns institutional investor activity in futures market.

    Returns:
        JSON: Futures institutional data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets('futures_institutional', 'finmind_futures_institutional', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.futures_institutional_investors(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('futures_institutional', response_data, 'finmind_futures_institutional', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/options_institutional')
def get_finmind_options_institutional():
    """
    Get options institutional investor data.

    Returns institutional investor activity in options market.

    Returns:
        JSON: Options institutional data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Prepare dates and check cache first
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cached = load_data_from_datasets('options_institutional', 'finmind_options_institutional', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.options_institutional_investors(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('options_institutional', response_data, 'finmind_options_institutional', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# OTHER DATA ROUTES
# =============================================================================

@bp.route('/api/finmind/news/<stock_id>')
def get_finmind_news(stock_id: str):
    """
    Get related news for a stock.

    URL Parameters:
        stock_id (str): Stock identifier

    Returns:
        JSON: News data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        cached = load_data_from_datasets(stock_id, 'finmind_news')
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_news(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_news')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/gold_price')
def get_finmind_gold_price():
    """
    Get gold price data.

    Returns high-frequency gold price data (dataset 'GoldPrice').

    Query Parameters:
        start_date (str): YYYY-MM-DD (optional)
        end_date (str): YYYY-MM-DD (optional)

    Returns:
        JSON: List of gold price records (from FinMind 'data' field)

    Status Codes:
        200: Success
        502: Upstream HTTP error
        500: API key not set or other server error
    """
    try:
        import requests

        # Accept optional start_date / end_date query params
        start_date = request.args.get('start_date') or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = request.args.get('end_date') or datetime.now().strftime('%Y-%m-%d')

        # Try cache first
        cached = load_data_from_datasets('gold', 'finmind_gold_price', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        url = 'https://api.finmindtrade.com/api/v4/data'
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "dataset": "GoldPrice",
            "start_date": start_date,
            "end_date": end_date,
        }

        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        response_data = data.get('data', [])

        # Save to datasets for caching/analysis
        save_data_to_datasets('gold', response_data, 'finmind_gold_price', start_date, end_date)

        return jsonify(response_data)
    except requests.HTTPError as he:
        return jsonify({'error': f'HTTP error: {str(he)}'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/crude_oil_price')
def get_finmind_crude_oil_price():
    """
    Get crude oil price data (Brent and WTI).

    Returns daily Brent and WTI crude oil prices.

    Returns:
        JSON: Crude oil price data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        import requests
        # Optional: data_id can be 'WTI' or 'Brent'; default to 'WTI'
        data_id = request.args.get('data_id', 'WTI')
        start_date = request.args.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        # Try cache first
        cached = load_data_from_datasets(data_id, 'finmind_crude_oil_price', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        url = 'https://api.finmindtrade.com/api/v4/data'
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "dataset": "CrudeOilPrices",
            "data_id": data_id,
            "start_date": start_date,
            "end_date": end_date,
        }

        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        response_data = data.get('data', [])

        # Save to datasets for caching/analysis
        save_data_to_datasets(data_id, response_data, 'finmind_crude_oil_price', start_date, end_date)

        return jsonify(response_data)
    except requests.HTTPError as he:
        return jsonify({'error': f'HTTP error: {str(he)}'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/us_stock/<stock_id>')
def get_finmind_us_stock(stock_id: str):
    """
    Get US stock price data.

    URL Parameters:
        stock_id (str): US stock symbol (e.g., 'AAPL')

    Returns:
        JSON: US stock data with chart

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Get date parameters (defaults to last 30 days)
        start_date = request.args.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_us_stock', start_date, end_date)
        if cached is not None:
            # Use 'Close' column for plotting if present, otherwise try lowercase 'close'
            prices = [d.get('Close') if 'Close' in d else d.get('close') for d in cached]
            chart_image_base64 = generate_stock_chart([d['date'] for d in cached], prices, stock_id)
            return jsonify({'data': cached, 'chart_image': chart_image_base64})

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.us_stock_daily(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        # Use 'Close' column for plotting if present, otherwise try lowercase 'close'
        prices = [d.get('Close') if 'Close' in d else d.get('close') for d in response_data]
        chart_image_base64 = generate_stock_chart([d['date'] for d in response_data], prices, stock_id)
        save_data_to_datasets(stock_id, response_data, 'finmind_us_stock', start_date, end_date)

        return jsonify({'data': response_data, 'chart_image': chart_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/exchange_rate')
def get_finmind_exchange_rate():
    """
    Get exchange rates against TWD (19 currencies).

    Returns exchange rates for 19 different currencies against TWD.

    Returns:
        JSON: Exchange rate data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        cached = load_data_from_datasets('exchange_rate', 'finmind_exchange_rate', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.exchange_rate(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('exchange_rate', response_data, 'finmind_exchange_rate', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/central_bank_rate')
def get_finmind_central_bank_rate():
    """
    Get central bank interest rates (12 countries).

    Returns interest rates for central banks of 12 different countries.

    Returns:
        JSON: Central bank rate data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        cached = load_data_from_datasets('central_bank_rate', 'finmind_central_bank_rate')
        if cached is not None:
            return jsonify(cached)

        df = api.central_bank_interest_rate()
        response_data = df.to_dict(orient='records')
        save_data_to_datasets('central_bank_rate', response_data, 'finmind_central_bank_rate')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/us_treasury_yield')
def get_finmind_us_treasury_yield():
    """
    Get US Treasury yields (1-30 year terms, 12 bonds).

    Returns yields for US Treasury bonds of various maturities.

    Returns:
        JSON: US Treasury yield data

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        cached = load_data_from_datasets('us_treasury', 'finmind_us_treasury_yield', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.us_treasury_yield(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_datasets('us_treasury', response_data, 'finmind_us_treasury_yield', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# EXISTING ROUTES (Updated)
# =============================================================================

@bp.route('/api/finmind/<stock_id>')
def get_finmind_data(stock_id):
    try:
        # Get date parameters from query string, with defaults
        start_date = request.args.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        # Try to load from cache first (exact start/end match)
        cached = load_data_from_datasets(stock_id, 'finmind_taiwan_stock_price', start_date, end_date)
        if cached is not None:
            df = pd.DataFrame(cached) if isinstance(cached, list) else pd.DataFrame(cached)
            chart_image_base64 = generate_kline_chart(df, stock_id, f"{stock_id} K線圖")
            return jsonify({'data': cached, 'chart_image': chart_image_base64})

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        # Use DataLoader as recommended in documentation
        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_daily(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')

        # Generate K-line chart
        chart_image_base64 = generate_kline_chart(df, stock_id, f"{stock_id} K線圖")

        # Save to datasets
        save_data_to_datasets(stock_id, response_data, 'finmind_taiwan_stock_price', start_date, end_date)

        return jsonify({'data': response_data, 'chart_image': chart_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/per/<stock_id>')
def get_finmind_per_data(stock_id):
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)
        
        # Get PER/PBR data for last 1 year
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        cached = load_data_from_datasets(stock_id, 'finmind_per_pbr', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_per_pbr(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_per_pbr', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/institutional/<stock_id>')
def get_finmind_institutional_data(stock_id):
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)
        
        # Get institutional investor data for last 3 months
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        cached = load_data_from_datasets(stock_id, 'finmind_institutional', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_institutional', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/margin/<stock_id>')
def get_finmind_margin_data(stock_id):
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)
        
        # Get margin purchase/short sale data for last 3 months
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        cached = load_data_from_datasets(stock_id, 'finmind_margin', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_margin', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/revenue/<stock_id>')
def get_finmind_revenue_data(stock_id):
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)
        
        # Get monthly revenue data for last 2 years
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        cached = load_data_from_datasets(stock_id, 'finmind_revenue', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_month_revenue(
            stock_id=stock_id,
            start_date=start_date
        )
        
        response_data = df.to_dict(orient='records')
        save_data_to_datasets(stock_id, response_data, 'finmind_revenue', start_date, end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/weekly/<stock_id>')
def get_finmind_weekly_data(stock_id):
    """
    Generate weekly stock data by aggregating daily data.

    This endpoint fetches daily data and aggregates it into weekly OHLC data.
    Available for free membership as it uses daily data aggregation.

    URL Parameters:
        stock_id (str): Taiwan stock symbol (e.g., "2330" for TSMC). Required.

    Query Parameters:
        start_date (str): Start date in YYYY-MM-DD format (e.g., "2023-01-01"). Optional; defaults to 1 year ago.
        end_date (str): End date in YYYY-MM-DD format (e.g., "2023-12-31"). Optional; defaults to today.

    Returns:
        JSON: Dictionary containing 'data' (list of weekly price records) and 'chart_image' (base64 encoded K-line chart image).
            Each record contains date, open, high, low, close, volume, etc.

    Status Codes:
        200: Data retrieved and aggregated successfully.
        500: API key not set or data aggregation error.

    Authentication:
        Required: FinMind API Key set as environment variable 'FINMIND_API_KEY'.

    Note:
        This is a free-tier compatible implementation that aggregates daily data.
        Weekly data is calculated as: open (first day's open), high (week's max), low (week's min), close (last day's close), volume (week's sum).
    """
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        # Get date parameters from query string, with defaults (longer period for weekly aggregation)
        start_date = request.args.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        # Fetch daily data using existing function
        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_daily(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            return jsonify({'error': 'No daily data available for aggregation'}), 500

        # Convert to pandas DataFrame and prepare for aggregation
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Resample to weekly data (W-FRI means week ending Friday)
        weekly_df = df.resample('W-FRI').agg({
            'open': 'first',      # First trading day's open
            'max': 'max',         # Highest high of the week
            'min': 'min',         # Lowest low of the week
            'close': 'last',      # Last trading day's close
            'Trading_Volume': 'sum',  # Sum of volumes
            'Trading_money': 'sum',   # Sum of trading money
            'Trading_turnover': 'sum',  # Sum of turnover
            'spread': 'last',      # Last spread
        }).dropna()

        # Reset index and format date
        weekly_df = weekly_df.reset_index()
        weekly_df['date'] = weekly_df['date'].dt.strftime('%Y-%m-%d')

        # Rename columns to match expected format
        column_mapping = {
            'Trading_Volume': 'Trading_Volume',
            'Trading_money': 'Trading_money',
            'Trading_turnover': 'Trading_turnover'
        }
        weekly_df = weekly_df.rename(columns=column_mapping)

        # Try cache first
        cached = load_data_from_datasets(stock_id, 'finmind_weekly', start_date, end_date)
        if cached is not None:
            df_cached = pd.DataFrame(cached)
            chart_image_base64 = generate_kline_chart(df_cached, stock_id, f"{stock_id} 週線K線圖")
            return jsonify({'data': cached, 'chart_image': chart_image_base64})

        response_data = weekly_df.to_dict(orient='records')

        # Generate K-line chart
        chart_image_base64 = generate_kline_chart(weekly_df, stock_id, f"{stock_id} 週線K線圖")

        # Save to datasets
        save_data_to_datasets(stock_id, response_data, 'finmind_weekly', start_date, end_date)

        return jsonify({'data': response_data, 'chart_image': chart_image_base64})
    except Exception as e:
        return jsonify({'error': f'Weekly data aggregation failed: {str(e)}'}), 500

@bp.route('/api/finmind/monthly/<stock_id>')
def get_finmind_monthly_data(stock_id):
    """
    Generate monthly stock data by aggregating daily data.

    This endpoint fetches daily data and aggregates it into monthly OHLC data.
    Available for free membership as it uses daily data aggregation.

    URL Parameters:
        stock_id (str): Taiwan stock symbol (e.g., "2330" for TSMC). Required.

    Query Parameters:
        start_date (str): Start date in YYYY-MM-DD format (e.g., "2023-01-01"). Optional; defaults to 2 years ago.
        end_date (str): End date in YYYY-MM-DD format (e.g., "2023-12-31"). Optional; defaults to today.

    Returns:
        JSON: Dictionary containing 'data' (list of monthly price records) and 'chart_image' (base64 encoded K-line chart image).
            Each record contains date, open, high, low, close, volume, etc.

    Status Codes:
        200: Data retrieved and aggregated successfully.
        500: API key not set or data aggregation error.

    Authentication:
        Required: FinMind API Key set as environment variable 'FINMIND_API_KEY'.

    Note:
        This is a free-tier compatible implementation that aggregates daily data.
        Monthly data is calculated as: open (first day's open), high (month's max), low (month's min), close (last day's close), volume (month's sum).
    """
    try:
        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        # Get date parameters from query string, with defaults
        start_date = request.args.get('start_date', (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        # Fetch daily data using existing function
        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_daily(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            return jsonify({'error': 'No daily data available for aggregation'}), 500

        # Convert to pandas DataFrame and prepare for aggregation
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Resample to monthly data (ME means month end)
        monthly_df = df.resample('ME').agg({
            'open': 'first',      # First trading day's open
            'max': 'max',         # Highest high of the month
            'min': 'min',         # Lowest low of the month
            'close': 'last',      # Last trading day's close
            'Trading_Volume': 'sum',  # Sum of volumes
            'Trading_money': 'sum',   # Sum of trading money
            'Trading_turnover': 'sum',  # Sum of turnover
            'spread': 'last',      # Last spread
        }).dropna()

        # Reset index and format date
        monthly_df = monthly_df.reset_index()
        monthly_df['date'] = monthly_df['date'].dt.strftime('%Y-%m-%d')

        # Rename columns to match expected format
        column_mapping = {
            'Trading_Volume': 'Trading_Volume',
            'Trading_money': 'Trading_money',
            'Trading_turnover': 'Trading_turnover'
        }
        monthly_df = monthly_df.rename(columns=column_mapping)

        cached = load_data_from_datasets(stock_id, 'finmind_monthly', start_date, end_date)
        if cached is not None:
            df_cached = pd.DataFrame(cached)
            chart_image_base64 = generate_kline_chart(df_cached, stock_id, f"{stock_id} 月線K線圖")
            return jsonify({'data': cached, 'chart_image': chart_image_base64})

        response_data = monthly_df.to_dict(orient='records')

        # Generate K-line chart
        chart_image_base64 = generate_kline_chart(monthly_df, stock_id, f"{stock_id} 月線K線圖")

        # Save to datasets
        save_data_to_datasets(stock_id, response_data, 'finmind_monthly', start_date, end_date)

        return jsonify({'data': response_data, 'chart_image': chart_image_base64})
    except Exception as e:
        return jsonify({'error': f'Monthly data aggregation failed: {str(e)}'}), 500

@bp.route('/api/finmind/translation/<dataset>')
def get_translation(dataset):
    """
    Get field name translations for a dataset.

    Returns Chinese translations for English field names.

    Args:
        dataset (str): Dataset name

    Returns:
        JSON: Dictionary mapping English field names to Chinese

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    # Try cache first (allow returning cached translations without API key)
    cached = load_data_from_datasets(dataset, 'finmind_translation')
    if cached is not None:
        return jsonify(cached)

    token = os.getenv('FINMIND_API_KEY')
    if not token:
        return jsonify({'error': 'FinMind API 金鑰未設定'}), 500
    
    url = "https://api.finmindtrade.com/api/v4/translation"
    params = {
        'dataset': dataset,
        'token': token
    }
    
    try:

        resp = requests.get(url, params=params)
        data = resp.json()
        # Check if the response contains translation data
        if 'data' in data and isinstance(data['data'], dict):
            save_data_to_datasets(dataset, data['data'], 'finmind_translation')
            return jsonify(data['data'])
        elif 'detail' in data:
            # API validation error - dataset not supported
            return jsonify({}), 200  # Return empty dict, frontend will use fallbacks
        else:
            # Try to return data directly if it's a dict
            if isinstance(data, dict):
                save_data_to_datasets(dataset, data, 'finmind_translation')
                return jsonify(data)
            else:
                return jsonify({}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/user_info')
def get_user_info():
    """Get FinMind API usage info (user_count and api_request_limit).

    This endpoint returns real-time usage info from the FinMind web API.
    It does NOT use cached data and does NOT save responses to the datasets
    directory.
    """
    token = os.getenv('FINMIND_API_KEY')
    if not token:
        return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

    url = "https://api.web.finmindtrade.com/v2/user_info"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        user_count = data.get('user_count') if isinstance(data, dict) else None
        api_request_limit = data.get('api_request_limit') if isinstance(data, dict) else None

        return jsonify({'user_count': user_count, 'api_request_limit': api_request_limit})
    except requests.HTTPError as he:
        return jsonify({'error': f'HTTP error: {str(he)}'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/finmind_dashboard', methods=['GET', 'POST'])
def finmind_dashboard() -> str:
    """
    Render the FinMind dashboard with stock data and charts.

    Handles POST requests to fetch and display stock data, institutional data, and margin data.
    Generates a Plotly K-line chart for visualization.

    Returns:
        str: Rendered HTML template for the dashboard.
    """
    if request.method == 'POST':
        stock_id: str = request.form.get('stock_id')
        start_date: str = request.form.get('start_date')
        end_date: str = request.form.get('end_date')
        
        # Fetch price data
        price_data = load_data_from_datasets(stock_id, 'finmind_taiwan_stock_price', start_date, end_date)
        if price_data is None:
            api_key = os.getenv('FINMIND_API_KEY')
            if api_key:
                api = DataLoader()
                api.login_by_token(api_token=api_key)
                df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
                # Rename columns to match chart expectations
                df.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'}, inplace=True)
                price_data = df.to_dict(orient='records')
                save_data_to_datasets(stock_id, price_data, 'finmind_taiwan_stock_price', start_date, end_date)
        else:
            # If loaded from cache, ensure columns are renamed
            df = pd.DataFrame(price_data)
            df.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'}, inplace=True)
            price_data = df.to_dict(orient='records')
        
        # Fetch institutional data
        institutional_data = load_data_from_datasets(stock_id, 'finmind_institutional', start_date, end_date)
        if institutional_data is None:
            if api_key:
                df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date, end_date=end_date)
                institutional_data = df.to_dict(orient='records')
                save_data_to_datasets(stock_id, institutional_data, 'finmind_institutional', start_date, end_date)
        
        # Fetch margin data
        margin_data = load_data_from_datasets(stock_id, 'finmind_margin', start_date, end_date)
        if margin_data is None:
            if api_key:
                df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_date, end_date=end_date)
                margin_data = df.to_dict(orient='records')
                save_data_to_datasets(stock_id, margin_data, 'finmind_margin', start_date, end_date)
        
        # Generate chart
        chart_html = generate_plotly_kline_chart(price_data, institutional_data, margin_data, stock_id)
        
        return render_template('finmind_dashboard.html', chart_html=chart_html)
    
    return render_template('finmind_dashboard.html')