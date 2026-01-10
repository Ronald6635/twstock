"""
FinMind API Integration Module

This module provides Flask routes for accessing FinMind's Taiwan stock market data.
It supports all Free-tier datasets including technical, fundamental, and other market data.

Key features:
- Comprehensive Taiwan stock data access
- Automatic data caching to cache
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
from .utils import save_data_to_cache, generate_stock_chart, generate_kline_chart, load_data_from_cache, generate_plotly_kline_chart, write_combined_files
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
        cached = load_data_from_cache('all', 'finmind_stock_info')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_info()
        response_data = df.to_dict(orient='records')
        save_data_to_cache('all', response_data, 'finmind_stock_info')

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
        cached = load_data_from_cache('trading_days', 'finmind_trading_days')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_trading_days()
        response_data = df.to_dict(orient='records')
        save_data_to_cache('trading_days', response_data, 'finmind_trading_days')

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
        cached = load_data_from_cache(sector, 'finmind_sector_price', start_date, end_date)
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
        save_data_to_cache(sector, response_data, 'finmind_sector_price', start_date, end_date)

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

        cached = load_data_from_cache('TAIEX', 'finmind_index', start_date, end_date)
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
        save_data_to_cache('TAIEX', response_data, 'finmind_index', start_date, end_date)

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
        cached = load_data_from_cache('index_return', 'finmind_index_return', start_date, end_date)
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
        save_data_to_cache('index_return', response_data, 'finmind_index_return', start_date, end_date)

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
        cached = load_data_from_cache('total', 'finmind_margin_total', start_date, end_date)
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
        save_data_to_cache('total', response_data, 'finmind_margin_total', start_date, end_date)

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
        cached = load_data_from_cache('total', 'finmind_institutional_total', start_date, end_date)
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
        save_data_to_cache('total', response_data, 'finmind_institutional_total', start_date, end_date)

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

        cached = load_data_from_cache(stock_id, 'finmind_holders')
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_holders(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_holders')

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
        cached = load_data_from_cache(stock_id, 'finmind_securities_lending', start_date, end_date)
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
        save_data_to_cache(stock_id, response_data, 'finmind_securities_lending', start_date, end_date)

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
        cached = load_data_from_cache(stock_id, 'finmind_cash_flow')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_cash_flows_statement(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_cash_flow')

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
        cached = load_data_from_cache(stock_id, 'finmind_income_statement')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_income_statement(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_income_statement')

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
        cached = load_data_from_cache(stock_id, 'finmind_balance_sheet')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_balance_sheet(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_balance_sheet')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/finmind/financial_statement/<stock_id>')
def get_finmind_financial_statement(stock_id: str):
    """
    Get consolidated financial statements (income statement, balance sheet, cash flow).

    URL Parameters:
        stock_id (str): Stock identifier

    Query Parameters:
        start_date (str): Optional YYYY-MM-DD to limit results (default: '2019-01-01')
        end_date (str): Optional YYYY-MM-DD to limit results (optional, used for caching filename accuracy)

    Returns:
        JSON: Financial statement records

    Status Codes:
        200: Success
        500: API key not set or API error
    """
    try:
        # Accept optional date params (default start_date per FinMind docs)
        start_date = request.args.get('start_date', '2019-01-01')
        end_date = request.args.get('end_date')  # optional, used for cache filename matching

        # Try cache first (use both start/end when available)
        cached = load_data_from_cache(stock_id, 'finmind_financial_statement', start_date, end_date) if end_date else load_data_from_cache(stock_id, 'finmind_financial_statement', start_date)
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        # Call the FinMind financial statement API
        df = api.taiwan_stock_financial_statement(
            stock_id=stock_id,
            start_date=start_date
        )

        response_data = df.to_dict(orient='records')
        # Determine end_date for filename prefix: prefer provided end_date, else infer from returned rows, else today
        if not end_date:
            inferred_end = None
            if 'date' in df.columns and len(df) > 0:
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    inferred_end = df['date'].max().strftime('%Y-%m-%d')
                except Exception:
                    inferred_end = None
            end_date = inferred_end or datetime.now().strftime('%Y-%m-%d')

        save_data_to_cache(stock_id, response_data, 'finmind_financial_statement', start_date=start_date, end_date=end_date)

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _process_financial_data(financial_data):
    """Helper to extract EPS, Gross Profit and series from financial JSON data.

    Supports both "wide" format (one record per date with many columns) and
    "long" format where each record has 'type' and 'value' (as produced in
    FinMind CSVs/JSONs). When long-form is detected, pivot into per-date
    records before extracting EPS and Gross Profit.
    """
    latest_eps = None
    latest_gross = None
    financial_series = []
    if not financial_data:
        return latest_eps, latest_gross, financial_series

    try:
        fdf = pd.DataFrame(financial_data)

        # If data is long form (type/value pairs), pivot to wide form by date
        if 'type' in fdf.columns and 'value' in fdf.columns:
            # Prefer a proper 'date' field if present
            if 'date' in fdf.columns:
                fdf['date'] = pd.to_datetime(fdf['date'], errors='coerce')
                # pivot: index date, columns type, values value
                try:
                    pivot = fdf.pivot_table(index='date', columns='type', values='value', aggfunc='first')
                    pivot = pivot.reset_index()
                except Exception:
                    # fallback: groupby and take first
                    grouped = fdf.groupby(['date', 'type'])['value'].first().unstack(fill_value=None).reset_index()
                    pivot = grouped
                fdf = pivot
            else:
                # No date column, try year/season combination
                if 'year' in fdf.columns and 'season' in fdf.columns:
                    fdf['season'] = fdf['season'].astype(str)
                    # create a synthetic date-like index 'YYYY-SS'
                    fdf['period'] = fdf['year'].astype(str) + '-' + fdf['season'].str.zfill(2)
                    try:
                        pivot = fdf.pivot_table(index='period', columns='type', values='value', aggfunc='first').reset_index()
                        fdf = pivot
                    except Exception:
                        pass

        # Normalize column names to lower for flexible lookup
        lower_cols = {c.lower(): c for c in fdf.columns}

        # Discover candidate columns for EPS and Gross Profit
        eps_candidates = [k for k in lower_cols.keys() if 'eps' in k]
        gross_candidates = [k for k in lower_cols.keys() if 'gross' in k or 'costofgoods' in k or 'cost_of_goods' in k]

        # Choose preferred columns
        eps_col = lower_cols.get(eps_candidates[0]) if eps_candidates else None
        gross_col = None
        # Prefer explicit gross_profit-like columns; otherwise compute from revenue - cogs if available
        for g in ['gross_profit', 'grossprofit', 'gross']:
            if g in lower_cols:
                gross_col = lower_cols[g]
                break
        if not gross_col:
            # try candidates list
            gross_col = lower_cols.get(gross_candidates[0]) if gross_candidates else None

        # Ensure we have a date-like index for ordering
        if 'date' in fdf.columns:
            try:
                fdf['date'] = pd.to_datetime(fdf['date'], errors='coerce')
                fdf = fdf.sort_values('date')
            except Exception:
                pass
        elif 'period' in fdf.columns:
            # keep as-is
            pass
        else:
            fdf = fdf.sort_index()

        # Prepare series for template (last up to 12 records, sorted descending)
        rows = fdf.sort_values('date', ascending=False).head(12) if 'date' in fdf.columns else fdf.tail(8)
        for _, row in rows.iterrows():
            rec = {'date': None, 'eps': None, 'gross_profit': None}

            # date formatting
            if 'date' in fdf.columns:
                val = row.get('date')
                rec['date'] = val.strftime('%Y-%m-%d') if hasattr(val, 'strftime') else str(val)
            elif 'period' in fdf.columns:
                rec['date'] = str(row.get('period'))
            else:
                # fallback to year/season
                if 'year' in row and 'season' in row:
                    rec['date'] = f"{int(row.get('year'))}-{str(int(row.get('season'))).zfill(2)}"
                else:
                    rec['date'] = ''

            # eps value
            if eps_col and eps_col in row:
                try:
                    val = row.get(eps_col)
                    rec['eps'] = float(val) if pd.notnull(val) else None
                except Exception:
                    rec['eps'] = None

            # gross profit value
            if gross_col and gross_col in row:
                try:
                    val = row.get(gross_col)
                    rec['gross_profit'] = float(val) if pd.notnull(val) else None
                except Exception:
                    rec['gross_profit'] = None
            else:
                # try computing from Revenue - CostOfGoodsSold if present
                rev_col = lower_cols.get('revenue')
                cogs_col = None
                for c in ['costofgoods', 'cost_of_goods', 'costofgoodssold', 'cost_of_goods_sold']:
                    if c in lower_cols:
                        cogs_col = lower_cols[c]
                        break
                if rev_col and cogs_col and rev_col in row and cogs_col in row:
                    try:
                        r = row.get(rev_col)
                        c = row.get(cogs_col)
                        rec['gross_profit'] = float(r) - float(c)
                    except Exception:
                        pass

            # capture latest non-null values
            if latest_eps is None and rec['eps'] is not None:
                latest_eps = rec['eps']
            if latest_gross is None and rec['gross_profit'] is not None:
                latest_gross = rec['gross_profit']

            financial_series.append(rec)

        # Reverse so most recent appears first
        financial_series = list(reversed(financial_series))
    except Exception:
        pass

    return latest_eps, latest_gross, financial_series

@bp.route('/api/finmind/financial_summary/<stock_id>')
def get_finmind_financial_summary(stock_id: str):
    """Async endpoint to get processed EPS/GrossProfit summary."""
    try:
        # Accept optional start/end params
        start_date = request.args.get('start_date', '2019-01-01')
        end_date = request.args.get('end_date')

        # Try cache first (use both start/end when available)
        financial_data = load_data_from_cache(stock_id, 'finmind_financial_statement', start_date, end_date) if end_date else load_data_from_cache(stock_id, 'finmind_financial_statement', start_date)

        if financial_data is None:
            api_key = os.getenv('FINMIND_API_KEY')
            if api_key:
                api = DataLoader()
                api.login_by_token(api_token=api_key)
                df = api.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date)
                financial_data = df.to_dict(orient='records')

                # Determine end_date for filename prefix: prefer provided end_date, else infer from returned rows, else today
                if not end_date:
                    inferred_end = None
                    if 'date' in df.columns and len(df) > 0:
                        try:
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')
                            inferred_end = df['date'].max().strftime('%Y-%m-%d')
                        except Exception:
                            inferred_end = None
                    end_date = inferred_end or datetime.now().strftime('%Y-%m-%d')

                save_data_to_cache(stock_id, financial_data, 'finmind_financial_statement', start_date=start_date, end_date=end_date)
            else:
                return jsonify({'error': 'No data found and no API key'}), 404

        latest_eps, latest_gross, financial_series = _process_financial_data(financial_data)
        return jsonify({
            'latest_eps': latest_eps,
            'latest_gross': latest_gross,
            'financial_series': financial_series
        })
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
        cached = load_data_from_cache(stock_id, 'finmind_dividend')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_dividend(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_dividend')

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
        cached = load_data_from_cache(stock_id, 'finmind_ex_dividend')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_ex_dividend(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_ex_dividend')

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
        cached = load_data_from_cache('delisted', 'finmind_delisted')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_delisted()
        response_data = df.to_dict(orient='records')
        save_data_to_cache('delisted', response_data, 'finmind_delisted')

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
        cached = load_data_from_cache(stock_id, 'finmind_split')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_split(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_split')

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
        cached = load_data_from_cache(stock_id, 'finmind_face_value_change')
        if cached is not None:
            return jsonify(cached)

        api_key = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_face_value_change(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_face_value_change')

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
        cached = load_data_from_cache('futures_options', 'finmind_futures_options_daily', start_date, end_date)
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
        save_data_to_cache('futures_options', response_data, 'finmind_futures_options_daily', start_date, end_date)

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
        cached = load_data_from_cache('futures', 'finmind_futures_daily', start_date, end_date)
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
        save_data_to_cache('futures', response_data, 'finmind_futures_daily', start_date, end_date)

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
        cached = load_data_from_cache('options', 'finmind_options_daily', start_date, end_date)
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
        save_data_to_cache('options', response_data, 'finmind_options_daily', start_date, end_date)

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
        cached = load_data_from_cache('futures_institutional', 'finmind_futures_institutional', start_date, end_date)
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
        save_data_to_cache('futures_institutional', response_data, 'finmind_futures_institutional', start_date, end_date)

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
        cached = load_data_from_cache('options_institutional', 'finmind_options_institutional', start_date, end_date)
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
        save_data_to_cache('options_institutional', response_data, 'finmind_options_institutional', start_date, end_date)

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

        cached = load_data_from_cache(stock_id, 'finmind_news')
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_news(stock_id=stock_id)
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_news')

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
        cached = load_data_from_cache('gold', 'finmind_gold_price', start_date, end_date)
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

        # Save to cache for caching/analysis
        save_data_to_cache('gold', response_data, 'finmind_gold_price', start_date, end_date)

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
        cached = load_data_from_cache(data_id, 'finmind_crude_oil_price', start_date, end_date)
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

        # Save to cache for caching/analysis
        save_data_to_cache(data_id, response_data, 'finmind_crude_oil_price', start_date, end_date)

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
        cached = load_data_from_cache(stock_id, 'finmind_us_stock', start_date, end_date)
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
        save_data_to_cache(stock_id, response_data, 'finmind_us_stock', start_date, end_date)

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

        cached = load_data_from_cache('exchange_rate', 'finmind_exchange_rate', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.exchange_rate(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_cache('exchange_rate', response_data, 'finmind_exchange_rate', start_date, end_date)

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

        cached = load_data_from_cache('central_bank_rate', 'finmind_central_bank_rate')
        if cached is not None:
            return jsonify(cached)

        df = api.central_bank_interest_rate()
        response_data = df.to_dict(orient='records')
        save_data_to_cache('central_bank_rate', response_data, 'finmind_central_bank_rate')

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

        cached = load_data_from_cache('us_treasury', 'finmind_us_treasury_yield', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.us_treasury_yield(
            start_date=start_date,
            end_date=end_date
        )

        response_data = df.to_dict(orient='records')
        save_data_to_cache('us_treasury', response_data, 'finmind_us_treasury_yield', start_date, end_date)

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
        cached = load_data_from_cache(stock_id, 'finmind_taiwan_stock_price', start_date, end_date)
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

        # Save to cache
        save_data_to_cache(stock_id, response_data, 'finmind_taiwan_stock_price', start_date, end_date)

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
        
        cached = load_data_from_cache(stock_id, 'finmind_per_pbr', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_per_pbr(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_per_pbr', start_date, end_date)

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
        
        cached = load_data_from_cache(stock_id, 'finmind_institutional', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_institutional', start_date, end_date)

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
        
        cached = load_data_from_cache(stock_id, 'finmind_margin', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_margin', start_date, end_date)

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
        
        cached = load_data_from_cache(stock_id, 'finmind_revenue', start_date, end_date)
        if cached is not None:
            return jsonify(cached)

        df = api.taiwan_stock_month_revenue(
            stock_id=stock_id,
            start_date=start_date
        )
        
        response_data = df.to_dict(orient='records')
        save_data_to_cache(stock_id, response_data, 'finmind_revenue', start_date, end_date)

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
        cached = load_data_from_cache(stock_id, 'finmind_weekly', start_date, end_date)
        if cached is not None:
            df_cached = pd.DataFrame(cached)
            chart_image_base64 = generate_kline_chart(df_cached, stock_id, f"{stock_id} 週線K線圖")
            return jsonify({'data': cached, 'chart_image': chart_image_base64})

        response_data = weekly_df.to_dict(orient='records')

        # Generate K-line chart
        chart_image_base64 = generate_kline_chart(weekly_df, stock_id, f"{stock_id} 週線K線圖")

        # Save to cache
        save_data_to_cache(stock_id, response_data, 'finmind_weekly', start_date, end_date)

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

        cached = load_data_from_cache(stock_id, 'finmind_monthly', start_date, end_date)
        if cached is not None:
            df_cached = pd.DataFrame(cached)
            chart_image_base64 = generate_kline_chart(df_cached, stock_id, f"{stock_id} 月線K線圖")
            return jsonify({'data': cached, 'chart_image': chart_image_base64})

        response_data = monthly_df.to_dict(orient='records')

        # Generate K-line chart
        chart_image_base64 = generate_kline_chart(monthly_df, stock_id, f"{stock_id} 月線K線圖")

        # Save to cache
        save_data_to_cache(stock_id, response_data, 'finmind_monthly', start_date, end_date)

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
    # Manual mappings for robustness and specific datasets
    manual_maps = {
        'TaiwanStockFinancialStatement': {
            'eps': '每股盈餘 (EPS)',
            'eps_basic': '每股盈餘 (基本)',
            'gross_profit': '營業毛利',
            'revenue': '營業收入',
            'date': '日期',
            'type': '會計項目說明'
        },
        'TaiwanStockMonthRevenue': {
            'revenue': '月營收',
            'revenue_year': '年度',
            'revenue_month': '月份'
        }
    }

    # Try cache first (allow returning cached translations without API key)
    cached = load_data_from_cache(dataset, 'finmind_translation')
    
    token = os.getenv('FINMIND_API_KEY')
    
    result_map = {}
    if cached is not None:
        result_map = cached
    elif token:
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
                save_data_to_cache(dataset, data['data'], 'finmind_translation')
                result_map = data['data']
            elif 'detail' in data:
                result_map = {}
            elif isinstance(data, dict):
                save_data_to_cache(dataset, data, 'finmind_translation')
                result_map = data
        except Exception:
            result_map = {}

    # Merge with manual map if available
    if dataset in manual_maps:
        if not result_map:
            result_map = {}
        result_map.update(manual_maps[dataset])
        
    return jsonify(result_map)

@bp.route('/api/finmind/user_info')
def get_user_info():
    """Get FinMind API usage info (user_count and api_request_limit).

    This endpoint returns real-time usage info from the FinMind web API.
    It does NOT use cached data and does NOT save responses to the cache
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
    import json
    with open('docs/tw_stock_info.json', 'r', encoding='utf-8') as f:
        stock_data = json.load(f)
    
    # Ensure api_key is available in all branches to avoid UnboundLocalError
    api_key = os.getenv('FINMIND_API_KEY')
    api = None
    if api_key:
        try:
            api = DataLoader()
            api.login_by_token(api_token=api_key)
        except Exception:
            api = None

    if request.method == 'POST':
        stock_id: str = request.form.get('stock_id')
        start_date: str = request.form.get('start_date')
        end_date: str = request.form.get('end_date')
        
        # Fetch price data
        price_data = load_data_from_cache(stock_id, 'finmind_taiwan_stock_price', start_date, end_date)
        if price_data is None:
            if api:
                df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
                # Rename columns to match chart expectations
                df.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'}, inplace=True)
                price_data = df.to_dict(orient='records')
                save_data_to_cache(stock_id, price_data, 'finmind_taiwan_stock_price', start_date, end_date)
        else:
            # If loaded from cache, ensure columns are renamed
            df = pd.DataFrame(price_data)
            df.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'}, inplace=True)
            price_data = df.to_dict(orient='records')
        
        # Fetch institutional data
        institutional_data = load_data_from_cache(stock_id, 'finmind_institutional', start_date, end_date)
        if institutional_data is None:
            if api:
                df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date, end_date=end_date)
                institutional_data = df.to_dict(orient='records')
                save_data_to_cache(stock_id, institutional_data, 'finmind_institutional', start_date, end_date)
        
        # Fetch margin data
        margin_data = load_data_from_cache(stock_id, 'finmind_margin', start_date, end_date)
        if margin_data is None:
            if api:
                df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_date, end_date=end_date)
                margin_data = df.to_dict(orient='records')
                save_data_to_cache(stock_id, margin_data, 'finmind_margin', start_date, end_date)

        # Fetch monthly revenue data
        revenue_data = load_data_from_cache(stock_id, 'finmind_revenue', start_date, end_date)
        if revenue_data is None:
            if api:
                df = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
                revenue_data = df.to_dict(orient='records')
                save_data_to_cache(stock_id, revenue_data, 'finmind_revenue', start_date, end_date)

        # Fetch financial statement (EPS, Gross Profit)
        # Use a broader start_date for financials if possible (defaulting to input start_date)
        financial_data = load_data_from_cache(stock_id, 'finmind_financial_statement', start_date)
        if financial_data is None and api:
            try:
                # Per FinMind tutor, financial statements often need a 2019 baseline or similar
                df = api.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date)
                financial_data = df.to_dict(orient='records')
                # infer inferred_end_date from returned rows when possible
                inferred_end_date = None
                if 'date' in df.columns and len(df) > 0:
                    try:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        inferred_end_date = df['date'].max().strftime('%Y-%m-%d')
                    except Exception:
                        inferred_end_date = None
                if not inferred_end_date:
                    inferred_end_date = datetime.now().strftime('%Y-%m-%d')
                # Removed early save of raw quarterly data
            except Exception:
                financial_data = None

        # Extract EPS and Gross Profit using helper
        latest_eps, latest_gross, financial_series = _process_financial_data(financial_data)

        # Build backward-filled daily financial series aligned to trading days (use next announcement's value for prior days)
        financial_daily = []
        try:
            if financial_data and price_data:
                import pandas as _pd
                df_fin = _pd.DataFrame(financial_data)
                if 'type' in df_fin.columns and 'value' in df_fin.columns:
                    df_fin_wide = df_fin.pivot_table(index='date', columns='type', values='value', aggfunc='first').reset_index()
                else:
                    df_fin_wide = df_fin.copy()

                eps_col = None
                gross_col = None
                for c in df_fin_wide.columns:
                    lc = c.lower()
                    if lc == 'eps' or lc.startswith('eps'):
                        eps_col = c
                    if lc in ('grossprofit','gross_profit','gross'):
                        gross_col = c

                df_fin_wide['date'] = _pd.to_datetime(df_fin_wide['date'])
                df_fin_wide = df_fin_wide.set_index('date').sort_index()
                df_price = _pd.DataFrame(price_data)
                df_price['date'] = _pd.to_datetime(df_price['date'])
                idx = _pd.Index(sorted(df_price['date'].unique()))
                fin_daily = df_fin_wide.reindex(idx).bfill()

                for ddate, r in fin_daily.iterrows():
                    financial_daily.append({
                        'date': ddate.strftime('%Y-%m-%d'),
                        'eps': float(r.get(eps_col)) if eps_col and not _pd.isna(r.get(eps_col)) else None,
                        'gross_profit': float(r.get(gross_col)) if gross_col and not _pd.isna(r.get(gross_col)) else None
                    })

                # Save daily sequential financial data to cache
                save_data_to_cache(stock_id, financial_daily, 'finmind_financial_statement', start_date=start_date, end_date=end_date)
        except Exception:
            financial_daily = []

        # Generate chart (include revenue)
        chart_html = generate_plotly_kline_chart(price_data, institutional_data, margin_data, revenue_data, stock_id)

        # Generate separate financial chart (EPS + GrossProfit)
        try:
            from app.utils import generate_financial_chart
            financial_chart_html = generate_financial_chart(financial_daily, stock_id)
        except Exception:
            financial_chart_html = ''

        return render_template('finmind_dashboard.html', chart_html=chart_html, stock_id=stock_id, start_date=start_date, end_date=end_date,
                               financial_series=financial_series, latest_eps=latest_eps, latest_gross=latest_gross, financial_chart_html=financial_chart_html, stock_data=stock_data)
    
    # Default values for GET
    default_end = datetime.now().strftime('%Y-%m-%d')
    return render_template('finmind_dashboard.html', stock_id='2379', start_date='2025-01-01', end_date=default_end, stock_data=stock_data)


@bp.route('/api/finmind/save_dashboard', methods=['POST'])
def save_dashboard():
    """Save combined cached data (JSON + CSV) for the dashboard's selected stock/date-range.

    Payload: { stock_id, start_date, end_date, force_refresh (optional bool) }
    """
    try:
        payload = request.get_json() or {}
        stock_id = payload.get('stock_id')
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        force_refresh = bool(payload.get('force_refresh', False))

        if not (stock_id and start_date and end_date):
            return jsonify({'error': 'stock_id/start_date/end_date required'}), 400

        api_key = os.getenv('FINMIND_API_KEY')
        api = None
        if api_key:
            api = DataLoader()
            api.login_by_token(api_token=api_key)
        else:
            if force_refresh:
                return jsonify({'error': 'FinMind API 金鑰未設定 (force_refresh requested)'}), 500

        # Determine company name
        try:
            import twstock
            company = twstock.codes.get(stock_id).name if stock_id in twstock.codes else stock_id
        except Exception:
            company = stock_id

        api_names = [
            'finmind_taiwan_stock_price',
            'finmind_institutional',
            'finmind_margin',
            'finmind_revenue',
            'finmind_financial_statement'
        ]

        combined = {
            'meta': {
                'stock_id': stock_id,
                'company': company,
                'start_date': start_date,
                'end_date': end_date,
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'saved_by': 'user:web',
                'api_names': api_names
            },
            'cache': {}
        }

        results = {}
        for api_name in api_names:
            data = None
            if not force_refresh:
                data = load_data_from_cache(stock_id, api_name, start_date, end_date)
            source = 'cache' if data is not None else None

            if data is None:
                if not api:
                    results[api_name] = {'ok': False, 'error': 'no cache and no FINMIND API key'}
                    continue
                try:
                    if api_name == 'finmind_taiwan_stock_price':
                        df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
                        df.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'}, inplace=True)
                    elif api_name == 'finmind_institutional':
                        df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date, end_date=end_date)
                    elif api_name == 'finmind_margin':
                        df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_date, end_date=end_date)
                    elif api_name == 'finmind_revenue':
                        df = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
                    elif api_name == 'finmind_financial_statement':
                        # Financial statements are typically reported quarterly; fetch range if provided
                        df = api.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date, end_date=end_date)

                    data = df.to_dict(orient='records')
                    if api_name != 'finmind_financial_statement':
                        save_data_to_cache(stock_id, data, api_name, start_date, end_date)
                    source = 'finmind'
                except Exception as e:
                    results[api_name] = {'ok': False, 'error': str(e)}
                    continue

            combined['cache'][api_name] = {'source': source or 'cache', 'cached': source == 'cache', 'records': data}
            results[api_name] = {'ok': True}

        # Derive monthly aggregates for revenue (cc. generate_plotly_kline_chart)
        try:
            revenue_ds = combined['cache'].get('finmind_revenue', {})
            rev_records = revenue_ds.get('records') or []
            if rev_records:
                df_rev = pd.DataFrame(rev_records)
                if 'revenue_year' in df_rev.columns and 'revenue_month' in df_rev.columns:
                    df_rev['year_month'] = pd.to_datetime(df_rev['revenue_year'].astype(str) + '-' + df_rev['revenue_month'].astype(str) + '-01').dt.to_period('M')
                elif 'date' in df_rev.columns:
                    df_rev['date'] = pd.to_datetime(df_rev['date'])
                    df_rev['year_month'] = df_rev['date'].dt.to_period('M')

                df_price = load_data_from_cache(stock_id, 'finmind_taiwan_stock_price', start_date, end_date) or []
                df_price = pd.DataFrame(df_price)
                if not df_price.empty:
                    df_price['date'] = pd.to_datetime(df_price['date'])
                    df_price['year_month'] = df_price['date'].dt.to_period('M')
                    td = df_price.groupby('year_month').size().reset_index(name='trading_days')
                else:
                    td = pd.DataFrame(columns=['year_month', 'trading_days'])

                if 'revenue' in df_rev.columns:
                    # Group revenue by month (sum in case multiple entries per month)
                    df_rev_grouped = df_rev.groupby('year_month', as_index=False).agg({'revenue': 'sum'})
                    df_rev_grouped = pd.merge(df_rev_grouped, td, on='year_month', how='left')
                    df_rev_grouped['est_flag'] = df_rev_grouped['trading_days'].isna()
                    df_rev_grouped['trading_days'] = df_rev_grouped['trading_days'].fillna(0).astype(int)
                    df_rev_grouped['avg_per_trading_day'] = df_rev_grouped.apply(lambda r: (r['revenue'] / r['trading_days']) if r['trading_days'] and r['trading_days'] > 0 else None, axis=1)

                    monthly = []
                    for _, r in df_rev_grouped.iterrows():
                        monthly.append({
                            'year_month': str(r['year_month']),
                            'revenue': int(r['revenue']),
                            'trading_days': int(r['trading_days']) if not pd.isna(r['trading_days']) else None,
                            'avg_per_trading_day': float(r['avg_per_trading_day']) if not pd.isna(r['avg_per_trading_day']) else None
                        })

                    combined['cache'].setdefault('finmind_revenue', {})['derived'] = {'monthly_aggregates': monthly}
        except Exception:
            pass

        # Build backward-filled daily financial series (EPS: seasonal; GrossProfit: absolute)
        try:
            fin_ds = combined['cache'].get('finmind_financial_statement', {})
            fin_records = fin_ds.get('records') or []
            price_ds = combined['cache'].get('finmind_taiwan_stock_price', {})
            price_records = price_ds.get('records') or []
            if fin_records and price_records:
                df_fin = pd.DataFrame(fin_records)
                # Pivot long-form (type/value) into wide form if needed
                if 'type' in df_fin.columns and 'value' in df_fin.columns:
                    df_fin_wide = df_fin.pivot_table(index='date', columns='type', values='value', aggfunc='first').reset_index()
                else:
                    df_fin_wide = df_fin.copy()

                # Normalize column names and find EPS / GrossProfit columns
                eps_col = None
                gross_col = None
                for c in df_fin_wide.columns:
                    lc = c.lower()
                    if 'eps' == lc or lc.startswith('eps'):
                        eps_col = c
                    if lc in ('grossprofit','gross_profit','gross'):
                        gross_col = c

                # Prepare index aligned to trading dates and backward-fill
                df_fin_wide['date'] = pd.to_datetime(df_fin_wide['date'])
                df_fin_wide = df_fin_wide.set_index('date').sort_index()
                df_price = pd.DataFrame(price_records)
                df_price['date'] = pd.to_datetime(df_price['date'])
                idx = pd.Index(sorted(df_price['date'].unique()))
                fin_daily = df_fin_wide.reindex(idx).bfill()

                daily = []
                for ddate, row in fin_daily.iterrows():
                    rec = {'date': ddate.strftime('%Y-%m-%d')}
                    rec['eps'] = float(row.get(eps_col)) if eps_col and not pd.isna(row.get(eps_col)) else None
                    rec['gross_profit'] = float(row.get(gross_col)) if gross_col and not pd.isna(row.get(gross_col)) else None
                    daily.append(rec)

                combined['cache'].setdefault('finmind_financial_statement', {}).setdefault('derived', {})['daily_financial_series'] = daily
                # Save processed daily financial series to its own dataset cache
                save_data_to_cache(stock_id, daily, 'finmind_financial_statement', start_date, end_date)
        except Exception:
            pass

        # Persist combined files
        try:
            files = write_combined_files(company, stock_id, start_date, end_date, combined)
        except Exception as e:
            return jsonify({'error': 'failed to write combined files', 'details': str(e), 'results': results}), 500

        return jsonify({'saved': True, 'files': [files['json'], files['csv']], 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
