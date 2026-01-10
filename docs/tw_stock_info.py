"""
Taiwan Stock Information Module

This module provides functionality to retrieve Taiwan stock overview information
using the FinMind API. It includes caching mechanisms to optimize performance
and reduce API calls.
"""

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.append('..')

from flask import Flask, jsonify
from FinMind.data import DataLoader

from app.utils import load_data_from_cache, save_data_to_cache


def get_finmind_stock_info() -> Any:
    """
    Get Taiwan stock overview information from FinMind API.

    Retrieves basic information for all listed Taiwan stocks including
    industry category, stock ID, stock name, type, and date.
    Uses caching to avoid repeated API calls for the same data.

    Returns:
        Flask response: JSON list of stock information dictionaries

    Raises:
        Exception: If API call fails or data processing error occurs

    Status Codes:
        200: Success - cached or fresh data returned
        500: API key not set or API error

    Example:
        Response format:
        [
            {
                "industry_category": "ETF",
                "stock_id": "0050",
                "stock_name": "元大台灣50",
                "type": "twse",
                "date": "2021-10-05"
            },
            ...
        ]

    Note:
        Data is cached to reduce API usage. Cache is checked before
        making API calls. API key must be set in environment variable
        FINMIND_API_KEY.
    """
    try:
        # Return cached data if available before requiring API key
        cached: Optional[List[Dict[str, Any]]] = load_data_from_cache('all', 'finmind_stock_info')
        if cached is not None:
            return jsonify(cached)

        api_key: Optional[str] = os.getenv('FINMIND_API_KEY')
        if not api_key:
            return jsonify({'error': 'FinMind API 金鑰未設定'}), 500

        api = DataLoader()
        api.login_by_token(api_token=api_key)

        df = api.taiwan_stock_info()
        response_data: List[Dict[str, Any]] = df.to_dict(orient='records')
        save_data_to_cache('all', response_data, 'finmind_stock_info')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add main block for testing
if __name__ == "__main__":
    import json
    # Create a minimal Flask app context for jsonify to work
    app = Flask(__name__)
    with app.app_context():
        response = get_finmind_stock_info()
        # Extract JSON data from the response and save to file
        if hasattr(response, 'get_json'):
            data = response.get_json()
            with open('tw_stock_info.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print("Output saved to tw_stock_info.json")
        else:
            print("Response is not JSON-serializable in this context.")