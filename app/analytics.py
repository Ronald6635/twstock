from flask import Blueprint, jsonify
import twstock
from .utils import save_data_to_cache

bp = Blueprint('analytics', __name__)

@bp.route('/api/analytics/<stock_id>')
def get_analytics_data(stock_id):
    try:
        stock = twstock.Stock(stock_id)
        stock.fetch_31()

        bfp = twstock.BestFourPoint(stock)
        best_four_point = bfp.best_four_point()

        response_data = {
            'stock_id': stock_id,
            'best_four_point': best_four_point[1] if best_four_point else None,
            'recommendation': best_four_point
        }

        # Save to cache
        save_data_to_cache(stock_id, response_data, 'analytics')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
