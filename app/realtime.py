from flask import Blueprint, jsonify
import twstock
from .utils import save_data_to_datasets

bp = Blueprint('realtime', __name__)

@bp.route('/api/realtime/<stock_id>')
def get_realtime_data(stock_id):
    try:
        realtime = twstock.realtime.get(stock_id)
        if not realtime.get('success', False):
            return jsonify({'error': f'Failed to fetch realtime data for {stock_id}'}), 500
        
        response_data = realtime

        # Save to datasets
        save_data_to_datasets(stock_id, response_data, 'realtime')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500