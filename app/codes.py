from flask import Blueprint, jsonify
import twstock
from .utils import save_data_to_datasets

bp = Blueprint('codes', __name__)

@bp.route('/api/codes')
def get_codes_data():
    try:
        codes = twstock.codes
        response_data = {k: {'name': v.name, 'type': v.type, 'isin': v.isin} for k, v in codes.items()}

        # Save to datasets (generic, since no stock_id)
        # For codes, save as general data
        save_data_to_datasets('all', response_data, 'codes')

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500