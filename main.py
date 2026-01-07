"""
Main entry point for the twstock Flask application.

This script configures Matplotlib for non-GUI use and runs the Flask app in debug mode.
"""
import matplotlib
from typing import NoReturn  # For type hints

matplotlib.use('Agg')  # Set backend to non-GUI to avoid Tkinter issues

from app import app

def main() -> NoReturn:
    """
    Runs the Flask application in debug mode.
    
    This function serves as the entry point, ensuring the app starts with debugging enabled.
    """
    app.run(debug=True, port=8888)

if __name__ == '__main__':
    main()
