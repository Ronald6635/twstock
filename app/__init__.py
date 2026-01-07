from flask import Flask, render_template
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env for API keys

app = Flask(__name__, static_folder='../static', template_folder='../templates')

# Register blueprints
from . import stock, realtime, analytics, codes, finmind
app.register_blueprint(stock.bp)
app.register_blueprint(realtime.bp)
app.register_blueprint(analytics.bp)
app.register_blueprint(codes.bp)
app.register_blueprint(finmind.bp)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/finmind_api')
def finmind_api():
    return render_template('finmind_api.html')

if __name__ == '__main__':
    app.run(debug=True)