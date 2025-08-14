#!/usr/bin/env python3
"""
Web-based Frontend for Time Series Analysis Tool
This provides a web interface for the time series analysis functionality
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import base64
import os
from main import TimeSeriesAnalyzer

app = Flask(__name__)

# Global analyzer instance
analyzer = TimeSeriesAnalyzer()

@app.route('/')
def index():
    """Main dashboard page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Time Series Analysis Tool</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #007bff; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .section { background: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; }
            .btn:hover { background: #0056b3; }
            .info { background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Time Series Analysis & Forecasting Tool</h1>
                <p>Web-based interface for comprehensive time series analysis</p>
            </div>
            
            <div class="section">
                <h2>Getting Started</h2>
                <div class="info">
                    <h3>Quick Start Guide:</h3>
                    <ol>
                        <li>Prepare your CSV file with date and value columns</li>
                        <li>Use the command line tool: python main.py</li>
                        <li>Follow the interactive prompts</li>
                        <li>Generate forecasts using ARIMA or Prophet models</li>
                    </ol>
                </div>
            </div>
            
            <div class="grid">
                <div class="section">
                    <h3>Features</h3>
                    <ul>
                        <li>Time series data loading & preprocessing</li>
                        <li>Exploratory data analysis</li>
                        <li>Seasonal decomposition</li>
                        <li>Stationarity testing</li>
                        <li>ARIMA forecasting</li>
                        <li>Prophet forecasting</li>
                        <li>Model evaluation</li>
                        <li>Visualization & reporting</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h3>Usage Instructions</h3>
                    <div class="info">
                        <h4>Command Line Usage:</h4>
                        <p>1. Run: <code>python main.py</code></p>
                        <p>2. Follow the interactive prompts</p>
                        <p>3. Upload your CSV file</p>
                        <p>4. Select analysis options</p>
                        <p>5. Generate forecasts</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>Sample Data</h3>
                <p>Sample data is available in <code>data/sample_data.csv</code> for testing purposes.</p>
                <a href="/static/sample_data.csv" class="btn" download>Download Sample Data</a>
            </div>
            
            <div class="section">
                <h3>API Endpoints</h3>
                <div class="info">
                    <p><strong>GET /</strong> - Main dashboard</p>
                    <p><strong>GET /static/sample_data.csv</strong> - Download sample data</p>
                    <p><strong>POST /analyze</strong> - Perform analysis</p>
                    <p><strong>POST /visualize</strong> - Generate visualizations</p>
                    <p><strong>POST /forecast</strong> - Generate forecasts</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis"""
    try:
        data = request.json
        # Process analysis here
        return jsonify({'status': 'success', 'message': 'Analysis completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
