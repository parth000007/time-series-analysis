#!/usr/bin/env python3
"""
Time Series Analysis and Forecasting Tool
Main entry point for the time series analysis project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    def __init__(self, data_path=None):
        """Initialize the Time Series Analyzer"""
        self.data = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.forecast = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load time series data from CSV file"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self, date_col='date', target_col='value'):
        """Preprocess the time series data"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False
        
        # Convert date column to datetime
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data = self.data.sort_values(date_col)
        self.data.set_index(date_col, inplace=True)
        
        # Handle missing values
        self.data[target_col] = self.data[target_col].interpolate(method='linear')
        
        print("Data preprocessing completed")
        return True
    
    def visualize_time_series(self, target_col='value'):
        """Create comprehensive time series visualizations"""
        if self.data is None:
            print("No data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original time series
        axes[0, 0].plot(self.data[target_col])
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value')
        
        # Seasonal decomposition
        decomposition = seasonal_decompose(self.data[target_col], model='additive', period=12)
        axes[0, 1].plot(decomposition.trend)
        axes[0, 1].set_title('Trend Component')
        
        axes[1, 0].plot(decomposition.seasonal)
        axes[1, 0].set_title('Seasonal Component')
        
        axes[1, 1].plot(decomposition.resid)
        axes[1, 1].set_title('Residual Component')
        
        plt.tight_layout()
        plt.savefig('time_series_analysis.png')
        plt.show()
    
    def train_test_split(self, test_size=0.2):
        """Split data into training and testing sets"""
        if self.data is None:
            print("No data available for splitting")
            return False
        
        split_point = int(len(self.data) * (1 - test_size))
        self.train_data = self.data[:split_point]
        self.test_data = self.data[split_point:]
        
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        return True
    
    def arima_forecast(self, target_col='value', order=(1, 1, 1)):
        """Perform ARIMA forecasting"""
        if self.train_data is None or self.test_data is None:
            print("Please split data first")
            return None
        
        try:
            model = ARIMA(self.train_data[target_col], order=order)
            model_fit = model.fit()
            
            # Forecast
            forecast_steps = len(self.test_data)
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Calculate metrics
            mae = mean_absolute_error(self.test_data[target_col], forecast)
            rmse = np.sqrt(mean_squared_error(self.test_data[target_col], forecast))
            
            print(f"ARIMA Model Results:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            
            return forecast
        except Exception as e:
            print(f"Error in ARIMA forecasting: {e}")
            return None
    
    def prophet_forecast(self, target_col='value'):
        """Perform Prophet forecasting"""
        if self.train_data is None or self.test_data is None:
            print("Please split data first")
            return None
        
        try:
            # Prepare data for Prophet
            train_prophet = self.train_data.reset_index()
            train_prophet = train_prophet.rename(columns={train_prophet.columns[0]: 'ds', target_col: 'y'})
            
            # Initialize and fit Prophet model
            model = Prophet()
            model.fit(train_prophet)
            
            # Create future dataframe
            future_steps = len(self.test_data)
            future = model.make_future_dataframe(periods=future_steps)
            
            # Forecast
            forecast = model.predict(future)
            
            # Calculate metrics
            test_forecast = forecast['yhat'][-len(self.test_data):]
            mae = mean_absolute_error(self.test_data[target_col], test_forecast)
            rmse = np.sqrt(mean_squared_error(self.test_data[target_col], test_forecast))
            
            print(f"Prophet Model Results:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            
            return forecast
        except Exception as e:
            print(f"Error in Prophet forecasting: {e}")
            return None
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            'data_shape': self.data.shape if self.data is not None else None,
            'train_shape': self.train_data.shape if self.train_data is not None else None,
            'test_shape': self.test_data.shape if self.test_data is not None else None,
            'columns': list(self.data.columns) if self.data is not None else None,
            'data_types': self.data.dtypes.to_dict() if self.data is not None else None,
            'missing_values': self.data.isnull().sum().to_dict() if self.data is not None else None
        }
        
        return report

def main():
    """Main function to run the time series analysis"""
    print("Time Series Analysis Tool")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer()
    
    # Example usage
    print("To use this tool:")
    print("1. Create a CSV file with date and value columns")
    print("2. Use: analyzer.load_data('your_data.csv')")
    print("3. Use: analyzer.preprocess_data()")
    print("4. Use: analyzer.visualize_time_series()")
    print("5. Use: analyzer.train_test_split()")
    print("6. Use: analyzer.arima_forecast() or analyzer.prophet_forecast()")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
