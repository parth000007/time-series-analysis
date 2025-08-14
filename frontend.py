import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, handle if not installed
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📈 Time Series Analysis & Forecasting Tool</h1>', unsafe_allow_html=True)
st.markdown("**Analyze your time series data with advanced statistical models and machine learning techniques**")

class TimeSeriesAnalyzer:
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            else:
                st.error("Please upload a CSV file")
                return False
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self, date_col, target_col):
        """Preprocess the time series data"""
        try:
            # Convert date column to datetime
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data = self.data.sort_values(date_col)
            self.data.set_index(date_col, inplace=True)
            
            # Handle missing values
            if self.data[target_col].isnull().any():
                self.data[target_col] = self.data[target_col].interpolate(method='linear')
                st.warning(f"Missing values detected and filled using linear interpolation")
            
            return True
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return False
    
    def train_test_split(self, test_size=0.2):
        """Split data into training and testing sets"""
        split_point = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_point]
        self.test_data = self.data.iloc[split_point:]
        return self.train_data, self.test_data

def adf_test(series):
    """Perform Augmented Dickey-Fuller test"""
    result = adfuller(series)
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }

def create_sample_data():
    """Create sample time series data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    trend = np.linspace(100, 200, len(dates))
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 10, len(dates))
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })

# Initialize analyzer
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = TimeSeriesAnalyzer()

# Sidebar
st.sidebar.header("📊 Data Input")

# Data input options
data_option = st.sidebar.radio(
    "Choose data source:",
    ["Upload CSV File", "Use Sample Data"]
)

if data_option == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with date and value columns"
    )
    
    if uploaded_file is not None:
        if st.session_state.analyzer.load_data(uploaded_file):
            st.sidebar.success("✅ Data loaded successfully!")
            data = st.session_state.analyzer.data
        else:
            st.stop()
    else:
        st.info("👆 Please upload a CSV file to get started")
        st.stop()

else:
    # Use sample data
    data = create_sample_data()
    data.set_index('date', inplace=True)
    st.session_state.analyzer.data = data
    st.sidebar.success("✅ Sample data loaded!")

# Column selection
if st.session_state.analyzer.data is not None:
    data = st.session_state.analyzer.data
    
    # Reset index to show date column for selection
    data_for_selection = data.reset_index()
    
    st.sidebar.subheader("📋 Column Configuration")
    date_col = st.sidebar.selectbox(
        "Select Date Column",
        options=data_for_selection.columns,
        index=0
    )
    
    target_col = st.sidebar.selectbox(
        "Select Target Column",
        options=[col for col in data_for_selection.columns if col != date_col],
        index=0
    )
    
    # Preprocess data if columns are selected
    if date_col and target_col:
        if st.sidebar.button("🔄 Process Data"):
            data_copy = data_for_selection.copy()
            temp_analyzer = TimeSeriesAnalyzer()
            temp_analyzer.data = data_copy
            
            if temp_analyzer.preprocess_data(date_col, target_col):
                st.session_state.analyzer.data = temp_analyzer.data
                data = st.session_state.analyzer.data
                st.sidebar.success("✅ Data processed successfully!")

# Main content
if st.session_state.analyzer.data is not None and len(st.session_state.analyzer.data.columns) > 0:
    data = st.session_state.analyzer.data
    target_column = data.columns[0]  # Use first column as target
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Analysis", 
        "📈 Decomposition", 
        "🤖 Modeling", 
        "📋 Results"
    ])
    
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Date Range", f"{len(data)} days")
        with col3:
            st.metric("Missing Values", data[target_column].isnull().sum())
        with col4:
            st.metric("Columns", len(data.columns))
        
        # Display data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Time Series Plot")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[target_column],
                mode='lines',
                name=target_column,
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title="Time Series Data",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Statistical Summary")
            st.dataframe(data.describe(), use_container_width=True)
            
            # Show first and last few rows
            st.subheader("📋 Data Preview")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**First 5 rows:**")
                st.dataframe(data.head())
            with col_b:
                st.write("**Last 5 rows:**")
                st.dataframe(data.tail())
    
    with tab2:
        st.header("🔍 Statistical Analysis")
        
        # Stationarity test
        st.subheader("📈 Stationarity Test (ADF)")
        adf_result = adf_test(data[target_column])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ADF Statistic", f"{adf_result['adf_statistic']:.4f}")
            st.metric("P-value", f"{adf_result['p_value']:.4f}")
            
            if adf_result['is_stationary']:
                st.success("✅ Series is stationary")
            else:
                st.warning("⚠️ Series is non-stationary")
        
        with col2:
            st.write("**Critical Values:**")
            for key, value in adf_result['critical_values'].items():
                st.write(f"- {key}: {value:.4f}")
        
        # Distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution Plot")
            fig = px.histogram(
                x=data[target_column],
                nbins=30,
                title="Value Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Box Plot")
            fig = px.box(y=data[target_column], title="Box Plot")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📈 Time Series Decomposition")
        
        # Decomposition parameters
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model Type", ["additive", "multiplicative"])
        with col2:
            period = st.number_input("Period", min_value=2, max_value=365, value=365)
        
        if st.button("🔄 Perform Decomposition"):
            try:
                decomposition = seasonal_decompose(
                    data[target_column], 
                    model=model_type, 
                    period=period
                )
                
                # Create subplots
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                    vertical_spacing=0.08
                )
                
                # Original
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[target_column],
                    mode='lines', name='Original'
                ), row=1, col=1)
                
                # Trend
                fig.add_trace(go.Scatter(
                    x=data.index, y=decomposition.trend,
                    mode='lines', name='Trend'
                ), row=2, col=1)
                
                # Seasonal
                fig.add_trace(go.Scatter(
                    x=data.index, y=decomposition.seasonal,
                    mode='lines', name='Seasonal'
                ), row=3, col=1)
                
                # Residual
                fig.add_trace(go.Scatter(
                    x=data.index, y=decomposition.resid,
                    mode='lines', name='Residual'
                ), row=4, col=1)
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Store decomposition in session state
                st.session_state.decomposition = decomposition
                
            except Exception as e:
                st.error(f"Error in decomposition: {e}")
    
    with tab4:
        st.header("🤖 Forecasting Models")
        
        # Model selection
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test Set Size (%)", 
                min_value=10, 
                max_value=50, 
                value=20,
                step=5
            ) / 100
            
            models_to_run = st.multiselect(
                "Select Models to Run",
                options=["ARIMA", "Prophet"] if PROPHET_AVAILABLE else ["ARIMA"],
                default=["ARIMA"]
            )
        
        with col2:
            # ARIMA parameters
            st.write("**ARIMA Parameters:**")
            p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
            d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
            q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1)
        
        if st.button("🚀 Run Models", type="primary"):
            # Split data
            train_data, test_data = st.session_state.analyzer.train_test_split(test_size)
            
            results = {}
            forecasts = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            model_count = len(models_to_run)
            
            for i, model_name in enumerate(models_to_run):
                status_text.text(f"Running {model_name} model...")
                
                try:
                    if model_name == "ARIMA":
                        # ARIMA Model
                        model = ARIMA(train_data[target_column], order=(p, d, q))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=len(test_data))
                        
                        # Calculate metrics
                        mae = mean_absolute_error(test_data[target_column], forecast)
                        rmse = np.sqrt(mean_squared_error(test_data[target_column], forecast))
                        mape = np.mean(np.abs((test_data[target_column] - forecast) / test_data[target_column])) * 100
                        
                        results[model_name] = {
                            'MAE': mae,
                            'RMSE': rmse,
                            'MAPE': mape
                        }
                        forecasts[model_name] = forecast
                    
                    elif model_name == "Prophet" and PROPHET_AVAILABLE:
                        # Prophet Model
                        prophet_df = train_data.reset_index().rename(
                            columns={train_data.index.name: 'ds', target_column: 'y'}
                        )
                        
                        model = Prophet()
                        model.fit(prophet_df)
                        
                        # Create future dataframe
                        future = model.make_future_dataframe(periods=len(test_data))
                        forecast_df = model.predict(future)
                        forecast = forecast_df['yhat'][-len(test_data):].values
                        
                        # Calculate metrics
                        mae = mean_absolute_error(test_data[target_column], forecast)
                        rmse = np.sqrt(mean_squared_error(test_data[target_column], forecast))
                        mape = np.mean(np.abs((test_data[target_column] - forecast) / test_data[target_column])) * 100
                        
                        results[model_name] = {
                            'MAE': mae,
                            'RMSE': rmse,
                            'MAPE': mape
                        }
                        forecasts[model_name] = forecast
                
                except Exception as e:
                    st.error(f"Error running {model_name}: {e}")
                
                progress_bar.progress((i + 1) / model_count)
            
            # Store results
            st.session_state.results = results
            st.session_state.forecasts = forecasts
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            
            status_text.text("✅ All models completed!")
            progress_bar.empty()
            
            # Display results immediately
            if results:
                st.subheader("📊 Model Performance")
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df.round(4))
                
                # Plot forecasts
                st.subheader("📈 Forecast Comparison")
                fig = go.Figure()
                
                # Add training data
                fig.add_trace(go.Scatter(
                    x=train_data.index,
                    y=train_data[target_column],
                    mode='lines',
                    name='Training Data',
                    line=dict(color='blue')
                ))
                
                # Add test data
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=test_data[target_column],
                    mode='lines',
                    name='Actual',
                    line=dict(color='black', width=3)
                ))
                
                # Add forecasts
                colors = ['red', 'green', 'purple', 'orange']
                for i, (model_name, forecast) in enumerate(forecasts.items()):
                    fig.add_trace(go.Scatter(
                        x=test_data.index,
                        y=forecast,
                        mode='lines',
                        name=f'{model_name} Forecast',
                        line=dict(color=colors[i % len(colors)], dash='dash')
                    ))
                
                fig.update_layout(
                    title="Model Forecasts Comparison",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("📋 Results Summary")
        
        if hasattr(st.session_state, 'results') and st.session_state.results:
            results = st.session_state.results
            forecasts = st.session_state.forecasts
            train_data = st.session_state.train_data
            test_data = st.session_state.test_data
            
            # Model comparison
            st.subheader("🏆 Model Ranking")
            results_df = pd.DataFrame(results).T
            
            # Rank by RMSE (lower is better)
            results_df['Rank'] = results_df['RMSE'].rank()
            results_df = results_df.sort_values('Rank')
            
            st.dataframe(results_df.round(4))
            
            # Best model
            best_model = results_df.index[0]
            st.success(f"🥇 Best Model: **{best_model}** (Lowest RMSE: {results_df.loc[best_model, 'RMSE']:.4f})")
            
            # Metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=results_df.index,
                    y=results_df['MAE'],
                    title="Mean Absolute Error (MAE)",
                    labels={'x': 'Model', 'y': 'MAE'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=results_df.index,
                    y=results_df['RMSE'],
                    title="Root Mean Square Error (RMSE)",
                    labels={'x': 'Model', 'y': 'RMSE'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download model results
                csv = results_df.to_csv()
                st.download_button(
                    label="📊 Download Model Results",
                    data=csv,
                    file_name="model_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download forecasts
                forecast_df = pd.DataFrame(forecasts, index=test_data.index)
                forecast_df['Actual'] = test_data[target_column]
                forecast_csv = forecast_df.to_csv()
                st.download_button(
                    label="📈 Download Forecasts",
                    data=forecast_csv,
                    file_name="forecasts.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("📝 Run models in the 'Modeling' tab to see results here.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🚀 Time Series Analysis & Forecasting Tool</p>
        <p>Built with Streamlit • Made with ❤️ by KIRTAN KUMAR + DIVYANSHU KAUSHIK</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    """
    This tool provides comprehensive time series analysis including:
    
    📊 **Data Exploration**
    - Statistical summaries
    - Visualization plots
    - Missing value detection
    
    🔍 **Analysis**
    - Stationarity testing
    - Distribution analysis
    - Seasonal decomposition
    
    🤖 **Forecasting**
    - ARIMA models
    - Prophet models
    - Performance evaluation
    
    📋 **Results**
    - Model comparison
    - Downloadable reports
    """
)

if not PROPHET_AVAILABLE:
    st.sidebar.warning(
        "⚠️ Prophet not available. Install with:\n`pip install prophet`"
    )