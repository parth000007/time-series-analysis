# 📈 IBM Stock Price Time Series Forecasting

This project performs **Exploratory Data Analysis (EDA)** and applies multiple **time series forecasting models** on IBM stock price data. It includes statistical tests, visualization, and model evaluation to identify the best forecasting approach for future price prediction.

---

## 🧭 Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Exploratory Data Analysis](#exploratory-data-analysis)  
- [Forecasting Models](#forecasting-models)  
- [Model Evaluation](#model-evaluation)  
- [Results & Visualizations](#results--visualizations)  
- [Usage Instructions](#usage-instructions)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 📝 Overview

The aim of this project is to:

- Explore historical trends in IBM stock price  
- Test time series stationarity and visualize patterns  
- Apply and compare different forecasting models including ARIMA, Prophet, and LSTM  
- Evaluate forecast accuracy using statistical metrics

---

## 📂 Dataset

- **Source**: Yahoo Finance (via `yfinance` library or downloaded CSV)
- **Ticker**: `IBM`
- **Features**: Date, Open, High, Low, Close, Volume, Adjusted Close
- **Timeframe**: Daily historical prices

---

## 🔍 Exploratory Data Analysis

- 📊 **Plots of price trends over time**  
- 🧪 **Stationarity testing** using the Augmented Dickey-Fuller (ADF) test  
- 🔄 **Rolling statistics** and **Autocorrelation (ACF/PACF)** plots  
- 🔧 **Decomposition** into trend, seasonality, and residuals using seasonal decomposition

---

## 🔮 Forecasting Models

| Model        | Description                                         |
|--------------|-----------------------------------------------------|
| ARIMA/SARIMA | Classical statistical model, tuned with ACF/PACF & AIC |
| Prophet      | Additive time series forecasting model by Facebook  |
| LSTM         | Recurrent neural network model for sequence prediction |
| Baseline     | Naive forecast using previous-day value             |

---

## 🧪 Model Evaluation

- **Train/Test Split**: Last ~20% of data reserved for testing  
- **Evaluation Metrics**:
  - RMSE (Root Mean Squared Error)  
  - MAE (Mean Absolute Error)  
  - MAPE (Mean Absolute Percentage Error)  
- **Visualization**:
  - Forecast vs. actual plots  
  - Confidence intervals for statistical models

---

## 📊 Results & Visualizations

| Model   | RMSE   | MAE   | MAPE   |
|---------|--------|-------|--------|
| ARIMA   | 0.104  |0.088  | 1.75%  |
> 📉 **Forecast Visualization**: Time series plots comparing model predictions with actual test data, including future forecasts.

---

## ⚙️ Usage Instructions

To run the analysis and generate the output:

```bash
cd notebook
jupyter nbconvert --execute time_series_forecasting_ibm.ipynb \
  --to html --output forecasts.html
```
## 🌐 View the Notebook on Kaggle

You can explore the full notebook and code on Kaggle:

🔗 [View on Kaggle](https://www.kaggle.com/code/kirtankumar/time-series-analysis-and-forecasting-for-ibm)

Feel free to upvote, comment, or suggest improvements!

## 🤝 Contributing

Contributions are welcome and appreciated!

If you’d like to improve this project, feel free to:

1. Fork the repository  
2. Create a new branch: `git checkout -b feature-branch-name`  
3. Commit your changes: `git commit -m 'Add some feature'`  
4. Push to the branch: `git push origin feature-branch-name`  
5. Open a pull request

For major changes, please open an issue first to discuss what you would like to change.

📌 Make sure your code follows consistent formatting and is well-documented.

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for full details.

You are free to use, modify, and distribute this project with proper attribution.

## 🙋‍♂️ Author

**Kirtan Kumar**  
📍 Student at NIT Rourkela  
🔗 [LinkedIn](https://www.linkedin.com/in/kirtankumar)  
📊 [Kaggle](https://www.kaggle.com/kirtankumar)   
📫 Email: kirtanfbd@gmail.com
