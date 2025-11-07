

# StockInsight:



---

**StockInsight** is an interactive **Streamlit** application for financial market analysis, featuring real-time data retrieval, technical indicator computation, and machine learning-based forecasting.  

It empowers users to explore market trends, visualize stock performance, and experiment with predictive models in a single, easy-to-use interface.

---

## Key Features

### Real-time Data Ingestion
- Fetches **historical and live stock data** via [Yahoo Finance](https://finance.yahoo.com/) using the `yfinance` API.

### ETL & Technical Indicators
- Cleans and transforms stock data.
- Computes essential indicators:
  - **MA_20**, **MA_50** — Moving Averages  
  - **20D Annualized Volatility**  
  - **RSI_14** — Relative Strength Index  

### PySpark Processing (Optional)
- Demonstrates distributed data processing using **PySpark**, including scalable rolling mean and volatility calculations.

### Interactive Visualizations
- **Candlestick charts** with Plotly  
- **Returns histograms**  
- **Volatility bands** and trend overlays  
- Fully interactive and dynamic UI built with Streamlit.

### Comparative Analytics
- Computes **correlation matrices** across multiple stocks.  
- Highlights **top gainers** within a selected watchlist.

### Machine Learning Demo
- Trains **Random Forest** models for:
  - **Regression:** Predicting next-day close price  
  - **Classification:** Predicting next-day direction (Up/Down)

---

## Technology Stack

| Component | Libraries / Tools |
|------------|------------------|
| **Frontend / Web App** | `streamlit` |
| **Data Source** | `yfinance` |
| **Data Processing** | `pandas`, `numpy` |
| **Big Data** | `pyspark` |
| **Machine Learning** | `scikit-learn` |
| **Visualization** | `plotly`, `matplotlib`, `seaborn` |

---

<img width="1920" height="1080" alt="Screenshot 2025-11-07 160509" src="https://github.com/user-attachments/assets/5d895b9d-fd3d-49f9-9519-686008315b56" />
<img width="1920" height="1080" alt="Screenshot 2025-11-07 160519" src="https://github.com/user-attachments/assets/94286631-aac0-4ffc-9d94-ffa3ed86c535" />
<img width="1920" height="1080" alt="Screenshot 2025-11-07 160336" src="https://github.com/user-attachments/assets/6867bbdc-8d5f-4bec-8414-1959c6355a0a" />
<img width="1920" height="1080" alt="Screenshot 2025-11-07 160354" src="https://github.com/user-attachments/assets/fbd3cd01-5bd8-4090-bdce-c546dacffbc4" />
<img width="1920" height="1080" alt="Screenshot 2025-11-07 160427" src="https://github.com/user-attachments/assets/aac559d4-a41c-448a-b297-cb5ed522a1c2" />
<img width="1920" height="1080" alt="Screenshot 2025-11-07 160411" src="https://github.com/user-attachments/assets/de160bf9-0f88-400d-b6e2-5e49079f3714" />
<img width="1920" height="1080" alt="Screenshot 2025-11-07 160814" src="https://github.com/user-attachments/assets/b60ae746-6330-47e2-82fe-26cfe33cc8ba" />

