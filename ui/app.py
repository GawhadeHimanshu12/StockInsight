import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import os
import io

try:
    from data_ingestion.yfinance_fetch import fetch_stock, _sane_fallbacks
    from data_processing.processing import pipeline_etl
    from storage.db import init_db, add_watched_symbol, get_watched_symbols, save_processed_snapshot
    from analytics.analytics import correlation_matrix, top_gainers
    from ml.models import train_regressor, train_classifier, save_model, DEFAULT_FEATURES
    from visualization.plots import (
        plot_price_line,
        plot_candlestick_plotly,
        plot_return_hist,
        plot_volatility_band,
        plot_correlation_heatmap,
        plot_prediction_vs_actual
    )
    from utils.config import DEFAULT_SYMBOLS, DEFAULT_PERIOD, DEFAULT_INTERVAL
    from utils.logger import setup_logger
except ImportError as e:
    st.error(f"Error importing project modules: {e}")
    st.error("Please ensure you are running the app from the root 'StockInsight' directory"
             " and that all 'requirements.txt' are installed.")
    st.stop()
    
st.set_page_config(
    page_title="StockInsight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = setup_logger(__name__)
logger.info("StockInsight Streamlit App Started.")

try:
    init_db()
except Exception as e:
    st.error(f"Failed to initialize database: {e}")
    logger.error(f"DB init failed: {e}")


def generate_synthetic_stock(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Generates a deterministic synthetic stock DataFrame for demo purposes.
    """
    logger.info(f"Generating synthetic data for {symbol} ({days} days)")
    np.random.seed(42) 
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='B')
    
    price = 100 + np.cumsum(np.random.normal(0.001, 0.02, days))
    price += np.sin(np.linspace(0, 5 * np.pi, days)) * 5
    price = price.round(2)
    price = np.maximum(price, 1.0)
    df = pd.DataFrame(index=dates)
    df['Open'] = (price - np.random.uniform(0.1, 0.5, days)).round(2)
    df['High'] = (price + np.random.uniform(0.1, 1.0, days)).round(2)
    df['Low'] = (price - np.random.uniform(0.1, 1.0, days)).round(2)
    df['Close'] = price
    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    df['Adj Close'] = df['Close']
    df['Volume'] = np.random.randint(1_000_000, 5_000_000, days)
    
    df = df.reset_index().rename(columns={'index': 'Date'})
    standard_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    return df[standard_cols]

@st.cache_data(ttl=600)
def load_and_process_data(symbol: str, period: str, interval: str, use_synthetic_on_fail: bool) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    logger.info(f"Loading data for {symbol} (Period: {period}, Interval: {interval})")
    raw_df = fetch_stock(symbol, period, interval)
    is_synthetic = False

    if raw_df.empty:
        if use_synthetic_on_fail:
            st.toast(f"Failed to fetch real data for {symbol}. Using synthetic demo data.", icon="⚠️")
            logger.warning(f"Fetch failed for {symbol}. Using synthetic fallback.")
            raw_df = generate_synthetic_stock(symbol, days=365)
            is_synthetic = True
        else:
            st.error(f"Failed to fetch data for {symbol}. Try a different period or check symbol.")
            return pd.DataFrame(), pd.DataFrame(), False

    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame(), is_synthetic
        
    processed_df = pipeline_etl(raw_df)
    
    if processed_df.empty:
        st.warning(f"Data processing failed for {symbol}. Displaying raw data only.")
        return raw_df, pd.DataFrame(), is_synthetic
        
    return raw_df, processed_df, is_synthetic

st.sidebar.image(
    "https://img.icons8.com/office/100/000000/bullish.png", 
    width=100
)
st.sidebar.title("StockInsight Controls")

st.sidebar.subheader("Watchlist")
new_symbol = st.sidebar.text_input("Add Symbol (e.g., RELIANCE.NS)", "").upper()

if st.sidebar.button("Add to Watchlist"):
    if new_symbol:
        if add_watched_symbol(new_symbol):
            st.sidebar.success(f"Added {new_symbol}")
            st.rerun()
        else:
            st.sidebar.warning(f"{new_symbol} already in list or invalid.")
    else:
        st.sidebar.error("Please enter a symbol.")
try:
    watched_symbols = get_watched_symbols()
    available_symbols = sorted(list(set(DEFAULT_SYMBOLS + watched_symbols)))
except Exception as e:
    logger.error(f"Failed to get watched symbols from DB: {e}")
    available_symbols = DEFAULT_SYMBOLS
    st.sidebar.error("Could not load watchlist from DB.")

st.sidebar.subheader("Analysis Parameters")
selected_symbol = st.sidebar.selectbox(
    "Select Primary Symbol",
    available_symbols,
    index=0
)
compare_symbols = st.sidebar.multiselect(
    "Select Symbols to Compare",
    available_symbols,
    default=available_symbols[:2] if len(available_symbols) >= 2 else available_symbols
)

period = st.sidebar.selectbox("Select Period", 
    ["6mo", "1y", "3mo", "ytd", "1mo", "5d", "max"], 
    index=0
)
interval = st.sidebar.selectbox("Select Interval", 
    ["1d", "1wk", "1mo", "5m", "15m", "1h"], 
    index=0
)

st.sidebar.subheader("App Controls")
use_synthetic = st.sidebar.checkbox("Use Synthetic Data on Fail", value=True)
if st.sidebar.button("Clear Cache & Reload Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info(
    "This is a demo application. "
    "Data is provided by Yahoo Finance and may be delayed. "
    "Not financial advice."
)


st.title(f"📈 StockInsight: {selected_symbol}")

try:
    raw_df, processed_df, is_synthetic = load_and_process_data(
        selected_symbol, period, interval, use_synthetic
    )
except Exception as e:
    st.error(f"An unexpected error occurred during data loading: {e}")
    logger.critical(f"Data loading failed: {e}", exc_info=True)
    st.stop()
    
if is_synthetic:
    st.warning("**Note:** You are currently viewing synthetically generated demo data. "
             "Real-time data fetching failed.", icon="⚠️")

if processed_df.empty:
    st.error(f"Could not load or process any data for {selected_symbol}.")
    if not raw_df.empty:
        st.subheader("Raw Data (Fetch Succeeded, Processing Failed)")
        st.dataframe(raw_df)
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "📊 Charts", "⚖️ Compare", "🤖 ML Demo", "📥 Download & Debug"]
)

with tab1:
    st.header(f"Overview: {selected_symbol}")
    
    try:
        latest = processed_df.iloc[-1]
        prev_close = processed_df.iloc[-2]['Close']
        st.subheader("Key Metrics (Latest)")
        cols = st.columns(4)
        
        cols[0].metric(
            label="Last Close Price",
            value=f"₹{latest['Close']:.2f}",
            delta=f"₹{latest['Close'] - prev_close:.2f} ({((latest['Close'] - prev_close) / prev_close) * 100:.2f}%)"
        )
        
        cols[1].metric(
            label="Volume",
            value=f"{latest['Volume'] / 1_000_000:.2f}M"
        )
        
        cols[2].metric(
            label="MA 20",
            value=f"₹{latest['MA_20']:.2f}",
            delta=f"₹{latest['Close'] - latest['MA_20']:.2f}",
            delta_color="off"
        )
        
        cols[3].metric(
            label="Volatility (20D Ann.)",
            value=f"{latest['Volatility_20'] * 100:.2f}%"
        )

    except (IndexError, KeyError) as e:
        st.warning(f"Could not display metrics. Data may be insufficient. Error: {e}")
        logger.warning(f"Metric calculation failed for {selected_symbol}: {e}")

    st.subheader("Price & Moving Averages")
    try:
        price_fig = plot_price_line(processed_df, selected_symbol)
        st.pyplot(price_fig)
    except Exception as e:
        st.error(f"Failed to plot price chart: {e}")
        logger.error(f"Price plot failed: {e}", exc_info=True)

    st.subheader("Recent Data")
    st.dataframe(processed_df.tail(8).set_index("Date").style.format(precision=2))


with tab2:
    st.header(f"Detailed Charts: {selected_symbol}")
    st.subheader("Interactive Candlestick Chart")
    try:
        candle_fig = plot_candlestick_plotly(processed_df, selected_symbol)
        st.plotly_chart(candle_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot candlestick chart: {e}")
        logger.error(f"Candlestick plot failed: {e}", exc_info=True)
    st.subheader("Price Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Daily Returns Distribution**")
        try:
            hist_fig = plot_return_hist(processed_df)
            st.pyplot(hist_fig)
        except Exception as e:
            st.error(f"Failed to plot return histogram: {e}")
            logger.error(f"Histogram plot failed: {e}", exc_info=True)
            
    with col2:
        st.markdown("**Volatility Band (Price +/- 1 SD)**")
        try:
            vol_fig = plot_volatility_band(processed_df, window=20)
            st.pyplot(vol_fig)
        except Exception as e:
            st.error(f"Failed to plot volatility band: {e}")
            logger.error(f"Volatility plot failed: {e}", exc_info=True)

with tab3:
    st.header("Compare Stocks")
    
    if len(compare_symbols) < 2:
        st.warning("Please select at least two stocks in the sidebar to compare.")
    else:
        symbol_df_map = {}
        with st.spinner("Loading data for comparison..."):
            for symbol in compare_symbols:
                _, proc_df, _ = load_and_process_data(symbol, period, interval, use_synthetic)
                if not proc_df.empty:
                    symbol_df_map[symbol] = proc_df
        
        if len(symbol_df_map) < 2:
            st.error("Could not load sufficient data for comparison.")
        else:
            st.subheader("Correlation Matrix (Close Prices)")
            try:
                corr_df = correlation_matrix(symbol_df_map)
                if not corr_df.empty:
                    st.dataframe(corr_df.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format(precision=2))
                    
                    st.subheader("Correlation Heatmap")
                    heatmap_fig = plot_correlation_heatmap(corr_df)
                    st.pyplot(heatmap_fig)
                else:
                    st.warning("Could not calculate correlation. No overlapping data.")
            except Exception as e:
                st.error(f"Failed to calculate correlation: {e}")
                logger.error(f"Correlation failed: {e}", exc_info=True)
            st.subheader(f"Top Gainers (Last 30 Days)")
            try:
                gainers_df = top_gainers(symbol_df_map, window_days=30)
                st.dataframe(gainers_df.style.format({"PercentChange": "{:.2f}%"}))
            except Exception as e:
                st.error(f"Failed to calculate top gainers: {e}")
                logger.error(f"Top gainers failed: {e}", exc_info=True)

with tab4:
    st.header(f"🤖 Machine Learning Demo (on {selected_symbol})")
    
    
    all_features = [col for col in processed_df.columns if col in DEFAULT_FEATURES]
    if not all_features:
        st.warning("No ML features (e.g., MA_20, RSI_14) could be computed. ML demo is unavailable.")
    else:
        features = st.multiselect(
            "Select Features for Training",
            options=all_features,
            default=all_features
        )
        
        if not features:
            st.error("Please select at least one feature.")
        else:
            st.subheader("Model 1: Predict Next Day's Close Price (Regression)")
            
            if st.button("Train Regression Model"):
                with st.spinner("Training RandomForestRegressor..."):
                    try:
                        model, metrics, plot_df = train_regressor(
                            processed_df, 
                            features, 
                            target_col='NextClose'
                        )
                        
                        if model:
                            st.success("Regression model trained successfully!")
                            st.metric("Test Samples", f"{metrics['test_samples']}")
                            st.metric("Test Root Mean Squared Error (RMSE)", f"₹{metrics['rmse']:.4f}")
                            st.metric("Test R-squared (R²)", f"{metrics['r2']:.4f}")
                            
                            pred_fig = plot_prediction_vs_actual(
                                plot_df.tail(100),
                                title="Regression: Actual vs. Predicted (Test Set)"
                            )
                            st.pyplot(pred_fig)
                            model_path = save_model(model, selected_symbol, 'regressor')
                            st.session_state['regressor_model_path'] = model_path
                            st.session_state['regressor_model'] = model
                            
                        else:
                            st.error("Model training failed. Insufficient data after cleaning.")
                    
                    except Exception as e:
                        st.error(f"An error occurred during regression training: {e}")
                        logger.error(f"Regressor training failed: {e}", exc_info=True)

            st.subheader("Model 2: Predict Next Day's Direction (Classifier)")
            
            if st.button("Train Classification Model"):
                with st.spinner("Training RandomForestClassifier..."):
                    try:
                        model, metrics, plot_df = train_classifier(
                            processed_df, 
                            features, 
                            target_col='Direction'
                        )
                        
                        if model:
                            st.success("Classifier model trained successfully!")
                            st.metric("Test Samples", f"{metrics['test_samples']}")
                            st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
                            st.metric("Test F1-Score (Weighted)", f"{metrics['f1_score']:.4f}")
                            
                            st.text("Test Set Classification Report:")
                            st.json(metrics['report'])
                            
                            model_path = save_model(model, selected_symbol, 'classifier')
                            st.session_state['classifier_model_path'] = model_path
                            st.session_state['classifier_model'] = model
                            
                        else:
                            st.error("Model training failed. Insufficient data or target not binary.")

                    except Exception as e:
                        st.error(f"An error occurred during classification training: {e}")
                        logger.error(f"Classifier training failed: {e}", exc_info=True)

with tab5:
    st.header("Download Data & Debug Info")

    st.subheader("Download Processed Data")
    try:
        csv_buffer = io.StringIO()
        processed_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{selected_symbol.replace('.', '_')}_processed.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Failed to prepare CSV for download: {e}")

    st.subheader("Download Trained ML Models")
    
    if 'regressor_model' in st.session_state and 'regressor_model_path' in st.session_state:
        try:
            with open(st.session_state['regressor_model_path'], 'rb') as f:
                st.download_button(
                    label="Download Regression Model (.pkl)",
                    data=f.read(),
                    file_name=f"{selected_symbol.replace('.', '_')}_regressor.pkl",
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.warning(f"Could not prepare regressor model for download: {e}")
            
    if 'classifier_model' in st.session_state and 'classifier_model_path' in st.session_state:
        try:
            with open(st.session_state['classifier_model_path'], 'rb') as f:
                st.download_button(
                    label="Download Classifier Model (.pkl)",
                    data=f.read(),
                    file_name=f"{selected_symbol.replace('.', '_')}_classifier.pkl",
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.warning(f"Could not prepare classifier model for download: {e}")
    st.subheader("Debug Information")
    
    with st.expander("Show Raw Fetched Data"):
        st.dataframe(raw_df)
    
    with st.expander("Show Processed Data (Head)"):
        st.dataframe(processed_df.head(20))
        
    with st.expander("Show Processed Data (Tail)"):
        st.dataframe(processed_df.tail(20))