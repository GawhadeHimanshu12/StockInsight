import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)
plt.style.use('seaborn-v0_8-darkgrid')

def plot_price_line(df: pd.DataFrame, symbol: str, ma_cols: list[str] = ['MA_20', 'MA_50']) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if df.empty or 'Close' not in df.columns or 'Date' not in df.columns:
        logger.warning(f"Cannot plot price line for {symbol}, data missing.")
        ax.set_title(f"{symbol} - No Data Available")
        return fig
    ax.plot(df['Date'], df['Close'], label='Close Price', alpha=0.9)
    for col in ma_cols:
        if col in df.columns:
            ax.plot(df['Date'], df[col], label=col, alpha=0.7, linestyle='--')

    ax.set_title(f"{symbol} Closing Price & Moving Averages", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (INR)", fontsize=12)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_candlestick_plotly(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Creates an interactive Plotly candlestick chart.
    """
    if df.empty or not all(col in df.columns for col in ['Date', 'Open', 'High', 'Low', 'Close']):
        logger.warning(f"Cannot plot candlestick for {symbol}, data missing.")
        fig = go.Figure()
        fig.update_layout(title=f"{symbol} - No Data Available",
                          xaxis_rangeslider_visible=False)
        return fig
    df = df.sort_values(by='Date')

    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['MA_20'], 
            name='MA 20', 
            line=dict(color='orange', width=1, dash='dash')
        ))
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['MA_50'], 
            name='MA 50', 
            line=dict(color='blue', width=1, dash='dash')
        ))

    fig.update_layout(
        title=f"{symbol} Interactive Candlestick Chart",
        yaxis_title="Price (INR)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=True,
        legend_title="Legend"
    )
    
    return fig

def plot_return_hist(df: pd.DataFrame) -> plt.Figure:
    """
    Plots a histogram of daily returns.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if df.empty or 'Return' not in df.columns:
        logger.warning("Cannot plot return histogram, data missing.")
        ax.set_title("Daily Returns Distribution - No Data")
        return fig
    returns = df['Return'].dropna()
    
    if returns.empty:
        ax.set_title("Daily Returns Distribution - No Data")
        return fig
        
    sns.histplot(returns, kde=True, ax=ax, bins=50, color='blue', alpha=0.7)
    
    mean_ret = returns.mean()
    median_ret = returns.median()
    
    ax.axvline(mean_ret, color='red', linestyle='--', label=f'Mean: {mean_ret:.4f}')
    ax.axvline(median_ret, color='green', linestyle=':', label=f'Median: {median_ret:.4f}')
    
    ax.set_title("Daily Returns Distribution", fontsize=16)
    ax.set_xlabel("Daily Return", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_volatility_band(df: pd.DataFrame, window: int = 20) -> plt.Figure:
    """
    Plots the closing price with a rolling volatility band (e.g., +/- 1 Std Dev).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    col_name = f'Volatility_{window}'
    
    if df.empty or 'Close' not in df.columns or 'Date' not in df.columns:
        logger.warning("Cannot plot volatility band, data missing.")
        ax.set_title(f"{window}-Day Volatility Band - No Data")
        return fig
    if 'RollingStd_20' not in df.columns:
        df['RollingStd_20'] = df['Close'].rolling(window=20).std()
    if 'MA_20' not in df.columns:
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
    if df['RollingStd_20'].isnull().all() or df['MA_20'].isnull().all():
        ax.set_title(f"{window}-Day Volatility Band - Not Enough Data")
        return fig

    upper_band = df['MA_20'] + (df['RollingStd_20'] * 1)
    lower_band = df['MA_20'] - (df['RollingStd_20'] * 1)
    
    ax.plot(df['Date'], df['Close'], label='Close Price', alpha=0.8, color='blue')
    ax.plot(df['Date'], df['MA_20'], label='MA 20', linestyle='--', color='orange')
    ax.fill_between(df['Date'], lower_band, upper_band, 
                    color='gray', alpha=0.2, label='Volatility Band (+/- 1 SD)')

    ax.set_title(f"{window}-Day Volatility Band", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (INR)", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(corr_df: pd.DataFrame) -> plt.Figure:
    """
    Plots a heatmap of the provided correlation matrix.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if corr_df.empty or len(corr_df) < 2:
        logger.warning("Cannot plot correlation heatmap, insufficient data.")
        ax.set_title("Correlation Heatmap - Not Enough Data")
        return fig
        
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": .8}
    )
    
    ax.set_title("Stock Correlation Heatmap", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    return fig

def plot_prediction_vs_actual(plot_df: pd.DataFrame, title: str = "Prediction vs Actual") -> plt.Figure:
    """
    Plots the predicted values against the actual values from an ML model.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if plot_df.empty or not all(c in plot_df.columns for c in ['Date', 'Actual', 'Predicted']):
        logger.warning("Cannot plot predictions, data missing.")
        ax.set_title("Prediction vs Actual - No Data")
        return fig
        
    plot_df = plot_df.sort_values(by='Date')

    ax.plot(plot_df['Date'], plot_df['Actual'], label='Actual', alpha=0.8)
    ax.plot(plot_df['Date'], plot_df['Predicted'], label='Predicted', linestyle='--', alpha=0.9)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    return fig