import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)

def top_gainers(symbol_df_map: dict[str, pd.DataFrame], window_days: int = 30) -> pd.DataFrame:
    performance = []
    
    for symbol, df in symbol_df_map.items():
        if df.empty or 'Close' not in df.columns or len(df) < window_days:
            logger.warning(f"Skipping {symbol} for top_gainers (insufficient data).")
            continue
            
        try:
            # Ensure data is sorted by date
            df = df.sort_values(by='Date')
            
            end_price = df['Close'].iloc[-1]
            start_price = df['Close'].iloc[-window_days]
            
            pct_change = ((end_price - start_price) / start_price) * 100
            
            performance.append({
                'Symbol': symbol,
                'PercentChange': pct_change,
                'StartPrice': start_price,
                'EndPrice': end_price
            })
        except Exception as e:
            logger.error(f"Error calculating gain for {symbol}: {e}")

    if not performance:
        return pd.DataFrame(columns=['Symbol', 'PercentChange'])

    return pd.DataFrame(performance).sort_values(by='PercentChange', ascending=False)

def correlation_matrix(symbol_df_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close_prices = {}
    
    for symbol, df in symbol_df_map.items():
        if df.empty or 'Close' not in df.columns:
            logger.warning(f"Skipping {symbol} for correlation (no 'Close' data).")
            continue
        
        df_indexed = df.set_index('Date')['Close'].rename(symbol)
        close_prices[symbol] = df_indexed
        
    if len(close_prices) < 2:
        logger.warning("Need at least two stocks to calculate correlation.")
        return pd.DataFrame()
    combined_df = pd.DataFrame(close_prices)

    combined_df = combined_df.dropna()
    
    if combined_df.empty:
        logger.error("No overlapping data found for correlation.")
        return pd.DataFrame()

    return combined_df.corr()

def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Computes rolling annualized volatility.
    (Helper function, similar to the one in processing.py)
    """
    if 'LogReturn' not in df.columns:
        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        
    volatility = df['LogReturn'].rolling(window=window).std() * np.sqrt(252)
    return volatility.rename(f"Volatility_{window}")

def rolling_statistics(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculates rolling mean and standard deviation for the 'Close' price.
    """
    if df.empty or 'Close' not in df.columns:
        return df

    df[f'RollingMean_{window}'] = df['Close'].rolling(window=window).mean()
    df[f'RollingStd_{window}'] = df['Close'].rolling(window=window).std()
    
    return df