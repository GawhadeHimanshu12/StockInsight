import yfinance as yf
import pandas as pd
import time
from utils.logger import setup_logger

logger = setup_logger(__name__)
EXPECTED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

def _sane_fallbacks(interval='1d', period='6mo'):
    attempts = [
        (period, interval),
        ('6mo', '1d'),
        ('1y', '1d'),
        ('3mo', '1d'),
        ('60d', '1d'),
        ('ytd', '1d'),
        ('1mo', '1d'),
    ]
    if 'm' in interval or 'h' in interval:
        attempts.extend([
            ('7d', '5m'),
            ('1d', '1m')
        ])
    seen = set()
    return [x for x in attempts if not (x in seen or seen.add(x))]

def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    if df.index.name and df.index.name.lower() in ['date', 'datetime']:
        df = df.reset_index()
    df.columns = [col.title() for col in df.columns]
    for col in df.columns:
        if col.lower() == 'date' or col.lower() == 'datetime':
            df.rename(columns={col: 'Date'}, inplace=True)
            break
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
    else:
        logger.warning("No 'Date' or 'Datetime' column found after normalization.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS) 
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col == 'Close' and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                df[col] = pd.NA
    df = df[EXPECTED_COLUMNS]
    df = df.dropna(subset=['Date', 'Close', 'Open'])
    
    return df

def fetch_stock(symbol: str, period: str = '6mo', interval: str = '1d') -> pd.DataFrame:
    logger.info(f"Initiating fetch for {symbol} (Period: {period}, Interval: {interval})")
    fallback_attempts = _sane_fallbacks(interval, period)
    df = pd.DataFrame()
    for (p, i) in fallback_attempts:
        for attempt in range(1, 4):
            try:
                logger.debug(f"Attempt {attempt}/3: yf.download('{symbol}', period='{p}', interval='{i}')")
                df = yf.download(symbol, period=p, interval=i, progress=False)
                
                if not df.empty:
                    logger.info(f"Success (yf.download): {symbol} [{p}, {i}] - {len(df)} rows.")
                    return _normalize_dataframe(df)
                
                logger.warning(f"yf.download returned empty for {symbol} [{p}, {i}]")
                
            except Exception as e:
                logger.error(f"Error during yf.download {symbol} [{p}, {i}]: {e}")
                wait_time = (2 ** attempt) * 0.5 
                logger.debug(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        time.sleep(0.5)
    logger.warning(f"yf.download failed for {symbol}. Trying yf.Ticker.history() fallbacks.")
    
    for (p, i) in fallback_attempts:
        try:
            logger.debug(f"Attempt: yf.Ticker('{symbol}').history(period='{p}', interval='{i}')")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=p, interval=i)
            if not df.empty:
                logger.info(f"Success (yf.Ticker.history): {symbol} [{p}, {i}] - {len(df)} rows.")
                return _normalize_dataframe(df)
            logger.warning(f"yf.Ticker.history returned empty for {symbol} [{p}, {i}]")
            
        except Exception as e:
            logger.error(f"Error during yf.Ticker.history {symbol} [{p}, {i}]: {e}")
        time.sleep(0.5)
    logger.error(f"All fetch attempts FAILED for {symbol}.")
    return pd.DataFrame(columns=EXPECTED_COLUMNS)