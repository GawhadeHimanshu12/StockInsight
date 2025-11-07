import os
DB_NAME = "stockinsight_history.db"
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'storage', DB_NAME)

LOG_FILE = "stockinsight.log"
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"
DEFAULT_SYMBOLS = [
    "TCS.NS",
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS"
]