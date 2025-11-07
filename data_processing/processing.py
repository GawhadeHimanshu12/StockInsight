import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("clean_basic received an empty DataFrame.")
        return df
    if 'Date' not in df.columns:
        logger.error("No 'Date' column found in raw data.")
        return pd.DataFrame()
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df.columns = [col.title() for col in df.columns]
    if 'Close' not in df.columns or df['Close'].isnull().all():
        if 'Adj Close' in df.columns:
            logger.info("Using 'Adj Close' to fill missing 'Close' column.")
            df['Close'] = df['Adj Close']
        else:
            logger.error("'Close' and 'Adj Close' are both missing. Cannot proceed.")
            return pd.DataFrame()
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    return df.reset_index()

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'Close' not in df.columns or len(df) < 15:
        logger.warning("Not enough data to compute technical indicators.")
        return df

    df = df.set_index('Date').sort_index()

    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))

    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    df['Volatility_20'] = df['LogReturn'].rolling(window=20).std() * np.sqrt(252)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    return df.reset_index()

def label_direction(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if df.empty or 'Close' not in df.columns:
        return df
        
    df = df.set_index('Date').sort_index()
    
    df['NextClose'] = df['Close'].shift(-horizon)
    df['Direction'] = (df['NextClose'] > df['Close']).astype(int)
    return df.reset_index()

def pipeline_etl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("ETL pipeline received empty DataFrame. Skipping.")
        return pd.DataFrame()
        
    logger.info(f"ETL pipeline started. Input rows: {len(df)}")
    
    try:
        df_clean = clean_basic(df.copy())
        if df_clean.empty:
            logger.error("ETL failed at: clean_basic")
            return pd.DataFrame()
            
        df_ta = compute_technical_indicators(df_clean)
        if df_ta.empty:
            logger.warning("ETL completed with no technical indicators (insufficient data).")
            return df_clean 
            
        df_labeled = label_direction(df_ta)
        if df_labeled.empty:
            logger.error("ETL failed at: label_direction")
            return df_ta
            
        logger.info(f"ETL pipeline finished. Output rows: {len(df_labeled)}")
        return df_labeled
        
    except Exception as e:
        logger.error(f"ETL pipeline failed with exception: {e}", exc_info=True)
        return pd.DataFrame()


def process_with_pyspark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: Demonstrates PySpark processing if installed.
    Converts Pandas DF to Spark DF, calculates a rolling mean, and returns Pandas DF.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import avg, col, lit
        from pyspark.sql.window import Window
        
        logger.info("PySpark found. Attempting Spark processing...")
        
        spark = (
            SparkSession.builder.appName("StockInsightSparkDemo")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .getOrCreate()
        )
        
        # Convert Pandas DF to Spark DF
        spark_df = spark.createDataFrame(df.where(pd.notnull(df), None)) # Handle NaNs
        
        # Define a window specification
        # PySpark windows are more complex; they require an orderable column
        # and non-timestamp types for rangeBetween. We'll use row-based.
        window_spec = (
            Window.orderBy("Date")
            .rowsBetween(-9, 0) # 10-day window
        )
        
        # Perform a simple calculation (e.g., MA_10)
        spark_df = spark_df.withColumn("MA_10_Spark", avg(col("Close")).over(window_spec))
        
        logger.info("Spark processing complete. Converting back to Pandas.")
        
        # Convert back to Pandas
        result_df = spark_df.toPandas()
        
        spark.stop()
        return result_df

    except ImportError:
        logger.warning("PySpark not installed. Skipping optional Spark processing.")
        return df
    except Exception as e:
        logger.error(f"PySpark processing failed: {e}", exc_info=True)
        return df