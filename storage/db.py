import sqlalchemy as sa
from sqlalchemy import create_engine, MetaData, Table, Column, String, insert, select
from utils.config import DB_PATH
from utils.logger import setup_logger
import pandas as pd
import os
import datetime

logger = setup_logger(__name__)

engine = None
metadata = MetaData()

watched_table = Table(
    'watched',
    metadata,
    Column('symbol', String, primary_key=True, unique=True),
    Column('added_at', String)
)

def get_engine():
    global engine
    if engine is None:
        try:
            db_dir = os.path.dirname(DB_PATH)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            engine = create_engine(f'sqlite:///{DB_PATH}')
        except Exception as e:
            logger.error(f"Failed to create database engine at {DB_PATH}: {e}")
            raise
    return engine

def init_db():
    try:
        eng = get_engine()
        metadata.create_all(eng)
        logger.info(f"Database initialized and 'watched' table ensured at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def add_watched_symbol(symbol: str) -> bool:
    if not symbol:
        return False
        
    symbol_upper = symbol.upper()
    stmt = insert(watched_table).values(
        symbol=symbol_upper,
        added_at=datetime.datetime.now().isoformat()
    )
    
    try:
        eng = get_engine()
        with eng.connect() as conn:
            try:
                conn.execute(stmt)
                conn.commit()
                logger.info(f"Added symbol to watchlist: {symbol_upper}")
                return True
            except sa.exc.IntegrityError:
                logger.warning(f"Symbol {symbol_upper} is already in the watchlist.")
                return False
    except Exception as e:
        logger.error(f"Failed to add symbol {symbol_upper} to database: {e}")
        return False

def get_watched_symbols() -> list[str]:
    stmt = select(watched_table.c.symbol).order_by(watched_table.c.added_at)
    
    try:
        eng = get_engine()
        with eng.connect() as conn:
            result = conn.execute(stmt).fetchall()
            symbols = [row[0] for row in result]
            return symbols
    except Exception as e:
        logger.error(f"Failed to get watched symbols: {e}")
        return []

def save_processed_snapshot(df: pd.DataFrame, symbol: str) -> str | None:
    try:
        output_dir = "snapshots"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol.replace('.', '_')}_{timestamp}.csv"
        file_path = os.path.join(output_dir, filename)
        
        df.to_csv(file_path, index=False)
        logger.info(f"Saved snapshot to: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save snapshot for {symbol}: {e}")
        return None