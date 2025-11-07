import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, classification_report
import pickle
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_FEATURES = ['MA_20', 'MA_50', 'Volatility_20', 'RSI_14']

def create_feature_target(df: pd.DataFrame, 
                          features: list[str], 
                          target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame] | tuple[None, None, None]:
    if df.empty:
        logger.warning("Cannot create features, DataFrame is empty.")
        return None, None, None
        
    required_cols = features + [target_col] + ['Date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for ML: {missing_cols}")
        return None, None, None
    df_ml = df[required_cols].dropna()
    
    if df_ml.empty:
        logger.warning("No data left after dropping NaNs for ML. Try different features.")
        return None, None, None

    X = df_ml[features]
    y = df_ml[target_col]
    
    return X, y, df_ml

def train_regressor(df: pd.DataFrame, 
                    features: list[str] = DEFAULT_FEATURES, 
                    target_col: str = 'NextClose') -> tuple[object, dict, pd.DataFrame]:
    X, y, df_ml = create_feature_target(df, features, target_col)
    
    if X is None or len(X) < 20:
        logger.warning("Insufficient data to train regressor.")
        return None, {}, pd.DataFrame()
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df_ml.iloc[split_idx:]['Date']

    logger.info(f"Training regressor. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    metrics = {
        "mse": mse,
        "rmse": mse**0.5,
        "r2": r2,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    logger.info(f"Regressor metrics: {metrics}")
    plot_df = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test,
        'Predicted': preds
    })

    return model, metrics, plot_df

def train_classifier(df: pd.DataFrame, 
                     features: list[str] = DEFAULT_FEATURES, 
                     target_col: str = 'Direction') -> tuple[object, dict, pd.DataFrame]:

    X, y, df_ml = create_feature_target(df, features, target_col)
    
    if X is None or len(X) < 20:
        logger.warning("Insufficient data to train classifier.")
        return None, {}, pd.DataFrame()
        
    if y.nunique() < 2:
        logger.warning(f"Target column '{target_col}' is not binary. Found values: {y.unique()}")
        return None, {}, pd.DataFrame()

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df_ml.iloc[split_idx:]['Date']
    
    logger.info(f"Training classifier. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "report": classification_report(y_test, preds, output_dict=True),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    logger.info(f"Classifier accuracy: {acc:.4f}, F1: {f1:.4f}")

    plot_df = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test,
        'Predicted': preds
    })

    return model, metrics, plot_df

def save_model(model: object, symbol: str, model_type: str = 'regressor') -> str | None:
    """
    Saves a trained model to disk using pickle.
    """
    try:
        output_dir = "ml_models"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{symbol.replace('.', '_')}_{model_type}.pkl"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Saved model to: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save model for {symbol}: {e}")
        return None

def load_model(file_path: str) -> object | None:
    """
    Loads a pickled model from disk.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            return None
            
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Loaded model from: {file_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {file_path}: {e}")
        return None