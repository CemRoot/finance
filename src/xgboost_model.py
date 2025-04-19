import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import time
from typing import Dict, Tuple, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostStockPredictor:
    """
    A class for training and evaluating XGBoost models for stock price prediction
    and direction classification.
    """
    def __init__(self, model_type='regressor', random_state=42,
                 learning_rate=0.1, n_estimators=100, max_depth=5,
                 subsample=0.8, colsample_bytree=0.8):
        """
        Initialize the XGBoost predictor.
        
        Args:
            model_type (str): 'regressor' for predicting price, 'classifier' for predicting direction
            random_state (int): Random state for reproducibility
            learning_rate (float): Learning rate for gradient boosting
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            subsample (float): Subsample ratio of the training instances
            colsample_bytree (float): Subsample ratio of columns when constructing each tree
        """
        self.model_type = model_type
        self.random_state = random_state
        self.features = []
        self.scaler = StandardScaler()  # XGBoost benefits from scaled features
        
        # Common parameters
        self.model_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # Initialize appropriate model based on type
        if model_type == 'regressor':
            self.model = XGBRegressor(**self.model_params)
        elif model_type == 'classifier':
            self.model = XGBClassifier(**self.model_params, use_label_encoder=False, eval_metric='logloss')
        else:
            raise ValueError("model_type must be 'regressor' or 'classifier'")
        
        self.metrics = {}
        self.feature_importance = None
        logger.info(f"Initialized XGBoostStockPredictor (type={model_type}, params={self.model_params})")
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        # Ensure prices is a Series
        if hasattr(prices, 'columns'):  # If it's a DataFrame
            logger.warning("Prices is a DataFrame in _calculate_rsi, using first column")
            prices = prices.iloc[:, 0]
            
        if not isinstance(prices, pd.Series): 
            logger.warning("Invalid prices type in _calculate_rsi, returning empty Series")
            return pd.Series(index=getattr(prices, 'index', pd.Index([])), dtype=float)
            
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0) 
        loss = -delta.where(delta < 0, 0.0)

        # Use exponential moving average
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

        # Prevent division by zero
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50)  # Fill NaN values with neutral RSI
    
    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepares features from the stock data DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with stock data (must include 'Close', 'High', 'Low').

        Returns:
            tuple: (X, y, feature_names)
                   X (pd.DataFrame): Feature matrix.
                   y (pd.Series): Target variable.
                   feature_names (list): List of columns used as features.
        """
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        
        # Check for MultiIndex columns and flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            logger.warning("MultiIndex columns found. Flattening...")
            try:
                df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns.values]
            except Exception as e:
                logger.error(f"Failed to flatten MultiIndex: {e}")
        
        # Handle columns with symbol suffixes
        rename_dict = {}
        for col in df.columns:
            if '_' in col:
                parts = col.split('_')
                if len(parts) == 2 and parts[0] in ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj']:
                    rename_dict[col] = parts[0]
        
        if rename_dict:
            logger.info(f"Renaming columns to standardize: {rename_dict}")
            df = df.rename(columns=rename_dict)
        
        # Check for essential columns
        required_cols = ['Close', 'High', 'Low']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            # Check for columns with similar names
            logger.warning(f"Missing columns: {missing}. Found: {df.columns.tolist()}")
            
            # Try to find suitable columns
            for req_col in missing:
                candidates = [c for c in df.columns if req_col.lower() in c.lower()]
                if candidates:
                    logger.info(f"Using column {candidates[0]} for required column {req_col}")
                    df[req_col] = df[candidates[0]]
            
            # Check again
            still_missing = [col for col in required_cols if col not in df.columns]
            if still_missing:
                raise ValueError(f"DataFrame must contain essential columns (Close, High, Low). Found: {df.columns.tolist()}")
        
        logger.info(f"Preparing features for {self.model_type}...")
        
        # Make a deep copy to avoid modifying original
        df_processed = df.copy()
        
        # Ensure Date column is datetime type and set as index if necessary
        if 'Date' in df_processed.columns:
            df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
            df_processed = df_processed.set_index('Date')
        elif isinstance(df_processed.index, pd.DatetimeIndex):
            pass  # Already has datetime index
        else:
            logger.warning("DataFrame should have a 'Date' column or a DatetimeIndex.")
        
        # Ensure data is sorted by date
        try:
            df_processed.sort_index(inplace=True)
            # Remove duplicated indices which can cause issues
            df_processed = df_processed[~df_processed.index.duplicated(keep='first')]
        except Exception as e:
            logger.warning(f"Error sorting index: {e}")
        
        # Get Series objects safely for each required column
        try:
            close_col = df_processed['Close']
            if hasattr(close_col, 'columns'):  # If it's a DataFrame
                close_col = close_col.iloc[:, 0]
                
            high_col = df_processed['High']
            if hasattr(high_col, 'columns'):
                high_col = high_col.iloc[:, 0]
                
            low_col = df_processed['Low']
            if hasattr(low_col, 'columns'):
                low_col = low_col.iloc[:, 0]
                
            volume_col = None
            if 'Volume' in df_processed.columns:
                volume_col = df_processed['Volume']
                if hasattr(volume_col, 'columns'):
                    volume_col = volume_col.iloc[:, 0]
                volume_col = pd.to_numeric(volume_col, errors='coerce').fillna(0)
        except Exception as e:
            logger.error(f"Error extracting base columns: {e}")
            raise ValueError(f"Error processing core columns: {e}")
        
        # --- Feature Engineering ---
        features_df = pd.DataFrame(index=df_processed.index)
        
        # 1. Basic price features
        features_df['close'] = close_col
        features_df['high'] = high_col
        features_df['low'] = low_col
        if volume_col is not None:
            features_df['volume'] = volume_col
            features_df['volume_log'] = np.log1p(volume_col)  # Log transform for volume
        
        # 2. Price differences and returns
        features_df['close_diff'] = close_col.diff()
        features_df['return_1d'] = close_col.pct_change(1)
        features_df['return_5d'] = close_col.pct_change(5)
        features_df['return_10d'] = close_col.pct_change(10)
        
        # 3. Moving Averages
        for window in [5, 10, 20, 50]:
            features_df[f'sma_{window}'] = close_col.rolling(window=window).mean()
            features_df[f'ema_{window}'] = close_col.ewm(span=window, adjust=False).mean()
        
        # 4. Price Volatility
        for window in [5, 10, 20]:
            features_df[f'volatility_{window}d'] = close_col.pct_change().rolling(window=window).std()
        
        # 5. RSI (Relative Strength Index)
        features_df['rsi_14'] = self._calculate_rsi(close_col, 14)
        
        # 6. Bollinger Bands
        for window in [20]:
            mid_band = close_col.rolling(window=window).mean()
            std_dev = close_col.rolling(window=window).std()
            features_df[f'bb_upper_{window}'] = mid_band + (std_dev * 2)
            features_df[f'bb_lower_{window}'] = mid_band - (std_dev * 2)
            features_df[f'bb_width_{window}'] = (features_df[f'bb_upper_{window}'] - features_df[f'bb_lower_{window}']) / mid_band
            
            # Position within Bollinger Bands (0 = lower band, 1 = upper band)
            features_df[f'bb_pos_{window}'] = (close_col - features_df[f'bb_lower_{window}']) / (features_df[f'bb_upper_{window}'] - features_df[f'bb_lower_{window}'] + 1e-9)
        
        # 7. MACD (Moving Average Convergence Divergence)
        ema12 = close_col.ewm(span=12, adjust=False).mean()
        ema26 = close_col.ewm(span=26, adjust=False).mean()
        features_df['macd'] = ema12 - ema26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        features_df['macd_hist'] = features_df['macd'] - features_df['macd_signal']
        
        # 8. Price Momentum
        for window in [5, 10, 20]:
            features_df[f'momentum_{window}'] = close_col / close_col.shift(window) - 1
        
        # 9. High-Low Range and Ratio
        features_df['hl_range'] = high_col - low_col
        features_df['hl_ratio'] = high_col / (low_col + 1e-9)  # Avoid division by zero
        
        # 10. Distance from Moving Averages (as percentage)
        for window in [10, 20, 50]:
            ma_col = f'sma_{window}'
            if ma_col in features_df.columns:
                features_df[f'dist_ma_{window}'] = (close_col - features_df[ma_col]) / features_df[ma_col]
        
        # Drop NaN values which appear due to rolling windows
        features_df.dropna(inplace=True)
        
        # Define target variable
        if self.model_type == 'regressor':
            # For regression, predict next day's closing price
            target = close_col.shift(-1).loc[features_df.index]
        else:  # classifier
            # For classification, predict price direction (1 = up, 0 = down)
            next_day_price = close_col.shift(-1)
            current_price = close_col
            target = (next_day_price > current_price).astype(int).loc[features_df.index]
        
        # Drop any remaining NaN targets and align features with target
        target = target[~target.isna()]
        features_df = features_df.loc[target.index]
        
        # Store feature names for later use
        self.features = features_df.columns.tolist()
        
        logger.info(f"Prepared {len(features_df)} samples with {len(self.features)} features")
        return features_df, target, self.features
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, auto_optimize: bool = True) -> Tuple[Dict, pd.DataFrame, pd.Series, np.ndarray, Optional[np.ndarray]]:
        """
        Train the XGBoost model on stock data.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            test_size (float): Proportion of data to use for testing
            auto_optimize (bool): Whether to perform hyperparameter optimization
            
        Returns:
            Tuple[Dict, pd.DataFrame, pd.Series, np.ndarray, Optional[np.ndarray]]: 
                (metrics, X_test, y_test, y_pred, y_pred_proba)
        """
        start_time = time.time()
        logger.info(f"Starting {self.model_type} training with test_size={test_size}, auto_optimize={auto_optimize}")
        
        try:
            # Prepare features
            X, y, feature_names = self.prepare_features(df)
            logger.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")
            
            # Split data with time series consideration
            if isinstance(X.index, pd.DatetimeIndex) or isinstance(X.index, pd.PeriodIndex):
                # Time-based split
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                logger.info(f"Time-based split: train={len(X_train)}, test={len(X_test)}")
            else:
                # Standard split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                logger.info(f"Random split: train={len(X_train)}, test={len(X_test)}")
            
            # Scale features if needed
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Hyperparameter optimization if requested
            if auto_optimize:
                logger.info("Starting hyperparameter optimization...")
                param_grid = self._get_default_param_grid()
                
                # Use time series cross-validation for more reliable results
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Define scoring metric based on model type
                scoring = 'neg_mean_squared_error' if self.model_type == 'regressor' else 'accuracy'
                
                # Create GridSearchCV
                grid_search = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit GridSearchCV
                grid_search.fit(X_train_scaled, y_train)
                
                # Get best parameters and model
                best_params = grid_search.best_params_
                self.model = grid_search.best_estimator_
                logger.info(f"Best parameters: {best_params}")
            else:
                # Train with default parameters
                self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Get probabilities if classifier
            y_pred_proba = None
            if self.model_type == 'classifier' and hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            if self.model_type == 'regressor':
                self.metrics = {
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2': r2_score(y_test, y_pred)
                }
            else:  # classifier
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='binary', zero_division=0
                )
                accuracy = accuracy_score(y_test, y_pred)
                self.metrics = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1
                }
            
            # Calculate feature importance
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            
            training_time = time.time() - start_time
            self.metrics['training_time'] = training_time
            logger.info(f"XGBoost model training completed in {training_time:.2f}s. Metrics: {self.metrics}")
            
            return self.metrics, X_test, y_test, y_pred, y_pred_proba
            
        except Exception as e:
            logger.error(f"Error in XGBoost training: {e}", exc_info=True)
            raise
    
    def _get_default_param_grid(self) -> dict:
        """Returns a default parameter grid for GridSearchCV based on model type."""
        if self.model_type == 'regressor':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]
            }
        else:  # classifier
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'scale_pos_weight': [1, 3, 5]  # For imbalanced classes
            }
    
    def evaluate_parameters(self, df: pd.DataFrame, param_grid: dict = None, cv: int = 3, verbose: int = 1) -> dict:
        """
        Evaluate different parameter combinations using cross-validation.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            param_grid (dict): Parameter grid to search
            cv (int): Number of cross-validation folds
            verbose (int): Verbosity level
            
        Returns:
            dict: Results of parameter evaluation
        """
        logger.info("Starting parameter evaluation...")
        
        # Prepare features
        X, y, _ = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use default parameter grid if not specified
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Define scoring metric based on model type
        scoring = 'neg_mean_squared_error' if self.model_type == 'regressor' else 'accuracy'
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=verbose
        )
        
        # Fit GridSearchCV
        grid_search.fit(X_scaled, y)
        
        # Get results
        cv_results = grid_search.cv_results_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}, best score: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': cv_results
        }
    
    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance for the model.
        
        Args:
            top_n: Number of top features to show
            
        Returns:
            Dictionary with plot information
        """
        try:
            if self.model is None:
                logger.warning("Model not trained yet")
                return {'error': 'Model not trained yet'}
                
            # Get feature importances
            importance_type = 'weight'  # Other options: 'gain', 'cover', 'total_gain', 'total_cover'
            importance = self.model.get_booster().get_score(importance_type=importance_type)
            
            # Convert to DataFrame
            importance_df = pd.DataFrame({
                'Feature': list(importance.keys()),
                'Importance': list(importance.values())
            })
            
            # Sort and filter top N
            importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            importance_plot = plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.title(f"XGBoost Feature Importance ({importance_type})")
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Create directory if it doesn't exist
            image_dir = os.path.join('static', 'images', 'analysis', 'features')
            os.makedirs(image_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = int(datetime.now().timestamp())
            
            # Save image
            image_path = os.path.join(image_dir, f'xgb_feature_importance_{timestamp}.png')
            plt.savefig(image_path)
            plt.close()
            
            logger.info(f"XGBoost feature importance plot saved: {image_path}")
            
            return {
                'image_path': image_path.replace('static/', ''),
                'top_features': importance_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error plotting XGBoost feature importance: {e}", exc_info=True)
            return {'error': str(e)}
    
    def save_model(self, path="models/xgb_model.joblib"):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model, scaler, and metadata
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'model_type': self.model_type,
                'params': self.model_params
            }
            
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise
    
    def load_model(self, path="models/xgb_model.joblib"):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the model file
        """
        try:
            model_data = joblib.load(path)
            
            # Restore model attributes
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.metrics = model_data['metrics']
            self.feature_importance = model_data['feature_importance']
            self.model_type = model_data['model_type']
            self.model_params = model_data['params']
            
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise


# Helper functions for model usage

def train_xgboost_model(data: pd.DataFrame, model_type: str = 'regressor', 
                       auto_optimize: bool = True, save_path: str = None) -> Tuple[XGBoostStockPredictor, Dict]:
    """
    Train an XGBoost model on stock data and return the model and metrics.
    
    Args:
        data (pd.DataFrame): DataFrame with stock data
        model_type (str): 'regressor' for price prediction or 'classifier' for direction prediction
        auto_optimize (bool): Whether to perform hyperparameter optimization
        save_path (str): Path to save the model
        
    Returns:
        Tuple[XGBoostStockPredictor, Dict]: Trained model and metrics
    """
    # Create model
    model = XGBoostStockPredictor(model_type=model_type)
    
    # Train model
    metrics, _, _, _, _ = model.train(data, auto_optimize=auto_optimize)
    
    # Save model if path is provided
    if save_path:
        model.save_model(save_path)
    
    return model, metrics

def load_xgboost_model(filepath: str) -> XGBoostStockPredictor:
    """
    Load a trained XGBoost model from disk.
    
    Args:
        filepath (str): Path to the model file
        
    Returns:
        XGBoostStockPredictor: Loaded model
    """
    model = XGBoostStockPredictor()
    model.load_model(filepath)
    return model 