# src/ml_models.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support, accuracy_score
import joblib
import os
import logging
import pandas_ta as ta # Import pandas_ta
import time  # Add this at the top of the file if not already imported
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.model_selection import learning_curve

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPricePredictor:
    """
    A class for training and evaluating Random Forest models for stock price prediction
    (either regression for price or classification for direction).
    """
    def __init__(self, model_type: str = 'regressor', n_estimators: int = 100, random_state: int = 42,
                 # Overfitting'i azaltmak için varsayılanları kısıtla
                 max_depth: Optional[int] = 10,
                 min_samples_leaf: int = 5,
                 max_features: float = 0.7): # Özelliklerin %70'ini kullan
        """
        Initializes the predictor.

        Args:
            model_type (str): 'regressor' for predicting price, 'classifier' for predicting direction.
            n_estimators (int): Number of trees in the forest.
            random_state (int): Random state for reproducibility.
            max_depth (Optional[int]): Maximum depth of the trees.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
            max_features (float): Maximum number of features to consider when looking for the best split.
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.features: List[str] = []

        model_params = {
            'n_estimators': n_estimators,
            'random_state': random_state,
            'n_jobs': -1,
            'warm_start': False,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

        if model_type == 'regressor':
            self.model = RandomForestRegressor(**model_params)
        elif model_type == 'classifier':
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError("model_type must be 'regressor' or 'classifier'")

        self.metrics: Dict[str, float] = {}
        self.feature_importance: Optional[Dict[str, float]] = None
        logger.info(f"Initialized StockPricePredictor (type={model_type}, params={model_params})")

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
                # Continue with the original columns
        
        # Eğer _SYMBOL soneki olan sütunlar varsa bunları temizle (Close_AAPL -> Close)
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
            # Bazı kolonlar hala eksik - sembol bazlı sütunları kontrol et
            logger.warning(f"Missing columns after first rename: {missing}. Found: {df.columns.tolist()}")
            
            # Sembol bazlı kolonlara uydurma işlemi
            for req_col in missing:
                candidates = [c for c in df.columns if req_col in c]
                if candidates:
                    logger.info(f"Using column {candidates[0]} for required column {req_col}")
                    df[req_col] = df[candidates[0]]
            
            # Son kontrol
            still_missing = [col for col in required_cols if col not in df.columns]
            if still_missing:
                raise ValueError(f"DataFrame must contain essential columns (Close, High, Low). Found: {df.columns.tolist()}")
        
        logger.info(f"Preparing features for {self.model_type}...")
        
        # Make a deep copy to avoid modifying original
        df_processed = df.copy()

        # Ensure Date column is datetime type and set as index for easier rolling calculations
        if 'Date' in df_processed.columns:
            df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
            df_processed = df_processed.set_index('Date')
        elif isinstance(df_processed.index, pd.DatetimeIndex):
            pass # Already has datetime index
        else:
            logger.warning("DataFrame must have a 'Date' column or a DatetimeIndex.")
            # Try to continue anyway
        
        # Ensure data is sorted by date
        try:
            df_processed.sort_index(inplace=True)
            # Remove duplicated indices which can cause issues
            df_processed = df_processed[~df_processed.index.duplicated(keep='first')]
        except Exception as e:
            logger.warning(f"Error sorting index: {e}")
        
        # --- Get Series objects safely ---
        # For each required column, ensure we get a Series, not a DataFrame
        try:
            close_col = df_processed['Close']
            if hasattr(close_col, 'columns'):  # If it's a DataFrame
                logger.warning("Close is a DataFrame, using first column")
                close_col = close_col.iloc[:, 0]
                
            high_col = None
            if 'High' in df_processed.columns:
                high_col = df_processed['High']
                if hasattr(high_col, 'columns'):
                    high_col = high_col.iloc[:, 0]
                    
            low_col = None
            if 'Low' in df_processed.columns:
                low_col = df_processed['Low']
                if hasattr(low_col, 'columns'):
                    low_col = low_col.iloc[:, 0]
                    
            volume_col = None
            if 'Volume' in df_processed.columns:
                volume_col = df_processed['Volume']
                if hasattr(volume_col, 'columns'):
                    volume_col = volume_col.iloc[:, 0]
                # Convert to numeric and fill NaNs
                volume_col = pd.to_numeric(volume_col, errors='coerce').fillna(0)
        except Exception as e:
            logger.error(f"Error extracting base columns: {e}")
            # Try to continue with original columns

        # --- Feature Engineering ---
        # 1. Lagged Close Prices (safe method)
        try:
            for i in range(1, 6): # Previous 5 days
                df_processed[f'prev_close_{i}'] = close_col.shift(i)
        except Exception as e:
            logger.warning(f"Error creating lagged prices: {e}")
        
        # 2. Returns
        try:
            df_processed['return_1d'] = close_col.pct_change(1)
            df_processed['return_5d'] = close_col.pct_change(5)
        except Exception as e:
            logger.warning(f"Error calculating returns: {e}")

        # 3. Moving Averages 
        try:
            df_processed['SMA_5'] = ta.sma(close_col, length=5)
            df_processed['SMA_20'] = ta.sma(close_col, length=20)
            df_processed['EMA_10'] = ta.ema(close_col, length=10)
            
            # MA Crossover indicator (if both MAs exist)
            if 'SMA_5' in df_processed.columns and 'SMA_20' in df_processed.columns:
                sma5 = df_processed['SMA_5']
                sma20 = df_processed['SMA_20']
                # Handle if they're DataFrames
                if hasattr(sma5, 'columns'): sma5 = sma5.iloc[:, 0]
                if hasattr(sma20, 'columns'): sma20 = sma20.iloc[:, 0]
                df_processed['SMA_5_gt_SMA_20'] = (sma5 > sma20).astype(int)
        except Exception as e:
            logger.warning(f"Error calculating moving averages: {e}")

        # 4. RSI with safe method
        try:
            df_processed['RSI_14'] = self._calculate_rsi(close_col, 14)
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")

        # 5. MACD (using pandas_ta)
        try:
            macd = ta.macd(close_col, fast=12, slow=26, signal=9)
            if macd is not None and isinstance(macd, pd.DataFrame):
                # Dynamically find MACD columns
                macd_col = next((col for col in macd.columns if col.startswith('MACD_')), None)
                signal_col = next((col for col in macd.columns if col.startswith('MACDs_')), None)
                hist_col = next((col for col in macd.columns if col.startswith('MACDh_')), None)
                
                if macd_col: df_processed['MACD'] = macd[macd_col]
                if signal_col: df_processed['MACD_signal'] = macd[signal_col]
                if hist_col: df_processed['MACD_hist'] = macd[hist_col]
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")

        # 6. Bollinger Bands
        try:
            bbands = ta.bbands(close_col, length=20, std=2)
            if bbands is not None and isinstance(bbands, pd.DataFrame):
                # Dynamically find column names by prefix
                bbu_col = next((c for c in bbands.columns if c.startswith('BBU_')), None)
                bbl_col = next((c for c in bbands.columns if c.startswith('BBL_')), None)
                bbm_col = next((c for c in bbands.columns if c.startswith('BBM_')), None)
                
                if bbu_col: df_processed['BB_upper'] = bbands[bbu_col]
                if bbl_col: df_processed['BB_lower'] = bbands[bbl_col]
                if bbm_col: df_processed['BB_mid'] = bbands[bbm_col]
                
                # Calculate bandwidth and position within bands
                if all(col in df_processed.columns for col in ['BB_upper', 'BB_lower', 'BB_mid']):
                    try:
                        bb_mid = df_processed['BB_mid']
                        bb_upper = df_processed['BB_upper']
                        bb_lower = df_processed['BB_lower']
                        
                        # Handle if they're DataFrames
                        if hasattr(bb_mid, 'columns'): bb_mid = bb_mid.iloc[:, 0]
                        if hasattr(bb_upper, 'columns'): bb_upper = bb_upper.iloc[:, 0]
                        if hasattr(bb_lower, 'columns'): bb_lower = bb_lower.iloc[:, 0]
                        
                        # Bandwidth calculation
                        bb_mid_safe = bb_mid.replace(0, 1e-9)  # Avoid division by zero
                        df_processed['BB_width_pct'] = (bb_upper - bb_lower) / bb_mid_safe
                    except Exception as bb_calc_err:
                        logger.warning(f"Bollinger Band additional calc error: {bb_calc_err}")
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")

        # 7. ATR (Average True Range)
        if all(col is not None for col in [high_col, low_col, close_col]):
            try:
                df_processed['ATR_14'] = ta.atr(high_col, low_col, close_col, length=14)
            except Exception as e:
                logger.warning(f"Error calculating ATR: {e}")

        # 8. Volume-based indicators
        if volume_col is not None:
            try:
                # Calculate OBV safely
                df_processed['OBV'] = ta.obv(close_col, volume_col)
                
                # Only calculate VWAP if we have all required columns
                if all(col is not None for col in [high_col, low_col, close_col]):
                    try:
                        df_processed['VWAP'] = ta.vwap(high_col, low_col, close_col, volume_col)
                    except Exception as vwap_err:
                        logger.warning(f"VWAP calculation failed: {vwap_err}")
                
                # Volume moving averages
                try:
                    df_processed['Volume_SMA_5'] = ta.sma(volume_col, length=5)
                    df_processed['Volume_SMA_20'] = ta.sma(volume_col, length=20)
                    
                    # Calculate volume ratio only if Volume_SMA_20 exists
                    if 'Volume_SMA_20' in df_processed.columns:
                        vol_sma20 = df_processed['Volume_SMA_20']
                        if hasattr(vol_sma20, 'columns'):  # If it's a DataFrame
                            vol_sma20 = vol_sma20.iloc[:, 0]
                        df_processed['Volume_Ratio'] = volume_col / (vol_sma20 + 1e-8)
                except Exception as vol_ma_err:
                    logger.warning(f"Volume moving average calculation failed: {vol_ma_err}")
            except Exception as volume_err:
                logger.warning(f"Volume indicators calculation failed: {volume_err}")

        # 9. Date-based features (safer extraction)
        try:
            if isinstance(df_processed.index, pd.DatetimeIndex):
                df_processed['day_of_week'] = df_processed.index.dayofweek
                df_processed['month'] = df_processed.index.month
                df_processed['day_of_year'] = df_processed.index.dayofyear
                try:
                    df_processed['week_of_year'] = df_processed.index.isocalendar().week.astype(int)
                except:
                    # Older pandas versions
                    df_processed['week_of_year'] = df_processed.index.week
        except Exception as date_err:
            logger.warning(f"Error calculating date features: {date_err}")

        # --- Fill NaNs from feature engineering ---
        df_processed = df_processed.ffill().bfill()
        logger.info("Filled NaNs in features.")

        # --- Define Target ---
        try:
            if self.model_type == 'regressor':
                # Predict the next day's closing price
                df_processed['target'] = close_col.shift(-1)
            else: # classifier
                # Predict if the price will go up (1) or down/stay same (0) next day
                df_processed['target'] = (close_col.shift(-1) > close_col).astype(int)
        except Exception as target_err:
            logger.error(f"Error creating target variable: {target_err}")
            # Create empty target as fallback
            df_processed['target'] = np.nan

        # --- Clean Data ---
        # Drop rows with NaNs in 'target' before separating X and y to keep alignment
        initial_rows = len(df_processed)
        df_processed.dropna(subset=['target'], inplace=True)
        rows_after_dropna = len(df_processed)
        logger.info(f"Dropped {initial_rows - rows_after_dropna} rows due to NaN target.")

        if df_processed.empty:
            logger.error("DataFrame empty after dropping NaN target.")
            return pd.DataFrame(), pd.Series(dtype='float64'), []

        # Select features (exclude target and potentially other non-numeric cols if any remain)
        exclude_cols = ['target', 'Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close', 'returns']
        self.features = [col for col in df_processed.columns if col not in exclude_cols]
        # Make sure features exist after potential drops/renames from pandas_ta
        self.features = [f for f in self.features if f in df_processed.columns]

        X = df_processed[self.features].copy() # Select final features
        y = df_processed['target'].copy()

        # Ensure all features are numeric (handle potential lingering non-numeric types)
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Non-numeric columns found: {non_numeric_cols.tolist()}.")
            for col in non_numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X.dropna(axis=1, inplace=True)
            self.features = X.columns.tolist()

        # Final check for NaNs
        if X.isnull().any().any():
            logger.warning("NaNs detected in final feature matrix X.")
            X = X.fillna(X.median()) # Fill any remaining NaNs

        logger.info(f"Prepared features: {len(self.features)} columns. Shape: X={X.shape}, y={y.shape}")
        return X, y, self.features

    def train(self, df: pd.DataFrame, test_size: float = 0.2, auto_optimize: bool = True) -> tuple:
        """
        Trains the model and calculates performance metrics. Optionally performs 
        automatic hyperparameter tuning to reduce overfitting.

        Args:
            df (pd.DataFrame): DataFrame with stock data.
            test_size (float): Proportion of data to use for testing.
            auto_optimize (bool): Whether to automatically run GridSearchCV optimization.

        Returns:
            tuple: (metrics, X_test, y_test, y_pred)
                  metrics (dict): Performance metrics.
                  X_test (pd.DataFrame): Test feature matrix.
                  y_test (pd.Series): Test target values.
                  y_pred (array): Predicted values.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Training failed: Input DataFrame is invalid or empty.")
            return {}, None, None, None

        X, y, _ = self.prepare_features(df)
        if X.empty or y.empty:
            logger.error("Training failed: No data available after feature preparation.")
            return {}, None, None, None

        # Use TimeSeriesSplit for splitting time-series data correctly
        tscv = TimeSeriesSplit(n_splits=5)  # 5 splits is usually a good balance

        # Get the indices for the last split as train/test
        train_index, test_index = list(tscv.split(X))[-1]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        logger.info(f"Training {self.model_type} model with {len(X_train)} train / {len(X_test)} test samples (TimeSeriesSplit)")

        # Perform automatic hyperparameter tuning if requested
        if auto_optimize:
            logger.info("Starting hyperparameter optimization with GridSearchCV...")
            try:
                param_grid = self._get_default_param_grid()
                self.model = self.evaluate_parameters(df, param_grid=param_grid, verbose=0)['best_model']
                logger.info(f"Using optimized model: {self.model.get_params()}")
            except Exception as e:
                logger.warning(f"Parameter optimization failed: {e}. Using default model.")

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        if self.model_type == 'classifier':
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Up)

        # Calculate metrics
        if self.model_type == 'regressor':
            self.metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred)
            }
        else:  # classifier
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            self.metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            }

        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        logger.info(f"Model training completed. Metrics: {self.metrics}")
        return self.metrics, X_test, y_test, y_pred, y_pred_proba

    def _get_default_param_grid(self) -> dict:
        """Returns a default parameter grid for GridSearchCV based on model type."""
        if self.model_type == 'regressor':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.7]
            }
        else:  # classifier
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.7],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }

    def evaluate_parameters(self, df: pd.DataFrame, param_grid: dict = None, cv: int = 3, verbose: int = 1) -> dict:
        """
        Evaluates different model parameters using GridSearchCV.

        Args:
            df (pd.DataFrame): DataFrame with stock data.
            param_grid (dict): Parameter grid to search, or None for defaults.
            cv (int): Number of cross-validation folds.
            verbose (int): Verbosity level (0-3).

        Returns:
            dict: Dictionary with best parameters, score, and model.
        """
        if not param_grid:
            param_grid = self._get_default_param_grid()

        X, y, _ = self.prepare_features(df)
        if X.empty or y.empty:
            logger.error("Parameter evaluation failed: No data available after feature preparation.")
            return {'error': 'No data available'}

        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv)

        # Scoring metric based on model type
        scoring = 'r2' if self.model_type == 'regressor' else 'f1'

        # Initialize base model according to type
        base_model = RandomForestRegressor(random_state=self.random_state) if self.model_type == 'regressor' else RandomForestClassifier(random_state=self.random_state)

        logger.info(f"Starting GridSearchCV with {len(param_grid)} parameters and {cv} TimeSeriesSplit folds")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=tscv,
            n_jobs=-1,  # Use all available cores
            verbose=verbose
        )

        try:
            grid_search.fit(X, y)
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best {scoring} score: {best_score:.3f}")

            # Create and return a new model with the best parameters
            if self.model_type == 'regressor':
                best_model = RandomForestRegressor(random_state=self.random_state, **best_params)
            else:
                best_model = RandomForestClassifier(random_state=self.random_state, **best_params)

            return {
                'best_params': best_params,
                'best_score': best_score,
                'best_model': best_model,
                'cv_results': grid_search.cv_results_
            }
        except Exception as e:
            logger.error(f"GridSearchCV failed: {e}", exc_info=True)
            return {'error': str(e)}

    def plot_learning_curve(self, df: pd.DataFrame, cv: int = 5) -> Dict[str, Any]:
        """
        Plot learning curve to evaluate model performance with different training sizes.
        
        Args:
            df: DataFrame to use for plotting learning curve
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with plot data
        """
        try:
            if self.model is None:
                logger.warning("Model not trained yet")
                return {'error': 'Model not trained yet'}
            
            # Prepare data
            X, y, _ = self.prepare_features(df)  # Unpack correctly, ignoring the third return value
            
            # Define train sizes
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            # Calculate learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                self.model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error'
            )
            
            # Calculate mean and std
            train_mean = -np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = -np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.grid()
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
            plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
            plt.title('Learning Curve (Random Forest)')
            plt.xlabel('Training examples')
            plt.ylabel('Mean Squared Error')
            plt.legend(loc='best')
            plt.tight_layout()
            
            # Create directory if it doesn't exist
            image_dir = os.path.join('static', 'images', 'analysis', 'learning')
            os.makedirs(image_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = int(datetime.now().timestamp())
            
            # Save plot
            image_path = os.path.join(image_dir, f'rf_learning_curve_{timestamp}.png')
            plt.savefig(image_path)
            plt.close()
            
            logger.info(f"Learning curve plot saved to {image_path}")
            
            return {
                'image_path': image_path.replace('static/', ''),
                'train_sizes': train_sizes.tolist(),
                'train_scores': train_mean.tolist(),
                'test_scores': test_mean.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error plotting learning curve: {e}", exc_info=True)
            return {'error': str(e)}

    def plot_feature_importance(self, top_n: int = 15) -> dict:
        """
        Plot feature importance graph for the model.
        
        Args:
            top_n: Number of top features to show
            
        Returns:
            Dictionary with plot data
        """
        try:
            if self.model is None:
                logger.warning("Model not trained yet")
                return {'error': 'Model not trained yet'}
            
            # Get feature importance
            feature_importance = self.model.feature_importances_
            
            # Create DataFrame for plotting
            feature_importance_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': feature_importance
            })
            
            # Sort by importance and take top N
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Create directory if it doesn't exist
            image_dir = os.path.join('static', 'images', 'analysis', 'features')
            os.makedirs(image_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = int(datetime.now().timestamp())
            
            # Save plot
            image_path = os.path.join(image_dir, f'rf_feature_importance_{timestamp}.png')
            plt.savefig(image_path)
            plt.close()
            
            logger.info(f"Feature importance plot saved to {image_path}")
            
            return {
                'image_path': image_path.replace('static/', ''),
                'top_features': feature_importance_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}", exc_info=True)
            return {'error': str(e)}

    def save_model(self, path="models/rf_model.joblib"):
        """Saves the trained model to a file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            return False

    def load_model(self, path="models/rf_model.joblib"):
        """Loads a trained model from a file."""
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False

def train_random_forest_model(df, test_size=0.2, params=None):
    """
    Train a random forest model on the given dataframe.
    
    Args:
        df: DataFrame with stock data
        test_size: Fraction of data to use for testing
        params: Parameters for the model (optional)
        
    Returns:
        Tuple of (metrics, X_test, y_test, y_pred, model)
    """
    predictor = StockPricePredictor()
    if params:
        predictor = StockPricePredictor(**params)
    
    metrics, X_test, y_test, y_pred = predictor.train(df, test_size=test_size)
    
    return metrics, X_test, y_test, y_pred, predictor.model