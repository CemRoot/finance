import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from typing import Dict, Tuple
import logging
import os
import datetime
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class SVMStockPredictor:
    """
    Support Vector Machine model for stock price prediction and classification.
    Provides regression (SVR) for price prediction and classification (SVC) for direction prediction.
    """
    
    def __init__(self, 
                 n_features: int = 10, 
                 prediction_horizon: int = 1,
                 model_type: str = 'regression',
                 kernel: str = 'rbf',
                 tune_hyperparams: bool = False,
                 verbose: bool = False):
        """
        Initialize the SVM predictor.
        
        Args:
            n_features (int): Number of features (lookback window) to use for prediction
            prediction_horizon (int): Number of days ahead to predict
            model_type (str): 'regression' for price prediction or 'classification' for direction prediction
            kernel (str): Kernel function to use ('linear', 'poly', 'rbf', 'sigmoid')
            tune_hyperparams (bool): Whether to perform hyperparameter tuning
            verbose (bool): Whether to print detailed information
        """
        self.n_features = n_features
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.kernel = kernel
        self.tune_hyperparams = tune_hyperparams
        self.verbose = verbose
        
        # Initialize model and scalers
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Initialize metrics
        self.metrics = {}
        
        # Initialize model params
        self.model_params = {
            'kernel': kernel,
            'gamma': 'scale',
            'C': 1.0,
            'epsilon': 0.1,  # Only for SVR
            'cache_size': 1000,  # Increase cache size for faster training
        }
        
        logger.info(f"Initialized SVM predictor with {n_features} features, "
                   f"{prediction_horizon} days horizon, {model_type} model type, "
                   f"and {kernel} kernel")
    
    def _create_model(self) -> None:
        """
        Create the SVM model based on model_type.
        """
        if self.model_type == 'regression':
            # Remove classification-specific parameters
            params = {k: v for k, v in self.model_params.items() if k != 'probability'}
            self.model = SVR(**params)
            logger.info(f"Created SVR model with parameters: {params}")
        else:  # classification
            # Add classification-specific parameters
            params = self.model_params.copy()
            params['probability'] = True  # Enable probability estimates
            if 'epsilon' in params:  # Remove regression-specific parameters
                params.pop('epsilon')
            self.model = SVC(**params)
            logger.info(f"Created SVC model with parameters: {params}")
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning using cross-validation.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training targets
            
        Returns:
            Dict: Best parameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter grid based on model type
        if self.model_type == 'regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
            }
            base_model = SVR(cache_size=1000)
        else:  # classification
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'class_weight': ['balanced', None]
            }
            base_model = SVC(probability=True, cache_size=1000)
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error' if self.model_type == 'regression' else 'accuracy',
            n_jobs=-1,
            verbose=1 if self.verbose else 0
        )
        
        # Fit the grid search
        grid_search.fit(X, y)
        
        # Get best parameters
        best_params = grid_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        
        # Return best parameters
        return best_params
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and targets from stock data.
        
        Args:
            data (pd.DataFrame): DataFrame with stock data (must have 'Close' column)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Prepared features and targets
        """
        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if 'Close' not in data.columns:
            # Check if this is a MultiIndex DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                close_cols = [col for col in data.columns if 'Close' in col[1:]]
                if close_cols:
                    # Use the first Close column found
                    close_col = close_cols[0]
                    logger.info(f"Using column {close_col} as Close price")
                    data = data[close_col].to_frame(name='Close')
                else:
                    raise ValueError("DataFrame with MultiIndex columns must contain a 'Close' column")
            else:
                raise ValueError("DataFrame must contain a 'Close' column")
        
        # Extract close price
        close_prices = data['Close'].values
        
        # Create features and targets based on model type
        if self.model_type == 'regression':
            # Prepare regression features (predict actual prices)
            features = []
            targets = []
            
            for i in range(len(close_prices) - self.n_features - self.prediction_horizon + 1):
                # Features: n_features previous prices
                feature = close_prices[i:i+self.n_features]
                # Target: price after prediction_horizon days
                target = close_prices[i+self.n_features+self.prediction_horizon-1]
                
                features.append(feature)
                targets.append(target)
        else:  # classification
            # Prepare classification features (predict price direction)
            features = []
            targets = []
            
            for i in range(len(close_prices) - self.n_features - self.prediction_horizon + 1):
                # Features: n_features previous prices
                feature = close_prices[i:i+self.n_features]
                # Target: price direction after prediction_horizon days (1 for up, 0 for down or same)
                current_price = close_prices[i+self.n_features-1]
                future_price = close_prices[i+self.n_features+self.prediction_horizon-1]
                target = 1 if future_price > current_price else 0
                
                features.append(feature)
                targets.append(target)
        
        # Convert to numpy arrays
        features = np.array(features)
        targets = np.array(targets)
        
        # Get corresponding dates (for the targets)
        dates = data.index[self.n_features+self.prediction_horizon-1:].values[:len(targets)]
        
        # Return as DataFrames with dates as index
        features_df = pd.DataFrame(features, index=dates)
        targets_df = pd.DataFrame(targets, index=dates, columns=['target'])
        
        return features_df, targets_df
    
    def prepare_additional_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare additional technical indicators as features.
        
        Args:
            data (pd.DataFrame): DataFrame with stock data (OHLCV columns)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Prepared features and targets
        """
        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}. Some features may not be calculated.")
        
        # Create a copy of the dataframe
        df = data.copy()
        
        # Calculate technical indicators
        
        # 1. Moving Averages
        if 'Close' in df.columns:
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 2. Price changes
        if 'Close' in df.columns:
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        
        # 3. Volatility
        if 'Close' in df.columns:
            df['Volatility'] = df['Close'].rolling(window=5).std()
        
        # 4. Relative Strength Index (RSI)
        if 'Close' in df.columns:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. MACD
        if 'Close' in df.columns:
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 6. Bollinger Bands
        if 'Close' in df.columns:
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # 7. Price/Volume relationship
        if 'Close' in df.columns and 'Volume' in df.columns:
            df['Price_Volume_Ratio'] = df['Close'] / df['Volume']
        
        # 8. Momentum
        if 'Close' in df.columns:
            df['Momentum'] = df['Close'] - df['Close'].shift(4)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Prepare features and targets based on model type
        if self.model_type == 'regression':
            # For regression, we predict the actual price
            features = []
            targets = []
            
            feature_columns = [col for col in df.columns if col != 'Close']
            
            for i in range(len(df) - self.prediction_horizon + 1):
                # Use all technical indicators as features
                feature = df.iloc[i][feature_columns].values
                # Target: price after prediction_horizon days
                if i + self.prediction_horizon < len(df):
                    target = df['Close'].iloc[i + self.prediction_horizon]
                    
                    features.append(feature)
                    targets.append(target)
        else:  # classification
            # For classification, we predict the price direction
            features = []
            targets = []
            
            feature_columns = [col for col in df.columns if col != 'Close']
            
            for i in range(len(df) - self.prediction_horizon + 1):
                # Use all technical indicators as features
                feature = df.iloc[i][feature_columns].values
                # Target: price direction after prediction_horizon days
                if i + self.prediction_horizon < len(df):
                    current_price = df['Close'].iloc[i]
                    future_price = df['Close'].iloc[i + self.prediction_horizon]
                    target = 1 if future_price > current_price else 0
                    
                    features.append(feature)
                    targets.append(target)
        
        # Convert to numpy arrays
        features = np.array(features)
        targets = np.array(targets)
        
        # Get corresponding dates (for the targets)
        dates = df.index[self.prediction_horizon:].values[:len(targets)]
        
        # Return as DataFrames with dates as index
        features_df = pd.DataFrame(features, index=dates)
        targets_df = pd.DataFrame(targets, index=dates, columns=['target'])
        
        return features_df, targets_df
    
    def train(self, data: pd.DataFrame, use_additional_features: bool = False) -> Dict:
        """
        Train the SVM model on stock data.
        
        Args:
            data (pd.DataFrame): DataFrame with stock data
            use_additional_features (bool): Whether to use additional technical indicators
            
        Returns:
            Dict: Training metrics
        """
        logger.info("Starting model training...")
        
        # Prepare features and targets
        if use_additional_features:
            features_df, targets_df = self.prepare_additional_features(data)
        else:
            features_df, targets_df = self.prepare_features(data)
        
        # Extract numpy arrays
        X = features_df.values
        y = targets_df['target'].values
        
        # Calculate train/test split index
        split_idx = int(len(X) * 0.8)
        
        # Split data
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train = self.feature_scaler.fit_transform(X_train)
        X_test = self.feature_scaler.transform(X_test)
        
        # Scale targets for regression
        if self.model_type == 'regression':
            y_train = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Perform hyperparameter tuning if requested
        if self.tune_hyperparams:
            best_params = self._tune_hyperparameters(X_train, y_train)
            self.model_params.update(best_params)
        
        # Create model
        self._create_model()
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Make predictions on test set
        if self.model_type == 'regression':
            y_pred = self.model.predict(X_test)
            # Inverse transform predictions
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_test_orig = y_test  # Original scale
            
            # Calculate metrics
            mse = mean_squared_error(y_test_orig, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_orig, y_pred)
            r2 = r2_score(y_test_orig, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"Regression metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        else:  # classification
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate directional accuracy (if price goes up and model predicts up, it's correct)
            metrics = {
                'accuracy': accuracy,
                'pred_positive_rate': np.mean(y_pred),
                'actual_positive_rate': np.mean(y_test)
            }
            
            logger.info(f"Classification metrics - Accuracy: {accuracy:.4f}")
        
        # Store metrics
        self.metrics = metrics
        
        return metrics
    
    def predict(self, data: pd.DataFrame, use_additional_features: bool = False) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            data (pd.DataFrame): DataFrame with stock data
            use_additional_features (bool): Whether to use additional technical indicators
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if self.model is None:
            logger.error("Model not trained yet. Call train() first.")
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info("Making predictions...")
        
        # Prepare features
        if use_additional_features:
            features_df, _ = self.prepare_additional_features(data)
        else:
            features_df, _ = self.prepare_features(data)
        
        # Extract numpy array
        X = features_df.values
        
        # Scale features
        X = self.feature_scaler.transform(X)
        
        # Make predictions
        if self.model_type == 'regression':
            # Get scaled predictions
            predictions = self.model.predict(X)
            # Inverse transform predictions
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
            
            # Create DataFrame with predictions
            results = pd.DataFrame({
                'Predicted_Price': predictions
            }, index=features_df.index)
            
            # Add actual values if available
            if 'Close' in data.columns:
                actuals = data.loc[features_df.index, 'Close']
                results['Actual_Price'] = actuals.values
                
                # Calculate prediction error
                results['Error'] = results['Actual_Price'] - results['Predicted_Price']
                results['Percent_Error'] = results['Error'] / results['Actual_Price'] * 100
        else:  # classification
            # Get class predictions
            predictions = self.model.predict(X)
            # Get class probabilities
            probabilities = self.model.predict_proba(X)[:, 1]  # Probability of positive class
            
            # Create DataFrame with predictions
            results = pd.DataFrame({
                'Predicted_Direction': predictions,
                'Probability_Up': probabilities
            }, index=features_df.index)
            
            # Add actual directions if available
            if 'Close' in data.columns:
                # Calculate actual directions
                shifted_close = data['Close'].shift(self.prediction_horizon)
                actuals = (data['Close'] > shifted_close).astype(int)
                results['Actual_Direction'] = actuals.loc[features_df.index].values
                
                # Calculate accuracy
                results['Correct'] = results['Predicted_Direction'] == results['Actual_Direction']
        
        return results
    
    def plot_predictions(self, predictions: pd.DataFrame, data: pd.DataFrame = None, 
                        title: str = 'SVM Model Predictions', figsize: Tuple[int, int] = (12, 6),
                        save_path: str = None) -> None:
        """
        Plot the model predictions against actual values.
        
        Args:
            predictions (pd.DataFrame): DataFrame with predictions
            data (pd.DataFrame): Original stock data (optional)
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save the plot
        """
        # Create figure
        plt.figure(figsize=figsize)
        
        if self.model_type == 'regression':
            # Plot actual vs. predicted prices
            if 'Actual_Price' in predictions.columns:
                plt.plot(predictions.index, predictions['Actual_Price'], 'b-', label='Actual Price')
            plt.plot(predictions.index, predictions['Predicted_Price'], 'r--', label='Predicted Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add prediction horizon info
            plt.figtext(0.01, 0.01, f'Prediction Horizon: {self.prediction_horizon} days', 
                      ha='left', fontsize=9)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
        else:  # classification
            # Plot probabilities for price going up
            plt.plot(predictions.index, predictions['Probability_Up'], 'g-', label='Probability Up')
            plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
            
            # Plot actual directions if available
            if 'Actual_Direction' in predictions.columns:
                # Create twin axis for direction
                ax2 = plt.twinx()
                ax2.plot(predictions.index, predictions['Actual_Direction'], 'bo', alpha=0.3, label='Actual Direction')
                ax2.set_ylabel('Direction (1=Up, 0=Down)')
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(['Down', 'Up'])
                
                # Add to legend
                lines, labels = plt.gca().get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(lines + lines2, labels + labels2)
            else:
                plt.legend()
            
            plt.xlabel('Date')
            plt.ylabel('Probability')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Add prediction horizon info
            plt.figtext(0.01, 0.01, f'Prediction Horizon: {self.prediction_horizon} days', 
                      ha='left', fontsize=9)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
        
        # Save plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, data: pd.DataFrame, use_additional_features: bool = False) -> Dict:
        """
        Evaluate the model on new data.
        
        Args:
            data (pd.DataFrame): DataFrame with stock data
            use_additional_features (bool): Whether to use additional technical indicators
            
        Returns:
            Dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained yet. Call train() first.")
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Make predictions
        predictions = self.predict(data, use_additional_features)
        
        # Calculate metrics
        if self.model_type == 'regression':
            if 'Actual_Price' in predictions.columns:
                mse = mean_squared_error(predictions['Actual_Price'], predictions['Predicted_Price'])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(predictions['Actual_Price'], predictions['Predicted_Price'])
                r2 = r2_score(predictions['Actual_Price'], predictions['Predicted_Price'])
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                logger.info(f"Evaluation metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            else:
                logger.warning("Cannot calculate regression metrics without actual prices")
                metrics = {}
        else:  # classification
            if 'Actual_Direction' in predictions.columns:
                accuracy = accuracy_score(predictions['Actual_Direction'], predictions['Predicted_Direction'])
                metrics = {
                    'accuracy': accuracy,
                    'pred_positive_rate': predictions['Predicted_Direction'].mean(),
                    'actual_positive_rate': predictions['Actual_Direction'].mean()
                }
                
                logger.info(f"Evaluation metrics - Accuracy: {accuracy:.4f}")
            else:
                logger.warning("Cannot calculate classification metrics without actual directions")
                metrics = {}
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model, scalers, and parameters to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            logger.error("Model not trained yet. Call train() first.")
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data to save
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'n_features': self.n_features,
            'prediction_horizon': self.prediction_horizon,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'metrics': self.metrics,
            'created_at': datetime.datetime.now().isoformat()
        }
        
        # Save model data
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model, scalers, and parameters from a file.
        
        Args:
            filepath (str): Path to the model file
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file {filepath} not found")
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Load model and parameters
        self.model = model_data['model']
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.n_features = model_data['n_features']
        self.prediction_horizon = model_data['prediction_horizon']
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        self.metrics = model_data['metrics']
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model type: {self.model_type}, n_features: {self.n_features}, "
                   f"prediction_horizon: {self.prediction_horizon}")


# Helper functions for model usage

def train_svm_model(data: pd.DataFrame, model_type: str = 'regression', 
                   n_features: int = 10, prediction_horizon: int = 1,
                   tune_hyperparams: bool = False, use_additional_features: bool = True,
                   save_path: str = None) -> Tuple[SVMStockPredictor, Dict]:
    """
    Train an SVM model on stock data and return the model and metrics.
    
    Args:
        data (pd.DataFrame): DataFrame with stock data
        model_type (str): 'regression' for price prediction or 'classification' for direction prediction
        n_features (int): Number of features (lookback window) to use for prediction
        prediction_horizon (int): Number of days ahead to predict
        tune_hyperparams (bool): Whether to perform hyperparameter tuning
        use_additional_features (bool): Whether to use additional technical indicators
        save_path (str): Path to save the model
        
    Returns:
        Tuple[SVMStockPredictor, Dict]: Trained model and metrics
    """
    # Create model
    model = SVMStockPredictor(
        n_features=n_features,
        prediction_horizon=prediction_horizon,
        model_type=model_type,
        tune_hyperparams=tune_hyperparams
    )
    
    # Train model
    metrics = model.train(data, use_additional_features=use_additional_features)
    
    # Save model if path is provided
    if save_path:
        model.save_model(save_path)
    
    return model, metrics

def predict_with_svm(model: SVMStockPredictor, data: pd.DataFrame, 
                    use_additional_features: bool = True,
                    plot: bool = True, save_plot_path: str = None) -> pd.DataFrame:
    """
    Make predictions using a trained SVM model.
    
    Args:
        model (SVMStockPredictor): Trained SVM model
        data (pd.DataFrame): DataFrame with stock data
        use_additional_features (bool): Whether to use additional technical indicators
        plot (bool): Whether to plot the predictions
        save_plot_path (str): Path to save the plot
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Make predictions
    predictions = model.predict(data, use_additional_features=use_additional_features)
    
    # Plot predictions if requested
    if plot:
        model.plot_predictions(predictions, data, save_path=save_plot_path)
    
    return predictions

def load_svm_model(filepath: str) -> SVMStockPredictor:
    """
    Load a trained SVM model from a file.
    
    Args:
        filepath (str): Path to the model file
        
    Returns:
        SVMStockPredictor: Loaded model
    """
    model = SVMStockPredictor()
    model.load_model(filepath)
    return model 