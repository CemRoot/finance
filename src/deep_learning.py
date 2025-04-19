import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
import time
import json
from typing import Dict, List, Any, Tuple

# Configure TensorFlow to use memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(f"GPU memory growth setting error: {e}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMStockPredictor:
    """
    A class for training and evaluating LSTM models for stock price prediction.
    Uses TensorFlow/Keras for deep learning implementation.
    """
    def __init__(self, sequence_length=60, prediction_horizon=1, batch_size=32, 
                 epochs=100, validation_split=0.2, patience=15, verbose=1):
        """
        Initialize the LSTM stock predictor.
        
        Args:
            sequence_length (int): Number of time steps to look back for prediction
            prediction_horizon (int): Number of days to predict ahead
            batch_size (int): Training batch size
            epochs (int): Maximum number of training epochs
            validation_split (float): Fraction of training data to use for validation
            patience (int): Number of epochs with no improvement before early stopping
            verbose (int): Verbosity level for training (0=silent, 1=progress bar, 2=one line per epoch)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        
        # Attributes to be set during training
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.features = []
        self.history = None
        self.metrics = {}
        
        logger.info(f"Initialized LSTMStockPredictor (sequence_length={sequence_length}, "
                   f"prediction_horizon={prediction_horizon})")
    
    def _create_model(self, input_shape, lstm_units=50, dropout_rate=0.2, 
                     learning_rate=0.001, architecture='simple'):
        """
        Create an LSTM model with the given architecture.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features)
            lstm_units (int or list): Number of units in LSTM layer(s)
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the Adam optimizer
            architecture (str): Model architecture ('simple', 'deep', or 'bidirectional')
            
        Returns:
            model: Compiled Keras model
        """
        model = Sequential()
        
        if architecture == 'simple':
            # Simple single-layer LSTM
            model.add(LSTM(units=lstm_units if isinstance(lstm_units, int) else lstm_units[0],
                           input_shape=input_shape,
                           return_sequences=False))
            model.add(Dropout(dropout_rate))
            model.add(Dense(units=1))
            
        elif architecture == 'deep':
            # Deep stacked LSTM
            units_list = lstm_units if isinstance(lstm_units, list) else \
                         [lstm_units, lstm_units//2, lstm_units//4]
            
            # First LSTM layer
            model.add(LSTM(units=units_list[0],
                           input_shape=input_shape,
                           return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            # Middle LSTM layers
            for i in range(1, len(units_list)-1):
                model.add(LSTM(units=units_list[i], 
                               return_sequences=True))
                model.add(Dropout(dropout_rate))
                model.add(BatchNormalization())
            
            # Final LSTM layer
            model.add(LSTM(units=units_list[-1], 
                           return_sequences=False))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            # Output layer
            model.add(Dense(units=1))
            
        elif architecture == 'bidirectional':
            # Bidirectional LSTM
            from tensorflow.keras.layers import Bidirectional
            
            model.add(Bidirectional(LSTM(
                units=lstm_units if isinstance(lstm_units, int) else lstm_units[0],
                input_shape=input_shape,
                return_sequences=True
            )))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            model.add(Bidirectional(LSTM(
                units=lstm_units//2 if isinstance(lstm_units, int) else lstm_units[1],
                return_sequences=False
            )))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            model.add(Dense(units=1))
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Compile model with appropriate optimizer
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepares features from the stock data DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data (must include 'Close')
            
        Returns:
            tuple: (X, y, feature_names)
                  X (pd.DataFrame): Feature matrix
                  y (pd.Series): Target variable (future close price)
                  feature_names (list): List of feature column names
        """
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        
        # Check for MultiIndex columns and flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            logger.warning("MultiIndex columns found. Flattening...")
            try:
                df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                             for col in df.columns.values]
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
        
        # Check for required column
        if 'Close' not in df.columns:
            # Try to find a column that might contain close price
            close_candidates = [c for c in df.columns if 'close' in c.lower()]
            if close_candidates:
                logger.info(f"Using {close_candidates[0]} as Close price")
                df['Close'] = df[close_candidates[0]]
            else:
                raise ValueError(f"DataFrame must contain 'Close' column. Found: {df.columns.tolist()}")
        
        logger.info("Preparing features for LSTM model...")
        
        # Make a deep copy to avoid modifying original
        df_processed = df.copy()
        
        # Ensure Date column is datetime type if it exists and set as index
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
        
        # Get Series objects safely
        close_prices = df_processed['Close']
        if hasattr(close_prices, 'columns'):  # If it's a DataFrame
            close_prices = close_prices.iloc[:, 0]
        
        # Feature engineering for LSTM
        features_df = pd.DataFrame(index=df_processed.index)
        
        # Basic price features
        features_df['close'] = close_prices
        
        # Include other price columns if available
        for col in ['Open', 'High', 'Low']:
            if col in df_processed.columns:
                features_df[col.lower()] = df_processed[col]
        
        # Add volume if available
        if 'Volume' in df_processed.columns:
            features_df['volume'] = df_processed['Volume']
            # Log transform for volume (handles large values better)
            features_df['volume_log'] = np.log1p(df_processed['Volume'])
        
        # Technical indicators (basic)
        # Moving averages
        features_df['ma5'] = close_prices.rolling(window=5).mean()
        features_df['ma20'] = close_prices.rolling(window=20).mean()
        
        # Price momentum (percent changes)
        features_df['returns_1d'] = close_prices.pct_change(periods=1)
        features_df['returns_5d'] = close_prices.pct_change(periods=5)
        
        # Volatility
        features_df['volatility_5d'] = features_df['returns_1d'].rolling(window=5).std()
        features_df['volatility_15d'] = features_df['returns_1d'].rolling(window=15).std()
        
        # MACD components
        ema12 = close_prices.ewm(span=12, adjust=False).mean()
        ema26 = close_prices.ewm(span=26, adjust=False).mean()
        features_df['macd'] = ema12 - ema26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        
        # Drop NaN values
        features_df.dropna(inplace=True)
        
        # Store feature names
        self.features = features_df.columns.tolist()
        
        # For LSTM, we want the future Close price as target
        target = close_prices.shift(-self.prediction_horizon).loc[features_df.index]
        target = target[~target.isna()]  # Remove NaN targets
        features_df = features_df.loc[target.index]  # Align with target
        
        logger.info(f"Prepared {len(features_df)} samples with {len(self.features)} features")
        
        return features_df, target, self.features
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences for LSTM model.
        
        Args:
            X (np.ndarray): Feature matrix 
            y (np.ndarray): Target values
            
        Returns:
            tuple: (X_seq, y_seq)
                  X_seq (np.ndarray): Sequence data with shape (samples, sequence_length, features)
                  y_seq (np.ndarray): Target values
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, df: pd.DataFrame, lstm_units=50, dropout_rate=0.2, 
              learning_rate=0.001, architecture='simple') -> Dict[str, Any]:
        """
        Train the LSTM model on stock data.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data (must include 'date' and 'Close')
            lstm_units (int or list): Number of units in LSTM layer(s)
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the Adam optimizer
            architecture (str): Model architecture ('simple', 'deep', or 'bidirectional')
            
        Returns:
            Dict[str, Any]: Dictionary with training results including metrics and predictions
        """
        try:
            # Prepare features from the input DataFrame
            X, y, feature_names = self.prepare_features(df)
            
            if X.empty or len(y) == 0:
                logger.error("Failed to prepare features. Empty X or y.")
                return {'error': 'Feature preparation failed - empty dataset'}
            
            logger.info(f"Prepared features with shape X: {X.shape}, y: {len(y)}")
            
            # Scale the features and target
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
            
            # Create training sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
            
            # Split into training and validation sets
            train_size = int(len(X_seq) * (1 - self.validation_split))
            X_train, X_val = X_seq[:train_size], X_seq[train_size:]
            y_train, y_val = y_seq[:train_size], y_seq[train_size:]
            
            # Create and compile the model
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            self.model = self._create_model(
                input_shape=input_shape, 
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                architecture=architecture
            )
            
            # Set up callbacks for training
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # Train the model
            logger.info(f"Starting LSTM model training with {self.epochs} epochs, batch size {self.batch_size}")
            training_start_time = time.time()
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=self.verbose
            )
            
            training_time = time.time() - training_start_time
            logger.info(f"LSTM model training completed in {training_time:.2f} seconds")
            
            # Make predictions for evaluation
            y_pred_scaled = self.model.predict(X_val)
            
            # Inverse transform predictions and actual values
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_true = self.scaler_y.inverse_transform(y_val)
            
            # Calculate metrics
            mse = np.mean((y_pred - y_true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_pred - y_true))
            
            # Calculate R²
            ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_residual = np.sum((y_true - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            # Store metrics
            self.metrics = {
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R2': float(r2),
                'training_time': training_time
            }
            
            logger.info(f"LSTM model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            
            # Return results for model comparison
            results = {
                'metrics': self.metrics,
                'model_summary': str(self.model.summary()),
                'history': {
                    'loss': self.history.history['loss'],
                    'val_loss': self.history.history['val_loss']
                }
            }
            
            # Add predictions for visualization in the comparison
            # Get original dates for the validation set
            val_indices = range(train_size, train_size + len(y_val))
            
            # Format dates based on input DataFrame
            if isinstance(df, pd.DataFrame) and 'date' in df.columns:
                dates = df['date'].iloc[val_indices].tolist()
                # Convert to string format if needed
                dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates]
            else:
                # Generate sequential dates if not available
                dates = [str(i) for i in range(len(y_val))]
            
            # Store predictions
            results['predictions'] = {
                'dates': dates,
                'actual': y_true.flatten().tolist(),
                'predicted': y_pred.flatten().tolist()
            }
            
            # Generate plots for feature importance and learning curves
            history_plot = self.plot_history()
            if history_plot:
                results['history_plot'] = history_plot.get('image_path', '')
            
            # Generate prediction plot (last 90 days)
            pred_plot = self.plot_predictions(df, prediction_steps=30, plot_last_n=90)
            if pred_plot:
                results['prediction_plot'] = pred_plot.get('image_path', '')
            
            return results
        
        except Exception as e:
            logger.error(f"Error in LSTM training: {str(e)}", exc_info=True)
            return {'error': str(e)}
    
    def predict(self, df: pd.DataFrame, steps_ahead: int = None) -> Dict[str, Any]:
        """
        Make predictions using the trained LSTM model.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            steps_ahead (int): Number of steps to forecast into the future (None = use prediction_horizon)
            
        Returns:
            dict: Dictionary with predictions and metadata
        """
        if self.model is None:
            logger.error("No trained model available for prediction.")
            return {'error': 'Model not trained'}
        
        try:
            # Prepare features
            X, _, _ = self.prepare_features(df)
            if X.empty:
                logger.error("No data available for prediction after feature preparation.")
                return {'error': 'No data available for prediction'}
            
            # Scale features
            X_scaled = self.scaler_X.transform(X)
            
            # Use the last sequence_length data points for prediction
            last_sequence = X_scaled[-self.sequence_length:]
            
            # If we want to predict multiple steps ahead
            if steps_ahead is not None and steps_ahead > 1:
                predictions = []
                current_sequence = last_sequence.copy()
                
                # Iteratively predict future values
                for _ in range(steps_ahead):
                    # Reshape for LSTM input (samples, time steps, features)
                    pred_input = current_sequence.reshape(1, self.sequence_length, X_scaled.shape[1])
                    
                    # Predict next value
                    next_scaled = self.model.predict(pred_input)
                    next_value = self.scaler_y.inverse_transform(next_scaled)[0, 0]
                    predictions.append(next_value)
                    
                    # Update sequence for next iteration
                    # This is approximate since we're only updating the close price feature
                    # A more sophisticated approach would update all features
                    close_idx = self.features.index('close')
                    
                    # Remove first element and add predicted value to end
                    next_row = current_sequence[-1].copy()
                    next_row[close_idx] = next_scaled[0, 0]  # Use scaled value for consistency
                    current_sequence = np.vstack([current_sequence[1:], next_row])
                
                # Generate prediction dates
                last_date = X.index[-1]
                if isinstance(last_date, pd.Timestamp):
                    freq = pd.infer_freq(X.index) or 'B'  # Default to business days if no freq found
                    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                              periods=steps_ahead, freq=freq)
                else:
                    # If index is not datetime, use simple integers
                    pred_dates = range(len(X) + 1, len(X) + steps_ahead + 1)
                
                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'predicted_close': predictions
                }, index=pred_dates)
                
                return {
                    'predictions': pred_df,
                    'steps_ahead': steps_ahead,
                    'last_date': X.index[-1]
                }
                
            else:
                # Single step prediction (default)
                # Reshape for LSTM input (samples, time steps, features)
                pred_input = last_sequence.reshape(1, self.sequence_length, X_scaled.shape[1])
                
                # Make prediction
                pred_scaled = self.model.predict(pred_input)
                prediction = float(self.scaler_y.inverse_transform(pred_scaled)[0, 0])
                
                # Generate prediction date
                last_date = X.index[-1]
                if isinstance(last_date, pd.Timestamp):
                    freq = pd.infer_freq(X.index) or 'B'  # Default to business days if no freq found
                    pred_date = last_date + pd.Timedelta(days=1)
                    if freq == 'B' and pred_date.weekday() >= 5:  # Skip weekend
                        pred_date = pd.date_range(start=last_date, periods=2, freq='B')[1]
                else:
                    pred_date = len(X) + 1
                
                return {
                    'prediction': prediction,
                    'prediction_date': pred_date,
                    'last_date': last_date,
                    'last_close': float(X['close'].iloc[-1])
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {'error': str(e)}
    
    def plot_history(self) -> Dict[str, Any]:
        """
        Plot the training history (loss curves).
        
        Returns:
            dict: Dictionary with image path
        """
        if self.history is None:
            logger.error("No training history available to plot.")
            return {'error': 'No training history available'}
        
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['loss'], label='Training Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Model Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            img_dir = 'static/images'
            os.makedirs(img_dir, exist_ok=True)
            save_path = os.path.join(img_dir, f'lstm_history_{int(time.time())}.png')
            rel_path = f'/static/images/lstm_history_{int(time.time())}.png'
            
            plt.savefig(save_path)
            plt.close()
            
            return {'image_path': rel_path}
        
        except Exception as e:
            logger.error(f"Error plotting training history: {e}", exc_info=True)
            return {'error': str(e)}
    
    def plot_predictions(self, df: pd.DataFrame, prediction_steps: int = 30,
                        plot_last_n: int = 90) -> Dict[str, Any]:
        """
        Plot historical data and future predictions.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            prediction_steps (int): Number of days to predict
            plot_last_n (int): Number of historical days to plot
            
        Returns:
            dict: Dictionary with image path
        """
        if self.model is None:
            logger.error("No trained model available for predictions.")
            return {'error': 'Model not trained'}
        
        try:
            # Get historical data
            X, _, _ = self.prepare_features(df)
            if X.empty:
                return {'error': 'No data available after feature preparation'}
            
            # Use only the last n days for plotting
            historical = X['close'].iloc[-plot_last_n:] if len(X) > plot_last_n else X['close']
            
            # Make future predictions
            predictions_result = self.predict(df, steps_ahead=prediction_steps)
            if 'error' in predictions_result:
                return predictions_result
            
            predictions = predictions_result['predictions']
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(historical.index, historical.values, 'b-', label='Historical Prices')
            
            # Plot predictions
            plt.plot(predictions.index, predictions['predicted_close'].values, 'r--', label='LSTM Forecast')
            
            # Add shading to indicate prediction region
            min_y = min(min(historical.min(), predictions['predicted_close'].min()) * 0.95,
                        min(historical.min(), predictions['predicted_close'].min()) * 1.05)
            max_y = max(max(historical.max(), predictions['predicted_close'].max()) * 0.95,
                        max(historical.max(), predictions['predicted_close'].max()) * 1.05)
            
            plt.axvspan(predictions.index[0], predictions.index[-1], alpha=0.2, color='gray')
            
            # Add labels and title
            plt.title(f'LSTM Stock Price Prediction: Historical vs Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Set y-axis limits to ensure consistent visualization
            plt.ylim(min_y, max_y)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save the figure
            img_dir = 'static/images'
            os.makedirs(img_dir, exist_ok=True)
            save_path = os.path.join(img_dir, f'lstm_predictions_{int(time.time())}.png')
            rel_path = f'/static/images/lstm_predictions_{int(time.time())}.png'
            
            plt.savefig(save_path)
            plt.close()
            
            return {
                'image_path': rel_path,
                'historical_dates': historical.index.tolist(),
                'historical_values': historical.values.tolist(),
                'prediction_dates': predictions.index.tolist(),
                'prediction_values': predictions['predicted_close'].values.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {e}", exc_info=True)
            return {'error': str(e)}
    
    def save_model(self, path: str = None) -> Dict[str, Any]:
        """
        Save the trained LSTM model and associated data.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            dict: Status of the save operation
        """
        if self.model is None:
            logger.error("No trained model available to save.")
            return {'error': 'No trained model available'}
        
        try:
            # Create default path if not provided
            if path is None:
                model_dir = 'models'
                os.makedirs(model_dir, exist_ok=True)
                path = os.path.join(model_dir, f'lstm_model_{int(time.time())}')
            
            # Create directory if not exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save Keras model
            model_path = f"{path}.h5"
            self.model.save(model_path)
            
            # Save metadata (scaler, features, etc.)
            metadata = {
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'features': self.features,
                'metrics': self.metrics,
                'batch_size': self.batch_size,
                'validation_split': self.validation_split,
                'epochs': self.epochs,
                'scaler_X_mean': self.scaler_X.mean_.tolist() if hasattr(self.scaler_X, 'mean_') else None,
                'scaler_X_scale': self.scaler_X.scale_.tolist() if hasattr(self.scaler_X, 'scale_') else None,
                'scaler_y_mean': self.scaler_y.mean_.tolist() if hasattr(self.scaler_y, 'mean_') else None,
                'scaler_y_scale': self.scaler_y.scale_.tolist() if hasattr(self.scaler_y, 'scale_') else None
            }
            
            metadata_path = f"{path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            return {
                'model_path': model_path,
                'metadata_path': metadata_path,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            return {'error': str(e)}
    
    def load_model(self, model_path: str, metadata_path: str = None) -> Dict[str, Any]:
        """
        Load a saved LSTM model and associated data.
        
        Args:
            model_path (str): Path to the saved model file
            metadata_path (str): Path to the saved metadata file (will try to infer if None)
            
        Returns:
            dict: Status of the load operation
        """
        try:
            # Try to infer metadata path if not provided
            if metadata_path is None:
                base_path = model_path.replace('.h5', '')
                metadata_path = f"{base_path}_metadata.json"
            
            # Load Keras model
            self.model = load_model(model_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Restore metadata
            self.sequence_length = metadata['sequence_length']
            self.prediction_horizon = metadata['prediction_horizon']
            self.features = metadata['features']
            self.metrics = metadata['metrics']
            self.batch_size = metadata.get('batch_size', 32)
            self.validation_split = metadata.get('validation_split', 0.2)
            self.epochs = metadata.get('epochs', 100)
            
            # Restore scalers if available
            if metadata.get('scaler_X_mean') is not None and metadata.get('scaler_X_scale') is not None:
                self.scaler_X = MinMaxScaler()
                self.scaler_X.mean_ = np.array(metadata['scaler_X_mean'])
                self.scaler_X.scale_ = np.array(metadata['scaler_X_scale'])
                
            if metadata.get('scaler_y_mean') is not None and metadata.get('scaler_y_scale') is not None:
                self.scaler_y = MinMaxScaler()
                self.scaler_y.mean_ = np.array(metadata['scaler_y_mean'])
                self.scaler_y.scale_ = np.array(metadata['scaler_y_scale'])
            
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Metadata loaded from {metadata_path}")
            
            return {'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return {'error': str(e)} 