# src/model_comparison.py
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import matplotlib.dates as mdates
import io
import time  # Add time module for execution timing
from src.ml_models import StockPricePredictor
from src.xgboost_model import XGBoostStockPredictor
from src.forecasting import get_prophet_forecast
from src.marketstack_api import MarketstackAPI
from src.deep_learning import LSTMStockPredictor  # Import LSTM model
from config import MARKETSTACK_API_KEY

# Assuming ml_models and forecasting are in the same src directory or accessible
from .ml_models import StockPricePredictor # Relative import
# Import the specific forecast function needed
from .forecasting import get_prophet_forecast # Relative import
# Import the XGBoost model
from .xgboost_model import XGBoostStockPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelComparator:
    """
    A class for comparing different stock price prediction models.
    """
    
    def __init__(self):
        """
        Initialize the model comparator.
        """
        self.models = {
            'Random Forest': StockPricePredictor(),
            'XGBoost': XGBoostStockPredictor(),
            'LSTM': None,  # LSTM model will be created at runtime
            'Prophet': None  # Prophet is handled separately
        }
        self.marketstack = MarketstackAPI(api_key=MARKETSTACK_API_KEY)
        
    def _get_data(self, symbol, period='5y'):
        """
        Get stock data using Marketstack API.
        
        Args:
            symbol: Stock symbol
            period: Time period (default: '5y')
            
        Returns:
            DataFrame with stock data
        """
        try:
            logger.info(f"Fetching data from Marketstack for {symbol}")
            df = self.marketstack.get_stock_history(symbol, period=period)
            
            if df is None or df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Log the data structure to help diagnose issues
            logger.info(f"Retrieved {len(df)} rows for {symbol} with columns: {df.columns.tolist()}")
            logger.info(f"Index type: {type(df.index).__name__}")
            logger.info(f"Sample data: \n{df.head(2).to_string()}")
            
            # Ensure column names are standardized
            if 'Close' not in df.columns:
                # Try to find a close column with different capitalization
                close_cols = [col for col in df.columns if col.lower() == 'close']
                if close_cols:
                    logger.info(f"Renaming {close_cols[0]} to Close")
                    df = df.rename(columns={close_cols[0]: 'Close'})
                else:
                    logger.error(f"No Close column found in data. Available columns: {df.columns.tolist()}")
                    return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
            return None
            
    def _prepare_features(self, df):
        """
        Prepare features for model training.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with features
        """
        if len(df) < 100:
            logger.warning(f"Not enough data points ({len(df)}) to calculate indicators")
            return df
            
        # Calculate technical indicators
        try:
            # Moving averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential moving averages
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # Price momentum
            df['Close_Pct_Change'] = df['Close'].pct_change()
            df['Volume_Pct_Change'] = df['Volume'].pct_change()
            
            # Add rolling mean and std of returns
            df['Return_5d_Mean'] = df['Close_Pct_Change'].rolling(window=5).mean()
            df['Return_5d_Std'] = df['Close_Pct_Change'].rolling(window=5).std()
            
            # Log returns
            df['Log_Return'] = np.log(df['Close']/df['Close'].shift(1))
            
            logger.info("Technical indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def compare_models(self, symbol):
        """
        Compare different models for a given stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Start timing execution
            start_time = time.time()
            
            # Get data
            df = self._get_data(symbol)
            if df is None or df.empty:
                return {'error': f"No data available for {symbol}"}
                
            # Prepare features
            df = self._prepare_features(df)
            if len(df) < 100:
                return {'error': f"Insufficient data for {symbol} after preprocessing"}
                
            # Reset index for Prophet
            df_with_date = df.reset_index()
                
            # Results storage
            results = {
                'metrics': {},
                'predictions': {},
                'model_errors': {},
                'feature_importance': {}
            }
            
            # Train and evaluate Random Forest
            try:
                logger.info("Training Random Forest model")
                rf_result = self.models['Random Forest'].train(df)
                if isinstance(rf_result, tuple) and len(rf_result) >= 4:
                    rf_metrics = rf_result[0]  # Extract metrics 
                    X_test_rf = rf_result[1]   # Extract test features
                    y_test_rf = rf_result[2]   # Extract test targets
                    y_pred_rf = rf_result[3]   # Extract predictions
                else:
                    logger.warning(f"RandomForest train() returned unexpected format: {type(rf_result)}")
                    rf_metrics = rf_result if isinstance(rf_result, dict) else {}
                    X_test_rf = pd.DataFrame()
                    y_test_rf = []
                    y_pred_rf = []
                
                if not rf_metrics:
                    logger.warning("Random Forest model returned no metrics")
                    results['model_errors']['Random Forest'] = "No metrics returned from model"
                else:
                    # Ensure uppercase keys for metrics
                    results['metrics']['Random Forest'] = {
                        'RMSE': rf_metrics.get('RMSE', rf_metrics.get('rmse', 0)),
                        'MAE': rf_metrics.get('MAE', rf_metrics.get('mae', 0)),
                        'R2': rf_metrics.get('R2', rf_metrics.get('r2', 0))
                    }
                    
                    if not X_test_rf.empty and len(y_test_rf) > 0 and len(y_pred_rf) > 0:
                        results['predictions']['Random Forest'] = {
                            'actual': y_test_rf if isinstance(y_test_rf, list) else y_test_rf.tolist(),
                            'predicted': y_pred_rf if isinstance(y_pred_rf, list) else y_pred_rf.tolist(),
                            'dates': X_test_rf.index.strftime('%Y-%m-%d').tolist() if hasattr(X_test_rf.index, 'strftime') else [str(x) for x in X_test_rf.index]
                        }
                    
                    # Safely get feature importance
                    try:
                        results['feature_importance']['Random Forest'] = self.models['Random Forest'].get_feature_importance()
                    except (AttributeError, Exception) as fe:
                        logger.warning(f"Could not get RandomForest feature importance: {fe}")
                        
                    logger.info("Random Forest model trained successfully")
            except Exception as e:
                logger.error(f"Error training Random Forest model: {e}", exc_info=True)
                results['model_errors']['Random Forest'] = str(e)
            
            # Train and evaluate XGBoost
            try:
                logger.info("Training XGBoost model")
                xgb_result = self.models['XGBoost'].train(df)
                if isinstance(xgb_result, tuple) and len(xgb_result) >= 5:
                    xgb_metrics = xgb_result[0]  # Extract metrics
                    X_test_xgb = xgb_result[1]   # Extract test features
                    y_test_xgb = xgb_result[2]   # Extract test targets
                    y_pred_xgb = xgb_result[3]   # Extract predictions
                    model = xgb_result[4]        # Extract model (if returned)
                else:
                    logger.warning(f"XGBoost train() returned unexpected format: {type(xgb_result)}")
                    xgb_metrics = xgb_result if isinstance(xgb_result, dict) else {}
                    X_test_xgb = pd.DataFrame()
                    y_test_xgb = []
                    y_pred_xgb = []
                    model = None
                
                if not xgb_metrics:
                    logger.warning("XGBoost model returned no metrics")
                    results['model_errors']['XGBoost'] = "No metrics returned from model"
                else:
                    # Ensure uppercase keys for metrics
                    results['metrics']['XGBoost'] = {
                        'RMSE': xgb_metrics.get('RMSE', xgb_metrics.get('rmse', 0)),
                        'MAE': xgb_metrics.get('MAE', xgb_metrics.get('mae', 0)),
                        'R2': xgb_metrics.get('R2', xgb_metrics.get('r2', 0))
                    }
                    
                    if not X_test_xgb.empty and len(y_test_xgb) > 0 and len(y_pred_xgb) > 0:
                        results['predictions']['XGBoost'] = {
                            'actual': y_test_xgb if isinstance(y_test_xgb, list) else y_test_xgb.tolist(),
                            'predicted': y_pred_xgb if isinstance(y_pred_xgb, list) else y_pred_xgb.tolist(),
                            'dates': X_test_xgb.index.strftime('%Y-%m-%d').tolist() if hasattr(X_test_xgb.index, 'strftime') else [str(x) for x in X_test_xgb.index]
                        }
                    
                    # Safely get feature importance - handle the missing method
                    try:
                        # Try the actual method name first
                        if hasattr(self.models['XGBoost'], 'get_feature_importance'):
                            results['feature_importance']['XGBoost'] = self.models['XGBoost'].get_feature_importance()
                        # Fall back to plot_feature_importance if available
                        elif hasattr(self.models['XGBoost'], 'plot_feature_importance'):
                            # Just get the feature importances without plotting
                            # This is a workaround since the method exists but with a different name
                            feature_imp = {}
                            # Check if model has feature_importances_ attribute (XGBoost models do)
                            if hasattr(self.models['XGBoost'], 'model') and hasattr(self.models['XGBoost'].model, 'feature_importances_'):
                                # Get feature names if available
                                feature_names = getattr(self.models['XGBoost'], 'feature_names', None)
                                if feature_names is None and hasattr(X_test_xgb, 'columns'):
                                    feature_names = X_test_xgb.columns.tolist()
                                else:
                                    feature_names = [f'feature_{i}' for i in range(len(self.models['XGBoost'].model.feature_importances_))]
                                
                                # Create feature importance dict with up to top 5 features
                                top_indices = np.argsort(self.models['XGBoost'].model.feature_importances_)[-5:]
                                for idx in top_indices:
                                    if idx < len(feature_names):
                                        feature_imp[feature_names[idx]] = float(self.models['XGBoost'].model.feature_importances_[idx])
                                
                                results['feature_importance']['XGBoost'] = feature_imp
                    except (AttributeError, Exception) as fe:
                        logger.warning(f"Could not get XGBoost feature importance: {fe}")
                    
                    logger.info("XGBoost model trained successfully")
            except Exception as e:
                logger.error(f"Error training XGBoost model: {e}", exc_info=True)
                results['model_errors']['XGBoost'] = str(e)
            
            # Train and evaluate Prophet
            try:
                logger.info("Training Prophet model")
                # Pass the prepared dataframe to Prophet to use Marketstack data instead of fetching again
                prophet_result = get_prophet_forecast(symbol, periods=30, df=df_with_date)
                if 'error' in prophet_result:
                    results['model_errors']['Prophet'] = prophet_result['error']
                else:
                    # Extract metrics from prophet_result
                    prophet_metrics = prophet_result.get('metrics', {})
                    if not prophet_metrics:
                        logger.warning("Prophet model returned no metrics")
                        results['model_errors']['Prophet'] = "No metrics returned from model"
                    else:
                        # Ensure uppercase keys for metrics
                        results['metrics']['Prophet'] = {
                            'RMSE': prophet_metrics.get('RMSE', prophet_metrics.get('rmse', 0)),
                            'MAE': prophet_metrics.get('MAE', prophet_metrics.get('mae', 0)),
                            'R2': prophet_metrics.get('R2', prophet_metrics.get('r2', 0))
                        }
                    
                    # Get the forecast values
                    forecast_data = prophet_result.get('forecast_values', {})
                    if forecast_data:
                        results['predictions']['Prophet'] = {
                            'dates': forecast_data.get('dates', []),
                            'predicted': forecast_data.get('values', []),
                            'upper_bound': forecast_data.get('upper_bound', []),
                            'lower_bound': forecast_data.get('lower_bound', [])
                        }
                logger.info("Prophet model trained successfully")
            except Exception as e:
                logger.error(f"Error training Prophet model: {e}", exc_info=True)
                results['model_errors']['Prophet'] = str(e)
            
            # Train and evaluate LSTM
            try:
                logger.info("Training LSTM model")
                lstm_model = LSTMStockPredictor(sequence_length=30, batch_size=32, epochs=50)
                
                # Copy and reset index to ensure we have a proper date column
                lstm_df = df.copy().reset_index()
                
                # Train the model
                lstm_result = lstm_model.train(lstm_df, lstm_units=50, dropout_rate=0.2, architecture='simple')
                
                if not lstm_result or 'metrics' not in lstm_result:
                    logger.warning("LSTM model returned no metrics")
                    results['model_errors']['LSTM'] = "No metrics returned from model"
                else:
                    # Extract metrics
                    lstm_metrics = lstm_result.get('metrics', {})
                    
                    # Ensure uppercase keys for metrics
                    results['metrics']['LSTM'] = {
                        'RMSE': lstm_metrics.get('RMSE', lstm_metrics.get('rmse', 0)),
                        'MAE': lstm_metrics.get('MAE', lstm_metrics.get('mae', 0)),
                        'R2': lstm_metrics.get('R2', lstm_metrics.get('r2', 0))
                    }
                    
                    # Get prediction data
                    prediction_data = lstm_result.get('predictions', {})
                    if prediction_data:
                        # Process predictions to match expected format
                        pred_dates = prediction_data.get('dates', [])
                        results['predictions']['LSTM'] = {
                            'dates': pred_dates,
                            'actual': prediction_data.get('actual', []),
                            'predicted': prediction_data.get('predicted', [])
                        }
                    
                    # Get feature importance if available
                    if 'feature_importance' in lstm_result:
                        results['feature_importance']['LSTM'] = lstm_result['feature_importance']
                    
                logger.info("LSTM model trained successfully")
            except Exception as e:
                logger.error(f"Error training LSTM model: {e}", exc_info=True)
                results['model_errors']['LSTM'] = str(e)
            
            # Add best model determination based on metrics
            if results['metrics']:
                try:
                    # Find the model with the lowest RMSE (primary metric)
                    models_with_metrics = {model: metrics for model, metrics in results['metrics'].items() 
                                          if 'RMSE' in metrics and metrics['RMSE'] > 0}
                    
                    if models_with_metrics:
                        best_model = min(models_with_metrics.items(), 
                                         key=lambda x: x[1]['RMSE'])
                        results['best_model'] = best_model[0]
                        
                        # Add comparison metadata
                        results['days_compared'] = len(next(iter(results['predictions'].values()))['dates']) if results['predictions'] else 0
                        results['model_metrics'] = results['metrics']  # For template compatibility
                        logger.info(f"Best model determined: {results['best_model']}")
                    else:
                        logger.warning("No models with valid RMSE metrics found")
                except Exception as e:
                    logger.error(f"Error determining best model: {e}", exc_info=True)
            else:
                logger.warning("No metrics available to determine best model")
            
            # Generate comparison plots
            try:
                logger.info("Generating comparison plots")
                plot_path = self._generate_comparison_plot(results, symbol)
                results['plot_path'] = plot_path
                
                performance_plot_path = self._generate_performance_plot(results, symbol)
                results['performance_plot_path'] = performance_plot_path
                logger.info("Comparison plots generated successfully")
            except Exception as e:
                logger.error(f"Error generating comparison plots: {e}", exc_info=True)
                results['plot_error'] = str(e)
            
            # Calculate execution time and add to results
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            logger.info(f"Model comparison completed in {execution_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing models for {symbol}: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _generate_comparison_plot(self, results, symbol):
        """
        Generate a comparison plot for the predictions.
        
        Args:
            results: Dictionary with comparison results
            symbol: Stock symbol
            
        Returns:
            Path to the saved plot
        """
        try:
            plt.figure(figsize=(12, 8))
            
            colors = {
                'Random Forest': 'forestgreen',
                'XGBoost': 'darkorange',
                'LSTM': 'purple',  # Add color for LSTM
                'Prophet': 'royalblue'
            }
            
            # Plot each model's predictions
            for model_name, color in colors.items():
                if model_name in results['predictions']:
                    pred_data = results['predictions'][model_name]
                    dates = pred_data['dates']
                    
                    # Convert string dates to datetime
                    if isinstance(dates[0], str):
                        dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
                    
                    plt.plot(dates, pred_data['actual'], '-', color='gray', alpha=0.5)
                    plt.plot(dates, pred_data['predicted'], '-', color=color, label=f"{model_name} Predictions")
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.title(f'Model Comparison for {symbol}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            # Save plot to memory
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Save plot to disk
            plot_dir = os.path.join('static', 'images', 'analysis', 'models')
            os.makedirs(plot_dir, exist_ok=True)
            timestamp = int(datetime.now().timestamp())
            plot_path = os.path.join(plot_dir, f'comparison_{symbol}_{timestamp}.png')
            plt.savefig(plot_path, format='png', dpi=100)
            plt.close()
            
            # Convert to relative path for the template
            return plot_path.replace('static/', '')
            
        except Exception as e:
            logger.error(f"Error generating comparison plot: {e}", exc_info=True)
            return None
    
    def _generate_performance_plot(self, results, symbol):
        """
        Generate a performance comparison plot.
        
        Args:
            results: Dictionary with comparison results
            symbol: Stock symbol
            
        Returns:
            Path to the saved plot
        """
        try:
            models = [model for model in results['metrics'].keys()]
            if not models:
                return None
                
            metrics = ['RMSE', 'MAE', 'R2']
            metrics_display = ['RMSE', 'MAE', 'R²']
            colors = ['crimson', 'navy', 'forestgreen']
            
            fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                values = [results['metrics'][model].get(metric, 0) for model in models]
                
                # Special handling for R²
                if metric == 'R2':
                    values = [max(0, min(v, 1)) for v in values]  # Clip to [0, 1]
                    
                ax.bar(models, values, color=colors[i], alpha=0.7)
                ax.set_title(f"{metrics_display[i]}")
                ax.set_ylim(bottom=0)
                
                # Rotate x-axis labels
                ax.set_xticklabels(models, rotation=45, ha='right')
                
                # Add value labels on top of each bar
                for j, v in enumerate(values):
                    ax.text(j, v + 0.01, f"{v:.3f}", ha='center')
            
            plt.suptitle(f'Model Performance Comparison for {symbol}')
            plt.tight_layout()
            
            # Save plot to disk
            plot_dir = os.path.join('static', 'images', 'analysis', 'models')
            os.makedirs(plot_dir, exist_ok=True)
            timestamp = int(datetime.now().timestamp())
            plot_path = os.path.join(plot_dir, f'performance_{symbol}_{timestamp}.png')
            plt.savefig(plot_path, format='png', dpi=100)
            plt.close()
            
            # Convert to relative path for the template
            return plot_path.replace('static/', '')
            
        except Exception as e:
            logger.error(f"Error generating performance plot: {e}", exc_info=True)
            return None