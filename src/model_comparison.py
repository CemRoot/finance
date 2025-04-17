# src/model_comparison.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import logging
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from typing import Callable, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Assuming ml_models and forecasting are in the same src directory or accessible
from .ml_models import StockPricePredictor # Relative import
# Import the specific forecast function needed
from .forecasting import get_prophet_forecast # Relative import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelComparator:
    """
    Compares the performance of different time series forecasting models,
    specifically Prophet and a Random Forest Regressor.
    """
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {} # Stores trained models/results
        self.results: Dict[str, Any] = {}           # Stores comparison results
        self.df_data: Optional[pd.DataFrame] = None # Stores fetched data for reuse
        self.symbol: Optional[str] = None           # Stores the stock symbol

    def _fetch_data(self, symbol: str, period: str = '5y') -> Optional[pd.DataFrame]:
        """Fetches and caches stock data using yfinance."""
        # Reuse cached data if available for the same symbol
        if self.df_data is not None and self.symbol == symbol:
             logger.info(f"Using cached data for {symbol}")
             return self.df_data

        logger.info(f"Fetching data for {symbol} (period: {period})")
        try:
            # Download data with yfinance
            df = yf.download(symbol, period=period)
            
            if df.empty:
                logger.error(f"No historical data returned for {symbol} (period={period}).")
                return None
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            logger.info(f"Downloaded data columns: {df.columns}")
            logger.info(f"Data types: {df.dtypes}")
            
            # Handle MultiIndex columns that may occur with some symbols
            if isinstance(df.columns, pd.MultiIndex):
                logger.info(f"MultiIndex columns detected: {df.columns}")
                
                # Create a new flat DataFrame with proper column names
                new_columns = []
                for col in df.columns:
                    if col[0] == 'Date' or col[0] == 'Datetime':
                        new_columns.append('Date')
                    elif col[0] == symbol:
                        # If first level is the symbol, use the second level
                        new_columns.append(col[1])
                    else:
                        # Otherwise use the first level
                        new_columns.append(col[0])
                
                logger.info(f"Flattened columns: {new_columns}")
                df.columns = new_columns
            
            # Ensure Date is datetime type
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Remove timezone if present
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            # Drop rows with missing Close prices
            before_dropna = len(df)
            df.dropna(subset=['Close'], inplace=True)
            after_dropna = len(df)
            if before_dropna > after_dropna:
                logger.info(f"Dropped {before_dropna - after_dropna} rows with missing Close values")
            
            # Sort by date for time series consistency
            df.sort_values('Date', inplace=True)
            
            # Final quality check
            logger.info(f"Successfully fetched and prepared {len(df)} data points for {symbol}")
            if len(df) < 100:
                logger.warning(f"Only {len(df)} data points available for {symbol}, which may be insufficient")
            
            # Cache the data
            self.df_data = df
            self.symbol = symbol
            return self.df_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
            self.df_data = None
            self.symbol = None
            return None

    def fetch_data(self, symbol: str, period: str = '5y') -> Optional[pd.DataFrame]:
        """Public method to fetch and cache stock data."""
        return self._fetch_data(symbol, period)

    def _add_random_forest(self, df: pd.DataFrame, name: str = "Random Forest") -> Dict[str, Any]:
        """Adds and trains the Random Forest model."""
        logger.info(f"Initializing and training Random Forest model: {name}")
        model = StockPricePredictor(model_type='regressor') # Ensure regressor type
        # Pass the DataFrame with 'Date' column for feature prep
        # train returns: metrics, X_test, y_test, y_pred, y_pred_proba (None for regressor)
        results = model.train(df.copy(), auto_optimize=True) # Use copy to avoid modifying original df
        metrics = results[0]
        test_data = {'X_test': results[1], 'y_test': results[2], 'y_pred': results[3]}

        self.models[name] = {
            'instance': model,
            'type': 'random_forest',
            'metrics': metrics,
            'test_data': test_data
        }
        logger.info(f"Random Forest model '{name}' added. Metrics: {metrics}")
        return metrics

    def _add_prophet(self, symbol: str, df: pd.DataFrame = None, name: str = "Prophet") -> Dict[str, Any]:
        """Adds and runs the Prophet model using the forecasting function."""
        logger.info(f"Running Prophet forecast for {symbol}: {name}")
        try:
            # Call the imported forecast function with historical=True to get predictions for past data
            forecast_data = get_prophet_forecast(symbol, historical=True, periods=30, df=df)

            if forecast_data.get('error'):
                error_msg = forecast_data['error']
                logger.error(f"Prophet forecast failed for {symbol}: {error_msg}")
                self.models[name] = {'type': 'prophet', 'error': error_msg, 'metrics': {}}
                return {'error': error_msg}

            metrics = forecast_data.get('metrics', {})
            self.models[name] = {
                'type': 'prophet',
                'forecast_data': forecast_data,
                'metrics': metrics,
                'comparison_metrics': {}
            }
            logger.info(f"Prophet model '{name}' added. Internal Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error running or adding Prophet model for {symbol}: {e}", exc_info=True)
            self.models[name] = {'type': 'prophet', 'error': str(e), 'metrics': {}}
            return {'error': str(e)}

    def _align_predictions_for_test_period(self, days: int = 30) -> Optional[Tuple[pd.DatetimeIndex, pd.Series, pd.Series, pd.Series]]:
        """
        Aligns RF test predictions with Prophet predictions for the same time period.
        Returns aligned test dates, actual values, RF predictions, and Prophet predictions.
        """
        rf_model_data = self.models.get("Random Forest")
        prophet_model_data = self.models.get("Prophet")

        if not rf_model_data or not prophet_model_data:
            logger.error("Cannot align predictions: Missing models.")
            return None
        
        if rf_model_data.get('error') or prophet_model_data.get('error'):
            logger.error("Cannot align predictions: One or both models had errors.")
            return None
        
        if 'test_data' not in rf_model_data:
            logger.error("Random Forest test data not available.")
            return None

        try:
            # Get Random Forest test data
            X_test = rf_model_data['test_data']['X_test']
            y_test = rf_model_data['test_data']['y_test']
            y_pred_rf = pd.Series(rf_model_data['test_data']['y_pred'], index=y_test.index)
            
            # Limit to last 'days' days if specified
            if days and len(y_test) > days:
                y_test = y_test.iloc[-days:]
                y_pred_rf = y_pred_rf.iloc[-days:]
                
            # Get Prophet predictions
            forecast_data = prophet_model_data['forecast_data']
            
            # Get historical predictions (should include the test period)
            if 'historical_forecast' in forecast_data:
                # Create prophet DataFrame with dates as index
                prophet_df = pd.DataFrame({
                    'ds': pd.to_datetime(forecast_data['historical_forecast']['dates']),
                    'yhat': forecast_data['historical_forecast']['values']
                })
                prophet_df.set_index('ds', inplace=True)
                
                # Convert y_test index to datetime if not already a DatetimeIndex
                if not isinstance(y_test.index, pd.DatetimeIndex):
                    logger.info(f"Converting y_test index to datetime. Current type: {type(y_test.index)}")
                    try:
                        # Try to get dates from the X_test dataframe and original data
                        if self.df_data is not None and 'Date' in self.df_data.columns:
                            # Get dates from the original data using indices from X_test
                            date_mapping = dict(zip(self.df_data.index, self.df_data['Date']))
                            date_values = [date_mapping.get(idx) for idx in X_test.index]
                            date_index = pd.DatetimeIndex(date_values)
                            y_test.index = date_index
                            y_pred_rf.index = date_index
                            logger.info(f"Successfully mapped dates from original data. Sample dates: {y_test.index[:5]}")
                        else:
                            # Fallback: Try to convert numeric index to dates by assuming they're sequential
                            # This is less reliable but better than nothing
                            logger.warning("No Date column available, attempting fallback date conversion")
                            prophet_dates = prophet_df.index.sort_values()
                            if len(prophet_dates) >= len(y_test):
                                # Use the last len(y_test) dates from prophet
                                date_index = prophet_dates[-len(y_test):]
                                y_test.index = date_index
                                y_pred_rf.index = date_index
                                logger.info(f"Used sequential prophet dates as fallback. Sample: {y_test.index[:5]}")
                            else:
                                logger.error("Insufficient prophet dates for fallback conversion")
                                return None
                    except Exception as e:
                        logger.error(f"Failed to convert y_test index to DatetimeIndex: {e}", exc_info=True)
                        return None

                # Ensure both have DatetimeIndex
                if not isinstance(y_test.index, pd.DatetimeIndex) or not isinstance(prophet_df.index, pd.DatetimeIndex):
                    logger.error(f"Index types not compatible: y_test={type(y_test.index)}, prophet={type(prophet_df.index)}")
                    return None

                # Align the indices - find common dates between RF test and Prophet predictions
                common_dates = y_test.index.intersection(prophet_df.index)
                
                if len(common_dates) == 0:
                    logger.error("No common dates between RF test set and Prophet predictions.")
                    return None
                    
                y_test_aligned = y_test.loc[common_dates]
                y_pred_rf_aligned = y_pred_rf.loc[common_dates]
                y_pred_prophet_aligned = prophet_df.loc[common_dates, 'yhat']
                
                if len(y_test_aligned) < 5:  # Require at least 5 common points for comparison
                    logger.error(f"Insufficient overlapping data points: {len(y_test_aligned)}, need at least 5")
                    return None
                    
                logger.info(f"Successfully aligned {len(y_test_aligned)} data points for comparison")
                logger.info(f"Sample dates: {common_dates[:5]}")
                return common_dates, y_test_aligned, y_pred_rf_aligned, y_pred_prophet_aligned
            else:
                logger.error("No historical forecast data available from Prophet.")
                return None
                
        except Exception as e:
            logger.error(f"Error aligning predictions: {e}", exc_info=True)
            return None

    def compare_models(self, symbol: str, df: Optional[pd.DataFrame] = None, days_to_compare: int = 30) -> Dict[str, Any]:
        """Compares Prophet and Random Forest predictions against actual values."""
        self.symbol = symbol # Store symbol for later use
        if df is None:
            df = self._fetch_data(symbol, period='5y')
            if df is None: return {'error': f'Could not fetch data for {symbol}'}
        elif self.df_data is None: # Store provided data
             self.df_data = df

        logger.info(f"Comparing prediction models for {symbol} over last {days_to_compare} days.")

        try:
            # Ensure both models are loaded/trained
            if "Random Forest" not in self.models: 
                self._add_random_forest(df)
            
            # Only add Prophet model after Random Forest so we know the test period
            if "Prophet" not in self.models: 
                self._add_prophet(symbol, df)

            # Check for errors during model addition
            if self.models.get("Random Forest", {}).get('error') or self.models.get("Prophet", {}).get('error'):
                 return {'error': f"One or more models failed to initialize for {symbol}.",
                         'rf_error': self.models.get("Random Forest", {}).get('error'),
                         'prophet_error': self.models.get("Prophet", {}).get('error')}

            # Align predictions
            aligned_data = self._align_predictions_for_test_period(days=days_to_compare)
            if aligned_data is None:
                 return {'error': f"Could not align predictions for {symbol}."}

            dates, y_test, y_pred_rf, y_pred_prophet = aligned_data

            # Calculate comparison metrics
            try:
                rf_mae = mean_absolute_error(y_test, y_pred_rf)
                rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
                rf_r2 = r2_score(y_test, y_pred_rf)

                prophet_mae = mean_absolute_error(y_test, y_pred_prophet)
                prophet_rmse = np.sqrt(mean_squared_error(y_test, y_pred_prophet))
                prophet_r2 = r2_score(y_test, y_pred_prophet)

                # Store these comparison-specific metrics separately
                self.models['Random Forest']['comparison_metrics'] = {'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2}
                self.models['Prophet']['comparison_metrics'] = {'MAE': prophet_mae, 'RMSE': prophet_rmse, 'R2': prophet_r2}

                # Calculate differences (Prophet - RF)
                comparison_metrics = {
                    'Random Forest': self.models['Random Forest']['comparison_metrics'],
                    'Prophet': self.models['Prophet']['comparison_metrics'],
                    'Difference': {
                        'MAE': prophet_mae - rf_mae,
                        'RMSE': prophet_rmse - rf_rmse,
                        'R2': prophet_r2 - rf_r2
                    }
                }
                self.results['comparison_metrics'] = comparison_metrics
                self.results['aligned_data'] = {
                    'dates': dates,
                    'y_test': y_test,
                    'y_pred_rf': y_pred_rf,
                    'y_pred_prophet': y_pred_prophet
                }
                logger.info(f"Model comparison metrics calculated for {symbol}")
                return comparison_metrics
                
            except Exception as metric_err:
                 logger.error(f"Error calculating comparison metrics: {metric_err}", exc_info=True)
                 return {'error': f"Error calculating comparison metrics: {str(metric_err)}"}
                 
        except Exception as e:
            logger.error(f"Error in compare_models: {e}", exc_info=True)
            return {'error': f"Model comparison failed: {str(e)}"}

    def plot_comparison(self, days_to_plot: int = 30) -> Dict[str, Any]:
        """
        Creates a plot comparing the actual values with predictions from both models.
        """
        if not self.results.get('aligned_data'):
            logger.error("No aligned data available for plotting. Run compare_models first.")
            return {'error': "No comparison data available. Run compare_models first."}

        # Ensure output directory exists
        os.makedirs('static/images', exist_ok=True)
        fig_path = f'static/images/model_comparison_{int(time.time())}.png'

        try:
            # Get data from the alignment
            aligned_data = self.results['aligned_data']
            dates = aligned_data['dates']
            y_test = aligned_data['y_test']
            y_pred_rf = aligned_data['y_pred_rf']
            y_pred_prophet = aligned_data['y_pred_prophet']

            # Limit to the specified days if needed
            if days_to_plot and len(dates) > days_to_plot:
                # Take the most recent days_to_plot days
                dates = dates[-days_to_plot:]
                y_test = y_test.loc[dates]
                y_pred_rf = y_pred_rf.loc[dates]
                y_pred_prophet = y_pred_prophet.loc[dates]

            # Create the comparison plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates, y_test, 'o-', label='Actual', color='black', alpha=0.7)
            plt.plot(dates, y_pred_rf, 's-', label='Random Forest', color='blue', alpha=0.7)
            plt.plot(dates, y_pred_prophet, '^-', label='Prophet', color='red', alpha=0.7)
            
            # Format x-axis to show dates clearly
            plt.gcf().autofmt_xdate()
            
            plt.title(f'{self.symbol} Price Prediction Comparison')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(fig_path)
            plt.close()
            
            return {
                'image_path': fig_path,
                'symbol': self.symbol,
                'dates': [d.strftime('%Y-%m-%d') for d in dates]
            }
            
        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}", exc_info=True)
            return {'error': f"Failed to create comparison plot: {str(e)}"}

    def plot_performance_comparison(self) -> Dict[str, Any]:
        """
        Creates a bar chart comparing the performance metrics of both models.
        """
        if not self.results.get('comparison_metrics'):
            logger.error("No comparison metrics available for plotting. Run compare_models first.")
            return {'error': "No comparison metrics available. Run compare_models first."}

        # Ensure output directory exists
        os.makedirs('static/images', exist_ok=True)
        fig_path = f'static/images/performance_comparison_{int(time.time())}.png'

        try:
            metrics = self.results['comparison_metrics']
            
            # Extract metrics for plotting
            rf_metrics = metrics['Random Forest']
            prophet_metrics = metrics['Prophet']
            
            metrics_to_plot = ['MAE', 'RMSE']  # Skip R2 in bar chart as it can be negative
            
            # Create figure with 2 subplots: bar chart for MAE/RMSE and separate for R2
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
            
            # Bar chart for MAE and RMSE
            x = np.arange(len(metrics_to_plot))
            width = 0.35
            
            rf_values = [rf_metrics[m] for m in metrics_to_plot]
            prophet_values = [prophet_metrics[m] for m in metrics_to_plot]
            
            ax1.bar(x - width/2, rf_values, width, label='Random Forest', color='blue', alpha=0.7)
            ax1.bar(x + width/2, prophet_values, width, label='Prophet', color='red', alpha=0.7)
            
            # Add value annotations above bars
            for i, v in enumerate(rf_values):
                ax1.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
            for i, v in enumerate(prophet_values):
                ax1.text(i + width/2, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics_to_plot)
            ax1.set_ylabel('Value (lower is better)')
            ax1.set_title('Error Metrics Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Separate plot for R2 (can be negative)
            x2 = ['R²']
            rf_r2 = [rf_metrics['R2']]
            prophet_r2 = [prophet_metrics['R2']]
            
            ax2.bar([0 - width/2], rf_r2, width, label='Random Forest', color='blue', alpha=0.7)
            ax2.bar([0 + width/2], prophet_r2, width, label='Prophet', color='red', alpha=0.7)
            
            # Add value annotations
            ax2.text(0 - width/2, rf_r2[0] + 0.02, f'{rf_r2[0]:.2f}', ha='center', fontsize=9)
            ax2.text(0 + width/2, prophet_r2[0] + 0.02, f'{prophet_r2[0]:.2f}', ha='center', fontsize=9)
            
            ax2.set_xticks([0])
            ax2.set_xticklabels(x2)
            ax2.set_ylabel('Value (higher is better)')
            ax2.set_title('R² Score Comparison')
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal line at y=0 for R² plot
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Adjust the y-range to ensure all bars and annotations are visible
            ax1.set_ylim(0, max(max(rf_values), max(prophet_values)) * 1.2)  # 20% margin
            r2_max = max(max(rf_r2), max(prophet_r2), 0.1)  # At least 0.1 for visibility
            r2_min = min(min(rf_r2), min(prophet_r2), 0)
            ax2.set_ylim(r2_min - 0.2, r2_max + 0.2)  # Add margin
            
            plt.suptitle(f'{self.symbol} Model Performance Comparison', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save the figure
            plt.savefig(fig_path)
            plt.close()
            
            return {
                'image_path': fig_path,
                'metrics': metrics,
                'symbol': self.symbol
            }
            
        except Exception as e:
            logger.error(f"Error creating performance comparison plot: {e}", exc_info=True)
            return {'error': f"Failed to create performance plot: {str(e)}"}