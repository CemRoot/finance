# src/forecasting.py
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import time
# Import rate limit function and constant from newsapi module
from .newsapi import _rate_limit, YFINANCE_MIN_INTERVAL # Relative import
import logging
import requests # Import for handling HTTP errors specifically

# Suppress verbose INFO logs from Prophet and cmdstanpy unless debugging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def get_prophet_forecast(stock_symbol, periods=30):
    """
    Generates stock price forecast using Facebook Prophet library.
    Includes rate limiting and enhanced error handling.

    Args:
        stock_symbol (str): The stock symbol (e.g., 'AAPL').
        periods (int): Number of days into the future to forecast.

    Returns:
        dict: A dictionary containing forecast results or an error message.
              On success: {'forecast_values': {...}, 'metrics': {...}, 'last_actual_...': ..., 'error': None}
              On failure: {'error': 'Error message string'}
    """
    try:
        logger.info(f"Starting Prophet forecast for {stock_symbol} ({periods} periods)")

        # --- Data Fetching ---
        ticker = yf.Ticker(stock_symbol)
        history = None
        # Try fetching sufficient history (Prophet prefers more data)
        for period in ["2y", "1y", "6mo"]: # Try longer periods first
            try:
                logger.info(f"Applying rate limit before fetching {period} history for Prophet ({stock_symbol})")
                _rate_limit('yfinance') # Apply rate limiting
                logger.info(f"Fetching {period} history for Prophet ({stock_symbol})")
                # Use '1d' interval for Prophet, auto_adjust=True is generally recommended
                history = ticker.history(period=period, interval="1d", auto_adjust=True, repair=True, timeout=30)

                if history is not None and not history.empty and len(history) >= 20: # Need at least ~20 points for Prophet
                    logger.info(f"Fetched {len(history)} data points ({period}) for Prophet ({stock_symbol}).")
                    break # Stop fetching if sufficient data is obtained
                else:
                     logger.warning(f"Insufficient data ({len(history) if history is not None else 0} rows) fetched for period '{period}' for Prophet ({stock_symbol}). Trying next period.")
                     history = None # Reset history if insufficient
            except requests.exceptions.HTTPError as http_err_hist:
                 # Handle HTTP errors during history fetch specifically
                 status_code = http_err_hist.response.status_code
                 logger.error(f"Prophet history fetch ({period}) failed for {stock_symbol} due to yfinance HTTP Error {status_code}: {http_err_hist}")
                 if status_code == 404: # Not found is critical
                     return {'error': f"Symbol '{stock_symbol}' not found on Yahoo Finance (HTTP 404)."}
                 # Other HTTP errors might be transient, allow loop to continue
            except Exception as e_hist:
                logger.error(f"Error fetching {period} history for Prophet ({stock_symbol}): {type(e_hist).__name__} - {e_hist}")
                history = None # Ensure history is None on error

            # Small delay if fetch failed before trying next period
            if history is None or history.empty: time.sleep(0.5)

        # If no history could be fetched after all attempts
        if history is None or history.empty:
            logger.error(f"Prophet forecast failed: Could not fetch sufficient historical data for {stock_symbol} after trying multiple periods.")
            return {'error': f"Could not retrieve enough historical data (at least ~20 days needed) for '{stock_symbol}' to make a forecast."}

        # --- Data Preparation ---
        logger.info(f"Preparing data for Prophet model ({stock_symbol}).")

        # Handle potential MultiIndex columns from yf.download (less likely with ticker.history)
        if isinstance(history.columns, pd.MultiIndex):
            logger.warning(f"MultiIndex detected in history for {stock_symbol}. Flattening columns.")
            # Flatten MultiIndex columns: ('Close', 'AAPL') -> 'Close_AAPL'
            # history.columns = ['_'.join(col).strip() for col in history.columns.values]
            # Or, more simply, if auto_adjust=True, just select standard columns
            try:
                history = history[['Open', 'High', 'Low', 'Close', 'Volume']]
            except KeyError as multi_err:
                logger.error(f"Could not select standard OHLCV columns from MultiIndex for {stock_symbol}: {multi_err}. Columns: {history.columns}")
                return {'error': f"Unexpected data column structure for '{stock_symbol}'. Cannot process."}


        # Ensure index is DatetimeIndex
        if not isinstance(history.index, pd.DatetimeIndex):
             history.index = pd.to_datetime(history.index, errors='coerce')
             history = history[pd.notna(history.index)] # Drop rows where index conversion failed

        # Prepare DataFrame for Prophet (ds, y)
        df = history.reset_index() # Get the date index as a column
        # Find the date column (could be 'Date', 'Datetime', or the reset index name)
        date_col_name = None
        if 'Date' in df.columns: date_col_name = 'Date'
        elif 'Datetime' in df.columns: date_col_name = 'Datetime'
        elif 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']): date_col_name = 'index'
        else: # Check index name if it was reset
             potential_date_col = df.columns[0] # Assume first column might be date after reset
             if pd.api.types.is_datetime64_any_dtype(df[potential_date_col]):
                  date_col_name = potential_date_col

        if not date_col_name:
            logger.error(f"Prophet forecast failed: Could not identify the date column for {stock_symbol}. Columns: {df.columns}")
            return {'error': f"Date column not found in historical data for '{stock_symbol}'."}

        # Rename columns for Prophet
        df = df.rename(columns={date_col_name: 'ds', 'Close': 'y'})

        # Ensure 'ds' is datetime and timezone-naive (Prophet requires this)
        try:
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        except Exception as date_err:
            logger.error(f"Error converting 'ds' column to timezone-naive datetime for {stock_symbol}: {date_err}")
            return {'error': f"Error processing date format for '{stock_symbol}'."}

        # Ensure 'y' (Close price) is numeric, coercing errors to NaN
        df['y'] = pd.to_numeric(df['y'], errors='coerce')

        # Handle NaNs in 'y' (fill forward then backward) - crucial for Prophet
        df['y'] = df['y'].ffill().bfill()

        # Check if NaNs remain in 'y' after filling
        if df['y'].isnull().any():
            logger.error(f"Prophet forecast failed: 'y' column contains NaNs after fill for {stock_symbol}")
            return {'error': f"Price data for '{stock_symbol}' contains gaps that could not be filled."}

        # Final check for sufficient data length after cleaning
        if len(df) < 5: # Prophet needs at least 2, but more is better
             logger.error(f"Prophet forecast failed: Insufficient data ({len(df)} rows) after cleaning for {stock_symbol}")
             return {'error': f"Too few valid data points remain for '{stock_symbol}' to make a forecast (need at least 5)."}
        elif len(df) < 20:
              logger.warning(f"Prophet forecast warning: Data for {stock_symbol} is short ({len(df)} rows). Forecast accuracy might be low.")

        # Select only necessary columns for fitting
        df_fit = df[['ds', 'y']].copy()


        # --- Model Training ---
        logger.info(f"Initializing and fitting Prophet model for {stock_symbol}...")
        # Initialize Prophet model with potentially adjusted parameters
        m = Prophet(
            yearly_seasonality='auto', # Auto-detect yearly patterns
            weekly_seasonality='auto', # Auto-detect weekly patterns
            daily_seasonality=False, # Daily patterns usually not relevant for '1d' interval stocks
            seasonality_mode='multiplicative', # Assumes seasonality scales with trend
            # changepoint_prior_scale=0.05, # Default flexibility for trend changes
            # holidays=... # Consider adding relevant market holidays if needed
        )

        # Optional: Add regressors (e.g., Volume) - Requires careful handling
        # if 'Volume' in df.columns and df['Volume'].nunique() > 1:
        #     df_fit['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        #     # Avoid log(0) issues
        #     df_fit['log_volume'] = np.log1p(df_fit['Volume'])
        #     # Check for NaN/Inf introduced by log
        #     if not df_fit['log_volume'].isnull().any() and not np.isinf(df_fit['log_volume']).any():
        #         try:
        #             m.add_regressor('log_volume')
        #             logger.info(f"Added 'log_volume' as regressor for {stock_symbol}")
        #         except Exception as reg_err:
        #              logger.warning(f"Could not add 'log_volume' regressor for {stock_symbol}: {reg_err}")
        #     else:
        #         logger.warning(f"NaN or Inf found in log_volume for {stock_symbol}, regressor not added.")
        #         df_fit.drop(columns=['log_volume'], inplace=True) # Remove column if not used
        # else: logger.info(f"Volume data not suitable as regressor for {stock_symbol}.")


        # Fit the model
        try:
             m.fit(df_fit) # Fit with prepared ds, y columns (and optional regressors)
        except Exception as fit_err:
             logger.error(f"Prophet model fitting failed for {stock_symbol}: {fit_err}", exc_info=True)
             return {'error': f"Failed to train the forecast model for '{stock_symbol}': {fit_err}"}
        logger.info(f"Prophet model fitted for {stock_symbol}.")

        # --- Prediction ---
        # Create future DataFrame for prediction
        future = m.make_future_dataframe(periods=periods, freq='D') # Use 'D' for daily frequency

        # Add regressor values to future dataframe if used
        # if 'log_volume' in m.extra_regressors:
        #      # Simple approach: use the mean or last value of the regressor for future dates
        #      last_log_volume = df_fit['log_volume'].iloc[-1] if 'log_volume' in df_fit and not df_fit['log_volume'].empty else 0
        #      future['log_volume'] = float(last_log_volume) if pd.notna(last_log_volume) else 0.0


        # Make prediction
        logger.info(f"Making future predictions for {stock_symbol}...")
        try:
            forecast = m.predict(future)
        except Exception as pred_err:
             logger.error(f"Prophet prediction failed for {stock_symbol}: {pred_err}", exc_info=True)
             return {'error': f"Failed to generate forecast predictions for '{stock_symbol}': {pred_err}"}
        logger.info(f"Predictions generated for {stock_symbol}.")

        # --- Metrics Calculation ---
        logger.info(f"Calculating forecast metrics for {stock_symbol}...")
        metrics = {}
        try:
             # Trend Direction & Strength (using forecast 'trend' component)
            trend_direction = 'Uncertain'
            trend_strength = 0.0
            trend_series = forecast['trend']
            if len(trend_series) > 7:
                # Compare last value to value 7 days prior within the forecast range
                trend_diff = trend_series.iloc[-1] - trend_series.iloc[-8] # Difference over last 7 days
                last_trend_val = abs(trend_series.iloc[-1]) # Use absolute value as base
                if last_trend_val > 1e-6 and pd.notna(trend_diff): # Avoid division by zero or NaN
                    trend_strength = abs(trend_diff / last_trend_val) # Relative strength
                    if trend_diff > (0.001 * last_trend_val): trend_direction = 'Upward'
                    elif trend_diff < (-0.001 * last_trend_val): trend_direction = 'Downward'
                    else: trend_direction = 'Sideways'
            metrics['trend_direction'] = trend_direction
            metrics['trend_strength'] = float(trend_strength) if pd.notna(trend_strength) else 0.0

            # Seasonality Strength (relative to trend magnitude)
            seasonality_strength = 0.0
            total_seasonality_amplitude = 0
            component_count = 0
            for component in ['weekly', 'yearly']:
                 if component in forecast.columns and forecast[component].nunique() > 1:
                      comp_mean_abs = np.mean(np.abs(forecast[component]))
                      total_seasonality_amplitude += comp_mean_abs
                      component_count += 1
            if component_count > 0:
                 trend_mean_abs = np.mean(np.abs(forecast['trend']))
                 if trend_mean_abs > 1e-6:
                     seasonality_strength = total_seasonality_amplitude / trend_mean_abs
            metrics['seasonality_strength'] = float(seasonality_strength) if pd.notna(seasonality_strength) else 0.0

            # Historical Volatility (using original cleaned data)
            historical_volatility = 0.0
            if len(df) >= 5: # Check original df length
                 # Calculate returns on the original 'y' column (cleaned close prices)
                 df['returns'] = df['y'].pct_change()
                 # Use standard deviation of returns over a window (e.g., last 30 trading days)
                 window_size = min(30, len(df)-1) # Ensure window fits data
                 if window_size >= 2:
                     rolling_std = df['returns'].rolling(window=window_size).std()
                     last_valid_std = rolling_std.dropna().iloc[-1] if not rolling_std.dropna().empty else None
                     if last_valid_std is not None:
                          # Annualize the standard deviation
                          historical_volatility = last_valid_std * np.sqrt(252) # Approx trading days in a year
            metrics['historical_volatility'] = float(historical_volatility) if pd.notna(historical_volatility) else None # Keep as None if not calculated

            # Confidence Interval Width (relative to the last predicted price 'yhat')
            confidence_interval_relative = 0.0
            last_forecast_point = forecast.iloc[-1]
            last_yhat = last_forecast_point.get('yhat')
            last_upper = last_forecast_point.get('yhat_upper')
            last_lower = last_forecast_point.get('yhat_lower')
            # Check if all values are valid numbers and yhat is not zero
            if all(pd.notna(v) for v in [last_yhat, last_upper, last_lower]) and abs(last_yhat) > 1e-6:
                 confidence_interval_width = last_upper - last_lower
                 confidence_interval_relative = confidence_interval_width / abs(last_yhat) # Relative width
            metrics['confidence_interval'] = float(confidence_interval_relative) if pd.notna(confidence_interval_relative) else 0.0

        except Exception as metrics_err:
            logger.error(f"Error calculating forecast metrics for {stock_symbol}: {metrics_err}", exc_info=True)
            # Return basic structure even if metrics fail
            metrics = {'error': 'Failed to calculate forecast metrics.'}


        # --- Format Results ---
        logger.info(f"Formatting forecast results for {stock_symbol}.")
        # Select only the future prediction rows (tail(periods))
        forecast_output = forecast.tail(periods).copy()

        # Convert dates to ISO strings for JSON compatibility
        # Ensure 'ds' column is datetime first
        forecast_output['ds'] = pd.to_datetime(forecast_output['ds'])
        output_dates = forecast_output['ds'].dt.strftime('%Y-%m-%d').tolist() # Keep simple date format

        # Helper to safely convert forecast values to float or None
        def safe_float(value, decimals=2):
            if pd.isna(value) or np.isinf(value): return None
            try: return round(float(value), decimals)
            except (ValueError, TypeError): return None

        result = {
            'forecast_values': {
                'dates': output_dates,
                'values': [safe_float(v) for v in forecast_output['yhat']],
                'upper_bound': [safe_float(v) for v in forecast_output['yhat_upper']],
                'lower_bound': [safe_float(v) for v in forecast_output['yhat_lower']]
            },
            'metrics': metrics, # Include calculated metrics
            'last_actual_date': df['ds'].iloc[-1].strftime('%Y-%m-%d') if not df.empty else None,
            'last_actual_price': safe_float(df['y'].iloc[-1]) if not df.empty else None,
            'error': None # Indicate success
        }

        # Final check for None values in forecast (can indicate model instability)
        if None in result['forecast_values']['values']:
             logger.warning(f"Forecast for {stock_symbol} contains None values in 'values'. Check model stability/input data.")
             # Optionally add a warning to the result dictionary if needed

        logger.info(f"Prophet forecast successful for {stock_symbol}")
        return result

    # --- Outer Exception Handling ---
    except requests.exceptions.HTTPError as http_err:
        # Handle yfinance HTTP errors encountered during initial Ticker() or history() calls
        status_code = http_err.response.status_code if http_err.response else 'Unknown'
        logger.error(f"Prophet forecast failed for {stock_symbol} due to yfinance HTTP Error {status_code}: {http_err}")
        error_msg = f"Failed to retrieve data for '{stock_symbol}' from Yahoo Finance (HTTP {status_code})."
        if status_code == 404: error_msg = f"Symbol '{stock_symbol}' not found on Yahoo Finance."
        elif status_code == 429: error_msg += " Rate limit may have been exceeded."
        return {'error': error_msg}
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"Prophet forecast failed for {stock_symbol}: {type(e).__name__} - {str(e)}", exc_info=True)
        return {'error': f"An unexpected error occurred while generating the forecast for '{stock_symbol}': {type(e).__name__}"}