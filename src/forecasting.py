# src/forecasting.py
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta, date # Keep specific imports
import pytz
import time
from .newsapi import rate_limit # Import rate_limit directly
import logging
import requests
from random import uniform # Import uniform

# Suppress Prophet/CmdStanPy logs
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def get_prophet_forecast(stock_symbol: str, periods: int = 30, historical: bool = False, df: pd.DataFrame = None) -> dict:
    """
    Generates stock price forecast using Facebook Prophet library.
    Includes logistic growth, error handling, and metrics calculation.

    Args:
        stock_symbol (str): The stock symbol (e.g., 'AAPL').
        periods (int): Number of future days to forecast.
        historical (bool): Whether to include historical predictions for existing data.
        df (pd.DataFrame): Optional pre-loaded DataFrame (to avoid re-fetching data).

    Returns:
        dict: A dictionary containing forecast data and metrics, or an 'error' key on failure.
    """
    if not stock_symbol: logger.error("Forecast failed: Symbol empty."); return {'error': "Symbol cannot be empty."}
    stock_symbol = stock_symbol.strip().upper(); logger.info(f"Starting Prophet forecast for {stock_symbol} ({periods} periods, historical={historical})")
    error_return = { 'error': f"Forecast failed for {stock_symbol}", 'forecast_values': None, 'metrics': {'error': 'Forecast failed'}, 'last_actual_date': None, 'last_actual_price': None, 'execution_time': 0 }
    
    # Define minimum required rows here, so it's available for all code paths
    min_required_rows = 20

    try:
        # Use provided dataframe if available
        if df is not None and not df.empty:
            history = df.copy()
            logger.info(f"Using provided DataFrame with {len(history)} rows for {stock_symbol}")
            # Ensure it has the required columns
            if 'Close' not in history.columns or 'Date' not in history.columns:
                logger.error(f"Provided DataFrame missing required columns for {stock_symbol}")
                return {**error_return, 'error': "Provided DataFrame missing required columns (needs 'Date' and 'Close')"}
        else:
            # Fetch data if not provided
            ticker = yf.Ticker(stock_symbol); history = None; fetched_period = None
            
            for period in ["5y", "2y", "1y", "6mo"]:
                try:
                    logger.info(f"Rate limit check before fetching {period} history for {stock_symbol}"); rate_limit('yfinance')
                    logger.info(f"Fetching {period} history for {stock_symbol}"); history = ticker.history(period=period, interval="1d", auto_adjust=True, repair=True, timeout=45)
                    history_len = len(history) if history is not None else 0
                    if history is not None and not history.empty and history_len >= min_required_rows: logger.info(f"Fetched {history_len} points ({period}) for {stock_symbol}."); fetched_period = period; break
                    else: logger.warning(f"Insufficient data ({history_len}<{min_required_rows}) for '{period}' ({stock_symbol}). Trying next."); history = None
                except requests.exceptions.HTTPError as http_err:
                    status = http_err.response.status_code if http_err.response else 'Unknown'; logger.error(f"History fetch ({period}) failed ({stock_symbol}, HTTP {status}): {http_err}")
                    if status == 404: logger.error(f"'{stock_symbol}' not found (404). Aborting forecast."); return {**error_return, 'error': f"Symbol '{stock_symbol}' not found."}
                    history = None
                except Exception as e_hist: logger.error(f"Error fetching {period} history ({stock_symbol}): {type(e_hist).__name__}", exc_info=False); history = None
                if history is None or history.empty: time.sleep(uniform(0.3, 0.7))

            if history is None or history.empty: logger.error(f"Failed: Could not fetch sufficient history ({min_required_rows}+ rows) for {stock_symbol}."); return {**error_return, 'error': f"Could not retrieve enough data for '{stock_symbol}'."}

        logger.info(f"Preparing {len(history)} points for Prophet ({stock_symbol}).")
        if isinstance(history.columns, pd.MultiIndex):
            logger.warning(f"MultiIndex columns found for {stock_symbol}. Flattening.");
            try: std_cols = ['Open', 'High', 'Low', 'Close', 'Volume']; avail = [c for c in std_cols if c in history.columns.get_level_values(0)]; history = history[avail]; history.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in history.columns.values]
            except KeyError as e: logger.error(f"Failed to flatten MultiIndex: {e}"); return {**error_return, 'error': "Unexpected data structure."}
        if not isinstance(history.index, pd.DatetimeIndex): history.index = pd.to_datetime(history.index, errors='coerce')
        if history.index.tz is not None: history.index = history.index.tz_localize(None)
        history = history[pd.notna(history.index)];
        if history.empty: logger.error(f"DataFrame empty after date processing ({stock_symbol})"); return {**error_return, 'error': "Date processing error."}

        df = history.reset_index(); date_col = next((c for c in ['Date', 'Datetime', 'index'] if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c])), None)
        if not date_col: logger.error(f"Date column not found for {stock_symbol}"); return {**error_return, 'error': "Date column missing."}
        df = df.rename(columns={date_col: 'ds', 'Close': 'y'}); df['y'] = pd.to_numeric(df['y'], errors='coerce'); df['y'] = df['y'].ffill().bfill()
        if df['y'].isnull().any(): logger.error(f"NaNs in 'y' column after fill ({stock_symbol})"); return {**error_return, 'error': "Price data gaps."}
        df_len = len(df);
        if df_len < 5: logger.error(f"Insufficient data ({df_len}<5) after cleaning ({stock_symbol})"); return {**error_return, 'error': "Too few valid data points."}
        elif df_len < min_required_rows: logger.warning(f"Using short data ({df_len} rows) for {stock_symbol}.")
        df_fit = df[['ds', 'y']].copy()

        logger.info(f"Initializing and fitting Prophet model (logistic) for {stock_symbol}...")
        m = Prophet(growth='logistic', seasonality_mode='multiplicative', yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality=False)
        df_fit['cap'] = df_fit['y'].max() * 1.5; df_fit['floor'] = df_fit['y'].min() * 0.7; df_fit['floor'] = df_fit['floor'].apply(lambda x: max(x, 0.01))

        start_time = time.time()
        try: m.fit(df_fit)
        except Exception as fit_err: logger.error(f"Prophet fitting failed for {stock_symbol}: {fit_err}", exc_info=True); return {**error_return, 'error': f"Model training failed: {fit_err}"}
        logger.info(f"Prophet model fitted for {stock_symbol}.")

        future = m.make_future_dataframe(periods=periods, freq='D'); future['cap'] = df_fit['cap'].iloc[-1]; future['floor'] = df_fit['floor'].iloc[-1]
        logger.info(f"Making future predictions for {stock_symbol}...")
        try: forecast = m.predict(future)
        except Exception as pred_err: logger.error(f"Prophet prediction failed for {stock_symbol}: {pred_err}", exc_info=True); return {**error_return, 'error': f"Prediction failed: {pred_err}"}
        execution_time = time.time() - start_time; logger.info(f"Predictions generated ({execution_time:.2f}s) for {stock_symbol}.")

        logger.info(f"Calculating forecast metrics for {stock_symbol}...")
        metrics = {}; trend_direction = 'Uncertain'; trend_strength = 0.0; seasonality_strength = 0.0; historical_volatility = None; confidence_interval_relative = 0.0
        try:
            trend = forecast['trend']
            # *** DÜZELTİLMİŞ BLOK ***
            if len(trend) > 7:
                last = trend.iloc[-1]
                prev = trend.iloc[-8]
                if abs(last) > 1e-6 and pd.notna(last) and pd.notna(prev):
                    diff = last - prev
                    trend_strength = abs(diff / last) # Standard division
                    # Trend yönünü ayrı satırlarda belirle
                    if diff > (0.005 * abs(last)):
                        trend_direction = 'Upward'
                    elif diff < (-0.005 * abs(last)):
                        trend_direction = 'Downward'
                    else:
                        trend_direction = 'Sideways'
            # *** DÜZELTME SONU ***
            metrics['trend_direction'] = trend_direction
            metrics['trend_strength'] = float(trend_strength) if pd.notna(trend_strength) else 0.0

            seas_amp = 0; comp_count = 0; trend_mean = np.mean(np.abs(trend))
            for comp in ['weekly', 'yearly']:
                if comp in forecast.columns and forecast[comp].nunique() > 1:
                    seas_amp += np.mean(np.abs(forecast[comp]))
                    comp_count += 1
            if comp_count > 0 and trend_mean > 1e-6:
                seasonality_strength = seas_amp / trend_mean
            metrics['seasonality_strength'] = float(seasonality_strength) if pd.notna(seasonality_strength) else 0.0

            # Calculate historical volatility
            if 'returns' not in df.columns and len(df) >= 2: # Ensure returns are calculated if missing
                 df['returns'] = df['y'].pct_change()
            if 'returns' in df.columns and len(df['returns'].dropna()) >= 5:
                win = min(30, len(df) - 1)
                if win >= 2:
                    std = df['returns'].rolling(window=win).std()
                    last_std = std.dropna().iloc[-1] if not std.dropna().empty else None
                    if last_std:
                        historical_volatility = last_std * np.sqrt(252)
            metrics['historical_volatility'] = float(historical_volatility) if pd.notna(historical_volatility) else None

            # Calculate confidence interval
            last_fc = forecast.iloc[-1]
            yhat = last_fc.get('yhat')
            up = last_fc.get('yhat_upper')
            low = last_fc.get('yhat_lower')
            if all(pd.notna(v) for v in [yhat, up, low]) and abs(yhat) > 1e-6:
                width = up - low
                confidence_interval_relative = width / abs(yhat)
            metrics['confidence_interval'] = float(confidence_interval_relative) if pd.notna(confidence_interval_relative) else 0.0

        except Exception as metrics_err:
            logger.error(f"Metrics calc error for {stock_symbol}: {metrics_err}", exc_info=True)
            metrics = {'error': 'Metrics calculation failed.'} # Include error in metrics dict

        logger.info(f"Formatting forecast results for {stock_symbol}. Metrics: {metrics}")
        
        # Future forecast (predictions after the last known date)
        forecast_output = forecast[forecast['ds'] > df['ds'].max()].copy()
        forecast_output['ds'] = pd.to_datetime(forecast_output['ds'])
        output_dates = forecast_output['ds'].dt.strftime('%Y-%m-%d').tolist()

        def safe_float(v, dec=2):
            try: return round(float(v), dec) if pd.notna(v) and np.isfinite(v) else None
            except (ValueError, TypeError): return None

        last_actual = df.iloc[-1] if not df.empty else None
        
        # Include historical predictions if requested
        historical_forecast_data = None
        if historical:
            # Include predictions for all historical dates
            historical_output = forecast[forecast['ds'] <= df['ds'].max()].copy()
            historical_output['ds'] = pd.to_datetime(historical_output['ds'])
            
            # Create dataframe with historical predictions
            historical_forecast_data = {
                'dates': historical_output['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'values': [safe_float(v) for v in historical_output['yhat']],
                'upper_bound': [safe_float(v) for v in historical_output['yhat_upper']],
                'lower_bound': [safe_float(v) for v in historical_output['yhat_lower']]
            }
            logger.info(f"Added {len(historical_output)} historical predictions for {stock_symbol}")

        result = {
            'forecast_values': {
                'dates': output_dates,
                'values': [safe_float(v) for v in forecast_output['yhat']],
                'upper_bound': [safe_float(v) for v in forecast_output['yhat_upper']],
                'lower_bound': [safe_float(v) for v in forecast_output['yhat_lower']]
            },
            'metrics': metrics,
            'last_actual_date': last_actual['ds'].strftime('%Y-%m-%d') if last_actual is not None else None,
            'last_actual_price': safe_float(last_actual['y']) if last_actual is not None else None,
            'execution_time': execution_time,
            'error': None # Indicate success if we reached here
        }
        
        # Include historical forecast if requested
        if historical_forecast_data:
            result['historical_forecast'] = historical_forecast_data

        if None in result['forecast_values']['values']:
            logger.warning(f"Forecast for {stock_symbol} contains None values (likely due to logistic cap/floor or model instability).")

        logger.info(f"Prophet forecast successful for {stock_symbol}.")
        return result

    except Exception as e:
        logger.error(f"Prophet forecast failed unexpectedly for {stock_symbol}: {type(e).__name__}", exc_info=True)
        # Return consistent error structure
        return {
            **error_return, # Use the predefined error structure
            'error': f"Unexpected error: {type(e).__name__}"
        }