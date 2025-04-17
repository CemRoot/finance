# src/forecasting.py
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta  # Keep datetime class import
import pytz
import time
# Import renamed function
from .newsapi import rate_limit, YFINANCE_MIN_INTERVAL  # Import rate_limit
import logging
import requests

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def get_prophet_forecast(stock_symbol, periods=30):
    """Generates stock price forecast using Facebook Prophet library."""
    try:
        # Using .format() for the initial log message
        logger.info("Starting Prophet forecast for {} ({} periods)".format(stock_symbol, periods))

        ticker = yf.Ticker(stock_symbol)
        history = None
        for period in ["2y", "1y", "6mo"]:
            try:
                # Using .format()
                logger.info("Applying rate limit before fetching {} history for Prophet ({})".format(period, stock_symbol))
                rate_limit('yfinance')
                # Using .format()
                logger.info("Fetching {} history for Prophet ({})".format(period, stock_symbol))
                history = ticker.history(period=period, interval="1d", auto_adjust=True, repair=True, timeout=30)
                history_len = len(history) if history is not None else 0
                if history is not None and not history.empty and history_len >= 20:
                    # Using .format()
                    logger.info("Fetched {} data points ({}) for Prophet ({}).".format(history_len, period, stock_symbol))
                    break
                else:
                    # Using .format()
                    logger.warning("Insufficient data ({} rows) for '{}' for Prophet ({}). Trying next period.".format(history_len, period, stock_symbol))
                    history = None
            except requests.exceptions.HTTPError as http_err_hist:
                status_code = http_err_hist.response.status_code if http_err_hist.response else 'Unknown'
                # Using .format()
                logger.error("Prophet history fetch ({}) failed for {} (HTTP {}): {}".format(period, stock_symbol, status_code, http_err_hist))
                if status_code == 404:
                    return {'error': f"Symbol '{stock_symbol}' not found (HTTP 404)."}
            except Exception as e_hist:
                err_type = type(e_hist).__name__
                # Using .format()
                logger.error("Error fetching {} history for Prophet ({}): {} - {}".format(period, stock_symbol, err_type, e_hist))
                history = None
            if history is None or history.empty:
                time.sleep(0.5)

        if history is None or history.empty:
            # Using .format()
            logger.error("Prophet forecast failed: Could not fetch sufficient history for {}.".format(stock_symbol))
            return {'error': f"Could not retrieve enough historical data for '{stock_symbol}'."}

        # Using .format()
        logger.info("Preparing data for Prophet model ({}).".format(stock_symbol))
        if isinstance(history.columns, pd.MultiIndex):
            logger.warning("MultiIndex detected for {}. Flattening.".format(stock_symbol))
            try:
                history = history[['Open', 'High', 'Low', 'Close', 'Volume']]
            except KeyError as multi_err:
                cols_str = str(history.columns)
                # Using .format()
                logger.error("Could not select standard columns from MultiIndex for {}: {}. Columns: {}".format(stock_symbol, multi_err, cols_str))
                return {'error': f"Unexpected data column structure for '{stock_symbol}'."}
        if not isinstance(history.index, pd.DatetimeIndex):
            history.index = pd.to_datetime(history.index, errors='coerce')
            history = history[pd.notna(history.index)]
        df = history.reset_index()
        date_col_name = None
        if 'Date' in df.columns:
            date_col_name = 'Date'
        elif 'Datetime' in df.columns:
            date_col_name = 'Datetime'
        elif 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']):
            date_col_name = 'index'
        else:
            potential_date_col = df.columns[0]
            if pd.api.types.is_datetime64_any_dtype(df[potential_date_col]):
                date_col_name = potential_date_col

        df = df.rename(columns={date_col_name: 'ds', 'Close': 'y'})
        try:
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        except Exception as date_err:
            err_msg = str(date_err)
            # Using .format()
            logger.error("Error converting 'ds' to datetime for {}: {}".format(stock_symbol, err_msg))
            return {'error': f"Error processing date format for '{stock_symbol}'."}
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['y'] = df['y'].ffill().bfill()
        if df['y'].isnull().any():
            # Using .format()
            logger.error("Prophet forecast failed: 'y' column contains NaNs after fill for {}".format(stock_symbol))
            return {'error': f"Price data gaps could not be filled for '{stock_symbol}'."}

        df_len = len(df)
        if df_len < 5:
            # Using .format()
            logger.error("Insufficient data ({} rows) after cleaning for {}".format(df_len, stock_symbol))
            return {'error': f"Too few valid data points for '{stock_symbol}'."}
        elif df_len < 20:
            # Using .format()
            logger.warning("Short data ({} rows) for {}. Forecast accuracy might be low.".format(df_len, stock_symbol))
        df_fit = df[['ds', 'y']].copy()

        # Using .format()
        logger.info("Initializing and fitting Prophet model for {}...".format(stock_symbol))
        m = Prophet(yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality=False, seasonality_mode='multiplicative')
        try:
            m.fit(df_fit)
        except Exception as fit_err:
            logger.error(f"Prophet fitting failed for {stock_symbol}: {fit_err}", exc_info=True)
            return {'error': f"Failed to train forecast model: {fit_err}"}
        # Using .format()
        logger.info("Prophet model fitted for {}.".format(stock_symbol))

        future = m.make_future_dataframe(periods=periods, freq='D')
        # Using .format()
        logger.info("Making future predictions for {}...".format(stock_symbol))
        try:
            forecast = m.predict(future)
        except Exception as pred_err:
            logger.error(f"Prophet prediction failed for {stock_symbol}: {pred_err}", exc_info=True)
            return {'error': f"Failed to generate predictions: {pred_err}"}
        # Using .format()
        logger.info("Predictions generated for {}.".format(stock_symbol))

        # Using .format()
        logger.info("Calculating forecast metrics for {}...".format(stock_symbol))
        metrics = {}
        trend_direction = 'Uncertain'
        trend_strength = 0.0
        seasonality_strength = 0.0
        historical_volatility = None
        confidence_interval_relative = 0.0
        try:
            # ... (metric calculation logic remains the same) ...
            trend_series = forecast['trend']
            if len(trend_series) > 7:
                trend_diff = trend_series.iloc[-1] - trend_series.iloc[-8]
                last_trend_val = abs(trend_series.iloc[-1])
                if last_trend_val > 1e-6 and pd.notna(trend_diff):
                    trend_strength = abs(trend_diff / last_trend_val)
                    if trend_diff > (0.001 * last_trend_val):
                        trend_direction = 'Upward'
                    elif trend_diff < (-0.001 * last_trend_val):
                        trend_direction = 'Downward'
                    else:
                        trend_direction = 'Sideways'
            metrics['trend_direction'] = trend_direction
            metrics['trend_strength'] = float(trend_strength) if pd.notna(trend_strength) else 0.0
            total_seasonality_amplitude = 0
            component_count = 0
            for component in ['weekly', 'yearly']:
                if component in forecast.columns and forecast[component].nunique() > 1:
                    total_seasonality_amplitude += np.mean(np.abs(forecast[component]))
                    component_count += 1
            if component_count > 0:
                trend_mean_abs = np.mean(np.abs(forecast['trend']))
                if trend_mean_abs > 1e-6:
                    seasonality_strength = total_seasonality_amplitude / trend_mean_abs
            metrics['seasonality_strength'] = float(seasonality_strength) if pd.notna(seasonality_strength) else 0.0
            if len(df) >= 5:
                df['returns'] = df['y'].pct_change()
                window_size = min(30, len(df)-1)
                if window_size >= 2:
                    rolling_std = df['returns'].rolling(window=window_size).std()
                    last_valid_std = rolling_std.dropna().iloc[-1] if not rolling_std.dropna().empty else None
                    if last_valid_std is not None:
                        historical_volatility = last_valid_std * np.sqrt(252)
            metrics['historical_volatility'] = float(historical_volatility) if pd.notna(historical_volatility) else None
            last_forecast_point = forecast.iloc[-1]
            last_yhat = last_forecast_point.get('yhat')
            last_upper = last_forecast_point.get('yhat_upper')
            last_lower = last_forecast_point.get('yhat_lower')
            if all(pd.notna(v) for v in [last_yhat, last_upper, last_lower]) and abs(last_yhat) > 1e-6:
                confidence_interval_width = last_upper - last_lower
                confidence_interval_relative = confidence_interval_width / abs(last_yhat)
            metrics['confidence_interval'] = float(confidence_interval_relative) if pd.notna(confidence_interval_relative) else 0.0
        except Exception as metrics_err:
            err_msg = str(metrics_err)
            # Using .format()
            logger.error("Error calculating forecast metrics for {}: {}".format(stock_symbol, err_msg), exc_info=True)
            metrics = {'error': 'Failed to calculate metrics.'}

        # Using .format()
        logger.info("Formatting forecast results for {}.".format(stock_symbol))
        forecast_output = forecast.tail(periods).copy()
        forecast_output['ds'] = pd.to_datetime(forecast_output['ds'])
        output_dates = forecast_output['ds'].dt.strftime('%Y-%m-%d').tolist()

        def safe_float(v, dec=2):
            try:
                if pd.notna(v) and np.isfinite(v):
                    return round(float(v), dec)
                else:
                    return None
            except Exception:
                return None

        result = {
            'forecast_values': {
                'dates': output_dates,
                'values': [safe_float(v) for v in forecast_output['yhat']],
                'upper_bound': [safe_float(v) for v in forecast_output['yhat_upper']],
                'lower_bound': [safe_float(v) for v in forecast_output['yhat_lower']]
            },
            'metrics': metrics,
            'last_actual_date': df['ds'].iloc[-1].strftime('%Y-%m-%d') if not df.empty else None,
            'last_actual_price': safe_float(df['y'].iloc[-1]) if not df.empty else None,
            'error': None
        }
        if None in result['forecast_values']['values']:
            logger.warning(f"Forecast for {stock_symbol} contains None values.")  # f-string likely okay here
        # Using .format()
        logger.info("Prophet forecast successful for {}.".format(stock_symbol))
        return result
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response else 'Unknown'
        logger.error(f"Prophet forecast failed (HTTP {status_code}): {http_err}")
        error_msg = f"Failed data retrieval (HTTP {status_code})."
        if status_code == 404:
            error_msg = f"Symbol '{stock_symbol}' not found."
        elif status_code == 429:
            error_msg += " Rate limit likely exceeded."
        return {'error': error_msg}
    except Exception as e:
        logger.error(f"Prophet forecast failed: {type(e).__name__} - {str(e)}", exc_info=True)
        return {'error': f"Unexpected error generating forecast: {type(e).__name__}"}
