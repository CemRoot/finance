# app.py
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from flask_caching import Cache
from dotenv import load_dotenv
import src.newsapi as newsapi
import src.decision as decision # Decision module
# Forecasting modülünü ve içindeki fonksiyonu import et
import src.forecasting as forecasting
import src.sentiment as sentiment # Sentiment module
# ML modellerini ve karşılaştırıcıyı import et
from src.ml_models import StockPricePredictor
from src.model_comparison import ModelComparator
from src.xgboost_model import XGBoostStockPredictor, train_xgboost_model
# Import AlphaVantageAPI for stock data in forest model and deep learning parts
from src.marketstack_api import MarketstackAPI

import json
from datetime import datetime, timedelta, timezone, UTC # Import specific classes
import time
from functools import wraps
import logging
import os
import pytz
import pandas as pd
import numpy as np
import pandas_ta as ta # Technical Analysis library
import jinja2 # Import jinja2 for exception handling
import plotly.io as pio
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask Application and Cache Configuration
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-should-be-changed')

# Configure caching with timeouts from config
from config import (CACHE_DEFAULT_TIMEOUT, STOCK_DATA_CACHE_TIMEOUT,
                   NEWS_CACHE_TIMEOUT, FORECAST_CACHE_TIMEOUT)

cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": CACHE_DEFAULT_TIMEOUT
}
cache = Cache(config=cache_config)
cache.init_app(app)

# Initialize Marketstack API
try:
    marketstack = MarketstackAPI()
except ValueError as e:
    logger.error(f"Failed to initialize Marketstack API: {e}")
    marketstack = None

# Configure Plotly for static image export
try:
    import kaleido  # ensure kaleido package is loaded
    if pio.kaleido and pio.kaleido.scope:
        pio.kaleido.scope.default_format = "png"
except Exception as e:
    logger.warning(f"Plotly kaleido scope unavailable: {e}")

@app.context_processor
def inject_current_time():
    return {'current_time': datetime.now()}

# --- Popular stock symbols with associated name fragments for lookup ---
# (POPULAR_SYMBOLS_FOR_NAME_LOOKUP listesi aynı kalıyor)
POPULAR_SYMBOLS_FOR_NAME_LOOKUP = {
    "AAPL": ["apple"], "MSFT": ["microsoft"], "GOOG": ["google", "alphabet"],
    "GOOGL": ["google", "alphabet"], "AMZN": ["amazon"], "TSLA": ["tesla"],
    "META": ["meta", "facebook"], "NVDA": ["nvidia"], "JPM": ["jpmorgan", "jp morgan"],
    "BAC": ["bank of america", "bofa"], "WFC": ["wells fargo"], "C": ["citigroup", "citi"],
    "PFE": ["pfizer"], "JNJ": ["johnson", "johnson & johnson"], "UNH": ["unitedhealth", "united health"],
    "HD": ["home depot"], "PG": ["procter", "gamble", "procter & gamble"], "XOM": ["exxon", "exxonmobil"],
    "CVX": ["chevron"], "KO": ["coca", "coca-cola", "coca cola"], "PEP": ["pepsi", "pepsico"],
    "T": ["at&t", "at and t"], "VZ": ["verizon"], "DIS": ["disney", "walt disney"],
    "NFLX": ["netflix"], "INTC": ["intel"], "AMD": ["amd", "advanced micro devices"],
    "CSCO": ["cisco"], "IBM": ["ibm", "international business machines"], "ORCL": ["oracle"],
    "CRM": ["salesforce"], "ADBE": ["adobe"], "PYPL": ["paypal"], "V": ["visa"], "MA": ["mastercard"],
    "WMT": ["walmart", "wal-mart"], "TGT": ["target"], "COST": ["costco"], "SBUX": ["starbucks"],
    "MCD": ["mcdonald", "mcdonalds", "mcdonald's"], "BABA": ["alibaba"], "GM": ["general motors"],
    "F": ["ford"], "GE": ["general electric"], "BA": ["boeing"],
    "THYAO.IS": ["thy", "turkish airlines"], "GARAN.IS": ["garanti"], "BIMAS.IS": ["bim"], "ASELS.IS": ["aselsan"]
}

# Logging Setup
# Flask'ın kendi logger'ını kullanmak genellikle daha iyidir.
# Basic config sadece Flask logger'ı başlatılmamışsa bir fallback sağlar.
if not app.debug: # Don't use Flask's default handlers in debug mode if you configured custom ones
    # Production logging can be configured here (e.g., file handler)
    pass
else:
    # Ensure Flask's handler uses our format in debug mode
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    # Clear default handlers and add the custom formatted one
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO) # Set level for Flask's logger


# --- Cached Functions ---
@cache.memoize(timeout=600)
def cached_get_stock_data(symbol):
    app.logger.info(f"CACHE MISS/expired for get_stock_data({symbol}). Calling API.")
    return newsapi.get_stock_data(symbol)

@cache.memoize(timeout=1800)
def cached_get_news(symbol):
    app.logger.info(f"CACHE MISS/expired for get_news({symbol}). Calling API.")
    return newsapi.get_news(symbol, lang='en')

@cache.memoize(timeout=3600)
def cached_get_ticker_info(symbol):
    """Stub ticker info: uses symbol as company name and no yfinance."""
    app.logger.info(f"CACHE MISS for ticker_info({symbol}). Using stub.")
    return {'shortName': symbol, 'longName': '', 'symbol': symbol, 'currency': None}

# --- Jinja2 Filters ---
# (Filtre fonksiyonları aynı kalıyor)
@app.template_filter('calculate_average')
def calculate_average(values):
    if not values: return 'N/A'
    try: valid_values = [float(v) for v in values if v is not None and isinstance(v, (int, float, np.number)) and pd.notna(v)];
    except (ValueError, TypeError): return 'N/A'
    if not valid_values: return 'N/A'
    try: avg = sum(valid_values) / len(valid_values)
    except ZeroDivisionError: return 'N/A'
    if abs(avg) >= 1e9: return f"{avg / 1e9:.1f}B"
    if abs(avg) >= 1e6: return f"{avg / 1e6:.1f}M"
    if abs(avg) >= 1e3: return f"{avg / 1e3:,.0f}K"
    return f"{avg:,.0f}"

@app.template_filter('format_number')
def format_number(value, decimals=2):
    if value is None or not isinstance(value, (int, float, np.number)) or not pd.notna(value): return 'N/A'
    try: format_string = '{:,.%df}' % decimals; return format_string.format(float(value))
    except (ValueError, TypeError): return 'N/A'

@app.template_filter('format_date')
def format_date(value, format_str='%d %b %Y'):
    if not value: return ''
    try:
        dt_obj = None
        if isinstance(value, datetime): dt_obj = value
        elif isinstance(value, pd.Timestamp): dt_obj = value.to_pydatetime()
        elif isinstance(value, np.datetime64): dt_obj = pd.Timestamp(value).to_pydatetime()
        elif isinstance(value, str):
            try:
                if 'T' in value: dt_obj = datetime.fromisoformat(value.split('.')[0].replace('Z', ''))
                else: dt_obj = datetime.strptime(value, '%Y-%m-%d')
            except ValueError: app.logger.debug(f"format_date: Could not parse date string '{value}'"); return value
        else: app.logger.warning(f"format_date: Unsupported type: {type(value)}"); return str(value)
        if dt_obj.tzinfo is None: dt_obj = pytz.utc.localize(dt_obj)
        else: dt_obj = dt_obj.astimezone(pytz.utc)
        return dt_obj.strftime(format_str)
    except Exception as e: app.logger.error(f"Error formatting date '{value}': {e}", exc_info=False); return str(value)

app.jinja_env.filters['calculate_average'] = calculate_average
app.jinja_env.filters['format_number'] = format_number
app.jinja_env.filters['format_date'] = format_date
app.jinja_env.globals.update(abs=abs, len=len, isinstance=isinstance, float=float, int=int, str=str, list=list, dict=dict, datetime=datetime, max=max, min=min, round=round, now=lambda: datetime.now(pytz.utc))

# --- Main Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    stock_symbol = session.get('last_symbol', None)
    current_utc_time = datetime.now(UTC)

    if request.method == 'POST':
        search_query = request.form.get('stock', '').strip()
        if not search_query: flash('Please enter stock symbol or company name.', 'warning'); return redirect(url_for('index'))
        stock_symbol_candidate = search_query.upper(); resolved_symbol = None
        is_likely_symbol = '.' in stock_symbol_candidate or (stock_symbol_candidate.isalnum() and not stock_symbol_candidate.isdigit() and len(stock_symbol_candidate) <= 6)

        if is_likely_symbol:
            try: info_check = cached_get_ticker_info(stock_symbol_candidate);
            except Exception as e: app.logger.warning(f"Symbol validation error for '{stock_symbol_candidate}': {e}."); info_check = None; is_likely_symbol = False;
            if info_check and (info_check.get('shortName') or info_check.get('longName')): app.logger.info(f"Search '{search_query}' validated as symbol."); resolved_symbol = stock_symbol_candidate
            else: is_likely_symbol = False; app.logger.info(f"'{search_query}' failed symbol validation.")

        if not resolved_symbol:
            app.logger.info(f"Treating '{search_query}' as company name...")
            search_query_lower = search_query.lower()
            for symbol, name_fragments in POPULAR_SYMBOLS_FOR_NAME_LOOKUP.items():
                if any(fragment in search_query_lower for fragment in name_fragments): resolved_symbol = symbol; app.logger.info(f"Matched {symbol} via fragments."); break

        if not resolved_symbol:
             if is_likely_symbol: resolved_symbol = stock_symbol_candidate; app.logger.warning(f"Could not validate '{resolved_symbol}', proceeding as symbol.")
             else: flash(f"Could not resolve '{search_query}'. Try exact symbol.", 'warning'); return redirect(url_for('index'))

        session['last_symbol'] = resolved_symbol; app.logger.info(f"Search resolved to '{resolved_symbol}'. Redirecting.")
        cache_keys = [f"cached_get_stock_data_{resolved_symbol}", f"cached_get_news_{resolved_symbol}", f"forecast_{resolved_symbol}", f"ticker_info_{resolved_symbol}"]
        for key in cache_keys: cache.delete(key)
        app.logger.info(f"Cache cleared for {resolved_symbol}")
        return redirect(url_for('index', stock=resolved_symbol))

    stock_symbol_arg = request.args.get('stock')
    if stock_symbol_arg: stock_symbol = stock_symbol_arg.strip().upper(); session['last_symbol'] = stock_symbol
    elif not stock_symbol: return render_template('index.html', stock=None, current_time=current_utc_time)

    app.logger.info(f"Processing GET for symbol: {stock_symbol}")
    stock_data, articles, tech_indicators, avg_sentiment = None, [], {}, 0.0
    decision_result = "Analysis Pending"; error_occurred = False; error_message = None

    try:
        stock_data = cached_get_stock_data(stock_symbol)
        if not stock_data: raise ValueError(f"No data retrieved for '{stock_symbol}'.")
        if stock_data.get('error'): raise ValueError(stock_data['error'])
        if not stock_data.get('labels') or not stock_data.get('values'): raise ValueError(f"Core data missing for '{stock_symbol}'.")
        app.logger.info(f"Stock data OK for {stock_symbol}. Labels: {len(stock_data.get('labels', []))}")
        if 'symbol' not in stock_data: stock_data['symbol'] = stock_symbol

        try:
            df = pd.DataFrame({'Open': stock_data['open_values'], 'High': stock_data['high_values'],'Low': stock_data['low_values'], 'Close': stock_data['values'],'Volume': stock_data['volume_values']}, index=pd.to_datetime(stock_data['labels'], errors='coerce'))
            df.index.name = 'Date'; df = df.loc[pd.notna(df.index)]; df.dropna(subset=['Close'], inplace=True)
            if len(df) >= 2:
                app.logger.info(f"Calculating TA for {stock_symbol} ({len(df)} points).")
                min_rows = 20; df['returns'] = df['Close'].pct_change(); std_dev = df['returns'].std(); tech_indicators['volatility'] = std_dev * np.sqrt(252) if pd.notna(std_dev) else None
                if len(df) >= min_rows:
                    df.ta.sma(length=20, append=True); df.ta.rsi(length=14, append=True); df.ta.macd(fast=12, slow=26, signal=9, append=True); df.ta.bbands(length=20, std=2, append=True)
                    last = df.iloc[-1]; cp = last.get('Close'); sma20 = last.get('SMA_20'); rsi14 = last.get('RSI_14'); macd = last.get('MACD_12_26_9'); macds = last.get('MACDs_12_26_9'); macdh = last.get('MACDh_12_26_9');
                    bbl_key = next((col for col in df.columns if col.startswith('BBL_')), None); bbu_key = next((col for col in df.columns if col.startswith('BBU_')), None); bbm_key = next((col for col in df.columns if col.startswith('BBM_')), None)
                    bbl = last.get(bbl_key) if bbl_key else None; bbu = last.get(bbu_key) if bbu_key else None; bbm = last.get(bbm_key) if bbm_key else None
                    if pd.notna(cp) and pd.notna(sma20): tech_indicators['sma'] = 'Above' if cp > sma20 else ('Below' if cp < sma20 else 'Equal')
                    if pd.notna(rsi14): tech_indicators['rsi'] = 'Overbought' if rsi14 > 70 else ('Oversold' if rsi14 < 30 else 'Neutral')
                    if pd.notna(macd) and pd.notna(macds): tech_indicators['macd'] = 'Buy Signal (Positive)' if macd > macds else 'Sell Signal (Negative)'
                    if pd.notna(macdh): tech_indicators['macd_hist'] = float(round(macdh, 3))
                    if pd.notna(cp) and pd.notna(bbl) and pd.notna(bbu): tech_indicators['bbands'] = 'Below Lower Band' if cp < bbl else ('Above Upper Band' if cp > bbu else 'Inside Bands')
                    if pd.notna(bbm): tech_indicators['bbands_mid'] = float(round(bbm, 2))
                    app.logger.info(f"TA Indicators calculated: {tech_indicators}")
                else: app.logger.warning(f"Insufficient data ({len(df)}<{min_rows}) for some TA.")
            else: app.logger.warning(f"DataFrame empty/short after cleaning for {stock_symbol}. Skipping TA.")
        except Exception as ta_error: app.logger.error(f"TA calc error: {ta_error}", exc_info=True); flash("Warning: TA calc failed.", 'warning'); tech_indicators = {}

        articles_result = cached_get_news(stock_symbol)
        if isinstance(articles_result, dict) and articles_result.get('error'): flash(f"News Error: {articles_result['error']}", 'warning'); articles = []
        elif isinstance(articles_result, list): articles = articles_result
        else: flash("Unexpected news format.", 'warning'); articles = []
        app.logger.info(f"News articles retrieved: {len(articles)}")

        if articles:
            try: avg_sentiment = sentiment.analyze_articles_sentiment(articles); app.logger.info(f"Avg sentiment: {avg_sentiment:.4f}")
            except Exception as sent_err: app.logger.error(f"Sentiment error: {sent_err}", exc_info=True); flash("Sentiment analysis error.", "warning")

        try: decision_result = decision.make_decision_with_indicators(tech_indicators, avg_sentiment); app.logger.info(f"Decision: {decision_result}")
        except Exception as dec_err: decision_result = "Decision Error"; app.logger.error(f"Decision logic error: {dec_err}", exc_info=True); flash("Decision analysis error.", "warning")

    except ValueError as ve: error_occurred = True; error_message = str(ve); app.logger.error(f"Data error for {stock_symbol}: {error_message}"); flash(error_message, 'danger'); stock_data, articles, tech_indicators, decision_result = None, [], {}, "Error"
    except Exception as e: error_occurred = True; error_message = "Unexpected server error."; app.logger.error(f"Unexpected error for {stock_symbol}: {type(e).__name__} - {e}", exc_info=True); flash(error_message, 'danger'); stock_data, articles, tech_indicators, decision_result = None, [], {}, "Error"

    context = { 'stock': stock_symbol, 'stock_data': stock_data, 'articles': articles, 'error': error_occurred, 'error_message': error_message, 'decision': decision_result, 'tech_indicators': tech_indicators }
    return render_template('index.html', **context)

# --- AJAX / Data Endpoints ---
@app.route('/refresh_stock')
def refresh_stock():
    stock = request.args.get("stock");
    if not stock: return jsonify({"status": "error", "message": "Symbol required"}), 400
    stock = stock.strip().upper(); app.logger.info(f"AJAX REFRESH request for: {stock}")
    try:
        stock_data = newsapi.get_stock_data(stock) # Direct call
        if not stock_data: raise ValueError(f"No data from get_stock_data for {stock}")
        if stock_data.get('error'):
            error_msg = stock_data['error']; status_code = 500;
            if any(k in error_msg.lower() for k in ["not found", "invalid", "no data", "delisted"]): status_code = 404
            elif any(k in error_msg.lower() for k in ["rate limit", "too many requests"]): status_code = 429
            elif "timeout" in error_msg.lower(): status_code = 504
            app.logger.warning(f"Refresh failed for {stock}: {error_msg} (Status: {status_code})")
            return jsonify({"status": "error", "message": error_msg}), status_code
        elif not stock_data.get('labels') or not stock_data.get('values'):
             app.logger.warning(f"Refreshed data for {stock} incomplete."); return jsonify({"status": "error", "message": f"Refreshed data for {stock} incomplete."}), 500
        else:
            if 'symbol' not in stock_data: stock_data['symbol'] = stock
            cache_key = f"cached_get_stock_data_{stock}"; cache.set(cache_key, stock_data)
            app.logger.info(f"Cache updated for {stock} after refresh."); return jsonify({"status": "success", "stock_data": stock_data })
    except Exception as e: app.logger.error(f"Refresh error ({stock}): {e}", exc_info=True); return jsonify({"status": "error", "message": "Server error refreshing data."}), 500

@app.route('/get_forecast_data', methods=['GET'])
def get_forecast_data():
    symbol = request.args.get('symbol');
    if not symbol: return jsonify({'error': 'Symbol required.'}), 400
    symbol = symbol.strip().upper(); cache_key = f"forecast_{symbol}"; cached_result = cache.get(cache_key)
    if cached_result: app.logger.info(f"Returning cached forecast for {symbol}"); return jsonify(cached_result)
    app.logger.info(f"CACHE MISS forecast: {symbol}. Generating...")
    try:
        forecast_result = forecasting.get_prophet_forecast(symbol, periods=30)
        if forecast_result is None: raise ValueError("Forecast returned None")
        if forecast_result.get('error'):
            error_msg = forecast_result['error']; status_code = 500;
            if any(k in error_msg.lower() for k in ["veri bulunamadı", "insufficient", "not found", "retrieve enough"]): status_code = 404
            elif "rate limit" in error_msg.lower(): status_code = 429
            app.logger.warning(f"Forecast failed for {symbol}: {error_msg} (Status: {status_code})")
            return jsonify({'error': error_msg}), status_code # Return only error field
        else:
            forecast_result['timestamp'] = datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S.000Z')
            cache.set(cache_key, forecast_result, timeout=3600) # Cache 1 hour
            app.logger.info(f"Forecast generated and cached for {symbol}.")
            return jsonify(forecast_result) # Return full result
    except Exception as e: app.logger.error(f"Forecast endpoint error ({symbol}): {e}", exc_info=True); return jsonify({'error': f"Server error generating forecast."}), 500

@app.route('/clear_cache')
def clear_cache_key():
    key = request.args.get('key');
    if key:
        try: prefix = cache.config.get('CACHE_KEY_PREFIX', ''); full_key = prefix + key; deleted = cache.delete(key) or cache.delete(full_key)
        except Exception as e: app.logger.error(f"Cache delete error for '{key}': {e}"); return jsonify({"status": "error", "message": "Error clearing cache."}), 500
        if deleted: app.logger.info(f"Cache key '{key}' deleted."); return jsonify({"status": "success", "message": f"Key '{key}' cleared."}), 200
        else: app.logger.warning(f"Cache key '{key}' not found."); return jsonify({"status": "warning", "message": f"Key '{key}' not found."}), 404
    else: return jsonify({"status": "error", "message": "No key provided."}), 400

# --- ML Analysis Routes ---
@app.route('/random_forest_analysis')
def random_forest_analysis():
    stock = request.args.get('stock')
    if not stock: flash("Select stock first.", "warning"); return redirect(url_for('index'))
    stock = stock.strip().upper()

    try:
        app.logger.info(f"Starting RF analysis for {stock}")
        
        # Use Marketstack API to fetch data for Random Forest analysis
        app.logger.info(f"Fetching 10-year data from Marketstack for {stock}")
        
        # Use the new method for fetching 10 years of historical data
        df = marketstack.fetch_full_historical_data(stock)
        
        if df is None or df.empty:
            flash(f"No data found in Marketstack for {stock}. Please check if this symbol is available.", "warning")
            return redirect(url_for('index', stock=stock))
        
        app.logger.info(f"Retrieved {len(df)} rows from Marketstack for {stock}")
        
        # Enhance data with additional technical indicators
        app.logger.info(f"Calculating technical indicators for {stock}")
        
        # Add various technical indicators
        if len(df) >= 100:
            # Moving averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential moving averages
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # Volatility indicators
            df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # Trend indicators
            macd_result = ta.macd(df['Close'])
            df['MACD'] = macd_result['MACD_12_26_9']
            df['MACD_Signal'] = macd_result['MACDs_12_26_9']
            df['MACD_Hist'] = macd_result['MACDh_12_26_9']
            
            # Momentum indicators
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
            df['ROC_10'] = ta.roc(df['Close'], length=10)
            
            # Volume indicators
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            
            # Bollinger Bands
            bbands = ta.bbands(df['Close'], length=20, std=2)
            
            # Log column names to debug the issue
            app.logger.info(f"Bollinger Bands columns: {bbands.columns.tolist()}")
            
            # Check for BBU column using pattern matching to handle different naming conventions
            bbu_col = next((col for col in bbands.columns if col.startswith('BBU_')), None)
            bbm_col = next((col for col in bbands.columns if col.startswith('BBM_')), None)
            bbl_col = next((col for col in bbands.columns if col.startswith('BBL_')), None)
            
            if bbu_col and bbm_col and bbl_col:
                df['BB_Upper'] = bbands[bbu_col]
                df['BB_Middle'] = bbands[bbm_col]
                df['BB_Lower'] = bbands[bbl_col]
                app.logger.info(f"Successfully mapped Bollinger Bands columns: {bbu_col}, {bbm_col}, {bbl_col}")
            else:
                # If columns not found, calculate manually as fallback
                app.logger.warning(f"Bollinger Band columns not found. Using manual calculation.")
                sma = df['Close'].rolling(window=20).mean()
                std = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = sma + (std * 2)
                df['BB_Middle'] = sma
                df['BB_Lower'] = sma - (std * 2)
            
            app.logger.info(f"Technical indicators calculated successfully for {stock}")
        else:
            app.logger.warning(f"Not enough data points for {stock} to calculate technical indicators")
        
        # Drop rows with NaN values that result from calculating indicators
        df.dropna(inplace=True)
        
        if len(df) < 100:
            flash(f"Insufficient historical data ({len(df)} points, need at least 100) for ML analysis for {stock}.", "warning")
            return redirect(url_for('index', stock=stock))

        # Reset index to make Date a column, which is expected by StockPricePredictor
        df = df.reset_index()
        
        app.logger.info(f"Final dataframe shape: {df.shape}, columns: {df.columns.tolist()}")
        
        model = StockPricePredictor(model_type='regressor')
        train_results = model.train(df.copy())
        metrics = train_results[0]
        if not metrics: raise ValueError("Model training failed or returned no metrics.")

        learning_curve_data = model.plot_learning_curve(df.copy())
        feature_importance_data = model.plot_feature_importance()

        app.logger.info(f"RF analysis completed for {stock} (using default/initial params)")
        current_utc_time = datetime.now(UTC)
        return render_template('ml_analysis.html', stock=stock, model_type="Random Forest", metrics=metrics,
                               learning_curve_path="static/" + learning_curve_data.get('image_path', ''),
                               feature_importance_path="static/" + feature_importance_data.get('image_path', ''),
                               current_time=current_utc_time)
    except Exception as e:
        app.logger.error(f"Error in RF analysis for {stock}: {str(e)}", exc_info=True)
        flash(f"Error generating RF analysis: {str(e)}", "danger")
        return redirect(url_for('index', stock=stock))

@app.route('/model_comparison')
def model_comparison():
    stock_symbol = request.args.get('stock')
    if not stock_symbol:
        flash('Stock symbol is required.', 'danger')
        return redirect(url_for('index'))
    
    stock_symbol = stock_symbol.strip().upper()
    app.logger.info(f"Starting model comparison for {stock_symbol}")
    
    try:
        # Get stock data
        comparator = ModelComparator()
        
        # Run comparison (will fetch data internally)
        comparison_result = comparator.compare_models(stock_symbol)
        
        if 'error' in comparison_result:
            flash(f"Model comparison error: {comparison_result['error']}", 'danger')
            # If there are model-specific errors, display them too
            if 'model_errors' in comparison_result:
                for model, error in comparison_result['model_errors'].items():
                    if error:
                        flash(f"{model} error: {error}", 'warning')
            return redirect(url_for('index', stock=stock_symbol))
        
        app.logger.info(f"Model comparison completed for {stock_symbol}")
        
        # Render the comparison template with results
        return render_template('model_comparison.html', 
                               stock=stock_symbol,
                               comparison=comparison_result,
                               plot_url=comparison_result.get('plot_path', ''),
                               performance_plot_url=comparison_result.get('performance_plot_path', ''),
                               description="Comparison of Random Forest, XGBoost, LSTM, and Prophet models for stock price prediction.",
                               current_time=datetime.now(UTC))
    
    except Exception as e:
        app.logger.error(f"Error in model comparison for {stock_symbol}: {e}", exc_info=True)
        flash(f"Error processing model comparison: {str(e)}", 'danger')
        return redirect(url_for('index', stock=stock_symbol))

@app.route('/xgboost_analysis')
def xgboost_analysis():
    stock_symbol = request.args.get('stock')
    prediction_type = request.args.get('type', 'regressor')  # Default to regressor
    auto_optimize = request.args.get('optimize', 'true').lower() == 'true'
    
    if not stock_symbol:
        flash('Stock symbol is required.', 'danger')
        return redirect(url_for('index'))
    
    stock_symbol = stock_symbol.strip().upper()
    app.logger.info(f"Starting XGBoost analysis for {stock_symbol} (type: {prediction_type}, optimize: {auto_optimize})")
    
    try:
        # Get 10 years of stock data from Marketstack
        app.logger.info(f"Fetching 10-year data from Marketstack for {stock_symbol}")
        
        # Use the new method for fetching 10 years of historical data
        df = marketstack.fetch_full_historical_data(stock_symbol)
        
        if df is None or df.empty:
            flash(f"No data found in Marketstack for {stock_symbol}. Please check if this symbol is available.", "warning")
            return redirect(url_for('index', stock=stock_symbol))
        
        app.logger.info(f"Retrieved {len(df)} rows from Marketstack for {stock_symbol}")
        
        # Calculate additional technical indicators
        app.logger.info(f"Calculating technical indicators for {stock_symbol}")
        
        # Add various technical indicators
        if len(df) >= 100:
            # Moving averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential moving averages
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # Volatility indicators
            df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # Trend indicators
            macd_result = ta.macd(df['Close'])
            df['MACD'] = macd_result['MACD_12_26_9']
            df['MACD_Signal'] = macd_result['MACDs_12_26_9']
            df['MACD_Hist'] = macd_result['MACDh_12_26_9']
            
            # Momentum indicators
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
            df['ROC_10'] = ta.roc(df['Close'], length=10)
            
            # Volume indicators
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            
            # Bollinger Bands
            bbands = ta.bbands(df['Close'], length=20, std=2)
            
            # Check for BBU column using pattern matching to handle different naming conventions
            bbu_col = next((col for col in bbands.columns if col.startswith('BBU_')), None)
            bbm_col = next((col for col in bbands.columns if col.startswith('BBM_')), None)
            bbl_col = next((col for col in bbands.columns if col.startswith('BBL_')), None)
            
            if bbu_col and bbm_col and bbl_col:
                df['BB_Upper'] = bbands[bbu_col]
                df['BB_Middle'] = bbands[bbm_col]
                df['BB_Lower'] = bbands[bbl_col]
                app.logger.info(f"Successfully mapped Bollinger Bands columns: {bbu_col}, {bbm_col}, {bbl_col}")
            else:
                # If columns not found, calculate manually as fallback
                app.logger.warning(f"Bollinger Band columns not found. Using manual calculation.")
                sma = df['Close'].rolling(window=20).mean()
                std = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = sma + (std * 2)
                df['BB_Middle'] = sma
                df['BB_Lower'] = sma - (std * 2)
            
            # Price rate of change
            df['Close_Pct_Change'] = df['Close'].pct_change()
            df['Volume_Pct_Change'] = df['Volume'].pct_change()
            
            # Add price direction (if classifier model)
            if prediction_type == 'classifier':
                df['Price_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            
            app.logger.info(f"Technical indicators calculated successfully for {stock_symbol}")
        else:
            app.logger.warning(f"Not enough data points for {stock_symbol} to calculate technical indicators")
        
        # Drop rows with NaN values that result from calculating indicators
        df.dropna(inplace=True)
        
        if len(df) < 100:
            flash(f"Insufficient historical data ({len(df)} points, need at least 100) for ML analysis for {stock_symbol}.", "danger")
            return redirect(url_for('index', stock=stock_symbol))
            
        # Reset index to make Date a column for the model
        df = df.reset_index()
        
        app.logger.info(f"Final dataframe shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Create and train XGBoost model
        model = XGBoostStockPredictor(model_type=prediction_type)
        metrics, X_test, y_test, y_pred, y_pred_proba = model.train(df, auto_optimize=auto_optimize)
        
        # Generate feature importance plot
        plot_result = model.plot_feature_importance(top_n=15)
        feature_importance_path = plot_result.get('image_path', '').replace('static/', '')
        
        # Save model
        model_dir = os.path.join('models', 'xgboost')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{stock_symbol}_{prediction_type}_{int(time.time())}.joblib")
        model.save_model(model_path)
        
        # Format prediction results
        prediction_results = []
        for i in range(min(10, len(X_test))):  # Show first 10 predictions
            date = X_test.index[i].strftime('%Y-%m-%d') if hasattr(X_test.index[i], 'strftime') else str(X_test.index[i])
            actual = float(y_test.iloc[i]) if hasattr(y_test, 'iloc') else float(y_test[i])
            predicted = float(y_pred[i])
            
            if prediction_type == 'regressor':
                error = actual - predicted
                error_pct = (error / actual) * 100 if actual != 0 else float('inf')
                result = {
                    'date': date,
                    'actual': round(actual, 2),
                    'predicted': round(predicted, 2),
                    'error': round(error, 2),
                    'error_pct': round(error_pct, 2)
                }
            else:  # classifier
                actual_label = 'Up' if actual == 1 else 'Down'
                predicted_label = 'Up' if predicted == 1 else 'Down'
                confidence = float(y_pred_proba[i][1]) if y_pred_proba is not None else None
                result = {
                    'date': date,
                    'actual': actual_label,
                    'predicted': predicted_label,
                    'correct': actual == predicted,
                    'confidence': round(confidence * 100, 2) if confidence is not None else None
                }
            
            prediction_results.append(result)
        
        # Format metrics
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                formatted_metrics[key] = round(float(value), 4)
            else:
                formatted_metrics[key] = value
        
        # Render template
        context = {
            'stock': stock_symbol,
            'model_type': prediction_type,
            'metrics': formatted_metrics,
            'feature_importance_path': feature_importance_path,
            'prediction_results': prediction_results,
            'current_time': datetime.now(UTC)
        }
        
        return render_template('xgboost_analysis.html', **context)
    
    except Exception as e:
        app.logger.error(f"Error in XGBoost analysis for {stock_symbol}: {e}", exc_info=True)
        flash(f"Error processing XGBoost analysis: {str(e)}", 'danger')
        return redirect(url_for('index', stock=stock_symbol))

# --- New Marketstack Full Analysis Route ---
@app.route('/full_stock_analysis')
def full_stock_analysis():
    """
    Comprehensive stock analysis using Marketstack API.
    This endpoint fetches 10 years of data, technical indicators, and 
    fundamental data for detailed ML/DL analysis.
    """
    symbol = request.args.get('stock')
    if not symbol:
        flash('Stock symbol is required.', 'danger')
        return redirect(url_for('index'))
    
    symbol = symbol.strip().upper()
    app.logger.info(f"Starting full stock analysis for {symbol}")
    
    # Check if Marketstack API is initialized
    if not marketstack:
        flash('Marketstack API is not properly configured. Please check your API key.', 'danger')
        return redirect(url_for('index'))
    
    # Check cache first
    cache_key = f"full_stock_analysis_{symbol}"
    cached_result = cache.get(cache_key)
    
    if cached_result:
        app.logger.info(f"Using cached full analysis for {symbol}")
        return render_template('full_analysis.html', **cached_result)
    
    try:
        # Get complete stock data including 10 years history, technical indicators, and fundamentals
        start_time = time.time()
        app.logger.info(f"Fetching complete stock data for {symbol}")
        result = marketstack.get_complete_stock_data(symbol)
        
        if 'error' in result:
            app.logger.error(f"Error retrieving data for {symbol}: {result['error']}")
            flash(f"Error retrieving data: {result['error']}", 'danger')
            return redirect(url_for('index', stock=symbol))
        
        # Extract key components
        historical_data = result['historical_data']
        enriched_data = result['enriched_data']
        fundamental_data = result['fundamental_data']
        data_range = result['data_range']
        
        # Log data structures to help with debugging
        app.logger.info(f"Historical data for {symbol} - shape: {historical_data.shape}, columns: {historical_data.columns.tolist()}")
        app.logger.info(f"Enriched data for {symbol} - shape: {enriched_data.shape}, columns: {enriched_data.columns.tolist()}")
        
        # Ensure the static directory exists
        chart_dir = 'static/images/analysis/price'
        os.makedirs(chart_dir, exist_ok=True)
        app.logger.info(f"Ensuring chart directory exists: {chart_dir}")
        
        # Generate price chart visualization
        app.logger.info(f"Generating price chart for {symbol}")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create OHLC chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.03, 
                              subplot_titles=('Price', 'Volume'),
                              row_heights=[0.7, 0.3])

            # Log the values we're plotting to check for issues
            app.logger.info(f"Price data sample for {symbol} OHLC chart: {historical_data.head(2).to_string()}")
            app.logger.info(f"Data types: Open: {historical_data['Open'].dtype}, High: {historical_data['High'].dtype}, Low: {historical_data['Low'].dtype}, Close: {historical_data['Close'].dtype}")
            
            fig.add_trace(go.Candlestick(x=historical_data.index,
                                        open=historical_data['Open'],
                                        high=historical_data['High'],
                                        low=historical_data['Low'],
                                        close=historical_data['Close'],
                                        name='OHLC'),
                         row=1, col=1)

            fig.add_trace(go.Bar(x=historical_data.index,
                                y=historical_data['Volume'],
                                name='Volume'),
                         row=2, col=1)

            fig.update_layout(
                title=f'{symbol} - Price History',
                yaxis_title='Price',
                yaxis2_title='Volume',
                xaxis_rangeslider_visible=False,
                height=800
            )
            
            # Save price chart with proper error handling
            price_chart_path = f'static/images/analysis/price/price_chart_{symbol}.png'
            try:
                app.logger.info(f"Attempting to save chart to {price_chart_path}")
                # Check if kaleido is available
                if 'kaleido' not in sys.modules:
                    app.logger.warning("Kaleido not available, trying to import it...")
                    import kaleido
                
                fig.write_image(price_chart_path, engine='kaleido')
                app.logger.info(f"Successfully saved price chart to {price_chart_path}")
            except Exception as chart_err:
                app.logger.error(f"Error saving price chart: {type(chart_err).__name__} - {chart_err}", exc_info=True)
                
                # Try alternate approach - save as HTML instead
                try:
                    html_path = f'static/images/analysis/price/price_chart_{symbol}.html'
                    app.logger.info(f"Trying to save as HTML instead to {html_path}")
                    fig.write_html(html_path)
                    app.logger.info(f"Saved chart as HTML to {html_path}")
                    price_chart_path = html_path
                except Exception as html_err:
                    app.logger.error(f"Error saving HTML chart: {html_err}")
                    price_chart_path = None
        except Exception as plot_err:
            app.logger.error(f"Error creating plot for {symbol}: {type(plot_err).__name__} - {plot_err}", exc_info=True)
            price_chart_path = None
        
        # Prepare analysis results
        analysis_result = {
            'symbol': symbol,
            'company_name': fundamental_data.get('overview', {}).get('Name', symbol),
            'company_info': fundamental_data.get('overview', {}),
            'price_chart_path': price_chart_path,
            'data_range': data_range,
            'fundamental_data': fundamental_data,
            'technical_indicators': {
                'sma_20': enriched_data['SMA_20'].iloc[-1] if 'SMA_20' in enriched_data else None,
                'rsi_14': enriched_data['RSI_14'].iloc[-1] if 'RSI_14' in enriched_data else None,
                'macd': enriched_data['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in enriched_data else None,
                'volatility': enriched_data['Volatility_20d'].iloc[-1] if 'Volatility_20d' in enriched_data else None
            },
            'execution_time': round(time.time() - start_time, 2),
            'stats': {
                'start_date': historical_data.index.min().strftime('%Y-%m-%d') if not historical_data.empty else 'N/A',
                'end_date': historical_data.index.max().strftime('%Y-%m-%d') if not historical_data.empty else 'N/A',
                'trading_days': len(historical_data),
                'avg_volume': historical_data['Volume'].mean() if 'Volume' in historical_data else 0,
                'min_price': historical_data['Low'].min() if 'Low' in historical_data else 0,
                'max_price': historical_data['High'].max() if 'High' in historical_data else 0,
                'features_count': len(enriched_data.columns),
                'processing_time': round(time.time() - start_time, 2)
            }
        }
        
        # Cache the results
        app.logger.info(f"Caching full analysis result for {symbol}")
        cache.set(cache_key, analysis_result, timeout=STOCK_DATA_CACHE_TIMEOUT)
        
        return render_template('full_analysis.html', **analysis_result)
        
    except Exception as e:
        app.logger.error(f"Error in full stock analysis for {symbol}: {type(e).__name__} - {str(e)}", exc_info=True)
        flash(f"An error occurred during analysis: {str(e)}", 'danger')
        return redirect(url_for('index', stock=symbol))

# --- Main Execution Block ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Financial Analysis Dashboard')
    parser.add_argument('--port', type=int, default=5001, help='Port to run')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    args = parser.parse_args()
    # Kullanım: app.logger Flask uygulamasının logger'ıdır
    app.logger.info(f"Starting Flask app on {args.host}:{args.port} (Debug: True, Reloader: False)")
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)