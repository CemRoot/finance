# app.py
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from flask_caching import Cache
import src.newsapi as newsapi
import src.decision as decision # Decision module
import src.forecasting as forecasting # Forecasting module
import src.sentiment as sentiment # Sentiment module
import json
from datetime import datetime, timedelta # Import specific classes
import time
from functools import wraps
import logging
import os
import pytz
import pandas as pd
import numpy as np
import pandas_ta as ta # Technical Analysis library
import jinja2 # Import jinja2 for exception handling
import yfinance as yf # For direct ticker info access

# Flask Application and Cache Configuration
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-should-be-changed')
cache_config = { "CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300 }
cache = Cache(config=cache_config)
cache.init_app(app)

# --- Popular stock symbols with associated name fragments for lookup ---
POPULAR_SYMBOLS_FOR_NAME_LOOKUP = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "GOOG": ["google", "alphabet"],
    "GOOGL": ["google", "alphabet"],
    "AMZN": ["amazon"],
    "TSLA": ["tesla"],
    "META": ["meta", "facebook"],
    "NVDA": ["nvidia"],
    "JPM": ["jpmorgan", "jp morgan"],
    "BAC": ["bank of america", "bofa"],
    "WFC": ["wells fargo"],
    "C": ["citigroup", "citi"],
    "PFE": ["pfizer"],
    "JNJ": ["johnson", "johnson & johnson"],
    "UNH": ["unitedhealth", "united health"],
    "HD": ["home depot"],
    "PG": ["procter", "gamble", "procter & gamble"],
    "XOM": ["exxon", "exxonmobil"],
    "CVX": ["chevron"],
    "KO": ["coca", "coca-cola", "coca cola"],
    "PEP": ["pepsi", "pepsico"],
    "T": ["at&t", "at and t"],
    "VZ": ["verizon"],
    "DIS": ["disney", "walt disney"],
    "NFLX": ["netflix"],
    "INTC": ["intel"],
    "AMD": ["amd", "advanced micro devices"],
    "CSCO": ["cisco"],
    "IBM": ["ibm", "international business machines"],
    "ORCL": ["oracle"],
    "CRM": ["salesforce"],
    "ADBE": ["adobe"],
    "PYPL": ["paypal"],
    "V": ["visa"],
    "MA": ["mastercard"],
    "WMT": ["walmart", "wal-mart"],
    "TGT": ["target"],
    "COST": ["costco"],
    "SBUX": ["starbucks"],
    "MCD": ["mcdonald", "mcdonalds", "mcdonald's"],
    "BABA": ["alibaba"],
    "GM": ["general motors"],
    "F": ["ford"],
    "GE": ["general electric"],
    "BA": ["boeing"]
}

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
app.logger.handlers.clear()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s] - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


# --- Cached Functions ---
@cache.memoize()
def cached_get_stock_data(symbol):
    app.logger.info(f"CACHE MISS or expired for get_stock_data({symbol}). Calling API.")
    return newsapi.get_stock_data(symbol)

@cache.memoize()
def cached_get_news(symbol):
    app.logger.info(f"CACHE MISS or expired for get_news({symbol}). Calling API.")
    return newsapi.get_news(symbol, lang='en')

@cache.memoize(timeout=3600)  # Cache company names for 1 hour
def cached_get_ticker_info(symbol):
    """
    Get and cache only the company name info for a ticker symbol.
    This is a lightweight alternative to getting all stock data.
    
    Args:
        symbol (str): Stock symbol to lookup
        
    Returns:
        dict: Dictionary with shortName and longName (if available)
    """
    app.logger.info(f"CACHE MISS or expired for ticker_info({symbol}). Fetching company info.")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract only the company name information to save cache space
        name_info = {
            'shortName': info.get('shortName', ''),
            'longName': info.get('longName', ''),
            'symbol': symbol
        }
        return name_info
    except Exception as e:
        app.logger.warning(f"Error fetching ticker info for {symbol}: {str(e)}")
        return {'shortName': '', 'longName': '', 'symbol': symbol}

# --- Jinja2 Filters ---
@app.template_filter('calculate_average')
def calculate_average(values):
    if not values: return 'N/A'
    try:
        valid_values = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
        if not valid_values: return 'N/A'
        avg = sum(valid_values) / len(valid_values)
        if abs(avg) >= 1e9: return f"{avg / 1e9:.1f}B"
        if abs(avg) >= 1e6: return f"{avg / 1e6:.1f}M"
        if abs(avg) >= 1e3: return f"{avg / 1e3:,.0f}K"
        return f"{avg:,.0f}"
    except (ValueError, TypeError, ZeroDivisionError): return 'N/A'

@app.template_filter('safe_sum')
def safe_sum(values):
     if not values: return 0
     try: valid_values = [float(v) for v in values if v is not None and isinstance(v, (int, float))]; return sum(valid_values)
     except (ValueError, TypeError): return 0

@app.template_filter('variance')
def calculate_variance(values):
    if not values: return None
    try: valid_values = [float(v) for v in values if v is not None and isinstance(v, (int, float))]; return np.var(valid_values) if len(valid_values) >= 2 else None
    except (ValueError, TypeError): return None

@app.template_filter('format_number')
def format_number(value, decimals=2):
    if value is None or not isinstance(value, (int, float)): return 'N/A'
    try: format_string = '{:,.%df}' % decimals; return format_string.format(value)
    except (ValueError, TypeError): return 'N/A'

@app.template_filter('format_date')
def format_date(value, format_str='%d %b %Y'):
    if not value: return ''
    try:
        dt_obj = None
        if isinstance(value, datetime): dt_obj = value
        elif isinstance(value, str):
            try: dt_obj = pd.to_datetime(value, errors='raise').to_pydatetime()
            except ValueError: app.logger.debug(f"format_date: Could not parse date string '{value}'"); return value
        else: app.logger.warning(f"format_date: Unsupported type: {type(value)}"); return str(value)
        if dt_obj.tzinfo is None: dt_obj = pytz.utc.localize(dt_obj)
        else: dt_obj = dt_obj.astimezone(pytz.utc)
        return dt_obj.strftime(format_str)
    except Exception as e: app.logger.error(f"Error formatting date '{value}': {e}", exc_info=False); return str(value)

# Register filters and globals
app.jinja_env.filters['calculate_average'] = calculate_average
app.jinja_env.filters['safe_sum'] = safe_sum
app.jinja_env.filters['variance'] = calculate_variance
app.jinja_env.filters['format_number'] = format_number
app.jinja_env.filters['format_date'] = format_date
app.jinja_env.globals.update(abs=abs, len=len, isinstance=isinstance, float=float, int=int, str=str, list=list, dict=dict, datetime=datetime, max=max, min=min, round=round, now=lambda: datetime.now(pytz.utc))

# --- Main Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    stock_symbol = session.get('last_symbol', None)

    # --- Handle POST request (Search) ---
    if request.method == 'POST':
        search_query = request.form.get('stock', '').strip()
        if not search_query:
            flash('Please enter a valid stock symbol or company name.', 'warning')
            return redirect(url_for('index'))
        
        # Convert to uppercase for symbol checking
        stock_symbol_candidate = search_query.upper()
        
        # Check if it looks like a stock symbol (all caps, short)
        is_likely_symbol = (stock_symbol_candidate.isalpha() or 
                           (stock_symbol_candidate.isalnum() and not stock_symbol_candidate.isdigit())) and \
                           len(stock_symbol_candidate) <= 5
        
        resolved_symbol = None
        
        if is_likely_symbol:
            # It looks like a symbol, try to use it directly
            app.logger.info(f"Search query '{search_query}' appears to be a stock symbol.")
            resolved_symbol = stock_symbol_candidate
        else:
            # It might be a company name, try to match with our popular symbols
            app.logger.info(f"Search query '{search_query}' appears to be a company name. Attempting to match...")
            search_query_lower = search_query.lower()
            
            # First check our dictionary with name fragments
            for symbol, name_fragments in POPULAR_SYMBOLS_FOR_NAME_LOOKUP.items():
                if any(fragment in search_query_lower for fragment in name_fragments):
                    app.logger.info(f"Found matching symbol {symbol} for '{search_query}' using name fragment list")
                    resolved_symbol = symbol
                    break
                    
            # If not found in fragments list, try full company name lookup
            if not resolved_symbol:
                for symbol in POPULAR_SYMBOLS_FOR_NAME_LOOKUP.keys():
                    ticker_info = cached_get_ticker_info(symbol)
                    
                    # Check if either shortName or longName contains the search query
                    short_name = ticker_info.get('shortName', '').lower()
                    long_name = ticker_info.get('longName', '').lower() 
                    
                    if search_query_lower in short_name or search_query_lower in long_name:
                        app.logger.info(f"Found matching symbol {symbol} for '{search_query}' in company name")
                        resolved_symbol = symbol
                        break
        
        if not resolved_symbol:
            flash(f"No valid stock symbol found for '{search_query}'. Please try using the exact symbol or another company name.", 'warning')
            return redirect(url_for('index'))

        # We have a valid symbol, store it and redirect
        session['last_symbol'] = resolved_symbol
        app.logger.info(f"Search query '{search_query}' resolved to symbol '{resolved_symbol}'. Redirecting after clearing cache.")
        
        # Clear cache for this symbol to ensure fresh data
        cache_key_data = f"cached_get_stock_data_{resolved_symbol}"
        cache_key_news = f"cached_get_news_{resolved_symbol}"
        cache_key_forecast = f"forecast_{resolved_symbol}"
        cache.delete(cache_key_data)
        cache.delete(cache_key_news)
        cache.delete(cache_key_forecast)
        
        return redirect(url_for('index', stock=resolved_symbol))

    # --- Process GET request ---
    stock_symbol_arg = request.args.get('stock')
    if stock_symbol_arg:
        stock_symbol = stock_symbol_arg.strip().upper()
        session['last_symbol'] = stock_symbol
    elif not stock_symbol:
        current_utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)
        return render_template('index.html', stock=None, current_time=current_utc_time)

    # --- Define variables with defaults BEFORE the main try block ---
    stock_data = None; articles = []; tech_indicators = {}; avg_sentiment = 0.0
    decision_result = "Analysis Pending"; error_occurred = False; error_message = None
    current_utc_time = datetime.utcnow().replace(tzinfo=pytz.utc) # Defined here

    app.logger.info(f"Processing GET request for symbol: {stock_symbol}")

    # --- Main Data Fetching and Processing Block ---
    try:
        # 1. Get Stock Data
        stock_data = cached_get_stock_data(stock_symbol)
        if stock_data.get('error'): raise ValueError(stock_data['error'])
        if not stock_data or not stock_data.get('labels') or not stock_data.get('values'): raise ValueError(f"Stock data for '{stock_symbol}' missing/incomplete.")

        # 2. Calculate Technical Indicators
        try:
            df = pd.DataFrame({'Open': stock_data.get('open_values'), 'High': stock_data.get('high_values'), 'Low': stock_data.get('low_values'), 'Close': stock_data.get('values'), 'Volume': stock_data.get('volume_values')}, index=pd.to_datetime(stock_data.get('labels'), format='ISO8601', errors='coerce'))
            df = df[pd.notna(df.index)]; df.dropna(subset=['Close'], inplace=True)

            if not df.empty:
                app.logger.info(f"Calculating TA indicators for {stock_symbol} with {len(df)} data points.")
                min_rows_for_full_ta = 20
                if len(df) >= 2: df['returns'] = df['Close'].pct_change(); std_dev = df['returns'].std(); tech_indicators['volatility'] = std_dev * np.sqrt(252) if pd.notna(std_dev) else None
                if len(df) >= min_rows_for_full_ta:
                    df.ta.sma(length=20, append=True); df.ta.rsi(length=14, append=True)
                    df.ta.macd(fast=12, slow=26, signal=9, append=True); df.ta.bbands(length=20, std=2, append=True)
                    # Debug print kaldırıldı, sütun adlarının doğru olduğu varsayıldı
                    last = df.iloc[-1]
                    cp = last.get('Close'); sma20 = last.get('SMA_20'); rsi14 = last.get('RSI_14')
                    macd = last.get('MACD_12_26_9'); macds = last.get('MACDs_12_26_9'); macdh = last.get('MACDh_12_26_9')
                    # *** DÜZELTME: Doğru Bollinger Bant sütun adlarını kullan ***
                    bbl = last.get('BBL_20_2'); bbu = last.get('BBU_20_2'); bbm = last.get('BBM_20_2')
                    # *** DÜZELTME SONU ***
                    if pd.notna(cp) and pd.notna(sma20): tech_indicators['sma'] = 'Above' if cp > sma20 else ('Below' if cp < sma20 else 'Equal')
                    if pd.notna(rsi14): tech_indicators['rsi'] = 'Overbought' if rsi14 > 70 else ('Oversold' if rsi14 < 30 else 'Neutral')
                    if pd.notna(macd) and pd.notna(macds): tech_indicators['macd'] = 'Buy Signal (Positive)' if macd > macds else 'Sell Signal (Negative)'
                    if pd.notna(macdh): tech_indicators['macd_hist'] = float(round(macdh, 3))
                    if pd.notna(cp) and pd.notna(bbl) and pd.notna(bbu): # Assign bbands status
                        if cp < bbl: tech_indicators['bbands'] = 'Below Lower Band'
                        elif cp > bbu: tech_indicators['bbands'] = 'Above Upper Band'
                        else: tech_indicators['bbands'] = 'Inside Bands'
                    # *** DÜZELTME: Hatalı logger çağrısı kaldırıldı ***
                    # else: app.logger.debug(...) # Kaldırıldı veya app.logger ile düzeltildi
                    if pd.notna(bbm): tech_indicators['bbands_mid'] = float(round(bbm, 2)) # Assign bbands_mid
                    # else: app.logger.debug(...) # Kaldırıldı veya app.logger ile düzeltildi
                else: app.logger.warning(f"Insufficient data ({len(df)} rows) for full TA for {stock_symbol}"); flash(f"Limited data for {stock_symbol}, some indicators unavailable.", 'info')
            else: app.logger.warning(f"DataFrame empty after cleaning for {stock_symbol}. Skipping TA.")
        # *** Granular except block for TA errors ***
        except Exception as ta_error:
             # Log the error with app.logger
             app.logger.error(f"Error calculating TA for {stock_symbol}: {ta_error}", exc_info=True)
             # Flash a user-friendly warning
             flash(f"Warning: Could not calculate technical indicators.", 'warning')
             # Reset tech_indicators to ensure it's empty, preventing partial data issues
             tech_indicators = {}

        # 3. Get News (Continues even if TA fails)
        articles = cached_get_news(stock_symbol)
        if isinstance(articles, dict) and articles.get('error'): flash(f"Could not fetch news: {articles['error']}", 'warning'); articles = []
        elif not isinstance(articles, list): flash(f"Unexpected news format.", 'warning'); articles = []

        # 4. Calculate Average Sentiment (Continues even if TA fails)
        sentiment_calculated = False
        if articles:
            try: avg_sentiment = sentiment.analyze_articles_sentiment(articles); sentiment_calculated = True; app.logger.info(f"Avg sentiment for {stock_symbol}: {avg_sentiment:.4f}")
            except Exception as sent_err: app.logger.error(f"Error calculating sentiment for {stock_symbol}: {sent_err}", exc_info=True); flash("Error during sentiment analysis.", "warning")

        # 5. Make Decision (Uses whatever indicators and sentiment are available)
        try:
            # decision function should handle potentially empty tech_indicators dict
            decision_result = decision.make_decision_with_indicators(tech_indicators, avg_sentiment)
            app.logger.info(f"Combined decision for {stock_symbol}: {decision_result}")
        except Exception as dec_err:
            decision_result = "Decision Error"
            app.logger.error(f"Error in combined decision logic for {stock_symbol}: {dec_err}", exc_info=True)
            flash("Error during decision analysis.", "warning")

    # --- Handle Errors from Main Data Fetching/Validation Block ---
    except ValueError as ve:
        error_occurred = True; error_message = str(ve)
        app.logger.error(f"Data fetching/validation error for {stock_symbol}: {error_message}")
        flash(error_message, 'danger'); stock_data = None; articles = []; tech_indicators = {}; decision_result = "Error"
    except Exception as e: # Catch-all for other unexpected errors
        error_occurred = True; error_message = f"An unexpected server error occurred."
        app.logger.error(f"Unexpected error processing {stock_symbol}: {type(e).__name__} - {str(e)}", exc_info=True)
        if isinstance(e, jinja2.exceptions.TemplateError): error_message = f"Template rendering error."; app.logger.error(f"Jinja Error: {e.message} (Template: {getattr(e, 'filename', 'N/A')}, Line: {getattr(e, 'lineno', 'N/A')})")
        flash(error_message, 'danger'); stock_data = None; articles = []; tech_indicators = {}; decision_result = "Error"

    # --- Prepare Context and Render ---
    context = { 'stock': stock_symbol, 'stock_data': stock_data, 'articles': articles, 'error': error_occurred, 'error_message': error_message, 'decision': decision_result, 'tech_indicators': tech_indicators, 'current_time': current_utc_time }
    return render_template('index.html', **context)


# --- AJAX / Data Endpoints ---
@app.route('/refresh_stock')
def refresh_stock():
    # ... (Endpoint code remains the same) ...
    stock = request.args.get("stock")
    if not stock: return jsonify({"status": "error", "message": "Stock symbol is required"}), 400
    app.logger.info(f"AJAX request to REFRESH stock data for: {stock}")
    try:
        stock_data = newsapi.get_stock_data(stock)
        if stock_data.get('error'):
            error_msg = stock_data['error']; status_code = 500
            if "not found" in error_msg.lower() or "invalid" in error_msg.lower(): status_code = 404
            elif "rate limit" in error_msg.lower(): status_code = 429
            return jsonify({"status": "error", "message": error_msg}), status_code
        elif not stock_data or not stock_data.get('labels'): return jsonify({"status": "error", "message": f"Refreshed data for {stock} is incomplete."}), 404
        else: cache_key = f"cached_get_stock_data_{stock}"; cache.set(cache_key, stock_data); app.logger.info(f"Cache updated for {stock} after refresh."); return jsonify({"status": "success", "stock_data": stock_data })
    except Exception as e: app.logger.error(f"Server error during stock refresh ({stock}): {str(e)}", exc_info=True); return jsonify({"status": "error", "message": "Server error while refreshing stock data."}), 500

@app.route('/get_forecast_data', methods=['GET'])
def get_forecast_data():
    # ... (Endpoint code remains the same) ...
    symbol = request.args.get('symbol')
    if not symbol: return jsonify({'error': 'Stock symbol is required.'}), 400
    cache_key = f"forecast_{symbol}"; cached_result = cache.get(cache_key)
    if cached_result: app.logger.info(f"Returning cached forecast data for {symbol}"); return jsonify(cached_result)
    app.logger.info(f"CACHE MISS for forecast data: {symbol}. Generating forecast...")
    try:
        forecast_result = forecasting.get_prophet_forecast(symbol, periods=30)
        if forecast_result is None: raise Exception("Forecast function returned None")
        if forecast_result.get('error'):
            error_msg = forecast_result['error']; status_code = 500
            if "veri bulunamadı" in error_msg.lower() or "insufficient" in error_msg.lower() or "not found" in error_msg.lower(): status_code = 404
            elif "rate limit" in error_msg.lower(): status_code = 429
            return jsonify({'error': error_msg}), status_code
        else:
            forecast_result['timestamp'] = datetime.utcnow().replace(tzinfo=pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
            cache.set(cache_key, forecast_result, timeout=3600); app.logger.info(f"Successfully generated forecast for {symbol}.")
            return jsonify(forecast_result)
    except Exception as e: app.logger.error(f"Forecast endpoint error ({symbol}): {str(e)}", exc_info=True); return jsonify({'error': f"Server error generating forecast for {symbol}."}), 500

@app.route('/clear_cache')
def clear_cache_key():
    # ... (Endpoint code remains the same) ...
    key = request.args.get('key')
    if key:
        try:
            prefix = cache.config.get('CACHE_KEY_PREFIX', ''); full_key_guess = prefix + key
            deleted1 = cache.delete(key); deleted2 = cache.delete(full_key_guess)
            if deleted1 or deleted2: app.logger.info(f"Cache key '{key}' deleted via API."); return jsonify({"status": "success", "message": f"Cache key '{key}' cleared."}), 200
            else: app.logger.warning(f"Cache key '{key}' not found for deletion."); return jsonify({"status": "warning", "message": f"Cache key '{key}' not found."}), 404
        except Exception as e: app.logger.error(f"Error deleting cache key '{key}': {e}", exc_info=True); return jsonify({"status": "error", "message": "Error clearing cache key."}), 500
    else: return jsonify({"status": "error", "message": "No cache key provided."}), 400


# --- Main Execution Block ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Financial Analysis Dashboard')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the Flask app on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host (0.0.0.0 for external access)')
    args = parser.parse_args()
    app.logger.info(f"Starting Flask app on host {args.host} port {args.port} with debug=True, use_reloader=False")
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)