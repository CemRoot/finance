# app.py
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from flask_caching import Cache
import src.newsapi as newsapi
import src.decision as decision # Decision module
# Forecasting modülünü ve içindeki fonksiyonu import et
import src.forecasting as forecasting
import src.sentiment as sentiment # Sentiment module
# ML modellerini ve karşılaştırıcıyı import et
from src.ml_models import StockPricePredictor
from src.model_comparison import ModelComparator

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
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
    """Get and cache minimal ticker info."""
    app.logger.info(f"CACHE MISS/expired for ticker_info({symbol}). Fetching.")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info or ticker.info
        name_info = {
            'shortName': info.get('short_name', info.get('shortName', '')),
            'longName': info.get('long_name', info.get('longName', '')),
            'currency': info.get('currency', 'USD'),
            'symbol': symbol }
        if not name_info['shortName'] and not name_info['longName']:
             app.logger.warning(f"Could not retrieve name for {symbol}")
             name_info['shortName'] = symbol
        return name_info
    except Exception as e:
        app.logger.warning(f"Error fetching ticker info for {symbol}: {str(e)}")
        return {'shortName': symbol, 'longName': '', 'symbol': symbol, 'currency': 'USD'}

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
    current_utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)

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

    context = { 'stock': stock_symbol, 'stock_data': stock_data, 'articles': articles, 'error': error_occurred, 'error_message': error_message, 'decision': decision_result, 'tech_indicators': tech_indicators, 'current_time': current_utc_time }
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
            forecast_result['timestamp'] = datetime.utcnow().replace(tzinfo=pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
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
        df = yf.download(stock, period='5y')
        df.reset_index(inplace=True)
        
        # Sütun adlarını düzenle - yfinance bazen sembol adını sütunlara ekler
        if isinstance(df.columns, pd.MultiIndex):
            app.logger.info(f"MultiIndex columns detected, flattening: {df.columns}")
            # MultiIndex'i düzleştir ve istenen sütun adlarına dönüştür
            df.columns = [col[0] if col[0] in ['Date', 'Datetime', 'index'] else col[0] for col in df.columns]
        else:
            # Eğer sütun adları sembol içeriyorsa (_AAPL gibi), bunları kaldır
            rename_dict = {}
            for col in df.columns:
                if '_' in col and col.split('_')[-1] == stock:
                    base_col = col.split('_')[0]
                    rename_dict[col] = base_col
            if rename_dict:
                app.logger.info(f"Renaming columns with symbol suffixes: {rename_dict}")
                df.rename(columns=rename_dict, inplace=True)
        
        app.logger.info(f"Final DataFrame columns: {df.columns.tolist()}")

        min_ml_data_points = 100
        if df.empty or len(df) < min_ml_data_points:
            flash(f"Insufficient historical data ({len(df)} points, need {min_ml_data_points}) for ML analysis for {stock}.", "warning")
            return redirect(url_for('index', stock=stock))

        model = StockPricePredictor(model_type='regressor')
        train_results = model.train(df.copy())
        metrics = train_results[0]
        if not metrics: raise ValueError("Model training failed or returned no metrics.")

        learning_curve_data = model.plot_learning_curve(df.copy())
        feature_importance_data = model.plot_feature_importance()

        app.logger.info(f"RF analysis completed for {stock} (using default/initial params)")
        current_utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)
        return render_template('ml_analysis.html', stock=stock, model_type="Random Forest", metrics=metrics,
                               learning_curve_path=learning_curve_data.get('image_path'),
                               feature_importance_path=feature_importance_data.get('image_path'),
                               current_time=current_utc_time)
    except Exception as e:
        app.logger.error(f"Error in RF analysis for {stock}: {e}", exc_info=True)
        flash(f"Error generating RF analysis: {str(e)}", "danger")
        return redirect(url_for('index', stock=stock))

@app.route('/model_comparison')
def model_comparison():
    stock = request.args.get('stock')
    if not stock: flash("Select stock first.", "warning"); return redirect(url_for('index'))
    stock = stock.strip().upper()

    try:
        app.logger.info(f"Starting model comparison for {stock}")
        comparator = ModelComparator()
        df = comparator._fetch_data(stock, period='5y')

        min_ml_data_points = 100
        if df is None or len(df) < min_ml_data_points:
             flash(f"Insufficient data ({len(df) if df is not None else 0} points, need {min_ml_data_points}) for comparison for {stock}.", "warning")
             return redirect(url_for('index', stock=stock))

        # Run model comparison
        comparison_metrics = comparator.compare_models(stock, df)
        if comparison_metrics.get('error'): 
            raise ValueError(comparison_metrics['error'])

        # Generate comparison plots
        comparison_plot_data = comparator.plot_comparison()
        if comparison_plot_data.get('error'): 
            raise ValueError(comparison_plot_data['error'])

        # Generate performance metrics plot
        performance_plot_data = comparator.plot_performance_comparison()
        if performance_plot_data.get('error'): 
            raise ValueError(performance_plot_data['error'])

        app.logger.info(f"Model comparison completed for {stock}")
        return render_template('model_comparison.html', 
                               stock=stock,
                               comparison_image=comparison_plot_data.get('image_path', '/static/images/model_comparison.png'),
                               performance_image=performance_plot_data.get('image_path', '/static/images/performance_comparison.png'),
                               metrics=performance_plot_data.get('metrics', {}))
    except Exception as e:
        app.logger.error(f"Error during model comparison for {stock}: {e}", exc_info=True)
        flash(f"Error generating model comparison: {str(e)}", "danger")
        return redirect(url_for('index', stock=stock))

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