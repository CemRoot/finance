# src/newsapi.py
import requests
import yfinance as yf
from datetime import datetime, timedelta, date # date eklendi
import pytz
import time
from random import uniform
import logging
import pandas as pd
from config import NEWSAPI_KEY, NEWSAPI_ENDPOINT, API_TIMEOUT, YFINANCE_MIN_INTERVAL # Import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LAST_YFINANCE_REQUEST_TIME = 0
LAST_NEWSAPI_REQUEST_TIME = 0

# *** FIX: Renamed from _rate_limit to rate_limit ***
def rate_limit(api_type='yfinance'):
    """Rate limiting helper function"""
    global LAST_YFINANCE_REQUEST_TIME, LAST_NEWSAPI_REQUEST_TIME
    current_time = time.time(); enforced = False
    last_request_time = 0; min_interval = 0

    if api_type == 'yfinance':
        min_interval = YFINANCE_MIN_INTERVAL; last_request_time = LAST_YFINANCE_REQUEST_TIME
    elif api_type == 'newsapi':
        min_interval = 1.5; last_request_time = LAST_NEWSAPI_REQUEST_TIME
    else: return

    time_since_last_request = current_time - last_request_time
    if time_since_last_request < min_interval:
        sleep_time = min_interval - time_since_last_request + uniform(0.1, 0.6)
        logger.warning(f"Rate limiting for {api_type}: sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time); enforced = True

    new_time = time.time()
    if api_type == 'yfinance': LAST_YFINANCE_REQUEST_TIME = new_time
    elif api_type == 'newsapi': LAST_NEWSAPI_REQUEST_TIME = new_time
    if enforced: logger.info(f"Rate limit enforced for {api_type}. Resuming...")

def get_news(stock, lang='en'):
    """Fetches news using the EventRegistry API."""
    if not stock: logger.error("Cannot fetch news: Stock symbol is empty"); return []
    result = []
    try:
        # *** FIX: Call updated function name ***
        rate_limit('newsapi')

        params = {
            'apiKey': NEWSAPI_KEY, 'resultType': 'articles', 'articlesPage': 1,
            'articlesCount': 20, 'articlesSortBy': 'date', 'articlesSortByAsc': False,
            'articleBodyLen': -1, 'dataType': ['news'], 'forceMaxDataTimeWindow': 7,
            'lang': lang, 'keyword': stock, #'categoryUri': 'dmoz/Business/Financial_Services',
            'minSentiment': -1.0, 'maxSentiment': 1.0, 'dateEnd': date.today().strftime('%Y-%m-%d')
        }
        params = {k: v for k, v in params.items() if v is not None and v != ''}

        logger.info(f"Sending EventRegistry API request: {NEWSAPI_ENDPOINT} for {stock} (lang={lang}) with params: {params}")
        response = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=API_TIMEOUT)
        logger.info(f"EventRegistry API response code ({stock}): {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', {}).get('results', [])
            logger.info(f"Number of news found ({stock}): {len(articles)}")
            if len(articles) == 0: logger.warning(f"No news articles returned by EventRegistry for query: {params}")

            for article in articles:
                if not isinstance(article, dict): continue
                title = article.get('title', 'No Title'); description = article.get('body', 'No Description')
                url = article.get('url', '#'); date_str = article.get('dateTimePub') or article.get('date')
                source = article.get('source', {}).get('title', 'Unknown Source'); image_url = article.get('image')
                published_at_str = article.get('dateTime')
                description = (description[:250] + '...') if description and len(description) > 250 else description
                parsed_date = None
                if date_str:
                    try:
                        if 'T' in date_str: parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).astimezone(pytz.utc)
                        else: parsed_date = pytz.utc.localize(datetime.strptime(date_str, '%Y-%m-%d'))
                    except ValueError: logger.warning(f"Could not parse news date ({stock}): {date_str}"); parsed_date = date_str
                parsed_published_at = None
                if published_at_str:
                     try: parsed_published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00')).astimezone(pytz.utc)
                     except ValueError: logger.warning(f"Could not parse news published_at date ({stock}): {published_at_str}"); parsed_published_at = published_at_str
                result.append({'title': title, 'description': description, 'url': url, 'date': parsed_date, 'published_at': parsed_published_at or parsed_date, 'source': source, 'image_url': image_url if image_url else ''})

            logger.info(f"Number of processed news ({stock}): {len(result)}")
            result.sort(key=lambda x: x['date'] if isinstance(x['date'], datetime) else datetime.min.replace(tzinfo=pytz.utc), reverse=True)
            return result
        elif response.status_code == 429:
            logger.error(f"EventRegistry API rate limit exceeded (HTTP 429).")
            return {'error': 'News API rate limit exceeded.'}
        else:
            logger.error(f"EventRegistry API error ({stock}): HTTP {response.status_code}")
            error_detail = "Unknown API error"
            try:
                error_detail = response.json()
            except:
                pass
            return {'error': f"News API error (HTTP {response.status_code}). Detail: {error_detail}"}
    except requests.exceptions.Timeout: logger.error(f"Timeout fetching news ({stock})"); return {'error': 'Timeout fetching news.'}
    except requests.exceptions.RequestException as e: logger.error(f"Network error fetching news ({stock}): {e}"); return {'error': f'Network error: {e}'}
    except Exception as e: logger.error(f"General error fetching news ({stock}): {e}", exc_info=True); return {'error': 'Unexpected error fetching news.'}


def get_stock_data(stock):
    """Fetches stock data using yfinance (called by cache in app.py)."""
    logger.info(f"Starting data fetch for {stock}")
    empty_result = {
        'labels': [], 'values': [], 'open_values': [], 'high_values': [], 'low_values': [], 'volume_values': [], 'candlestick_data': [],
        'market_status': 'UNKNOWN', 'current_price': None, 'change_percent': None, 'company_name': stock, 'currency': None,
        'timestamp': datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'), # Use datetime class here
        'error': f"Symbol '{stock}' not found or invalid." }
    try:
        # *** FIX: Call updated function name ***
        rate_limit('yfinance')
        logger.info(f"Fetching ticker data for {stock} with extended timeout")
        ticker = yf.Ticker(stock)
        info = None; fast_info = None
        try:
            logger.info(f"Attempting to get basic info for {stock}")
            fast_info = ticker.fast_info; logger.info(f"Got fast_info for {stock}: {len(str(fast_info)) if fast_info else 'None'} chars")
            info = ticker.info; logger.info(f"Successfully retrieved info for {stock}: {len(str(info)) if info else 'None'} chars")
        except Exception as info_err:
            logger.error(f"Failed to get ticker info for {stock}: {info_err}", exc_info=False)
            try:
                if fast_info and isinstance(fast_info, dict) and fast_info.get('last_price'):
                    logger.info(f"Using fast_info as fallback for {stock}")
                    info = {"regularMarketPrice": fast_info.get('last_price'), "previousClose": fast_info.get('previous_close'), "marketState": fast_info.get('market_state', 'UNKNOWN'), "shortName": fast_info.get('name', stock), "currency": fast_info.get('currency')}
                else: raise ValueError("No ticker information available")
            except Exception as fallback_err: logger.error(f"Both info methods failed: {fallback_err}"); empty_result['error'] = f"Could not retrieve basic info for '{stock}'."; return empty_result

        if not info or not isinstance(info, dict) or len(info) < 2:
            logger.error(f"Invalid or insufficient info data for {stock}"); empty_result['error'] = f"Insufficient data for '{stock}'."; empty_result['company_name'] = info.get('shortName', info.get('longName', stock)) if isinstance(info, dict) else stock; return empty_result

        market_status = info.get('marketState', 'UNKNOWN').upper(); logger.info(f"Market status for {stock}: {market_status}")
        hist = None; logger.info(f"Attempting to fetch historical data for {stock}")
        for period in ['6mo', '1y', '3mo', '1mo']:
            try:
                logger.info(f"Fetching {period} historical data for {stock}")
                hist = ticker.history(period=period, interval='1d', auto_adjust=True, repair=True, timeout=30)
                if hist is not None and not hist.empty: logger.info(f"Successfully fetched {len(hist)} data points for {stock} using period='{period}'"); break
                else: logger.warning(f"Empty data for {period} {stock}")
            except Exception as e_hist: logger.error(f"Fetching {period} history failed: {e_hist}"); hist = None
            if hist is None or hist.empty: time.sleep(uniform(0.5, 1.0))

        if hist is None or hist.empty:
            try:
                logger.info(f"Attempting backup download (last 90 days) for {stock}")
                end_date = date.today(); start_date = end_date - timedelta(days=90)
                hist_download = yf.download(stock, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False, timeout=30)
                if hist_download is not None and not hist_download.empty: logger.info(f"Backup download successful, {len(hist_download)} rows"); hist = hist_download
                else: logger.error(f"Backup download failed for {stock}")
            except Exception as download_err: logger.error(f"Error in backup download: {download_err}")

        if hist is None or hist.empty:
            logger.error(f"Could not fetch historical data for {stock}"); empty_result['error'] = f"Historical data unavailable for '{stock}'."; empty_result['company_name'] = info.get('shortName', info.get('longName', stock)); empty_result['market_status'] = market_status; return empty_result

        logger.info(f"Processing {len(hist)} historical data points for {stock}")
        try:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in hist.columns: logger.warning(f"Missing column {col}, adding NaN"); hist[col] = float('nan')
            if not isinstance(hist.index, pd.DatetimeIndex): hist.index = pd.to_datetime(hist.index, errors='coerce'); hist = hist[pd.notna(hist.index)]
            if hist.empty: logger.error("Empty DataFrame after date conversion"); empty_result['error'] = "Date processing failed."; return empty_result
            if hist.index.tz is None: labels = hist.index.tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%S.000Z').tolist()
            else: labels = hist.index.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S.000Z').tolist()
            logger.info("Successfully processed dates")
        except Exception as date_fmt_err: logger.error(f"Error formatting dates: {date_fmt_err}", exc_info=True); empty_result['error'] = "Date processing failed."; return empty_result

        hist = hist.ffill().bfill(); hist['Volume'] = hist['Volume'].fillna(0).astype(int)
        def safe_round(v, decimals=2):
            try:
                return round(float(v), decimals) if pd.notna(v) else None
            except:
                return None

        try:
            close_values = [safe_round(v) for v in hist['Close']]; open_values = [safe_round(v) for v in hist['Open']]
            high_values = [safe_round(v) for v in hist['High']]; low_values = [safe_round(v) for v in hist['Low']]
            volume_values = [int(v) if pd.notna(v) else 0 for v in hist['Volume']]
            if not any(v is not None for v in close_values): logger.error("No valid close prices found"); empty_result['error'] = "No valid price data available."; return empty_result
            logger.info("Successfully converted data types")
        except Exception as conv_err: logger.error(f"Error converting data types: {conv_err}", exc_info=True); empty_result['error'] = "Data conversion error."; return empty_result

        current_price = None; change_percent = None; prev_close_info = None
        last_valid_close_index = next((i for i, v in enumerate(reversed(close_values)) if v is not None), -1)
        if last_valid_close_index != -1:
            actual_index = len(close_values) - 1 - last_valid_close_index; current_price = close_values[actual_index]; prev_close_hist = None
            if actual_index > 0: prev_close_hist = next((v for v in reversed(close_values[:actual_index]) if v is not None), None)
            try:
                 prev_close_info = info.get('previousClose')
                 if prev_close_info is not None and prev_close_info != 0: change_percent = ((current_price - prev_close_info) / prev_close_info * 100); logger.debug(f"Used info.previousClose ({prev_close_info})")
                 elif fast_info and fast_info.get('previous_close') is not None and fast_info['previous_close'] != 0: prev_close_fast = fast_info['previous_close']; change_percent = ((current_price - prev_close_fast) / prev_close_fast * 100); logger.debug(f"Used fast_info.previous_close ({prev_close_fast})")
                 elif prev_close_hist is not None and prev_close_hist != 0: change_percent = ((current_price - prev_close_hist) / prev_close_hist * 100); logger.debug(f"Used historical prev close ({prev_close_hist})")
                 else: logger.warning("Could not determine previous close price")
            except Exception as prev_err:
                 logger.warning(f"Error accessing previous close: {prev_err}")
                 if prev_close_hist is not None and prev_close_hist != 0: change_percent = ((current_price - prev_close_hist) / prev_close_hist * 100); logger.debug(f"Used historical prev close ({prev_close_hist}) after error")
                 else: logger.warning("Could not determine previous close after error")
        else:
            logger.warning("No valid close price in history"); current_price = info.get('regularMarketPrice', info.get('currentPrice')); prev_close_info = info.get('previousClose')
            if current_price is not None and prev_close_info is not None and prev_close_info != 0: change_percent = ((current_price - prev_close_info) / prev_close_info * 100); logger.debug(f"Used info price ({current_price}) and prev close ({prev_close_info})")

        candlestick_data = [{'t': labels[i], 'o': open_values[i], 'h': high_values[i], 'l': low_values[i], 'c': close_values[i]} for i in range(len(labels)) if all(pd.notna(v) for v in [open_values[i], high_values[i], low_values[i], close_values[i]])]

        company_name = info.get('shortName', info.get('longName', stock)); currency = info.get('currency', 'USD')
        timestamp_now = datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z') # Use datetime class here

        logger.info(f"Successfully processed {stock}. Candles: {len(candlestick_data)}, Status: {market_status}")
        return { 'labels': labels, 'values': close_values, 'open_values': open_values, 'high_values': high_values, 'low_values': low_values, 'volume_values': volume_values, 'candlestick_data': candlestick_data, 'market_status': market_status, 'current_price': safe_round(current_price), 'change_percent': safe_round(change_percent) if change_percent is not None else None, 'company_name': company_name, 'currency': currency, 'timestamp': timestamp_now, 'error': None }
    except Exception as e:
        logger.error(f"Unexpected error processing '{stock}': {type(e).__name__} - {str(e)}", exc_info=True)
        final_error_result = empty_result.copy(); final_error_result['error'] = f"Unexpected server error processing '{stock}'. Please try again."
        try:
            if 'info' in locals() and info and isinstance(info, dict): final_error_result['company_name'] = info.get('shortName', info.get('longName', stock)); final_error_result['market_status'] = info.get('marketState', 'UNKNOWN').upper(); final_error_result['currency'] = info.get('currency')
        except: pass
        return final_error_result