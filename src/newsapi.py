# src/newsapi.py
import requests
import yfinance as yf
from config import NEWSAPI_KEY, NEWSAPI_ENDPOINT, API_TIMEOUT, YFINANCE_MIN_INTERVAL
import datetime
import pytz
import time
from random import uniform
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LAST_YFINANCE_REQUEST_TIME = 0
LAST_NEWSAPI_REQUEST_TIME = 0

def _rate_limit(api_type='yfinance'):
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

def get_news(stock, lang='en'): # Changed default lang to 'en'
    """Fetches news using the EventRegistry API."""
    if not stock: logger.error("Cannot fetch news: Stock symbol is empty"); return []
    result = []
    try:
        _rate_limit('newsapi')
        params = {
            'apiKey': NEWSAPI_KEY, 'resultType': 'articles', 'articlesPage': 1,
            'articlesCount': 10, 'articlesSortBy': 'date', 'articlesSortByAsc': False,
            'articleBodyLen': 250, 'dataType': 'news',
            'forceMaxDataTimeWindow': 3, # Last 3 days
            'lang': lang, # Use language from function argument
            'keyword': stock
        }
        logger.info(f"Sending EventRegistry API request: {NEWSAPI_ENDPOINT} for {stock} (lang={lang})")
        response = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=API_TIMEOUT)
        logger.info(f"EventRegistry API response code ({stock}): {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', {}).get('results', [])
            logger.info(f"Number of news found ({stock}): {len(articles)}")
            for article in articles:
                if not isinstance(article, dict): continue
                title = article.get('title', 'Title not found'); description = article.get('body', 'No description'); url = article.get('url', '#')
                # *** FIX: Use dateTimePub first, then date ***
                date_str = article.get('dateTimePub') or article.get('date'); source = article.get('source', {}).get('title', 'Unknown Source'); image_url = article.get('image')
                published_at_str = article.get('dateTime') # EventRegistry specific publish time

                parsed_date = None
                if date_str:
                    try:
                        # Try ISO format first (common)
                        if 'T' in date_str: parsed_date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00')).astimezone(pytz.utc)
                        # Then try YYYY-MM-DD
                        else: parsed_date = pytz.utc.localize(datetime.datetime.strptime(date_str, '%Y-%m-%d'))
                    except ValueError: logger.warning(f"Could not parse news date ({stock}): {date_str}"); parsed_date = date_str # Keep original string if parsing fails

                parsed_published_at = None
                if published_at_str:
                     try: parsed_published_at = datetime.datetime.fromisoformat(published_at_str.replace('Z', '+00:00')).astimezone(pytz.utc)
                     except ValueError: logger.warning(f"Could not parse news published_at date ({stock}): {published_at_str}"); parsed_published_at = published_at_str

                result.append({
                    'title': title, 'description': description, 'url': url,
                    'date': parsed_date, # Main date used for sorting
                    'published_at': parsed_published_at or parsed_date, # Fallback for display if specific time exists
                    'source': source,
                    'image_url': image_url if image_url else ''
                })
            logger.info(f"Number of processed news ({stock}): {len(result)}")
            # Sort by date (datetime object priority)
            result.sort(key=lambda x: x['date'] if isinstance(x['date'], datetime.datetime) else datetime.datetime.min.replace(tzinfo=pytz.utc), reverse=True)
            return result
        elif response.status_code == 429:
            logger.error(f"EventRegistry API rate limit exceeded (HTTP 429).")
            # *** FIX: Use English error message ***
            return {'error': 'News API rate limit exceeded. Please try again later.'} # Return error dict
        else:
            logger.error(f"EventRegistry API error ({stock}): HTTP {response.status_code}")
            error_detail = "Unknown API error"
            try: error_detail = response.json(); logger.error(f"Error detail ({stock}): {error_detail}")
            except Exception as ex: logger.error(f"Could not get error detail ({stock}): {ex}")
            # *** FIX: Use English error message ***
            return {'error': f"News API error (HTTP {response.status_code}). Detail: {error_detail}"} # Return error dict
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching news ({stock}) : {NEWSAPI_ENDPOINT}")
        # *** FIX: Use English error message ***
        return {'error': 'Timeout while fetching news.'}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while fetching news ({stock}): {str(e)}")
        # *** FIX: Use English error message ***
        return {'error': f'Network error while fetching news: {e}'}
    except Exception as e:
        logger.error(f"General error while fetching news ({stock}): {str(e)}", exc_info=True)
        # *** FIX: Use English error message ***
        return {'error': f'An unexpected error occurred while fetching news.'}


def get_stock_data(stock):
    """Fetches stock data using yfinance (called by cache in app.py)."""
    logger.info(f"Starting data fetch for {stock}")

    # *** FIX: Use English error message ***
    empty_result = {
        'labels': [], 'values': [], 'open_values': [], 'high_values': [],
        'low_values': [], 'volume_values': [], 'candlestick_data': [],
        'market_status': 'UNKNOWN', 'current_price': None, 'change_percent': None,
        'company_name': stock, 'currency': None, # Add currency here too
        'timestamp': datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'error': f"Symbol '{stock}' not found or invalid."
    }

    try:
        _rate_limit('yfinance')
        logger.info(f"Fetching ticker data for {stock} with extended timeout")
        ticker = yf.Ticker(stock)

        info = None
        fast_info = None
        try:
            logger.info(f"Attempting to get basic info for {stock}")
            # Try fast_info first as it's sometimes more reliable
            fast_info = ticker.fast_info
            logger.info(f"Got fast_info for {stock}: {len(str(fast_info)) if fast_info else 'None'} chars")

            # Now try to get the full info
            info = ticker.info
            logger.info(f"Successfully retrieved info for {stock}: {len(str(info)) if info else 'None'} chars")
        except Exception as info_err:
            logger.error(f"Failed to get ticker info for {stock}: {info_err}", exc_info=False) # Less noisy log
            # If fast_info worked but regular info failed, use what we have
            try:
                if fast_info and isinstance(fast_info, dict) and fast_info.get('last_price'): # Check if fast_info is useful
                    logger.info(f"Using fast_info as fallback for {stock}")
                    # Construct a basic info dict from fast_info
                    info = {
                        "regularMarketPrice": fast_info.get('last_price'),
                        "previousClose": fast_info.get('previous_close'),
                        "marketState": fast_info.get('market_state', 'UNKNOWN'), # Use market_state if available
                        "shortName": fast_info.get('name', stock),
                        "currency": fast_info.get('currency') # Get currency if available
                    }
                else:
                    # If even fast_info failed, raise an error to be caught below
                    raise ValueError("No ticker information available from fast_info or info")
            except Exception as fallback_err:
                logger.error(f"Both info methods failed or provided insufficient data for {stock}: {fallback_err}")
                 # *** FIX: Use English error message ***
                empty_result['error'] = f"Could not retrieve basic info for '{stock}'. It might be delisted or invalid."
                return empty_result

        # Check if essential info is present
        if not info or not isinstance(info, dict) or len(info) < 2: # More robust check
            logger.error(f"Invalid or insufficient info data retrieved for {stock}")
             # *** FIX: Use English error message ***
            empty_result['error'] = f"Insufficient data found for '{stock}'. Symbol might be invalid or delisted."
            # Try to get name even if other info failed
            empty_result['company_name'] = info.get('shortName', info.get('longName', stock)) if isinstance(info, dict) else stock
            return empty_result

        market_status = info.get('marketState', 'UNKNOWN').upper()
        logger.info(f"Market status for {stock}: {market_status}")

        # Try multiple period lengths to get historical data
        hist = None
        logger.info(f"Attempting to fetch historical data for {stock}")
        # *** FIX: Use a more common default period first like 6mo or 1y ***
        for period in ['6mo', '1y', '3mo', '1mo']:
            try:
                logger.info(f"Fetching {period} historical data for {stock}")
                hist = ticker.history(period=period, interval='1d',
                                     auto_adjust=True,
                                     # back_adjust=True, # auto_adjust handles splits/dividends
                                     repair=True,
                                     timeout=30) # Increased timeout
                if hist is not None and not hist.empty:
                    logger.info(f"Successfully fetched {len(hist)} data points for {stock} using period='{period}'")
                    break # Exit loop once data is fetched
                else:
                    logger.warning(f"Empty or no data returned for {period} {stock}")
            except Exception as e_hist:
                logger.error(f"Fetching {period} history failed for {stock}: {type(e_hist).__name__} - {e_hist}")
                hist = None # Ensure hist is None on failure
            # Add a small delay if a period fails before trying the next
            if hist is None or hist.empty: time.sleep(uniform(0.5, 1.0))

        # Backup download method (less reliable for intervals/periods)
        if hist is None or hist.empty:
            try:
                logger.info(f"Attempting backup download method (last 90 days) for {stock}")
                end_date = datetime.now()
                start_date = end_date - datetime.timedelta(days=90)
                hist_download = yf.download(
                    stock,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False,
                    timeout=30
                )
                if hist_download is not None and not hist_download.empty:
                    logger.info(f"Backup download successful for {stock}, got {len(hist_download)} rows")
                    hist = hist_download # Use downloaded data
                else: logger.error(f"Backup download also failed or returned empty for {stock}")
            except Exception as download_err: logger.error(f"Error in backup download for {stock}: {download_err}")

        # Final check for historical data
        if hist is None or hist.empty:
            logger.error(f"Could not fetch any valid historical data for {stock} after trying multiple methods.")
             # *** FIX: Use English error message ***
            empty_result['error'] = f"Historical price data could not be retrieved for '{stock}'. API might be unavailable or the symbol is problematic."
            empty_result['company_name'] = info.get('shortName', info.get('longName', stock))
            empty_result['market_status'] = market_status
            return empty_result

        # Data processing
        logger.info(f"Processing {len(hist)} historical data points for {stock}")
        try:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in hist.columns:
                    logger.warning(f"Missing column {col} in data for {stock}, adding as NaN")
                    hist[col] = float('nan')

            # Ensure index is DatetimeIndex and convert to UTC ISO format strings
            if not isinstance(hist.index, pd.DatetimeIndex):
                 hist.index = pd.to_datetime(hist.index, errors='coerce')
                 hist = hist[pd.notna(hist.index)] # Remove rows where date parsing failed

            if hist.empty:
                 logger.error(f"Date conversion resulted in empty DataFrame for {stock}")
                 empty_result['error'] = f"Date data for '{stock}' could not be processed."
                 return empty_result

            # Convert timezone IF it exists, otherwise assume UTC? Or local? Let's assume UTC if naive.
            if hist.index.tz is None:
                 labels = hist.index.tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%S.000Z').tolist()
            else:
                 labels = hist.index.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S.000Z').tolist()

            logger.info(f"Successfully processed dates for {stock}")
        except Exception as date_fmt_err:
            logger.error(f"Error formatting dates for {stock}: {date_fmt_err}", exc_info=True)
             # *** FIX: Use English error message ***
            empty_result['error'] = f"Date data for '{stock}' could not be processed."
            return empty_result

        # Fill missing values
        hist = hist.ffill() # Forward fill first
        hist = hist.bfill() # Then back fill any remaining NaNs at the beginning
        # Ensure Volume is integer, fill NaNs with 0 before converting
        hist['Volume'] = hist['Volume'].fillna(0).astype(int)

        # Safely convert OHLCV data to floats/ints, rounding prices
        def safe_round(v, decimals=2):
            try: return round(float(v), decimals) if pd.notna(v) else None
            except (TypeError, ValueError): return None

        try:
            close_values = [safe_round(v) for v in hist['Close']]
            open_values = [safe_round(v) for v in hist['Open']]
            high_values = [safe_round(v) for v in hist['High']]
            low_values = [safe_round(v) for v in hist['Low']]
            volume_values = [int(v) if pd.notna(v) else 0 for v in hist['Volume']]

            # Check if we got any valid close prices at all
            if not any(v is not None for v in close_values):
                logger.error(f"No valid close prices found after processing for {stock}")
                # *** FIX: Use English error message ***
                empty_result['error'] = f"No valid price data available for '{stock}'."
                return empty_result
            logger.info(f"Successfully converted data types for {stock}")
        except Exception as conv_err:
            logger.error(f"Error converting OHLCV data types for {stock}: {conv_err}", exc_info=True)
             # *** FIX: Use English error message ***
            empty_result['error'] = f"Error during data conversion for '{stock}'."
            return empty_result

        # Calculate current price and change percent
        current_price = None
        change_percent = None
        # *** FIX: Initialize prev_close_info BEFORE the logic block ***
        prev_close_info = None

        # Find the last valid close price from history
        last_valid_close_index = next((i for i, v in enumerate(reversed(close_values)) if v is not None), -1)

        if last_valid_close_index != -1:
            # Index needs to be adjusted because we reversed the list
            actual_index = len(close_values) - 1 - last_valid_close_index
            current_price = close_values[actual_index]

            # Find the second to last valid close price from history for change calculation
            prev_close_hist = None
            if actual_index > 0:
                 prev_close_hist = next((v for v in reversed(close_values[:actual_index]) if v is not None), None)

            # Try using info/fast_info first for previous close, fall back to history
            try:
                 prev_close_info = info.get('previousClose') # Try full info
                 if prev_close_info is not None and prev_close_info != 0:
                     change_percent = ((current_price - prev_close_info) / prev_close_info * 100)
                     logger.debug(f"Used info.previousClose ({prev_close_info}) for {stock} change %")
                 elif fast_info and fast_info.get('previous_close') is not None and fast_info['previous_close'] != 0: # Fallback to fast_info
                     prev_close_fast = fast_info['previous_close']
                     change_percent = ((current_price - prev_close_fast) / prev_close_fast * 100)
                     logger.debug(f"Used fast_info.previous_close ({prev_close_fast}) for {stock} change %")
                 elif prev_close_hist is not None and prev_close_hist != 0: # Fallback to historical close
                     change_percent = ((current_price - prev_close_hist) / prev_close_hist * 100)
                     logger.debug(f"Used historical previous close ({prev_close_hist}) for {stock} change %")
                 else:
                     logger.warning(f"Could not determine a valid previous close price for {stock} from info, fast_info, or history.")
            except Exception as prev_err:
                 logger.warning(f"Error accessing previous close price for {stock}: {prev_err}. Falling back to historical if available.")
                 # Fallback to historical if error during info access
                 if prev_close_hist is not None and prev_close_hist != 0:
                     change_percent = ((current_price - prev_close_hist) / prev_close_hist * 100)
                     logger.debug(f"Used historical previous close ({prev_close_hist}) for {stock} change % after error.")
                 else:
                     logger.warning(f"Could not determine previous close after error for {stock}.")

        else:
            logger.warning(f"Could not find last valid close price in historical data for {stock}")
            # Try getting current price from info if history failed
            current_price = info.get('regularMarketPrice', info.get('currentPrice')) # Try different keys
            prev_close_info = info.get('previousClose')
            if current_price is not None and prev_close_info is not None and prev_close_info != 0:
                 change_percent = ((current_price - prev_close_info) / prev_close_info * 100)
                 logger.debug(f"Used info.regularMarketPrice ({current_price}) and info.previousClose ({prev_close_info}) for {stock} change %")


        # Generate candlestick data
        candlestick_data = []
        logger.debug(f"Generating candlestick data for {stock}")
        for i in range(len(labels)):
             # Check index bounds carefully
            if i < len(open_values) and i < len(high_values) and i < len(low_values) and i < len(close_values):
                o, h, l, c = open_values[i], high_values[i], low_values[i], close_values[i]
                # Ensure all OHLC values for this candle are valid numbers
                if all(v is not None and pd.notna(v) for v in [o, h, l, c]):
                    candlestick_data.append({'t': labels[i], 'o': o, 'h': h, 'l': l, 'c': c})

        company_name = info.get('shortName', info.get('longName', stock))
        currency = info.get('currency', 'USD') # Get currency, default to USD
        timestamp_now = datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')

        logger.info(f"Successfully processed stock data for {stock}. Candlestick points: {len(candlestick_data)}, Market Status: {market_status}")

        return {
            'labels': labels, 'values': close_values, 'open_values': open_values,
            'high_values': high_values, 'low_values': low_values, 'volume_values': volume_values,
            'candlestick_data': candlestick_data, 'market_status': market_status,
            'current_price': safe_round(current_price), # Ensure current price is rounded
            'change_percent': safe_round(change_percent) if change_percent is not None else None,
            'company_name': company_name,
            'currency': currency, # Add currency
            'timestamp': timestamp_now,
            'error': None # Explicitly set error to None on success
        }

    except Exception as e:
        # *** FIX: Catch specific exceptions if possible, improve logging ***
        logger.error(f"Unexpected server error while processing '{stock}': {type(e).__name__} - {str(e)}", exc_info=True)
        final_error_result = empty_result.copy()
        # *** FIX: Use English error message ***
        final_error_result['error'] = f"An unexpected server error occurred while processing '{stock}'. Please check the symbol or try again later."
        # Try to populate company name and market status even on error, if info was fetched
        try:
            if 'info' in locals() and info and isinstance(info, dict):
                final_error_result['company_name'] = info.get('shortName', info.get('longName', stock))
                final_error_result['market_status'] = info.get('marketState', 'UNKNOWN').upper()
                final_error_result['currency'] = info.get('currency') # Try to get currency
        except Exception: pass # Ignore errors during error handling
        return final_error_result