# src/newsapi.py
import logging
import time
from datetime import datetime, timedelta, date
from random import uniform

import pandas as pd
import pytz
import requests
import yfinance as yf
from eventregistry import EventRegistry, QueryArticlesIter

from config import (API_TIMEOUT, NEWSAPI_ENDPOINT, NEWSAPI_KEY,
                    YFINANCE_MIN_INTERVAL)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Globals for Rate Limiting ---
LAST_YFINANCE_REQUEST_TIME = 0
LAST_NEWSAPI_REQUEST_TIME = 0

# --- Fallback News Sources (Ensure these URIs are recognized by EventRegistry) ---
FALLBACK_SOURCE_URIS = [
    "reuters.com", "bloomberg.com", "apnews.com", "cnbc.com",
    "marketwatch.com", "investing.com", "finance.yahoo.com",
    "wsj.com", "ft.com", "nytimes.com", 
    "barrons.com", "thestreet.com", "fool.com", "seekingalpha.com",
    "forbes.com", "businessinsider.com", "money.cnn.com", "morningstar.com",
    "economist.com", "zacks.com"
]

# --- Helper Functions ---

def rate_limit(api_type='yfinance'):
    """
    Applies rate limiting by pausing execution if calls are too frequent.

    Args:
        api_type (str): The type of API being called ('yfinance' or 'newsapi').
    """
    global LAST_YFINANCE_REQUEST_TIME, LAST_NEWSAPI_REQUEST_TIME
    current_time = time.time()
    enforced = False
    last_request_time = 0
    min_interval = 0

    if api_type == 'yfinance':
        min_interval = YFINANCE_MIN_INTERVAL
        last_request_time = LAST_YFINANCE_REQUEST_TIME
    elif api_type == 'newsapi':
        min_interval = 1.5  # Specific interval for news API
        last_request_time = LAST_NEWSAPI_REQUEST_TIME
    else:
        return # Unknown API type

    time_since_last_request = current_time - last_request_time
    if time_since_last_request < min_interval:
        sleep_time = min_interval - time_since_last_request + uniform(0.1, 0.6)
        logger.warning(
            "Rate limiting for {}: sleeping for {:.2f} seconds.".format(
                api_type, sleep_time
            )
        )
        time.sleep(sleep_time)
        enforced = True

    # Update last request time after waiting (or immediately if no wait)
    new_time = time.time()
    if api_type == 'yfinance':
        LAST_YFINANCE_REQUEST_TIME = new_time
    elif api_type == 'newsapi':
        LAST_NEWSAPI_REQUEST_TIME = new_time

    if enforced:
        logger.info("Rate limit enforced for {}. Resuming...".format(api_type))


def _parse_news_date(date_str: str, context: str = "news date") -> datetime | str | None:
    """
    Parses a date string from news API into a timezone-aware datetime object.

    Args:
        date_str (str): The date string from the API.
        context (str): Logging context (e.g., 'news date', 'published_at date').

    Returns:
        datetime | str | None: Parsed datetime object (UTC) or original string on failure.
    """
    if not date_str:
        return None
    try:
        # Prefer ISO format with time component
        if 'T' in date_str:
            # Handle 'Z' for UTC or timezone offsets
            dt_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            # Ensure it's UTC
            return dt_obj.astimezone(pytz.utc)
        else:
            # Assume YYYY-MM-DD format, localize to UTC
            dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return pytz.utc.localize(dt_obj)
    except ValueError:
        logger.warning("Could not parse {}: {}".format(context, date_str))
        return date_str # Return original string if parsing fails


def _process_articles(articles: list, stock_symbol: str, is_fallback: bool) -> list:
    """
    Processes a list of raw article dictionaries from EventRegistry API.

    Args:
        articles (list): List of article dictionaries from API response.
        stock_symbol (str): The stock symbol (for logging context).
        is_fallback (bool): Whether these articles are from a fallback search.

    Returns:
        list: A list of processed article dictionaries.
    """
    processed_articles = []
    logger.info("Processing {} articles (Fallback: {})...".format(len(articles), is_fallback))

    for article in articles:
        if not isinstance(article, dict):
            continue

        title = article.get('title', 'No Title Provided')
        description = article.get('body', 'No Description Available')
        url = article.get('url', '#')
        source = article.get('source', {}).get('title', 'Unknown Source')
        image_url = article.get('image') # Can be None

        # Use dateTimePub first, then date for main sorting/display date
        date_str = article.get('dateTimePub') or article.get('date')
        # Use dateTime for more precise published time if available
        published_at_str = article.get('dateTime')

        # Limit description length for display
        if description and len(description) > 250:
            description = description[:250] + '...'

        # Parse dates
        parsed_date = _parse_news_date(date_str, "news date ({})".format(stock_symbol))
        parsed_published_at = _parse_news_date(published_at_str, "published_at date ({})".format(stock_symbol))

        processed_articles.append({
            'title': title,
            'description': description,
            'url': url,
            'date': parsed_date, # Primary date for sorting
            'published_at': parsed_published_at or parsed_date, # Best available publish time
            'source': source,
            'image_url': image_url if image_url else '' # Ensure empty string if None
        })

    logger.info("Number of processed news articles: {}".format(len(processed_articles)))

    # Sort by primary date (most recent first)
    processed_articles.sort(
        key=lambda x: x['date'] if isinstance(x['date'], datetime) else datetime.min.replace(tzinfo=pytz.utc),
        reverse=True
    )
    return processed_articles


def get_news(stock: str, lang: str = 'en') -> list | dict:
    """
    Fetches news using the EventRegistry API with a complex query tailored for stocks.
    If no stock-specific news is found, falls back to fetching general financial news.

    Args:
        stock (str): The stock symbol (e.g., 'AAPL').
        lang (str): Language code (e.g., 'en' for English).

    Returns:
        list | dict: A list of processed articles, or a dictionary with an 'error' key on failure.
    """
    if not stock:
        logger.error("Cannot fetch news: Stock symbol is empty")
        return [] # Return empty list for invalid input

    articles = []
    is_fallback = False
    language_code = 'eng' if lang == 'en' else lang  # EventRegistry uses 'eng' rather than 'en'

    # --- Step 1: Attempt Primary Search for Stock-Specific News ---
    try:
        rate_limit('newsapi')
        
        # Calculate date range (last 14 days - daha uzun bir süre için)
        end_date = date.today()
        start_date = end_date - timedelta(days=14)
        
        # Initialize EventRegistry
        er = EventRegistry(apiKey=NEWSAPI_KEY)
        
        # Build stock-specific query
        query = {
            "$query": {
                "$and": [
                    {
                        "keyword": stock  # Search by stock symbol
                    },
                    {
                        "categoryUri": "dmoz/Business/Financial_Services"
                    },
                    {
                        "$or": [{"sourceUri": source} for source in FALLBACK_SOURCE_URIS]
                    },
                    {
                        "dateStart": start_date.strftime("%Y-%m-%d"),
                        "dateEnd": end_date.strftime("%Y-%m-%d"),
                        "lang": language_code
                    }
                ]
            }
        }
        
        logger.info("Attempting PRIMARY news search for {} with EventRegistry query".format(stock))
        q = QueryArticlesIter.initWithComplexQuery(query)
        
        # Execute the query with a limited number of results
        articles_list = []
        try:
            # Get up to 15 articles (değiştirildi: 20 → 15)
            for article in q.execQuery(er, maxItems=15):
                articles_list.append(article)
            
            articles = articles_list
            logger.info("Primary search found {} news articles for {}.".format(len(articles), stock))
        except Exception as query_err:
            logger.error("Error executing EventRegistry query: {}".format(query_err), exc_info=True)
            return {'error': 'Error querying news API: {}'.format(str(query_err))}

    except Exception as e:
        logger.error("General error during primary news search ({}): {}".format(stock, e), exc_info=True)
        return {'error': 'Unexpected error fetching news: {}'.format(str(e))}

    # --- Step 2: Fallback to General Financial News if Primary Search Found Nothing ---
    if not articles:
        logger.warning("No specific news found for {}. Attempting fallback.".format(stock))
        is_fallback = True
        try:
            rate_limit('newsapi')  # Apply rate limit again for fallback
            
            # Build fallback query (general financial news without stock keyword)
            fallback_query = {
                "$query": {
                    "$and": [
                        {
                            "categoryUri": "dmoz/Business/Financial_Services"
                        },
                        {
                            "$or": [{"sourceUri": source} for source in FALLBACK_SOURCE_URIS]
                        },
                        {
                            "dateStart": start_date.strftime("%Y-%m-%d"),
                            "dateEnd": end_date.strftime("%Y-%m-%d"),
                            "lang": language_code
                        }
                    ]
                }
            }
            
            logger.info("Attempting FALLBACK news search with EventRegistry query")
            fallback_q = QueryArticlesIter.initWithComplexQuery(fallback_query)
            
            # Execute the fallback query
            fallback_articles = []
            for article in fallback_q.execQuery(er, maxItems=15):
                fallback_articles.append(article)
            
            articles = fallback_articles
            logger.info("Fallback search found {} general news articles.".format(len(articles)))

        except Exception as fb_e:
            logger.error("Error during fallback news search: {}".format(fb_e), exc_info=True)
            articles = [] # Ensure articles is empty on fallback exception

    # --- Step 3: Process and Return Articles (either primary or fallback) ---
    return _process_articles(articles, stock, is_fallback)


# --- Stock Data Function (Cleaned - Functionality unchanged from previous fixes) ---
def get_stock_data(stock: str) -> dict:
    """
    Fetches and processes stock data using yfinance.

    Args:
        stock (str): The stock symbol.

    Returns:
        dict: A dictionary containing processed stock data or an error message.
    """
    logger.info("Starting data fetch for {}".format(stock))
    # Use datetime class from import
    empty_result = {
        'labels': [], 'values': [], 'open_values': [], 'high_values': [],
        'low_values': [], 'volume_values': [], 'candlestick_data': [],
        'market_status': 'UNKNOWN', 'current_price': None, 'change_percent': None,
        'company_name': stock, 'currency': None,
        'timestamp': datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'error': "Symbol '{}' not found or invalid.".format(stock)
    }

    try:
        rate_limit('yfinance') # Use renamed function
        logger.info("Fetching ticker data for {} with extended timeout".format(stock))
        ticker = yf.Ticker(stock)
        info = None
        fast_info = None

        # --- Get Ticker Info ---
        try:
            logger.info("Attempting to get basic info for {}".format(stock))
            fast_info = ticker.fast_info
            logger.info("Got fast_info for {}: {} chars".format(stock, len(str(fast_info)) if fast_info else 'None'))
            info = ticker.info
            logger.info("Successfully retrieved info for {}: {} chars".format(stock, len(str(info)) if info else 'None'))
        except Exception as info_err:
            logger.error("Failed to get ticker info for {}: {}".format(stock, info_err), exc_info=False)
            # Fallback using fast_info if primary fails
            try:
                if fast_info and isinstance(fast_info, dict) and fast_info.get('last_price'):
                    logger.info("Using fast_info as fallback for {}".format(stock))
                    info = {
                        "regularMarketPrice": fast_info.get('last_price'),
                        "previousClose": fast_info.get('previous_close'),
                        "marketState": fast_info.get('market_state', 'UNKNOWN'),
                        "shortName": fast_info.get('name', stock),
                        "currency": fast_info.get('currency')
                    }
                else:
                    raise ValueError("No usable ticker information available from fast_info or info.")
            except Exception as fallback_err:
                logger.error("Both info methods failed or insufficient: {}".format(fallback_err))
                empty_result['error'] = "Could not retrieve basic info for '{}'.".format(stock)
                return empty_result

        # --- Validate Info ---
        if not info or not isinstance(info, dict) or len(info) < 2:
            logger.error("Invalid or insufficient info data retrieved for {}".format(stock))
            empty_result['error'] = "Insufficient data found for '{}'.".format(stock)
            empty_result['company_name'] = info.get('shortName', info.get('longName', stock)) if isinstance(info, dict) else stock
            return empty_result

        market_status = info.get('marketState', 'UNKNOWN').upper()
        logger.info("Market status for {}: {}".format(stock, market_status))

        # --- Get Historical Data ---
        hist = None
        logger.info("Attempting to fetch historical data for {}".format(stock))
        for period in ['6mo', '1y', '3mo', '1mo']:
            try:
                logger.info("Fetching {} historical data for {}".format(period, stock))
                hist = ticker.history(period=period, interval='1d', auto_adjust=True, repair=True, timeout=30)
                if hist is not None and not hist.empty:
                    logger.info("Fetched {} data points for {} (period='{}')".format(len(hist), stock, period))
                    break
                else:
                    logger.warning("Empty data for {} {}".format(period, stock))
            except Exception as e_hist:
                logger.error("Fetching {} history failed for {}: {}".format(period, stock, e_hist))
                hist = None
            if hist is None or hist.empty:
                time.sleep(uniform(0.5, 1.0))

        # --- Backup Download Method ---
        if hist is None or hist.empty:
            try:
                logger.info("Attempting backup download (90 days) for {}".format(stock))
                end_date = date.today()
                start_date = end_date - timedelta(days=90)
                hist_download = yf.download(
                    stock, start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"), progress=False, timeout=30
                )
                if hist_download is not None and not hist_download.empty:
                    logger.info("Backup download successful, {} rows".format(len(hist_download)))
                    hist = hist_download
                else:
                    logger.error("Backup download failed for {}".format(stock))
            except Exception as download_err:
                logger.error("Error in backup download for {}: {}".format(stock, download_err))

        # --- Validate History ---
        if hist is None or hist.empty:
            logger.error("Could not fetch historical data for {}".format(stock))
            empty_result['error'] = "Historical data unavailable for '{}'.".format(stock)
            empty_result['company_name'] = info.get('shortName', info.get('longName', stock))
            empty_result['market_status'] = market_status
            return empty_result

        # --- Process Data ---
        logger.info("Processing {} historical data points for {}".format(len(hist), stock))
        try:
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in hist.columns:
                    logger.warning("Missing column {}, adding NaN".format(col))
                    hist[col] = float('nan')

            # Ensure index is DatetimeIndex and format labels
            if not isinstance(hist.index, pd.DatetimeIndex):
                 hist.index = pd.to_datetime(hist.index, errors='coerce')
                 hist = hist[pd.notna(hist.index)]
            if hist.empty:
                raise ValueError("DataFrame empty after date conversion.")

            if hist.index.tz is None:
                 labels = hist.index.tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%S.000Z').tolist()
            else:
                 labels = hist.index.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S.000Z').tolist()
            logger.info("Processed dates successfully.")

        except Exception as date_fmt_err:
            logger.error("Error processing dates: {}".format(date_fmt_err), exc_info=True)
            empty_result['error'] = "Date processing failed for '{}'.".format(stock)
            return empty_result

        # Fill NaNs and ensure types
        hist = hist.ffill().bfill()
        hist['Volume'] = hist['Volume'].fillna(0).astype(int)

        def safe_round(v, decimals=2):
            try:
                return round(float(v), decimals) if pd.notna(v) else None
            except (TypeError, ValueError):
                return None

        try:
            close_values = [safe_round(v) for v in hist['Close']]
            open_values = [safe_round(v) for v in hist['Open']]
            high_values = [safe_round(v) for v in hist['High']]
            low_values = [safe_round(v) for v in hist['Low']]
            volume_values = [int(v) if pd.notna(v) else 0 for v in hist['Volume']]
            if not any(v is not None for v in close_values):
                raise ValueError("No valid close prices found after processing.")
            logger.info("Converted data types successfully.")
        except Exception as conv_err:
            logger.error("Error converting data types: {}".format(conv_err), exc_info=True)
            empty_result['error'] = "Data conversion error for '{}'.".format(stock)
            return empty_result

        # --- Calculate Current Price & Change ---
        current_price = None
        change_percent = None
        prev_close_info = None
        last_valid_close_index = next((i for i, v in enumerate(reversed(close_values)) if v is not None), -1)

        if last_valid_close_index != -1:
            actual_index = len(close_values) - 1 - last_valid_close_index
            current_price = close_values[actual_index]
            prev_close_hist = None
            if actual_index > 0:
                 prev_close_hist = next((v for v in reversed(close_values[:actual_index]) if v is not None), None)
            # Try getting previous close from info/fast_info first
            try:
                 prev_close_info = info.get('previousClose')
                 if prev_close_info is not None and prev_close_info != 0:
                     change_percent = ((current_price - prev_close_info) / prev_close_info * 100)
                     logger.debug("Used info.previousClose ({})".format(prev_close_info))
                 elif fast_info and fast_info.get('previous_close') is not None and fast_info['previous_close'] != 0:
                     prev_close_fast = fast_info['previous_close']
                     change_percent = ((current_price - prev_close_fast) / prev_close_fast * 100)
                     logger.debug("Used fast_info.previous_close ({})".format(prev_close_fast))
                 elif prev_close_hist is not None and prev_close_hist != 0:
                     change_percent = ((current_price - prev_close_hist) / prev_close_hist * 100)
                     logger.debug("Used historical prev close ({})".format(prev_close_hist))
                 else:
                     logger.warning("Could not determine previous close price.")
            except Exception as prev_err:
                 logger.warning("Error accessing previous close: {}. Falling back.".format(prev_err))
                 if prev_close_hist is not None and prev_close_hist != 0:
                     change_percent = ((current_price - prev_close_hist) / prev_close_hist * 100)
                     logger.debug("Used historical prev close ({}) after error.".format(prev_close_hist))
                 else:
                     logger.warning("Could not determine previous close after error.")
        else:
            # Fallback if no history close price found
            logger.warning("No valid close price found in history.")
            current_price = info.get('regularMarketPrice', info.get('currentPrice'))
            prev_close_info = info.get('previousClose')
            if current_price is not None and prev_close_info is not None and prev_close_info != 0:
                 change_percent = ((current_price - prev_close_info) / prev_close_info * 100)
                 logger.debug("Used info price ({}) and prev close ({})".format(current_price, prev_close_info))

        # --- Generate Candlestick Data ---
        candlestick_data = []
        for i in range(len(labels)):
            # Check index bounds
            if i < len(open_values) and i < len(high_values) and i < len(low_values) and i < len(close_values):
                o, h, l, c = open_values[i], high_values[i], low_values[i], close_values[i]
                # Add candle only if all values are valid numbers
                if all(v is not None and pd.notna(v) for v in [o, h, l, c]):
                    candlestick_data.append({'t': labels[i], 'o': o, 'h': h, 'l': l, 'c': c})

        # --- Prepare Final Result ---
        company_name = info.get('shortName', info.get('longName', stock))
        currency = info.get('currency', 'USD')
        timestamp_now = datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')

        logger.info("Successfully processed {}. Candles: {}, Status: {}".format(
            stock, len(candlestick_data), market_status)
        )
        return {
            'labels': labels, 'values': close_values, 'open_values': open_values,
            'high_values': high_values, 'low_values': low_values, 'volume_values': volume_values,
            'candlestick_data': candlestick_data, 'market_status': market_status,
            'current_price': safe_round(current_price),
            'change_percent': safe_round(change_percent) if change_percent is not None else None,
            'company_name': company_name, 'currency': currency,
            'timestamp': timestamp_now, 'error': None
        }

    except Exception as e:
        err_type = type(e).__name__
        logger.error("Unexpected error processing '{}': {} - {}".format(stock, err_type, str(e)), exc_info=True)
        final_error_result = empty_result.copy()
        final_error_result['error'] = "Unexpected server error processing '{}'. Please try again.".format(stock)
        # Try to add partial info if available
        try:
            if 'info' in locals() and info and isinstance(info, dict):
                final_error_result['company_name'] = info.get('shortName', info.get('longName', stock))
                final_error_result['market_status'] = info.get('marketState', 'UNKNOWN').upper()
                final_error_result['currency'] = info.get('currency')
        except Exception:
            pass
        return final_error_result