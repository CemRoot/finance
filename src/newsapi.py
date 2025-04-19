# src/newsapi.py
import logging
import time
from datetime import datetime, timedelta, date, UTC
from random import uniform

import pandas as pd
import pytz

from config import NEWSAPI_KEY

from src.marketstack_api import MarketstackAPI
marketstack_api = MarketstackAPI()

# Try to import EventRegistry
event_registry_available = False
try:
    from eventregistry import EventRegistry, QueryArticlesIter
    event_registry_available = True
    logger = logging.getLogger(__name__)
    logger.info("EventRegistry successfully imported")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("EventRegistry not available. News functionality will be limited.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Globals for Rate Limiting ---
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

def rate_limit(api_type='newsapi'):
    """
    Applies rate limiting by pausing execution if calls are too frequent.

    Args:
        api_type (str): The type of API being called (only 'newsapi' now).
    """
    global LAST_NEWSAPI_REQUEST_TIME
    current_time = time.time()
    enforced = False
    last_request_time = 0
    min_interval = 0

    if api_type == 'newsapi':
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
    if api_type == 'newsapi':
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
            if dt_obj.tzinfo is None:
                return dt_obj.replace(tzinfo=UTC)
            return dt_obj.astimezone(UTC)
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
        key=lambda x: x['date'] if isinstance(x['date'], datetime) else datetime.min.replace(tzinfo=UTC),
        reverse=True
    )
    return processed_articles


def get_stock_data(stock: str) -> dict:
    """
    Fetches stock data using the Marketstack API.
    
    Args:
        stock (str): The stock symbol (e.g., 'AAPL').
        
    Returns:
        dict: A dictionary containing stock data, or a dictionary with an 'error' key.
    """
    if not stock:
        logger.error("Cannot fetch stock data: Stock symbol is empty")
        return {'error': "Stock symbol cannot be empty"}
    
    stock = stock.strip().upper()
    logger.info(f"Fetching stock data for {stock} using Marketstack API")
    
    try:
        # Get 1 year of daily data
        df = marketstack_api.get_stock_history(stock, period='1y')
        
        if df is None or df.empty:
            logger.error(f"No stock data found for {stock}")
            return {'error': f"No data available for {stock}"}
        
        # Calculate basic stats
        daily_returns = df['Close'].pct_change().dropna()
        
        # Prepare output data
        labels = df.index.strftime('%Y-%m-%d').tolist()
        values = df['Close'].tolist()
        
        # Prepare candlestick data for the chart
        candlestick_data = []
        for i, date_str in enumerate(labels):
            if i < len(values):
                candlestick_data.append({
                    't': date_str,  # timestamp in string format
                    'o': float(df['Open'].iloc[i]) if not pd.isna(df['Open'].iloc[i]) else None,
                    'h': float(df['High'].iloc[i]) if not pd.isna(df['High'].iloc[i]) else None,
                    'l': float(df['Low'].iloc[i]) if not pd.isna(df['Low'].iloc[i]) else None,
                    'c': float(df['Close'].iloc[i]) if not pd.isna(df['Close'].iloc[i]) else None
                })
        
        # Get current time in UTC for the timestamp
        current_time = datetime.now(UTC)
        
        # Determine market status based on current day and time
        # USA market hours: 9:30 AM - 4:00 PM Eastern Time, Monday to Friday
        current_time_et = current_time.astimezone(pytz.timezone('US/Eastern'))
        is_weekday = current_time_et.weekday() < 5  # 0-4 are Monday to Friday
        market_open_time = current_time_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = current_time_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_market_hours = market_open_time <= current_time_et <= market_close_time
        
        if is_weekday and is_market_hours:
            market_status = "OPEN"
        elif is_weekday and current_time_et < market_open_time:
            market_status = "PRE-MARKET"
        elif is_weekday and current_time_et > market_close_time:
            market_status = "AFTER-HOURS"
        else:
            market_status = "CLOSED"
        
        # Get the last available price as current price
        current_price = values[-1] if values else None
        
        result = {
            'symbol': stock,
            'labels': labels,
            'values': values,
            'open_values': df['Open'].tolist(),
            'high_values': df['High'].tolist(),
            'low_values': df['Low'].tolist(),
            'volume_values': df['Volume'].tolist(),
            'change_percent': round(((values[-1] / values[0]) - 1) * 100, 2) if len(values) > 1 else 0,
            'last_price': values[-1] if values else None,
            'last_date': labels[-1] if labels else None,
            'volatility': round(daily_returns.std() * (252 ** 0.5) * 100, 2),  # Annualized volatility
            'source': 'Marketstack',
            'currency': 'USD',  # Default currency
            'company_name': stock,  # Default to stock symbol if company name not available
            'candlestick_data': candlestick_data,  # Add candlestick data for the financial chart
            'current_price': current_price,  # Current price (latest available)
            'market_status': market_status,  # Market status
            'timestamp': current_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')  # Current timestamp in ISO format
        }
        
        logger.info(f"Successfully fetched {len(labels)} days of data for {stock}")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching stock data for {stock}: {e}", exc_info=True)
        return {'error': f"Failed to retrieve stock data: {str(e)}"}


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

    # Check if EventRegistry is available
    if not event_registry_available:
        logger.warning(f"EventRegistry not available. Cannot fetch news for {stock}.")
        return [
            {
                'title': 'News functionality unavailable',
                'description': 'EventRegistry library is not installed. Please install it to enable news functionality.',
                'url': '#',
                'date': datetime.now(UTC),
                'published_at': datetime.now(UTC),
                'source': 'System',
                'image_url': ''
            }
        ]

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
            # *** HABER LİMİTİ ARTIRILDI ***
            articles = list(q.execQuery(er, sortBy="date", sortByAsc=False, maxItems=30)) # <-- 30 haber
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
            # Fallback to general financial news
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
            
            logger.info("Attempting FALLBACK general financial news search")
            fallback_q = QueryArticlesIter.initWithComplexQuery(fallback_query)
            try:
                articles = list(fallback_q.execQuery(er, sortBy="date", sortByAsc=False, maxItems=15))
                logger.info("Fallback search found {} general financial news articles.".format(len(articles)))
            except Exception as fallback_err:
                logger.error("Error executing fallback query: {}".format(fallback_err), exc_info=True)
                return {'error': 'Error in fallback news query: {}'.format(str(fallback_err))}
                
        except Exception as gen_err:
            logger.error("General error in fallback search: {}".format(gen_err), exc_info=True)
            return {'error': 'Unexpected error in fallback news search: {}'.format(str(gen_err))}

    # --- Step 3: Process and Return Articles (either primary or fallback) ---
    return _process_articles(articles, stock, is_fallback)