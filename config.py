# config.py
import os
import logging

# --- API Keys & Endpoints ---

# Marketstack API Key (Used for Stock Data)
# Get from environment variable 'MARKETSTACK_API_KEY'
MARKETSTACK_API_KEY = os.environ.get('MARKETSTACK_API_KEY')

# EventRegistry API Key (Used for News Fetching)
# Get from environment variable 'EVENTREGISTRY_API_KEY'
NEWSAPI_KEY = os.environ.get('EVENTREGISTRY_API_KEY')

# API Endpoints
NEWSAPI_ENDPOINT = 'https://eventregistry.org/api/v1/article/getArticles'
MARKETSTACK_BASE_URL = 'https://api.marketstack.com/v2'

# --- Timeouts and Intervals ---

# API Request Timeout (seconds)
API_TIMEOUT = 30  # Increased for slower API responses

# Marketstack Request Rate Limits
MARKETSTACK_REQUESTS_PER_SECOND = 5  # Free plan allows 5 requests per second

# --- Cache Configuration ---
CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
STOCK_DATA_CACHE_TIMEOUT = 900  # 15 minutes
NEWS_CACHE_TIMEOUT = 1800  # 30 minutes
FORECAST_CACHE_TIMEOUT = 3600  # 1 hour

# --- Error Messages ---
ERROR_MESSAGES = {
    'missing_api_key': 'API key is missing. Please set the {} environment variable.',
    'rate_limit': 'Rate limit exceeded. Please try again in {} seconds.',
    'api_error': 'API error occurred: {}',
    'no_data': 'No data available for symbol: {}'
}

# --- Input Validation ---
def validate_api_keys():
    logger = logging.getLogger(__name__)
    
    if not MARKETSTACK_API_KEY:
        logger.error(ERROR_MESSAGES['missing_api_key'].format('MARKETSTACK_API_KEY'))
        raise ValueError(ERROR_MESSAGES['missing_api_key'].format('MARKETSTACK_API_KEY'))
    
    if not NEWSAPI_KEY:
        logger.error(ERROR_MESSAGES['missing_api_key'].format('EVENTREGISTRY_API_KEY'))
        raise ValueError(ERROR_MESSAGES['missing_api_key'].format('EVENTREGISTRY_API_KEY'))

# Validate API keys on import
validate_api_keys()

# --- Marketstack API Configuration ---
MARKETSTACK_CONFIG = {
    'eod_endpoint': '/eod',
    'intraday_endpoint': '/intraday/latest',
    'max_symbols_per_request': 100,
    'max_limit_per_request': 1000,
    'supported_intervals': ['1d'],  # EOD API only supports daily data
    'default_sort': 'DESC'  # Newest first
}

# --- Model Configuration (Optional) ---
# You could place model-related configurations here if needed,
# although they are currently handled within their respective modules.
# Example: FINBERT_MODEL_NAME = "ProsusAI/finbert"