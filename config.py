# config.py
import os

# --- API Keys & Endpoints ---

# EventRegistry API Key (Used for News Fetching)
# Get from environment variable 'EVENTREGISTRY_API_KEY' or use the provided default.
# Note: It's best practice to use environment variables for sensitive keys.
NEWSAPI_KEY = os.environ.get('EVENTREGISTRY_API_KEY', '7c13d593-f204-4d98-9ac4-75ee489a336d') # Replace default if needed Do NOT THING THIS IS REAL API KEY IM NOT STUPID :)

# EventRegistry API Endpoint for fetching news articles
NEWSAPI_ENDPOINT = 'https://eventregistry.org/api/v1/article/getArticles'

# --- Timeouts and Intervals ---

# API Request Timeout (seconds)
# How long to wait for a response from external APIs (NewsAPI, potentially others)
API_TIMEOUT = 20  # Increased slightly for potentially slower APIs

# yfinance Request Rate Limit (seconds)
# Minimum time interval between consecutive calls to yfinance to avoid being rate-limited.
# Adjust based on observed behavior; too low might cause 429 errors.
YFINANCE_MIN_INTERVAL = 2 # Reduced slightly, adjust if rate limited

# --- Model Configuration (Optional) ---
# You could place model-related configurations here if needed,
# although they are currently handled within their respective modules.
# Example: FINBERT_MODEL_NAME = "ProsusAI/finbert"

# --- Input Validation / Sanity Check ---
# Optional: Check if the essential API key is present and log a warning if not.
if NEWSAPI_KEY == 'YOUR_DEFAULT_EVENTREGISTRY_API_KEY_HERE' or not NEWSAPI_KEY:
    import logging
    logger = logging.getLogger(__name__) # Use logger if available
    # Check if logger has handlers to avoid errors during startup if logging isn't configured yet
    if logger.hasHandlers():
         logger.warning("NEWSAPI_KEY is using the default value or is not set. "
                        "Please set the EVENTREGISTRY_API_KEY environment variable "
                        "or update the default value in config.py for news fetching to work.")
    else:
        # Fallback to print if logging isn't ready
        print("WARNING: NEWSAPI_KEY is using the default value or is not set. "
              "Please set the EVENTREGISTRY_API_KEY environment variable "
              "or update the default value in config.py for news fetching to work.")