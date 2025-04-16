# config.py
import os
# EventRegistry API anahtarını ortam değişkeninden veya doğrudan al
NEWSAPI_KEY = os.environ.get('EVENTREGISTRY_API_KEY', '7c13d593-f204-4d98-9ac4-75ee489a336d')

# Haber istekleri için API endpoint
NEWSAPI_ENDPOINT = 'https://eventregistry.org/api/v1/article/getArticles' # EventRegistry API endpoint

# Ek API anahtarları ve yapılandırmalar
API_TIMEOUT = 15  # API istekleri için zaman aşımı süresi (saniye) - Biraz artırıldı
YFINANCE_MIN_INTERVAL = 4 # yfinance istekleri arası minimum bekleme (saniye) - Artırıldı