# src/newsapi.py
import requests
import yfinance as yf
from config import NEWSAPI_KEY, NEWSAPI_ENDPOINT, API_TIMEOUT, YFINANCE_MIN_INTERVAL
import datetime
import pytz
import time
from random import uniform
import logging
import pandas as pd # pd import edildi

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting için global değişkenler
LAST_YFINANCE_REQUEST_TIME = 0
LAST_NEWSAPI_REQUEST_TIME = 0

def _rate_limit(api_type='yfinance'):
    """Rate limiting için yardımcı fonksiyon"""
    global LAST_YFINANCE_REQUEST_TIME, LAST_NEWSAPI_REQUEST_TIME

    current_time = time.time()
    enforced = False # Limit uygulandı mı?

    if api_type == 'yfinance':
        min_interval = YFINANCE_MIN_INTERVAL
        last_request_time = LAST_YFINANCE_REQUEST_TIME
    elif api_type == 'newsapi':
        # EventRegistry için de bir limit belirleyelim (örneğin 1.5 saniye)
        min_interval = 1.5
        last_request_time = LAST_NEWSAPI_REQUEST_TIME
    else:
        return # Bilinmeyen API tipi

    time_since_last_request = current_time - last_request_time
    if time_since_last_request < min_interval:
        sleep_time = min_interval - time_since_last_request + uniform(0.1, 0.6) # Biraz daha rastgelelik
        logger.warning(f"Rate limiting for {api_type}: sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
        enforced = True

    # Zaman damgasını güncelle (yeni zamanla)
    new_time = time.time()
    if api_type == 'yfinance':
        LAST_YFINANCE_REQUEST_TIME = new_time
    elif api_type == 'newsapi':
        LAST_NEWSAPI_REQUEST_TIME = new_time

    if enforced:
        logger.info(f"Rate limit enforced for {api_type}. Resuming...")


def get_news(stock):
    """
    Belirtilen stok veya şirket adı için EventRegistry API'sini kullanarak
    haberleri getirir.
    """
    if not stock:
        logger.error("Haber çekilemedi: Stok sembolü boş")
        return []

    try:
        _rate_limit('newsapi') # NewsAPI için rate limit
        params = {
            'apiKey': NEWSAPI_KEY,
            'resultType': 'articles',
            'articlesPage': 1,
            'articlesCount': 7, # Sayıyı biraz daha azaltalım (7)
            'articlesSortBy': 'date',
            'articlesSortByAsc': False,
            'articleBodyLen': 250, # Kısa özet yeterli
            'dataType': 'news',
            'forceMaxDataTimeWindow': 7, # Son 7 gün
            'lang': 'tur', # Türkçe haberleri önceliklendir (destekliyorsa)
            'keyword': stock
            # 'categoryUri': 'dmoz/Business/Financial_Services' # Kategori ekleyebiliriz
        }

        logger.info(f"EventRegistry API isteği gönderiliyor: {NEWSAPI_ENDPOINT} for {stock}")
        response = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=API_TIMEOUT)
        logger.info(f"EventRegistry API yanıt kodu ({stock}): {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', {}).get('results', [])
            logger.info(f"Bulunan haber sayısı ({stock}): {len(articles)}")

            result = []
            for article in articles:
                if not isinstance(article, dict):
                    continue

                title = article.get('title', 'Başlık bulunamadı')
                description = article.get('body', 'Açıklama yok')
                url = article.get('url', '#')
                # dateTimePub veya date kullan, UTC ise timezone bilgisi ekle
                date_str = article.get('dateTimePub') or article.get('date')
                source = article.get('source', {}).get('title', 'Bilinmeyen Kaynak')
                image_url = article.get('image') # None olabilir

                parsed_date = None
                if date_str:
                    try:
                        # EventRegistry genellikle 'YYYY-MM-DD' veya ISO formatında verir
                        if 'T' in date_str: # ISO formatı gibi
                            parsed_date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00')).astimezone(pytz.utc)
                        else: # Sadece tarih ise
                            parsed_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                            # Bunu da UTC yapalım (veya yerel saat dilimi?) - Şimdilik UTC varsayalım
                            parsed_date = pytz.utc.localize(parsed_date)
                    except ValueError:
                        logger.warning(f"Haber tarihi ({stock}) ayrıştırılamadı: {date_str}")
                        parsed_date = date_str # Orijinal string kalsın

                result.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'date': parsed_date, # datetime objesi (UTC) veya string
                    'source': source,
                    'image_url': image_url if image_url else ''
                })

            logger.info(f"İşlenen haber sayısı ({stock}): {len(result)}")
            return result
        elif response.status_code == 429:
             logger.error(f"EventRegistry API rate limit aşıldı (HTTP 429).")
             return []
        else:
            logger.error(f"EventRegistry API hatası ({stock}): HTTP {response.status_code}")
            try:
                error_detail = response.json()
                logger.error(f"Hata detayı ({stock}): {error_detail}")
            except Exception as ex:
                logger.error(f"Hata detayı ({stock}) alınamadı: {ex}")
            return []
    except requests.exceptions.Timeout:
        logger.error(f"Haber çekilirken zaman aşımı ({stock}) : {NEWSAPI_ENDPOINT}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Haber çekilirken ağ hatası ({stock}): {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Haber çekilirken genel hata ({stock}): {str(e)}", exc_info=True)
        return []


def get_stock_data(stock):
    """
    yfinance kütüphanesini kullanarak, girilen stok sembolüne ait verileri çeker.
    Rate limiting ve hata yönetimi iyileştirildi.
    """
    empty_result = {
        'labels': [], 'values': [], 'open_values': [], 'high_values': [], 'low_values': [],
        'volume_values': [], 'candlestick_data': [], 'market_status': 'Bilinmiyor',
        'current_price': None, 'change_percent': None, 'company_name': stock, # None yapıldı
        'timestamp': datetime.datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S %Z'), # UTC eklendi
        'error': None # Hata mesajı ekleyelim
    }
    try:
        logger.info(f"Getting stock data for {stock}")
        # Ticker nesnesi oluşturmadan önce rate limit UYGULAMAYALIM, bu henüz API çağrısı değil
        # _rate_limit('yfinance')
        ticker = yf.Ticker(stock)

        # Hisse senedi hakkında detaylı bilgi alma - .info çağrısından ÖNCE rate limit
        info = {}
        try:
            logger.info(f"Applying rate limit before fetching info for {stock}")
            _rate_limit('yfinance') # Rate limiting BURADA
            logger.info(f"Fetching info for {stock}")
            info = ticker.info
            # yfinance bazen boş dict dönebilir veya anahtar eksik olabilir, kontrol edelim
            if not info or not info.get('symbol'):
                logger.error(f"Hisse bilgisi alınamadı (muhtemelen geçersiz sembol): {stock}. Info: {info}")
                empty_result['error'] = f"'{stock}' sembolü bulunamadı veya geçersiz."
                return empty_result
            logger.info(f"Successfully fetched info for {stock}")
        except requests.exceptions.HTTPError as http_err:
            # Hata kodlarını işle
            status_code = http_err.response.status_code
            logger.error(f"Hisse bilgisi alınamadı (HTTP Error {status_code}): {stock} - {str(http_err)}")
            if status_code == 404:
                empty_result['error'] = f"'{stock}' sembolü bulunamadı veya geçersiz."
            elif status_code == 429:
                empty_result['error'] = f"'{stock}' için veri alınırken rate limit aşıldı. Lütfen biraz bekleyip tekrar deneyin."
            else:
                 empty_result['error'] = f"'{stock}' bilgisi alınırken HTTP hatası: {status_code}"
            return empty_result
        except Exception as e:
            # Diğer hatalar (örn. JSON decode hatası 429 sonrası)
            logger.error(f"Hisse bilgisi alınamadı ({stock}): {type(e).__name__} - {str(e)}")
            # Rate limit hatası olabilir, history'yi denemeye devam edelim
            # empty_result['error'] = f"'{stock}' bilgisi alınırken beklenmedik hata: {type(e).__name__}"
            # return empty_result # Şimdilik return etmeyelim

        # Piyasa açık/kapalı durumu (Lokal, API yok)
        market_status = "Bilinmiyor" # Varsayılan
        try:
            eastern = pytz.timezone('US/Eastern')
            now_eastern = datetime.datetime.now(eastern)
            # info'dan gelen marketState daha güvenilir
            market_state = info.get('marketState', '').upper()
            logger.info(f"Market state from info for {stock}: {market_state}")

            if market_state in ['REGULAR']:
                 market_status = "Açık"
            elif market_state in ['PRE', 'PREPRE']: # Piyasa öncesi
                 market_status = "Açık (Piyasa Öncesi)"
            elif market_state in ['POST', 'POSTPOST', 'EXTENDED']: # Piyasa sonrası
                 market_status = "Kapalı (Piyasa Sonrası)"
            elif market_state in ['CLOSED']:
                 market_status = "Kapalı"
            else: # Eski yönteme fallback (eğer marketState yoksa veya tanımsızsa)
                 logger.warning(f"Unknown or missing marketState for {stock}: '{market_state}'. Falling back to time check.")
                 is_weekday = now_eastern.weekday() < 5 # Pazartesi-Cuma
                 # Saatleri 9:30 ve 16:00 olarak alalım
                 is_market_hours = (now_eastern.time() >= datetime.time(9, 30)) and (now_eastern.time() < datetime.time(16, 0))
                 if is_weekday and is_market_hours:
                     market_status = "Açık"
                 else:
                     market_status = "Kapalı"
        except Exception as e:
             logger.warning(f"Market status determination failed for {stock}: {e}. Defaulting to 'Bilinmiyor'.")


        # Tarihsel veri alma - Ana periyot ve yedekler
        hist = None
        primary_period = "6mo" # Daha uzun veri, daha iyi tahmin/analiz
        fallback_periods = ["3mo", "1mo", "2wk", "5d"] # Daha kısa yedekler

        all_periods_to_try = [primary_period] + fallback_periods

        for period in all_periods_to_try:
            logger.info(f"Attempting to fetch history for {stock} with period={period}")
            try:
                logger.info(f"Applying rate limit before fetching history ({period}) for {stock}")
                _rate_limit('yfinance') # Rate limiting BURADA
                logger.info(f"Fetching history ({period}) for {stock}")
                # interval'i dinamik seçelim: kısa periyotlar için daha sık (örn 1d, 1h), uzunlar için 1d
                interval = "1d"
                if period in ["5d", "1wk", "2wk"]:
                    interval = "1h" # Daha kısa periyotlar için saatlik veri deneyelim? Veya 30m?
                elif period == "1mo":
                     interval = "90m" # veya 60m

                hist = ticker.history(period=period, interval=interval, auto_adjust=True) # auto_adjust=True önemli

                if hist is not None and not hist.empty:
                    logger.info(f"Successfully fetched history for {stock} with period={period}, interval={interval}. Data points: {len(hist)}")
                    break # Veri bulunduysa döngüden çık
                else:
                    logger.warning(f"No history data found for {stock} with period={period}, interval={interval}.")
                    hist = None # Boş DataFrame geldiyse None yapalım
            except requests.exceptions.HTTPError as http_err:
                status_code = http_err.response.status_code
                logger.error(f"{period} history fetch failed for {stock} (HTTP Error {status_code}): {str(http_err)}")
                if status_code == 429: # Rate limit ise bekleyip sonraki periyodu deneyelim
                    logger.warning("Rate limit hit during history fetch. Waiting before next attempt...")
                    time.sleep(YFINANCE_MIN_INTERVAL * 1.5) # Ekstra bekleme
                hist = None
            except ValueError as ve: # yfinance bazen interval/period uyumsuzluğunda ValueError verebilir
                logger.error(f"{period} history fetch failed for {stock} (ValueError): {str(ve)}. Trying next period.")
                hist = None
            except Exception as e:
                logger.error(f"{period} history fetch failed for {stock}: {type(e).__name__} - {str(e)}")
                hist = None
            # Her deneme sonrası kısa bir bekleme ekleyelim (rate limiti zorlamamak için)
            if hist is None:
                 time.sleep(uniform(0.5, 1.5))

        # Son kontrol
        if hist is None or hist.empty:
            logger.error(f"Could not fetch any valid historical data for {stock} after trying multiple periods.")
            empty_result['error'] = f"'{stock}' için geçmiş fiyat verisi alınamadı. Sembolü kontrol edin veya daha sonra tekrar deneyin."
            # info'dan alınan ismi yine de döndürelim
            empty_result['company_name'] = info.get('shortName') or info.get('longName', stock) if info else stock
            empty_result['market_status'] = market_status # Piyasa durumunu döndürelim
            return empty_result

        # --- Veri İşleme Başlangıcı ---
        logger.info(f"Processing {len(hist)} data points for {stock}")

        # Gelen index'in saat dilimini kontrol et (genellikle yerel veya UTC olur)
        # Eğer timezone varsa UTC'ye çevirip kaldıralım, yoksa direkt formatlayalım
        try:
            if isinstance(hist.index, pd.DatetimeIndex):
                 if hist.index.tz:
                     # logger.debug(f"Original timezone for {stock}: {hist.index.tz}")
                     labels = hist.index.tz_convert('UTC').tz_localize(None).strftime('%Y-%m-%d %H:%M:%S').tolist() # Saat bilgisiyle alalım
                 else:
                     # logger.debug(f"No timezone found for {stock}. Assuming local/UTC.")
                     labels = hist.index.strftime('%Y-%m-%d %H:%M:%S').tolist() # Saat bilgisiyle alalım
            else:
                 # Index datetime değilse, dönüştürmeyi dene
                 logger.warning(f"Index for {stock} is not DatetimeIndex. Type: {type(hist.index)}. Attempting conversion.")
                 hist.index = pd.to_datetime(hist.index)
                 labels = hist.index.strftime('%Y-%m-%d %H:%M:%S').tolist()

        except Exception as date_fmt_err:
             logger.error(f"Error formatting dates for {stock}: {date_fmt_err}", exc_info=True)
             empty_result['error'] = f"'{stock}' için tarih verileri işlenemedi."
             return empty_result


        # NaN veya Inf değerlerini kontrol et ve temizle/doldur
        hist = hist.replace([float('inf'), -float('inf')], float('nan'))
        # Doldurmadan önce sütunların varlığından emin ol
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in hist.columns]
        if missing_cols:
             logger.error(f"Missing required columns in historical data for {stock}: {missing_cols}. Available: {list(hist.columns)}")
             empty_result['error'] = f"'{stock}' için gerekli fiyat sütunları ({', '.join(missing_cols)}) eksik."
             return empty_result

        # Eksik verileri doldur (özellikle kısa interval'lerde olabilir)
        # hist = hist.ffill().bfill() # Bu bazen yanıltıcı olabilir, NaN bırakmak daha iyi? Ya da interpolasyon?
        # Şimdilik NaN bırakalım, JS tarafı handle etsin veya sadece NaN olmayanları alalım.
        # Sadece Volume'u dolduralım
        hist['Volume'] = hist['Volume'].fillna(0)


        # Değerleri float/int'e çevir ve yuvarla (NaN kontrolü ile)
        try:
            close_values = [float(round(v, 2)) if pd.notna(v) else None for v in hist['Close']]
            open_values = [float(round(v, 2)) if pd.notna(v) else None for v in hist['Open']]
            high_values = [float(round(v, 2)) if pd.notna(v) else None for v in hist['High']]
            low_values = [float(round(v, 2)) if pd.notna(v) else None for v in hist['Low']]
            volume_values = [int(v) if pd.notna(v) else 0 for v in hist['Volume']] # Hacim NaN ise 0 yap
        except (ValueError, TypeError) as conv_err:
            logger.error(f"Error converting historical data types for {stock}: {conv_err}", exc_info=True)
            empty_result['error'] = f"'{stock}' fiyat verileri sayısal formata dönüştürülürken hata oluştu."
            return empty_result

        # Güncel fiyat ve değişim bilgisi (NaN kontrolü)
        current_price = None
        change_percent = None
        # Listenin sonundaki None olmayan ilk değeri bul
        last_valid_close_index = next((i for i, v in enumerate(reversed(close_values)) if v is not None), None)
        if last_valid_close_index is not None:
            current_price = close_values[-(last_valid_close_index + 1)]

            # Önceki kapanışı bulmak için daha geriye git
            prev_valid_close_index = next((i for i, v in enumerate(reversed(close_values), last_valid_close_index + 1) if v is not None), None)
            if prev_valid_close_index is not None:
                prev_close = close_values[-(prev_valid_close_index + 1)]
                if prev_close is not None and prev_close != 0 and current_price is not None:
                    change_percent = ((current_price - prev_close) / prev_close * 100)
            elif info and info.get('previousClose'): # Önceki gün yoksa info'dan almayı dene
                 prev_close_info = info.get('previousClose')
                 if prev_close_info and prev_close_info != 0 and current_price is not None:
                      change_percent = ((current_price - prev_close_info) / prev_close_info * 100)

        # Mum grafiği verisi (NaN kontrolü ile)
        candlestick_data = []
        for i in range(len(labels)):
            # İlgili indexteki tüm değerlerin geçerli (None olmayan) olduğundan emin ol
            if i < len(open_values) and i < len(high_values) and i < len(low_values) and i < len(close_values) and \
               open_values[i] is not None and high_values[i] is not None and \
               low_values[i] is not None and close_values[i] is not None:
                candlestick_data.append({
                    't': labels[i], # Tarih/saat string formatında
                    'o': open_values[i],
                    'h': high_values[i],
                    'l': low_values[i],
                    'c': close_values[i]
                })
            # else: logger.debug(f"Skipping candlestick data point at index {i} due to None values for {stock}")


        # Şirket ismi alma (info'dan)
        company_name = info.get('shortName') or info.get('longName', stock) if info else stock

        logger.info(f"Successfully processed stock data for {stock}. Candlestick points: {len(candlestick_data)}")

        return {
            'labels': labels,
            'values': close_values,
            'open_values': open_values,
            'high_values': high_values,
            'low_values': low_values,
            'volume_values': volume_values,
            'candlestick_data': candlestick_data,
            'market_status': market_status,
            'current_price': current_price, # None olabilir
            'change_percent': float(round(change_percent, 2)) if change_percent is not None else None,
            'company_name': company_name,
            'timestamp': datetime.datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'error': None # Başarılı ise hata yok
        }
    except Exception as e:
        logger.error(f"Hisse verisi çekilirken genel hata oluştu ({stock}): {type(e).__name__} - {str(e)}", exc_info=True)
        empty_result['error'] = f"'{stock}' verileri işlenirken beklenmedik bir sunucu hatası oluştu."
        return empty_result