# app.py
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import src.newsapi as newsapi
import src.decision as decision
import json
import src.forecasting as forecasting
import src.sentiment # sentiment modülünü import et (karar için gerekli)
import pandas as pd # pd import edildi
import numpy as np
from datetime import datetime
import time
from functools import wraps
import logging # Logging ekle
import os # os import edildi
import pytz # pytz import edildi
import base64 # base64 import edildi (JSON encode/decode için)

# Flask Uygulaması
app = Flask(__name__)

# Gizli Anahtar
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-should-be-changed') # Geliştirme anahtarı

# Logging Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Flask'ın kendi logger'ını da yapılandıralım
app.logger.handlers.clear() # Varsayılan handler'ları temizle
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO) # Flask logger seviyesini ayarla

# --- Jinja2 Filtreleri ---
@app.template_filter('calculate_average')
def calculate_average(values):
    if not values: return 'N/A'
    try:
        valid_values = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
        if not valid_values: return 'N/A'
        return '{:,.0f}'.format(sum(valid_values) / len(valid_values))
    except (ValueError, TypeError): return 'N/A'

@app.template_filter('safe_sum')
def safe_sum(values):
     if not values: return 0
     try:
         valid_values = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
         return sum(valid_values)
     except (ValueError, TypeError): return 0

@app.template_filter('variance')
def calculate_variance(values):
    if not values: return None
    try:
        valid_values = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
        if len(valid_values) < 2: return None
        return np.var(valid_values)
    except (ValueError, TypeError): return None

@app.template_filter('format_number')
def format_number(value, decimals=2):
    if value is None or not isinstance(value, (int, float)): return 'N/A'
    try:
        format_string = '{:,.%df}' % decimals
        return format_string.format(value)
    except (ValueError, TypeError): return str(value)

@app.template_filter('format_date')
def format_date(value, format_str='%d.%m.%Y'):
    if not value: return ''
    try:
        dt_obj = None
        if isinstance(value, datetime):
            dt_obj = value
        elif isinstance(value, str):
            try: # ISO formatını veya sadece tarihi dene
                if 'T' in value or ' ' in value: # Saat bilgisi varsa
                    dt_obj = pd.to_datetime(value).to_pydatetime()
                else: # Sadece tarihse
                     dt_obj = datetime.strptime(value, '%Y-%m-%d')
            except ValueError:
                 app.logger.warning(f"Tarih formatlama: Anlaşılamayan string formatı: {value}")
                 return value # Anlaşılamıyorsa orijinal string kalsın
        else:
             app.logger.warning(f"Tarih formatlama: Desteklenmeyen tip: {type(value)} - Değer: {value}")
             return str(value)

        # Eğer timezone bilgisi varsa, yerel saate çevirip formatlayalım (veya UTC?)
        # Şimdilik UTC varsayıp timezone'u kaldıralım
        if dt_obj and dt_obj.tzinfo:
             dt_obj = dt_obj.astimezone(pytz.utc).replace(tzinfo=None)

        return dt_obj.strftime(format_str) if dt_obj else str(value)

    except Exception as e:
        app.logger.error(f"Tarih formatlama hatası: {e} - Değer: {value}", exc_info=True)
        return str(value)

# Jinja environment'a filtreleri ve global fonksiyonları ekle
app.jinja_env.filters['calculate_average'] = calculate_average
app.jinja_env.filters['safe_sum'] = safe_sum
app.jinja_env.filters['variance'] = calculate_variance
app.jinja_env.filters['format_number'] = format_number
app.jinja_env.filters['format_date'] = format_date
# Base64 encode filtresi (JSON'u JS'e güvenli aktarmak için - artık kullanılmıyor)
# app.jinja_env.filters['b64encode'] = lambda s: base64.b64encode(s.encode()).decode()
app.jinja_env.globals.update(abs=abs)
app.jinja_env.globals.update(len=len)
app.jinja_env.globals.update(now=datetime.utcnow) # Template'de güncel UTC zamanı için


# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    stock_symbol = None
    stock_data = None
    articles = []
    decision_result = "Veri Yok"
    stock_data_json = '{}' # JS için başlangıç değeri
    error_occurred = False
    error_message = None
    # Volatilite ve teknik gösterge sonuçları için değişkenler
    volatility_metric = None
    sma_status = None
    rsi_status = None


    if request.method == 'POST':
        stock_symbol = request.form.get('stock', '').strip().upper()
        if not stock_symbol:
            flash('Lütfen geçerli bir hisse senedi sembolü girin.', 'warning')
            return redirect(url_for('index'))
        else:
            app.logger.info(f"POST request for symbol: {stock_symbol}. Redirecting to GET.")
            return redirect(url_for('index', stock=stock_symbol))

    # GET isteği veya yönlendirme sonrası
    stock_symbol = request.args.get('stock', '').strip().upper()

    if not stock_symbol:
        return render_template('index.html', stock=None) # Hoşgeldin ekranı

    app.logger.info(f"Processing GET request for symbol: {stock_symbol}")
    try:
        # Hisse senedi verilerini çek
        stock_data = newsapi.get_stock_data(stock_symbol)

        if stock_data.get('error'):
            error_occurred = True
            error_message = stock_data['error']
            app.logger.error(f"Error fetching stock data for {stock_symbol}: {error_message}")
            flash(error_message, 'danger')
            stock_data = None # Hata varsa template için None yap
        elif not stock_data or not stock_data.get('labels'):
             error_occurred = True
             error_message = f"{stock_symbol} için hisse senedi verisi alınamadı (boş veya eksik veri)."
             app.logger.error(error_message)
             flash(error_message, 'danger')
             stock_data = None
        else:
             # Veri başarıyla alındı
             stock_data_json = json.dumps(stock_data)

             # Teknik Göstergeleri ve Volatiliteyi burada hesaplayalım (opsiyonel)
             # Bu, Jinja içindeki karmaşık hesaplamaları önler.
             try:
                 values = [v for v in stock_data.get('values', []) if v is not None] # None olmayanları al
                 if len(values) >= 2:
                    returns = pd.Series(values).pct_change().dropna()
                    if len(returns) >= 2: # Varyans için en az 2 return lazım
                        # Yıllık volatilite
                        volatility_metric = returns.std() * np.sqrt(252) # 252 işlem günü

                 sma_period = 20
                 if len(values) >= sma_period:
                     sma_20 = np.mean(values[-sma_period:])
                     current_price = values[-1]
                     if current_price > sma_20: sma_status = 'Üzerinde'
                     elif current_price < sma_20: sma_status = 'Altında'
                     else: sma_status = 'Eşit'

                 rsi_period = 14
                 if len(values) >= rsi_period + 1:
                     delta = pd.Series(values).diff()
                     gain = delta.where(delta > 0, 0.0)
                     loss = -delta.where(delta < 0, 0.0)
                     # EWMA (Exponential Weighted Moving Average) kullanmak daha standarttır
                     avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
                     avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
                     rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else np.inf
                     rsi = 100 - (100 / (1 + rs))
                     if rsi > 70: rsi_status = 'Aşırı Alım'
                     elif rsi < 30: rsi_status = 'Aşırı Satım'
                     else: rsi_status = 'Nötr'

             except Exception as calc_err:
                 app.logger.warning(f"Error calculating metrics for {stock_symbol}: {calc_err}")


        # Haberleri çek ve duygu analizi yap (Hata olsa bile devam et)
        app.logger.info(f"Fetching news for {stock_symbol}")
        articles = newsapi.get_news(stock_symbol)
        if not articles:
            app.logger.warning(f"No news found for {stock_symbol}")
            # flash(f"{stock_symbol} için güncel haber bulunamadı.", 'info') # Çok sık çıkabilir, kapatalım

        # Karar mekanizmasını çalıştır
        if articles:
             app.logger.info(f"Getting decision based on {len(articles)} articles for {stock_symbol}")
             decision_result = decision.get_decision(articles)
        else:
             decision_result = "Veri Yok"


        # Template'e verileri gönder
        return render_template('index.html',
                               stock=stock_symbol,
                               stock_data=stock_data,
                               articles=articles,
                               decision=decision_result,
                               stock_data_json=stock_data_json,
                               error=error_occurred,
                               error_message=error_message,
                               # Hesaplanan metrikleri de gönder
                               volatility_metric=volatility_metric,
                               sma_status=sma_status,
                               rsi_status=rsi_status
                               )

    except Exception as e:
        app.logger.error(f"Genel hata (route '/') işlenirken {stock_symbol}: {str(e)}", exc_info=True)
        flash(f'Beklenmedik bir sunucu hatası oluştu: {str(e)}', 'danger')
        return render_template('index.html', stock=stock_symbol, error=True, error_message=f'Beklenmedik bir sunucu hatası: {str(e)}')


@app.route('/refresh_stock')
def refresh_stock():
    """
    AJAX ile çağrılacak endpoint. Sadece hisse verilerini yeniler.
    """
    stock = request.args.get("stock")
    if not stock:
        return jsonify({"status": "error", "message": "Stock sembolü gerekli"}), 400

    app.logger.info(f"AJAX request to refresh stock data for: {stock}")
    try:
        stock_data = newsapi.get_stock_data(stock)

        if stock_data.get('error'):
            app.logger.error(f"Error refreshing stock data for {stock}: {stock_data['error']}")
            # Hata kodunu belirlemeye çalışalım
            status_code = 500
            if "bulunamadı" in stock_data['error'].lower() or "geçersiz" in stock_data['error'].lower():
                 status_code = 404
            elif "rate limit" in stock_data['error'].lower():
                 status_code = 429
            return jsonify({
                "status": "error",
                "message": stock_data['error']
            }), status_code
        elif not stock_data or not stock_data.get('labels'):
             app.logger.warning(f"No stock data returned after refresh for {stock}")
             return jsonify({
                 "status": "error",
                 "message": f"{stock} için veri yenilenemedi (boş veri)."
             }), 404
        else:
             app.logger.info(f"Successfully refreshed stock data for {stock}")
             # Başarı durumunda sadece gerekli verileri döndür (JS'in ihtiyaç duyduğu)
             response_data = {
                 'current_price': stock_data.get('current_price'),
                 'change_percent': stock_data.get('change_percent'),
                 'market_status': stock_data.get('market_status'),
                 'labels': stock_data.get('labels'),
                 'values': stock_data.get('values'),
                 'volume_values': stock_data.get('volume_values'),
                 'candlestick_data': stock_data.get('candlestick_data'),
                 'timestamp': stock_data.get('timestamp'),
                 'company_name': stock_data.get('company_name'), # JS tarafında gerekebilir
                 # Diğerleri (open, high, low) gerekirse eklenebilir
             }
             return jsonify({
                 "status": "success",
                 "stock_data": response_data
             })

    except Exception as e:
        app.logger.error(f"Hisse yenileme sırasında sunucu hatası ({stock}): {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Veri yenilenirken sunucu hatası oluştu."
        }), 500


@app.route('/get_forecast_data', methods=['GET'])
def get_forecast_data():
    """
    AJAX ile çağrılacak endpoint. Prophet tahmin verilerini döndürür.
    """
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Hisse senedi sembolü gerekli.'}), 400

    app.logger.info(f"AJAX request for forecast data: {symbol}")

    try:
        # forecasting modülündeki fonksiyonu çağır
        forecast_result = forecasting.get_prophet_forecast(symbol, periods=30) # 30 günlük tahmin

        if forecast_result is None:
             app.logger.error(f"Forecast function returned None for {symbol}")
             return jsonify({'error': f'{symbol} için tahmin hesaplanırken bilinmeyen bir hata oluştu.'}), 500
        elif forecast_result.get('error'):
            error_msg = forecast_result['error']
            app.logger.error(f"Error generating forecast for {symbol}: {error_msg}")
            # Hata mesajına göre uygun HTTP kodu döndür
            status_code = 500 # Varsayılan sunucu hatası
            if "bulunamadı" in error_msg.lower() or "yeterli veri yok" in error_msg.lower():
                status_code = 404
            elif "rate limit" in error_msg.lower():
                 status_code = 429
            return jsonify({'error': error_msg}), status_code
        else:
            app.logger.info(f"Successfully generated forecast for {symbol}")
            return jsonify(forecast_result) # Başarılı sonucu JSON olarak döndür

    except Exception as e:
        app.logger.error(f"Tahmin endpoint (/get_forecast_data) hatası ({symbol}): {str(e)}", exc_info=True)
        return jsonify({'error': f'Tahmin hesaplanırken genel sunucu hatası oluştu.'}), 500


# Ana Çalıştırma Bloğu
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # debug=True geliştirme için, production'da False yapın.
    # use_reloader=False rate limiting hatalarını ve çift başlatmayı önler.
    # host='0.0.0.0' ağdan erişim için.
    app.logger.info(f"Starting Flask app on host 0.0.0.0 port {port} with debug=True, use_reloader=False")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)