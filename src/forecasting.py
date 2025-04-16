# src/forecasting.py
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import time
# newsapi'den rate limit fonksiyonunu ve interval'i al (relative import)
from .newsapi import _rate_limit, YFINANCE_MIN_INTERVAL
import logging
import requests # Hata yakalamak için import et

# Gerekirse Prophet'in INFO loglarını bastır
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_prophet_forecast(stock_symbol, periods=30):
    """
    Prophet kütüphanesini kullanarak hisse senedi fiyat tahmini yapar.
    Rate limiting ve hata yönetimi eklendi.

    Args:
        stock_symbol (str): Hisse senedi sembolü (örn. 'AAPL')
        periods (int): Kaç gün ileriye tahmin yapılacağı

    Returns:
        dict: Tahmin sonuçlarını içeren sözlük veya hata durumunda None
    """
    try:
        logger.info(f"Starting Prophet forecast for {stock_symbol}")
        # Veri çekme - son 1 yıllık veri (Rate Limit ile)
        ticker = yf.Ticker(stock_symbol)
        logger.info(f"Applying rate limit before fetching history for Prophet ({stock_symbol})")
        _rate_limit('yfinance') # Rate limit
        logger.info(f"Fetching 1y history for Prophet ({stock_symbol})")
        # Prophet için genellikle günlük veri daha iyidir
        history = ticker.history(period="1y", interval="1d", auto_adjust=True)

        if history.empty:
            logger.error(f"Prophet forecast failed: No historical data found for {stock_symbol} (1y period).")
            # Daha kısa periyot deneyebilir miyiz? Örneğin 6 ay?
            logger.info(f"Applying rate limit before fetching 6mo history for Prophet ({stock_symbol})")
            _rate_limit('yfinance')
            logger.info(f"Fetching 6mo history for Prophet ({stock_symbol})")
            history = ticker.history(period="6mo", interval="1d", auto_adjust=True)
            if history.empty:
                 logger.error(f"Prophet forecast failed: No historical data found for {stock_symbol} (6mo period either).")
                 return {'error': f"{stock_symbol} için tahmin yapacak yeterli geçmiş veri (en az ~20 gün) bulunamadı."}

        logger.info(f"Fetched {len(history)} data points for Prophet ({stock_symbol}).")

        # Prophet veri formatına dönüştürme (ds, y)
        df = history.reset_index()
        date_col = None
        if 'Date' in df.columns: date_col = 'Date'
        elif 'Datetime' in df.columns: date_col = 'Datetime' # Bazen Datetime olabilir
        # Eğer index sütunu varsa ve tipi datetime ise onu kullan
        elif isinstance(df.index, pd.DatetimeIndex):
            df['ds'] = df.index
            date_col = 'ds' # Index'i ds yaptık
            df = df.reset_index(drop=True) # Eski index'i kaldır
        else: # Index'i kontrol et
             index_name = df.index.name
             if index_name and isinstance(df.index, pd.DatetimeIndex):
                  df.index.name = 'ds' # Index adını ds yap
                  df = df.reset_index()
                  date_col = 'ds'


        if not date_col:
             logger.error(f"Prophet forecast failed: Date column ('Date', 'Datetime', or DatetimeIndex) not found in history for {stock_symbol}. Columns: {df.columns}")
             return {'error': f"{stock_symbol} geçmiş verisinde tarih sütunu bulunamadı."}

        df = df.rename(columns={date_col: 'ds', 'Close': 'y'})

        # Tarih sütununu datetime yap ve zaman dilimini kaldır
        try:
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        except Exception as date_err:
            logger.error(f"Error converting 'ds' column to datetime for {stock_symbol}: {date_err}")
            return {'error': f"{stock_symbol} için tarih formatı dönüştürülemedi."}


        # 'y' sütununu sayısal yap, hataları NaN yap
        df['y'] = pd.to_numeric(df['y'], errors='coerce')

        # NaN değerleri doldur (önce ffill, sonra bfill)
        df['y'] = df['y'].ffill().bfill()

        # Hala NaN varsa hata ver
        if df['y'].isnull().any():
            logger.error(f"Prophet forecast failed: 'y' column contains NaNs after fill for {stock_symbol}")
            return {'error': f"{stock_symbol} için fiyat verisinde doldurulamayan boşluklar var."}

        # Çok az veri varsa Prophet hata verebilir (en az 2 gerektirir, ama pratik olarak daha fazla lazım)
        if len(df) < 5:
             logger.error(f"Prophet forecast failed: Insufficient data for {stock_symbol} (found {len(df)} rows, need at least 5)")
             return {'error': f"{stock_symbol} için güvenilir tahmin yapmak için çok az veri var (en az 5 gün gerekli)."}
        elif len(df) < 20:
              logger.warning(f"Prophet forecast: Data for {stock_symbol} is short ({len(df)} rows). Forecast accuracy might be low.")


        # Model kurulumu
        m = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05 # Default
            # growth='logistic', # Kapasite/taban belirtirsek kullanılabilir
            # holidays=... # Tatilleri ekleyebiliriz
        )

        # Regressor Ekleme (opsiyonel, hata olursa atla)
        if 'Volume' in df.columns and df['Volume'].nunique() > 1:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            if df['Volume'].sum() > 0: # Hacim sıfır değilse
                df['log_volume'] = np.log1p(df['Volume'])
                # NaN/Inf kontrolü
                if not df['log_volume'].isnull().any() and not np.isinf(df['log_volume']).any():
                    try:
                        m.add_regressor('log_volume')
                        logger.info(f"Added 'log_volume' as regressor for {stock_symbol}")
                    except Exception as reg_err:
                        logger.warning(f"Could not add 'log_volume' regressor for {stock_symbol}: {reg_err}")
                else: logger.warning(f"NaN or Inf found in log_volume for {stock_symbol}, not adding.")
            else: logger.warning(f"Volume data is all zero for {stock_symbol}, not adding regressor.")


        # Modeli eğit
        logger.info(f"Fitting Prophet model for {stock_symbol} with {len(df)} data points...")
        try:
             m.fit(df[['ds', 'y'] + ([col for col in ['log_volume'] if col in m.extra_regressors])]) # Sadece eklenen regressor'ları ver
        except Exception as fit_err:
             logger.error(f"Prophet model fitting failed for {stock_symbol}: {fit_err}", exc_info=True)
             return {'error': f"{stock_symbol} için Prophet modeli eğitilemedi: {fit_err}"}
        logger.info(f"Prophet model fitted for {stock_symbol}.")

        # Gelecek tarihleri oluştur
        future = m.make_future_dataframe(periods=periods, freq='D') # Günlük frekans
        future['ds'] = pd.to_datetime(future['ds']).dt.tz_localize(None)


        # Regressor'ları gelecek veriye ekle (son değeri veya ortalamayı kullan)
        if 'log_volume' in m.extra_regressors:
             # Gelecek için hacmi tahmin etmek zor, son değeri veya ortalamayı kullanabiliriz
             # Veya basitçe sıfır bırakabiliriz veya hiç eklemeyebiliriz. Şimdilik son değeri alalım.
             last_volume = df['log_volume'].iloc[-1] if 'log_volume' in df and not df['log_volume'].empty else 0
             future['log_volume'] = float(last_volume) if pd.notna(last_volume) else 0.0

        # Tahmini yap
        logger.info(f"Making predictions for {stock_symbol}...")
        try:
            forecast = m.predict(future)
        except Exception as pred_err:
             logger.error(f"Prophet prediction failed for {stock_symbol}: {pred_err}", exc_info=True)
             return {'error': f"{stock_symbol} için tahmin yapılamadı: {pred_err}"}
        logger.info(f"Predictions made for {stock_symbol}.")


        # --- Metrik Hesaplama ---
        # Trend (Son 7 güne bakarak)
        trend_direction = 'Yatay'
        trend_strength = 0.0
        if len(forecast['trend']) > 7:
             trend_segment = forecast['trend'].iloc[-7:] # Son 7 gün
             # Basit lineer regresyon eğimi veya ilk-son farkı
             trend_diff = trend_segment.iloc[-1] - trend_segment.iloc[0]
             last_trend_val = trend_segment.iloc[-1]
             if last_trend_val != 0 and pd.notna(trend_diff) and pd.notna(last_trend_val):
                 # Trend gücünü normalize edelim (7 günlük değişim / son değer)
                 trend_strength = abs(trend_diff / last_trend_val)
                 if trend_diff > 0.001 * last_trend_val: trend_direction = 'Yükseliş' # Küçük değişimleri yatay say
                 elif trend_diff < -0.001 * last_trend_val: trend_direction = 'Düşüş'
                 else: trend_direction = 'Yatay'
             else: trend_direction = 'Belirsiz'
        elif len(forecast['trend']) > 1 : # Daha az veri varsa ilk-son farkı
             trend_diff = forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]
             last_trend_val = forecast['trend'].iloc[-1]
             if last_trend_val != 0 and pd.notna(trend_diff) and pd.notna(last_trend_val):
                 trend_strength = abs(trend_diff / last_trend_val)
                 if trend_diff > 0.001 * last_trend_val: trend_direction = 'Yükseliş'
                 elif trend_diff < -0.001 * last_trend_val: trend_direction = 'Düşüş'
                 else: trend_direction = 'Yatay'
             else: trend_direction = 'Belirsiz'
        else: trend_direction = 'Belirsiz'

        # Mevsimsellik Gücü (Prophet'in kendi bileşenlerini kullan)
        seasonality_strength = 0.0
        total_seasonality_amplitude = 0
        component_count = 0
        for component in ['weekly', 'yearly']: # Mevsimsel bileşenler
             if component in forecast.columns and forecast[component].nunique() > 1:
                  # Ortalama mutlak değerini trendin ortalama mutlak değerine oranlayalım
                  comp_mean_abs = np.mean(np.abs(forecast[component]))
                  total_seasonality_amplitude += comp_mean_abs
                  component_count += 1

        if component_count > 0:
             trend_mean_abs = np.mean(np.abs(forecast['trend']))
             if trend_mean_abs > 1e-6:
                 seasonality_strength = total_seasonality_amplitude / trend_mean_abs


        # Tarihsel Volatilite (Son 30 gün veya mevcut veri)
        historical_volatility = 0.0
        if len(df) >= 5:
             df['returns'] = df['y'].pct_change().fillna(0)
             window_size = min(30, len(df)-1) # En az 2 veri noktası gerekli
             if window_size >= 2:
                 rolling_std = df['returns'].rolling(window=window_size).std()
                 # Son geçerli volatiliteyi al
                 last_valid_vol = rolling_std.dropna().iloc[-1] if not rolling_std.dropna().empty else None
                 if last_valid_vol is not None:
                      historical_volatility = last_valid_vol * np.sqrt(252) # Yıllık

        # Güven Aralığı Genişliği (Son tahmin noktası için, fiyata oranla)
        confidence_interval_relative = 0.0
        if not forecast.empty:
            last_yhat = forecast['yhat'].iloc[-1]
            last_upper = forecast['yhat_upper'].iloc[-1]
            last_lower = forecast['yhat_lower'].iloc[-1]
            if pd.notna(last_yhat) and pd.notna(last_upper) and pd.notna(last_lower) and last_yhat != 0:
                 confidence_interval_width = last_upper - last_lower
                 confidence_interval_relative = confidence_interval_width / abs(last_yhat)


        # Sonuçları formatlama
        result = {
            'forecast_values': {
                'dates': forecast['ds'].tail(periods).dt.strftime('%Y-%m-%d').tolist(),
                 'values': [float(v) if pd.notna(v) and np.isfinite(v) else None for v in forecast['yhat'].tail(periods).round(2).tolist()],
                 'upper_bound': [float(v) if pd.notna(v) and np.isfinite(v) else None for v in forecast['yhat_upper'].tail(periods).round(2).tolist()],
                 'lower_bound': [float(v) if pd.notna(v) and np.isfinite(v) else None for v in forecast['yhat_lower'].tail(periods).round(2).tolist()]
            },
            'metrics': {
                'trend_direction': trend_direction,
                 'trend_strength': float(trend_strength) if pd.notna(trend_strength) and np.isfinite(trend_strength) else 0.0,
                 'seasonality_strength': float(seasonality_strength) if pd.notna(seasonality_strength) and np.isfinite(seasonality_strength) else 0.0,
                 'historical_volatility': float(historical_volatility) if pd.notna(historical_volatility) and np.isfinite(historical_volatility) else 0.0,
                 'confidence_interval': float(confidence_interval_relative) if pd.notna(confidence_interval_relative) and np.isfinite(confidence_interval_relative) else 0.0 # Göreceli genişlik
            },
            'last_actual_date': df['ds'].iloc[-1].strftime('%Y-%m-%d') if not df.empty else None,
            'last_actual_price': float(df['y'].iloc[-1]) if not df.empty and pd.notna(df['y'].iloc[-1]) else None,
            'error': None # Başarılı ise hata yok
        }

        # Tahmin değerlerinde None varsa logla
        if None in result['forecast_values']['values']:
             logger.warning(f"Forecast for {stock_symbol} contains None values in 'values'. Check model output.")
        # Diğer None kontrolleri...

        logger.info(f"Prophet forecast successful for {stock_symbol}")
        return result

    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        logger.error(f"Prophet forecast failed for {stock_symbol} due to yfinance HTTP Error {status_code}: {str(http_err)}")
        error_msg = f"'{stock_symbol}' için veri alınırken HTTP hatası ({status_code})."
        if status_code == 429: error_msg += " Rate limit aşıldı."
        return {'error': error_msg}
    except Exception as e:
        logger.error(f"Prophet forecast failed for {stock_symbol}: {type(e).__name__} - {str(e)}", exc_info=True)
        return {'error': f"'{stock_symbol}' için tahmin hesaplanırken beklenmedik bir hata oluştu: {type(e).__name__}"}