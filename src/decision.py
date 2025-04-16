# src/decision.py
import src.sentiment as sentiment
import logging

logger = logging.getLogger(__name__)

def get_decision(articles):
    """
    Haberlerin ortalama duygu skoruna göre basit bir Al/Sat/Bekle kararı üretir.
    """
    if not articles:
        logger.info("Decision: Not enough data (no articles).")
        return "Veri Yok"

    # Ortalama duygu skorunu hesapla (sentiment.py içinden)
    logger.info("Calculating average sentiment for decision...")
    avg_sentiment = sentiment.analyze_articles_sentiment(articles)
    logger.info(f"Decision based on average sentiment: {avg_sentiment:.4f}")

    # Eşik değerleri (Ayarlanabilir)
    # Daha hassas olmak için eşikleri biraz daraltabiliriz
    buy_threshold = 0.15
    sell_threshold = -0.15

    if avg_sentiment > buy_threshold:
        decision = "Al"
    elif avg_sentiment < sell_threshold:
        decision = "Sat"
    else:
        decision = "Bekle"

    logger.info(f"Final Decision: {decision}")
    return decision