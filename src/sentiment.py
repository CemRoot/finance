# src/sentiment.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import logging
import os
import time

# Daha az ayrıntılı loglama için transformers log seviyesini ayarla
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__) # Bu modül için logger

class FinBERTSentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_pipeline = None # Pipeline'ı burada None başlat
        # Modeli hemen yüklemek yerine ilk kullanımda yükleyebiliriz (Lazy Loading)
        # self._load_model()

    def _load_model(self):
        """Modeli ve tokenizer'ı yükler (Eğer henüz yüklenmediyse)."""
        # Eğer pipeline zaten yüklüyse tekrar yükleme
        if self.sentiment_pipeline:
            return True

        try:
            logger.info(f"Loading FinBERT model ({self.model_name}) to {self.device}...")
            # Pipeline kullanmak daha verimli
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1, # device index: 0 for cuda, -1 for cpu
                top_k=None # Tüm etiketleri döndür (positive, negative, neutral)
            )
            logger.info(f"FinBERT model ({self.model_name}) loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading FinBERT model ({self.model_name}): {e}", exc_info=True)
            self.sentiment_pipeline = None # Hata durumunda None yap
            return False

    def analyze_sentiment(self, text):
        """
        Metni analiz eder ve duygu skorunu döndürür:
        -1.0 (çok negatif) ile 1.0 (çok pozitif) arasında.
        """
        # Model yüklenmemişse yüklemeyi dene
        if not self.sentiment_pipeline:
            if not self._load_model(): # Yükleme başarısız olursa
                 logger.error("Sentiment model could not be loaded. Returning neutral score.")
                 return 0.0

        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0 # Boş metin için nötr skor

        try:
            # Metni kısalt (çok uzunsa hata verebilir)
            # Tokenizer'ın max_length'i genellikle 512'dir
            max_length = 510 # Güvenlik payı bırakalım
            if len(text) > max_length:
                text = text[:max_length]

            # Pipeline ile analiz yap
            results = self.sentiment_pipeline(text)
            # Pipeline top_k=None ile liste içinde dict döndürür: [{'label': 'positive', 'score': 0.9}, ...]

            if not results: # Eğer sonuç boşsa
                logger.warning(f"Sentiment analysis returned empty results for: '{text[:50]}...'")
                return 0.0

            # Skorları ayıkla
            scores = {item['label']: item['score'] for item in results[0]} # İlk eleman liste içeriyor
            positive_score = scores.get('positive', 0.0)
            negative_score = scores.get('negative', 0.0)
            # neutral_score = scores.get('neutral', 0.0) # Nötr skorunu da alabiliriz

            # Skoru hesapla: pozitif - negatif
            final_score = positive_score - negative_score

            return float(final_score)

        except Exception as e:
            # Hata logunu daha bilgilendirici yap
            logger.error(f"Error analyzing sentiment for text (len={len(text)}): '{text[:50]}...': {e}", exc_info=False) # Sadece mesajı logla
            return 0.0 # Hata durumunda nötr dön

# Singleton instance oluştur (Lazy loading ile model ilk istekte yüklenecek)
sentiment_analyzer = FinBERTSentimentAnalyzer()

def analyze_sentiment(text):
    """
    FinBERT kullanarak metnin duygu analizini yapar.
    Singleton instance'ı kullanır.
    """
    return sentiment_analyzer.analyze_sentiment(text)


def analyze_articles_sentiment(articles):
    """
    Verilen haber makaleleri listesindeki metinlerden duygu analizi yapar
    ve ortalama duygu skorunu döner.
    """
    if not articles:
        return 0.0

    total_score = 0.0
    valid_articles = 0
    logger.info(f"Analyzing sentiment for {len(articles)} articles...")
    start_time = time.time()

    for i, article in enumerate(articles):
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        text_to_analyze = (title + " " + description).strip()

        if text_to_analyze:
            score = analyze_sentiment(text_to_analyze)
            # logger.debug(f"Article {i+1} score: {score:.3f} for '{title[:30]}...'")
            total_score += score
            valid_articles += 1
        # else: logger.debug(f"Article {i+1} skipped (empty text).")


    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Sentiment analysis for {valid_articles} articles took {duration:.2f} seconds.")

    if valid_articles == 0:
        return 0.0

    average_score = total_score / valid_articles
    logger.info(f"Calculated average sentiment score: {average_score:.4f}")
    return average_score