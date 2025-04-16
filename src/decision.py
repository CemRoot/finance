# src/decision.py
import src.sentiment as sentiment # Import the sentiment module
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Decision Thresholds (Used by sentiment-only logic) ---
# BUY_THRESHOLD = 0.15
# SELL_THRESHOLD = -0.15

# --- Original Sentiment-Only Decision Function (Now Commented Out) ---
# def make_decision(articles: list) -> str:
#     """
#     Generates a simple Buy/Sell/Hold decision based *only* on average news sentiment.
#     (This version is now commented out in favor of the combined logic below)
#     """
#     if not articles or not isinstance(articles, list) or len(articles) == 0:
#         logger.warning("Decision analysis skipped: Input 'articles' is empty or invalid.")
#         return "No Data"
#     logger.info(f"Calculating average sentiment for decision from {len(articles)} articles...")
#     try:
#         avg_sentiment = sentiment.analyze_articles_sentiment(articles)
#         logger.info(f"Decision analysis based on average sentiment: {avg_sentiment:.4f}")
#     except Exception as e:
#         logger.error(f"Unexpected error during sentiment calculation call in decision module: {e}", exc_info=True)
#         return "Analysis Unavailable"
#
#     if avg_sentiment > BUY_THRESHOLD:
#         decision = "Buy"
#     elif avg_sentiment < SELL_THRESHOLD:
#         decision = "Sell"
#     else:
#         decision = "Hold"
#
#     logger.info(f"Final Decision based *only* on sentiment: {decision} (Score: {avg_sentiment:.4f})")
#     return decision

# --- Combined Decision Logic (Using Indicators and Sentiment) ---
# *** This function is now intended to be called from app.py ***
def make_decision_with_indicators(indicators: dict, avg_sentiment: float) -> str:
    """
    Generates a decision combining technical indicator statuses and news sentiment.

    Args:
        indicators (dict): Dictionary containing calculated technical indicator statuses
                           (e.g., {'rsi': 'Neutral', 'macd': 'Buy Signal', 'sma': 'Above'}).
                           Expected keys: 'rsi', 'macd', 'sma'.
        avg_sentiment (float): The average sentiment score from news articles.

    Returns:
        str: 'Buy', 'Sell', or 'Hold' decision. Returns 'Hold' if indicators are missing.
    """
    # Check if indicators dictionary is provided and seems valid
    if not indicators or not isinstance(indicators, dict):
        logger.warning("Decision with indicators skipped: Indicators dictionary is missing or invalid. Falling back to Hold.")
        # Fallback to Hold if indicators are missing, as sentiment alone might not be enough here.
        return "Hold"

    # Check for essential indicator keys (optional but recommended)
    # required_keys = ['rsi', 'macd', 'sma']
    # if not all(key in indicators for key in required_keys):
    #     logger.warning(f"Decision with indicators skipped: Missing one or more required indicators ({required_keys}). Falling back to Hold.")
    #     return "Hold"


    logger.info(f"Making decision with indicators: {indicators}, Sentiment: {avg_sentiment:.4f}")

    # --- Scoring System Example ---
    # Initialize score
    score = 0

    # 1. Sentiment Component
    # More granular thresholds for sentiment's contribution
    if avg_sentiment > 0.20: score += 1.0   # Strongly positive
    elif avg_sentiment > 0.05: score += 0.5  # Mildly positive
    elif avg_sentiment < -0.20: score -= 1.0 # Strongly negative
    elif avg_sentiment < -0.05: score -= 0.5 # Mildly negative
    # Neutral sentiment (between -0.05 and 0.05) contributes 0 points

    # 2. RSI Component
    rsi_status = indicators.get('rsi') # Use .get() for safe access
    if rsi_status == 'Oversold': score += 1.0
    elif rsi_status == 'Overbought': score -= 1.0
    # Neutral RSI contributes 0 points

    # 3. MACD Component
    macd_status = indicators.get('macd')
    # Check for specific signal strings
    if macd_status == 'Buy Signal (Positive)': score += 1.0
    elif macd_status == 'Sell Signal (Negative)': score -= 1.0
    # No signal or unclear contributes 0 points

    # 4. SMA Component (Price vs SMA)
    sma_status = indicators.get('sma')
    if sma_status == 'Above': score += 0.5 # Price above SMA is bullish bias
    elif sma_status == 'Below': score -= 0.5 # Price below SMA is bearish bias
    # 'Equal' or missing contributes 0 points

    # --- Final Decision Based on Score ---
    # Adjust score thresholds based on strategy and backtesting
    # These thresholds determine how strong the combined signal needs to be.
    buy_score_threshold = 1.5  # Need multiple positive signals
    sell_score_threshold = -1.5 # Need multiple negative signals

    final_decision = "Hold" # Default to Hold
    if score >= buy_score_threshold:
        final_decision = "Buy"
    elif score <= sell_score_threshold:
        final_decision = "Sell"

    logger.info(f"Combined Decision: Score = {score:.1f}, Final Decision = {final_decision}")
    return final_decision