# src/sentiment.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import logging
import os
import time

# Configure logger for this module
logger = logging.getLogger(__name__)
# Suppress excessive logging from transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)

class FinBERTSentimentAnalyzer:
    """
    A class to perform sentiment analysis using a pre-trained FinBERT model.
    Implements lazy loading for the model.
    """
    def __init__(self, model_name="ProsusAI/finbert"):
        self.model_name = model_name
        # Determine device (use GPU if available, otherwise CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_pipeline = None
        self._model_loaded = False
        logger.info(f"FinBERTSentimentAnalyzer initialized. Model: '{self.model_name}', Device: {self.device}")

    def _load_model(self):
        """
        Loads the FinBERT model and tokenizer lazily when first needed.
        Uses the Hugging Face pipeline for sentiment analysis.
        """
        if self._model_loaded:
            return True # Already loaded

        try:
            logger.info(f"LAZY LOADING: Loading FinBERT model ('{self.model_name}') to device '{self.device}'...")
            start_time = time.time()

            # Determine device index for pipeline (-1 for CPU, 0 for first GPU)
            device_index = 0 if self.device == "cuda" else -1

            # Create the sentiment analysis pipeline
            # Ensure top_k=None to get scores for all labels (positive, negative, neutral)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device_index,
                top_k=None # Get all scores
            )
            end_time = time.time()
            logger.info(f"LAZY LOADING: FinBERT model loaded successfully in {end_time - start_time:.2f} seconds.")
            self._model_loaded = True
            return True
        except Exception as e:
            # Log detailed error during model loading
            logger.error(f"Error loading FinBERT model ('{self.model_name}'): {e}", exc_info=True)
            self.sentiment_pipeline = None
            self._model_loaded = False
            return False

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyzes the sentiment of a given text.

        Args:
            text (str): The input text to analyze.

        Returns:
            float: A sentiment score between -1.0 (very negative) and +1.0 (very positive).
                   Returns 0.0 for invalid input or if analysis fails.
        """
        # Ensure the model is loaded before proceeding
        if not self._model_loaded:
            if not self._load_model():
                logger.error("Sentiment model could not be loaded. Cannot analyze.")
                return 0.0 # Return neutral score if model failed to load

        # Check if pipeline is available (should be if _load_model succeeded)
        if not self.sentiment_pipeline:
            logger.error("Sentiment analysis pipeline is not available.")
            return 0.0

        # Input validation
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            # logger.debug("Input text is empty or invalid. Returning neutral sentiment.")
            return 0.0 # Return neutral for empty/invalid input

        try:
            # Truncate text to avoid issues with models having max sequence length limits
            # FinBERT's max length is typically 512 tokens. Use slightly less for safety.
            # Note: Token count is not exactly character count, but this is a practical approximation.
            max_length = 510
            truncated_text = text[:max_length] if len(text) > max_length else text

            # Perform sentiment analysis using the pipeline
            results = self.sentiment_pipeline(truncated_text)

            # The pipeline with top_k=None returns a list containing a list of dictionaries,
            # e.g., [[{'label': 'positive', 'score': 0.9}, {'label': 'negative', 'score': 0.05}, ...]]
            if not results or not isinstance(results, list) or not results[0]:
                logger.warning(f"Sentiment analysis returned empty or unexpected results for text: '{truncated_text[:50]}...'")
                return 0.0

            # Extract the list of scores (it's nested inside the first element)
            scores_list = results[0]
            if not isinstance(scores_list, list):
                 logger.error(f"Unexpected sentiment result format (expected list of dicts): {scores_list}")
                 return 0.0

            # Convert the list of score dicts into a single dictionary mapping label to score
            scores = {item['label'].lower(): item['score'] for item in scores_list if isinstance(item, dict) and 'label' in item and 'score' in item}

            # Extract positive and negative scores (default to 0.0 if a label is missing)
            positive_score = scores.get('positive', 0.0)
            negative_score = scores.get('negative', 0.0)
            # Neutral score is available but not used in the final score calculation here
            # neutral_score = scores.get('neutral', 0.0)

            # Calculate final score: positive - negative
            # This score ranges from -1.0 (100% negative) to +1.0 (100% positive).
            # A score around 0 indicates neutral or mixed sentiment.
            final_score = positive_score - negative_score

            # Ensure the score is float
            return float(final_score)

        except Exception as e:
            # Log errors during analysis, but avoid flooding logs with full text
            logger.error(f"Error analyzing sentiment (text length={len(text)}): '{text[:50]}...' - {e}", exc_info=False) # exc_info=False reduces log noise
            return 0.0 # Return neutral score on error

# --- Global Instance ---
# Instantiate the analyzer globally. Lazy loading handles the actual model loading on first use.
sentiment_analyzer = FinBERTSentimentAnalyzer()

# --- Public Access Functions ---

def analyze_sentiment(text: str) -> float:
    """
    Public function to easily access sentiment analysis for a single text.
    Uses the global sentiment_analyzer instance.

    Args:
        text (str): The text to analyze.

    Returns:
        float: Sentiment score (-1.0 to +1.0).
    """
    # Delegate to the method of the global analyzer instance
    return sentiment_analyzer.analyze_sentiment(text)

def analyze_articles_sentiment(articles: list) -> float:
    """
    Calculates the average sentiment score for a list of articles.
    Each article is expected to be a dictionary with 'title' and 'description'.

    Args:
        articles (list): A list of article dictionaries.

    Returns:
        float: The average sentiment score (-1.0 to +1.0) across all valid articles.
               Returns 0.0 if no valid articles are found or input is invalid.
    """
    if not articles or not isinstance(articles, list):
        logger.warning("Invalid input: 'articles' must be a non-empty list.")
        return 0.0

    total_score = 0.0
    valid_articles_count = 0
    logger.info(f"Analyzing sentiment for {len(articles)} articles...")
    start_time = time.time()

    # Ensure model is loaded before processing batch (more efficient)
    if not sentiment_analyzer._model_loaded:
        if not sentiment_analyzer._load_model():
             logger.error("Sentiment model could not be loaded. Cannot analyze articles.")
             return 0.0

    for i, article in enumerate(articles):
        # Basic validation for each article
        if not isinstance(article, dict):
            # logger.debug(f"Skipping item {i+1}: Not a dictionary.")
            continue # Skip if not a dictionary

        # Combine title and description for analysis, ensuring they are strings
        title = str(article.get("title", "")) # Ensure string
        description = str(article.get("description", "")) # Ensure string
        text_to_analyze = (title + " " + description).strip() # Combine and remove leading/trailing whitespace

        if text_to_analyze: # Only analyze if there's text content
            # Use the single text analysis function (which uses the global instance)
            score = analyze_sentiment(text_to_analyze)
            total_score += score
            valid_articles_count += 1
            # Optional: Log individual scores for debugging
            # logger.debug(f"Article {i+1}/{len(articles)} sentiment: {score:.4f}")
        # else:
            # logger.debug(f"Skipping article {i+1}: No text content found in title/description.")

    end_time = time.time()
    duration = end_time - start_time

    if valid_articles_count == 0:
        logger.warning("No valid articles with text content found to analyze sentiment.")
        return 0.0 # Return neutral if no articles were analyzed

    average_score = total_score / valid_articles_count
    logger.info(f"Sentiment analysis for {valid_articles_count} articles completed in {duration:.2f} seconds. Average score: {average_score:.4f}")
    return average_score