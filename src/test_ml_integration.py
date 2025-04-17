#!/usr/bin/env python3
# src/test_ml_integration.py
# A simple test script to verify the ML integration

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from src.ml_models import StockPricePredictor
from src.model_comparison import ModelComparator
from src import forecasting
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_random_forest():
    logger.info("Testing RandomForest model")
    symbol = "AAPL"
    
    # Fetch data
    logger.info(f"Fetching data for {symbol}")
    df = yf.download(symbol, period='2y')
    df.reset_index(inplace=True)
    
    if len(df) < 100:
        logger.error(f"Insufficient data for {symbol}")
        return False
    
    try:
        # Create model
        logger.info("Creating RandomForest model")
        model = StockPricePredictor(model_type='regressor')
        
        # Train model
        logger.info("Training model")
        metrics = model.train(df)
        logger.info(f"Training metrics: {metrics}")
        
        # Test plotting
        logger.info("Testing plot_learning_curve")
        model.plot_learning_curve(df)
        
        logger.info("Testing plot_feature_importance")
        model.plot_feature_importance()
        
        logger.info("RandomForest test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in RandomForest test: {e}", exc_info=True)
        return False

def test_model_comparison():
    logger.info("Testing ModelComparator")
    symbol = "AAPL"
    
    try:
        # Create comparator
        logger.info("Creating ModelComparator")
        comparator = ModelComparator()
        
        # Fetch data
        logger.info(f"Fetching data for {symbol}")
        df = comparator.fetch_data(symbol, period='2y')
        
        if df is None or len(df) < 100:
            logger.error(f"Insufficient data for {symbol}")
            return False
        
        # Add RandomForest model
        logger.info("Adding RandomForest model")
        rf_metrics = comparator.add_random_forest_model(df)
        logger.info(f"RandomForest metrics: {rf_metrics}")
        
        # Add Prophet model
        logger.info("Adding Prophet model")
        prophet_metrics = comparator.add_prophet_model(symbol, forecasting.get_prophet_forecast)
        logger.info(f"Prophet metrics: {prophet_metrics}")
        
        # Compare models
        logger.info("Comparing models")
        comparison = comparator.compare_predictions(symbol, forecasting.get_prophet_forecast, df)
        logger.info(f"Comparison results: {comparison}")
        
        # Plot comparison
        logger.info("Plotting comparison")
        comparator.plot_comparison(symbol, forecasting.get_prophet_forecast, df)
        
        logger.info("Model comparison test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in model comparison test: {e}", exc_info=True)
        return False

def main():
    logger.info("Starting ML integration tests")
    
    # Ensure output directory exists
    os.makedirs('static/images', exist_ok=True)
    
    # Test RandomForest
    rf_success = test_random_forest()
    logger.info(f"RandomForest test {'passed' if rf_success else 'failed'}")
    
    # Test ModelComparator
    mc_success = test_model_comparison()
    logger.info(f"ModelComparator test {'passed' if mc_success else 'failed'}")
    
    logger.info("ML integration tests completed")
    
    if rf_success and mc_success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("One or more tests failed")
        return 1

if __name__ == "__main__":
    exit(main())