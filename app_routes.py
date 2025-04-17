# app_routes.py
# Add these routes to your app.py file

from flask import app, render_template, request, flash, redirect, url_for
import yfinance as yf
import os
import logging
from src.ml_models import StockPricePredictor
from src.model_comparison import ModelComparator
import src.forecasting as forecasting  # Import your existing forecasting module

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Route for Random Forest Analysis
@app.route('/random_forest_analysis')
def random_forest_analysis():
    stock = request.args.get('stock')
    if not stock:
        flash("Please select a stock first.", "warning")
        return redirect(url_for('index'))
    
    try:
        logger.info(f"Starting Random Forest analysis for {stock}")
        
        # Fetch stock data
        df = yf.download(stock, period='2y')
        df.reset_index(inplace=True)
        
        if len(df) < 100:
            flash(f"Insufficient data for {stock} to perform analysis.", "warning")
            return redirect(url_for('index'))
        
        # Create and train the Random Forest model
        model = StockPricePredictor(model_type='regressor')
        metrics = model.train(df)
        
        # Generate learning curve and get image path
        learning_curve_data = model.plot_learning_curve(df)
        learning_curve_path = learning_curve_data.get('image_path', '/static/images/learning_curve.png')
        
        # Evaluate with different data sizes
        different_sizes = model.evaluate_with_different_sizes(df)
        
        # Plot feature importance and get image path
        feature_importance_data = model.plot_feature_importance()
        feature_importance_path = feature_importance_data.get('image_path', '/static/images/feature_importance.png')
        
        logger.info(f"Random Forest analysis completed for {stock}")
        
        return render_template(
            'ml_analysis.html',
            stock=stock,
            metrics=metrics,
            different_sizes=different_sizes,
            learning_curve_path=learning_curve_path,
            feature_importance_path=feature_importance_path,
            model_type="Random Forest"
        )
    except Exception as e:
        logger.error(f"Error in Random Forest analysis for {stock}: {e}", exc_info=True)
        flash(f"Error generating Random Forest analysis: {str(e)}", "danger")
        return redirect(url_for('index'))

# Route for Model Comparison
@app.route('/model_comparison')
def model_comparison():
    stock = request.args.get('stock')
    if not stock:
        flash("Please select a stock first.", "warning")
        return redirect(url_for('index'))
    
    try:
        logger.info(f"Starting model comparison for {stock}")
        
        # Create model comparator
        comparator = ModelComparator()
        
        # Fetch data
        df = comparator.fetch_data(stock, period='5y')
        
        if df is None or len(df) < 100:
            flash(f"Insufficient data for {stock} to perform comparison.", "warning")
            return redirect(url_for('index'))
        
        # Run model comparison
        comparison_metrics = comparator.compare_models(stock, df)
        
        if comparison_metrics.get('error'):
            flash(f"Error in model comparison: {comparison_metrics['error']}", "warning")
            return redirect(url_for('index'))
        
        # Generate plots
        comparison_plot_data = comparator.plot_comparison()
        performance_plot_data = comparator.plot_performance_comparison()
        
        logger.info(f"Model comparison completed for {stock}")
        
        return render_template(
            'model_comparison.html',
            stock=stock,
            comparison_image=comparison_plot_data.get('image_path', '/static/images/model_comparison.png'),
            performance_image=performance_plot_data.get('image_path', '/static/images/performance_comparison.png'),
            metrics=performance_plot_data.get('metrics', {})
        )
    except Exception as e:
        logger.error(f"Error in model comparison for {stock}: {e}", exc_info=True)
        flash(f"Error generating model comparison: {str(e)}", "danger")
        return redirect(url_for('index'))