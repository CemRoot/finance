# Machine Learning Model Integration

This document explains how to integrate the new machine learning models and comparison features into the Finance Dashboard application.

## Files Added

1. `src/ml_models.py` - RandomForestRegressor model implementation for stock price prediction
2. `src/model_comparison.py` - Utility for comparing the RandomForest and Prophet models
3. `templates/ml_analysis.html` - Template for displaying RandomForest analysis results
4. `templates/model_comparison.html` - Template for displaying model comparison results
5. `templates/partials/_sidebar_ml.html` - Modified sidebar template with ML options
6. `static/js/ml-analysis.js` - JavaScript for ML analysis interactions
7. `app_routes.py` - New routes for ML features to be added to app.py

## Integration Steps

### 1. Add Routes to app.py

Copy the routes from `app_routes.py` into your main `app.py` file at an appropriate location, such as before the main execution block.

### 2. Use the ML Sidebar (Optional)

If you want to use the ML-enabled sidebar, modify your `base.html` template:

```html
<div class="col-lg-2 col-md-3 sidebar px-0 d-none d-md-flex flex-column">
    {% include 'partials/_sidebar_ml.html' %}
</div>
```

Alternatively, you can add links to the ML features in your existing sidebar or navigation.

### 3. Create Static Image Directory

Ensure the static/images directory exists for saving chart images:

```bash
mkdir -p static/images
```

### 4. Add Model Tabs to Dashboard (Optional)

If you want to add ML model tabs directly to the dashboard, modify your `index.html`:

```html
<!-- ML Analysis Tab Pane -->
<div class="tab-pane fade" id="ml-analysis" role="tabpanel" aria-labelledby="ml-analysis-tab">
    <div class="row g-4">
        <div class="col-12">
            <div class="card shadow-sm border-0">
                <div class="card-header bg-light border-0">
                    <h5 class="card-title mb-0 fs-6 text-dark">Machine Learning Analysis</h5>
                </div>
                <div class="card-body">
                    <div class="d-flex gap-3 flex-wrap">
                        <a href="{{ url_for('random_forest_analysis', stock=stock) }}" class="btn btn-primary">
                            <i class="fas fa-tree me-1"></i> Random Forest Analysis
                        </a>
                        <a href="{{ url_for('model_comparison', stock=stock) }}" class="btn btn-primary">
                            <i class="fas fa-balance-scale me-1"></i> Model Comparison
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
```

And add a new tab to the sidebar:

```html
<li class="nav-item" role="presentation">
    <a class="nav-link" id="ml-analysis-tab" data-bs-toggle="tab" href="#ml-analysis" role="tab"
        aria-controls="ml-analysis" aria-selected="false">
        <i class="fas fa-brain fa-fw me-2"></i> ML Analysis
    </a>
</li>
```

## Features

1. **Random Forest Analysis**
   - Stock price prediction using RandomForestRegressor
   - Feature importance visualization
   - Learning curves
   - Evaluation with different training sizes

2. **Model Comparison**
   - Compare RandomForest and Prophet predictions
   - Performance metrics (MAE, RMSE, R²)
   - Execution time comparison
   - Visual comparison of predictions

## Usage

1. Navigate to any stock using the search function
2. Click on "Random Forest" in the sidebar to see the RF analysis
3. Click on "Model Comparison" to compare Prophet and RandomForest models

## Dependencies

Make sure the following Python packages are installed:

- scikit-learn
- matplotlib
- joblib
- pandas
- numpy
- yfinance

You can install them with:

```bash
pip install scikit-learn matplotlib joblib pandas numpy yfinance
```