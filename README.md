# Finance Analysis Dashboard

## Project Overview

The Finance Analysis Dashboard is a comprehensive web application designed for detailed financial data analysis, visualization, and prediction. It empowers users to investigate stock performance through advanced technical indicators, track financial news with sentiment analysis, and forecast future price movements using machine learning algorithms. This full-stack solution combines real-time data integration, interactive visualizations, and intelligent analysis tools to provide a complete financial research platform.

## Core Features

### Data Acquisition and Visualization
- **Multi-modal Stock Search**: Access stocks via ticker symbol (e.g., AAPL) or company name (e.g., Apple) with intelligent name resolution
- **Customizable Historical Data**: View adjustable time periods with daily, weekly, or monthly aggregation
- **Interactive Charts**: Professional candlestick charts with volume indicators and customizable parameters
- **Real-time Updates**: Refresh stock data on demand with the latest market information

### Technical Analysis
- **Comprehensive Technical Indicators**:
  - **Simple Moving Average (SMA)**: Trend identification using various time periods (7, 20, 50, 200 days)
  - **Relative Strength Index (RSI)**: Overbought/oversold momentum identification (14-day period)
  - **Moving Average Convergence Divergence (MACD)**: Signal line crossovers for trend strength/direction
  - **Bollinger Bands**: Volatility channels with 20-day period and 2 standard deviations
  - **Volume Analysis**: Trading volume patterns with moving averages
  - **Volatility Metrics**: Historical and implied volatility calculations

### News and Sentiment Integration
- **Curated News Feed**: Latest articles from financial sources filtered by relevance
- **Sentiment Analysis**: NLP-based sentiment scoring of news articles (positive/negative/neutral)
- **News Impact Visualization**: Correlation between news sentiment and price movements
- **Historical News Context**: Archive of significant news events mapped to price action

### Trading Signal Generation
- **Automated Decision Support**: Buy/Sell/Hold recommendations based on technical indicator confluence
- **Signal Strength Metrics**: Confidence scores for trading signals
- **Multi-timeframe Validation**: Signal confirmation across different time periods

### Forecasting and Machine Learning
- **Time Series Forecasting**: Price predictions using statistical models
- **Machine Learning Integration**:
  - **Random Forest Analysis**: Non-linear regression for price prediction with feature importance
  - **XGBoost Model**: Gradient boosting for enhanced prediction accuracy
  - **Prophet Model**: Time series decomposition with trend and seasonality
  - **Model Comparison**: Performance metrics and visual comparison between prediction models
  - **Learning Curve Analysis**: Model validation with varying training sizes

### Market Research Tools
- **Multi-stock Comparison**: Side-by-side analysis of related stocks
- **Sector Performance**: Industry-specific metrics and benchmarking
- **International Markets**: Support for global exchanges including US and Turkish markets

## Technical Implementation

### Architecture
- **Backend**: Flask web framework with Python
- **Frontend**: HTML5, CSS3, Bootstrap 5, and JavaScript
- **Data Visualization**: Chart.js and custom rendering
- **Caching System**: Flask-Caching with tiered timeout strategy
- **Database**: In-memory data storage with option for persistent storage
- **Logging**: Comprehensive logging system with configurable verbosity

### API Integrations
- **yfinance**: General historical and real-time market data
- **MarketStack API**: Data source for machine learning models and technical indicators
- **EventRegistry**: News article aggregation and filtering
- **Custom APIs**: Internal data processing and analysis endpoints

### Machine Learning Pipeline
- **Data Preprocessing**: Automated cleaning, normalization, and feature engineering
- **Model Training**: Scikit-learn and XGBoost implementation with cross-validation
- **Feature Importance**: Identification of key price-influencing factors
- **Model Persistence**: Saved models for rapid redeployment
- **Evaluation Metrics**: MAE, RMSE, R² for performance assessment

### Performance Optimizations
- **Strategic Caching**: Time-based caching strategy for API responses
- **Asynchronous Loading**: Non-blocking UI updates for seamless user experience
- **Lazy Chart Initialization**: On-demand rendering of complex visualizations
- **Efficient Data Processing**: Vectorized operations with pandas and numpy

## Installation Guide

### Prerequisites
- Python 3.8+ installed
- Git (for repository cloning)
- Internet connection for API access
- EventRegistry API key (for news functionality)
- MarketStack API key (for machine learning models)

### Setup Process
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Finance
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   ```bash
   # For Unix/Linux/macOS
   export EVENTREGISTRY_API_KEY=your_api_key
   export MARKETSTACK_API_KEY=your_marketstack_api_key
   export FLASK_DEBUG=true  # For development

   # For Windows
   set EVENTREGISTRY_API_KEY=your_api_key
   set MARKETSTACK_API_KEY=your_marketstack_api_key
   set FLASK_DEBUG=true
   ```

5. **Create Required Directories** (if not present):
   ```bash
   mkdir -p static/images
   ```

6. **Launch the Application**:
   ```bash
   python app.py
   ```

7. **Access the Dashboard**:
   Open your browser and navigate to:
   ```
   http://localhost:5001
   ```

## Detailed Usage Guide

### Stock Search and Navigation
1. **Search Methods**:
   - **Symbol Search**: Enter ticker symbols directly (e.g., "AAPL", "MSFT")
   - **Company Name Search**: Use full or partial company names (e.g., "Apple", "Microsoft")
   - **Quick Access**: Click popular stock buttons for immediate analysis

2. **Supported Markets**:
   - **US Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, etc.
   - **Turkish Stocks**: THYAO.IS, GARAN.IS, BIMAS.IS, ASELS.IS, etc.

### Dashboard Navigation
The application provides a multi-tab interface for comprehensive analysis:

1. **Overview Tab**:
   - Main price chart with selectable timeframes
   - Current market data (open, high, low, close, volume)
   - Performance metrics (daily change, volatility)

2. **News Tab**:
   - Latest news articles related to the selected stock
   - Sentiment classification with visual indicators
   - Source attribution and publication timestamps

3. **Details Tab**:
   - Advanced technical indicators with explanations
   - Volume trend analysis
   - Support and resistance levels

4. **Forecasting Tab**:
   - Statistical price projections
   - Confidence intervals
   - Trend analysis

5. **ML Analysis**:
   - Random Forest model insights
   - XGBoost analysis for improved predictions
   - Feature importance visualization
   - Model performance metrics

6. **Model Comparison**:
   - Side-by-side evaluation of different forecasting approaches
   - Accuracy metrics (MAE, RMSE, R²)
   - Execution time comparison

### Technical Analysis Tools
Instructions for utilizing technical indicators:

1. **Moving Averages**:
   - Interpret trend direction by comparing price to SMAs
   - Identify potential support/resistance at SMA lines
   - Look for crossovers between different period SMAs

2. **RSI Analysis**:
   - Values above 70 indicate overbought conditions
   - Values below 30 indicate oversold conditions
   - Divergence between price and RSI can signal potential reversals

3. **MACD Interpretation**:
   - Crossovers of MACD line and signal line generate trading signals
   - MACD histogram shows momentum strength
   - Zero line crossovers indicate trend changes

4. **Bollinger Bands Strategy**:
   - Price touching upper band may indicate overbought conditions
   - Price touching lower band may indicate oversold conditions
   - Band contraction suggests decreased volatility, often preceding significant moves

5. **Volume Analysis**:
   - Confirm price movements with corresponding volume
   - Look for volume divergence as potential reversal signal
   - Identify institutional participation through volume spikes

## System Architecture

The application follows a modular architecture with these components:

1. **Core Application (app.py)**:
   - Flask application initialization
   - Route definitions
   - Caching configuration
   - Request handling

2. **Data Processing Modules**:
   - Stock data retrieval and processing
   - Technical indicator calculation
   - News API integration
   - Data transformation and normalization

3. **Analysis Modules**:
   - Decision support system
   - Sentiment analysis
   - Forecasting algorithms
   - Machine learning models

4. **Template Structure**:
   - Base layout templates
   - Functional component templates
   - Reusable partial templates
   - Specialized visualization templates

5. **Static Assets**:
   - Custom CSS styling
   - JavaScript functionality
   - Image resources
   - Generated visualizations

## Technical Limitations and Considerations

1. **API Dependencies**:
   - EventRegistry API requires a valid API key
   - yfinance is subject to rate limiting and data availability
   - Some functionality may be limited without API access

2. **Data Limitations**:
   - Historical data availability varies by stock
   - Newer or less liquid stocks may have insufficient data for ML analysis
   - News coverage varies by company size and market

3. **Processing Considerations**:
   - Machine learning analysis requires sufficient historical data (minimum 100 data points)
   - Complex visualizations may impact performance on slower devices
   - Simultaneous analysis of multiple stocks may be resource-intensive

4. **Accuracy Considerations**:
   - Forecasting models provide estimates not guarantees
   - Technical indicators should be used in conjunction, not isolation
   - Financial markets are inherently unpredictable and subject to external factors

## Future Development Roadmap

1. **Enhanced Analysis**:
   - Additional technical indicators
   - Advanced pattern recognition
   - Options analytics integration

2. **Expanded ML Capabilities**:
   - Deep learning models (LSTM, Transformers)
   - Ensemble model approaches
   - Hyperparameter optimization

3. **UI Enhancements**:
   - Customizable dashboards
   - Advanced charting tools
   - Mobile-optimized experience

4. **Data Integration**:
   - Fundamental data analysis
   - Economic indicator correlation
   - Social media sentiment

## Troubleshooting

### Common Issues and Solutions

1. **Data Retrieval Failures**:
   - Check internet connection
   - Verify API key configuration
   - Ensure stock symbol is valid

2. **Visualization Problems**:
   - Clear browser cache
   - Update to latest browser version
   - Check for JavaScript console errors

3. **ML Analysis Errors**:
   - Ensure sufficient historical data exists
   - Verify required packages are installed
   - Check disk space for image generation

## Dependencies

The application relies on these key Python packages:

- **Web Framework**: Flask, Flask-Caching
- **Data Processing**: pandas, numpy, pandas_ta
- **Data Retrieval**: yfinance, EventRegistry
- **Machine Learning**: scikit-learn, Prophet
- **Visualization**: matplotlib, Chart.js
- **Utility**: joblib, pytz, logging

## Contributing Guidelines

Contributions to the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- yfinance for providing market data access
- EventRegistry for news aggregation capabilities
- Open-source community for various libraries and tools
- Faculty advisors for project guidance and support
