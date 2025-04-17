# Finance Analysis Dashboard

A web application that allows users to analyze financial data, news, and forecasts for stocks. The application provides detailed information, technical indicators, and news articles for a given stock symbol or company name.

## Features

- **Company Name Search**: Search for stocks using either the ticker symbol (e.g., AAPL) or the company name (e.g., Apple)
- **Stock Data Visualization**: View historical price charts with candlestick data
- **Technical Indicators**: Analyze stocks using various technical indicators including:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Volatility
- **News Integration**: Get the latest news related to specific stocks
- **Sentiment Analysis**: Analyze the sentiment of news articles
- **Market Decision Support**: Get automated trading signals based on technical indicators
- **Forecasting**: View price forecasts using time series analysis

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Finance
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set environment variables:
   ```
   export EVENTREGISTRY_API_KEY=your_api_key  # On Windows: set EVENTREGISTRY_API_KEY=your_api_key
   export FLASK_DEBUG=true  # For development
   ```

5. Run the application:
   ```
   python app.py
   ```

6. Open your browser and go to:
   ```
   http://localhost:5000
   ```

## Usage

### Searching for Stocks

The application supports two ways to search for stocks:

1. **Using Stock Symbols**: Enter a ticker symbol like "AAPL" for Apple, "MSFT" for Microsoft, etc.
2. **Using Company Names**: Enter a company name like "Apple", "Microsoft", "Coca Cola", etc.

The application will automatically determine if you've entered a company name or a stock symbol and will resolve it to the appropriate ticker symbol.

### Popular Stocks

For quick access, you can click on any of the popular stock buttons on the home page:

- US Stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA
- Turkey Stocks: THYAO.IS, GARAN.IS, BIMAS.IS, ASELS.IS

### Dashboard Tabs

The dashboard provides several tabs for analyzing stocks:

1. **Overview**: Shows the stock price chart, current price, and basic market data
2. **News**: Displays recent news articles related to the stock
3. **Details**: Shows additional details about the stock
4. **Forecasting**: Provides price forecasts based on historical data

## Technical Implementation Details

### Company Name Search

The application maintains a dictionary of popular stock symbols with associated name fragments for lookup. When a user enters a search query, the application:

1. Checks if the input looks like a stock symbol (uppercase, 1-5 characters)
2. If not, it tries to match the input against company names in the dictionary
3. If a match is found, it retrieves the corresponding stock symbol
4. If no match is found in the dictionary, it tries to use yfinance to look up the company name

This functionality is implemented in the `index` route in `app.py`.

### Data Caching

To improve performance and reduce API calls, the application implements caching for:

- Stock data
- News articles
- Company information
- Forecasting data

## API Integrations

The application uses the following APIs:

- **yfinance**: Used for fetching historical stock data and company information
- **EventRegistry**: Used for fetching news articles related to stocks

## Known Issues and Limitations

- The EventRegistry API requires a valid API key to fetch news articles
- Some stocks may not have sufficient news coverage
- Search by company name works best with well-known companies in the predefined dictionary

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
