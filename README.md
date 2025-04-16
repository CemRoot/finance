```markdown
# Financial Analysis Dashboard

## Overview

This web application provides a dashboard for financial analysis, forecasting, and news aggregation for specific stock symbols. It allows users to:

*   View historical stock price data (OHLCV) with interactive charts (Candlestick and Line).
*   Analyze recent news articles related to a stock symbol.
*   View sentiment analysis scores derived from news headlines and descriptions using FinBERT.
*   Receive a simple trading decision suggestion (Buy/Sell/Hold) based on news sentiment and/or technical indicators.
*   View short-term price forecasts generated using the Prophet library.
*   Examine basic technical indicators (SMA, RSI, MACD, Bollinger Bands, Volatility).

## Features

*   **Stock Data Visualization:** Interactive Candlestick and Line charts using Chart.js. Displays Open, High, Low, Close, and Volume.
*   **News Aggregation:** Fetches latest news articles relevant to the selected stock symbol using the EventRegistry API.
*   **Sentiment Analysis:** Analyzes news sentiment using the `ProsusAI/finbert` model via the Hugging Face `transformers` library. Calculates an average sentiment score.
*   **Trading Decision Support:** Provides a 'Buy', 'Sell', or 'Hold' suggestion based on configurable logic (currently combines news sentiment and technical indicators).
*   **Time Series Forecasting:** Uses Facebook Prophet to forecast future stock prices (next 30 days).
*   **Technical Indicators:** Calculates and displays SMA, RSI, MACD, Bollinger Bands, and Annualized Volatility using `pandas-ta`.
*   **Caching:** Uses Flask-Caching to cache results from external APIs (yfinance, EventRegistry, Prophet forecast) to improve performance and reduce API calls.
*   **Responsive Design:** Basic responsiveness using Bootstrap 5 for usability on different screen sizes.

## Project Structure

```
├── app.py                    # Main Flask application logic, routes
├── config.py                 # Configuration (API keys, constants)
├── requirements.txt          # Python package dependencies
├── README.md                 # This file
├── .env.example              # Example environment variables file
├── src/                      # Source code modules directory
│   ├── __init__.py           # Makes 'src' a Python package
│   ├── decision.py           # Logic for generating Buy/Sell/Hold decision
│   ├── forecasting.py        # Prophet forecasting logic
│   ├── newsapi.py            # Functions for fetching stock data (yfinance) and news (EventRegistry)
│   └── sentiment.py          # FinBERT sentiment analysis logic
├── static/                   # Static files (CSS, JavaScript, Images)
│   ├── css/
│   │   └── style.css         # Custom stylesheets
│   └── js/
│       ├── forecast.js       # JS for forecast tab (Plotly charts, AJAX)
│       ├── main.js           # General JS (sidebar interaction, helpers)
│       ├── script.js         # JS for overview tab (Chart.js charts, AJAX refresh)
│       └── stock-data-loader.js # Helper JS for initializing data from Flask
└── templates/                # HTML templates (Jinja2)
    ├── base.html             # Base HTML structure, includes CSS/JS libraries
    ├── index.html            # Main page, contains welcome screen or dashboard content
    ├── partials/             # Reusable template snippets
    │   ├── _dashboard_header.html # Header shown when stock is selected
    │   ├── _flash_messages.html # Renders flashed messages
    │   ├── _sidebar.html       # Left sidebar navigation and search
    │   └── _welcome.html     # Content shown when no stock is selected
    └── tabs/                 # Templates for individual dashboard tabs
        ├── details.html      # Placeholder (content moved to overview)
        ├── forecast.html     # Structure for forecast tab
        ├── news.html         # Structure for news tab
        └── overview.html     # Structure for overview tab (charts, indicators)

```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd Finance
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `torch` and `cmdstanpy` might take some time.*

4.  **Configure API Keys (Optional but Recommended):**
    *   Rename `.env.example` to `.env`.
    *   Edit the `.env` file and add your EventRegistry API key:
        ```dotenv
        EVENTREGISTRY_API_KEY=your_actual_api_key_here
        # FLASK_SECRET_KEY=a_strong_random_secret_key # Optional: Set a secret key
        ```
    *   The application will read this key using `python-dotenv`. If not set, it falls back to the default in `config.py`.

5.  **Run the Application:**
    ```bash
    python app.py --port 5001 # Or any port you prefer
    ```
    *   The app will be accessible at `http://127.0.0.1:5001` (or the host/port you specify).
    *   Use `--host 0.0.0.0` to make it accessible on your local network.

## Usage

*   Open the application URL in your web browser.
*   Use the search bar in the sidebar to enter a stock symbol (e.g., `AAPL`, `GOOGL`, `MSFT`, `TSLA`, `THYAO.IS`).
*   Alternatively, click on one of the popular symbols on the welcome screen.
*   Navigate through the tabs (Overview, News, Forecast) to view different analyses.
*   Use the refresh buttons within the charts/sections to update data without a full page reload.

## Future Enhancements (Ideas)

*   More advanced technical indicators and charting features.
*   User accounts and watchlists.
*   Backtesting integration for decision strategies.
*   More sophisticated forecasting models (e.g., LSTM, ARIMA).
*   Improved error handling and user feedback.
*   Internationalization/Localization support.
*   Dockerization for easier deployment.

```

**Summary of Changes/Verification in `README.md`:**

1.  **Overview/Features:** Updated to accurately reflect the current capabilities, including the specific libraries used (Chart.js, Plotly, Prophet, FinBERT, pandas-ta), the type of analysis performed (sentiment, technical indicators, forecasting), and features like caching. Explicitly mentions the combined decision logic.
2.  **Project Structure:** Correctly lists the key files and directories based on the provided codebase.
3.  **Setup Instructions:** Provides clear, step-by-step instructions for cloning, setting up a virtual environment, installing requirements, configuring the optional `.env` file for the API key, and running the application. Includes platform-specific commands. Added a note about potentially long install times for ML libraries.
4.  **Usage:** Briefly explains how to interact with the application.
5.  **Future Enhancements:** Added a section with ideas for potential improvements.
6.  **Formatting:** Uses Markdown for clear headings, lists, and code blocks.
