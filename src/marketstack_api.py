import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class MarketstackAPI:
    """
    Client for Marketstack API v2 to fetch end-of-day stock data.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('MARKETSTACK_API_KEY')
        if not self.api_key:
            raise ValueError("MARKETSTACK_API_KEY not set")
        self.base_url = "https://api.marketstack.com/v2"
        logger.info("MarketstackAPI initialized")

    def fetch_eod_data(self, symbol: str, limit: int = 100, date_from: str = None, date_to: str = None) -> pd.DataFrame:
        """
        Fetch end-of-day (EOD) price data for a given symbol.
        Returns a DataFrame indexed by date with columns: Open, High, Low, Close, Volume.
        
        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of results (up to 1000)
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
        """
        logger.info(f"Fetching EOD data for {symbol} (limit={limit})")
        url = f"{self.base_url}/eod"
        params = {
            "access_key": self.api_key, 
            "symbols": symbol,
            "limit": min(limit, 1000)
        }
        
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
            
        logger.debug(f"API request to {url} with params: {params}")
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if 'error' in data:
                logger.error(f"Marketstack API error: {data['error']}")
                raise ValueError(f"Marketstack API error: {data['error']}")
                
            records = data.get('data', [])
            if not records:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = [col.capitalize() for col in df.columns]
            df.sort_index(inplace=True)
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching EOD data for {symbol}: {e}")
            raise ValueError(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching EOD data for {symbol}: {e}")
            raise ValueError(f"Error processing data: {str(e)}")

    def fetch_intraday_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch latest intraday (real-time) data for a given symbol.
        Returns a DataFrame indexed by datetime with Open, High, Low, Close, Volume columns.
        """
        logger.info(f"Fetching intraday data for {symbol}")
        url = f"{self.base_url}/intraday/latest"
        params = {"access_key": self.api_key, "symbols": symbol, "limit": min(limit, 1000)}
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if 'error' in data:
                logger.error(f"Marketstack API intraday error: {data['error']}")
                raise ValueError(f"Marketstack API intraday error: {data['error']}")
                
            records = data.get('data', [])
            if not records:
                logger.warning(f"No intraday data returned for {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = [col.capitalize() for col in df.columns]
            df.sort_index(inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} intraday records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            raise ValueError(f"Error processing intraday data: {str(e)}")

    def fetch_daily_data(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """
        Fetch daily data for a symbol, compatible with AlphaVantage API replacement.
        
        Args:
            symbol: Stock ticker symbol
            outputsize: 'full' for up to 10 years of data, 'compact' for 100 days
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching daily data for {symbol} (outputsize={outputsize})")
        limit = 1000 if outputsize == 'full' else 100
        
        # Calculate date range for full data (up to 10 years)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = None
        if outputsize == 'full':
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')  # ~10 years
        
        return self.fetch_eod_data(symbol, limit=limit, date_from=start_date, date_to=end_date)

    def fetch_full_historical_data(self, symbol: str, years: int = 10) -> pd.DataFrame:
        """
        Fetch up to 10 years of historical data for a symbol using multiple API calls if needed.
        
        Args:
            symbol: Stock ticker symbol
            years: Number of years of data to fetch (max 10)
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {years} years of historical data for {symbol}")
        years = min(years, 10)  # Limit to 10 years maximum
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # Format dates for API
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Initialize with first batch (most recent)
        df = self.fetch_eod_data(symbol, limit=1000, date_from=start_date_str, date_to=end_date_str)
        
        if df.empty:
            logger.warning(f"No historical data found for {symbol}")
            return df
            
        # If we need more data, make additional calls with pagination
        all_data = [df]
        while len(df) == 1000 and len(all_data) < 10:  # Limit to 10 API calls
            # Get the oldest date from the last batch
            oldest_date = df.index.min()
            
            # New date range is from start date to the day before the oldest date
            new_end_date = (oldest_date - timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.info(f"Fetching additional data for {symbol} before {new_end_date}")
            df = self.fetch_eod_data(symbol, limit=1000, date_from=start_date_str, date_to=new_end_date)
            
            if df.empty:
                break
                
            all_data.append(df)
            
        # Combine all dataframes
        full_df = pd.concat(all_data)
        
        # Remove duplicates
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        
        # Sort by date
        full_df.sort_index(inplace=True)
        
        logger.info(f"Successfully fetched {len(full_df)} historical records for {symbol}")
        return full_df

    def get_complete_stock_data(self, symbol: str) -> dict:
        """
        Fetch EOD data and compute technical indicators.
        Returns a dict with keys: historical_data, enriched_data, fundamental_data, data_range.
        """
        df = self.fetch_full_historical_data(symbol)
        if df.empty:
            return {'error': f"No data returned for symbol {symbol}"}
            
        # Calculate technical indicators
        enriched = df.copy()
        try:
            import pandas_ta as ta
            
            # Log DataFrame structure for debugging
            logger.info(f"Computing technical indicators for {symbol} dataframe with shape {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame index type: {type(df.index).__name__}")
            
            # Make sure we're working with a proper DataFrame with the right columns
            if 'Close' not in enriched.columns:
                logger.error(f"Missing 'Close' column in {symbol} data. Available columns: {enriched.columns.tolist()}")
                # Try to identify potential close column and rename it
                close_cols = [col for col in enriched.columns if 'close' in col.lower()]
                if close_cols:
                    logger.info(f"Using {close_cols[0]} as Close column")
                    enriched = enriched.rename(columns={close_cols[0]: 'Close'})
                else:
                    logger.error(f"No suitable price column found for technical indicators")
                    return {'error': f"Cannot compute technical indicators: missing price data for {symbol}"}
            
            # Price-based indicators
            # Use pandas functions directly instead of ta accessor to avoid Series issues
            enriched['SMA_20'] = enriched['Close'].rolling(window=20).mean()
            enriched['SMA_50'] = enriched['Close'].rolling(window=50).mean()
            enriched['SMA_200'] = enriched['Close'].rolling(window=200).mean()
            enriched['EMA_20'] = enriched['Close'].ewm(span=20, adjust=False).mean()
            
            # Use ta library function calls instead of the accessor method
            # These functions take DataFrame columns as inputs
            try:
                rsi = ta.rsi(enriched['Close'], length=14)
                enriched['RSI_14'] = rsi
            except Exception as e:
                logger.error(f"Error calculating RSI for {symbol}: {e}")
                enriched['RSI_14'] = None
            
            # MACD - use the function directly
            try:
                macd_result = ta.macd(enriched['Close'])
                if isinstance(macd_result, pd.DataFrame):
                    for col in macd_result.columns:
                        enriched[col] = macd_result[col]
                else:
                    logger.warning(f"MACD calculation returned unexpected type: {type(macd_result)}")
            except Exception as e:
                logger.error(f"Error calculating MACD for {symbol}: {e}")
            
            # Bollinger Bands - use the function directly
            try:
                bbands_result = ta.bbands(enriched['Close'])
                if isinstance(bbands_result, pd.DataFrame):
                    # Check for columns using pattern matching to handle different versions
                    bbu_col = next((col for col in bbands_result.columns if col.startswith('BBU_')), None)
                    bbm_col = next((col for col in bbands_result.columns if col.startswith('BBM_')), None)
                    bbl_col = next((col for col in bbands_result.columns if col.startswith('BBL_')), None)
                    
                    if bbu_col and bbm_col and bbl_col:
                        # Use detected column names
                        enriched['BB_Upper'] = bbands_result[bbu_col]
                        enriched['BB_Middle'] = bbands_result[bbm_col]
                        enriched['BB_Lower'] = bbands_result[bbl_col]
                        logger.info(f"Mapped Bollinger Bands columns: {bbu_col}, {bbm_col}, {bbl_col}")
                    else:
                        # If columns not found in the expected format, copy all columns
                        for col in bbands_result.columns:
                            enriched[col] = bbands_result[col]
                        logger.info(f"Copied all Bollinger Bands columns: {bbands_result.columns.tolist()}")
                else:
                    logger.warning(f"BBands calculation returned unexpected type: {type(bbands_result)}")
                    # Manual calculation as fallback
                    sma = enriched['Close'].rolling(window=20).mean()
                    std = enriched['Close'].rolling(window=20).std()
                    enriched['BB_Upper'] = sma + (std * 2)
                    enriched['BB_Middle'] = sma
                    enriched['BB_Lower'] = sma - (std * 2)
                    logger.info("Used manual calculation for Bollinger Bands")
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
                # Manual calculation as fallback
                try:
                    sma = enriched['Close'].rolling(window=20).mean()
                    std = enriched['Close'].rolling(window=20).std()
                    enriched['BB_Upper'] = sma + (std * 2)
                    enriched['BB_Middle'] = sma
                    enriched['BB_Lower'] = sma - (std * 2)
                    logger.info("Used manual fallback calculation for Bollinger Bands after error")
                except Exception as e2:
                    logger.error(f"Fallback Bollinger Bands calculation also failed: {e2}")
            
            # Volatility
            enriched['Daily_Return'] = enriched['Close'].pct_change()
            enriched['Volatility_20d'] = enriched['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            # Log successful indicators calculation
            logger.info(f"Successfully calculated technical indicators for {symbol}")
            logger.info(f"Final indicators columns: {enriched.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
            enriched = pd.DataFrame()
            
        data_range = (df.index.min(), df.index.max())
        
        # Build result dict
        return {
            'historical_data': df,
            'enriched_data': enriched,
            'fundamental_data': {
                'overview': {
                    'Name': symbol,
                    'Symbol': symbol,
                    'Exchange': 'Unknown',
                    'Industry': 'Unknown'
                }
            },
            'data_range': [
                data_range[0].strftime('%Y-%m-%d') if data_range[0] else None,
                data_range[1].strftime('%Y-%m-%d') if data_range[1] else None
            ]
        }
        
    def get_stock_history(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Drop-in replacement for yfinance Ticker.history() method.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (only 1d supported by Marketstack EOD)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Getting stock history for {symbol} (period={period}, interval={interval})")
        
        if interval != '1d':
            logger.warning(f"Interval {interval} not supported by Marketstack EOD API. Using daily data.")
        
        # Convert period to date_from
        end_date = datetime.now()
        
        if period == '1d':
            start_date = end_date - timedelta(days=1)
        elif period == '5d':
            start_date = end_date - timedelta(days=5)
        elif period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif period == '3mo':
            start_date = end_date - timedelta(days=90)
        elif period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '2y':
            start_date = end_date - timedelta(days=2*365)
        elif period == '5y':
            start_date = end_date - timedelta(days=5*365)
        elif period == '10y':
            start_date = end_date - timedelta(days=10*365)
        elif period == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        elif period == 'max':
            start_date = end_date - timedelta(days=10*365)  # Maximum 10 years
        else:
            logger.warning(f"Unknown period {period}. Using 1 year.")
            start_date = end_date - timedelta(days=365)
        
        # Format dates for API
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Fetch data with appropriate limit
        days_diff = (end_date - start_date).days
        limit = min(days_diff + 10, 1000)  # Add buffer for weekends
        
        df = self.fetch_eod_data(symbol, limit=limit, date_from=start_date_str, date_to=end_date_str)
        return df
