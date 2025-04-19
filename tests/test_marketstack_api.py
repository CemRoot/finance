import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from src.marketstack_api import MarketstackAPI


class TestMarketstackAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env
        load_dotenv()
        api_key = os.getenv('MARKETSTACK_API_KEY')
        if not api_key:
            raise unittest.SkipTest("MARKETSTACK_API_KEY not set in environment")
        cls.api = MarketstackAPI(api_key=api_key)

    def test_fetch_eod_data(self):
        """Test fetching end-of-day data for AAPL."""
        df = self.api.fetch_eod_data('AAPL')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "EOD DataFrame is empty")
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            self.assertIn(col, df.columns, f"Column {col} missing in EOD data")

    def test_get_complete_stock_data(self):
        """Test getting complete stock data structure for AAPL."""
        result = self.api.get_complete_stock_data('AAPL')
        self.assertIsInstance(result, dict)
        self.assertIn('historical_data', result)
        self.assertIn('enriched_data', result)
        self.assertIn('data_range', result)
        hist = result['historical_data']
        self.assertIsInstance(hist, pd.DataFrame)
        self.assertFalse(hist.empty, "Historical data is empty")


if __name__ == '__main__':
    unittest.main()
