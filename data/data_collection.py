# Data Collection Module for Hybrid Gold Price Prediction
# Implements historical data fetching as per instruction manual Phase 1.2
# Research purposes only - academic dissertation

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all necessary modules directly
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class DataCollector:
    """
    Data collection class for fetching historical market data
    As specified in instruction manual Step 1.2
    
    Reference: Instruction manual Phase 1.2 - Historical Data Collection
    """
    
    def __init__(self, start_date, end_date):
        """
        Initialize data collector with date range
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.start_date = start_date
        self.end_date = end_date
        print(f"DataCollector initialized for period: {start_date} to {end_date}")
        
    def fetch_gold_data(self):
        """
        Fetch gold prices using GLD ETF or GC=F futures
        
        Returns:
            pd.DataFrame: DataFrame with Date, Open, High, Low, Close, Volume
            
        Reference: Instruction manual - "Fetch gold prices (GLD ETF or GC=F futures)"
        """
        try:
            # Try GLD ETF first
            gold_data = yf.download('GLD', start=self.start_date, end=self.end_date)
            if gold_data.empty:
                # Fallback to gold futures
                gold_data = yf.download('GC=F', start=self.start_date, end=self.end_date)
            
            # Reset index to make Date a column
            gold_data = gold_data.reset_index()
            
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in gold_data.columns:
                    if col == 'Date':
                        continue
                    gold_data[col] = gold_data.get('Adj Close', gold_data['Close'])
            
            print(f"Successfully fetched {len(gold_data)} days of gold data")
            return gold_data
            
        except Exception as e:
            print(f"Error fetching gold data: {e}")
            return pd.DataFrame()
    
    def fetch_oil_data(self):
        """
        Fetch Brent oil prices (BZ=F) or WTI (CL=F)
        
        Returns:
            pd.DataFrame: DataFrame with Date, Close
            
        Reference: Instruction manual - "Fetch Brent oil prices (BZ=F) or WTI (CL=F)"
        """
        try:
            # Try Brent oil first
            oil_data = yf.download('BZ=F', start=self.start_date, end=self.end_date)
            if oil_data.empty:
                # Fallback to WTI
                oil_data = yf.download('CL=F', start=self.start_date, end=self.end_date)
            
            # Reset index and select required columns
            oil_data = oil_data.reset_index()
            oil_data = oil_data[['Date', 'Close']].rename(columns={'Close': 'Oil_Close'})
            
            print(f"Successfully fetched {len(oil_data)} days of oil data")
            return oil_data
            
        except Exception as e:
            print(f"Error fetching oil data: {e}")
            return pd.DataFrame()
    
    def fetch_market_indices(self):
        """
        Fetch S&P 500 (^GSPC) and USD Index (DX-Y.NYB)
        
        Returns:
            pd.DataFrame: DataFrame with Date, Close for each index
            
        Reference: Instruction manual - "Fetch S&P 500 (^GSPC) and USD Index (DX-Y.NYB)"
        """
        try:
            # Fetch S&P 500
            sp500_data = yf.download('^GSPC', start=self.start_date, end=self.end_date)
            sp500_data = sp500_data.reset_index()
            sp500_data = sp500_data[['Date', 'Close']].rename(columns={'Close': 'SP500_Close'})
            
            # Fetch USD Index
            usd_data = yf.download('DX-Y.NYB', start=self.start_date, end=self.end_date)
            usd_data = usd_data.reset_index()
            usd_data = usd_data[['Date', 'Close']].rename(columns={'Close': 'USD_Close'})
            
            # Merge the indices
            indices_data = pd.merge(sp500_data, usd_data, on='Date', how='outer')
            
            print(f"Successfully fetched {len(indices_data)} days of market indices data")
            return indices_data
            
        except Exception as e:
            print(f"Error fetching market indices: {e}")
            return pd.DataFrame()
    
    def merge_market_data(self):
        """
        Combine all market data into single DataFrame
        Handle missing values with forward fill or interpolation
        
        Returns:
            pd.DataFrame: Merged DataFrame indexed by Date
            
        Reference: Instruction manual - "Combine all market data into single DataFrame"
        """
        try:
            # Fetch all data
            gold_data = self.fetch_gold_data()
            oil_data = self.fetch_oil_data()
            indices_data = self.fetch_market_indices()
            
            # Start with gold data as base
            merged_data = gold_data.copy()
            
            # Merge oil data
            if not oil_data.empty:
                merged_data = pd.merge(merged_data, oil_data, on='Date', how='left')
            
            # Merge indices data
            if not indices_data.empty:
                merged_data = pd.merge(merged_data, indices_data, on='Date', how='left')
            
            # Set Date as index
            merged_data.set_index('Date', inplace=True)
            
            # Handle missing values with forward fill then backward fill
            merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove any remaining NaN values
            merged_data = merged_data.dropna()
            
            print(f"Successfully merged market data: {len(merged_data)} trading days")
            print(f"Columns available: {list(merged_data.columns)}")
            
            return merged_data
            
        except Exception as e:
            print(f"Error merging market data: {e}")
            return pd.DataFrame()
    
    def save_data(self, data, filename):
        """
        Save merged data to CSV file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        try:
            data.to_csv(filename)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def collect_all_data(self, start_date=None, end_date=None):
        """
        Collect and merge all required market data for comprehensive analysis
        Based on Fama & French (1993) multi-factor model approach
        
        Args:
            start_date (str, optional): Override start date
            end_date (str, optional): Override end date
            
        Returns:
            pd.DataFrame: Comprehensive market dataset
        """
        try:
            if start_date:
                self.start_date = start_date
            if end_date:
                self.end_date = end_date
            
            print(f"Collecting comprehensive market data for {self.start_date} to {self.end_date}")
            
            gold_data = self.fetch_gold_data()
            if gold_data.empty:
                print("Failed to fetch gold data")
                return pd.DataFrame()
            
            oil_data = self.fetch_oil_data()
            if oil_data.empty:
                print("Using placeholder oil data")
                oil_data = self.create_placeholder_oil_data(gold_data.index)
            
            market_data = self.fetch_market_data()
            if market_data.empty:
                print("Using placeholder market data")
                market_data = self.create_placeholder_market_data(gold_data.index)
            
            currency_data = self.fetch_currency_data()
            if currency_data.empty:
                print("Using placeholder currency data")
                currency_data = self.create_placeholder_currency_data(gold_data.index)
            
            merged_data = self.merge_comprehensive_data(gold_data, oil_data, market_data, currency_data)
            
            print(f"Comprehensive data collection completed: {len(merged_data)} observations")
            print(f"Data columns: {list(merged_data.columns)}")
            
            return merged_data
            
        except Exception as e:
            print(f"Error in comprehensive data collection: {e}")
            return pd.DataFrame()
    
    def create_placeholder_oil_data(self, date_index):
        """
        Create placeholder oil data when real data unavailable
        
        Args:
            date_index (pd.DatetimeIndex): Date index to match
            
        Returns:
            pd.DataFrame: Placeholder oil data
        """
        try:
            np.random.seed(42)
            base_price = 70.0
            
            oil_data = pd.DataFrame(index=date_index)
            oil_data['Oil_Close'] = base_price + np.cumsum(np.random.normal(0, 2, len(date_index)))
            oil_data['Oil_Volume'] = np.random.lognormal(15, 0.5, len(date_index))
            oil_data['Oil_Return'] = oil_data['Oil_Close'].pct_change()
            oil_data['Oil_Volatility'] = oil_data['Oil_Return'].rolling(window=20).std()
            
            return oil_data
            
        except Exception as e:
            print(f"Error creating placeholder oil data: {e}")
            return pd.DataFrame()
    
    def create_placeholder_market_data(self, date_index):
        """
        Create placeholder market data when real data unavailable
        
        Args:
            date_index (pd.DatetimeIndex): Date index to match
            
        Returns:
            pd.DataFrame: Placeholder market data
        """
        try:
            np.random.seed(42)
            base_price = 3000.0
            
            market_data = pd.DataFrame(index=date_index)
            market_data['Market_Close'] = base_price + np.cumsum(np.random.normal(0, 20, len(date_index)))
            market_data['Market_Volume'] = np.random.lognormal(20, 0.3, len(date_index))
            market_data['Market_Return'] = market_data['Market_Close'].pct_change()
            market_data['Market_Volatility'] = market_data['Market_Return'].rolling(window=20).std()
            
            return market_data
            
        except Exception as e:
            print(f"Error creating placeholder market data: {e}")
            return pd.DataFrame()
    
    def create_placeholder_currency_data(self, date_index):
        """
        Create placeholder currency data when real data unavailable
        
        Args:
            date_index (pd.DatetimeIndex): Date index to match
            
        Returns:
            pd.DataFrame: Placeholder currency data
        """
        try:
            np.random.seed(42)
            base_rate = 1.25
            
            currency_data = pd.DataFrame(index=date_index)
            currency_data['USD_Index'] = base_rate + np.cumsum(np.random.normal(0, 0.01, len(date_index)))
            currency_data['USD_Return'] = currency_data['USD_Index'].pct_change()
            currency_data['USD_Volatility'] = currency_data['USD_Return'].rolling(window=20).std()
            
            return currency_data
            
        except Exception as e:
            print(f"Error creating placeholder currency data: {e}")
            return pd.DataFrame()
    
    def merge_comprehensive_data(self, gold_data, oil_data, market_data, currency_data):
        """
        Merge all market data sources with proper alignment
        
        Args:
            gold_data (pd.DataFrame): Gold price data
            oil_data (pd.DataFrame): Oil price data
            market_data (pd.DataFrame): Market index data
            currency_data (pd.DataFrame): Currency data
            
        Returns:
            pd.DataFrame: Merged market dataset
        """
        try:
            merged_data = gold_data.copy()
            
            if not oil_data.empty:
                merged_data = pd.merge(merged_data, oil_data, left_index=True, right_index=True, how='left')
            
            if not market_data.empty:
                merged_data = pd.merge(merged_data, market_data, left_index=True, right_index=True, how='left')
            
            if not currency_data.empty:
                merged_data = pd.merge(merged_data, currency_data, left_index=True, right_index=True, how='left')
            
            merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
            
            return merged_data
            
        except Exception as e:
            print(f"Error merging comprehensive data: {e}")
            return gold_data

# Example usage and testing
if __name__ == "__main__":
    # Initialize data collector
    collector = DataCollector('2020-01-01', '2024-01-01')
    
    # Fetch and merge all data
    market_data = collector.merge_market_data()
    
    # Display summary
    if not market_data.empty:
        print("\nData Summary:")
        print(market_data.describe())
        print("\nFirst few rows:")
        print(market_data.head())
        
        # Save to file
        collector.save_data(market_data, 'data/market_data.csv')
