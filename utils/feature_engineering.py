# Feature Engineering Module for Hybrid Gold Price Prediction
# Implements feature engineering as per instruction manual Phase 1.4
# Research purposes only - academic dissertation

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    Feature engineering class for creating market features
    As specified in instruction manual Step 1.4
    
    Reference: Instruction manual Phase 1.4 - Feature Engineering
    """
    
    def __init__(self, data):
        """
        Initialize feature engineering with market data
        
        Args:
            data (pd.DataFrame): Market data with price information
            
        Reference: Instruction manual - "def __init__(self, data):"
        """
        self.data = data.copy()
        self.scaler = StandardScaler()
        self.original_columns = list(data.columns)
        print(f"FeatureEngineering initialized with {len(data)} observations")
        print(f"Original columns: {self.original_columns}")
        
    def calculate_returns(self):
        """
        Calculate log returns for gold and oil
        Add columns: gold_return, oil_return
        
        Reference: Instruction manual - "Calculate log returns for gold and oil"
        """
        try:
            # Calculate gold returns
            if 'Close' in self.data.columns:
                self.data['gold_return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            
            # Calculate oil returns
            if 'Oil_Close' in self.data.columns:
                self.data['oil_return'] = np.log(self.data['Oil_Close'] / self.data['Oil_Close'].shift(1))
            
            print("Successfully calculated returns")
            return self.data
            
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return self.data
    
    def calculate_moving_averages(self, windows=[5, 10, 20]):
        """
        Calculate moving averages for gold prices
        Add columns: gold_ma_5, gold_ma_10, gold_ma_20
        
        Args:
            windows (list): List of window sizes for moving averages
            
        Reference: Instruction manual - "Calculate moving averages for gold prices"
        """
        try:
            if 'Close' in self.data.columns:
                for window in windows:
                    col_name = f'gold_ma_{window}'
                    self.data[col_name] = self.data['Close'].rolling(window=window, min_periods=1).mean()
                    
                    # Calculate price relative to moving average
                    self.data[f'gold_ma_{window}_ratio'] = self.data['Close'] / self.data[col_name]
            
            print(f"Successfully calculated moving averages for windows: {windows}")
            return self.data
            
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
            return self.data
    
    def calculate_volatility(self, window=20):
        """
        Calculate rolling volatility
        Add column: gold_volatility
        
        Args:
            window (int): Rolling window size for volatility calculation
            
        Reference: Instruction manual - "Calculate rolling volatility"
        """
        try:
            if 'gold_return' in self.data.columns:
                self.data['gold_volatility'] = self.data['gold_return'].rolling(
                    window=window, min_periods=1
                ).std()
            
            print(f"Successfully calculated volatility with window: {window}")
            return self.data
            
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return self.data
    
    def create_ca_grid_features(self):
        """
        Create 3x3 grid features for CA
        Center: current gold return
        8 neighbors: past 4 days returns + 4 days oil/sentiment
        
        Reference: Instruction manual - "Create 3x3 grid features for CA"
        """
        try:
            # Create lagged features for CA grid
            if 'gold_return' in self.data.columns:
                # Past 4 days of gold returns
                for i in range(1, 5):
                    self.data[f'gold_return_lag_{i}'] = self.data['gold_return'].shift(i)
            
            # Oil return lags
            if 'oil_return' in self.data.columns:
                for i in range(1, 5):
                    self.data[f'oil_return_lag_{i}'] = self.data['oil_return'].shift(i)
            
            # Market index features
            if 'SP500_Close' in self.data.columns:
                self.data['sp500_return'] = np.log(self.data['SP500_Close'] / self.data['SP500_Close'].shift(1))
                for i in range(1, 3):
                    self.data[f'sp500_return_lag_{i}'] = self.data['sp500_return'].shift(i)
            
            if 'USD_Close' in self.data.columns:
                self.data['usd_return'] = np.log(self.data['USD_Close'] / self.data['USD_Close'].shift(1))
                for i in range(1, 3):
                    self.data[f'usd_return_lag_{i}'] = self.data['usd_return'].shift(i)
            
            # Create CA grid representation
            ca_features = []
            
            # Center cell (current gold return)
            if 'gold_return' in self.data.columns:
                ca_features.append('gold_return')
            
            # Neighbor cells (lagged features)
            for i in range(1, 5):
                if f'gold_return_lag_{i}' in self.data.columns:
                    ca_features.append(f'gold_return_lag_{i}')
                if f'oil_return_lag_{i}' in self.data.columns:
                    ca_features.append(f'oil_return_lag_{i}')
            
            # Add market context
            if 'sp500_return' in self.data.columns:
                ca_features.append('sp500_return')
            if 'usd_return' in self.data.columns:
                ca_features.append('usd_return')
            
            # Store CA feature names for later use
            self.ca_features = ca_features
            
            print(f"Successfully created CA grid features: {len(ca_features)} features")
            return self.data
            
        except Exception as e:
            print(f"Error creating CA grid features: {e}")
            return self.data
    
    def create_technical_indicators(self):
        """
        Create additional technical indicators
        """
        try:
            if 'Close' in self.data.columns:
                # RSI calculation
                delta = self.data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                self.data['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                ma_20 = self.data['Close'].rolling(window=20).mean()
                std_20 = self.data['Close'].rolling(window=20).std()
                self.data['bollinger_upper'] = ma_20 + (std_20 * 2)
                self.data['bollinger_lower'] = ma_20 - (std_20 * 2)
                self.data['bollinger_width'] = self.data['bollinger_upper'] - self.data['bollinger_lower']
                
                # Price position within Bollinger Bands
                self.data['bollinger_position'] = (self.data['Close'] - self.data['bollinger_lower']) / self.data['bollinger_width']
                
                # MACD
                exp1 = self.data['Close'].ewm(span=12).mean()
                exp2 = self.data['Close'].ewm(span=26).mean()
                self.data['macd'] = exp1 - exp2
                self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
                self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
            
            print("Successfully created technical indicators")
            return self.data
            
        except Exception as e:
            print(f"Error creating technical indicators: {e}")
            return self.data
    
    def normalize_features(self):
        """
        Normalize all features to comparable scales
        Return normalized DataFrame
        
        Reference: Instruction manual - "Normalize all features to comparable scales"
        """
        try:
            # Identify numerical columns to normalize
            numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude original price columns from normalization
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Oil_Close', 'SP500_Close', 'USD_Close']
            columns_to_normalize = [col for col in numerical_columns if col not in exclude_cols]
            
            # Create normalized data
            normalized_data = self.data.copy()
            
            if columns_to_normalize:
                # Fit scaler and transform
                normalized_values = self.scaler.fit_transform(self.data[columns_to_normalize])
                
                # Replace normalized columns
                for i, col in enumerate(columns_to_normalize):
                    normalized_data[col] = normalized_values[:, i]
            
            # Remove rows with NaN values
            normalized_data = normalized_data.dropna()
            
            print(f"Successfully normalized {len(columns_to_normalize)} features")
            print(f"Final dataset shape: {normalized_data.shape}")
            
            return normalized_data
            
        except Exception as e:
            print(f"Error normalizing features: {e}")
            return self.data
    
    def create_target_variable(self, horizon=1):
        """
        Create target variable for prediction
        
        Args:
            horizon (int): Number of days ahead to predict
            
        Returns:
            pd.Series: Target variable
        """
        try:
            if 'Close' in self.data.columns:
                # Future return as target
                future_return = self.data['Close'].shift(-horizon) / self.data['Close'] - 1
                
                # Create categorical target (up/down)
                target_binary = (future_return > 0).astype(int)
                
                # Create continuous target
                target_continuous = future_return
                
                return target_binary, target_continuous
            else:
                return None, None
                
        except Exception as e:
            print(f"Error creating target variable: {e}")
            return None, None
    
    def get_feature_summary(self):
        """
        Get summary of created features
        
        Returns:
            dict: Feature summary
        """
        try:
            summary = {
                'total_features': len(self.data.columns),
                'original_features': len(self.original_columns),
                'created_features': len(self.data.columns) - len(self.original_columns),
                'feature_names': list(self.data.columns),
                'data_shape': self.data.shape,
                'null_values': self.data.isnull().sum().sum(),
                'date_range': f"{self.data.index.min()} to {self.data.index.max()}"
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting feature summary: {e}")
            return {}
    
    def save_features(self, filename):
        """
        Save engineered features to CSV file
        
        Args:
            filename (str): Output filename
        """
        try:
            self.data.to_csv(filename)
            print(f"Features saved to {filename}")
        except Exception as e:
            print(f"Error saving features: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Close': 1800 + np.cumsum(np.random.randn(len(dates)) * 5),
        'Oil_Close': 80 + np.cumsum(np.random.randn(len(dates)) * 2),
        'SP500_Close': 4000 + np.cumsum(np.random.randn(len(dates)) * 20),
        'USD_Close': 100 + np.cumsum(np.random.randn(len(dates)) * 1)
    }, index=dates)
    
    # Initialize feature engineering
    fe = FeatureEngineering(sample_data)
    
    # Apply all feature engineering steps
    fe.calculate_returns()
    fe.calculate_moving_averages()
    fe.calculate_volatility()
    fe.create_ca_grid_features()
    fe.create_technical_indicators()
    
    # Normalize features
    normalized_data = fe.normalize_features()
    
    # Get summary
    summary = fe.get_feature_summary()
    print("\nFeature Engineering Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save features
    fe.save_features('data/engineered_features.csv')
