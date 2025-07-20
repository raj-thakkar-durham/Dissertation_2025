# CA Rule Optimizer for Hybrid Gold Price Prediction
# Implements CA rule optimization as per instruction manual Phase 2.2
# Research purposes only - academic dissertation

import numpy as np
from scipy.optimize import differential_evolution, minimize
from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class CARuleOptimizer:
    """
    CA Rule Optimizer for optimizing cellular automaton rules
    As specified in instruction manual Step 2.2
    
    Reference: Instruction manual Phase 2.2 - CA Rule Optimization
    """
    
    def __init__(self, ca_model, historical_data):
        """
        Initialize CA rule optimizer
        
        Args:
            ca_model: CellularAutomaton instance
            historical_data (pd.DataFrame): Historical market data
            
        Reference: Instruction manual - "def __init__(self, ca_model, historical_data):"
        """
        self.ca_model = ca_model
        self.historical_data = historical_data
        self.best_rules = None
        self.optimization_history = []
        
        print(f"CARuleOptimizer initialized with {len(historical_data)} historical observations")
        
    def objective_function(self, rule_parameters):
        """
        Evaluate how well CA rules predict historical price movements
        Return error metric (lower is better)
        
        Args:
            rule_parameters (array): Rule parameters to evaluate
            
        Returns:
            float: Error metric (lower is better)
            
        Reference: Instruction manual - "Evaluate how well CA rules predict historical price movements"
        """
        try:
            # Convert parameter array to dictionary
            param_dict = self._array_to_params(rule_parameters)
            
            # Create a copy of CA model for testing
            test_ca = self._create_test_ca()
            test_ca.define_rules(param_dict)
            
            # Prepare data for simulation
            predictions = []
            actual_movements = []
            
            # Run simulation on historical data
            for i in range(len(self.historical_data) - 1):
                # Get current market features
                current_features = self._extract_features(i)
                
                # Initialize CA with current state
                initial_state = {
                    'sentiment': current_features.get('sentiment', 0),
                    'volatility': current_features.get('volatility', 0),
                    'trend': current_features.get('trend', 0)
                }
                test_ca.initialize_grid(initial_state)
                
                # Run CA for a few steps
                for _ in range(5):
                    test_ca.step(current_features)
                
                # Get prediction
                ca_signal = test_ca.get_market_signal()
                predictions.append(ca_signal)
                
                # Get actual movement
                if i + 1 < len(self.historical_data):
                    actual_return = self._calculate_return(i, i + 1)
                    actual_movements.append(actual_return)
            
            # Calculate error metrics
            if len(predictions) > 0 and len(actual_movements) > 0:
                # Ensure equal length
                min_len = min(len(predictions), len(actual_movements))
                predictions = predictions[:min_len]
                actual_movements = actual_movements[:min_len]
                
                # Calculate MSE
                mse = mean_squared_error(actual_movements, predictions)
                
                # Calculate directional accuracy
                pred_directions = np.sign(predictions)
                actual_directions = np.sign(actual_movements)
                directional_accuracy = accuracy_score(actual_directions, pred_directions)
                
                # Combined error (lower is better)
                error = mse + (1 - directional_accuracy)
                
                return error
            else:
                return 1.0  # High error if no valid predictions
                
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1.0  # High error on failure
    
    def _array_to_params(self, param_array):
        """
        Convert parameter array to dictionary
        
        Args:
            param_array (np.ndarray): Parameter array
            
        Returns:
            dict: Parameter dictionary
        """
        return {
            'bullish_threshold': param_array[0],
            'bearish_threshold': param_array[1],
            'momentum_weight': param_array[2],
            'contrarian_weight': param_array[3],
            'random_weight': param_array[4],
            'external_weight': param_array[5]
        }
    
    def _create_test_ca(self):
        """
        Create a test CA instance
        
        Returns:
            CellularAutomaton: Test CA instance
        """
        from .cellular_automaton import CellularAutomaton
        return CellularAutomaton(
            grid_size=self.ca_model.grid_size,
            num_states=self.ca_model.num_states
        )
    
    def _extract_features(self, index):
        """
        Extract market features at given index
        
        Args:
            index (int): Data index
            
        Returns:
            dict: Market features
        """
        try:
            row = self.historical_data.iloc[index]
            
            features = {
                'sentiment': row.get('Sentiment', 0),
                'volatility': row.get('gold_volatility', 0),
                'trend': row.get('gold_return', 0),
                'oil_return': row.get('oil_return', 0),
                'sp500_return': row.get('sp500_return', 0),
                'usd_return': row.get('usd_return', 0)
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features at index {index}: {e}")
            return {}
    
    def _calculate_return(self, start_idx, end_idx):
        """
        Calculate return between two indices
        
        Args:
            start_idx (int): Start index
            end_idx (int): End index
            
        Returns:
            float: Return value
        """
        try:
            if 'Close' in self.historical_data.columns:
                start_price = self.historical_data.iloc[start_idx]['Close']
                end_price = self.historical_data.iloc[end_idx]['Close']
                return (end_price - start_price) / start_price
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def optimize_rules(self, method='differential_evolution', n_jobs=1):
        """
        Optimize CA rule parameters to fit historical data
        Use parallel evaluation for speed
        
        Args:
            method (str): Optimization method
            n_jobs (int): Number of parallel jobs
            
        Returns:
            dict: Optimized rule parameters
            
        Reference: Instruction manual - "Optimize CA rule parameters to fit historical data"
        """
        try:
            print(f"Starting rule optimization using {method}...")
            
            # Parameter bounds
            bounds = [
                (0.1, 0.9),   # bullish_threshold
                (-0.9, -0.1), # bearish_threshold
                (0.0, 1.0),   # momentum_weight
                (0.0, 1.0),   # contrarian_weight
                (0.0, 0.5),   # random_weight
                (0.0, 1.0)    # external_weight
            ]
            
            if method == 'differential_evolution':
                # Use differential evolution
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    maxiter=50,
                    popsize=15,
                    seed=42,
                    workers=n_jobs if n_jobs > 1 else 1
                )
                
                optimized_params = self._array_to_params(result.x)
                final_error = result.fun
                
            else:
                # Use minimize with initial guess
                initial_guess = [0.3, -0.3, 0.4, 0.3, 0.2, 0.5]
                
                result = minimize(
                    self.objective_function,
                    initial_guess,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                optimized_params = self._array_to_params(result.x)
                final_error = result.fun
            
            # Store best rules
            self.best_rules = optimized_params
            
            print(f"Optimization completed. Final error: {final_error:.4f}")
            print(f"Optimized parameters: {optimized_params}")
            
            return optimized_params
            
        except Exception as e:
            print(f"Error in rule optimization: {e}")
            return {}
    
    def validate_rules(self, test_data):
        """
        Test optimized rules on out-of-sample data
        Return validation metrics
        
        Args:
            test_data (pd.DataFrame): Test data
            
        Returns:
            dict: Validation metrics
            
        Reference: Instruction manual - "Test optimized rules on out-of-sample data"
        """
        try:
            if self.best_rules is None:
                print("No optimized rules found. Run optimize_rules first.")
                return {}
            
            print("Validating optimized rules on test data...")
            
            # Create test CA
            test_ca = self._create_test_ca()
            test_ca.define_rules(self.best_rules)
            
            # Run validation
            predictions = []
            actual_movements = []
            ca_signals = []
            
            for i in range(len(test_data) - 1):
                # Extract features
                current_features = self._extract_features_from_data(test_data, i)
                
                # Initialize CA
                initial_state = {
                    'sentiment': current_features.get('sentiment', 0),
                    'volatility': current_features.get('volatility', 0),
                    'trend': current_features.get('trend', 0)
                }
                test_ca.initialize_grid(initial_state)
                
                # Run CA
                for _ in range(5):
                    test_ca.step(current_features)
                
                # Get prediction
                ca_signal = test_ca.get_market_signal()
                predictions.append(ca_signal)
                ca_signals.append(ca_signal)
                
                # Get actual movement
                if i + 1 < len(test_data):
                    actual_return = self._calculate_return_from_data(test_data, i, i + 1)
                    actual_movements.append(actual_return)
            
            # Calculate validation metrics
            if len(predictions) > 0 and len(actual_movements) > 0:
                min_len = min(len(predictions), len(actual_movements))
                predictions = predictions[:min_len]
                actual_movements = actual_movements[:min_len]
                
                # MSE
                mse = mean_squared_error(actual_movements, predictions)
                
                # RMSE
                rmse = np.sqrt(mse)
                
                # MAE
                mae = np.mean(np.abs(np.array(actual_movements) - np.array(predictions)))
                
                # Directional accuracy
                pred_directions = np.sign(predictions)
                actual_directions = np.sign(actual_movements)
                directional_accuracy = accuracy_score(actual_directions, pred_directions)
                
                # Correlation
                correlation = np.corrcoef(predictions, actual_movements)[0, 1]
                
                validation_metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'directional_accuracy': directional_accuracy,
                    'correlation': correlation,
                    'num_predictions': len(predictions),
                    'mean_ca_signal': np.mean(ca_signals),
                    'std_ca_signal': np.std(ca_signals)
                }
                
                print("Validation Results:")
                for metric, value in validation_metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                return validation_metrics
            
            else:
                print("No valid predictions for validation")
                return {}
                
        except Exception as e:
            print(f"Error in rule validation: {e}")
            return {}
    
    def _extract_features_from_data(self, data, index):
        """
        Extract features from specific data
        
        Args:
            data (pd.DataFrame): Data
            index (int): Index
            
        Returns:
            dict: Features
        """
        try:
            row = data.iloc[index]
            
            features = {
                'sentiment': row.get('Sentiment', 0),
                'volatility': row.get('gold_volatility', 0),
                'trend': row.get('gold_return', 0),
                'oil_return': row.get('oil_return', 0),
                'sp500_return': row.get('sp500_return', 0),
                'usd_return': row.get('usd_return', 0)
            }
            
            return features
            
        except Exception as e:
            return {}
    
    def _calculate_return_from_data(self, data, start_idx, end_idx):
        """
        Calculate return from specific data
        
        Args:
            data (pd.DataFrame): Data
            start_idx (int): Start index
            end_idx (int): End index
            
        Returns:
            float: Return
        """
        try:
            if 'Close' in data.columns:
                start_price = data.iloc[start_idx]['Close']
                end_price = data.iloc[end_idx]['Close']
                return (end_price - start_price) / start_price
            else:
                return 0.0
        except Exception as e:
            return 0.0
    
    def run_parameter_sweep(self, param_ranges, n_samples=100):
        """
        Run parameter sweep to explore parameter space
        
        Args:
            param_ranges (dict): Parameter ranges
            n_samples (int): Number of samples
            
        Returns:
            pd.DataFrame: Parameter sweep results
        """
        try:
            print(f"Running parameter sweep with {n_samples} samples...")
            
            # Generate parameter samples
            samples = []
            for _ in range(n_samples):
                sample = []
                for param, (min_val, max_val) in param_ranges.items():
                    sample.append(np.random.uniform(min_val, max_val))
                samples.append(sample)
            
            # Evaluate samples
            results = []
            for i, sample in enumerate(samples):
                error = self.objective_function(sample)
                param_dict = self._array_to_params(sample)
                
                result = {
                    'sample_id': i,
                    'error': error,
                    **param_dict
                }
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Evaluated {i + 1}/{n_samples} samples")
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Sort by error
            results_df = results_df.sort_values('error')
            
            print(f"Parameter sweep completed. Best error: {results_df['error'].min():.4f}")
            
            return results_df
            
        except Exception as e:
            print(f"Error in parameter sweep: {e}")
            return pd.DataFrame()
    
    def save_optimization_results(self, filename):
        """
        Save optimization results
        
        Args:
            filename (str): Output filename
        """
        try:
            results = {
                'best_rules': self.best_rules,
                'optimization_history': self.optimization_history
            }
            
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"Optimization results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving optimization results: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    from .cellular_automaton import CellularAutomaton
    
    # Initialize CA
    ca = CellularAutomaton(grid_size=(10, 10), num_states=3)
    
    # Create sample historical data
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    sample_data = pd.DataFrame({
        'Close': 1800 + np.cumsum(np.random.randn(len(dates)) * 5),
        'Sentiment': np.random.uniform(-0.5, 0.5, len(dates)),
        'gold_volatility': np.random.uniform(0.01, 0.05, len(dates)),
        'gold_return': np.random.randn(len(dates)) * 0.02,
        'oil_return': np.random.randn(len(dates)) * 0.03,
        'sp500_return': np.random.randn(len(dates)) * 0.01,
        'usd_return': np.random.randn(len(dates)) * 0.005
    }, index=dates)
    
    # Initialize optimizer
    optimizer = CARuleOptimizer(ca, sample_data)
    
    # Optimize rules
    optimized_rules = optimizer.optimize_rules(method='differential_evolution', n_jobs=1)
    
    # Validate on test data (using same data for demonstration)
    test_data = sample_data.iloc[len(sample_data)//2:]
    validation_results = optimizer.validate_rules(test_data)
    
    print("\nOptimization completed successfully!")
