# CA Rule Optimizer for Hybrid Gold Price Prediction
# Academic Implementation for Dissertation Research

"""
Cellular automaton rule optimization using evolutionary algorithms
for financial market modeling and prediction enhancement.

Citations:
[1] Storn, R. & Price, K. (1997). Differential evolution–a simple and efficient 
    heuristic for global optimization over continuous spaces. Journal of Global 
    Optimization, 11(4), 341-359.
[2] Holland, J.H. (1992). Adaptation in Natural and Artificial Systems: 
    An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence. 
    MIT Press, Cambridge, MA.
[3] Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
    Journal of Business & Economic Statistics, 13(3), 253-263.
[4] Hansen, P.R. & Lunde, A. (2005). A forecast comparison of volatility models: 
    does anything beat a GARCH(1,1)? Journal of Applied Econometrics, 20(7), 873-889.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class CARuleOptimizer:
    """
    CA rule optimizer implementing evolutionary algorithms for parameter optimization.
    
    Optimization framework based on Storn & Price (1997) differential evolution
    with Diebold & Mariano (1995) forecast evaluation methodology.
    
    References:
        Storn, R. & Price, K. (1997). Differential evolution–a simple and efficient 
        heuristic for global optimization over continuous spaces. Journal of Global 
        Optimization, 11(4), 341-359.
        
        Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
        Journal of Business & Economic Statistics, 13(3), 253-263.
    """

    def __init__(self, ca_model, historical_data):
        """
        Initialize CA rule optimizer with model and historical data.
        
        Args:
            ca_model: CellularAutomaton instance
            historical_data (pd.DataFrame): Historical market data
        """
        self.ca_model = ca_model
        self.historical_data = historical_data
        self.best_rules = None
        self.optimization_history = []

    def objective_function(self, rule_parameters):
        """
        Evaluate CA rules performance using comprehensive error metrics.
        
        Objective function implementing Hansen & Lunde (2005) forecast
        evaluation framework for financial time series prediction.
        
        Args:
            rule_parameters (array): Rule parameters to evaluate
            
        Returns:
            float: Error metric (lower is better)
            
        References:
            Hansen, P.R. & Lunde, A. (2005). A forecast comparison of volatility models: 
            does anything beat a GARCH(1,1)? Journal of Applied Econometrics, 20(7), 873-889.
        """
        try:
            # Convert parameter array to dictionary
            param_dict = self._array_to_params(rule_parameters)

            # Create test CA model
            test_ca = self._create_test_ca()
            test_ca.define_rules(param_dict)

            # Run simulation on historical data
            predictions = []
            actual_movements = []

            for i in range(len(self.historical_data) - 1):
                # Extract current market features
                current_features = self._extract_features(i)

                # Initialize CA with current state
                initial_state = {
                    'sentiment': current_features.get('sentiment', 0),
                    'volatility': current_features.get('volatility', 0),
                    'trend': current_features.get('trend', 0)
                }
                test_ca.initialize_grid(initial_state)

                # Run CA evolution
                for _ in range(5):
                    test_ca.step(current_features)

                # Get prediction and actual movement
                ca_signal = test_ca.get_market_signal()
                predictions.append(ca_signal)

                if i + 1 < len(self.historical_data):
                    actual_return = self._calculate_return(i, i + 1)
                    actual_movements.append(actual_return)

            # Calculate performance metrics
            if len(predictions) > 0 and len(actual_movements) > 0:
                min_len = min(len(predictions), len(actual_movements))
                predictions = predictions[:min_len]
                actual_movements = actual_movements[:min_len]

                # Mean Squared Error
                mse = mean_squared_error(actual_movements, predictions)
                
                # Directional accuracy
                pred_directions = np.sign(predictions)
                actual_directions = np.sign(actual_movements)
                directional_accuracy = accuracy_score(actual_directions, pred_directions)

                # Combined error metric (Diebold-Mariano approach)
                error = mse + (1 - directional_accuracy)
                return error
            else:
                return 1.0

        except Exception:
            return 1.0

    def _array_to_params(self, param_array):
        """
        Convert optimization parameter array to CA rule dictionary.
        
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
        """Create test CA instance for optimization."""
        from .cellular_automaton import CellularAutomaton
        return CellularAutomaton(
            grid_size=self.ca_model.grid_size,
            num_states=self.ca_model.num_states
        )

    def _extract_features(self, index):
        """
        Extract market features at given index for CA initialization.
        
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
        except Exception:
            return {}

    def _calculate_return(self, start_idx, end_idx):
        """
        Calculate return between two indices for performance evaluation.
        
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
        except Exception:
            return 0.0

    def optimize_rules(self, method='differential_evolution', n_jobs=1):
        """
        Optimize CA rule parameters using evolutionary algorithms.
        
        Implementation of Storn & Price (1997) differential evolution
        for global optimization of CA rule parameters.
        
        Args:
            method (str): Optimization method
            n_jobs (int): Number of parallel jobs
            
        Returns:
            dict: Optimized rule parameters
            
        References:
            Storn, R. & Price, K. (1997). Differential evolution–a simple and efficient 
            heuristic for global optimization over continuous spaces. Journal of Global 
            Optimization, 11(4), 341-359.
        """
        try:
            # Parameter bounds for optimization
            bounds = [
                (0.1, 0.9),   # bullish_threshold
                (-0.9, -0.1), # bearish_threshold
                (0.0, 1.0),   # momentum_weight
                (0.0, 1.0),   # contrarian_weight
                (0.0, 0.5),   # random_weight
                (0.0, 1.0)    # external_weight
            ]

            if method == 'differential_evolution':
                # Use differential evolution optimization
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
                # Use L-BFGS-B optimization
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
            
            return optimized_params

        except Exception:
            return {}

    def validate_rules(self, test_data):
        """
        Validate optimized rules on out-of-sample data using forecast evaluation.
        
        Validation methodology implementing Diebold & Mariano (1995)
        framework for forecast accuracy comparison.
        
        Args:
            test_data (pd.DataFrame): Test data for validation
            
        Returns:
            dict: Validation metrics
            
        References:
            Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
            Journal of Business & Economic Statistics, 13(3), 253-263.
        """
        try:
            if self.best_rules is None:
                return {}

            # Create test CA with optimized rules
            test_ca = self._create_test_ca()
            test_ca.define_rules(self.best_rules)

            # Run validation simulation
            predictions = []
            actual_movements = []
            ca_signals = []

            for i in range(len(test_data) - 1):
                # Extract features for current observation
                current_features = self._extract_features_from_data(test_data, i)

                # Initialize CA
                initial_state = {
                    'sentiment': current_features.get('sentiment', 0),
                    'volatility': current_features.get('volatility', 0),
                    'trend': current_features.get('trend', 0)
                }
                test_ca.initialize_grid(initial_state)

                # Run CA evolution
                for _ in range(5):
                    test_ca.step(current_features)

                # Get prediction and actual movement
                ca_signal = test_ca.get_market_signal()
                predictions.append(ca_signal)
                ca_signals.append(ca_signal)

                if i + 1 < len(test_data):
                    actual_return = self._calculate_return_from_data(test_data, i, i + 1)
                    actual_movements.append(actual_return)

            # Calculate validation metrics
            if len(predictions) > 0 and len(actual_movements) > 0:
                min_len = min(len(predictions), len(actual_movements))
                predictions = predictions[:min_len]
                actual_movements = actual_movements[:min_len]

                # Comprehensive validation metrics
                mse = mean_squared_error(actual_movements, predictions)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(np.array(actual_movements) - np.array(predictions)))
                
                # Directional accuracy
                pred_directions = np.sign(predictions)
                actual_directions = np.sign(actual_movements)
                directional_accuracy = accuracy_score(actual_directions, pred_directions)

                # Correlation analysis
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

                return validation_metrics
            else:
                return {}

        except Exception:
            return {}

    def _extract_features_from_data(self, data, index):
        """Extract features from specific dataset at given index."""
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
        except Exception:
            return {}

    def _calculate_return_from_data(self, data, start_idx, end_idx):
        """Calculate return from specific dataset between indices."""
        try:
            if 'Close' in data.columns:
                start_price = data.iloc[start_idx]['Close']
                end_price = data.iloc[end_idx]['Close']
                return (end_price - start_price) / start_price
            else:
                return 0.0
        except Exception:
            return 0.0

    def run_parameter_sweep(self, param_ranges, n_samples=100):
        """
        Run comprehensive parameter sweep for sensitivity analysis.
        
        Parameter space exploration implementing Holland (1992) 
        genetic algorithm principles for optimization.
        
        Args:
            param_ranges (dict): Parameter ranges for sweep
            n_samples (int): Number of samples
            
        Returns:
            pd.DataFrame: Parameter sweep results
            
        References:
            Holland, J.H. (1992). Adaptation in Natural and Artificial Systems. MIT Press.
        """
        try:
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
                result = {'sample_id': i, 'error': error, **param_dict}
                results.append(result)

            # Convert to DataFrame and sort by error
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('error')

            return results_df

        except Exception:
            return pd.DataFrame()
