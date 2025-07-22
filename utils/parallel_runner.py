# Parallel Simulation Runner for Hybrid Gold Price Prediction
# Academic Implementation for Dissertation Research

"""
Parallel execution framework for efficient simulation runs and parameter studies.

Citations:
[1] Dagum, L. & Menon, R. (1998). OpenMP: an industry standard API for 
    shared-memory programming. IEEE Computational Science and Engineering, 5(1), 46-55.
[2] Masad, D. & Kazil, J. (2015). Mesa: An agent-based modeling framework 
    in Python. Proceedings of the 14th Python in Science Conference, 53-60.
[3] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: 
    Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
"""

import multiprocessing as mp
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import time
from itertools import product
import pickle
import os

class ParallelSimulationRunner:
    """
    Parallel simulation runner for efficient parameter studies and Monte Carlo analysis.
    
    Implementation based on Masad & Kazil (2015) Mesa framework extended
    with high-performance computing principles for financial modeling.
    
    References:
        Masad, D. & Kazil, J. (2015). Mesa: An agent-based modeling framework 
        in Python. Proceedings of the 14th Python in Science Conference, 53-60.
    """

    def __init__(self, model_class, num_cores=None):
        """
        Initialize parallel simulation runner.
        
        Args:
            model_class: Model class or function to run
            num_cores (int): Number of CPU cores to use
        """
        self.model_class = model_class
        self.num_cores = num_cores or mp.cpu_count()
        self.execution_times = []
        self.results_cache = {}

    def run_single_simulation(self, params, run_id=None):
        """
        Execute single simulation with given parameters.
        
        Single simulation execution implementing academic standards
        for computational reproducibility (Peng, 2011).
        
        Args:
            params (dict): Simulation parameters
            run_id (int): Unique run identifier
            
        Returns:
            dict: Simulation results with metadata
            
        References:
            Peng, R.D. (2011). Reproducible research in computational science. 
            Science, 334(6060), 1226-1227.
        """
        try:
            start_time = time.time()
            
            # Set random seed for reproducibility
            if 'seed' in params:
                np.random.seed(params['seed'])
                
            # Execute simulation
            if callable(self.model_class):
                results = self.model_class(**params)
            else:
                model_instance = self.model_class(**params)
                results = model_instance.run()
                
            execution_time = time.time() - start_time
            
            # Add metadata
            if isinstance(results, dict):
                results['execution_metadata'] = {
                    'run_id': run_id,
                    'execution_time': execution_time,
                    'parameters': params.copy(),
                    'timestamp': time.time()
                }
            
            return results
        except Exception as e:
            return {
                'error': str(e),
                'run_id': run_id,
                'parameters': params.copy(),
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            }

    def run_batch_simulations(self, parameters_list, num_iterations=1):
        """
        Run batch simulations with parameter variations.
        
        Batch execution implementing Dagum & Menon (1998) parallel computing
        principles for scientific computing applications.
        
        Args:
            parameters_list (list): List of parameter dictionaries
            num_iterations (int): Number of iterations per parameter set
            
        Returns:
            list: List of simulation results
            
        References:
            Dagum, L. & Menon, R. (1998). OpenMP: an industry standard API for 
            shared-memory programming. IEEE Computational Science and Engineering, 5(1), 46-55.
        """
        try:
            # Prepare parameter sets with run IDs
            all_params = []
            for i, base_params in enumerate(parameters_list):
                for j in range(num_iterations):
                    params = base_params.copy()
                    params['iteration'] = j
                    params['seed'] = i * num_iterations + j
                    all_params.append((params, f"{i}_{j}"))
            
            # Parallel execution
            results = Parallel(n_jobs=self.num_cores, verbose=1)(
                delayed(self.run_single_simulation)(params, run_id) 
                for params, run_id in all_params
            )
            
            return results
        except Exception:
            return []

    def run_monte_carlo_simulation(self, base_params, num_runs=100):
        """
        Run Monte Carlo simulations for statistical robustness.
        
        Monte Carlo methodology implementing academic standards for
        financial modeling uncertainty quantification (Glasserman, 2003).
        
        Args:
            base_params (dict): Base parameter set
            num_runs (int): Number of Monte Carlo runs
            
        Returns:
            list: Monte Carlo simulation results
            
        References:
            Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. 
            Springer, New York.
        """
        try:
            # Generate parameter sets with different seeds
            param_sets = []
            for i in range(num_runs):
                params = base_params.copy()
                params['seed'] = i
                params['run_type'] = 'monte_carlo'
                param_sets.append(params)
            
            # Execute Monte Carlo runs
            results = self.run_batch_simulations(param_sets, num_iterations=1)
            
            # Calculate Monte Carlo statistics
            mc_stats = self.calculate_monte_carlo_statistics(results)
            
            return {
                'individual_results': results,
                'monte_carlo_statistics': mc_stats,
                'num_runs': num_runs,
                'base_parameters': base_params
            }
        except Exception:
            return {}

    def run_parameter_sweep(self, parameter_ranges, num_samples=None):
        """
        Run systematic parameter space exploration.
        
        Parameter sweep implementation following Pedregosa et al. (2011)
        grid search methodology for hyperparameter optimization.
        
        Args:
            parameter_ranges (dict): Parameter ranges for exploration
            num_samples (int): Number of samples per parameter
            
        Returns:
            pd.DataFrame: Parameter sweep results
            
        References:
            Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: 
            Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
        """
        try:
            # Generate parameter combinations
            param_combinations = []
            
            if num_samples:
                # Random sampling approach
                for _ in range(num_samples):
                    params = {}
                    for param_name, (min_val, max_val) in parameter_ranges.items():
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[param_name] = np.random.randint(min_val, max_val + 1)
                        else:
                            params[param_name] = np.random.uniform(min_val, max_val)
                    param_combinations.append(params)
            else:
                # Grid search approach
                param_names = list(parameter_ranges.keys())
                param_values = []
                
                for param_name, (min_val, max_val) in parameter_ranges.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        values = list(range(min_val, max_val + 1))
                    else:
                        values = np.linspace(min_val, max_val, 5)
                    param_values.append(values)
                
                # Generate all combinations
                for combination in product(*param_values):
                    params = dict(zip(param_names, combination))
                    param_combinations.append(params)
            
            # Run parameter sweep
            results = self.run_batch_simulations(param_combinations)
            
            # Convert to DataFrame for analysis
            sweep_results = []
            for result in results:
                if 'error' not in result:
                    result_row = result.get('execution_metadata', {}).get('parameters', {})
                    
                    # Extract key metrics
                    if 'final_price' in result:
                        result_row['final_price'] = result['final_price']
                    if 'price_series' in result:
                        prices = result['price_series']
                        result_row['price_change'] = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0
                        result_row['volatility'] = np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 0
                    
                    sweep_results.append(result_row)
            
            return pd.DataFrame(sweep_results)
        except Exception:
            return pd.DataFrame()

    def calculate_monte_carlo_statistics(self, results):
        """
        Calculate comprehensive Monte Carlo statistics.
        
        Statistical analysis implementing academic standards for
        Monte Carlo result interpretation (Robert & Casella, 2004).
        
        Args:
            results (list): Monte Carlo simulation results
            
        Returns:
            dict: Monte Carlo statistics
            
        References:
            Robert, C. & Casella, G. (2004). Monte Carlo Statistical Methods. 
            Springer, New York.
        """
        try:
            # Extract key metrics from results
            final_prices = []
            price_changes = []
            volatilities = []
            execution_times = []
            
            for result in results:
                if 'error' not in result:
                    if 'final_price' in result:
                        final_prices.append(result['final_price'])
                    
                    if 'price_series' in result:
                        prices = result['price_series']
                        if len(prices) > 1:
                            change = (prices[-1] - prices[0]) / prices[0]
                            price_changes.append(change)
                            
                            returns = np.diff(prices) / prices[:-1]
                            volatilities.append(np.std(returns))
                    
                    if 'execution_metadata' in result:
                        execution_times.append(result['execution_metadata']['execution_time'])
            
            # Calculate statistics
            stats = {
                'num_successful_runs': len([r for r in results if 'error' not in r]),
                'num_failed_runs': len([r for r in results if 'error' in r]),
                'success_rate': len([r for r in results if 'error' not in r]) / len(results) if results else 0
            }
            
            # Price statistics
            if final_prices:
                stats['price_statistics'] = {
                    'mean': np.mean(final_prices),
                    'std': np.std(final_prices),
                    'min': np.min(final_prices),
                    'max': np.max(final_prices),
                    'percentile_5': np.percentile(final_prices, 5),
                    'percentile_95': np.percentile(final_prices, 95)
                }
            
            # Price change statistics
            if price_changes:
                stats['return_statistics'] = {
                    'mean': np.mean(price_changes),
                    'std': np.std(price_changes),
                    'skewness': self._calculate_skewness(price_changes),
                    'kurtosis': self._calculate_kurtosis(price_changes),
                    'positive_returns_ratio': np.mean(np.array(price_changes) > 0)
                }
            
            # Volatility statistics
            if volatilities:
                stats['volatility_statistics'] = {
                    'mean': np.mean(volatilities),
                    'std': np.std(volatilities),
                    'min': np.min(volatilities),
                    'max': np.max(volatilities)
                }
            
            # Performance statistics
            if execution_times:
                stats['performance_statistics'] = {
                    'mean_execution_time': np.mean(execution_times),
                    'total_execution_time': np.sum(execution_times),
                    'min_execution_time': np.min(execution_times),
                    'max_execution_time': np.max(execution_times)
                }
            
            return stats
        except Exception:
            return {}

    def _calculate_skewness(self, data):
        """Calculate skewness statistic."""
        try:
            from scipy import stats
            return stats.skew(data)
        except:
            return 0.0

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis statistic."""
        try:
            from scipy import stats
            return stats.kurtosis(data)
        except:
            return 0.0

    def save_results(self, results, filename):
        """
        Save simulation results to file.
        
        Args:
            results: Results to save
            filename (str): Output filename
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        except Exception:
            pass

    def load_results(self, filename):
        """
        Load simulation results from file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            Results object or None
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def optimize_parameters(self, objective_function, parameter_ranges, method='random', num_evaluations=100):
        """
        Optimize simulation parameters using parallel evaluation.
        
        Parameter optimization implementing academic methodology for
        simulation-based optimization (Fu, 2002).
        
        Args:
            objective_function (callable): Function to optimize
            parameter_ranges (dict): Parameter search space
            method (str): Optimization method
            num_evaluations (int): Number of evaluations
            
        Returns:
            dict: Optimization results
            
        References:
            Fu, M.C. (2002). Optimization for simulation: Theory vs. practice. 
            INFORMS Journal on Computing, 14(3), 192-215.
        """
        try:
            # Generate parameter samples
            param_samples = []
            for _ in range(num_evaluations):
                params = {}
                for param_name, (min_val, max_val) in parameter_ranges.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
                param_samples.append(params)
            
            # Parallel evaluation
            results = Parallel(n_jobs=self.num_cores)(
                delayed(objective_function)(params) for params in param_samples
            )
            
            # Find best parameters
            best_idx = np.argmin(results) if results else 0
            best_params = param_samples[best_idx] if param_samples else {}
            best_value = results[best_idx] if results else float('inf')
            
            optimization_results = {
                'best_parameters': best_params,
                'best_value': best_value,
                'all_evaluations': list(zip(param_samples, results)),
                'num_evaluations': len(results),
                'optimization_method': method
            }
            
            return optimization_results
        except Exception:
            return {}

    def run_sensitivity_analysis(self, base_params, sensitivity_params, perturbation_range=0.1):
        """
        Run sensitivity analysis for model parameters.
        
        Sensitivity analysis implementation following academic methodology
        for model robustness testing (Saltelli et al., 2008).
        
        Args:
            base_params (dict): Base parameter set
            sensitivity_params (list): Parameters to analyze
            perturbation_range (float): Perturbation range for parameters
            
        Returns:
            pd.DataFrame: Sensitivity analysis results
            
        References:
            Saltelli, A., Ratto, M., Andres, T., et al. (2008). Global Sensitivity 
            Analysis: The Primer. John Wiley & Sons.
        """
        try:
            sensitivity_results = []
            
            for param_name in sensitivity_params:
                if param_name not in base_params:
                    continue
                    
                base_value = base_params[param_name]
                
                # Generate perturbations
                if isinstance(base_value, (int, float)):
                    perturbations = [
                        base_value * (1 - perturbation_range),
                        base_value,
                        base_value * (1 + perturbation_range)
                    ]
                else:
                    continue
                
                # Run simulations with perturbations
                for perturbation in perturbations:
                    params = base_params.copy()
                    params[param_name] = perturbation
                    
                    result = self.run_single_simulation(params)
                    
                    if 'error' not in result:
                        sensitivity_results.append({
                            'parameter': param_name,
                            'parameter_value': perturbation,
                            'parameter_change': (perturbation - base_value) / base_value if base_value != 0 else 0,
                            'final_price': result.get('final_price', 0),
                            'price_change': 0  # Calculate based on result
                        })
                        
                        # Calculate price change if price series available
                        if 'price_series' in result and len(result['price_series']) > 1:
                            prices = result['price_series']
                            price_change = (prices[-1] - prices[0]) / prices[0]
                            sensitivity_results[-1]['price_change'] = price_change
            
            return pd.DataFrame(sensitivity_results)
        except Exception:
            return pd.DataFrame()

# Example usage and testing
if __name__ == "__main__":
    # Mock simulation function for testing
    def mock_simulation(num_agents=100, volatility=0.02, seed=42):
        np.random.seed(seed)
        prices = 1800 + np.cumsum(np.random.randn(252) * volatility * 1800)
        return {
            'price_series': prices.tolist(),
            'final_price': prices[-1],
            'num_agents': num_agents,
            'volatility': volatility
        }
    
    # Initialize parallel runner
    runner = ParallelSimulationRunner(mock_simulation, num_cores=2)
    
    # Test Monte Carlo simulation
    base_params = {'num_agents': 100, 'volatility': 0.02}
    mc_results = runner.run_monte_carlo_simulation(base_params, num_runs=10)
    
    # Test parameter sweep
    param_ranges = {
        'num_agents': (50, 150),
        'volatility': (0.01, 0.05)
    }
    sweep_results = runner.run_parameter_sweep(param_ranges, num_samples=20)
