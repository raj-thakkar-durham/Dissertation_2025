# Parallel Execution Framework for Hybrid Gold Price Prediction
# Implements parallel simulation execution as per instruction manual Phase 4.2
# Research purposes only - academic dissertation

from joblib import Parallel, delayed
import multiprocessing as mp
from mesa import batch_run
import pandas as pd
import numpy as np
from itertools import product
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

class ParallelSimulationRunner:
    """
    Parallel simulation runner for efficient execution
    As specified in instruction manual Step 4.2
    
    Reference: Instruction manual Phase 4.2 - Parallel Execution Framework
    """
    
    def __init__(self, model_class, num_cores=None):
        """
        Initialize parallel simulation runner
        
        Args:
            model_class: Model class to run
            num_cores (int): Number of CPU cores to use
            
        Reference: Instruction manual - "def __init__(self, model_class, num_cores=None):"
        """
        self.model_class = model_class
        self.num_cores = num_cores or mp.cpu_count()
        self.results_history = []
        
        print(f"ParallelSimulationRunner initialized with {self.num_cores} cores")
        
    def run_single_simulation(self, params, run_id=0):
        """
        Run a single simulation with given parameters
        
        Args:
            params (dict): Simulation parameters
            run_id (int): Run identifier
            
        Returns:
            dict: Simulation results
        """
        try:
            # Set random seed for reproducibility
            np.random.seed(run_id + params.get('seed', 42))
            
            # Create model instance
            model = self.model_class(**params['model_params'])
            
            # Run simulation
            results = model.run_simulation(
                num_days=params['num_days'],
                external_data=params.get('external_data', None)
            )
            
            # Add run metadata
            results['run_id'] = run_id
            results['parameters'] = params
            results['success'] = True
            
            return results
            
        except Exception as e:
            print(f"Error in simulation run {run_id}: {e}")
            return {
                'run_id': run_id,
                'parameters': params,
                'success': False,
                'error': str(e)
            }
    
    def run_batch_simulations(self, parameters, num_iterations):
        """
        Use Mesa's BatchRunner for parallel execution
        
        Args:
            parameters (dict): Fixed and variable parameters
            num_iterations (int): Number of iterations per parameter set
            
        Returns:
            pd.DataFrame: Batch results
            
        Reference: Instruction manual - "Use Mesa's BatchRunner for parallel execution"
        """
        try:
            print(f"Running batch simulations with {num_iterations} iterations...")
            
            # Prepare parameters for BatchRunner
            fixed_params = parameters.get('fixed_params', {})
            variable_params = parameters.get('variable_params', {})
            
            # Model reporters
            model_reporters = {
                "Final_Price": lambda m: m.current_price,
                "Total_Volume": lambda m: m.get_volume(),
                "Price_Change": lambda m: ((m.current_price - m.price_history[0]) / 
                                         m.price_history[0] if m.price_history else 0),
                "Volatility": lambda m: m.calculate_volatility(),
                "CA_Signal": lambda m: m.get_ca_signal(),
                "Num_Trades": lambda m: m.daily_trades
            }
            
            # Agent reporters
            agent_reporters = {
                "Agent_Type": "agent_type",
                "Portfolio_Value": lambda a: a.get_portfolio_value(a.model.current_price),
                "Position": "position",
                "Trades_Count": "trades_count",
                "PnL": lambda a: sum(a.pnl_history) if a.pnl_history else 0
            }
            
            # Use Mesa's batch_run for parallel execution
            results = batch_run(
                model_cls=self.model_class,
                parameters=variable_params,
                iterations=num_iterations,
                max_steps=fixed_params.get('num_days', 252),
                number_processes=self.num_cores,
                data_collection_period=1,
                display_progress=True
            )
            
            # Convert results to DataFrame
            model_data = pd.DataFrame(results)
            
            print(f"Batch simulation completed: {len(model_data)} model runs")
            
            return {
                'model_data': model_data,
                'agent_data': pd.DataFrame(),  # batch_run doesn't separate agent data
                'results': results
            }
            
        except Exception as e:
            print(f"Error in batch simulations: {e}")
            return {}
    
    def run_parameter_sweep(self, parameter_ranges, num_iterations=10):
        """
        Run simulations across parameter space
        
        Args:
            parameter_ranges (dict): Parameter ranges to sweep
            num_iterations (int): Iterations per parameter combination
            
        Returns:
            pd.DataFrame: Parameter sweep results
            
        Reference: Instruction manual - "Run simulations across parameter space"
        """
        try:
            print(f"Running parameter sweep with {num_iterations} iterations per combination...")
            
            # Generate parameter combinations
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            # Create all combinations
            combinations = list(product(*param_values))
            
            print(f"Generated {len(combinations)} parameter combinations")
            
            # Prepare simulation parameters
            all_params = []
            for i, combo in enumerate(combinations):
                for iteration in range(num_iterations):
                    params = {
                        'model_params': dict(zip(param_names, combo)),
                        'num_days': 252,
                        'seed': i * num_iterations + iteration
                    }
                    all_params.append(params)
            
            # Run simulations in parallel
            print(f"Running {len(all_params)} simulations on {self.num_cores} cores...")
            
            results = Parallel(n_jobs=self.num_cores)(
                delayed(self.run_single_simulation)(params, i) 
                for i, params in enumerate(all_params)
            )
            
            # Process results
            successful_results = [r for r in results if r.get('success', False)]
            failed_results = [r for r in results if not r.get('success', False)]
            
            print(f"Completed: {len(successful_results)} successful, {len(failed_results)} failed")
            
            # Convert to DataFrame
            if successful_results:
                sweep_data = []
                for result in successful_results:
                    row = {
                        'run_id': result['run_id'],
                        'final_price': result['final_price'],
                        'price_change': result['market_statistics']['price_change'],
                        'volatility': result['market_statistics']['price_volatility'],
                        'total_volume': result['total_volume'],
                        'average_volatility': result['average_volatility']
                    }
                    
                    # Add parameter values
                    for param_name, param_value in result['parameters']['model_params'].items():
                        row[param_name] = param_value
                    
                    sweep_data.append(row)
                
                sweep_df = pd.DataFrame(sweep_data)
                
                # Store results
                self.results_history.append({
                    'type': 'parameter_sweep',
                    'timestamp': time.time(),
                    'results': sweep_df,
                    'failed_runs': failed_results
                })
                
                return sweep_df
            
            else:
                print("No successful results from parameter sweep")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error in parameter sweep: {e}")
            return pd.DataFrame()
    
    def run_monte_carlo_simulation(self, base_params, num_runs=100):
        """
        Run Monte Carlo simulation with random parameter variations
        
        Args:
            base_params (dict): Base parameter configuration
            num_runs (int): Number of Monte Carlo runs
            
        Returns:
            pd.DataFrame: Monte Carlo results
        """
        try:
            print(f"Running Monte Carlo simulation with {num_runs} runs...")
            
            # Generate random parameter variations
            all_params = []
            for i in range(num_runs):
                params = base_params.copy()
                
                # Add random variations
                if 'random_variations' in base_params:
                    for param_name, (min_val, max_val) in base_params['random_variations'].items():
                        params['model_params'][param_name] = np.random.uniform(min_val, max_val)
                
                params['seed'] = i
                all_params.append(params)
            
            # Run simulations
            results = Parallel(n_jobs=self.num_cores)(
                delayed(self.run_single_simulation)(params, i) 
                for i, params in enumerate(all_params)
            )
            
            # Process results
            successful_results = [r for r in results if r.get('success', False)]
            
            if successful_results:
                mc_data = []
                for result in successful_results:
                    row = {
                        'run_id': result['run_id'],
                        'final_price': result['final_price'],
                        'price_change': result['market_statistics']['price_change'],
                        'volatility': result['market_statistics']['price_volatility'],
                        'total_volume': result['total_volume'],
                        'max_price': result['market_statistics']['max_price'],
                        'min_price': result['market_statistics']['min_price']
                    }
                    
                    # Add agent performance data
                    for agent_type, performance in result['agent_performances'].items():
                        row[f'{agent_type}_pnl'] = performance['total_pnl']
                        row[f'{agent_type}_win_rate'] = performance['win_rate']
                    
                    mc_data.append(row)
                
                mc_df = pd.DataFrame(mc_data)
                
                print(f"Monte Carlo simulation completed: {len(mc_df)} successful runs")
                return mc_df
            
            else:
                print("No successful Monte Carlo results")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {e}")
            return pd.DataFrame()
    
    def optimize_parameters(self, objective_function, param_bounds, max_iterations=50):
        """
        Parallel parameter optimization
        
        Args:
            objective_function (callable): Function to optimize
            param_bounds (list): Parameter bounds
            max_iterations (int): Maximum optimization iterations
            
        Returns:
            dict: Optimization results
            
        Reference: Instruction manual - "Parallel parameter optimization"
        """
        try:
            from scipy.optimize import differential_evolution
            
            print(f"Running parameter optimization with {max_iterations} iterations...")
            
            # Define objective wrapper for parallel evaluation
            def parallel_objective(params):
                # Run simulation with given parameters
                sim_params = {
                    'model_params': self._params_array_to_dict(params),
                    'num_days': 252,
                    'seed': np.random.randint(0, 10000)
                }
                
                result = self.run_single_simulation(sim_params)
                
                if result.get('success', False):
                    return objective_function(result)
                else:
                    return 1e6  # High penalty for failed simulations
            
            # Run optimization
            optimization_result = differential_evolution(
                parallel_objective,
                param_bounds,
                maxiter=max_iterations,
                popsize=15,
                workers=self.num_cores,
                seed=42
            )
            
            # Convert result to readable format
            optimized_params = self._params_array_to_dict(optimization_result.x)
            
            optimization_results = {
                'optimized_parameters': optimized_params,
                'optimization_value': optimization_result.fun,
                'num_iterations': optimization_result.nit,
                'success': optimization_result.success,
                'message': optimization_result.message
            }
            
            print(f"Optimization completed: {optimization_result.success}")
            print(f"Best value: {optimization_result.fun:.6f}")
            
            return optimization_results
            
        except Exception as e:
            print(f"Error in parameter optimization: {e}")
            return {}
    
    def _params_array_to_dict(self, params_array):
        """
        Convert parameter array to dictionary (implementation specific)
        
        Args:
            params_array (np.ndarray): Parameter array
            
        Returns:
            dict: Parameter dictionary
        """
        # This is a placeholder - should be customized based on actual parameters
        return {
            'num_agents': int(params_array[0]),
            'grid_size': (int(params_array[1]), int(params_array[1])),
            'volatility': params_array[2],
            'liquidity': params_array[3]
        }
    
    def run_sensitivity_analysis(self, base_params, sensitivity_params, num_runs=20):
        """
        Run sensitivity analysis for specific parameters
        
        Args:
            base_params (dict): Base parameter configuration
            sensitivity_params (dict): Parameters to vary for sensitivity analysis
            num_runs (int): Number of runs per parameter variation
            
        Returns:
            pd.DataFrame: Sensitivity analysis results
        """
        try:
            print(f"Running sensitivity analysis...")
            
            all_results = []
            
            for param_name, param_values in sensitivity_params.items():
                for param_value in param_values:
                    # Create parameter set
                    params = base_params.copy()
                    params['model_params'][param_name] = param_value
                    
                    # Run multiple simulations for this parameter value
                    param_results = []
                    for run in range(num_runs):
                        params['seed'] = run
                        result = self.run_single_simulation(params, run)
                        
                        if result.get('success', False):
                            param_results.append({
                                'parameter': param_name,
                                'parameter_value': param_value,
                                'run': run,
                                'final_price': result['final_price'],
                                'price_change': result['market_statistics']['price_change'],
                                'volatility': result['market_statistics']['price_volatility']
                            })
                    
                    all_results.extend(param_results)
            
            sensitivity_df = pd.DataFrame(all_results)
            
            print(f"Sensitivity analysis completed: {len(sensitivity_df)} results")
            return sensitivity_df
            
        except Exception as e:
            print(f"Error in sensitivity analysis: {e}")
            return pd.DataFrame()
    
    def save_results(self, results, filename):
        """
        Save simulation results to file
        
        Args:
            results: Results to save
            filename (str): Output filename
        """
        try:
            if isinstance(results, pd.DataFrame):
                results.to_csv(filename)
            else:
                with open(filename, 'wb') as f:
                    pickle.dump(results, f)
            
            print(f"Results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def get_performance_statistics(self):
        """
        Get performance statistics for the runner
        
        Returns:
            dict: Performance statistics
        """
        try:
            stats = {
                'num_cores': self.num_cores,
                'total_runs_completed': len(self.results_history),
                'results_history': self.results_history
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting performance statistics: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Mock model class for testing
    class MockModel:
        def __init__(self, num_agents=100, grid_size=(10, 10), volatility=0.02):
            self.num_agents = num_agents
            self.grid_size = grid_size
            self.volatility = volatility
            self.current_price = 1800
            self.price_history = [1800]
            
        def run_simulation(self, num_days, external_data=None):
            # Mock simulation
            final_price = self.current_price * (1 + np.random.normal(0, self.volatility))
            
            return {
                'final_price': final_price,
                'total_volume': np.random.uniform(1000, 10000),
                'average_volatility': self.volatility,
                'market_statistics': {
                    'price_change': (final_price - self.current_price) / self.current_price,
                    'price_volatility': self.volatility
                },
                'agent_performances': {
                    'herder': {'total_pnl': np.random.uniform(-100, 100), 'win_rate': 0.5},
                    'contrarian': {'total_pnl': np.random.uniform(-100, 100), 'win_rate': 0.5}
                }
            }
        
        def get_volume(self):
            return np.random.uniform(100, 1000)
        
        def calculate_volatility(self):
            return self.volatility
        
        def get_ca_signal(self):
            return np.random.uniform(-0.5, 0.5)
    
    # Test parallel runner
    runner = ParallelSimulationRunner(MockModel, num_cores=2)
    
    # Test parameter sweep
    param_ranges = {
        'num_agents': [50, 100, 150],
        'volatility': [0.01, 0.02, 0.03]
    }
    
    sweep_results = runner.run_parameter_sweep(param_ranges, num_iterations=2)
    
    print(f"Parameter sweep completed: {len(sweep_results)} results")
    if not sweep_results.empty:
        print(sweep_results.head())
    
    # Test Monte Carlo simulation
    base_params = {
        'model_params': {'num_agents': 100, 'grid_size': (10, 10), 'volatility': 0.02},
        'num_days': 252,
        'random_variations': {
            'volatility': (0.01, 0.03),
            'num_agents': (50, 150)
        }
    }
    
    mc_results = runner.run_monte_carlo_simulation(base_params, num_runs=5)
    
    print(f"Monte Carlo simulation completed: {len(mc_results)} results")
    if not mc_results.empty:
        print(mc_results.describe())
    
    print("Parallel runner testing completed successfully!")
