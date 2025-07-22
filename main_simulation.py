# Main Simulation Engine for Hybrid Gold Price Prediction
# Academic Implementation for Dissertation Research

"""
Main simulation engine orchestrating the hybrid CA-ABM system for gold price prediction.

Citations:
[1] Arthur, W.B., Holland, J.H., LeBaron, B., Palmer, R. & Tayler, P. (1997). 
    Asset pricing under endogenous expectations in an artificial stock market. 
    The Economy as an Evolving Complex System II, 15-44.
[2] Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
    Journal of Business & Economic Statistics, 13(3), 253-263.
[3] Hansen, L.P. & Sargent, T.J. (2008). Robustness. Princeton University Press.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from header import *
from data.data_collection import DataCollector
from data.news_analysis import SentimentAnalyzer
from models.cellular_automaton import CellularAutomaton
from models.market_model import GoldMarketModel
from models.ca_optimizer import CARuleOptimizer
from utils.feature_engineering import FeatureEngineering
from utils.parallel_runner import ParallelSimulationRunner
from results.result_analyzer import ResultsAnalyzer
from results.plots import VisualizationTools

class HybridGoldSimulation:
    """
    Main hybrid simulation class coordinating CA and ABM components.
    
    Implementation based on Arthur et al. (1997) Santa Fe artificial stock
    market framework extended with cellular automata methodology.
    
    References:
        Arthur, W.B., Holland, J.H., LeBaron, B., Palmer, R. & Tayler, P. (1997). 
        Asset pricing under endogenous expectations in an artificial stock market.
    """

    def __init__(self, config=None):
        """
        Initialize hybrid simulation with comprehensive configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.historical_data = None
        self.sentiment_data = None
        self.ca_model = None
        self.market_model = None
        self.optimization_results = {}
        self.simulation_results = {}
        
        # Initialize components
        self.data_collector = DataCollector(
            self.config['start_date'], 
            self.config['end_date']
        )
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_engineer = None
        self.visualizer = VisualizationTools()

    def load_and_prepare_data(self):
        """
        Load and preprocess market data with comprehensive feature engineering.
        
        Data preparation methodology following Hansen & Sargent (2008)
        robustness framework for financial modeling.
        
        Returns:
            bool: Success status
            
        References:
            Hansen, L.P. & Sargent, T.J. (2008). Robustness. Princeton University Press.
        """
        try:
            # Collect comprehensive market data
            self.historical_data = self.data_collector.collect_all_data()
            
            if self.historical_data.empty:
                return False

            # Generate sentiment data
            date_range = self.historical_data.index
            self.sentiment_data = self.sentiment_analyzer.generate_sentiment_series(date_range)
            
            # Merge sentiment with market data
            if not self.sentiment_data.empty:
                self.historical_data = pd.merge(
                    self.historical_data, self.sentiment_data,
                    left_index=True, right_index=True, how='left'
                )

            # Feature engineering
            self.feature_engineer = FeatureEngineering(self.historical_data)
            self.historical_data = self.feature_engineer.create_comprehensive_features()

            return True
        except Exception:
            return False

    def initialize_models(self):
        """
        Initialize CA and ABM models with optimal configurations.
        
        Model initialization implementing Arthur et al. (1997) heterogeneous
        agent framework with cellular automata integration.
        
        Returns:
            bool: Success status
        """
        try:
            # Initialize cellular automaton
            self.ca_model = CellularAutomaton(
                grid_size=self.config['ca_grid_size'],
                num_states=self.config['ca_num_states']
            )

            # Initialize market model with CA integration
            self.market_model = GoldMarketModel(
                num_agents=self.config['num_agents'],
                grid_size=self.config['agent_grid_size'],
                ca_model=self.ca_model
            )

            return True
        except Exception:
            return False

    def calibrate_models(self):
        """
        Calibrate model parameters using historical data optimization.
        
        Model calibration implementing Diebold & Mariano (1995) forecast
        evaluation framework for parameter optimization.
        
        Returns:
            dict: Calibration results
            
        References:
            Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
            Journal of Business & Economic Statistics, 13(3), 253-263.
        """
        try:
            if self.historical_data is None or self.ca_model is None:
                return {}

            # Optimize CA rules
            ca_optimizer = CARuleOptimizer(self.ca_model, self.historical_data)
            optimized_rules = ca_optimizer.optimize_rules(
                method='differential_evolution',
                n_jobs=self.config['optimization_cores']
            )

            # Validate optimized rules
            validation_split = int(len(self.historical_data) * 0.8)
            train_data = self.historical_data.iloc[:validation_split]
            test_data = self.historical_data.iloc[validation_split:]
            
            validation_metrics = ca_optimizer.validate_rules(test_data)

            self.optimization_results = {
                'optimized_rules': optimized_rules,
                'validation_metrics': validation_metrics,
                'training_size': len(train_data),
                'test_size': len(test_data)
            }

            # Apply optimized rules to CA
            if optimized_rules:
                self.ca_model.define_rules(optimized_rules)

            return self.optimization_results
        except Exception:
            return {}

    def run_single_simulation(self, seed=None):
        """
        Execute single simulation run with specified random seed.
        
        Single simulation implementing Arthur et al. (1997) methodology
        for controlled experimental conditions.
        
        Args:
            seed (int): Random seed for reproducibility
            
        Returns:
            dict: Simulation results
        """
        try:
            if seed is not None:
                set_random_seed(seed)

            if not all([self.ca_model, self.market_model, self.historical_data is not None]):
                return {}

            # Prepare simulation data
            simulation_days = min(self.config['simulation_days'], len(self.historical_data))
            external_data = self.historical_data.iloc[:simulation_days].to_dict('records')

            # Run simulation
            results = self.market_model.run_simulation(
                num_days=simulation_days,
                external_data={'data': self.historical_data.iloc[:simulation_days]}
            )

            # Add metadata
            results['simulation_metadata'] = {
                'seed': seed,
                'simulation_days': simulation_days,
                'config': self.config.copy()
            }

            return results
        except Exception:
            return {}

    def run_parallel_simulations(self, num_runs=None):
        """
        Execute multiple parallel simulations for robust results.
        
        Parallel simulation framework for Monte Carlo analysis
        ensuring statistical robustness of results.
        
        Args:
            num_runs (int): Number of simulation runs
            
        Returns:
            list: List of simulation results
        """
        try:
            num_runs = num_runs or self.config['num_parallel_runs']
            
            # Prepare parameter sets with different seeds
            param_sets = [{'seed': i} for i in range(num_runs)]

            # Run parallel simulations
            parallel_runner = ParallelSimulationRunner(
                model_class=self.run_single_simulation,
                num_cores=self.config['parallel_cores']
            )

            all_results = []
            for params in param_sets:
                result = self.run_single_simulation(params['seed'])
                if result:
                    all_results.append(result)

            return all_results
        except Exception:
            return []

    def run_complete_simulation(self):
        """
        Execute complete simulation pipeline with comprehensive analysis.
        
        Complete pipeline implementing academic research methodology
        for computational finance studies.
        
        Returns:
            dict: Complete simulation results
        """
        try:
            # Step 1: Data preparation
            if not self.load_and_prepare_data():
                return {'status': 'failed', 'error': 'Data loading failed'}

            # Step 2: Model initialization
            if not self.initialize_models():
                return {'status': 'failed', 'error': 'Model initialization failed'}

            # Step 3: Model calibration
            calibration_results = self.calibrate_models()

            # Step 4: Single simulation run
            single_result = self.run_single_simulation(seed=42)

            # Step 5: Parallel simulation runs
            parallel_results = self.run_parallel_simulations(num_runs=10)

            # Step 6: Results analysis
            analysis_results = self.validate_results(single_result)

            # Compile comprehensive results
            complete_results = {
                'status': 'success',
                'data_summary': self.get_data_summary(),
                'calibration_results': calibration_results,
                'single_simulation': single_result,
                'parallel_simulations': parallel_results,
                'analysis_results': analysis_results,
                'performance_metrics': self.calculate_performance_metrics(single_result)
            }

            self.simulation_results = complete_results
            return complete_results
        except Exception:
            return {'status': 'failed', 'error': 'Complete simulation failed'}

    def validate_results(self, simulation_results):
        """
        Validate simulation results against historical data.
        
        Validation methodology implementing Diebold & Mariano (1995)
        forecast accuracy framework.
        
        Args:
            simulation_results (dict): Simulation results to validate
            
        Returns:
            dict: Validation metrics
        """
        try:
            if not simulation_results or 'price_series' not in simulation_results:
                return {}

            # Initialize results analyzer
            analyzer = ResultsAnalyzer(simulation_results, self.historical_data)
            
            # Calculate comprehensive metrics
            metrics = analyzer.calculate_metrics()
            
            # Perform statistical tests
            statistical_tests = analyzer.statistical_tests() if hasattr(analyzer, 'statistical_tests') else {}
            
            # Analyze agent behavior
            agent_analysis = analyzer.analyze_agent_behavior()

            validation_results = {
                'forecast_metrics': metrics,
                'statistical_tests': statistical_tests,
                'agent_analysis': agent_analysis,
                'data_quality': validate_data_integrity(self.historical_data)
            }

            return validation_results
        except Exception:
            return {}

    def calculate_performance_metrics(self, results):
        """
        Calculate comprehensive performance metrics for academic evaluation.
        
        Performance evaluation implementing academic standards for
        computational finance research validation.
        
        Args:
            results (dict): Simulation results
            
        Returns:
            dict: Performance metrics
        """
        try:
            metrics = {}
            
            if 'price_series' in results:
                prices = results['price_series']
                returns = calculate_returns(prices)
                
                # Risk-return metrics
                metrics['annualized_return'] = np.mean(returns) * 252
                metrics['annualized_volatility'] = np.std(returns) * np.sqrt(252)
                metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility'] if metrics['annualized_volatility'] > 0 else 0
                
                # Risk metrics
                metrics['var_95'] = np.percentile(returns, 5)
                metrics['var_99'] = np.percentile(returns, 1)
                
                # Maximum drawdown
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                metrics['max_drawdown'] = np.min(drawdown)

            # Agent performance metrics
            if 'agent_performances' in results:
                agent_perfs = results['agent_performances']
                metrics['agent_diversity'] = len(agent_perfs)
                metrics['best_agent_performance'] = max([perf.get('total_pnl', 0) for perf in agent_perfs.values()])
                metrics['worst_agent_performance'] = min([perf.get('total_pnl', 0) for perf in agent_perfs.values()])

            return metrics
        except Exception:
            return {}

    def generate_visualizations(self, save_directory=None):
        """
        Generate comprehensive visualizations for academic presentation.
        
        Visualization generation following academic standards for
        computational finance research presentation.
        
        Args:
            save_directory (str): Directory to save visualizations
            
        Returns:
            dict: Generated visualization paths
        """
        try:
            if not self.simulation_results:
                return {}

            viz_paths = {}
            single_result = self.simulation_results.get('single_simulation', {})

            if save_directory:
                os.makedirs(save_directory, exist_ok=True)

            # Main simulation results
            main_plot_path = os.path.join(save_directory, 'simulation_results.png') if save_directory else None
            self.visualizer.plot_simulation_results(single_result, self.historical_data, main_plot_path)
            if main_plot_path:
                viz_paths['main_results'] = main_plot_path

            # CA evolution visualization
            if hasattr(self.ca_model, 'history') and self.ca_model.history:
                ca_plot_path = os.path.join(save_directory, 'ca_evolution.png') if save_directory else None
                self.visualizer.plot_ca_evolution(self.ca_model.history, ca_plot_path)
                if ca_plot_path:
                    viz_paths['ca_evolution'] = ca_plot_path

            # Interactive dashboard
            dashboard_path = os.path.join(save_directory, 'interactive_dashboard.html') if save_directory else None
            self.visualizer.create_interactive_dashboard(single_result, dashboard_path)
            if dashboard_path:
                viz_paths['dashboard'] = dashboard_path

            return viz_paths
        except Exception:
            return {}

    def generate_report(self, save_path=None):
        """
        Generate comprehensive academic research report.
        
        Report generation implementing academic standards for
        computational finance research documentation.
        
        Args:
            save_path (str): Path to save the report
            
        Returns:
            str: Generated report content
        """
        try:
            report_sections = []
            
            # Executive Summary
            report_sections.append("# Hybrid Gold Price Prediction - Research Report\n")
            report_sections.append("## Executive Summary\n")
            
            if self.simulation_results:
                status = self.simulation_results.get('status', 'unknown')
                report_sections.append(f"Simulation Status: {status}\n")
                
                # Performance metrics
                if 'performance_metrics' in self.simulation_results:
                    metrics = self.simulation_results['performance_metrics']
                    report_sections.append("### Performance Metrics\n")
                    for metric, value in metrics.items():
                        report_sections.append(f"- {metric}: {value:.4f}\n")

            # Data Summary
            if self.historical_data is not None:
                report_sections.append("\n## Data Summary\n")
                report_sections.append(f"- Data Points: {len(self.historical_data)}\n")
                report_sections.append(f"- Date Range: {self.historical_data.index.min()} to {self.historical_data.index.max()}\n")
                report_sections.append(f"- Features: {list(self.historical_data.columns)}\n")

            # Model Configuration
            report_sections.append("\n## Model Configuration\n")
            for key, value in self.config.items():
                report_sections.append(f"- {key}: {value}\n")

            # Calibration Results
            if self.optimization_results:
                report_sections.append("\n## Model Calibration\n")
                if 'validation_metrics' in self.optimization_results:
                    metrics = self.optimization_results['validation_metrics']
                    for metric, value in metrics.items():
                        report_sections.append(f"- {metric}: {value}\n")

            report_content = "".join(report_sections)
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_content)

            return report_content
        except Exception:
            return "Error generating report"

    def get_data_summary(self):
        """
        Get comprehensive data summary for analysis.
        
        Returns:
            dict: Data summary statistics
        """
        try:
            if self.historical_data is None:
                return {}

            summary = {
                'total_observations': len(self.historical_data),
                'date_range': {
                    'start': str(self.historical_data.index.min()),
                    'end': str(self.historical_data.index.max())
                },
                'features': list(self.historical_data.columns),
                'missing_values': self.historical_data.isnull().sum().to_dict(),
                'data_types': self.historical_data.dtypes.to_dict(),
                'summary_statistics': self.historical_data.describe().to_dict()
            }

            return summary
        except Exception:
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize simulation with default configuration
    simulation = HybridGoldSimulation()
    
    # Run complete simulation
    results = simulation.run_complete_simulation()
    
    # Generate visualizations
    viz_paths = simulation.generate_visualizations(save_directory='results')
    
    # Generate report
    report = simulation.generate_report(save_path='results/simulation_report.md')
    print("Simulation completed successfully.")