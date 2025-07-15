# Results Analysis Module for Hybrid Gold Price Prediction
# Implements results analysis as per instruction manual Phase 5.1
# Research purposes only - academic dissertation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
from scipy.stats import jarque_bera, shapiro
import warnings
warnings.filterwarnings('ignore')

class ResultsAnalyzer:
    """
    Results analyzer for simulation output analysis
    As specified in instruction manual Step 5.1
    
    Reference: Instruction manual Phase 5.1 - Results Analysis
    """
    
    def __init__(self, simulation_results, actual_data=None):
        """
        Initialize results analyzer
        
        Args:
            simulation_results (dict or pd.DataFrame): Simulation results
            actual_data (pd.DataFrame): Actual market data for comparison
            
        Reference: Instruction manual - "def __init__(self, simulation_results, actual_data):"
        """
        self.results = simulation_results
        self.actual = actual_data
        self.analysis_cache = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("ResultsAnalyzer initialized")
        if actual_data is not None:
            print(f"Actual data available: {len(actual_data)} observations")
    
    def calculate_metrics(self):
        """
        Calculate RMSE, MAE, MAPE, directional accuracy
        
        Returns:
            dict: Calculated metrics
            
        Reference: Instruction manual - "RMSE, MAE, MAPE, directional accuracy"
        """
        try:
            metrics = {}
            
            if isinstance(self.results, dict) and 'price_series' in self.results:
                simulated_prices = self.results['price_series']
                
                if self.actual is not None and 'Close' in self.actual.columns:
                    actual_prices = self.actual['Close'].values
                    
                    # Align lengths
                    min_length = min(len(simulated_prices), len(actual_prices))
                    sim_prices = simulated_prices[:min_length]
                    act_prices = actual_prices[:min_length]
                    
                    # Calculate metrics
                    metrics['RMSE'] = np.sqrt(mean_squared_error(act_prices, sim_prices))
                    metrics['MAE'] = mean_absolute_error(act_prices, sim_prices)
                    
                    # MAPE (Mean Absolute Percentage Error)
                    mape = np.mean(np.abs((act_prices - sim_prices) / act_prices)) * 100
                    metrics['MAPE'] = mape
                    
                    # Directional accuracy
                    act_directions = np.sign(np.diff(act_prices))
                    sim_directions = np.sign(np.diff(sim_prices))
                    
                    directional_accuracy = np.mean(act_directions == sim_directions) * 100
                    metrics['Directional_Accuracy'] = directional_accuracy
                    
                    # Correlation
                    correlation = np.corrcoef(act_prices, sim_prices)[0, 1]
                    metrics['Correlation'] = correlation
                    
                    # Theil's U statistic
                    theil_u = np.sqrt(mean_squared_error(act_prices, sim_prices)) / \
                             np.sqrt(np.mean(act_prices**2))
                    metrics['Theil_U'] = theil_u
                    
                    print("Comparison metrics calculated successfully")
                
                # Price series statistics
                metrics['Price_Mean'] = np.mean(simulated_prices)
                metrics['Price_Std'] = np.std(simulated_prices)
                metrics['Price_Min'] = np.min(simulated_prices)
                metrics['Price_Max'] = np.max(simulated_prices)
                metrics['Price_Range'] = np.max(simulated_prices) - np.min(simulated_prices)
                
                # Calculate returns
                returns = np.diff(simulated_prices) / simulated_prices[:-1]
                metrics['Return_Mean'] = np.mean(returns)
                metrics['Return_Std'] = np.std(returns)
                metrics['Return_Skewness'] = stats.skew(returns)
                metrics['Return_Kurtosis'] = stats.kurtosis(returns)
                
                # Volatility clustering (ARCH test)
                squared_returns = returns**2
                if len(squared_returns) > 10:
                    autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                    metrics['Volatility_Clustering'] = autocorr
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                metrics['Max_Drawdown'] = np.min(drawdown)
                
                # Sharpe ratio (assuming risk-free rate = 0)
                if np.std(returns) > 0:
                    metrics['Sharpe_Ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
                
                print(f"Calculated {len(metrics)} metrics")
                
            else:
                print("No price series found in results")
            
            # Cache results
            self.analysis_cache['metrics'] = metrics
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def plot_price_comparison(self, save_path=None):
        """
        Plot simulated vs actual prices
        
        Args:
            save_path (str): Path to save the plot
            
        Reference: Instruction manual - "Plot simulated vs actual prices"
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Price Analysis: Simulated vs Actual', fontsize=16)
            
            if isinstance(self.results, dict) and 'price_series' in self.results:
                simulated_prices = self.results['price_series']
                dates = pd.date_range(start='2023-01-01', periods=len(simulated_prices), freq='D')
                
                # Plot 1: Price series comparison
                ax1 = axes[0, 0]
                ax1.plot(dates, simulated_prices, label='Simulated', color='blue', alpha=0.7)
                
                if self.actual is not None and 'Close' in self.actual.columns:
                    actual_prices = self.actual['Close'].values
                    min_length = min(len(simulated_prices), len(actual_prices))
                    actual_dates = dates[:min_length]
                    
                    ax1.plot(actual_dates, actual_prices[:min_length], 
                            label='Actual', color='red', alpha=0.7)
                
                ax1.set_title('Price Series Comparison')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Returns comparison
                ax2 = axes[0, 1]
                sim_returns = np.diff(simulated_prices) / simulated_prices[:-1]
                ax2.plot(dates[1:], sim_returns, label='Simulated Returns', alpha=0.7)
                
                if self.actual is not None and 'Close' in self.actual.columns:
                    actual_returns = np.diff(actual_prices[:min_length]) / actual_prices[:min_length-1]
                    ax2.plot(actual_dates[1:], actual_returns, label='Actual Returns', alpha=0.7)
                
                ax2.set_title('Returns Comparison')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Returns')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Price distribution
                ax3 = axes[1, 0]
                ax3.hist(simulated_prices, bins=30, alpha=0.7, label='Simulated', density=True)
                
                if self.actual is not None and 'Close' in self.actual.columns:
                    ax3.hist(actual_prices[:min_length], bins=30, alpha=0.7, 
                            label='Actual', density=True)
                
                ax3.set_title('Price Distribution')
                ax3.set_xlabel('Price')
                ax3.set_ylabel('Density')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Q-Q plot for returns
                ax4 = axes[1, 1]
                if len(sim_returns) > 10:
                    stats.probplot(sim_returns, dist="norm", plot=ax4)
                    ax4.set_title('Q-Q Plot: Simulated Returns vs Normal')
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {save_path}")
                
                plt.show()
                
            else:
                print("No price series data available for plotting")
                
        except Exception as e:
            print(f"Error plotting price comparison: {e}")
    
    def analyze_agent_behavior(self):
        """
        Analyze agent position changes over time
        
        Returns:
            dict: Agent behavior analysis
            
        Reference: Instruction manual - "Analyze agent position changes over time"
        """
        try:
            agent_analysis = {}
            
            if isinstance(self.results, dict) and 'agent_performances' in self.results:
                performances = self.results['agent_performances']
                
                # Analyze performance by agent type
                for agent_type, performance in performances.items():
                    analysis = {
                        'total_pnl': performance.get('total_pnl', 0),
                        'win_rate': performance.get('win_rate', 0),
                        'total_trades': performance.get('total_trades', 0),
                        'portfolio_value': performance.get('final_portfolio_value', 0),
                        'risk_adjusted_return': 0
                    }
                    
                    # Calculate risk-adjusted return
                    if performance.get('total_pnl', 0) != 0:
                        analysis['risk_adjusted_return'] = performance['total_pnl'] / abs(performance['total_pnl'])
                    
                    agent_analysis[agent_type] = analysis
                
                print(f"Analyzed behavior for {len(agent_analysis)} agent types")
                
            # Additional analysis if data collector results available
            if isinstance(self.results, dict) and 'data_collector' in self.results:
                try:
                    agent_data = self.results['data_collector'].get_agent_vars_dataframe()
                    
                    if not agent_data.empty:
                        # Analyze position changes over time
                        position_analysis = {}
                        
                        for agent_type in agent_data['Agent_Type'].unique():
                            type_data = agent_data[agent_data['Agent_Type'] == agent_type]
                            
                            position_analysis[agent_type] = {
                                'avg_position': type_data['Position'].mean(),
                                'position_volatility': type_data['Position'].std(),
                                'max_position': type_data['Position'].max(),
                                'min_position': type_data['Position'].min(),
                                'avg_portfolio_value': type_data['Portfolio_Value'].mean(),
                                'portfolio_volatility': type_data['Portfolio_Value'].std()
                            }
                        
                        agent_analysis['position_analysis'] = position_analysis
                        
                except Exception as e:
                    print(f"Error analyzing agent data: {e}")
            
            # Cache results
            self.analysis_cache['agent_behavior'] = agent_analysis
            
            return agent_analysis
            
        except Exception as e:
            print(f"Error analyzing agent behavior: {e}")
            return {}
    
    def plot_agent_behavior(self, save_path=None):
        """
        Plot agent behavior analysis
        
        Args:
            save_path (str): Path to save the plot
        """
        try:
            agent_analysis = self.analyze_agent_behavior()
            
            if agent_analysis:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Agent Behavior Analysis', fontsize=16)
                
                # Extract agent types and metrics
                agent_types = list(agent_analysis.keys())
                if 'position_analysis' in agent_types:
                    agent_types.remove('position_analysis')
                
                if agent_types:
                    # Plot 1: P&L by agent type
                    ax1 = axes[0, 0]
                    pnl_values = [agent_analysis[agent]['total_pnl'] for agent in agent_types]
                    ax1.bar(agent_types, pnl_values, alpha=0.7)
                    ax1.set_title('Total P&L by Agent Type')
                    ax1.set_ylabel('P&L')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Win rate by agent type
                    ax2 = axes[0, 1]
                    win_rates = [agent_analysis[agent]['win_rate'] for agent in agent_types]
                    ax2.bar(agent_types, win_rates, alpha=0.7, color='green')
                    ax2.set_title('Win Rate by Agent Type')
                    ax2.set_ylabel('Win Rate')
                    ax2.set_ylim(0, 1)
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)
                    
                    # Plot 3: Total trades by agent type
                    ax3 = axes[1, 0]
                    total_trades = [agent_analysis[agent]['total_trades'] for agent in agent_types]
                    ax3.bar(agent_types, total_trades, alpha=0.7, color='orange')
                    ax3.set_title('Total Trades by Agent Type')
                    ax3.set_ylabel('Number of Trades')
                    ax3.tick_params(axis='x', rotation=45)
                    ax3.grid(True, alpha=0.3)
                    
                    # Plot 4: Portfolio value by agent type
                    ax4 = axes[1, 1]
                    portfolio_values = [agent_analysis[agent]['portfolio_value'] for agent in agent_types]
                    ax4.bar(agent_types, portfolio_values, alpha=0.7, color='purple')
                    ax4.set_title('Final Portfolio Value by Agent Type')
                    ax4.set_ylabel('Portfolio Value')
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    if save_path:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"Agent behavior plot saved to {save_path}")
                    
                    plt.show()
                
                else:
                    print("No agent data available for plotting")
                    
        except Exception as e:
            print(f"Error plotting agent behavior: {e}")
    
    def statistical_tests(self):
        """
        Perform statistical tests on simulation results
        
        Returns:
            dict: Statistical test results
        """
        try:
            test_results = {}
            
            if isinstance(self.results, dict) and 'price_series' in self.results:
                simulated_prices = self.results['price_series']
                returns = np.diff(simulated_prices) / simulated_prices[:-1]
                
                # Normality tests
                if len(returns) > 3:
                    # Jarque-Bera test
                    jb_stat, jb_pvalue = jarque_bera(returns)
                    test_results['jarque_bera'] = {
                        'statistic': jb_stat,
                        'p_value': jb_pvalue,
                        'is_normal': jb_pvalue > 0.05
                    }
                    
                    # Shapiro-Wilk test (for smaller samples)
                    if len(returns) < 5000:
                        sw_stat, sw_pvalue = shapiro(returns)
                        test_results['shapiro_wilk'] = {
                            'statistic': sw_stat,
                            'p_value': sw_pvalue,
                            'is_normal': sw_pvalue > 0.05
                        }
                
                # Stationarity test (Augmented Dickey-Fuller)
                try:
                    from statsmodels.tsa.stattools import adfuller
                    
                    adf_result = adfuller(returns)
                    test_results['adf_stationarity'] = {
                        'statistic': adf_result[0],
                        'p_value': adf_result[1],
                        'is_stationary': adf_result[1] < 0.05,
                        'critical_values': adf_result[4]
                    }
                    
                except ImportError:
                    print("Statsmodels not available for stationarity test")
                
                # Autocorrelation test
                if len(returns) > 10:
                    # Ljung-Box test for autocorrelation
                    from scipy import stats
                    
                    # Calculate autocorrelation for lag 1
                    autocorr_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    test_results['autocorrelation'] = {
                        'lag_1_autocorr': autocorr_1,
                        'significant': abs(autocorr_1) > 2/np.sqrt(len(returns))
                    }
                
                # Volatility clustering test
                squared_returns = returns**2
                if len(squared_returns) > 10:
                    vol_autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                    test_results['volatility_clustering'] = {
                        'autocorr': vol_autocorr,
                        'significant': abs(vol_autocorr) > 2/np.sqrt(len(squared_returns))
                    }
                
                print(f"Performed {len(test_results)} statistical tests")
                
            # Cache results
            self.analysis_cache['statistical_tests'] = test_results
            
            return test_results
            
        except Exception as e:
            print(f"Error performing statistical tests: {e}")
            return {}
    
    def generate_report(self, save_path=None):
        """
        Generate comprehensive analysis report
        
        Args:
            save_path (str): Path to save the report
            
        Returns:
            str: Analysis report
            
        Reference: Instruction manual - "Generate comprehensive analysis report"
        """
        try:
            report = []
            report.append("="*60)
            report.append("HYBRID GOLD PRICE PREDICTION SIMULATION - ANALYSIS REPORT")
            report.append("="*60)
            report.append("")
            
            # Basic information
            report.append("1. SIMULATION OVERVIEW")
            report.append("-" * 30)
            
            if isinstance(self.results, dict):
                if 'market_statistics' in self.results:
                    stats = self.results['market_statistics']
                    report.append(f"Initial Price: ${stats.get('initial_price', 'N/A'):.2f}")
                    report.append(f"Final Price: ${stats.get('final_price', 'N/A'):.2f}")
                    report.append(f"Price Change: {stats.get('price_change', 0)*100:.2f}%")
                    report.append(f"Simulation Days: {stats.get('total_simulation_days', 'N/A')}")
                    report.append(f"Max Price: ${stats.get('max_price', 'N/A'):.2f}")
                    report.append(f"Min Price: ${stats.get('min_price', 'N/A'):.2f}")
                
                if 'total_volume' in self.results:
                    report.append(f"Total Volume: {self.results['total_volume']:.2f}")
                
                if 'average_volatility' in self.results:
                    report.append(f"Average Volatility: {self.results['average_volatility']:.4f}")
            
            report.append("")
            
            # Performance metrics
            report.append("2. PERFORMANCE METRICS")
            report.append("-" * 30)
            
            metrics = self.calculate_metrics()
            if metrics:
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report.append(f"{metric}: {value:.4f}")
                    else:
                        report.append(f"{metric}: {value}")
            
            report.append("")
            
            # Agent behavior analysis
            report.append("3. AGENT BEHAVIOR ANALYSIS")
            report.append("-" * 30)
            
            agent_analysis = self.analyze_agent_behavior()
            if agent_analysis:
                for agent_type, analysis in agent_analysis.items():
                    if agent_type != 'position_analysis':
                        report.append(f"\n{agent_type.upper()}:")
                        report.append(f"  Total P&L: {analysis.get('total_pnl', 0):.2f}")
                        report.append(f"  Win Rate: {analysis.get('win_rate', 0):.2f}")
                        report.append(f"  Total Trades: {analysis.get('total_trades', 0)}")
                        report.append(f"  Portfolio Value: {analysis.get('portfolio_value', 0):.2f}")
            
            report.append("")
            
            # Statistical tests
            report.append("4. STATISTICAL TESTS")
            report.append("-" * 30)
            
            test_results = self.statistical_tests()
            if test_results:
                for test_name, result in test_results.items():
                    report.append(f"\n{test_name.upper()}:")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            report.append(f"  {key}: {value}")
                    else:
                        report.append(f"  Result: {result}")
            
            report.append("")
            
            # Conclusions
            report.append("5. CONCLUSIONS")
            report.append("-" * 30)
            
            # Generate automatic conclusions based on results
            if metrics:
                if 'Correlation' in metrics and metrics['Correlation'] > 0.7:
                    report.append("- High correlation with actual data indicates good model fit")
                elif 'Correlation' in metrics and metrics['Correlation'] < 0.3:
                    report.append("- Low correlation suggests model may need calibration")
                
                if 'Directional_Accuracy' in metrics and metrics['Directional_Accuracy'] > 60:
                    report.append("- Good directional accuracy for trading applications")
                
                if 'Sharpe_Ratio' in metrics and metrics['Sharpe_Ratio'] > 0.5:
                    report.append("- Positive risk-adjusted returns achieved")
            
            report.append("")
            report.append("="*60)
            report.append("End of Report")
            report.append("="*60)
            
            # Join report
            full_report = "\n".join(report)
            
            # Save report
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(full_report)
                print(f"Report saved to {save_path}")
            
            # Cache report
            self.analysis_cache['report'] = full_report
            
            return full_report
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return f"Error generating report: {e}"
    
    def export_results(self, filename_prefix):
        """
        Export all analysis results to files
        
        Args:
            filename_prefix (str): Prefix for output files
        """
        try:
            # Export metrics
            metrics = self.calculate_metrics()
            if metrics:
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(f"{filename_prefix}_metrics.csv", index=False)
                print(f"Metrics exported to {filename_prefix}_metrics.csv")
            
            # Export agent analysis
            agent_analysis = self.analyze_agent_behavior()
            if agent_analysis:
                agent_df = pd.DataFrame(agent_analysis).T
                agent_df.to_csv(f"{filename_prefix}_agent_analysis.csv")
                print(f"Agent analysis exported to {filename_prefix}_agent_analysis.csv")
            
            # Export statistical tests
            test_results = self.statistical_tests()
            if test_results:
                import json
                with open(f"{filename_prefix}_statistical_tests.json", 'w') as f:
                    json.dump(test_results, f, indent=2)
                print(f"Statistical tests exported to {filename_prefix}_statistical_tests.json")
            
            # Export report
            report = self.generate_report()
            with open(f"{filename_prefix}_report.txt", 'w') as f:
                f.write(report)
            print(f"Report exported to {filename_prefix}_report.txt")
            
        except Exception as e:
            print(f"Error exporting results: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create mock simulation results
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    mock_results = {
        'price_series': 1800 + np.cumsum(np.random.randn(252) * 5),
        'final_price': 1850,
        'total_volume': 50000,
        'average_volatility': 0.025,
        'market_statistics': {
            'initial_price': 1800,
            'final_price': 1850,
            'price_change': 0.027,
            'total_simulation_days': 252,
            'max_price': 1920,
            'min_price': 1750
        },
        'agent_performances': {
            'herder': {'total_pnl': 150, 'win_rate': 0.65, 'total_trades': 20, 'final_portfolio_value': 10150},
            'contrarian': {'total_pnl': -50, 'win_rate': 0.45, 'total_trades': 25, 'final_portfolio_value': 9950},
            'trend_follower': {'total_pnl': 200, 'win_rate': 0.70, 'total_trades': 18, 'final_portfolio_value': 10200}
        }
    }
    
    # Create mock actual data
    actual_data = pd.DataFrame({
        'Close': 1800 + np.cumsum(np.random.randn(252) * 4)
    }, index=dates)
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(mock_results, actual_data)
    
    # Run analysis
    metrics = analyzer.calculate_metrics()
    print("\nCalculated Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    agent_analysis = analyzer.analyze_agent_behavior()
    print("\nAgent Analysis:")
    for agent_type, analysis in agent_analysis.items():
        print(f"{agent_type}: {analysis}")
    
    statistical_tests = analyzer.statistical_tests()
    print("\nStatistical Tests:")
    for test, result in statistical_tests.items():
        print(f"{test}: {result}")
    
    # Generate report
    report = analyzer.generate_report()
    print("\nGenerated Report:")
    print(report[:500] + "..." if len(report) > 500 else report)
    
    print("\nResults analyzer testing completed successfully!")
