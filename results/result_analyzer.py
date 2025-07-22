# Results Analysis Module for Hybrid Gold Price Prediction
# Academic Implementation for Dissertation Research

"""
Comprehensive results analysis implementing academic standards for
computational finance research validation and statistical testing.

Citations:
[1] Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
    Journal of Business & Economic Statistics, 13(3), 253-263.
[2] Hansen, P.R. & Lunde, A. (2005). A forecast comparison of volatility models: 
    does anything beat a GARCH(1,1)? Journal of Applied Econometrics, 20(7), 873-889.
[3] Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of 
    prediction mean squared errors. International Journal of Forecasting, 13(2), 281-291.
[4] Jarque, C.M. & Bera, A.K. (1987). A test for normality of observations and 
    regression residuals. International Statistical Review, 55(2), 163-172.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import scipy.stats as stats
from scipy.stats import jarque_bera, shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')

class ResultsAnalyzer:
    """
    Comprehensive results analyzer implementing academic evaluation standards.
    
    Analysis framework based on Diebold & Mariano (1995) forecast evaluation
    methodology with Hansen & Lunde (2005) model comparison techniques.
    
    References:
        Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
        Journal of Business & Economic Statistics, 13(3), 253-263.
        
        Hansen, P.R. & Lunde, A. (2005). A forecast comparison of volatility models.
    """

    def __init__(self, simulation_results, actual_data=None):
        """
        Initialize results analyzer with comprehensive validation framework.
        
        Args:
            simulation_results (dict): Simulation results for analysis
            actual_data (pd.DataFrame): Actual market data for comparison
        """
        self.results = simulation_results
        self.actual = actual_data
        self.analysis_cache = {}
        
        # Configure academic visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Initialize analysis components
        self.forecast_errors = None
        self.validation_metrics = {}

    def calculate_metrics(self):
        """
        Calculate comprehensive evaluation metrics using academic standards.
        
        Metric calculation implementing Diebold & Mariano (1995) forecast
        evaluation framework with additional financial performance measures.
        
        Returns:
            dict: Comprehensive evaluation metrics
            
        References:
            Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
            Journal of Business & Economic Statistics, 13(3), 253-263.
        """
        try:
            metrics = {}
            
            if isinstance(self.results, dict) and 'price_series' in self.results:
                simulated_prices = np.array(self.results['price_series'])
                
                # Comparison metrics with actual data
                if self.actual is not None and 'Close' in self.actual.columns:
                    actual_prices = self.actual['Close'].values
                    
                    # Align series lengths
                    min_length = min(len(simulated_prices), len(actual_prices))
                    sim_prices = simulated_prices[:min_length]
                    act_prices = actual_prices[:min_length]
                    
                    # Forecast accuracy metrics
                    metrics['RMSE'] = np.sqrt(mean_squared_error(act_prices, sim_prices))
                    metrics['MAE'] = mean_absolute_error(act_prices, sim_prices)
                    metrics['MAPE'] = np.mean(np.abs((act_prices - sim_prices) / act_prices)) * 100
                    
                    # Directional accuracy (Harvey et al., 1997)
                    act_directions = np.sign(np.diff(act_prices))
                    sim_directions = np.sign(np.diff(sim_prices))
                    valid_indices = ~(np.isnan(act_directions) | np.isnan(sim_directions))
                    
                    if np.sum(valid_indices) > 0:
                        metrics['Directional_Accuracy'] = accuracy_score(
                            act_directions[valid_indices], 
                            sim_directions[valid_indices]
                        ) * 100
                    
                    # Correlation measures
                    metrics['Pearson_Correlation'] = np.corrcoef(act_prices, sim_prices)[0, 1]
                    metrics['Spearman_Correlation'] = stats.spearmanr(act_prices, sim_prices)[0]
                    
                    # Theil's U statistic for forecast quality
                    metrics['Theil_U'] = (np.sqrt(mean_squared_error(act_prices, sim_prices)) / 
                                        np.sqrt(np.mean(act_prices**2)))
                    
                    # Store forecast errors for further analysis
                    self.forecast_errors = sim_prices - act_prices

                # Price series statistical properties
                metrics['Price_Mean'] = np.mean(simulated_prices)
                metrics['Price_Std'] = np.std(simulated_prices)
                metrics['Price_Skewness'] = stats.skew(simulated_prices)
                metrics['Price_Kurtosis'] = stats.kurtosis(simulated_prices)
                
                # Return-based metrics
                if len(simulated_prices) > 1:
                    returns = np.diff(simulated_prices) / simulated_prices[:-1]
                    returns = returns[~np.isnan(returns)]  # Remove NaN values
                    
                    if len(returns) > 0:
                        metrics['Return_Mean'] = np.mean(returns)
                        metrics['Return_Std'] = np.std(returns)
                        metrics['Return_Skewness'] = stats.skew(returns)
                        metrics['Return_Kurtosis'] = stats.kurtosis(returns)
                        
                        # Annualized metrics
                        metrics['Annualized_Return'] = np.mean(returns) * 252
                        metrics['Annualized_Volatility'] = np.std(returns) * np.sqrt(252)
                        
                        # Sharpe ratio (assuming risk-free rate = 0)
                        if np.std(returns) > 0:
                            metrics['Sharpe_Ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
                        
                        # Maximum drawdown
                        cumulative_returns = np.cumprod(1 + returns)
                        running_max = np.maximum.accumulate(cumulative_returns)
                        drawdown = (cumulative_returns - running_max) / running_max
                        metrics['Max_Drawdown'] = np.min(drawdown)
                        
                        # Value at Risk (VaR)
                        metrics['VaR_95'] = np.percentile(returns, 5)
                        metrics['VaR_99'] = np.percentile(returns, 1)
                        
                        # Conditional Value at Risk (CVaR)
                        var_95 = metrics['VaR_95']
                        metrics['CVaR_95'] = np.mean(returns[returns <= var_95])

            # Cache metrics for future use
            self.validation_metrics = metrics
            self.analysis_cache['metrics'] = metrics
            
            return metrics

        except Exception:
            return {}

    def statistical_tests(self):
        """
        Perform comprehensive statistical tests on simulation results.
        
        Statistical testing implementing Jarque & Bera (1987) normality tests
        and additional diagnostic tests for model validation.
        
        Returns:
            dict: Statistical test results with p-values and interpretations
            
        References:
            Jarque, C.M. & Bera, A.K. (1987). A test for normality of observations 
            and regression residuals. International Statistical Review, 55(2), 163-172.
        """
        try:
            test_results = {}
            
            if isinstance(self.results, dict) and 'price_series' in self.results:
                simulated_prices = np.array(self.results['price_series'])
                
                # Calculate returns for testing
                if len(simulated_prices) > 1:
                    returns = np.diff(simulated_prices) / simulated_prices[:-1]
                    returns = returns[~np.isnan(returns)]
                    
                    if len(returns) > 3:
                        # Normality tests
                        
                        # Jarque-Bera test
                        jb_stat, jb_pvalue = jarque_bera(returns)
                        test_results['jarque_bera'] = {
                            'statistic': jb_stat,
                            'p_value': jb_pvalue,
                            'is_normal': jb_pvalue > 0.05,
                            'interpretation': 'Normal' if jb_pvalue > 0.05 else 'Non-normal'
                        }
                        
                        # Shapiro-Wilk test (for smaller samples)
                        if len(returns) <= 5000:
                            sw_stat, sw_pvalue = shapiro(returns)
                            test_results['shapiro_wilk'] = {
                                'statistic': sw_stat,
                                'p_value': sw_pvalue,
                                'is_normal': sw_pvalue > 0.05,
                                'interpretation': 'Normal' if sw_pvalue > 0.05 else 'Non-normal'
                            }
                        
                        # D'Agostino's normality test
                        da_stat, da_pvalue = normaltest(returns)
                        test_results['dagostino_test'] = {
                            'statistic': da_stat,
                            'p_value': da_pvalue,
                            'is_normal': da_pvalue > 0.05,
                            'interpretation': 'Normal' if da_pvalue > 0.05 else 'Non-normal'
                        }
                        
                        # Stationarity test (Augmented Dickey-Fuller)
                        try:
                            from statsmodels.tsa.stattools import adfuller
                            adf_result = adfuller(returns, maxlag=int(len(returns)**(1/3)))
                            test_results['adf_stationarity'] = {
                                'statistic': adf_result[0],
                                'p_value': adf_result[1],
                                'is_stationary': adf_result[1] < 0.05,
                                'critical_values': adf_result[4],
                                'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
                            }
                        except ImportError:
                            pass
                        
                        # Serial correlation tests
                        if len(returns) > 10:
                            # Ljung-Box test for autocorrelation
                            try:
                                from statsmodels.stats.diagnostic import acorr_ljungbox
                                lb_result = acorr_ljungbox(returns, lags=min(10, len(returns)//4), 
                                                         return_df=True)
                                test_results['ljung_box'] = {
                                    'statistic': lb_result['lb_stat'].iloc[-1],
                                    'p_value': lb_result['lb_pvalue'].iloc[-1],
                                    'has_autocorr': lb_result['lb_pvalue'].iloc[-1] < 0.05,
                                    'interpretation': 'Autocorrelated' if lb_result['lb_pvalue'].iloc[-1] < 0.05 else 'No autocorrelation'
                                }
                            except ImportError:
                                # Simple autocorrelation test
                                autocorr_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                                test_results['simple_autocorr'] = {
                                    'lag_1_autocorr': autocorr_1,
                                    'significant': abs(autocorr_1) > 2/np.sqrt(len(returns))
                                }
                        
                        # Volatility clustering test (ARCH effects)
                        squared_returns = returns**2
                        if len(squared_returns) > 10:
                            try:
                                from statsmodels.stats.diagnostic import het_arch
                                arch_result = het_arch(returns)
                                test_results['arch_test'] = {
                                    'lm_statistic': arch_result[0],
                                    'p_value': arch_result[1],
                                    'has_arch': arch_result[1] < 0.05,
                                    'interpretation': 'ARCH effects present' if arch_result[1] < 0.05 else 'No ARCH effects'
                                }
                            except ImportError:
                                # Simple volatility clustering test
                                vol_autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                                test_results['volatility_clustering'] = {
                                    'autocorr': vol_autocorr,
                                    'significant': abs(vol_autocorr) > 2/np.sqrt(len(squared_returns))
                                }
                
                # Forecast error tests (if actual data available)
                if self.forecast_errors is not None and len(self.forecast_errors) > 3:
                    # Test for forecast bias
                    bias_stat, bias_pvalue = stats.ttest_1samp(self.forecast_errors, 0)
                    test_results['forecast_bias'] = {
                        'statistic': bias_stat,
                        'p_value': bias_pvalue,
                        'is_unbiased': bias_pvalue > 0.05,
                        'mean_error': np.mean(self.forecast_errors),
                        'interpretation': 'Unbiased' if bias_pvalue > 0.05 else 'Biased'
                    }
                    
                    # Test for forecast efficiency (errors should be unpredictable)
                    if len(self.forecast_errors) > 1:
                        error_autocorr = np.corrcoef(self.forecast_errors[:-1], self.forecast_errors[1:])[0, 1]
                        test_results['forecast_efficiency'] = {
                            'error_autocorr': error_autocorr,
                            'is_efficient': abs(error_autocorr) < 2/np.sqrt(len(self.forecast_errors)),
                            'interpretation': 'Efficient' if abs(error_autocorr) < 2/np.sqrt(len(self.forecast_errors)) else 'Inefficient'
                        }

            # Cache results
            self.analysis_cache['statistical_tests'] = test_results
            
            return test_results

        except Exception:
            return {}

    def analyze_agent_behavior(self):
        """
        Comprehensive analysis of agent behavior patterns and performance.
        
        Agent analysis implementing behavioral finance methodology for
        heterogeneous agent performance evaluation.
        
        Returns:
            dict: Detailed agent behavior analysis
        """
        try:
            agent_analysis = {}
            
            # Performance analysis by agent type
            if isinstance(self.results, dict) and 'agent_performances' in self.results:
                performances = self.results['agent_performances']
                
                for agent_type, performance in performances.items():
                    analysis = {
                        'total_pnl': performance.get('total_pnl', 0),
                        'win_rate': performance.get('win_rate', 0),
                        'total_trades': performance.get('total_trades', 0),
                        'final_portfolio_value': performance.get('final_portfolio_value', 0),
                        'average_trade_size': 0,
                        'risk_adjusted_return': 0,
                        'information_ratio': 0
                    }
                    
                    # Calculate derived metrics
                    if analysis['total_trades'] > 0:
                        analysis['average_trade_pnl'] = analysis['total_pnl'] / analysis['total_trades']
                    
                    # Risk-adjusted measures
                    if analysis['total_pnl'] != 0:
                        # Simplified risk adjustment
                        analysis['risk_adjusted_return'] = analysis['total_pnl'] / abs(analysis['total_pnl'])
                    
                    # Performance ranking metrics
                    analysis['consistency_score'] = analysis['win_rate']
                    analysis['profitability_score'] = max(0, analysis['total_pnl'] / 1000)  # Normalized
                    
                    agent_analysis[agent_type] = analysis

                # Cross-agent comparison
                if len(agent_analysis) > 1:
                    pnl_values = [agent_analysis[agent]['total_pnl'] for agent in agent_analysis]
                    win_rates = [agent_analysis[agent]['win_rate'] for agent in agent_analysis]
                    
                    agent_analysis['comparison_metrics'] = {
                        'best_performer': max(agent_analysis.keys(), 
                                            key=lambda x: agent_analysis[x]['total_pnl']),
                        'worst_performer': min(agent_analysis.keys(), 
                                             key=lambda x: agent_analysis[x]['total_pnl']),
                        'most_consistent': max(agent_analysis.keys(), 
                                             key=lambda x: agent_analysis[x]['win_rate']),
                        'pnl_dispersion': np.std(pnl_values),
                        'win_rate_dispersion': np.std(win_rates)
                    }

            # Behavioral pattern analysis from data collector
            if isinstance(self.results, dict) and 'data_collector' in self.results:
                try:
                    agent_data = self.results['data_collector'].get_agent_vars_dataframe()
                    if not agent_data.empty and 'Agent_Type' in agent_data.columns:
                        # Position dynamics analysis
                        position_analysis = {}
                        
                        for agent_type in agent_data['Agent_Type'].unique():
                            if pd.notna(agent_type):
                                type_data = agent_data[agent_data['Agent_Type'] == agent_type]
                                
                                position_analysis[agent_type] = {
                                    'avg_position': type_data['Position'].mean(),
                                    'position_volatility': type_data['Position'].std(),
                                    'max_position': type_data['Position'].max(),
                                    'min_position': type_data['Position'].min(),
                                    'position_changes': len(type_data[type_data['Position'].diff() != 0]),
                                    'avg_portfolio_value': type_data['Portfolio_Value'].mean(),
                                    'portfolio_volatility': type_data['Portfolio_Value'].std()
                                }
                        
                        agent_analysis['position_dynamics'] = position_analysis
                        
                        # Trading frequency analysis
                        trading_frequency = {}
                        for agent_type in agent_data['Agent_Type'].unique():
                            if pd.notna(agent_type):
                                type_data = agent_data[agent_data['Agent_Type'] == agent_type]
                                position_changes = type_data['Position'].diff().abs().sum()
                                total_periods = len(type_data)
                                
                                trading_frequency[agent_type] = {
                                    'activity_ratio': position_changes / total_periods if total_periods > 0 else 0,
                                    'total_position_changes': position_changes
                                }
                        
                        agent_analysis['trading_patterns'] = trading_frequency
                        
                except Exception:
                    pass

            # Cache results
            self.analysis_cache['agent_behavior'] = agent_analysis
            
            return agent_analysis

        except Exception:
            return {}

    def backtesting_analysis(self, train_ratio=0.8):
        """
        Perform rolling window backtesting analysis for out-of-sample validation.
        
        Backtesting methodology implementing academic standards for
        time series model validation (Diebold & Mariano, 1995).
        
        Args:
            train_ratio (float): Ratio of data for training (default 0.8)
            
        Returns:
            dict: Comprehensive backtesting results
            
        References:
            Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. 
            Journal of Business & Economic Statistics, 13(3), 253-263.
        """
        try:
            backtest_results = {}
            
            if isinstance(self.results, dict) and 'price_series' in self.results:
                simulated_prices = np.array(self.results['price_series'])
                
                if self.actual is not None and 'Close' in self.actual.columns:
                    actual_prices = self.actual['Close'].values
                    
                    # Align data lengths
                    min_length = min(len(simulated_prices), len(actual_prices))
                    sim_prices = simulated_prices[:min_length]
                    act_prices = actual_prices[:min_length]
                    
                    # Split data into train/test
                    split_point = int(len(act_prices) * train_ratio)
                    
                    # Training period analysis
                    train_sim = sim_prices[:split_point]
                    train_act = act_prices[:split_point]
                    
                    # Testing period analysis
                    test_sim = sim_prices[split_point:]
                    test_act = act_prices[split_point:]
                    
                    if len(test_sim) > 0 and len(test_act) > 0:
                        # Out-of-sample metrics
                        backtest_results['train_period'] = {
                            'rmse': np.sqrt(mean_squared_error(train_act, train_sim)),
                            'mae': mean_absolute_error(train_act, train_sim),
                            'correlation': np.corrcoef(train_act, train_sim)[0, 1],
                            'data_points': len(train_sim)
                        }
                        
                        backtest_results['test_period'] = {
                            'rmse': np.sqrt(mean_squared_error(test_act, test_sim)),
                            'mae': mean_absolute_error(test_act, test_sim),
                            'correlation': np.corrcoef(test_act, test_sim)[0, 1],
                            'data_points': len(test_sim)
                        }
                        
                        # Rolling window validation
                        window_size = max(20, len(act_prices) // 10)
                        rolling_errors = []
                        
                        for i in range(split_point, len(act_prices) - window_size):
                            window_sim = sim_prices[i:i+window_size]
                            window_act = act_prices[i:i+window_size]
                            
                            if len(window_sim) == len(window_act):
                                window_rmse = np.sqrt(mean_squared_error(window_act, window_sim))
                                rolling_errors.append(window_rmse)
                        
                        if rolling_errors:
                            backtest_results['rolling_validation'] = {
                                'mean_rmse': np.mean(rolling_errors),
                                'std_rmse': np.std(rolling_errors),
                                'min_rmse': np.min(rolling_errors),
                                'max_rmse': np.max(rolling_errors),
                                'stability_ratio': np.std(rolling_errors) / np.mean(rolling_errors)
                            }
                        
                        # Directional accuracy over time
                        test_act_returns = np.diff(test_act) / test_act[:-1]
                        test_sim_returns = np.diff(test_sim) / test_sim[:-1]
                        
                        if len(test_act_returns) > 0:
                            test_act_dirs = np.sign(test_act_returns)
                            test_sim_dirs = np.sign(test_sim_returns)
                            
                            directional_accuracy = np.mean(test_act_dirs == test_sim_dirs)
                            backtest_results['directional_analysis'] = {
                                'test_directional_accuracy': directional_accuracy * 100,
                                'bullish_predictions': np.sum(test_sim_dirs > 0),
                                'bearish_predictions': np.sum(test_sim_dirs < 0),
                                'neutral_predictions': np.sum(test_sim_dirs == 0)
                            }
                        
                        # Model degradation analysis
                        train_rmse = backtest_results['train_period']['rmse']
                        test_rmse = backtest_results['test_period']['rmse']
                        
                        backtest_results['model_stability'] = {
                            'performance_degradation': (test_rmse - train_rmse) / train_rmse * 100,
                            'overfitting_indicator': test_rmse / train_rmse,
                            'generalization_score': 1 - min(1, (test_rmse - train_rmse) / train_rmse)
                        }
                        
                        # Statistical significance testing
                        from scipy import stats
                        
                        # Diebold-Mariano test for forecast accuracy
                        if len(test_act) > 10:
                            # Forecast errors
                            errors_1 = (test_act - test_sim) ** 2
                            errors_2 = (test_act - np.mean(test_act)) ** 2  # Naive forecast
                            
                            error_diff = errors_1 - errors_2
                            
                            if len(error_diff) > 3:
                                dm_stat, dm_pvalue = stats.ttest_1samp(error_diff, 0)
                                
                                backtest_results['statistical_tests'] = {
                                    'diebold_mariano_stat': dm_stat,
                                    'diebold_mariano_pvalue': dm_pvalue,
                                    'forecast_significantly_better': dm_pvalue < 0.05 and dm_stat < 0,
                                    'interpretation': 'Better than naive' if (dm_pvalue < 0.05 and dm_stat < 0) else 'Not significantly better'
                                }
                    
                    # Cache results
                    self.analysis_cache['backtesting'] = backtest_results
                    
                    return backtest_results
                else:
                    return {'error': 'No actual data available for backtesting'}
            else:
                return {'error': 'No price series in simulation results'}
                
        except Exception as e:
            return {'error': f'Backtesting analysis failed: {str(e)}'}

def plot_backtesting_results(self, backtest_results, save_path=None):
    """
    Visualize backtesting analysis results.
    
    Args:
        backtest_results (dict): Results from backtesting analysis
        save_path (str): Path to save the plot
    """
    try:
        if not backtest_results or 'error' in backtest_results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtesting Analysis Results', fontsize=16)
        
        # Plot 1: Train vs Test Performance
        ax1 = axes[0, 0]
        if 'train_period' in backtest_results and 'test_period' in backtest_results:
            metrics = ['rmse', 'mae', 'correlation']
            train_values = [backtest_results['train_period'][m] for m in metrics]
            test_values = [backtest_results['test_period'][m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
            ax1.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Values')
            ax1.set_title('Train vs Test Performance')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling Validation
        ax2 = axes[0, 1]
        if 'rolling_validation' in backtest_results:
            rolling_stats = backtest_results['rolling_validation']
            labels = list(rolling_stats.keys())
            values = list(rolling_stats.values())
            
            ax2.bar(range(len(labels)), values, alpha=0.7)
            ax2.set_xlabel('Rolling Statistics')
            ax2.set_ylabel('RMSE Values')
            ax2.set_title('Rolling Window Validation')
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Directional Accuracy
        ax3 = axes[1, 0]
        if 'directional_analysis' in backtest_results:
            dir_data = backtest_results['directional_analysis']
            predictions = [dir_data['bullish_predictions'], 
                          dir_data['bearish_predictions'], 
                          dir_data['neutral_predictions']]
            labels = ['Bullish', 'Bearish', 'Neutral']
            colors = ['green', 'red', 'gray']
            
            ax3.pie(predictions, labels=labels, colors=colors, autopct='%1.1f%%')
            ax3.set_title(f"Prediction Distribution\n(Accuracy: {dir_data['test_directional_accuracy']:.1f}%)")
        
        # Plot 4: Model Stability
        ax4 = axes[1, 1]
        if 'model_stability' in backtest_results:
            stability = backtest_results['model_stability']
            metrics = list(stability.keys())
            values = list(stability.values())
            
            colors = ['red' if v > 1 else 'green' for v in values]
            bars = ax4.bar(range(len(metrics)), values, color=colors, alpha=0.7)
            
            ax4.set_xlabel('Stability Metrics')
            ax4.set_ylabel('Values')
            ax4.set_title('Model Stability Analysis')
            ax4.set_xticks(range(len(metrics)))
            ax4.set_xticklabels(metrics, rotation=45)
            ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        pass
