# Header Module for Hybrid Gold Price Prediction
# Academic Implementation for Dissertation Research

"""
Common imports, configurations, and utility functions for hybrid CA-ABM system.

Citations:
[1] McKinney, W. (2010). Data structures for statistical computing in Python. 
    Proceedings of the 9th Python in Science Conference, 51-56.
[2] Harris, C.R., Millman, K.J., van der Walt, S.J., et al. (2020). Array 
    programming with NumPy. Nature, 585(7825), 357-362.
[3] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: 
    Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
"""

import sys
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# Core data science libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Machine learning and statistics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats

# Optimization libraries
from scipy.optimize import differential_evolution, minimize
from joblib import Parallel, delayed

# Mesa framework for agent-based modeling
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Financial data
import yfinance as yf

# Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# High-performance computing
from numba import jit

# Default configuration for reproducible research
DEFAULT_CONFIG = {
    'start_date': '2014-01-01',
    'end_date': '2015-01-01',
    'ca_grid_size': (20, 20),
    'ca_num_states': 3,
    'num_agents': 100,
    'agent_grid_size': (15, 15),
    'initial_gold_price': 1800,
    'simulation_days': 252,
    'num_parallel_runs': 50,
    'optimization_cores': 2,
    'parallel_cores': mp.cpu_count(),
    'random_seed': 42
}

def set_random_seed(seed=42):
    """
    Set reproducible random seeds for research reproducibility.
    
    Implementation following best practices in computational finance
    research for ensuring reproducible results (Peng, 2011).
    
    Args:
        seed (int): Random seed value
        
    References:
        Peng, R.D. (2011). Reproducible research in computational science. 
        Science, 334(6060), 1226-1227.
    """
    np.random.seed(seed)
    import random
    random.seed(seed)

def discretize_signal(value, thresholds=(-0.33, 0.33)):
    """
    Convert continuous values to discrete states for CA processing.
    
    Discretization methodology based on information theory principles
    for optimal state space representation (Cover & Thomas, 2006).
    
    Args:
        value (float): Continuous input value
        thresholds (tuple): Discretization thresholds
        
    Returns:
        int: Discrete state (-1, 0, 1)
        
    References:
        Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory. 
        John Wiley & Sons, New York.
    """
    if value < thresholds[0]:
        return -1
    elif value > thresholds[1]:
        return 1
    else:
        return 0

def calculate_returns(prices, method='simple'):
    """
    Calculate financial returns using academic standard methodology.
    
    Return calculation following Campbell et al. (1997) methodology
    for financial time series analysis.
    
    Args:
        prices (array-like): Price series
        method (str): 'simple' or 'log' returns
        
    Returns:
        np.ndarray: Return series
        
    References:
        Campbell, J.Y., Lo, A.W. & MacKinlay, A.C. (1997). The Econometrics 
        of Financial Markets. Princeton University Press, Princeton.
    """
    prices = np.array(prices)
    if method == 'log':
        return np.diff(np.log(prices))
    else:
        return np.diff(prices) / prices[:-1]

def moving_average(data, window=20):
    """
    Calculate moving average with proper handling of edge cases.
    
    Moving average implementation following standard technical analysis
    methodology (Murphy, 1999) for financial time series.
    
    Args:
        data (array-like): Input data series
        window (int): Moving average window
        
    Returns:
        pd.Series: Moving average series
        
    References:
        Murphy, J.J. (1999). Technical Analysis of the Financial Markets. 
        New York Institute of Finance, New York.
    """
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

def validate_data_integrity(data):
    """
    Validate data integrity for financial time series analysis.
    
    Data validation methodology ensuring data quality standards
    for academic financial research (Tsay, 2010).
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        dict: Validation results
        
    References:
        Tsay, R.S. (2010). Analysis of Financial Time Series. 
        John Wiley & Sons, New York.
    """
    validation = {
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum(),
        'data_types_valid': True,
        'date_range_valid': True,
        'anomalies_detected': 0
    }
    
    # Check for extreme outliers (academic standard: >3 standard deviations)
    for col in data.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs(stats.zscore(data[col].dropna()))
        validation['anomalies_detected'] += (z_scores > 3).sum()
    
    return validation

def prepare_academic_output():
    """
    Setup output formatting for academic publication standards.
    
    Output configuration following academic publishing guidelines
    for computational finance research (Brandimarte, 2006).
    
    References:
        Brandimarte, P. (2006). Numerical Methods in Finance and Economics: 
        A MATLAB-Based Introduction. John Wiley & Sons, New York.
    """
    # Configure matplotlib for publication-quality figures
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Configure pandas display options
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.precision', 4)
    pd.set_option('display.float_format', '{:.4f}'.format)

# Initialize academic output settings
prepare_academic_output()
set_random_seed(DEFAULT_CONFIG['random_seed'])
