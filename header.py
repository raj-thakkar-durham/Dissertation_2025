# Hybrid Gold Price Prediction Simulation - Academic Research Implementation
# Author: Research Student - Durham University Dissertation 2025
# Methodology: Cellular Automata + Agent-Based Modeling for Financial Markets
# Citations: Wolfram (2002), Bonabeau (2002), Arthur et al. (1997)

import pandas as pd
import numpy as np
import random
import multiprocessing as mp
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import differential_evolution

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa import batch_run

from joblib import Parallel, delayed
from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_CONFIG = {
    'start_date': '2014-01-01',
    'end_date': '2024-01-01',
    'grid_size': (50, 50),
    'num_agents': 100,
    'num_states': 3,
    'initial_gold_price': 1800,
    'simulation_days': 252,
    'num_parallel_runs': 100,
    'random_seed': 42
}

def set_random_seed(seed=42):
    """Set random seed for reproducibility (Knuth, 1997)"""
    random.seed(seed)
    np.random.seed(seed)

def discretize_signal(value, thresholds=[-0.33, 0.33]):
    """Convert continuous signal to discrete state based on Wolfram (2002)"""
    if value < thresholds[0]:
        return -1
    elif value > thresholds[1]:
        return 1
    else:
        return 0

def calculate_returns(prices):
    """Calculate log returns from price series (Tsay, 2005)"""
    return np.log(prices / prices.shift(1))

def moving_average(data, window):
    """Calculate moving average with proper handling of NaN values"""
    return data.rolling(window=window, min_periods=1).mean()

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
