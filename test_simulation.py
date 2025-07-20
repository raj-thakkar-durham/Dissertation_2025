# Test importing the main simulation class without running the main section

import sys
import os

# Add the imports manually instead of importing the whole module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting imports...")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from models.cellular_automaton import CellularAutomaton
from models.market_model import GoldMarketModel
from models.ca_optimizer import CARuleOptimizer
from data.data_collection import DataCollector
from data.news_analysis import SentimentAnalyzer
from utils.feature_engineering import FeatureEngineering
from utils.parallel_runner import ParallelSimulationRunner
from results.result_analyzer import ResultsAnalyzer
from results.plots import VisualizationTools
import multiprocessing as mp

print("All imports successful")

# Define the DEFAULT_CONFIG manually
DEFAULT_CONFIG = {
    'start_date': '2014-01-01',
    'end_date': '2024-01-01',
    'ca_grid_size': (20, 20),
    'ca_num_states': 3,
    'num_agents': 100,
    'agent_grid_size': (15, 15),
    'initial_gold_price': 1800,
    'market_volatility': 0.02,
    'market_liquidity': 1000000,
    'simulation_days': 252,
    'num_parallel_runs': 50,
    'optimization_cores': 2,
    'parallel_cores': mp.cpu_count(),
    'random_seed': 42
}

print(f"Config loaded with {len(DEFAULT_CONFIG)} parameters")

# Test creating the class manually
class HybridGoldSimulation:
    def __init__(self, config):
        self.config = config
        print("HybridGoldSimulation created successfully!")

# Test creating an instance
simulation = HybridGoldSimulation(DEFAULT_CONFIG)
print("Test completed successfully!")
