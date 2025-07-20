# Test imports one by one to find the issue

print("Testing basic imports...")
import pandas as pd
import numpy as np
import sys
import os
print("Basic imports OK")

print("Testing project imports...")
try:
    from models.cellular_automaton import CellularAutomaton
    print("CellularAutomaton OK")
except Exception as e:
    print(f"CellularAutomaton failed: {e}")

try:
    from models.market_model import GoldMarketModel
    print("GoldMarketModel OK")
except Exception as e:
    print(f"GoldMarketModel failed: {e}")

try:
    from models.ca_optimizer import CARuleOptimizer
    print("CARuleOptimizer OK")
except Exception as e:
    print(f"CARuleOptimizer failed: {e}")

try:
    from data.data_collection import DataCollector
    print("DataCollector OK")
except Exception as e:
    print(f"DataCollector failed: {e}")

try:
    from data.news_analysis import SentimentAnalyzer
    print("SentimentAnalyzer OK")
except Exception as e:
    print(f"SentimentAnalyzer failed: {e}")

try:
    from utils.feature_engineering import FeatureEngineering
    print("FeatureEngineering OK")
except Exception as e:
    print(f"FeatureEngineering failed: {e}")

try:
    from utils.parallel_runner import ParallelSimulationRunner
    print("ParallelSimulationRunner OK")
except Exception as e:
    print(f"ParallelSimulationRunner failed: {e}")

try:
    from results.result_analyzer import ResultsAnalyzer
    print("ResultsAnalyzer OK")
except Exception as e:
    print(f"ResultsAnalyzer failed: {e}")

try:
    from results.plots import VisualizationTools
    print("VisualizationTools OK")
except Exception as e:
    print(f"VisualizationTools failed: {e}")

print("All imports tested!")
