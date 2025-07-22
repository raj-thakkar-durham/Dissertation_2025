# Cellular Automaton Implementation for Hybrid Gold Price Prediction
# Academic Implementation for Dissertation Research

"""
Cellular automaton implementation for market sentiment modeling
based on Wolfram's cellular automata principles for complex systems.

Citations:
[1] Wolfram, S. (2002). A New Kind of Science. Wolfram Media, Champaign, IL.
[2] von Neumann, J. (1966). Theory of Self-Reproducing Automata. University of 
    Illinois Press, Urbana, IL.
[3] Gardner, M. (1970). Mathematical Games: The fantastic combinations of John 
    Conway's new solitaire game "life". Scientific American, 223(4), 120-123.
[4] Packard, N.H. & Wolfram, S. (1985). Two-dimensional cellular automata. 
    Journal of Statistical Physics, 38(5-6), 901-946.
"""

import numpy as np
from numba import jit
import pandas as pd
from typing import Dict, Tuple, List

class CellularAutomaton:
    """
    Cellular automaton implementation for market sentiment modeling.
    
    Implementation based on Wolfram (2002) cellular automata principles
    adapted for financial market complex systems modeling.
    
    References:
        Wolfram, S. (2002). A New Kind of Science. Wolfram Media, Champaign, IL.
        
        Packard, N.H. & Wolfram, S. (1985). Two-dimensional cellular automata. 
        Journal of Statistical Physics, 38(5-6), 901-946.
    """

    def __init__(self, grid_size=(50, 50), num_states=3):
        """
        Initialize cellular automaton with specified grid dimensions.
        
        Args:
            grid_size (tuple): Size of the CA grid
            num_states (int): Number of states (-1, 0, 1 for bearish, neutral, bullish)
        """
        self.grid_size = grid_size
        self.num_states = num_states
        self.grid = np.zeros(grid_size, dtype=int)
        self.rules = {}
        self.history = []

    def initialize_grid(self, initial_state):
        """
        Initialize grid based on market features using von Neumann approach.
        
        Grid initialization methodology based on von Neumann (1966) 
        self-reproducing automata principles adapted for market sentiment.
        
        Args:
            initial_state (dict or np.ndarray): Initial state configuration
            
        References:
            von Neumann, J. (1966). Theory of Self-Reproducing Automata. 
            University of Illinois Press, Urbana, IL.
        """
        try:
            if isinstance(initial_state, dict):
                # Market feature-based initialization
                sentiment = initial_state.get('sentiment', 0)
                volatility = initial_state.get('volatility', 0)
                trend = initial_state.get('trend', 0)

                # Map continuous values to discrete states
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        # Combine multiple factors with spatial variation
                        combined_signal = (sentiment * 0.4 + trend * 0.4 + 
                                         volatility * 0.2 + np.random.normal(0, 0.1))
                        
                        # Discretize to states
                        if combined_signal < -0.33:
                            self.grid[i, j] = -1  # Bearish
                        elif combined_signal > 0.33:
                            self.grid[i, j] = 1   # Bullish
                        else:
                            self.grid[i, j] = 0   # Neutral

            elif isinstance(initial_state, np.ndarray):
                self.grid = initial_state.copy()
            else:
                # Random initialization
                self.grid = np.random.choice([-1, 0, 1], size=self.grid_size)

        except Exception:
            self.grid = np.zeros(self.grid_size, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def get_neighbors(grid, x, y):
        """
        Get 8-neighborhood values for cell at (x, y) with periodic boundaries.
        
        Neighborhood calculation implementing Gardner (1970) Conway's Life
        methodology adapted for market sentiment propagation.
        
        Args:
            grid (np.ndarray): Current grid state
            x (int): Row index
            y (int): Column index
            
        Returns:
            np.ndarray: 8-neighborhood values
            
        References:
            Gardner, M. (1970). Mathematical Games: The fantastic combinations 
            of John Conway's new solitaire game "life". Scientific American, 223(4), 120-123.
        """
        rows, cols = grid.shape
        neighbors = np.zeros(8, dtype=np.int32)
        
        # 8-neighborhood offsets (Moore neighborhood)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i, (dx, dy) in enumerate(offsets):
            # Periodic boundary conditions
            nx = (x + dx) % rows
            ny = (y + dy) % cols
            neighbors[i] = grid[nx, ny]
            
        return neighbors

    def define_rules(self, rule_parameters):
        """
        Define CA update rules based on neighbor configurations.
        
        Rule definition methodology based on Wolfram (2002) elementary
        cellular automata extended to two-dimensional market systems.
        
        Args:
            rule_parameters (dict): Parameters defining the CA rules
            
        References:
            Wolfram, S. (2002). A New Kind of Science. Wolfram Media, Champaign, IL.
        """
        try:
            # Default rule parameters
            default_params = {
                'bullish_threshold': 0.3,
                'bearish_threshold': -0.3,
                'momentum_weight': 0.4,
                'contrarian_weight': 0.3,
                'random_weight': 0.3,
                'external_weight': 0.5
            }
            
            self.rules = {**default_params, **rule_parameters}
        except Exception:
            self.rules = {}

    def update_cell(self, x, y, external_features=None):
        """
        Update single cell based on neighbors and external inputs.
        
        Cell update mechanism implementing Wolfram (2002) local rule
        application with external market feature integration.
        
        Args:
            x (int): Row index
            y (int): Column index
            external_features (dict): External market features
            
        Returns:
            int: New state for cell
            
        References:
            Wolfram, S. (2002). A New Kind of Science. Wolfram Media, Champaign, IL.
        """
        try:
            current_state = self.grid[x, y]
            neighbors = self.get_neighbors(self.grid, x, y)

            # Calculate neighbor influence
            neighbor_sum = np.sum(neighbors)
            neighbor_mean = neighbor_sum / 8.0

            # Apply rules
            if not self.rules:
                # Simple majority rule
                if neighbor_sum > 2:
                    new_state = 1
                elif neighbor_sum < -2:
                    new_state = -1
                else:
                    new_state = 0
            else:
                # Complex rule based on parameters
                momentum_signal = neighbor_mean * self.rules['momentum_weight']
                contrarian_signal = -neighbor_mean * self.rules['contrarian_weight']
                random_signal = np.random.normal(0, 0.1) * self.rules['random_weight']

                # External influence
                external_signal = 0
                if external_features:
                    external_signal = (external_features.get('sentiment', 0) * 0.3 +
                                     external_features.get('volatility', 0) * 0.2 +
                                     external_features.get('trend', 0) * 0.5)
                    external_signal *= self.rules['external_weight']

                # Combine signals
                total_signal = momentum_signal + contrarian_signal + random_signal + external_signal

                # Discretize
                if total_signal > self.rules['bullish_threshold']:
                    new_state = 1
                elif total_signal < self.rules['bearish_threshold']:
                    new_state = -1
                else:
                    new_state = 0

            return new_state
        except Exception:
            return self.grid[x, y]

    def step(self, external_features=None):
        """
        Update entire grid for one time step using synchronous update.
        
        Grid evolution implementing Packard & Wolfram (1985) synchronous
        update mechanism for two-dimensional cellular automata.
        
        Args:
            external_features (dict): External market features
            
        References:
            Packard, N.H. & Wolfram, S. (1985). Two-dimensional cellular automata. 
            Journal of Statistical Physics, 38(5-6), 901-946.
        """
        try:
            # Create new grid for simultaneous update
            new_grid = np.zeros_like(self.grid)

            # Update each cell
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    new_grid[i, j] = self.update_cell(i, j, external_features)

            # Update grid
            self.grid = new_grid

            # Store history (limited to last 100 steps)
            self.history.append(self.grid.copy())
            if len(self.history) > 100:
                self.history.pop(0)

        except Exception:
            pass

    def get_market_signal(self):
        """
        Aggregate grid state into market signal between -1 and 1.
        
        Signal aggregation based on Wolfram (2002) global behavior
        emergence from local cellular interactions.
        
        Returns:
            float: Market signal between -1 and 1
            
        References:
            Wolfram, S. (2002). A New Kind of Science. Wolfram Media, Champaign, IL.
        """
        try:
            total_cells = self.grid_size[0] * self.grid_size[1]
            bullish_cells = np.sum(self.grid == 1)
            bearish_cells = np.sum(self.grid == -1)
            
            # Market signal as proportion difference
            market_signal = (bullish_cells - bearish_cells) / total_cells
            return market_signal
        except Exception:
            return 0.0

    def get_grid_statistics(self):
        """
        Get comprehensive grid statistics for analysis.
        
        Statistical analysis implementing information-theoretic measures
        for cellular automaton state characterization.
        
        Returns:
            dict: Grid statistics including entropy measures
        """
        try:
            total_cells = self.grid_size[0] * self.grid_size[1]
            bullish_cells = np.sum(self.grid == 1)
            neutral_cells = np.sum(self.grid == 0)
            bearish_cells = np.sum(self.grid == -1)

            stats = {
                'total_cells': total_cells,
                'bullish_cells': bullish_cells,
                'neutral_cells': neutral_cells,
                'bearish_cells': bearish_cells,
                'bullish_ratio': bullish_cells / total_cells,
                'bearish_ratio': bearish_cells / total_cells,
                'neutral_ratio': neutral_cells / total_cells,
                'market_signal': self.get_market_signal(),
                'grid_entropy': self.calculate_entropy()
            }
            
            return stats
        except Exception:
            return {}

    def calculate_entropy(self):
        """
        Calculate Shannon entropy of current grid state.
        
        Entropy calculation for measuring information content and
        complexity of cellular automaton configurations.
        
        Returns:
            float: Shannon entropy value
        """
        try:
            total_cells = self.grid_size[0] * self.grid_size[1]
            states, counts = np.unique(self.grid, return_counts=True)
            probabilities = counts / total_cells

            # Calculate Shannon entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        except Exception:
            return 0.0

    def run_steps(self, num_steps, external_data=None):
        """
        Run multiple CA evolution steps with external data integration.
        
        Multi-step evolution for long-term CA behavior analysis
        with market data integration capability.
        
        Args:
            num_steps (int): Number of evolution steps
            external_data (list): External features for each step
            
        Returns:
            list: History of market signals
        """
        try:
            signals = []
            
            for step in range(num_steps):
                # Get external features for current step
                features = None
                if external_data and step < len(external_data):
                    features = external_data[step]
                
                # Evolve CA
                self.step(features)
                
                # Record market signal
                signal = self.get_market_signal()
                signals.append(signal)
            
            return signals
        except Exception:
            return []
