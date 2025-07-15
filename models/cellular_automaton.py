# Cellular Automaton Implementation for Hybrid Gold Price Prediction
# Implements CA as per instruction manual Phase 2.1
# Research purposes only - academic dissertation

import numpy as np
from numba import jit
import pandas as pd
from typing import Dict, Tuple, List

class CellularAutomaton:
    """
    Cellular Automaton implementation for market sentiment modeling
    As specified in instruction manual Step 2.1
    
    Reference: Instruction manual Phase 2.1 - CA Grid Structure
    """
    
    def __init__(self, grid_size=(50, 50), num_states=3):
        """
        Initialize cellular automaton
        
        Args:
            grid_size (tuple): Size of the CA grid
            num_states (int): Number of states (e.g., -1, 0, 1 for bearish, neutral, bullish)
            
        Reference: Instruction manual - "def __init__(self, grid_size=(50, 50), num_states=3):"
        """
        self.grid_size = grid_size
        self.num_states = num_states  # e.g., -1, 0, 1 for bearish, neutral, bullish
        self.grid = np.zeros(grid_size, dtype=int)
        self.rules = {}
        self.history = []
        
        print(f"CellularAutomaton initialized with grid size {grid_size} and {num_states} states")
        
    def initialize_grid(self, initial_state):
        """
        Initialize grid based on market features
        Map features to discrete states (-1, 0, 1)
        
        Args:
            initial_state (dict or np.ndarray): Initial state configuration
            
        Reference: Instruction manual - "Initialize grid based on market features"
        """
        try:
            if isinstance(initial_state, dict):
                # Initialize based on market features
                sentiment = initial_state.get('sentiment', 0)
                volatility = initial_state.get('volatility', 0)
                trend = initial_state.get('trend', 0)
                
                # Map continuous values to discrete states
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        # Combine multiple factors with some randomness
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
                # Direct initialization with array
                self.grid = initial_state.copy()
                
            else:
                # Random initialization
                self.grid = np.random.choice([-1, 0, 1], size=self.grid_size)
            
            print(f"Grid initialized with {np.sum(self.grid == 1)} bullish, "
                  f"{np.sum(self.grid == 0)} neutral, {np.sum(self.grid == -1)} bearish cells")
            
        except Exception as e:
            print(f"Error initializing grid: {e}")
            self.grid = np.zeros(self.grid_size, dtype=int)
    
    @staticmethod
    @jit(nopython=True)
    def get_neighbors(grid, x, y):
        """
        Get 8-neighborhood values for cell at (x, y)
        Handle boundary conditions
        
        Args:
            grid (np.ndarray): Current grid state
            x (int): Row index
            y (int): Column index
            
        Returns:
            np.ndarray: 8-neighborhood values
            
        Reference: Instruction manual - "Get 8-neighborhood values for cell at (x, y)"
        """
        rows, cols = grid.shape
        neighbors = np.zeros(8, dtype=np.int32)
        
        # 8-neighborhood offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i, (dx, dy) in enumerate(offsets):
            # Handle boundary conditions with periodic boundary
            nx = (x + dx) % rows
            ny = (y + dy) % cols
            neighbors[i] = grid[nx, ny]
        
        return neighbors
    
    def define_rules(self, rule_parameters):
        """
        Define CA update rules based on neighbor configurations
        Rules map 8-neighbor pattern to new center state
        
        Args:
            rule_parameters (dict): Parameters defining the CA rules
            
        Reference: Instruction manual - "Define CA update rules based on neighbor configurations"
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
            
            # Update with provided parameters
            self.rules = {**default_params, **rule_parameters}
            
            print(f"CA rules defined with parameters: {self.rules}")
            
        except Exception as e:
            print(f"Error defining rules: {e}")
            self.rules = {}
    
    def update_cell(self, x, y, external_features=None):
        """
        Update single cell based on neighbors and external inputs
        
        Args:
            x (int): Row index
            y (int): Column index
            external_features (dict): External market features
            
        Returns:
            int: New state for cell
            
        Reference: Instruction manual - "Update single cell based on neighbors and external inputs"
        """
        try:
            # Get current state and neighbors
            current_state = self.grid[x, y]
            neighbors = self.get_neighbors(self.grid, x, y)
            
            # Calculate neighbor influence
            neighbor_sum = np.sum(neighbors)
            neighbor_mean = neighbor_sum / 8.0
            
            # Apply rules
            if not self.rules:
                # Simple majority rule if no rules defined
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
            
        except Exception as e:
            print(f"Error updating cell at ({x}, {y}): {e}")
            return self.grid[x, y]
    
    def step(self, external_features=None):
        """
        Update entire grid for one time step
        Use vectorized operations for efficiency
        
        Args:
            external_features (dict): External market features
            
        Reference: Instruction manual - "Update entire grid for one time step"
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
            
            # Store history
            self.history.append(self.grid.copy())
            
            # Keep only last 100 steps in history
            if len(self.history) > 100:
                self.history.pop(0)
            
        except Exception as e:
            print(f"Error in CA step: {e}")
    
    def get_market_signal(self):
        """
        Aggregate grid state into market signal
        Return value between -1 and 1
        
        Returns:
            float: Market signal between -1 and 1
            
        Reference: Instruction manual - "Aggregate grid state into market signal"
        """
        try:
            # Calculate aggregate signal
            total_cells = self.grid_size[0] * self.grid_size[1]
            bullish_cells = np.sum(self.grid == 1)
            bearish_cells = np.sum(self.grid == -1)
            
            # Market signal as proportion difference
            market_signal = (bullish_cells - bearish_cells) / total_cells
            
            return market_signal
            
        except Exception as e:
            print(f"Error calculating market signal: {e}")
            return 0.0
    
    def get_grid_statistics(self):
        """
        Get current grid statistics
        
        Returns:
            dict: Grid statistics
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
            
        except Exception as e:
            print(f"Error calculating grid statistics: {e}")
            return {}
    
    def calculate_entropy(self):
        """
        Calculate entropy of current grid state
        
        Returns:
            float: Entropy value
        """
        try:
            total_cells = self.grid_size[0] * self.grid_size[1]
            states, counts = np.unique(self.grid, return_counts=True)
            probabilities = counts / total_cells
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return entropy
            
        except Exception as e:
            return 0.0
    
    def visualize_grid(self, title="CA Grid State"):
        """
        Visualize current grid state
        
        Args:
            title (str): Plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create color map
            colors = ['red', 'gray', 'green']  # bearish, neutral, bullish
            cmap = plt.matplotlib.colors.ListedColormap(colors)
            
            # Plot grid
            im = ax.imshow(self.grid, cmap=cmap, vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
            cbar.set_ticklabels(['Bearish', 'Neutral', 'Bullish'])
            
            ax.set_title(title)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing grid: {e}")
    
    def save_grid_state(self, filename):
        """
        Save current grid state
        
        Args:
            filename (str): Output filename
        """
        try:
            np.save(filename, self.grid)
            print(f"Grid state saved to {filename}")
        except Exception as e:
            print(f"Error saving grid state: {e}")
    
    def load_grid_state(self, filename):
        """
        Load grid state from file
        
        Args:
            filename (str): Input filename
        """
        try:
            self.grid = np.load(filename)
            print(f"Grid state loaded from {filename}")
        except Exception as e:
            print(f"Error loading grid state: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize CA
    ca = CellularAutomaton(grid_size=(20, 20), num_states=3)
    
    # Define initial state
    initial_state = {
        'sentiment': 0.2,
        'volatility': 0.1,
        'trend': 0.3
    }
    
    # Initialize grid
    ca.initialize_grid(initial_state)
    
    # Define rules
    rule_params = {
        'bullish_threshold': 0.3,
        'bearish_threshold': -0.3,
        'momentum_weight': 0.4,
        'contrarian_weight': 0.3,
        'random_weight': 0.3,
        'external_weight': 0.5
    }
    ca.define_rules(rule_params)
    
    # Run simulation
    print("Running CA simulation...")
    external_features = {
        'sentiment': 0.1,
        'volatility': 0.05,
        'trend': 0.2
    }
    
    for step in range(10):
        ca.step(external_features)
        stats = ca.get_grid_statistics()
        print(f"Step {step+1}: Market Signal = {stats['market_signal']:.3f}, "
              f"Entropy = {stats['grid_entropy']:.3f}")
    
    # Final statistics
    final_stats = ca.get_grid_statistics()
    print("\nFinal Grid Statistics:")
    for key, value in final_stats.items():
        print(f"{key}: {value}")
