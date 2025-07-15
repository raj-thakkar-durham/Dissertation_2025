# Agent Classes for Hybrid Gold Price Prediction
# Implements agent-based modeling as per instruction manual Phase 3.1
# Research purposes only - academic dissertation

from mesa import Agent
import numpy as np
import random
from abc import ABC, abstractmethod

class GoldTrader(Agent):
    """
    Base class for gold trading agents
    As specified in instruction manual Step 3.1
    
    Reference: Instruction manual Phase 3.1 - Agent Classes
    """
    
    def __init__(self, unique_id, model, agent_type='contrarian'):
        """
        Initialize gold trader agent
        
        Args:
            unique_id (int): Unique agent identifier
            model: Mesa model instance
            agent_type (str): Type of agent
            
        Reference: Instruction manual - "def __init__(self, unique_id, model, agent_type='contrarian'):"
        """
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.cash = 10000
        self.gold_holdings = 0
        self.risk_tolerance = random.uniform(0.1, 0.9)
        self.sentiment_bias = random.uniform(-0.5, 0.5)
        
        # Trading history
        self.trade_history = []
        self.pnl_history = []
        self.position_history = []
        
        # Decision parameters
        self.last_trade_price = 0
        self.trades_count = 0
        self.profitable_trades = 0
        
    def observe_environment(self):
        """
        Observe CA signals and neighbor actions
        Return dictionary of observations
        
        Returns:
            dict: Environment observations
            
        Reference: Instruction manual - "Observe CA signals and neighbor actions"
        """
        try:
            observations = {}
            
            # Get CA signal from model
            if hasattr(self.model, 'ca_model'):
                observations['ca_signal'] = self.model.ca_model.get_market_signal()
                observations['ca_stats'] = self.model.ca_model.get_grid_statistics()
            else:
                observations['ca_signal'] = 0.0
                observations['ca_stats'] = {}
            
            # Get current market price
            observations['current_price'] = self.model.current_price
            
            # Get price history
            if hasattr(self.model, 'price_history') and len(self.model.price_history) > 0:
                observations['price_history'] = self.model.price_history[-10:]  # Last 10 prices
                
                # Calculate price trend
                if len(observations['price_history']) > 1:
                    recent_return = (observations['price_history'][-1] - 
                                   observations['price_history'][-2]) / observations['price_history'][-2]
                    observations['recent_return'] = recent_return
                else:
                    observations['recent_return'] = 0.0
            else:
                observations['price_history'] = []
                observations['recent_return'] = 0.0
            
            # Observe neighbor actions
            neighbor_positions = []
            if hasattr(self.model, 'grid'):
                neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
                for neighbor in neighbors:
                    if hasattr(neighbor, 'position'):
                        neighbor_positions.append(neighbor.position)
            
            observations['neighbor_positions'] = neighbor_positions
            observations['neighbor_avg_position'] = np.mean(neighbor_positions) if neighbor_positions else 0
            
            # Market sentiment
            observations['market_sentiment'] = self.model.external_data.get('sentiment', 0) if self.model.external_data else 0
            
            # Volatility
            if len(observations['price_history']) > 5:
                returns = np.diff(observations['price_history']) / observations['price_history'][:-1]
                observations['volatility'] = np.std(returns)
            else:
                observations['volatility'] = 0.02  # Default volatility
            
            return observations
            
        except Exception as e:
            print(f"Error observing environment for agent {self.unique_id}: {e}")
            return {}
    
    @abstractmethod
    def make_decision(self, observations):
        """
        Decision logic based on agent type and observations
        Return action: 'buy', 'sell', or 'hold'
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action to take
            
        Reference: Instruction manual - "Decision logic based on agent type and observations"
        """
        pass
    
    def execute_trade(self, action, price):
        """
        Execute the decided action
        Update position, cash, and holdings
        
        Args:
            action (str): Action to execute
            price (float): Current market price
            
        Reference: Instruction manual - "Execute the decided action"
        """
        try:
            trade_size = self.calculate_trade_size(action, price)
            
            if action == 'buy' and trade_size > 0:
                # Buy gold
                cost = trade_size * price
                if cost <= self.cash:
                    self.cash -= cost
                    self.gold_holdings += trade_size
                    self.position = 1 if self.gold_holdings > 0 else 0
                    self.last_trade_price = price
                    self.trades_count += 1
                    
                    # Record trade
                    self.trade_history.append({
                        'action': 'buy',
                        'price': price,
                        'quantity': trade_size,
                        'timestamp': self.model.current_day
                    })
                    
            elif action == 'sell' and trade_size > 0:
                # Sell gold
                if trade_size <= self.gold_holdings:
                    self.cash += trade_size * price
                    self.gold_holdings -= trade_size
                    self.position = 1 if self.gold_holdings > 0 else 0
                    
                    # Calculate P&L
                    if self.last_trade_price > 0:
                        pnl = (price - self.last_trade_price) * trade_size
                        self.pnl_history.append(pnl)
                        if pnl > 0:
                            self.profitable_trades += 1
                    
                    self.trades_count += 1
                    
                    # Record trade
                    self.trade_history.append({
                        'action': 'sell',
                        'price': price,
                        'quantity': trade_size,
                        'timestamp': self.model.current_day
                    })
            
            # Update position history
            self.position_history.append(self.position)
            
        except Exception as e:
            print(f"Error executing trade for agent {self.unique_id}: {e}")
    
    def calculate_trade_size(self, action, price):
        """
        Calculate trade size based on agent characteristics
        
        Args:
            action (str): Intended action
            price (float): Current price
            
        Returns:
            float: Trade size
        """
        try:
            # Base trade size as percentage of wealth
            wealth = self.cash + self.gold_holdings * price
            base_size = wealth * 0.1 * self.risk_tolerance
            
            if action == 'buy':
                # Maximum we can buy
                max_buy = self.cash / price
                return min(base_size / price, max_buy)
                
            elif action == 'sell':
                # Maximum we can sell
                return min(base_size / price, self.gold_holdings)
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def get_portfolio_value(self, current_price):
        """
        Calculate current portfolio value
        
        Args:
            current_price (float): Current gold price
            
        Returns:
            float: Portfolio value
        """
        return self.cash + self.gold_holdings * current_price
    
    def step(self):
        """
        Agent's step function called each simulation day
        
        Reference: Instruction manual - "Agent's step function called each simulation day"
        """
        try:
            # Observe environment
            observations = self.observe_environment()
            
            # Make decision
            action = self.make_decision(observations)
            
            # Execute trade
            if action in ['buy', 'sell']:
                self.execute_trade(action, self.model.current_price)
                
        except Exception as e:
            print(f"Error in agent {self.unique_id} step: {e}")


class HerderAgent(GoldTrader):
    """
    Agent that follows majority of neighbors
    
    Reference: Instruction manual - "class HerderAgent(GoldTrader):"
    """
    
    def __init__(self, unique_id, model):
        """
        Initialize herder agent
        
        Reference: Instruction manual - "def __init__(self, unique_id, model):"
        """
        super().__init__(unique_id, model, agent_type='herder')
        self.herd_strength = random.uniform(0.6, 0.9)
        
    def make_decision(self, observations):
        """
        Follow majority of neighbors
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action decision
            
        Reference: Instruction manual - "Follow majority of neighbors"
        """
        try:
            # Weight factors
            ca_weight = 0.3
            neighbor_weight = 0.4
            sentiment_weight = 0.3
            
            # CA signal influence
            ca_signal = observations.get('ca_signal', 0)
            ca_influence = ca_signal * ca_weight
            
            # Neighbor influence
            neighbor_avg = observations.get('neighbor_avg_position', 0)
            neighbor_influence = neighbor_avg * neighbor_weight * self.herd_strength
            
            # Sentiment influence
            sentiment = observations.get('market_sentiment', 0)
            sentiment_influence = sentiment * sentiment_weight
            
            # Combine influences
            total_influence = ca_influence + neighbor_influence + sentiment_influence
            
            # Add some randomness
            total_influence += random.uniform(-0.1, 0.1)
            
            # Decision thresholds
            if total_influence > 0.2:
                return 'buy'
            elif total_influence < -0.2:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            print(f"Error in herder decision making: {e}")
            return 'hold'


class ContrarianAgent(GoldTrader):
    """
    Agent that acts opposite to majority
    
    Reference: Instruction manual - "class ContrarianAgent(GoldTrader):"
    """
    
    def __init__(self, unique_id, model):
        """
        Initialize contrarian agent
        
        Reference: Instruction manual - "def __init__(self, unique_id, model):"
        """
        super().__init__(unique_id, model, agent_type='contrarian')
        self.contrarian_strength = random.uniform(0.5, 0.8)
        
    def make_decision(self, observations):
        """
        Act opposite to majority
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action decision
            
        Reference: Instruction manual - "Act opposite to majority"
        """
        try:
            # Weight factors
            ca_weight = 0.4
            neighbor_weight = 0.3
            volatility_weight = 0.3
            
            # CA signal influence (contrarian)
            ca_signal = observations.get('ca_signal', 0)
            ca_influence = -ca_signal * ca_weight * self.contrarian_strength
            
            # Neighbor influence (contrarian)
            neighbor_avg = observations.get('neighbor_avg_position', 0)
            neighbor_influence = -neighbor_avg * neighbor_weight * self.contrarian_strength
            
            # Volatility influence (contrarians like volatility)
            volatility = observations.get('volatility', 0)
            volatility_influence = volatility * volatility_weight * 10  # Scale up volatility
            
            # Combine influences
            total_influence = ca_influence + neighbor_influence + volatility_influence
            
            # Add randomness
            total_influence += random.uniform(-0.1, 0.1)
            
            # Decision thresholds
            if total_influence > 0.25:
                return 'buy'
            elif total_influence < -0.25:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            print(f"Error in contrarian decision making: {e}")
            return 'hold'


class TrendFollowerAgent(GoldTrader):
    """
    Agent that follows price trends and momentum
    
    Reference: Instruction manual - "class TrendFollowerAgent(GoldTrader):"
    """
    
    def __init__(self, unique_id, model):
        """
        Initialize trend follower agent
        
        Reference: Instruction manual - "def __init__(self, unique_id, model):"
        """
        super().__init__(unique_id, model, agent_type='trend_follower')
        self.trend_sensitivity = random.uniform(0.3, 0.7)
        self.momentum_threshold = random.uniform(0.01, 0.03)
        
    def make_decision(self, observations):
        """
        Follow price trends and momentum
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action decision
            
        Reference: Instruction manual - "Follow price trends and momentum"
        """
        try:
            # Weight factors
            trend_weight = 0.5
            ca_weight = 0.3
            momentum_weight = 0.2
            
            # Trend influence
            recent_return = observations.get('recent_return', 0)
            trend_influence = recent_return * trend_weight * self.trend_sensitivity
            
            # CA signal influence
            ca_signal = observations.get('ca_signal', 0)
            ca_influence = ca_signal * ca_weight
            
            # Momentum influence
            price_history = observations.get('price_history', [])
            momentum_influence = 0
            if len(price_history) >= 3:
                # Calculate momentum as average of recent returns
                recent_returns = []
                for i in range(len(price_history) - 3, len(price_history)):
                    if i > 0:
                        ret = (price_history[i] - price_history[i-1]) / price_history[i-1]
                        recent_returns.append(ret)
                
                if recent_returns:
                    momentum = np.mean(recent_returns)
                    if abs(momentum) > self.momentum_threshold:
                        momentum_influence = momentum * momentum_weight
            
            # Combine influences
            total_influence = trend_influence + ca_influence + momentum_influence
            
            # Add randomness
            total_influence += random.uniform(-0.05, 0.05)
            
            # Decision thresholds
            if total_influence > 0.15:
                return 'buy'
            elif total_influence < -0.15:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            print(f"Error in trend follower decision making: {e}")
            return 'hold'


class NoiseTraderAgent(GoldTrader):
    """
    Agent that makes random trades with some market awareness
    """
    
    def __init__(self, unique_id, model):
        """Initialize noise trader agent"""
        super().__init__(unique_id, model, agent_type='noise_trader')
        self.noise_level = random.uniform(0.7, 1.0)
        
    def make_decision(self, observations):
        """
        Make mostly random decisions with slight market awareness
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action decision
        """
        try:
            # Mostly random with slight market influence
            random_factor = random.uniform(-1, 1) * self.noise_level
            
            # Small influence from market
            ca_signal = observations.get('ca_signal', 0) * 0.1
            sentiment = observations.get('market_sentiment', 0) * 0.1
            
            total_influence = random_factor + ca_signal + sentiment
            
            # Decision thresholds
            if total_influence > 0.3:
                return 'buy'
            elif total_influence < -0.3:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            return 'hold'


# Example usage and testing
if __name__ == "__main__":
    # This would normally be run within the Mesa model framework
    print("Agent classes defined successfully!")
    
    # Create mock model for testing
    class MockModel:
        def __init__(self):
            self.current_price = 1800
            self.current_day = 0
            self.external_data = {'sentiment': 0.1}
            self.price_history = [1795, 1798, 1800]
    
    # Test agent creation
    mock_model = MockModel()
    
    # Create different agent types
    herder = HerderAgent(1, mock_model)
    contrarian = ContrarianAgent(2, mock_model)
    trend_follower = TrendFollowerAgent(3, mock_model)
    noise_trader = NoiseTraderAgent(4, mock_model)
    
    print(f"Created agents: {herder.agent_type}, {contrarian.agent_type}, "
          f"{trend_follower.agent_type}, {noise_trader.agent_type}")
    
    # Test observation
    observations = herder.observe_environment()
    print(f"Sample observations: {list(observations.keys())}")
    
    # Test decision making
    for agent in [herder, contrarian, trend_follower, noise_trader]:
        decision = agent.make_decision(observations)
        print(f"{agent.agent_type} decision: {decision}")
    
    print("Agent testing completed successfully!")
