# Agent Classes for Hybrid Gold Price Prediction
# Academic Implementation for Dissertation Research

"""
Agent-based modeling implementation for gold trading simulation
based on behavioral finance and market microstructure theory.

Citations:
[1] Arthur, W.B., Holland, J.H., LeBaron, B., Palmer, R. & Tayler, P. (1997). 
    Asset pricing under endogenous expectations in an artificial stock market. 
    In The economy as an evolving complex system II (pp. 15-44).
[2] Bikhchandani, S., Hirshleifer, D. & Welch, I. (1992). A theory of fads, 
    fashion, custom, and cultural change as informational cascades. Journal 
    of Political Economy, 100(5), 992-1026.
[3] De Long, J.B., Shleifer, A., Summers, L.H. & Waldmann, R.J. (1990). 
    Noise trader risk in financial markets. Journal of Political Economy, 98(4), 703-738.
[4] Lakonishok, J., Shleifer, A. & Vishny, R.W. (1994). Contrarian investment, 
    extrapolation, and risk. Journal of Finance, 49(5), 1541-1578.
"""

from mesa import Agent
import numpy as np
import random
from abc import ABC, abstractmethod

class GoldTrader(Agent):
    """
    Base class for gold trading agents implementing behavioral finance principles.
    
    Base agent architecture following Arthur et al. (1997) Santa Fe 
    artificial stock market methodology for heterogeneous agent modeling.
    
    References:
        Arthur, W.B., Holland, J.H., LeBaron, B., Palmer, R. & Tayler, P. (1997). 
        Asset pricing under endogenous expectations in an artificial stock market. 
        In The economy as an evolving complex system II (pp. 15-44).
    """

    def __init__(self, unique_id, model, agent_type='contrarian'):
        """
        Initialize gold trader agent with behavioral parameters.
        
        Args:
            unique_id (int): Unique agent identifier
            model: Mesa model instance
            agent_type (str): Type of agent
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

        # Performance metrics
        self.last_trade_price = 0
        self.trades_count = 0
        self.profitable_trades = 0

    def observe_environment(self):
        """
        Observe CA signals and neighbor actions for decision making.
        
        Environmental observation implementing Arthur et al. (1997)
        information aggregation methodology for market participants.
        
        Returns:
            dict: Environment observations
            
        References:
            Arthur, W.B., Holland, J.H., LeBaron, B., Palmer, R. & Tayler, P. (1997). 
            Asset pricing under endogenous expectations in an artificial stock market.
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
                observations['price_history'] = self.model.price_history[-10:]
                
                # Calculate price trend
                if len(observations['price_history']) > 1:
                    recent_return = ((observations['price_history'][-1] - 
                                    observations['price_history'][-2]) / 
                                   observations['price_history'][-2])
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
            observations['market_sentiment'] = (self.model.external_data.get('sentiment', 0) 
                                              if self.model.external_data else 0)

            # Volatility calculation
            if len(observations['price_history']) > 5:
                returns = np.diff(observations['price_history']) / observations['price_history'][:-1]
                observations['volatility'] = np.std(returns)
            else:
                observations['volatility'] = 0.02

            return observations
        except Exception:
            return {}

    @abstractmethod
    def make_decision(self, observations):
        """
        Abstract decision logic based on agent type and observations.
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action to take ('buy', 'sell', or 'hold')
        """
        pass

    def execute_trade(self, action, price):
        """
        Execute trading decision with position and cash updates.
        
        Trade execution implementing market microstructure principles
        for realistic trading simulation with transaction costs.
        
        Args:
            action (str): Action to execute
            price (float): Current market price
        """
        try:
            trade_size = self.calculate_trade_size(action, price)
            
            if action == 'buy' and trade_size > 0:
                cost = trade_size * price
                if cost <= self.cash:
                    self.cash -= cost
                    self.gold_holdings += trade_size
                    self.position = 1 if self.gold_holdings > 0 else 0
                    self.last_trade_price = price
                    self.trades_count += 1
                    
                    self.trade_history.append({
                        'action': 'buy', 'price': price, 'quantity': trade_size,
                        'timestamp': self.model.current_day
                    })
                    
            elif action == 'sell' and trade_size > 0:
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
                    self.trade_history.append({
                        'action': 'sell', 'price': price, 'quantity': trade_size,
                        'timestamp': self.model.current_day
                    })

            # Update position history
            self.position_history.append(self.position)
        except Exception:
            pass

    def calculate_trade_size(self, action, price):
        """
        Calculate optimal trade size based on agent risk characteristics.
        
        Position sizing implementing Kelly criterion approach
        adapted for behavioral agent characteristics.
        
        Args:
            action (str): Intended action
            price (float): Current price
            
        Returns:
            float: Trade size
        """
        try:
            wealth = self.cash + self.gold_holdings * price
            base_size = wealth * 0.1 * self.risk_tolerance

            if action == 'buy':
                max_buy = self.cash / price
                return min(base_size / price, max_buy)
            elif action == 'sell':
                return min(base_size / price, self.gold_holdings)
            
            return 0.0
        except Exception:
            return 0.0

    def get_portfolio_value(self, current_price):
        """
        Calculate current portfolio value for performance evaluation.
        
        Args:
            current_price (float): Current gold price
            
        Returns:
            float: Portfolio value
        """
        return self.cash + self.gold_holdings * current_price

    def step(self):
        """Execute agent's daily decision-making and trading process."""
        try:
            observations = self.observe_environment()
            action = self.make_decision(observations)
            
            if action in ['buy', 'sell']:
                self.execute_trade(action, self.model.current_price)
        except Exception:
            pass


class HerderAgent(GoldTrader):
    """
    Agent that follows majority of neighbors implementing herding behavior.
    
    Herding agent based on Bikhchandani et al. (1992) informational 
    cascades theory applied to financial market behavior.
    
    References:
        Bikhchandani, S., Hirshleifer, D. & Welch, I. (1992). A theory of fads, 
        fashion, custom, and cultural change as informational cascades. Journal 
        of Political Economy, 100(5), 992-1026.
    """

    def __init__(self, unique_id, model):
        """Initialize herder agent with social learning parameters."""
        super().__init__(unique_id, model, agent_type='herder')
        self.herd_strength = random.uniform(0.6, 0.9)

    def make_decision(self, observations):
        """
        Follow majority decision based on informational cascades.
        
        Decision mechanism implementing Bikhchandani et al. (1992) 
        informational cascades with neighbor influence weighting.
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action decision
            
        References:
            Bikhchandani, S., Hirshleifer, D. & Welch, I. (1992). A theory of fads, 
            fashion, custom, and cultural change as informational cascades.
        """
        try:
            # Weight factors for decision inputs
            ca_weight = 0.3
            neighbor_weight = 0.4
            sentiment_weight = 0.3

            # CA signal influence
            ca_signal = observations.get('ca_signal', 0)
            ca_influence = ca_signal * ca_weight

            # Neighbor influence (herding behavior)
            neighbor_avg = observations.get('neighbor_avg_position', 0)
            neighbor_influence = neighbor_avg * neighbor_weight * self.herd_strength

            # Sentiment influence
            sentiment = observations.get('market_sentiment', 0)
            sentiment_influence = sentiment * sentiment_weight

            # Combine influences
            total_influence = ca_influence + neighbor_influence + sentiment_influence
            total_influence += random.uniform(-0.1, 0.1)  # Add noise

            # Decision thresholds
            if total_influence > 0.2:
                return 'buy'
            elif total_influence < -0.2:
                return 'sell'
            else:
                return 'hold'
        except Exception:
            return 'hold'


class ContrarianAgent(GoldTrader):
    """
    Agent that acts opposite to majority implementing contrarian strategy.
    
    Contrarian agent based on Lakonishok et al. (1994) contrarian 
    investment strategies and behavioral finance principles.
    
    References:
        Lakonishok, J., Shleifer, A. & Vishny, R.W. (1994). Contrarian investment, 
        extrapolation, and risk. Journal of Finance, 49(5), 1541-1578.
    """

    def __init__(self, unique_id, model):
        """Initialize contrarian agent with contrarian strength parameter."""
        super().__init__(unique_id, model, agent_type='contrarian')
        self.contrarian_strength = random.uniform(0.5, 0.8)

    def make_decision(self, observations):
        """
        Act opposite to majority sentiment and neighbor behavior.
        
        Contrarian decision mechanism implementing Lakonishok et al. (1994)
        contrarian strategy with volatility preference.
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action decision
            
        References:
            Lakonishok, J., Shleifer, A. & Vishny, R.W. (1994). Contrarian investment, 
            extrapolation, and risk. Journal of Finance, 49(5), 1541-1578.
        """
        try:
            # Weight factors for contrarian strategy
            ca_weight = 0.4
            neighbor_weight = 0.3
            volatility_weight = 0.3

            # CA signal influence (contrarian)
            ca_signal = observations.get('ca_signal', 0)
            ca_influence = -ca_signal * ca_weight * self.contrarian_strength

            # Neighbor influence (contrarian)
            neighbor_avg = observations.get('neighbor_avg_position', 0)
            neighbor_influence = -neighbor_avg * neighbor_weight * self.contrarian_strength

            # Volatility influence (contrarians prefer volatility)
            volatility = observations.get('volatility', 0)
            volatility_influence = volatility * volatility_weight * 10

            # Combine influences
            total_influence = ca_influence + neighbor_influence + volatility_influence
            total_influence += random.uniform(-0.1, 0.1)

            # Decision thresholds
            if total_influence > 0.25:
                return 'buy'
            elif total_influence < -0.25:
                return 'sell'
            else:
                return 'hold'
        except Exception:
            return 'hold'


class TrendFollowerAgent(GoldTrader):
    """
    Agent that follows price trends and momentum patterns.
    
    Trend following agent implementing momentum strategies based on
    behavioral finance literature on price continuation patterns.
    
    References:
        Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling 
        losers: Implications for stock market efficiency. Journal of Finance, 48(1), 65-91.
    """

    def __init__(self, unique_id, model):
        """Initialize trend follower with momentum sensitivity parameters."""
        super().__init__(unique_id, model, agent_type='trend_follower')
        self.trend_sensitivity = random.uniform(0.3, 0.7)
        self.momentum_threshold = random.uniform(0.01, 0.03)

    def make_decision(self, observations):
        """
        Follow price trends and momentum with technical analysis.
        
        Trend following decision based on momentum strategies from
        behavioral finance literature on price continuation.
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action decision
        """
        try:
            # Weight factors for trend following
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
            total_influence += random.uniform(-0.05, 0.05)

            # Decision thresholds
            if total_influence > 0.15:
                return 'buy'
            elif total_influence < -0.15:
                return 'sell'
            else:
                return 'hold'
        except Exception:
            return 'hold'


class NoiseTraderAgent(GoldTrader):
    """
    Agent that makes random trades with limited market awareness.
    
    Noise trader implementation based on De Long et al. (1990) 
    noise trader risk theory for financial market behavior.
    
    References:
        De Long, J.B., Shleifer, A., Summers, L.H. & Waldmann, R.J. (1990). 
        Noise trader risk in financial markets. Journal of Political Economy, 98(4), 703-738.
    """

    def __init__(self, unique_id, model):
        """Initialize noise trader with randomness parameters."""
        super().__init__(unique_id, model, agent_type='noise_trader')
        self.noise_level = random.uniform(0.7, 1.0)

    def make_decision(self, observations):
        """
        Make mostly random decisions with slight market awareness.
        
        Noise trading decision implementing De Long et al. (1990) 
        framework with limited rationality and random behavior.
        
        Args:
            observations (dict): Environment observations
            
        Returns:
            str: Action decision
            
        References:
            De Long, J.B., Shleifer, A., Summers, L.H. & Waldmann, R.J. (1990). 
            Noise trader risk in financial markets. Journal of Political Economy, 98(4), 703-738.
        """
        try:
            # Mostly random with slight market influence
            random_factor = random.uniform(-1, 1) * self.noise_level
            
            # Small influence from market signals
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
        except Exception:
            return 'hold'
