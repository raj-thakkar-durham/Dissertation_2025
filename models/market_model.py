# Market Model for Hybrid Gold Price Prediction
# Implements agent-based market model as per instruction manual Phase 3.2
# Research purposes only - academic dissertation

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
import random
from models.agents import HerderAgent, ContrarianAgent, TrendFollowerAgent, NoiseTraderAgent

class GoldMarketModel(Model):
    """
    Agent-based market model for gold price simulation
    As specified in instruction manual Step 3.2
    
    Reference: Instruction manual Phase 3.2 - Market Model
    """
    
    def __init__(self, num_agents, grid_size, ca_model):
        """
        Initialize gold market model
        
        Args:
            num_agents (int): Number of agents
            grid_size (tuple): Grid size for agent placement
            ca_model: Cellular automaton model
            
        Reference: Instruction manual - "def __init__(self, num_agents, grid_size, ca_model):"
        """
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(grid_size[0], grid_size[1], True)
        self.schedule = RandomActivation(self)
        self.ca_model = ca_model
        self.current_price = 1800  # Starting gold price
        self.price_history = [self.current_price]
        self.external_data = None
        self.current_day = 0
        
        # Market parameters
        self.volatility = 0.02
        self.liquidity = 1000000  # Market liquidity
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.market_maker_spread = 0.002  # 0.2% spread
        
        # Trading statistics
        self.daily_volume = 0
        self.daily_trades = 0
        self.buy_pressure = 0
        self.sell_pressure = 0
        
        # Agent statistics
        self.agent_types = {}
        self.agent_performances = {}
        
        # Create agents
        self.create_agents()
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Price": "current_price",
                "Volume": self.get_volume,
                "Buy_Pressure": "buy_pressure",
                "Sell_Pressure": "sell_pressure",
                "CA_Signal": self.get_ca_signal,
                "Volatility": self.calculate_volatility,
                "Number_of_Trades": "daily_trades"
            },
            agent_reporters={
                "Position": "position",
                "Cash": "cash",
                "Holdings": "gold_holdings",
                "Portfolio_Value": lambda agent: agent.get_portfolio_value(agent.model.current_price),
                "Agent_Type": "agent_type"
            }
        )
        
        print(f"GoldMarketModel initialized with {num_agents} agents on {grid_size} grid")
        
    def create_agents(self):
        """
        Create different types of agents
        Distribute them on grid
        
        Reference: Instruction manual - "Create different types of agents"
        """
        try:
            # Agent type distribution
            agent_distribution = {
                'herder': 0.3,
                'contrarian': 0.25,
                'trend_follower': 0.25,
                'noise_trader': 0.2
            }
            
            # Create agents
            for i in range(self.num_agents):
                # Determine agent type
                rand_val = random.random()
                cumulative = 0
                agent_type = 'herder'  # default
                
                for atype, prob in agent_distribution.items():
                    cumulative += prob
                    if rand_val <= cumulative:
                        agent_type = atype
                        break
                
                # Create agent based on type
                if agent_type == 'herder':
                    agent = HerderAgent(i, self)
                elif agent_type == 'contrarian':
                    agent = ContrarianAgent(i, self)
                elif agent_type == 'trend_follower':
                    agent = TrendFollowerAgent(i, self)
                else:  # noise_trader
                    agent = NoiseTraderAgent(i, self)
                
                # Add to scheduler
                self.schedule.add(agent)
                
                # Place on grid
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
                self.grid.place_agent(agent, (x, y))
                
                # Track agent types
                if agent_type not in self.agent_types:
                    self.agent_types[agent_type] = 0
                self.agent_types[agent_type] += 1
            
            print(f"Created agents: {self.agent_types}")
            
        except Exception as e:
            print(f"Error creating agents: {e}")
    
    def update_price(self):
        """
        Calculate net demand from all agents
        Update price based on supply/demand
        
        Reference: Instruction manual - "Calculate net demand from all agents"
        """
        try:
            # Reset daily statistics
            self.daily_volume = 0
            self.daily_trades = 0
            self.buy_pressure = 0
            self.sell_pressure = 0
            
            # Collect all agent trades
            buy_orders = []
            sell_orders = []
            
            for agent in self.schedule.agents:
                if hasattr(agent, 'trade_history') and agent.trade_history:
                    # Get today's trades
                    today_trades = [trade for trade in agent.trade_history 
                                  if trade['timestamp'] == self.current_day]
                    
                    for trade in today_trades:
                        if trade['action'] == 'buy':
                            buy_orders.append(trade['quantity'])
                            self.buy_pressure += trade['quantity']
                        elif trade['action'] == 'sell':
                            sell_orders.append(trade['quantity'])
                            self.sell_pressure += trade['quantity']
                        
                        self.daily_volume += trade['quantity']
                        self.daily_trades += 1
            
            # Calculate net demand
            total_buy_volume = sum(buy_orders)
            total_sell_volume = sum(sell_orders)
            net_demand = total_buy_volume - total_sell_volume
            
            # Price impact model
            if abs(net_demand) > 0:
                # Price impact depends on net demand relative to liquidity
                price_impact = (net_demand / self.liquidity) * self.current_price
                
                # Add random market noise
                market_noise = np.random.normal(0, self.volatility) * self.current_price
                
                # CA influence on price
                ca_signal = self.get_ca_signal()
                ca_influence = ca_signal * 0.1 * self.current_price
                
                # External data influence
                external_influence = 0
                if self.external_data:
                    sentiment = self.external_data.get('sentiment', 0)
                    volatility = self.external_data.get('volatility', 0)
                    external_influence = (sentiment * 0.05 + volatility * 0.02) * self.current_price
                
                # Calculate new price
                price_change = price_impact + market_noise + ca_influence + external_influence
                new_price = self.current_price + price_change
                
                # Apply price bounds (gold can't go negative)
                new_price = max(new_price, 100)  # Minimum price
                new_price = min(new_price, 10000)  # Maximum price for stability
                
                self.current_price = new_price
            
            # Update price history
            self.price_history.append(self.current_price)
            
            # Keep only last 252 days (trading year)
            if len(self.price_history) > 252:
                self.price_history.pop(0)
                
        except Exception as e:
            print(f"Error updating price: {e}")
    
    def get_volume(self):
        """
        Calculate total trading volume
        
        Returns:
            float: Total trading volume
            
        Reference: Instruction manual - "Calculate total trading volume"
        """
        return self.daily_volume
    
    def get_ca_signal(self):
        """
        Get current CA signal
        
        Returns:
            float: CA signal
        """
        try:
            if self.ca_model:
                return self.ca_model.get_market_signal()
            return 0.0
        except Exception as e:
            return 0.0
    
    def calculate_volatility(self):
        """
        Calculate current market volatility
        
        Returns:
            float: Market volatility
        """
        try:
            if len(self.price_history) < 2:
                return 0.02  # Default volatility
            
            # Calculate returns
            returns = []
            for i in range(1, len(self.price_history)):
                ret = (self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]
                returns.append(ret)
            
            # Calculate volatility as standard deviation
            if len(returns) > 1:
                volatility = np.std(returns)
                return volatility
            else:
                return 0.02
                
        except Exception as e:
            return 0.02
    
    def step(self):
        """
        One simulation day
        Update CA, agents act, update price
        
        Reference: Instruction manual - "One simulation day"
        """
        try:
            # Update CA with current market state
            if self.ca_model:
                # Prepare external features for CA
                external_features = {
                    'sentiment': self.external_data.get('sentiment', 0) if self.external_data else 0,
                    'volatility': self.calculate_volatility(),
                    'trend': self.calculate_trend(),
                    'price_momentum': self.calculate_momentum()
                }
                
                # Step CA
                self.ca_model.step(external_features)
            
            # Agents observe and act
            self.schedule.step()
            
            # Update market price based on agent actions
            self.update_price()
            
            # Collect data
            self.datacollector.collect(self)
            
            # Increment day counter
            self.current_day += 1
            
        except Exception as e:
            print(f"Error in simulation step {self.current_day}: {e}")
    
    def calculate_trend(self):
        """
        Calculate price trend
        
        Returns:
            float: Price trend
        """
        try:
            if len(self.price_history) < 5:
                return 0.0
            
            # Calculate trend as average return over last 5 days
            recent_prices = self.price_history[-5:]
            returns = []
            for i in range(1, len(recent_prices)):
                ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                returns.append(ret)
            
            return np.mean(returns)
            
        except Exception as e:
            return 0.0
    
    def calculate_momentum(self):
        """
        Calculate price momentum
        
        Returns:
            float: Price momentum
        """
        try:
            if len(self.price_history) < 10:
                return 0.0
            
            # Calculate momentum as difference between recent and older averages
            recent_avg = np.mean(self.price_history[-5:])
            older_avg = np.mean(self.price_history[-10:-5])
            
            momentum = (recent_avg - older_avg) / older_avg
            return momentum
            
        except Exception as e:
            return 0.0
    
    def run_simulation(self, num_days, external_data=None):
        """
        Run complete simulation
        Return price series and metrics
        
        Args:
            num_days (int): Number of days to simulate
            external_data (dict): External market data
            
        Returns:
            dict: Simulation results
            
        Reference: Instruction manual - "Run complete simulation"
        """
        try:
            print(f"Starting simulation for {num_days} days...")
            
            # Store external data
            self.external_data = external_data or {}
            
            # Run simulation
            for day in range(num_days):
                # Update external data for current day
                if isinstance(external_data, dict) and 'data' in external_data:
                    day_data = external_data['data'].iloc[day] if day < len(external_data['data']) else {}
                    self.external_data = dict(day_data) if hasattr(day_data, 'items') else {}
                
                # Step simulation
                self.step()
                
                # Print progress
                if (day + 1) % 50 == 0:
                    print(f"Day {day + 1}/{num_days}: Price = {self.current_price:.2f}")
            
            # Collect results
            results = {
                'price_series': self.price_history.copy(),
                'final_price': self.current_price,
                'total_volume': sum([self.datacollector.model_vars['Volume'][i] 
                                   for i in range(len(self.datacollector.model_vars['Volume']))]),
                'average_volatility': np.mean([self.datacollector.model_vars['Volatility'][i] 
                                             for i in range(len(self.datacollector.model_vars['Volatility']))]),
                'agent_performances': self.get_agent_performances(),
                'market_statistics': self.get_market_statistics(),
                'data_collector': self.datacollector
            }
            
            print(f"Simulation completed. Final price: {self.current_price:.2f}")
            return results
            
        except Exception as e:
            print(f"Error running simulation: {e}")
            return {}
    
    def get_agent_performances(self):
        """
        Get performance statistics for each agent type
        
        Returns:
            dict: Agent performance statistics
        """
        try:
            performances = {}
            
            for agent in self.schedule.agents:
                agent_type = agent.agent_type
                if agent_type not in performances:
                    performances[agent_type] = {
                        'total_pnl': [],
                        'win_rate': [],
                        'total_trades': [],
                        'final_portfolio_value': []
                    }
                
                # Calculate performance metrics
                portfolio_value = agent.get_portfolio_value(self.current_price)
                total_pnl = sum(agent.pnl_history) if agent.pnl_history else 0
                win_rate = (agent.profitable_trades / agent.trades_count) if agent.trades_count > 0 else 0
                
                performances[agent_type]['total_pnl'].append(total_pnl)
                performances[agent_type]['win_rate'].append(win_rate)
                performances[agent_type]['total_trades'].append(agent.trades_count)
                performances[agent_type]['final_portfolio_value'].append(portfolio_value)
            
            # Calculate averages
            for agent_type in performances:
                for metric in performances[agent_type]:
                    if performances[agent_type][metric]:
                        performances[agent_type][metric] = np.mean(performances[agent_type][metric])
                    else:
                        performances[agent_type][metric] = 0
            
            return performances
            
        except Exception as e:
            print(f"Error calculating agent performances: {e}")
            return {}
    
    def get_market_statistics(self):
        """
        Get market statistics
        
        Returns:
            dict: Market statistics
        """
        try:
            stats = {
                'initial_price': self.price_history[0] if self.price_history else 0,
                'final_price': self.current_price,
                'price_change': ((self.current_price - self.price_history[0]) / 
                               self.price_history[0]) if self.price_history else 0,
                'max_price': max(self.price_history) if self.price_history else 0,
                'min_price': min(self.price_history) if self.price_history else 0,
                'average_price': np.mean(self.price_history) if self.price_history else 0,
                'price_volatility': np.std(self.price_history) if len(self.price_history) > 1 else 0,
                'total_simulation_days': self.current_day,
                'agent_type_distribution': self.agent_types
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating market statistics: {e}")
            return {}
    
    def save_results(self, filename):
        """
        Save simulation results
        
        Args:
            filename (str): Output filename
        """
        try:
            # Get model data
            model_data = self.datacollector.get_model_vars_dataframe()
            agent_data = self.datacollector.get_agent_vars_dataframe()
            
            # Save to CSV
            model_data.to_csv(f"{filename}_model_data.csv")
            agent_data.to_csv(f"{filename}_agent_data.csv")
            
            print(f"Results saved to {filename}_model_data.csv and {filename}_agent_data.csv")
            
        except Exception as e:
            print(f"Error saving results: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create mock CA model for testing
    class MockCA:
        def __init__(self):
            self.signal = 0.0
            
        def get_market_signal(self):
            return self.signal
            
        def step(self, external_features):
            # Simple mock behavior
            self.signal = np.random.uniform(-0.5, 0.5)
    
    # Initialize model
    ca_model = MockCA()
    model = GoldMarketModel(num_agents=50, grid_size=(10, 10), ca_model=ca_model)
    
    # Create mock external data
    external_data = {
        'sentiment': 0.1,
        'volatility': 0.02,
        'trend': 0.05
    }
    
    # Run short simulation
    results = model.run_simulation(num_days=10, external_data=external_data)
    
    # Display results
    print("\nSimulation Results:")
    print(f"Price change: {results['price_series'][0]:.2f} -> {results['final_price']:.2f}")
    print(f"Total volume: {results['total_volume']:.2f}")
    print(f"Average volatility: {results['average_volatility']:.4f}")
    
    print("\nAgent Performances:")
    for agent_type, performance in results['agent_performances'].items():
        print(f"{agent_type}: PnL = {performance['total_pnl']:.2f}, "
              f"Win Rate = {performance['win_rate']:.2f}")
    
    print("\nMarket Statistics:")
    for key, value in results['market_statistics'].items():
        print(f"{key}: {value}")
    
    print("\nModel testing completed successfully!")
