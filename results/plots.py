# Visualization Tools for Hybrid Gold Price Prediction
# Implements visualization as per instruction manual Phase 5.2
# Research purposes only - academic dissertation

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

class VisualizationTools:
    """
    Visualization tools for simulation results
    As specified in instruction manual Step 5.2
    
    Reference: Instruction manual Phase 5.2 - Visualization Tools
    """
    
    def __init__(self):
        """
        Initialize visualization tools
        
        Reference: Instruction manual - "def __init__(self):"
        """
        self.setup_style()
        self.color_palette = sns.color_palette("husl", 8)
        self.fig_size = (12, 8)
        
        print("VisualizationTools initialized")
    
    def setup_style(self):
        """
        Setup plotting style
        
        Reference: Instruction manual - "def setup_style(self):"
        """
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Set default parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'grid.alpha': 0.3
        })
    
    def plot_simulation_results(self, results, actual_data=None, save_path=None):
        """
        Plot comprehensive simulation results
        
        Args:
            results (dict): Simulation results
            actual_data (pd.DataFrame): Actual market data
            save_path (str): Path to save the plot
            
        Reference: Instruction manual - "def plot_simulation_results(self, results):"
        """
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Hybrid Gold Price Prediction - Simulation Results', fontsize=16)
            
            # Plot 1: Price series
            ax1 = axes[0, 0]
            if 'price_series' in results:
                price_series = results['price_series']
                dates = pd.date_range(start='2023-01-01', periods=len(price_series), freq='D')
                
                ax1.plot(dates, price_series, label='Simulated Price', color='blue', linewidth=2)
                
                if actual_data is not None and 'Close' in actual_data.columns:
                    min_len = min(len(price_series), len(actual_data))
                    ax1.plot(dates[:min_len], actual_data['Close'][:min_len], 
                            label='Actual Price', color='red', linewidth=2, alpha=0.7)
                
                ax1.set_title('Gold Price Evolution')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price ($)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Returns distribution
            ax2 = axes[0, 1]
            if 'price_series' in results:
                returns = np.diff(price_series) / price_series[:-1]
                ax2.hist(returns, bins=50, alpha=0.7, color='skyblue', density=True)
                ax2.axvline(np.mean(returns), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(returns):.4f}')
                ax2.set_title('Returns Distribution')
                ax2.set_xlabel('Returns')
                ax2.set_ylabel('Density')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Volatility over time
            ax3 = axes[0, 2]
            if 'price_series' in results:
                # Calculate rolling volatility
                returns = np.diff(price_series) / price_series[:-1]
                rolling_vol = pd.Series(returns).rolling(window=20).std()
                
                ax3.plot(dates[1:], rolling_vol, color='green', linewidth=2)
                ax3.set_title('Rolling Volatility (20-day)')
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Volatility')
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Agent performance
            ax4 = axes[1, 0]
            if 'agent_performances' in results:
                agent_types = list(results['agent_performances'].keys())
                pnl_values = [results['agent_performances'][agent]['total_pnl'] 
                             for agent in agent_types]
                
                bars = ax4.bar(agent_types, pnl_values, color=self.color_palette[:len(agent_types)])
                ax4.set_title('Agent Performance (Total P&L)')
                ax4.set_ylabel('P&L ($)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, pnl_values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'${value:.0f}', ha='center', va='bottom')
            
            # Plot 5: Trading volume
            ax5 = axes[1, 1]
            if 'data_collector' in results:
                try:
                    model_data = results['data_collector'].get_model_vars_dataframe()
                    if 'Volume' in model_data.columns:
                        ax5.plot(model_data.index, model_data['Volume'], 
                                color='orange', linewidth=2)
                        ax5.set_title('Trading Volume Over Time')
                        ax5.set_xlabel('Time Step')
                        ax5.set_ylabel('Volume')
                        ax5.grid(True, alpha=0.3)
                except:
                    ax5.text(0.5, 0.5, 'Volume data not available', 
                            ha='center', va='center', transform=ax5.transAxes)
            
            # Plot 6: CA signal evolution
            ax6 = axes[1, 2]
            if 'data_collector' in results:
                try:
                    model_data = results['data_collector'].get_model_vars_dataframe()
                    if 'CA_Signal' in model_data.columns:
                        ax6.plot(model_data.index, model_data['CA_Signal'], 
                                color='purple', linewidth=2)
                        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                        ax6.set_title('Cellular Automaton Signal')
                        ax6.set_xlabel('Time Step')
                        ax6.set_ylabel('CA Signal')
                        ax6.grid(True, alpha=0.3)
                except:
                    ax6.text(0.5, 0.5, 'CA signal data not available', 
                            ha='center', va='center', transform=ax6.transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Simulation results plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting simulation results: {e}")
    
    def plot_ca_evolution(self, ca_states, save_path=None):
        """
        Plot cellular automaton evolution
        
        Args:
            ca_states (list): List of CA grid states over time
            save_path (str): Path to save the plot
            
        Reference: Instruction manual - "def plot_ca_evolution(self, ca_states):"
        """
        try:
            if not ca_states:
                print("No CA states provided for visualization")
                return
            
            # Create subplots for different time steps
            num_plots = min(6, len(ca_states))
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Cellular Automaton Evolution', fontsize=16)
            
            # Flatten axes for easier indexing
            axes = axes.flatten()
            
            # Color map for CA states
            colors = ['red', 'gray', 'green']  # bearish, neutral, bullish
            cmap = plt.matplotlib.colors.ListedColormap(colors)
            
            # Select time steps to display
            time_steps = np.linspace(0, len(ca_states)-1, num_plots, dtype=int)
            
            for i, step in enumerate(time_steps):
                ax = axes[i]
                
                # Plot CA grid
                im = ax.imshow(ca_states[step], cmap=cmap, vmin=-1, vmax=1)
                ax.set_title(f'Time Step {step}')
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')
                
                # Add grid statistics
                grid = ca_states[step]
                bullish_ratio = np.sum(grid == 1) / grid.size
                bearish_ratio = np.sum(grid == -1) / grid.size
                
                ax.text(0.02, 0.98, f'Bullish: {bullish_ratio:.2%}', 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.text(0.02, 0.88, f'Bearish: {bearish_ratio:.2%}', 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes, ticks=[-1, 0, 1], shrink=0.6)
            cbar.set_ticklabels(['Bearish', 'Neutral', 'Bullish'])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"CA evolution plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting CA evolution: {e}")
    
    def create_interactive_dashboard(self, results, save_path=None):
        """
        Create interactive dashboard using Plotly
        
        Args:
            results (dict): Simulation results
            save_path (str): Path to save the dashboard
            
        Reference: Instruction manual - "def create_dashboard(self, results):"
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Price Evolution', 'Returns Distribution', 
                              'Agent Performance', 'Trading Volume',
                              'CA Signal', 'Volatility'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Plot 1: Price evolution
            if 'price_series' in results:
                price_series = results['price_series']
                dates = pd.date_range(start='2023-01-01', periods=len(price_series), freq='D')
                
                fig.add_trace(
                    go.Scatter(x=dates, y=price_series, name='Simulated Price',
                              line=dict(color='blue', width=2)),
                    row=1, col=1
                )
            
            # Plot 2: Returns distribution
            if 'price_series' in results:
                returns = np.diff(price_series) / price_series[:-1]
                
                fig.add_trace(
                    go.Histogram(x=returns, name='Returns', nbinsx=50,
                                histnorm='probability density'),
                    row=1, col=2
                )
            
            # Plot 3: Agent performance
            if 'agent_performances' in results:
                agent_types = list(results['agent_performances'].keys())
                pnl_values = [results['agent_performances'][agent]['total_pnl'] 
                             for agent in agent_types]
                
                fig.add_trace(
                    go.Bar(x=agent_types, y=pnl_values, name='Agent P&L'),
                    row=2, col=1
                )
            
            # Plot 4: Trading volume
            if 'data_collector' in results:
                try:
                    model_data = results['data_collector'].get_model_vars_dataframe()
                    if 'Volume' in model_data.columns:
                        fig.add_trace(
                            go.Scatter(x=model_data.index, y=model_data['Volume'],
                                      name='Volume', line=dict(color='orange')),
                            row=2, col=2
                        )
                except:
                    pass
            
            # Plot 5: CA signal
            if 'data_collector' in results:
                try:
                    model_data = results['data_collector'].get_model_vars_dataframe()
                    if 'CA_Signal' in model_data.columns:
                        fig.add_trace(
                            go.Scatter(x=model_data.index, y=model_data['CA_Signal'],
                                      name='CA Signal', line=dict(color='purple')),
                            row=3, col=1
                        )
                except:
                    pass
            
            # Plot 6: Volatility
            if 'price_series' in results:
                returns = np.diff(price_series) / price_series[:-1]
                rolling_vol = pd.Series(returns).rolling(window=20).std()
                
                fig.add_trace(
                    go.Scatter(x=dates[1:], y=rolling_vol, name='Volatility',
                              line=dict(color='green')),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title='Hybrid Gold Price Prediction - Interactive Dashboard',
                showlegend=True,
                height=900
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            
            fig.update_xaxes(title_text="Returns", row=1, col=2)
            fig.update_yaxes(title_text="Density", row=1, col=2)
            
            fig.update_xaxes(title_text="Agent Type", row=2, col=1)
            fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
            
            fig.update_xaxes(title_text="Time Step", row=2, col=2)
            fig.update_yaxes(title_text="Volume", row=2, col=2)
            
            fig.update_xaxes(title_text="Time Step", row=3, col=1)
            fig.update_yaxes(title_text="CA Signal", row=3, col=1)
            
            fig.update_xaxes(title_text="Date", row=3, col=2)
            fig.update_yaxes(title_text="Volatility", row=3, col=2)
            
            if save_path:
                fig.write_html(save_path)
                print(f"Interactive dashboard saved to {save_path}")
            
            # Show the dashboard
            fig.show()
            
        except Exception as e:
            print(f"Error creating interactive dashboard: {e}")
    
    def plot_parameter_sensitivity(self, sensitivity_data, save_path=None):
        """
        Plot parameter sensitivity analysis
        
        Args:
            sensitivity_data (pd.DataFrame): Sensitivity analysis results
            save_path (str): Path to save the plot
        """
        try:
            if sensitivity_data.empty:
                print("No sensitivity data provided")
                return
            
            # Get unique parameters
            parameters = sensitivity_data['parameter'].unique()
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
            axes = axes.flatten()
            
            for i, param in enumerate(parameters[:4]):  # Plot first 4 parameters
                ax = axes[i]
                param_data = sensitivity_data[sensitivity_data['parameter'] == param]
                
                # Group by parameter value and calculate statistics
                grouped = param_data.groupby('parameter_value').agg({
                    'final_price': ['mean', 'std'],
                    'price_change': ['mean', 'std']
                }).reset_index()
                
                # Plot mean with error bars
                x = grouped['parameter_value']
                y_mean = grouped[('final_price', 'mean')]
                y_std = grouped[('final_price', 'std')]
                
                ax.errorbar(x, y_mean, yerr=y_std, marker='o', capsize=5)
                ax.set_title(f'Sensitivity to {param}')
                ax.set_xlabel(f'{param} Value')
                ax.set_ylabel('Final Price ($)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Parameter sensitivity plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting parameter sensitivity: {e}")
    
    def create_agent_heatmap(self, agent_data, save_path=None):
        """
        Create heatmap of agent positions over time
        
        Args:
            agent_data (pd.DataFrame): Agent data from simulation
            save_path (str): Path to save the plot
        """
        try:
            if agent_data.empty:
                print("No agent data provided")
                return
            
            # Pivot data to create heatmap
            pivot_data = agent_data.pivot_table(
                index='AgentID', 
                columns='Step', 
                values='Position',
                aggfunc='mean'
            )
            
            # Create heatmap
            plt.figure(figsize=(15, 8))
            sns.heatmap(pivot_data, cmap='RdYlGn', center=0, 
                       cbar_kws={'label': 'Position'})
            plt.title('Agent Positions Over Time')
            plt.xlabel('Time Step')
            plt.ylabel('Agent ID')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Agent heatmap saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating agent heatmap: {e}")
    
    def animate_ca_evolution(self, ca_states, save_path=None):
        """
        Create animated visualization of CA evolution
        
        Args:
            ca_states (list): List of CA grid states over time
            save_path (str): Path to save the animation
        """
        try:
            if not ca_states:
                print("No CA states provided for animation")
                return
            
            # Setup figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color map
            colors = ['red', 'gray', 'green']
            cmap = plt.matplotlib.colors.ListedColormap(colors)
            
            # Initialize plot
            im = ax.imshow(ca_states[0], cmap=cmap, vmin=-1, vmax=1)
            ax.set_title('Cellular Automaton Evolution')
            
            # Add colorbar
            cbar = plt.colorbar(im, ticks=[-1, 0, 1])
            cbar.set_ticklabels(['Bearish', 'Neutral', 'Bullish'])
            
            # Animation function
            def animate(frame):
                im.set_array(ca_states[frame])
                ax.set_title(f'Cellular Automaton Evolution - Step {frame}')
                return [im]
            
            # Create animation
            anim = FuncAnimation(fig, animate, frames=len(ca_states), 
                               interval=200, blit=True, repeat=True)
            
            if save_path:
                anim.save(save_path, writer='pillow', fps=5)
                print(f"CA animation saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating CA animation: {e}")
    
    def create_correlation_matrix(self, results, save_path=None):
        """
        Create correlation matrix of key variables
        
        Args:
            results (dict): Simulation results
            save_path (str): Path to save the plot
        """
        try:
            if 'data_collector' in results:
                model_data = results['data_collector'].get_model_vars_dataframe()
                
                # Select numerical columns
                numeric_cols = model_data.select_dtypes(include=[np.number]).columns
                
                # Calculate correlation matrix
                corr_matrix = model_data[numeric_cols].corr()
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5)
                plt.title('Correlation Matrix of Key Variables')
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Correlation matrix saved to {save_path}")
                
                plt.show()
                
        except Exception as e:
            print(f"Error creating correlation matrix: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize visualization tools
    viz = VisualizationTools()
    
    # Create mock results for testing
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    mock_results = {
        'price_series': 1800 + np.cumsum(np.random.randn(252) * 5),
        'final_price': 1850,
        'agent_performances': {
            'herder': {'total_pnl': 150, 'win_rate': 0.65},
            'contrarian': {'total_pnl': -50, 'win_rate': 0.45},
            'trend_follower': {'total_pnl': 200, 'win_rate': 0.70}
        }
    }
    
    # Test basic plotting
    viz.plot_simulation_results(mock_results)
    
    # Test CA visualization
    ca_states = [np.random.choice([-1, 0, 1], size=(20, 20)) for _ in range(10)]
    viz.plot_ca_evolution(ca_states)
    
    # Test interactive dashboard
    viz.create_interactive_dashboard(mock_results)
    
    print("Visualization tools testing completed successfully!")
