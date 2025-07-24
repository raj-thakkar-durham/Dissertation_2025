# Visualization Tools for Hybrid Gold Price Prediction
# Academic Implementation for Dissertation Research

"""
Comprehensive visualization suite implementing academic standards for
computational finance research presentation and analysis.

Citations:
[1] Tufte, E.R. (2001). The Visual Display of Quantitative Information. 
    Graphics Press, Cheshire, CT.
[2] Cleveland, W.S. (1993). Visualizing Data. Hobart Press, Summit, NJ.
[3] Wilkinson, L. (2005). The Grammar of Graphics. Springer, New York.
[4] Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. Computing 
    in Science & Engineering, 9(3), 90-95.
"""

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
    Comprehensive visualization toolkit for hybrid gold price prediction analysis.
    
    Implementation following Tufte (2001) principles of statistical graphics
    and Cleveland (1993) data visualization methodology.
    
    References:
        Tufte, E.R. (2001). The Visual Display of Quantitative Information. 
        Graphics Press, Cheshire, CT.
        
        Cleveland, W.S. (1993). Visualizing Data. Hobart Press, Summit, NJ.
    """

    def __init__(self):
        """Initialize visualization environment with academic standards."""
        self.setup_style()
        self.color_palette = self.create_academic_palette()
        self.fig_size = (12, 8)

    def setup_style(self):
        """
        Configure plotting styles for academic publication quality.
        
        Style configuration based on academic publishing standards
        and Wilkinson (2005) grammar of graphics principles.
        
        References:
            Wilkinson, L. (2005). The Grammar of Graphics. Springer, New York.
        """
        # Academic matplotlib style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        
        # Publication-quality parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'grid.alpha': 0.3,
            'axes.grid': True,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        # Seaborn academic style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)

    def create_academic_palette(self, n_colors=8):
        """
        Create color palette suitable for academic publications.
        
        Color selection based on Hunter (2007) matplotlib guidelines
        for scientific visualization accessibility.
        
        Args:
            n_colors (int): Number of distinct colors needed
            
        Returns:
            list: Academic-appropriate color palette
            
        References:
            Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. 
            Computing in Science & Engineering, 9(3), 90-95.
        """
        # Colorblind-friendly academic palette
        academic_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange  
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f'   # Gray
        ]
        return academic_colors[:n_colors]

    def plot_simulation_results(self, results, actual_data=None, save_path=None):
        """
        Generate comprehensive simulation results visualization.
        
        Multi-panel visualization implementing Cleveland (1993) principles
        for effective statistical graphics presentation.
        
        Args:
            results (dict): Simulation results
            actual_data (pd.DataFrame): Actual market data for comparison
            save_path (str): File path for saving
            
        References:
            Cleveland, W.S. (1993). Visualizing Data. Hobart Press, Summit, NJ.
        """
        try:
            # Validate inputs
            if not isinstance(results, dict):
                print("Warning: Results must be a dictionary")
                return
                
            # Create comprehensive figure layout
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

            # Main title with academic formatting
            fig.suptitle('Hybrid Gold Price Prediction: Simulation Analysis\n'
                        'Cellular Automata and Agent-Based Model Results', 
                        fontsize=16, fontweight='bold', y=0.95)

            # Panel 1: Price Evolution Comparison
            ax1 = fig.add_subplot(gs[0, :2])
            if 'price_series' in results and results['price_series'] is not None:
                price_series = np.array(results['price_series'])
                if len(price_series) > 0:
                    dates = pd.date_range(start='2023-01-01', periods=len(price_series), freq='D')
                    
                    ax1.plot(dates, price_series, label='Simulated Price', 
                            color=self.color_palette[0], linewidth=2.5, alpha=0.8)
                    
                    if actual_data is not None and isinstance(actual_data, pd.DataFrame) and 'Close' in actual_data.columns:
                        min_len = min(len(price_series), len(actual_data))
                        actual_subset = actual_data['Close'].iloc[:min_len]
                        ax1.plot(dates[:min_len], actual_subset, 
                                label='Actual Price', color=self.color_palette[1], 
                                linewidth=2.5, alpha=0.8, linestyle='--')
                    
                    ax1.set_title('Gold Price Evolution: Simulation vs Reality', fontweight='bold')
                    ax1.set_xlabel('Trading Days')
                    ax1.set_ylabel('Gold Price (USD/oz)')
                    ax1.legend(frameon=True, fancybox=True, shadow=True)
                    ax1.grid(True, alpha=0.3)
                    
                    # Format x-axis for better readability
                    ax1.tick_params(axis='x', rotation=45)
            else:
                ax1.text(0.5, 0.5, 'Price Series\nNot Available', 
                        ha='center', va='center', transform=ax1.transAxes,
                        fontsize=12, style='italic')

            # Panel 2: Return Distribution Analysis
            ax2 = fig.add_subplot(gs[0, 2])
            if 'price_series' in results and results['price_series'] is not None:
                price_series = np.array(results['price_series'])
                if len(price_series) > 1:
                    returns = np.diff(price_series) / price_series[:-1]
                    returns = returns[np.isfinite(returns)]  # Remove inf/nan values
                    
                    if len(returns) > 0:
                        # Histogram with statistical overlay
                        n_bins = min(30, max(10, len(returns) // 10))
                        ax2.hist(returns, bins=n_bins, alpha=0.7, color=self.color_palette[2], 
                                density=True, edgecolor='black', linewidth=0.5)
                        
                        # Normal distribution overlay
                        mu, sigma = np.mean(returns), np.std(returns)
                        if sigma > 0:
                            x = np.linspace(returns.min(), returns.max(), 100)
                            ax2.plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * 
                                    np.exp(-0.5 * ((x - mu) / sigma) ** 2), 
                                    'r--', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
                        
                        ax2.axvline(mu, color='red', linestyle='-', alpha=0.8, 
                                   label=f'Mean: {mu:.4f}')
                        ax2.set_title('Return Distribution\nAnalysis', fontweight='bold')
                        ax2.set_xlabel('Daily Returns')
                        ax2.set_ylabel('Density')
                        ax2.legend(fontsize=9)
                        ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Return Data\nNot Available', 
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=12, style='italic')

            # Panel 3: Volatility Clustering
            ax3 = fig.add_subplot(gs[1, 0])
            if 'price_series' in results and results['price_series'] is not None:
                price_series = np.array(results['price_series'])
                if len(price_series) > 20:
                    returns = np.diff(price_series) / price_series[:-1]
                    returns = returns[np.isfinite(returns)]
                    
                    if len(returns) > 20:
                        rolling_vol = pd.Series(returns).rolling(window=20, min_periods=1).std()
                        dates = pd.date_range(start='2023-01-01', periods=len(rolling_vol), freq='D')
                        
                        ax3.plot(dates, rolling_vol, color=self.color_palette[3], linewidth=2)
                        ax3.set_title('Volatility Clustering\n(20-day Rolling)', fontweight='bold')
                        ax3.set_xlabel('Date')
                        ax3.set_ylabel('Volatility')
                        ax3.grid(True, alpha=0.3)
                        ax3.tick_params(axis='x', rotation=45)
            else:
                ax3.text(0.5, 0.5, 'Volatility Data\nNot Available', 
                        ha='center', va='center', transform=ax3.transAxes,
                        fontsize=12, style='italic')

            # Panel 4: Agent Performance Analysis
            ax4 = fig.add_subplot(gs[1, 1])
            if 'agent_performances' in results and results['agent_performances']:
                agent_types = list(results['agent_performances'].keys())
                pnl_values = []
                
                for agent in agent_types:
                    pnl = results['agent_performances'][agent].get('total_pnl', 0)
                    pnl_values.append(pnl)
                
                if agent_types and pnl_values:
                    bars = ax4.bar(agent_types, pnl_values, 
                                  color=self.color_palette[:len(agent_types)], 
                                  alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, pnl_values):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., 
                                height + (max(pnl_values) - min(pnl_values)) * 0.01,
                                f'${value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
                    ax4.set_title('Agent Performance\n(Total P&L)', fontweight='bold')
                    ax4.set_ylabel('Profit/Loss (USD)')
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Agent Performance\nData Not Available', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=12, style='italic')

            # Panel 5: Trading Volume Evolution
            ax5 = fig.add_subplot(gs[1, 2])
            if 'data_collector' in results and results['data_collector'] is not None:
                try:
                    model_data = results['data_collector'].get_model_vars_dataframe()
                    if isinstance(model_data, pd.DataFrame) and 'Volume' in model_data.columns:
                        volume_data = model_data['Volume'].dropna()
                        if len(volume_data) > 0:
                            ax5.plot(volume_data.index, volume_data, 
                                    color=self.color_palette[4], linewidth=2)
                            ax5.set_title('Trading Volume\nEvolution', fontweight='bold')
                            ax5.set_xlabel('Time Step')
                            ax5.set_ylabel('Volume')
                            ax5.grid(True, alpha=0.3)
                        else:
                            raise ValueError("No volume data available")
                    else:
                        raise ValueError("Volume column not found")
                except:
                    ax5.text(0.5, 0.5, 'Volume Data\nUnavailable', 
                            ha='center', va='center', transform=ax5.transAxes,
                            fontsize=12, style='italic')
            else:
                ax5.text(0.5, 0.5, 'Volume Data\nUnavailable', 
                        ha='center', va='center', transform=ax5.transAxes,
                        fontsize=12, style='italic')

            # Panel 6: Cellular Automaton Signal
            ax6 = fig.add_subplot(gs[2, 0])
            if 'data_collector' in results and results['data_collector'] is not None:
                try:
                    model_data = results['data_collector'].get_model_vars_dataframe()
                    if isinstance(model_data, pd.DataFrame) and 'CA_Signal' in model_data.columns:
                        ca_signal_data = model_data['CA_Signal'].dropna()
                        if len(ca_signal_data) > 0:
                            ax6.plot(ca_signal_data.index, ca_signal_data, 
                                    color=self.color_palette[5], linewidth=2.5)
                            ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                            ax6.fill_between(ca_signal_data.index, ca_signal_data, 0, 
                                           alpha=0.3, color=self.color_palette[5])
                            ax6.set_title('Cellular Automaton\nMarket Signal', fontweight='bold')
                            ax6.set_xlabel('Time Step')
                            ax6.set_ylabel('CA Signal')
                            ax6.grid(True, alpha=0.3)
                        else:
                            raise ValueError("No CA signal data available")
                    else:
                        raise ValueError("CA_Signal column not found")
                except:
                    ax6.text(0.5, 0.5, 'CA Signal\nUnavailable', 
                            ha='center', va='center', transform=ax6.transAxes,
                            fontsize=12, style='italic')
            else:
                ax6.text(0.5, 0.5, 'CA Signal\nUnavailable', 
                        ha='center', va='center', transform=ax6.transAxes,
                        fontsize=12, style='italic')

            # Panel 7: Risk-Return Scatter
            ax7 = fig.add_subplot(gs[2, 1])
            if 'agent_performances' in results and results['agent_performances']:
                agent_types = list(results['agent_performances'].keys())
                returns_data = []
                risk_data = []
                
                for agent in agent_types:
                    pnl = results['agent_performances'][agent].get('total_pnl', 0)
                    returns_data.append(pnl)
                    # Proxy risk measure based on absolute PnL
                    risk_data.append(abs(pnl) * 0.1 + np.random.normal(0, 0.1))
                
                if len(returns_data) > 0 and len(risk_data) > 0:
                    scatter = ax7.scatter(risk_data, returns_data, 
                                         c=range(len(agent_types)), 
                                         cmap='viridis', alpha=0.7, s=100, edgecolors='black')
                    
                    for i, agent_type in enumerate(agent_types):
                        clean_name = agent_type.replace('_', ' ').replace('Agent', '').title()
                        ax7.annotate(clean_name, 
                                   (risk_data[i], returns_data[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=9)
                    
                    ax7.set_title('Agent Risk-Return\nProfile', fontweight='bold')
                    ax7.set_xlabel('Risk Measure')
                    ax7.set_ylabel('Returns (USD)')
                    ax7.grid(True, alpha=0.3)
            else:
                ax7.text(0.5, 0.5, 'Risk-Return\nData Not Available', 
                        ha='center', va='center', transform=ax7.transAxes,
                        fontsize=12, style='italic')

            # Panel 8: Performance Summary Statistics
            ax8 = fig.add_subplot(gs[2, 2])
            if 'market_statistics' in results and results['market_statistics']:
                stats = results['market_statistics']
                
                # Create text-based summary with error handling
                ax8.axis('off')
                ax8.text(0.5, 0.9, 'Market Statistics', ha='center', va='top', 
                        fontsize=14, fontweight='bold', transform=ax8.transAxes)
                
                metrics = ['Final Price', 'Price Change (%)', 'Max Price', 'Min Price']
                values = [
                    f"${stats.get('final_price', 0):.2f}",
                    f"{stats.get('price_change', 0)*100:.2f}%",
                    f"${stats.get('max_price', 0):.2f}",
                    f"${stats.get('min_price', 0):.2f}"
                ]
                
                for i, (metric, value) in enumerate(zip(metrics, values)):
                    y_pos = 0.7 - i * 0.15
                    ax8.text(0.1, y_pos, metric + ':', ha='left', va='center', 
                            fontweight='bold', transform=ax8.transAxes, fontsize=11)
                    ax8.text(0.9, y_pos, value, ha='right', va='center', 
                            transform=ax8.transAxes, fontsize=11)
            else:
                # Default statistics panel
                ax8.axis('off')
                ax8.text(0.5, 0.5, 'Market Statistics\nNot Available', 
                        ha='center', va='center', transform=ax8.transAxes,
                        fontsize=12, style='italic')

            # Add academic footer
            fig.text(0.5, 0.02, 
                    'Academic Implementation: Cellular Automata + Agent-Based Modeling\n'
                    'Methodology: Wolfram (2002), Arthur et al. (1997), Bonabeau (2002)',
                    ha='center', va='bottom', fontsize=10, style='italic')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if save_path:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {save_path}")
                except Exception as e:
                    print(f"Warning: Could not save plot - {str(e)}")
            
            plt.show()

        except Exception as e:
            print(f"Error in plot_simulation_results: {str(e)}")
            # Create a simple fallback plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error generating visualization:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title('Visualization Error')
            plt.show()

    def plot_ca_evolution(self, ca_states, save_path=None):
        """
        Visualize cellular automaton evolution with academic presentation.
        
        CA evolution visualization implementing Wolfram (2002) methodology
        for displaying complex system dynamics.
        
        Args:
            ca_states (list): List of CA grid states over time
            save_path (str): File path for saving
            
        References:
            Wolfram, S. (2002). A New Kind of Science. Wolfram Media.
        """
        try:
            # Input validation
            if not ca_states or not isinstance(ca_states, (list, tuple)):
                print("Warning: CA states must be a non-empty list")
                return
                
            # Filter out invalid states
            valid_states = []
            for state in ca_states:
                if state is not None and hasattr(state, 'shape') and len(state.shape) == 2:
                    valid_states.append(state)
                    
            if not valid_states:
                print("Warning: No valid CA states found")
                return

            # Academic figure layout
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle('Cellular Automaton Evolution: Market Sentiment Dynamics\n'
                        'Based on Wolfram (2002) Complex Systems Theory', 
                        fontsize=16, fontweight='bold')

            axes = axes.flatten()
            
            # Academic color mapping for CA states
            colors = ['#d62728', '#808080', '#2ca02c']  # Red, Gray, Green
            labels = ['Bearish', 'Neutral', 'Bullish']
            cmap = plt.matplotlib.colors.ListedColormap(colors)

            # Select representative time steps
            num_plots = min(6, len(valid_states))
            if num_plots == 0:
                print("Warning: No valid states to plot")
                return
                
            time_steps = np.linspace(0, len(valid_states)-1, num_plots, dtype=int)

            for i, step in enumerate(time_steps):
                ax = axes[i]
                grid = np.array(valid_states[step])
                
                # Ensure grid values are in valid range
                grid = np.clip(grid, -1, 1)
                
                # Create heatmap visualization
                im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect='equal', interpolation='nearest')
                ax.set_title(f'Evolution Step {step}\n'
                           f'Market Sentiment Configuration', fontweight='bold', fontsize=11)
                
                # Remove axis ticks for cleaner appearance
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Statistical annotations
                try:
                    bullish_ratio = np.sum(grid == 1) / grid.size if grid.size > 0 else 0
                    bearish_ratio = np.sum(grid == -1) / grid.size if grid.size > 0 else 0
                    neutral_ratio = np.sum(grid == 0) / grid.size if grid.size > 0 else 0
                    
                    # Add statistical text box
                    stats_text = (f'Bullish: {bullish_ratio:.1%}\n'
                                f'Bearish: {bearish_ratio:.1%}\n'
                                f'Neutral: {neutral_ratio:.1%}')
                    
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                except Exception as e:
                    # Fallback for statistical calculation errors
                    ax.text(0.02, 0.98, 'Stats\nUnavailable', transform=ax.transAxes,
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Hide unused subplots
            for i in range(num_plots, len(axes)):
                axes[i].axis('off')

            # Add shared colorbar
            try:
                cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                                  fraction=0.05, pad=0.08, shrink=0.8)
                cbar.set_ticks([-1, 0, 1])
                cbar.set_ticklabels(labels)
                cbar.set_label('Market Sentiment State', fontweight='bold')
            except:
                pass

            # Academic citation
            fig.text(0.5, 0.02, 
                    'Methodology: Wolfram (2002) Cellular Automata for Complex Systems',
                    ha='center', va='bottom', fontsize=10, style='italic')

            plt.tight_layout(rect=[0, 0.08, 1, 0.92])
            
            if save_path:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"CA evolution plot saved to {save_path}")
                except Exception as e:
                    print(f"Warning: Could not save CA evolution plot - {str(e)}")
                    
            plt.show()

        except Exception as e:
            print(f"Error in plot_ca_evolution: {str(e)}")
            # Create fallback visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error generating CA evolution plot:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title('CA Evolution Visualization Error')
            plt.show()

    def create_interactive_dashboard(self, results, save_path=None):
        """
        Create interactive dashboard using Plotly for dynamic analysis.
        
        Interactive visualization implementing modern web standards
        for scientific data exploration and presentation.
        
        Args:
            results (dict): Simulation results
            save_path (str): HTML file save path for dashboard
        """
        try:
            # Input validation
            if not isinstance(results, dict):
                print("Warning: Results must be a dictionary")
                return
                
            # Create comprehensive subplot layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Price Evolution with Confidence Bands',
                    'Agent Performance Matrix', 
                    'Return Distribution Analysis',
                    'Volatility Surface',
                    'CA Signal Dynamics',
                    'Market Regime Analysis'
                ),
                specs=[[{"secondary_y": True}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "heatmap"}],
                       [{"secondary_y": True}, {"type": "scatter"}]]
            )

            # Panel 1: Enhanced Price Evolution
            if 'price_series' in results and results['price_series'] is not None:
                price_series = np.array(results['price_series'])
                if len(price_series) > 0:
                    dates = pd.date_range(start='2023-01-01', periods=len(price_series), freq='D')
                    
                    # Main price line
                    fig.add_trace(
                        go.Scatter(x=dates, y=price_series, mode='lines', name='Simulated Price',
                                 line=dict(color='#1f77b4', width=3),
                                 hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'),
                        row=1, col=1
                    )
                    
                    # Add confidence bands if we have enough data
                    if len(price_series) > 20:
                        try:
                            returns = np.diff(price_series) / price_series[:-1]
                            returns = returns[np.isfinite(returns)]
                            if len(returns) > 0:
                                volatility = np.std(returns)
                                upper_band = [p * (1 + 2*volatility) for p in price_series]
                                lower_band = [p * (1 - 2*volatility) for p in price_series]
                                
                                fig.add_trace(
                                    go.Scatter(x=dates, y=upper_band, mode='lines', 
                                             line=dict(width=0), showlegend=False, name='Upper'),
                                    row=1, col=1
                                )
                                fig.add_trace(
                                    go.Scatter(x=dates, y=lower_band, mode='lines', 
                                             line=dict(width=0), fill='tonexty', 
                                             fillcolor='rgba(31, 119, 180, 0.2)',
                                             showlegend=True, name='Confidence Band'),
                                    row=1, col=1
                                )
                        except:
                            pass  # Skip confidence bands if calculation fails

            # Panel 2: Agent Performance Matrix
            if 'agent_performances' in results and results['agent_performances']:
                try:
                    agent_types = list(results['agent_performances'].keys())
                    pnl_values = []
                    win_rates = []
                    
                    for agent in agent_types:
                        pnl = results['agent_performances'][agent].get('total_pnl', 0)
                        win_rate = results['agent_performances'][agent].get('win_rate', 0) * 100
                        pnl_values.append(pnl)
                        win_rates.append(win_rate)
                    
                    if agent_types and pnl_values:
                        # Color-coded bars by performance
                        colors = ['red' if pnl < 0 else 'green' for pnl in pnl_values]
                        
                        fig.add_trace(
                            go.Bar(x=agent_types, y=pnl_values, name='P&L Performance',
                                  marker_color=colors, 
                                  customdata=win_rates,
                                  hovertemplate='Agent: %{x}<br>P&L: $%{y:.2f}<br>Win Rate: %{customdata:.1f}%<extra></extra>'),
                            row=1, col=2
                        )
                except:
                    pass

            # Panel 3: Interactive Return Distribution
            if 'price_series' in results and results['price_series'] is not None:
                try:
                    price_series = np.array(results['price_series'])
                    if len(price_series) > 1:
                        returns = np.diff(price_series) / price_series[:-1]
                        returns = returns[np.isfinite(returns)]
                        
                        if len(returns) > 0:
                            fig.add_trace(
                                go.Histogram(x=returns, nbinsx=min(40, max(10, len(returns)//10)), 
                                           name='Return Distribution',
                                           histnorm='probability density', opacity=0.7,
                                           marker_color='lightblue'),
                                row=2, col=1
                            )
                except:
                    pass

            # Panel 4: CA Signal Evolution
            if 'data_collector' in results and results['data_collector'] is not None:
                try:
                    model_data = results['data_collector'].get_model_vars_dataframe()
                    if isinstance(model_data, pd.DataFrame) and 'CA_Signal' in model_data.columns:
                        ca_signal_data = model_data['CA_Signal'].dropna()
                        if len(ca_signal_data) > 0:
                            fig.add_trace(
                                go.Scatter(x=ca_signal_data.index, y=ca_signal_data,
                                         mode='lines+markers', name='CA Signal',
                                         line=dict(color='purple', width=2),
                                         hovertemplate='Step: %{x}<br>Signal: %{y:.4f}<extra></extra>'),
                                row=3, col=1
                            )
                            
                            # Add zero reference line
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                                        row=3, col=1)
                except:
                    pass

            # Panel 5: Market Regime Analysis (simplified scatter plot)
            if 'agent_performances' in results and results['agent_performances']:
                try:
                    agent_types = list(results['agent_performances'].keys())
                    x_values = list(range(len(agent_types)))
                    y_values = [results['agent_performances'][agent].get('total_pnl', 0) 
                               for agent in agent_types]
                    
                    if len(x_values) > 0 and len(y_values) > 0:
                        fig.add_trace(
                            go.Scatter(x=x_values, y=y_values, mode='markers+text',
                                     text=agent_types, textposition="top center",
                                     marker=dict(size=10, color='blue', opacity=0.7),
                                     name='Agent Performance',
                                     hovertemplate='Agent: %{text}<br>P&L: $%{y:.2f}<extra></extra>'),
                            row=3, col=2
                        )
                except:
                    pass

            # Update layout with academic styling
            fig.update_layout(
                title=dict(
                    text='Interactive Hybrid Gold Price Prediction Dashboard<br>'
                         '<sub>Cellular Automata + Agent-Based Modeling Analysis</sub>',
                    font=dict(size=20, family='Times New Roman'),
                    x=0.5
                ),
                template='plotly_white',
                showlegend=True,
                height=1000,
                font=dict(family='Times New Roman', size=12)
            )

            # Update axes labels with error handling
            try:
                fig.update_xaxes(title_text="Trading Days", row=1, col=1)
                fig.update_yaxes(title_text="Gold Price (USD/oz)", row=1, col=1)
                fig.update_xaxes(title_text="Agent Type", row=1, col=2)
                fig.update_yaxes(title_text="P&L (USD)", row=1, col=2)
                fig.update_xaxes(title_text="Daily Returns", row=2, col=1)
                fig.update_yaxes(title_text="Density", row=2, col=1)
                fig.update_xaxes(title_text="Time Step", row=3, col=1)
                fig.update_yaxes(title_text="CA Signal Strength", row=3, col=1)
                fig.update_xaxes(title_text="Agent Index", row=3, col=2)
                fig.update_yaxes(title_text="Performance", row=3, col=2)
            except:
                pass

            # Save and display
            if save_path:
                try:
                    fig.write_html(save_path)
                    print(f"Interactive dashboard saved to {save_path}")
                except Exception as e:
                    print(f"Warning: Could not save dashboard - {str(e)}")
                    
            fig.show()

        except Exception as e:
            print(f"Error in create_interactive_dashboard: {str(e)}")
            # Create simple fallback dashboard
            try:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error creating dashboard:<br>{str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(
                    title="Dashboard Generation Error",
                    template='plotly_white'
                )
                fig.show()
            except:
                print("Could not create fallback dashboard")