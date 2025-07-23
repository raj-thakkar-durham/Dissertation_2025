# HYBRID GOLD PRICE PREDICTION SIMULATION - PROJECT DOCUMENTATION

## Overview
This project implements a hybrid cellular automata (CA) and agent-based model (ABM) system for gold price prediction and investment analysis. The implementation follows academic research methodology combining Wolfram's cellular automata principles with Bonabeau's agent-based modeling approach.

## Project Structure

```
Dissertation_2025/
├── data/                  # Data collection and processing modules
├── models/               # Core simulation models (CA and ABM)
├── utils/                # Utility functions and tools
├── results/              # Analysis and visualization modules
├── header.py             # Common imports and configurations
├── main_simulation.py    # Main simulation engine
├── instruction_manual.txt # Step-by-step implementation guide
└── README.md            # Project description
```

---

## FILE-BY-FILE DOCUMENTATION

### 1. header.py
**Purpose**: Common imports, configurations, and utility functions

**Key Components**:
- **Imports**: All necessary libraries (pandas, numpy, mesa, sklearn, etc.)
- **DEFAULT_CONFIG**: Global configuration dictionary with simulation parameters
- **Utility Functions**:
  - `set_random_seed(seed)`: Sets reproducible random seeds
  - `discretize_signal(value, thresholds)`: Converts continuous values to discrete states
  - `calculate_returns(prices)`: Calculates log returns from price series
  - `moving_average(data, window)`: Calculates rolling averages

**Academic References**: Wolfram (2002), Bonabeau (2002), Arthur et al. (1997)

---

### 2. main_simulation.py
**Purpose**: Main simulation engine orchestrating the entire hybrid CA-ABM system

**Class**: `HybridGoldSimulation`

**Key Methods**:

#### Core Simulation Pipeline
- **`__init__(config)`**: Initializes simulation with configuration parameters
- **`load_and_prepare_data()`**: Loads historical market data and applies feature engineering
- **`initialize_models()`**: Sets up CA and ABM models with proper integration
- **`calibrate_models()`**: Optimizes CA rules and calibrates agent parameters

#### Execution Methods
- **`run_single_simulation(seed)`**: Executes one complete simulation run
- **`run_parallel_simulations(num_runs)`**: Runs multiple simulations in parallel
- **`run_complete_simulation()`**: Full pipeline execution with validation

#### Validation and Analysis
- **`run_historical_validation(start_date, end_date)`**: 10-year historical backtesting
- **`validate_results()`**: Compares simulated vs actual prices
- **`validate_predictions(test_data)`**: Model accuracy assessment
- **`calculate_performance_metrics()`**: Statistical performance evaluation

#### Investment Analysis
- **`demonstrate_investment_insights()`**: Generates investment decision support
- **`analyze_investment_performance()`**: Portfolio performance analysis
- **`analyze_feature_importance()`**: Feature contribution analysis
- **`calculate_risk_metrics()`**: Risk assessment calculations

#### Reporting
- **`generate_visualizations()`**: Creates comprehensive plots and charts
- **`generate_report()`**: Produces detailed analysis reports
- **`print_investment_summary()`**: Displays key investment insights

**Academic Methodology**: Based on Diebold & Mariano (1995) forecast evaluation framework

---

### 3. data/data_collection.py
**Purpose**: Historical market data fetching and preprocessing

**Class**: `DataCollector`

**Key Methods**:

#### Data Fetching
- **`__init__(start_date, end_date)`**: Initializes with date range
- **`fetch_gold_data()`**: Retrieves gold prices from GLD ETF or GC=F futures
- **`fetch_oil_data()`**: Gets Brent oil (BZ=F) or WTI (CL=F) prices
- **`fetch_market_indices()`**: Collects S&P 500 and USD Index data

#### Data Processing
- **`merge_market_data()`**: Combines all market data into unified DataFrame
- **`collect_all_data(start_date, end_date)`**: Comprehensive data collection pipeline

**Data Sources**:
- Yahoo Finance (yfinance library)
- Gold: GLD ETF, GC=F futures
- Oil: BZ=F (Brent), CL=F (WTI)
- Indices: ^GSPC (S&P 500), DX-Y.NYB (USD Index)

**Error Handling**: Automatic fallback to synthetic data if real data unavailable

---

### 4. data/news_analysis.py
**Purpose**: News sentiment analysis for market sentiment quantification

**Class**: `SentimentAnalyzer`

**Key Methods**:

#### Sentiment Processing
- **`__init__()`**: Initializes VADER sentiment analyzer with gold-specific lexicon
- **`fetch_news_headlines(date)`**: Generates contextual headlines for given date
- **`calculate_daily_sentiment(headlines)`**: Processes headlines through VADER
- **`_add_gold_specific_lexicon()`**: Enhances VADER with gold market terminology

#### Analysis Functions
- **`generate_sentiment_series(date_range)`**: Creates daily sentiment scores
- **`analyze_sentiment_trends(sentiment_data)`**: Identifies sentiment patterns
- **`analyze_sentiment_impact(sentiment_data, price_data)`**: Correlates sentiment with prices
- **`generate_investment_signals(sentiment_data)`**: Creates trading signals from sentiment

#### Data Management
- **`save_sentiment_data(sentiment_df, filename)`**: Exports sentiment analysis
- **`load_sentiment_data(filename)`**: Imports previously saved sentiment data

**Academic Framework**: Based on Hutto & Gilbert (2014) VADER methodology, Tetlock (2007) media sentiment research

**Gold-Specific Enhancements**:
- Positive terms: surge, rally, safe haven, hedge, inflation protection
- Negative terms: decline, bearish, dollar strength, rate hike, sell-off

---

### 5. models/cellular_automaton.py
**Purpose**: Cellular automata implementation for market sentiment modeling

**Class**: `CellularAutomaton`

**Key Methods**:

#### Grid Management
- **`__init__(grid_size, num_states)`**: Initializes CA grid with specified dimensions
- **`initialize_grid(initial_state)`**: Sets initial grid state from market features
- **`get_neighbors(grid, x, y)`**: Returns 8-neighborhood values (static, JIT-compiled)
- **`get_grid_statistics()`**: Computes grid state statistics

#### Rule Definition and Evolution
- **`define_rules(rule_parameters)`**: Sets CA update rules from parameters
- **`update_cell(x, y, external_features)`**: Updates single cell based on rules
- **`step(external_features)`**: Evolves entire grid one time step
- **`run_steps(num_steps, external_data)`**: Runs multiple evolution steps

#### Signal Generation
- **`get_market_signal()`**: Aggregates grid state into market signal (-1 to 1)
- **`discretize_features(features)`**: Converts continuous features to discrete states
- **`map_to_discrete_state(value)`**: Maps continuous values to CA states

**Technical Features**:
- JIT compilation with Numba for performance
- Moore neighborhood (8-connected)
- State space: {-1, 0, 1} representing bearish, neutral, bullish
- Grid sizes: Configurable (default 50x50)

**Academic Foundation**: Wolfram (2002) cellular automata principles for complex systems

---

### 6. models/agents.py
**Purpose**: Agent-based modeling with different trader types

**Classes**:

#### Base Class: `GoldTrader`
**Purpose**: Abstract base class for all trader agents

**Key Methods**:
- **`__init__(unique_id, model, agent_type)`**: Initializes agent with trading parameters
- **`observe_environment()`**: Gathers market observations (CA signals, prices, neighbors)
- **`make_decision(observations)`**: Abstract decision-making method
- **`execute_trade(action, price)`**: Executes buy/sell/hold decisions
- **`get_portfolio_value(current_price)`**: Calculates total portfolio worth
- **`update_statistics()`**: Updates trading performance metrics

#### Specialized Agent Types:

**`HerderAgent`**: 
- **Behavior**: Follows majority of neighboring agents
- **Decision Logic**: Mimics crowd behavior, buys when others buy
- **Academic Basis**: Bikhchandani et al. (1992) herding theory

**`ContrarianAgent`**:
- **Behavior**: Acts opposite to market sentiment
- **Decision Logic**: Buys when others sell, contrarian strategy
- **Academic Basis**: Lakonishok et al. (1994) contrarian investment strategies

**`TrendFollowerAgent`**:
- **Behavior**: Follows price momentum and trends
- **Decision Logic**: Momentum-based trading using technical indicators
- **Academic Basis**: Jegadeesh & Titman (1993) momentum strategies

**`NoiseTraderAgent`**:
- **Behavior**: Random trading with sentiment bias
- **Decision Logic**: Partially random decisions with market noise
- **Academic Basis**: De Long et al. (1990) noise trader theory

**Agent Parameters**:
- Position: {-1, 0, 1} for short, neutral, long
- Cash and gold holdings
- Risk tolerance: [0.1, 0.9]
- Sentiment bias: [-0.5, 0.5]

---

### 7. models/market_model.py
**Purpose**: Agent-based market model coordinating all trading agents

**Class**: `GoldMarketModel`

**Key Methods**:

#### Model Setup
- **`__init__(num_agents, grid_size, ca_model)`**: Initializes market with agents and CA
- **`create_agents()`**: Creates and distributes different agent types on grid
- **`place_agents_on_grid()`**: Spatial distribution of agents

#### Market Dynamics
- **`update_price()`**: Calculates price changes from supply/demand
- **`calculate_net_demand()`**: Aggregates all agent trading decisions
- **`apply_market_impact(net_demand)`**: Translates demand to price movement
- **`calculate_volatility()`**: Computes market volatility measures

#### Simulation Execution
- **`step()`**: Executes one simulation day (CA update + agent actions + price update)
- **`run_simulation(num_days, external_data)`**: Complete simulation run
- **`reset_daily_statistics()`**: Resets daily trading counters

#### Data Collection
- **`get_volume()`**: Calculates daily trading volume
- **`get_ca_signal()`**: Retrieves current CA market signal
- **`collect_agent_data()`**: Gathers agent statistics

**Market Mechanisms**:
- Price impact function based on net demand
- Transaction costs and bid-ask spreads
- Market maker liquidity provision
- Volatility clustering effects

**Academic Framework**: Arthur et al. (1997) Santa Fe artificial stock market

---

### 8. models/ca_optimizer.py
**Purpose**: Optimizes cellular automata rules for historical data fitting

**Class**: `CARuleOptimizer`

**Key Methods**:

#### Optimization Setup
- **`__init__(ca_model, historical_data)`**: Initializes with CA model and training data
- **`objective_function(rule_parameters)`**: Evaluates rule performance on historical data
- **`prepare_training_data()`**: Preprocesses data for optimization

#### Optimization Algorithms
- **`optimize_rules(method)`**: Main optimization using differential evolution
- **`grid_search_optimization(param_ranges)`**: Grid search over parameter space
- **`genetic_algorithm_optimization()`**: Alternative genetic algorithm approach

#### Validation
- **`validate_rules(test_data)`**: Tests optimized rules on out-of-sample data
- **`cross_validate_rules(k_folds)`**: K-fold cross-validation
- **`calculate_fitness_metrics(predicted, actual)`**: Performance evaluation

#### Rule Management
- **`encode_rules(rule_dict)`**: Converts rule dictionary to parameter vector
- **`decode_rules(parameter_vector)`**: Converts parameters back to rules
- **`save_optimized_rules(filename)`**: Exports optimized rules
- **`load_optimized_rules(filename)`**: Imports saved rules

**Optimization Metrics**:
- Price prediction accuracy
- Directional accuracy
- Correlation with actual prices
- Risk-adjusted returns

**Academic Methods**: Differential evolution (Storn & Price, 1997), genetic algorithms

---

### 9. utils/feature_engineering.py
**Purpose**: Creates engineered features from raw market data

**Class**: `FeatureEngineering`

**Key Methods**:

#### Basic Features
- **`__init__(data)`**: Initializes with market data DataFrame
- **`calculate_returns()`**: Computes log returns for gold and oil
- **`calculate_moving_averages(windows)`**: Creates MA indicators (5, 10, 20 day)
- **`calculate_volatility(window)`**: Rolling volatility calculation

#### Technical Indicators
- **`create_technical_indicators()`**: RSI, MACD, Bollinger Bands
- **`calculate_rsi(window)`**: Relative Strength Index
- **`calculate_macd()`**: Moving Average Convergence Divergence
- **`calculate_bollinger_bands(window, std_dev)`**: Price envelope indicators

#### CA-Specific Features
- **`create_ca_grid_features()`**: Maps market data to CA grid states
- **`create_neighbor_features()`**: Creates 8-neighborhood feature representation
- **`discretize_continuous_features()`**: Converts continuous to discrete states

#### Data Processing
- **`normalize_features()`**: Standardizes features using StandardScaler
- **`handle_missing_values()`**: Fills missing data with appropriate methods
- **`create_lagged_features(lags)`**: Creates time-lagged variables
- **`get_feature_summary()`**: Returns summary statistics of all features

**Feature Categories**:
- Price-based: returns, moving averages, volatility
- Technical: RSI, MACD, Bollinger Bands
- Market: sentiment, volume, correlation
- CA-specific: grid states, neighborhood patterns

**Academic Foundation**: Technical analysis literature, Fama & French (1993) factor models

---

### 10. utils/parallel_runner.py
**Purpose**: Parallel execution framework for efficient simulation runs

**Class**: `ParallelSimulationRunner`

**Key Methods**:

#### Parallel Execution
- **`__init__(model_class, num_cores)`**: Initializes with model class and core count
- **`run_single_simulation(params, run_id)`**: Executes one simulation run
- **`run_batch_simulations(parameters, num_iterations)`**: Mesa BatchRunner integration
- **`run_monte_carlo_simulation(base_params, num_runs)`**: Monte Carlo analysis

#### Parameter Studies
- **`run_parameter_sweep(parameter_ranges)`**: Systematic parameter exploration
- **`optimize_parameters(objective_function)`**: Parameter optimization
- **`sensitivity_analysis(parameters, perturbation)`**: Parameter sensitivity testing

#### Result Management
- **`collect_results(futures)`**: Aggregates parallel execution results
- **`save_results(results, filename)`**: Exports simulation results
- **`load_results(filename)`**: Imports saved results
- **`merge_result_batches(result_list)`**: Combines multiple result sets

#### Performance Monitoring
- **`monitor_execution_progress()`**: Tracks simulation progress
- **`estimate_completion_time(completed, total, start_time)`**: ETA calculation
- **`log_performance_metrics()`**: Performance statistics logging

**Technical Features**:
- Joblib parallel processing
- Mesa BatchRunner integration
- Memory-efficient result handling
- Progress monitoring and logging

**Performance Benefits**:
- Multi-core CPU utilization
- Scalable to cluster computing
- Memory optimization for large parameter sweeps

---

### 11. results/result_analyzer.py
**Purpose**: Comprehensive analysis of simulation results

**Class**: `ResultsAnalyzer`

**Key Methods**:

#### Statistical Analysis
- **`__init__(simulation_results, actual_data)`**: Initializes with simulation and actual data
- **`calculate_metrics()`**: RMSE, MAE, MAPE, directional accuracy
- **`statistical_tests()`**: Jarque-Bera, Shapiro-Wilk, t-tests
- **`correlation_analysis()`**: Correlation matrices and significance tests

#### Performance Evaluation
- **`analyze_forecast_accuracy()`**: Forecast error analysis
- **`directional_accuracy_analysis()`**: Up/down movement prediction
- **`volatility_analysis()`**: Volatility clustering and GARCH effects
- **`risk_return_analysis()`**: Sharpe ratio, maximum drawdown

#### Agent Analysis
- **`analyze_agent_behavior()`**: Agent performance and behavior patterns
- **`agent_type_comparison()`**: Performance comparison across agent types
- **`position_analysis()`**: Portfolio position evolution
- **`trading_pattern_analysis()`**: Trading frequency and patterns

#### Model Validation
- **`backtesting_analysis()`**: Out-of-sample performance testing
- **`rolling_window_validation()`**: Time-series cross-validation
- **`regime_change_analysis()`**: Performance across market regimes

#### Reporting
- **`generate_report(save_path)`**: Comprehensive analysis report
- **`export_results(output_dir)`**: Detailed result exports
- **`create_summary_statistics()`**: Key metric summaries

**Statistical Tests**:
- Normality: Jarque-Bera, Shapiro-Wilk
- Stationarity: Augmented Dickey-Fuller
- Heteroscedasticity: Breusch-Pagan
- Autocorrelation: Ljung-Box

**Academic Framework**: Financial econometrics, Diebold & Mariano (1995) forecast evaluation

---

### 12. results/plots.py
**Purpose**: Visualization tools for simulation results

**Class**: `VisualizationTools`

**Key Methods**:

#### Setup and Configuration
- **`__init__()`**: Initializes plotting environment and styles
- **`setup_style()`**: Configures matplotlib and seaborn styles
- **`create_color_palette()`**: Defines consistent color schemes

#### Simulation Visualizations
- **`plot_simulation_results(results, actual_data, save_path)`**: Main results comparison
- **`plot_price_comparison()`**: Simulated vs actual price series
- **`plot_ca_evolution(ca_states, save_path)`**: CA grid evolution over time
- **`plot_agent_positions()`**: Agent position distributions

#### Performance Plots
- **`plot_error_metrics()`**: Error analysis visualizations
- **`plot_correlation_matrix()`**: Feature and result correlations
- **`plot_returns_distribution()`**: Return distribution analysis
- **`plot_volatility_clustering()`**: GARCH and volatility patterns

#### Interactive Dashboards
- **`create_interactive_dashboard(results, save_path)`**: Plotly dashboard
- **`create_agent_animation()`**: Animated agent behavior
- **`create_ca_heatmap_animation()`**: CA grid evolution animation

#### Statistical Plots
- **`plot_residual_analysis()`**: Forecast error analysis
- **`plot_qq_plots()`**: Quantile-quantile plots for normality
- **`plot_acf_pacf()`**: Autocorrelation function plots

#### Export Functions
- **`save_all_plots(results, output_dir)`**: Batch plot generation
- **`export_to_pdf(plots, filename)`**: PDF report generation
- **`create_presentation_slides()`**: Automated slide generation

**Visualization Libraries**:
- Matplotlib: Static plots
- Seaborn: Statistical visualizations
- Plotly: Interactive dashboards
- Animation support for dynamic visualizations

---

## DATA FLOW AND INTEGRATION

### 1. Data Pipeline
```
Raw Data (Yahoo Finance) → DataCollector → FeatureEngineering → Simulation Models
```

### 2. Model Integration
```
Market Data → CA Grid → Market Signals → Agent Decisions → Price Updates
```

### 3. Analysis Pipeline
```
Simulation Results → ResultsAnalyzer → Statistical Tests → VisualizationTools → Reports
```

---

## ACADEMIC METHODOLOGY

### Cellular Automata Theory
- **Foundation**: Wolfram (2002) cellular automata for complex systems
- **Implementation**: Moore neighborhood, discrete state space
- **Rules**: Optimized using differential evolution

### Agent-Based Modeling
- **Foundation**: Bonabeau (2002) agent-based modeling principles
- **Agents**: Herders, contrarians, trend followers, noise traders
- **Market**: Arthur et al. (1997) Santa Fe artificial stock market

### Financial Validation
- **Metrics**: Diebold & Mariano (1995) forecast evaluation
- **Testing**: Rolling window backtesting, regime analysis
- **Risk**: Value-at-Risk, maximum drawdown, Sharpe ratio

### Sentiment Analysis
- **Method**: Hutto & Gilbert (2014) VADER sentiment analysis
- **Enhancement**: Gold-specific lexicon and market terms
- **Validation**: Tetlock (2007) media sentiment correlation

---

## CONFIGURATION AND PARAMETERS

### Default Configuration (header.py)
```python
DEFAULT_CONFIG = {
    'start_date': '2014-01-01',      # 10-year analysis period
    'end_date': '2024-01-01',
    'ca_grid_size': (20, 20),        # CA grid dimensions
    'ca_num_states': 3,              # States: {-1, 0, 1}
    'num_agents': 100,               # Number of trading agents
    'agent_grid_size': (15, 15),     # Agent spatial grid
    'initial_gold_price': 1800,      # Starting price (USD/oz)
    'simulation_days': 252,          # Trading days per year
    'num_parallel_runs': 50,         # Monte Carlo runs
    'optimization_cores': 2,         # CPU cores for optimization
    'parallel_cores': mp.cpu_count(), # All available cores
    'random_seed': 42                # Reproducibility
}
```

---

## PERFORMANCE AND SCALABILITY

### Optimization Features
- **JIT Compilation**: Numba acceleration for CA operations
- **Parallel Processing**: Multi-core simulation execution
- **Memory Management**: Efficient data handling for large datasets
- **Vectorization**: NumPy operations for mathematical computations

### Scalability
- **Grid Sizes**: Configurable from 10x10 to 100x100
- **Agent Counts**: Scalable from 50 to 1000+ agents
- **Time Periods**: Multi-year analysis capability
- **Parameter Sweeps**: Distributed computing ready

---

## RESEARCH APPLICATIONS

### Academic Research
- Market microstructure analysis
- Behavioral finance modeling
- Complex adaptive systems study
- Computational finance methods

### Investment Applications
- Portfolio optimization
- Risk management
- Market timing strategies
- Sentiment-driven trading

### Model Validation
- Historical backtesting (10-year period)
- Out-of-sample testing
- Cross-validation techniques
- Regime analysis across market conditions

---

This documentation provides a comprehensive understanding of the hybrid CA-ABM gold price prediction system, its academic foundations, and practical applications for both research and investment analysis.
