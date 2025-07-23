# FUNCTION REFERENCE GUIDE - HYBRID GOLD PRICE PREDICTION SIMULATION

## Table of Contents
1. [Data Collection Functions](#data-collection-functions)
2. [News Analysis Functions](#news-analysis-functions)
3. [Cellular Automata Functions](#cellular-automata-functions)
4. [Agent-Based Model Functions](#agent-based-model-functions)
5. [Feature Engineering Functions](#feature-engineering-functions)
6. [Optimization Functions](#optimization-functions)
7. [Parallel Processing Functions](#parallel-processing-functions)
8. [Analysis Functions](#analysis-functions)
9. [Visualization Functions](#visualization-functions)
10. [Main Simulation Functions](#main-simulation-functions)

---

## DATA COLLECTION FUNCTIONS

### DataCollector Class (`data/data_collection.py`)

#### `__init__(start_date, end_date)`
**Purpose**: Initialize data collector with date range
**Parameters**:
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date in 'YYYY-MM-DD' format
**Returns**: None
**Example**: `collector = DataCollector('2020-01-01', '2024-01-01')`

#### `fetch_gold_data()`
**Purpose**: Fetch gold prices using GLD ETF or GC=F futures
**Parameters**: None
**Returns**: `pd.DataFrame` with columns [Date, Open, High, Low, Close, Volume]
**Data Sources**: Yahoo Finance (GLD ETF, GC=F futures)
**Error Handling**: Returns empty DataFrame if fetching fails

#### `fetch_oil_data()`
**Purpose**: Fetch Brent oil prices (BZ=F) or WTI (CL=F)
**Parameters**: None
**Returns**: `pd.DataFrame` with columns [Date, Oil_Close]
**Data Sources**: BZ=F (Brent), CL=F (WTI) from Yahoo Finance
**Fallback**: Default oil price of $75/barrel if data unavailable

#### `fetch_market_indices()`
**Purpose**: Fetch S&P 500 and USD Index data
**Parameters**: None
**Returns**: `pd.DataFrame` with columns [Date, SP500_Close, USD_Close]
**Data Sources**: ^GSPC (S&P 500), DX-Y.NYB (USD Index)
**Default Values**: S&P 500: 4000, USD Index: 100

#### `merge_market_data()`
**Purpose**: Combine all market data into single DataFrame
**Parameters**: None
**Returns**: `pd.DataFrame` indexed by Date with all market data
**Processing**: Forward fill → backward fill → drop NaN values
**Fallback**: Creates synthetic data if real data unavailable

#### `collect_all_data(start_date=None, end_date=None)`
**Purpose**: Comprehensive data collection for analysis
**Parameters**:
- `start_date` (str, optional): Override start date
- `end_date` (str, optional): Override end date
**Returns**: `pd.DataFrame` with complete market dataset
**Features**: Enhanced error handling and data validation

---

## NEWS ANALYSIS FUNCTIONS

### SentimentAnalyzer Class (`data/news_analysis.py`)

#### `__init__()`
**Purpose**: Initialize VADER sentiment analyzer with gold market context
**Parameters**: None
**Setup**: 
- Loads VADER sentiment analyzer
- Defines gold-specific positive/negative terms
- Sets up news source URLs

#### `fetch_news_headlines(date)`
**Purpose**: Generate contextual news headlines for gold market analysis
**Parameters**:
- `date` (str): Date in 'YYYY-MM-DD' format
**Returns**: `list` of contextual headlines
**Method**: Template-based headline generation with market variables
**Academic Basis**: Tetlock (2007) media sentiment methodology

#### `calculate_daily_sentiment(headlines)`
**Purpose**: Process headlines through VADER sentiment analysis
**Parameters**:
- `headlines` (list): List of news headlines
**Returns**: `float` compound sentiment score (-1 to 1)
**Processing**: 
- VADER analysis for each headline
- Gold-specific term weighting
- Aggregate compound score calculation

#### `_add_gold_specific_lexicon()`
**Purpose**: Enhance VADER with gold market-specific terminology
**Parameters**: None
**Enhancement**: Adds weights for gold trading terms
**Terms**: 
- Positive: surge, rally, safe haven, hedge, inflation protection
- Negative: decline, bearish, dollar strength, rate hike, sell-off

#### `generate_sentiment_series(date_range)`
**Purpose**: Generate daily sentiment scores for entire date range
**Parameters**:
- `date_range` (pd.DatetimeIndex): Range of dates to analyze
**Returns**: `pd.DataFrame` with columns [Date, Sentiment]
**Processing**: Daily headline generation → sentiment calculation → series compilation

#### `analyze_sentiment_trends(sentiment_data)`
**Purpose**: Identify patterns and trends in sentiment data
**Parameters**:
- `sentiment_data` (pd.DataFrame): Sentiment time series
**Returns**: `dict` with trend statistics
**Analysis**: Moving averages, volatility, trend changes

#### `analyze_sentiment_impact(sentiment_data, price_data)`
**Purpose**: Correlate sentiment with price movements
**Parameters**:
- `sentiment_data` (pd.DataFrame): Sentiment time series
- `price_data` (pd.DataFrame): Price data
**Returns**: `dict` with correlation metrics
**Statistics**: Pearson/Spearman correlation, high/low sentiment returns

#### `generate_investment_signals(sentiment_data, lookback_days=20)`
**Purpose**: Create trading signals from sentiment analysis
**Parameters**:
- `sentiment_data` (pd.DataFrame): Sentiment scores
- `lookback_days` (int): Window for signal generation
**Returns**: `pd.DataFrame` with investment signals
**Signals**: buy, hold, sell based on sentiment thresholds

#### `save_sentiment_data(sentiment_df, filename)`
**Purpose**: Export sentiment analysis to file
**Parameters**:
- `sentiment_df` (pd.DataFrame): Sentiment data
- `filename` (str): Output file path
**Format**: CSV with timestamp index

#### `load_sentiment_data(filename)`
**Purpose**: Import previously saved sentiment data
**Parameters**:
- `filename` (str): Input file path
**Returns**: `pd.DataFrame` with sentiment data
**Error Handling**: Returns empty DataFrame if file not found

---

## CELLULAR AUTOMATA FUNCTIONS

### CellularAutomaton Class (`models/cellular_automaton.py`)

#### `__init__(grid_size=(50, 50), num_states=3)`
**Purpose**: Initialize cellular automaton for market sentiment modeling
**Parameters**:
- `grid_size` (tuple): CA grid dimensions
- `num_states` (int): Number of discrete states
**Setup**: Creates grid, initializes rules dictionary, sets up history tracking

#### `initialize_grid(initial_state)`
**Purpose**: Set initial grid state from market features
**Parameters**:
- `initial_state` (dict or np.ndarray): Initial configuration
**Processing**: Maps sentiment, volatility, trend to discrete states {-1, 0, 1}
**States**: -1 (bearish), 0 (neutral), 1 (bullish)

#### `get_neighbors(grid, x, y)` [Static, JIT-compiled]
**Purpose**: Get 8-neighborhood values for cell at (x, y)
**Parameters**:
- `grid` (np.ndarray): Current grid state
- `x, y` (int): Cell coordinates
**Returns**: `np.ndarray` of 8 neighbor values
**Boundary**: Periodic boundary conditions (torus topology)
**Performance**: Numba JIT compilation for speed

#### `define_rules(rule_parameters)`
**Purpose**: Set CA update rules from parameters
**Parameters**:
- `rule_parameters` (dict): Rule configuration
**Rule Types**: Majority rule, threshold rule, custom rules
**Academic Basis**: Wolfram (2002) CA rule classification

#### `update_cell(x, y, external_features)`
**Purpose**: Update single cell based on neighbors and external inputs
**Parameters**:
- `x, y` (int): Cell coordinates
- `external_features` (dict): External market data
**Returns**: New state for the cell
**Factors**: Neighbor states, external sentiment, market conditions

#### `step(external_features)`
**Purpose**: Evolve entire grid one time step
**Parameters**:
- `external_features` (dict): External market inputs
**Processing**: Synchronous update of all cells
**Performance**: Vectorized operations for efficiency

#### `run_steps(num_steps, external_data=None)`
**Purpose**: Run multiple evolution steps
**Parameters**:
- `num_steps` (int): Number of time steps
- `external_data` (dict): Time series of external data
**Returns**: Grid evolution history
**Tracking**: Saves grid state at each step

#### `get_market_signal()`
**Purpose**: Aggregate grid state into market signal
**Parameters**: None
**Returns**: `float` market signal (-1 to 1)
**Calculation**: Weighted average of grid states

#### `get_grid_statistics()`
**Purpose**: Compute statistical measures of grid state
**Parameters**: None
**Returns**: `dict` with grid statistics
**Metrics**: Mean state, clustering coefficient, spatial correlation

#### `discretize_features(features)`
**Purpose**: Convert continuous features to discrete CA states
**Parameters**:
- `features` (dict): Continuous market features
**Returns**: `dict` with discrete states
**Thresholds**: Configurable thresholds for state boundaries

#### `map_to_discrete_state(value, thresholds=[-0.33, 0.33])`
**Purpose**: Map continuous value to discrete state
**Parameters**:
- `value` (float): Continuous input value
- `thresholds` (list): State boundary thresholds
**Returns**: `int` discrete state {-1, 0, 1}

---

## AGENT-BASED MODEL FUNCTIONS

### GoldTrader Base Class (`models/agents.py`)

#### `__init__(unique_id, model, agent_type='contrarian')`
**Purpose**: Initialize gold trader agent
**Parameters**:
- `unique_id` (int): Unique agent identifier
- `model`: Mesa model instance
- `agent_type` (str): Agent behavior type
**Setup**: Position, cash, holdings, risk tolerance, sentiment bias

#### `observe_environment()`
**Purpose**: Gather market observations (CA signals, prices, neighbors)
**Parameters**: None
**Returns**: `dict` with environment observations
**Observations**: CA signal, current price, price history, neighbor actions, market sentiment

#### `make_decision(observations)` [Abstract]
**Purpose**: Decision-making method (implemented by subclasses)
**Parameters**:
- `observations` (dict): Environment observations
**Returns**: `str` action ('buy', 'sell', 'hold')
**Note**: Must be implemented by each agent type

#### `execute_trade(action, price)`
**Purpose**: Execute buy/sell/hold decisions
**Parameters**:
- `action` (str): Trading action
- `price` (float): Current market price
**Processing**: Updates position, cash, holdings based on action
**Constraints**: Cash limits, position limits, transaction costs

#### `get_portfolio_value(current_price)`
**Purpose**: Calculate total portfolio worth
**Parameters**:
- `current_price` (float): Current gold price
**Returns**: `float` total portfolio value
**Calculation**: Cash + (gold_holdings × current_price)

#### `update_statistics()`
**Purpose**: Update trading performance metrics
**Parameters**: None
**Metrics**: Trade count, profit/loss, win rate, Sharpe ratio

#### `step()`
**Purpose**: Agent's step function called each simulation day
**Parameters**: None
**Process**: Observe → Decide → Execute → Update statistics

### Specialized Agent Classes

#### HerderAgent

#### `make_decision(observations)`
**Purpose**: Follow majority of neighboring agents
**Parameters**:
- `observations` (dict): Environment data
**Returns**: `str` trading action
**Logic**: Mimics average neighbor position
**Threshold**: Follows majority if consensus > 60%

#### ContrarianAgent

#### `make_decision(observations)`
**Purpose**: Act opposite to market sentiment
**Parameters**:
- `observations` (dict): Environment data
**Returns**: `str` trading action
**Logic**: Inverts CA signal and neighbor actions
**Strategy**: Buy when others sell, sell when others buy

#### TrendFollowerAgent

#### `make_decision(observations)`
**Purpose**: Follow price momentum and trends
**Parameters**:
- `observations` (dict): Environment data
**Returns**: `str` trading action
**Logic**: Moving average crossovers, momentum indicators
**Indicators**: Price trend, recent returns, volatility

#### NoiseTraderAgent

#### `make_decision(observations)`
**Purpose**: Random trading with sentiment bias
**Parameters**:
- `observations` (dict): Environment data
**Returns**: `str` trading action
**Logic**: Partially random with market sentiment influence
**Randomness**: 50% random, 50% sentiment-driven

---

## MARKET MODEL FUNCTIONS

### GoldMarketModel Class (`models/market_model.py`)

#### `__init__(num_agents, grid_size, ca_model)`
**Purpose**: Initialize agent-based market model
**Parameters**:
- `num_agents` (int): Number of trading agents
- `grid_size` (tuple): Spatial grid for agents
- `ca_model`: Cellular automaton instance
**Setup**: Grid, scheduler, data collector, market parameters

#### `create_agents()`
**Purpose**: Create and distribute different agent types on grid
**Parameters**: None
**Distribution**: 30% herders, 25% contrarians, 25% trend followers, 20% noise traders
**Placement**: Random spatial distribution

#### `place_agents_on_grid()`
**Purpose**: Spatial distribution of agents
**Parameters**: None
**Method**: Random placement with collision avoidance

#### `update_price()`
**Purpose**: Calculate price changes from supply/demand
**Parameters**: None
**Method**: Aggregate net demand → price impact function → price update
**Factors**: Net demand, volatility, liquidity, market maker spread

#### `calculate_net_demand()`
**Purpose**: Aggregate all agent trading decisions
**Parameters**: None
**Returns**: `float` net market demand
**Calculation**: Sum of all agent position changes

#### `apply_market_impact(net_demand)`
**Purpose**: Translate demand to price movement
**Parameters**:
- `net_demand` (float): Aggregate demand
**Returns**: `float` price change
**Function**: Non-linear impact function with liquidity constraints

#### `calculate_volatility(window=20)`
**Purpose**: Compute market volatility measures
**Parameters**:
- `window` (int): Rolling window size
**Returns**: `float` volatility measure
**Method**: Rolling standard deviation of returns

#### `step()`
**Purpose**: Execute one simulation day
**Parameters**: None
**Process**: CA update → agent actions → price update → data collection
**Synchronization**: Ensures proper order of operations

#### `run_simulation(num_days, external_data=None)`
**Purpose**: Complete simulation run
**Parameters**:
- `num_days` (int): Number of simulation days
- `external_data` (dict): External market data
**Returns**: `dict` with simulation results
**Output**: Price series, agent data, market statistics

#### `reset_daily_statistics()`
**Purpose**: Reset daily trading counters
**Parameters**: None
**Reset**: Volume, trades, buy/sell pressure

#### `get_volume()`
**Purpose**: Calculate daily trading volume
**Parameters**: None
**Returns**: `float` total trading volume
**Calculation**: Sum of all executed trades

#### `get_ca_signal()`
**Purpose**: Retrieve current CA market signal
**Parameters**: None
**Returns**: `float` CA signal (-1 to 1)
**Source**: Cellular automaton market signal

#### `collect_agent_data()`
**Purpose**: Gather agent statistics
**Parameters**: None
**Returns**: `dict` with agent performance data
**Data**: Positions, cash, portfolio values, trade counts

---

## FEATURE ENGINEERING FUNCTIONS

### FeatureEngineering Class (`utils/feature_engineering.py`)

#### `__init__(data)`
**Purpose**: Initialize feature engineering with market data
**Parameters**:
- `data` (pd.DataFrame): Market data
**Setup**: Copy data, initialize scaler, record original columns

#### `calculate_returns()`
**Purpose**: Calculate log returns for gold and oil
**Parameters**: None
**Returns**: `pd.DataFrame` with return columns
**Formula**: log(price_t / price_t-1)
**Columns**: gold_return, oil_return

#### `calculate_moving_averages(windows=[5, 10, 20])`
**Purpose**: Create moving average indicators
**Parameters**:
- `windows` (list): Window sizes for MA calculation
**Returns**: `pd.DataFrame` with MA columns
**Columns**: gold_ma_5, gold_ma_10, gold_ma_20, gold_ma_X_ratio

#### `calculate_volatility(window=20)`
**Purpose**: Calculate rolling volatility
**Parameters**:
- `window` (int): Rolling window size
**Returns**: `pd.DataFrame` with volatility column
**Method**: Rolling standard deviation of returns
**Column**: gold_volatility

#### `create_technical_indicators()`
**Purpose**: Generate RSI, MACD, Bollinger Bands
**Parameters**: None
**Returns**: `pd.DataFrame` with technical indicators
**Indicators**: RSI(14), MACD(12,26,9), Bollinger Bands(20,2)

#### `calculate_rsi(window=14)`
**Purpose**: Relative Strength Index calculation
**Parameters**:
- `window` (int): RSI calculation window
**Returns**: `pd.DataFrame` with RSI column
**Range**: 0-100, overbought > 70, oversold < 30

#### `calculate_macd(fast=12, slow=26, signal=9)`
**Purpose**: Moving Average Convergence Divergence
**Parameters**:
- `fast` (int): Fast EMA period
- `slow` (int): Slow EMA period
- `signal` (int): Signal line EMA period
**Returns**: `pd.DataFrame` with MACD columns
**Columns**: macd_line, macd_signal, macd_histogram

#### `calculate_bollinger_bands(window=20, std_dev=2)`
**Purpose**: Bollinger Bands calculation
**Parameters**:
- `window` (int): Moving average window
- `std_dev` (float): Standard deviation multiplier
**Returns**: `pd.DataFrame` with Bollinger columns
**Columns**: bb_upper, bb_middle, bb_lower, bb_width, bb_position

#### `create_ca_grid_features()`
**Purpose**: Map market data to CA grid states
**Parameters**: None
**Returns**: `pd.DataFrame` with CA features
**Method**: Convert continuous features to discrete states
**States**: {-1, 0, 1} for bearish, neutral, bullish

#### `create_neighbor_features()`
**Purpose**: Create 8-neighborhood feature representation
**Parameters**: None
**Returns**: `pd.DataFrame` with neighborhood features
**Method**: Map current and lagged features to 3x3 grid

#### `discretize_continuous_features(thresholds=[-0.33, 0.33])`
**Purpose**: Convert continuous to discrete states
**Parameters**:
- `thresholds` (list): State boundary values
**Returns**: `pd.DataFrame` with discrete features
**Method**: Threshold-based discretization

#### `normalize_features()`
**Purpose**: Standardize features using StandardScaler
**Parameters**: None
**Returns**: `pd.DataFrame` with normalized features
**Method**: Z-score normalization (mean=0, std=1)

#### `handle_missing_values(method='forward_fill')`
**Purpose**: Fill missing data with appropriate methods
**Parameters**:
- `method` (str): Filling method
**Methods**: forward_fill, backward_fill, interpolate, drop
**Returns**: `pd.DataFrame` with complete data

#### `create_lagged_features(lags=[1, 2, 3, 5])`
**Purpose**: Create time-lagged variables
**Parameters**:
- `lags` (list): Lag periods to create
**Returns**: `pd.DataFrame` with lagged features
**Purpose**: Incorporate historical information

#### `get_feature_summary()`
**Purpose**: Return summary statistics of all features
**Parameters**: None
**Returns**: `dict` with feature statistics
**Statistics**: Count, mean, std, min, max, skewness, kurtosis

---

## OPTIMIZATION FUNCTIONS

### CARuleOptimizer Class (`models/ca_optimizer.py`)

#### `__init__(ca_model, historical_data)`
**Purpose**: Initialize CA rule optimizer
**Parameters**:
- `ca_model`: Cellular automaton instance
- `historical_data` (pd.DataFrame): Training data
**Setup**: Model reference, data preparation, optimization parameters

#### `objective_function(rule_parameters)`
**Purpose**: Evaluate rule performance on historical data
**Parameters**:
- `rule_parameters` (array): CA rule parameters
**Returns**: `float` fitness score (lower is better)
**Metrics**: Price prediction error, directional accuracy, correlation

#### `prepare_training_data()`
**Purpose**: Preprocess data for optimization
**Parameters**: None
**Processing**: Feature extraction, normalization, train/validation split
**Returns**: Prepared datasets for optimization

#### `optimize_rules(method='differential_evolution', n_jobs=2)`
**Purpose**: Main optimization using differential evolution
**Parameters**:
- `method` (str): Optimization algorithm
- `n_jobs` (int): Number of parallel processes
**Returns**: `dict` optimized rule parameters
**Algorithm**: Differential evolution with parallel evaluation

#### `grid_search_optimization(param_ranges)`
**Purpose**: Grid search over parameter space
**Parameters**:
- `param_ranges` (dict): Parameter ranges for search
**Returns**: `dict` best parameters from grid search
**Method**: Exhaustive search over discrete parameter grid

#### `genetic_algorithm_optimization(population_size=50, generations=100)`
**Purpose**: Alternative genetic algorithm approach
**Parameters**:
- `population_size` (int): GA population size
- `generations` (int): Number of GA generations
**Returns**: `dict` evolved rule parameters
**Selection**: Tournament selection, crossover, mutation

#### `validate_rules(test_data)`
**Purpose**: Test optimized rules on out-of-sample data
**Parameters**:
- `test_data` (pd.DataFrame): Validation dataset
**Returns**: `dict` validation metrics
**Metrics**: RMSE, MAE, directional accuracy, Sharpe ratio

#### `cross_validate_rules(k_folds=5)`
**Purpose**: K-fold cross-validation
**Parameters**:
- `k_folds` (int): Number of CV folds
**Returns**: `dict` cross-validation results
**Method**: Time series cross-validation with proper ordering

#### `calculate_fitness_metrics(predicted, actual)`
**Purpose**: Performance evaluation metrics
**Parameters**:
- `predicted` (array): Predicted values
- `actual` (array): Actual values
**Returns**: `dict` fitness metrics
**Metrics**: RMSE, correlation, directional accuracy, information ratio

#### `encode_rules(rule_dict)`
**Purpose**: Convert rule dictionary to parameter vector
**Parameters**:
- `rule_dict` (dict): Rule configuration
**Returns**: `array` parameter vector for optimization
**Method**: Flatten dictionary to optimization-compatible format

#### `decode_rules(parameter_vector)`
**Purpose**: Convert parameters back to rules
**Parameters**:
- `parameter_vector` (array): Optimization parameters
**Returns**: `dict` rule configuration
**Method**: Reconstruct rule dictionary from parameter vector

#### `save_optimized_rules(filename)`
**Purpose**: Export optimized rules
**Parameters**:
- `filename` (str): Output file path
**Format**: JSON or pickle format
**Content**: Rule parameters, optimization history, validation results

#### `load_optimized_rules(filename)`
**Purpose**: Import saved rules
**Parameters**:
- `filename` (str): Input file path
**Returns**: `dict` rule parameters
**Error Handling**: Default rules if file not found

---

## PARALLEL PROCESSING FUNCTIONS

### ParallelSimulationRunner Class (`utils/parallel_runner.py`)

#### `__init__(model_class, num_cores=None)`
**Purpose**: Initialize parallel simulation runner
**Parameters**:
- `model_class`: Model class to run in parallel
- `num_cores` (int): Number of CPU cores (default: all available)
**Setup**: Core allocation, result storage, progress tracking

#### `run_single_simulation(params, run_id=0)`
**Purpose**: Execute one simulation run
**Parameters**:
- `params` (dict): Simulation parameters
- `run_id` (int): Run identifier for seeding
**Returns**: `dict` simulation results
**Features**: Error handling, result metadata, reproducible seeding

#### `run_batch_simulations(parameters, num_iterations)`
**Purpose**: Mesa BatchRunner integration
**Parameters**:
- `parameters` (dict): Fixed and variable parameters
- `num_iterations` (int): Iterations per parameter set
**Returns**: `pd.DataFrame` batch results
**Method**: Mesa's BatchRunner with parallel execution

#### `run_monte_carlo_simulation(base_params, num_runs=100)`
**Purpose**: Monte Carlo analysis with parameter variation
**Parameters**:
- `base_params` (dict): Base simulation parameters
- `num_runs` (int): Number of Monte Carlo runs
**Returns**: `pd.DataFrame` ensemble results
**Variation**: Random parameter perturbation across runs

#### `run_parameter_sweep(parameter_ranges)`
**Purpose**: Systematic parameter exploration
**Parameters**:
- `parameter_ranges` (dict): Parameter ranges to explore
**Returns**: `pd.DataFrame` parameter sweep results
**Method**: Grid search or Latin hypercube sampling

#### `optimize_parameters(objective_function, param_bounds)`
**Purpose**: Parameter optimization using parallel evaluation
**Parameters**:
- `objective_function` (callable): Function to optimize
- `param_bounds` (list): Parameter bounds for optimization
**Returns**: `dict` optimal parameters
**Algorithm**: Differential evolution with parallel fitness evaluation

#### `sensitivity_analysis(parameters, perturbation=0.1)`
**Purpose**: Parameter sensitivity testing
**Parameters**:
- `parameters` (dict): Base parameters
- `perturbation` (float): Perturbation magnitude
**Returns**: `dict` sensitivity metrics
**Method**: One-at-a-time sensitivity analysis

#### `collect_results(futures)`
**Purpose**: Aggregate parallel execution results
**Parameters**:
- `futures` (list): List of concurrent futures
**Returns**: `list` collected results
**Error Handling**: Handles failed simulations gracefully

#### `save_results(results, filename)`
**Purpose**: Export simulation results
**Parameters**:
- `results` (list/DataFrame): Results to save
- `filename` (str): Output file path
**Formats**: CSV, pickle, HDF5 for large datasets

#### `load_results(filename)`
**Purpose**: Import saved results
**Parameters**:
- `filename` (str): Input file path
**Returns**: Loaded results data
**Format Detection**: Automatic format detection and loading

#### `merge_result_batches(result_list)`
**Purpose**: Combine multiple result sets
**Parameters**:
- `result_list` (list): List of result DataFrames
**Returns**: `pd.DataFrame` merged results
**Method**: Concatenation with index management

#### `monitor_execution_progress(completed, total)`
**Purpose**: Track simulation progress
**Parameters**:
- `completed` (int): Completed simulations
- `total` (int): Total simulations
**Display**: Progress bar, ETA, completion percentage

#### `estimate_completion_time(completed, total, start_time)`
**Purpose**: ETA calculation
**Parameters**:
- `completed` (int): Completed runs
- `total` (int): Total runs
- `start_time` (datetime): Execution start time
**Returns**: `str` estimated completion time

#### `log_performance_metrics()`
**Purpose**: Performance statistics logging
**Parameters**: None
**Metrics**: CPU utilization, memory usage, execution time per core

---

## ANALYSIS FUNCTIONS

### ResultsAnalyzer Class (`results/result_analyzer.py`)

#### `__init__(simulation_results, actual_data=None)`
**Purpose**: Initialize results analyzer
**Parameters**:
- `simulation_results` (dict/DataFrame): Simulation output
- `actual_data` (DataFrame): Actual market data for comparison
**Setup**: Data storage, plotting style, analysis cache

#### `calculate_metrics()`
**Purpose**: Calculate comprehensive performance metrics
**Parameters**: None
**Returns**: `dict` with all calculated metrics
**Metrics**: RMSE, MAE, MAPE, directional accuracy, correlation, Theil's U

#### `statistical_tests()`
**Purpose**: Perform statistical hypothesis tests
**Parameters**: None
**Returns**: `dict` with test results
**Tests**: Jarque-Bera, Shapiro-Wilk, Augmented Dickey-Fuller, Ljung-Box

#### `correlation_analysis()`
**Purpose**: Correlation matrices and significance tests
**Parameters**: None
**Returns**: `dict` with correlation results
**Analysis**: Pearson, Spearman, Kendall correlations with p-values

#### `analyze_forecast_accuracy()`
**Purpose**: Detailed forecast error analysis
**Parameters**: None
**Returns**: `dict` with accuracy metrics
**Analysis**: Error distribution, bias, efficiency tests

#### `directional_accuracy_analysis()`
**Purpose**: Up/down movement prediction analysis
**Parameters**: None
**Returns**: `dict` with directional metrics
**Metrics**: Hit rate, false positive/negative rates, confusion matrix

#### `volatility_analysis()`
**Purpose**: Volatility clustering and GARCH effects
**Parameters**: None
**Returns**: `dict` with volatility statistics
**Analysis**: ARCH effects, volatility persistence, clustering

#### `risk_return_analysis()`
**Purpose**: Risk-adjusted performance metrics
**Parameters**: None
**Returns**: `dict` with risk metrics
**Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, VaR

#### `analyze_agent_behavior()`
**Purpose**: Agent performance and behavior patterns
**Parameters**: None
**Returns**: `dict` with agent analysis
**Analysis**: Position evolution, trading patterns, performance by type

#### `agent_type_comparison()`
**Purpose**: Performance comparison across agent types
**Parameters**: None
**Returns**: `dict` with comparative analysis
**Comparison**: Return, risk, trading frequency by agent type

#### `position_analysis()`
**Purpose**: Portfolio position evolution analysis
**Parameters**: None
**Returns**: `dict` with position statistics
**Analysis**: Position distribution, concentration, turnover

#### `trading_pattern_analysis()`
**Purpose**: Trading frequency and pattern analysis
**Parameters**: None
**Returns**: `dict` with trading statistics
**Patterns**: Trade size distribution, timing, clustering

#### `backtesting_analysis()`
**Purpose**: Out-of-sample performance testing
**Parameters**: None
**Returns**: `dict` with backtesting results
**Method**: Walk-forward analysis, expanding window validation

#### `rolling_window_validation(window_size=252)`
**Purpose**: Time-series cross-validation
**Parameters**:
- `window_size` (int): Rolling window size
**Returns**: `dict` with rolling validation results
**Method**: Rolling window with proper temporal ordering

#### `regime_change_analysis()`
**Purpose**: Performance across different market regimes
**Parameters**: None
**Returns**: `dict` with regime analysis
**Regimes**: Bull, bear, sideways markets; high/low volatility periods

#### `generate_report(save_path=None)`
**Purpose**: Comprehensive analysis report
**Parameters**:
- `save_path` (str): Output file path (optional)
**Returns**: `str` formatted report
**Content**: All analyses, tables, statistical tests, conclusions

#### `export_results(output_dir)`
**Purpose**: Detailed result exports
**Parameters**:
- `output_dir` (str): Output directory path
**Exports**: CSV files, statistical tables, figures
**Organization**: Structured directory with categorized outputs

#### `create_summary_statistics()`
**Purpose**: Key metric summaries
**Parameters**: None
**Returns**: `dict` with summary statistics
**Content**: Executive summary of key findings

---

## VISUALIZATION FUNCTIONS

### VisualizationTools Class (`results/plots.py`)

#### `__init__()`
**Purpose**: Initialize plotting environment and styles
**Parameters**: None
**Setup**: Matplotlib/seaborn styling, color palettes, figure parameters

#### `setup_style()`
**Purpose**: Configure plotting styles
**Parameters**: None
**Settings**: Figure size, DPI, color scheme, font settings

#### `create_color_palette(n_colors=8)`
**Purpose**: Generate consistent color schemes
**Parameters**:
- `n_colors` (int): Number of colors needed
**Returns**: Color palette for consistent plotting

#### `plot_simulation_results(results, actual_data=None, save_path=None)`
**Purpose**: Main results comparison visualization
**Parameters**:
- `results` (dict): Simulation results
- `actual_data` (DataFrame): Actual data for comparison
- `save_path` (str): File save path
**Plots**: Price comparison, error analysis, summary statistics

#### `plot_price_comparison(simulated, actual, save_path=None)`
**Purpose**: Simulated vs actual price series
**Parameters**:
- `simulated` (array): Simulated prices
- `actual` (array): Actual prices
- `save_path` (str): File save path
**Features**: Confidence intervals, correlation display

#### `plot_ca_evolution(ca_states, save_path=None)`
**Purpose**: CA grid evolution visualization
**Parameters**:
- `ca_states` (list): Grid states over time
- `save_path` (str): File save path
**Visualization**: Heatmaps, evolution animation, state statistics

#### `plot_agent_positions(agent_data, save_path=None)`
**Purpose**: Agent position distributions
**Parameters**:
- `agent_data` (DataFrame): Agent position data
- `save_path` (str): File save path
**Plots**: Position histograms, spatial distribution, time evolution

#### `plot_error_metrics(errors, save_path=None)`
**Purpose**: Error analysis visualizations
**Parameters**:
- `errors` (array): Prediction errors
- `save_path` (str): File save path
**Plots**: Error distribution, Q-Q plots, residual analysis

#### `plot_correlation_matrix(data, save_path=None)`
**Purpose**: Feature and result correlations
**Parameters**:
- `data` (DataFrame): Data for correlation analysis
- `save_path` (str): File save path
**Visualization**: Heatmap with significance indicators

#### `plot_returns_distribution(returns, save_path=None)`
**Purpose**: Return distribution analysis
**Parameters**:
- `returns` (array): Return series
- `save_path` (str): File save path
**Plots**: Histogram, normal comparison, tail analysis

#### `plot_volatility_clustering(prices, save_path=None)`
**Purpose**: Volatility pattern visualization
**Parameters**:
- `prices` (array): Price series
- `save_path` (str): File save path
**Analysis**: Volatility clustering, GARCH effects

#### `create_interactive_dashboard(results, save_path=None)`
**Purpose**: Interactive Plotly dashboard
**Parameters**:
- `results` (dict): Simulation results
- `save_path` (str): HTML file save path
**Features**: Interactive plots, dropdown menus, zoom/pan capabilities

#### `create_agent_animation(agent_data, save_path=None)`
**Purpose**: Animated agent behavior visualization
**Parameters**:
- `agent_data` (DataFrame): Agent data over time
- `save_path` (str): Animation file save path
**Animation**: Agent positions, trading actions over time

#### `create_ca_heatmap_animation(ca_states, save_path=None)`
**Purpose**: CA grid evolution animation
**Parameters**:
- `ca_states` (list): Grid states sequence
- `save_path` (str): Animation file save path
**Animation**: Grid state evolution with color coding

#### `plot_residual_analysis(residuals, save_path=None)`
**Purpose**: Forecast error analysis plots
**Parameters**:
- `residuals` (array): Prediction residuals
- `save_path` (str): File save path
**Plots**: Residual plots, autocorrelation, heteroscedasticity tests

#### `plot_qq_plots(data, distribution='normal', save_path=None)`
**Purpose**: Quantile-quantile plots for distribution testing
**Parameters**:
- `data` (array): Data to test
- `distribution` (str): Reference distribution
- `save_path` (str): File save path
**Purpose**: Test distributional assumptions

#### `plot_acf_pacf(data, lags=40, save_path=None)`
**Purpose**: Autocorrelation function plots
**Parameters**:
- `data` (array): Time series data
- `lags` (int): Number of lags to plot
- `save_path` (str): File save path
**Plots**: ACF, PACF with confidence intervals

#### `save_all_plots(results, output_dir)`
**Purpose**: Batch plot generation
**Parameters**:
- `results` (dict): Complete simulation results
- `output_dir` (str): Output directory
**Generation**: All standard plots with consistent naming

#### `export_to_pdf(plots, filename)`
**Purpose**: PDF report generation
**Parameters**:
- `plots` (list): List of plot objects
- `filename` (str): PDF output filename
**Format**: Multi-page PDF with plots and captions

#### `create_presentation_slides(results, template='academic')`
**Purpose**: Automated slide generation
**Parameters**:
- `results` (dict): Simulation results
- `template` (str): Slide template style
**Output**: PowerPoint-compatible slides with key results

---

## MAIN SIMULATION FUNCTIONS

### HybridGoldSimulation Class (`main_simulation.py`)

#### `__init__(config)`
**Purpose**: Initialize hybrid gold simulation system
**Parameters**:
- `config` (dict): Complete simulation configuration
**Setup**: Component initialization, model integration, result storage

#### `load_and_prepare_data()`
**Purpose**: Complete data loading and preprocessing pipeline
**Parameters**: None
**Returns**: `bool` success indicator
**Process**: Data collection → sentiment analysis → feature engineering → train/test split

#### `initialize_models()`
**Purpose**: Set up CA and ABM models with proper integration
**Parameters**: None
**Returns**: `bool` success indicator
**Setup**: CA model, market model, agent creation, model coupling

#### `calibrate_models()`
**Purpose**: Optimize CA rules and calibrate agent parameters
**Parameters**: None
**Returns**: `bool` success indicator
**Process**: CA rule optimization, agent parameter tuning, validation

#### `run_single_simulation(seed=None)`
**Purpose**: Execute one complete simulation run
**Parameters**:
- `seed` (int): Random seed for reproducibility
**Returns**: `dict` simulation results
**Process**: Model initialization → time evolution → result collection

#### `run_parallel_simulations(num_runs=100)`
**Purpose**: Execute multiple simulations in parallel
**Parameters**:
- `num_runs` (int): Number of parallel simulations
**Returns**: `DataFrame` ensemble results
**Method**: Monte Carlo simulation with parallel execution

#### `run_complete_simulation()`
**Purpose**: Full simulation pipeline execution
**Parameters**: None
**Returns**: `dict` complete results
**Pipeline**: Data → Models → Calibration → Simulation → Validation → Analysis

#### `run_historical_validation(start_date, end_date)`
**Purpose**: 10-year historical backtesting
**Parameters**:
- `start_date` (str): Validation start date
- `end_date` (str): Validation end date
**Returns**: `dict` validation results
**Method**: Historical data simulation with comprehensive metrics

#### `validate_results()`
**Purpose**: Model validation against actual data
**Parameters**: None
**Returns**: `bool` validation success
**Validation**: Statistical tests, performance metrics, agent analysis

#### `validate_predictions(test_data)`
**Purpose**: Prediction accuracy assessment
**Parameters**:
- `test_data` (DataFrame): Test dataset
**Returns**: `dict` validation metrics
**Metrics**: RMSE, correlation, directional accuracy

#### `calculate_performance_metrics(validation_results, test_data)`
**Purpose**: Comprehensive performance evaluation
**Parameters**:
- `validation_results` (dict): Validation output
- `test_data` (DataFrame): Test data
**Returns**: `dict` performance metrics
**Metrics**: Accuracy, precision, recall, F1-score

#### `demonstrate_investment_insights()`
**Purpose**: Generate investment decision support
**Parameters**: None
**Returns**: `dict` investment insights
**Insights**: Market analysis, risk assessment, trading recommendations

#### `analyze_investment_performance(investment_signals, historical_data)`
**Purpose**: Portfolio performance analysis
**Parameters**:
- `investment_signals` (DataFrame): Trading signals
- `historical_data` (DataFrame): Historical prices
**Returns**: `dict` performance analysis
**Metrics**: Total return, Sharpe ratio, maximum drawdown

#### `analyze_feature_importance(features)`
**Purpose**: Feature contribution analysis
**Parameters**:
- `features` (DataFrame): Feature dataset
**Returns**: `dict` feature importance
**Method**: Correlation analysis, mutual information

#### `analyze_market_regimes(historical_data)`
**Purpose**: Market regime identification and analysis
**Parameters**:
- `historical_data` (DataFrame): Historical market data
**Returns**: `dict` regime analysis
**Regimes**: Bull/bear markets, volatility regimes

#### `calculate_risk_metrics(historical_data)`
**Purpose**: Comprehensive risk assessment
**Parameters**:
- `historical_data` (DataFrame): Historical price data
**Returns**: `dict` risk metrics
**Metrics**: VaR, CVaR, maximum drawdown, beta

#### `assess_simulation_accuracy(validation_results)`
**Purpose**: Overall simulation accuracy evaluation
**Parameters**:
- `validation_results` (dict): Validation metrics
**Returns**: `float` accuracy score
**Method**: Composite score from multiple metrics

#### `run_simulation(data)`
**Purpose**: Core simulation execution on given data
**Parameters**:
- `data` (DataFrame): Input market data
**Returns**: `dict` simulation results
**Process**: Data-driven simulation with external inputs

#### `generate_visualizations(save_plots=True)`
**Purpose**: Create comprehensive visualization suite
**Parameters**:
- `save_plots` (bool): Whether to save plots to files
**Generation**: Price plots, CA evolution, agent behavior, dashboards

#### `generate_report(save_report=True)`
**Purpose**: Comprehensive simulation report generation
**Parameters**:
- `save_report` (bool): Whether to save report to file
**Returns**: `str` formatted report
**Content**: Executive summary, methodology, results, conclusions

#### `print_investment_summary(results)`
**Purpose**: Display key investment insights
**Parameters**:
- `results` (dict): Complete simulation results
**Display**: Investment recommendations, risk analysis, performance summary

---

This function reference provides detailed documentation for all major functions in the hybrid gold price prediction simulation system, organized by module and including parameters, returns, processing methods, and academic foundations.
