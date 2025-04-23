import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from Models.DirectionClassificationModel import DirectionClassificationModel
from Models.LTSMModel import LSTMModel
from Models.ModelBase import ModelBase
from Models.ModelFactory import ModelFactory
from Models.PricePredictionModel import PricePredictionModel
from Processing.DataStorage import DataStorage
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.PathResolver import PathResolver


class ModelPredictionStrategy(Strategy):
    """
    A strategy that uses trained ML models for trading decisions.

    - Direction model predicts whether to buy (1) or sell (-1) or hold (0)
    - Magnitude model predicts the expected price movement for setting take-profit levels

    Default parameters that can be optimized:
    - tp_ratio: Take-profit ratio relative to expected magnitude
    - sl_ratio: Stop-loss ratio relative to expected magnitude
    - min_confidence: Minimum confidence threshold for direction model
    - volatility_scale: Scaling factor for volatility-based position sizing
    """
    # Default parameters (these can be optimized)
    tp_ratio = 2.0  # Take-profit to stop-loss ratio (e.g., 2:1)
    sl_ratio = 1.0  # Base stop-loss ratio
    min_confidence = 0.55  # Minimum confidence for direction prediction
    volatility_scale = 1.0  # Scale position size by volatility

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)

        self.direction_predictions = self.data.direction_pred
        self.magnitude_predictions = self.data.magnitude_pred
        self.confidence_scores = self.data.confidence
        self.atr_values = self.data.atr

        # Keep track of current position info
        self.current_take_profit = None
        self.current_stop_loss = None

    def init(self):
        """Initialize indicators and signals required for the strategy."""
        # We'll use ATR for volatility-adjusted position sizing
        # Note: ATR should be available in the data

        # If needed, create additional indicators or signals
        pass

    def next(self):
        """Main strategy logic executed for each candle."""
        # Get latest predictions and values
        direction = self.direction_predictions[-1]
        magnitude = abs(self.magnitude_predictions[-1])  # Use absolute value for sizing
        confidence = self.confidence_scores[-1]
        atr = self.atr_values[-1]

        # Only enter positions with sufficient confidence
        if confidence < self.min_confidence:
            return

        # Scale position sizing based on volatility and magnitude
        # More predicted magnitude = larger position, higher volatility = smaller position
        risk_amount = magnitude / (atr * self.volatility_scale) if atr > 0 else 1.0

        # Cap risk amount between 0.1 and 1.0 (10% to 100% of normal position)
        risk_amount = max(0.1, min(1.0, risk_amount))

        # Calculate take-profit and stop-loss levels based on magnitude prediction
        # and the configured ratios
        take_profit_pips = magnitude * self.tp_ratio
        stop_loss_pips = magnitude * self.sl_ratio

        # Execute trading logic based on direction prediction

        # Long position logic
        if direction > 0 and not self.position:
            # Enter long position
            self.buy(size=risk_amount)
            self.current_take_profit = self.data.Close[-1] + take_profit_pips
            self.current_stop_loss = self.data.Close[-1] - stop_loss_pips

        # Short position logic
        elif direction < 0 and not self.position:
            # Enter short position
            self.sell(size=risk_amount)
            self.current_take_profit = self.data.Close[-1] - take_profit_pips
            self.current_stop_loss = self.data.Close[-1] + stop_loss_pips

        # Exit position logic based on take-profit and stop-loss levels
        if self.position:
            # For long positions
            if self.position.is_long:
                if self.data.Close[-1] >= self.current_take_profit or self.data.Close[-1] <= self.current_stop_loss:
                    self.position.close()

            # For short positions
            elif self.position.is_short:
                if self.data.Close[-1] <= self.current_take_profit or self.data.Close[-1] >= self.current_stop_loss:
                    self.position.close()


class BacktestManager:
    """Manages the backtesting process using the trained models."""

    def __init__(self, config: Config, logger: Logger, model_factory: ModelFactory, data_storage: DataStorage,
                 path_resolver: PathResolver):
        self.config = config
        self.logger = logger
        self.model_factory = model_factory
        self.data_storage = data_storage
        self.path_resolver = path_resolver

        self.backtest_config = config.get('BacktestSettings', {})
        self.results_dir = self.path_resolver.resolve_path("BacktestResults")
        os.makedirs(self.results_dir, exist_ok=True)

    def prepare_data(self, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Prepare data for backtesting by loading test data and adding model predictions.
        """
        try:
            # Load test dataset
            X_test, y_test = self.data_storage.load_processed_data(pair, timeframe, "testing")

            if X_test.empty:
                self.logger.error(f"No test data available for {pair} {timeframe}")
                raise ValueError(f"No test data available for {pair} {timeframe}")

            # We need price data in OHLC format for backtesting
            ohlc_data = self.get_raw_ohlc_data(pair, timeframe, "testing")

            if ohlc_data is None or ohlc_data.empty:
                self.logger.error(f"No OHLC data available for {pair} {timeframe}")
                raise ValueError(f"No OHLC data available for {pair} {timeframe}")

            # Get time from feature data and align with OHLC data
            if 'time' in X_test.columns:
                aligned_data = pd.merge(
                    ohlc_data,
                    X_test,
                    left_on='time',
                    right_on='time',
                    how='inner'
                )
            else:
                self.logger.error("Time column missing from feature data")
                raise ValueError("Time column missing from feature data")

            # Data now contains both OHLC and features
            return aligned_data

        except Exception as e:
            self.logger.error(f"Error preparing backtest data: {e}", exc_info=True)
            raise

    def get_raw_ohlc_data(self, pair: str, timeframe: str, dataset_type: str) -> pd.DataFrame:
        """
        Get raw OHLC data from the database for a specific symbol and timeframe.
        """
        try:
            # Create table name
            table_name = f"{pair}_{timeframe.lower()}_{dataset_type}"

            # Query to get all data
            query = f"SELECT * FROM {table_name} ORDER BY time"

            # Use data_storage's engine
            with self.data_storage.engine.connect() as conn:
                ohlc_data = pd.read_sql(query, conn)

            if ohlc_data.empty:
                self.logger.warning(f"No OHLC data found for {table_name}")
                return pd.DataFrame()

            # Ensure time column is datetime
            if 'time' in ohlc_data.columns:
                ohlc_data['time'] = pd.to_datetime(ohlc_data['time'])

            self.logger.info(f"Loaded {len(ohlc_data)} rows of OHLC data from {table_name}")
            return ohlc_data

        except Exception as e:
            self.logger.error(f"Failed to retrieve OHLC data: {e}")
            return pd.DataFrame()

    def load_models(self, pair: str, timeframe: str) -> Dict[str, ModelBase]:
        """
        Load the direction and magnitude prediction models.
        """
        try:
            models = {}

            # Log the directory where we're looking for models
            model_dir = self.path_resolver.resolve_path("TrainedModels")
            self.logger.info(f"Looking for models in: {model_dir}")

            # List all files in the model directory to help with debugging
            if os.path.exists(model_dir):
                model_files = os.listdir(model_dir)
                self.logger.info(f"Found {len(model_files)} files in model directory: {model_files}")

            # Try to load direction model
            direction_model_options = [
                "RandomForest", "GradientBoosting", "LSTM"
            ]

            # Try each model type until one loads successfully
            for model_type in direction_model_options:
                try:
                    self.logger.info(f"Attempting to load {model_type} direction model for {pair} {timeframe}")
                    direction_model = self.model_factory.load_model(
                        model_type, pair, timeframe, "direction_1"
                    )

                    if direction_model:
                        models['direction'] = direction_model
                        self.logger.info(f"Loaded direction model: {direction_model.name}")
                        break
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_type} direction model: {e}")

            # Try to load magnitude model
            magnitude_model_options = [
                "GradientBoosting", "RandomForest", "LSTM"
            ]

            # Try each model type until one loads successfully
            for model_type in magnitude_model_options:
                try:
                    self.logger.info(f"Attempting to load {model_type} magnitude model for {pair} {timeframe}")
                    magnitude_model = self.model_factory.load_model(
                        model_type, pair, timeframe, "future_price_1"
                    )

                    if magnitude_model:
                        models['magnitude'] = magnitude_model
                        self.logger.info(f"Loaded magnitude model: {magnitude_model.name}")
                        break
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_type} magnitude model: {e}")

            # If neither model loaded, try a more exhaustive search
            if not models:
                self.logger.warning(f"No models loaded with standard names, trying direct file search")

                # Scan the directory for files that match the pattern
                direction_pattern = f"{pair}_{timeframe}_direction_1_"
                magnitude_pattern = f"{pair}_{timeframe}_future_price_1_"

                for filename in model_files:
                    # Check if it's a model file (has .keras or .h5 extension)
                    if not (filename.endswith('.keras') or filename.endswith('.h5')):
                        continue

                    # Check if it's a direction model
                    if direction_pattern in filename and 'direction' not in models:
                        model_name = filename.split('.')[0]  # Remove extension
                        model_type = model_name.split('_')[-1]  # Get model type from name

                        try:
                            self.logger.info(f"Trying to load direction model by filename: {filename}")
                            path = os.path.join(model_dir, model_name)

                            if "randomforest" in model_name.lower():
                                model = DirectionClassificationModel(self.config, self.logger, model_name)
                            elif "gradientboosting" in model_name.lower():
                                model = DirectionClassificationModel(self.config, self.logger, model_name)
                            elif "lstm" in model_name.lower():
                                model = LSTMModel(self.config, self.logger, model_name)
                                model.is_classification = True
                            else:
                                continue

                            if model.load(path):
                                models['direction'] = model
                                self.logger.info(f"Loaded direction model from file: {filename}")
                        except Exception as e:
                            self.logger.warning(f"Failed to load direction model from file {filename}: {e}")

                    # Check if it's a magnitude model
                    if magnitude_pattern in filename and 'magnitude' not in models:
                        model_name = filename.split('.')[0]  # Remove extension
                        model_type = model_name.split('_')[-1]  # Get model type from name

                        try:
                            self.logger.info(f"Trying to load magnitude model by filename: {filename}")
                            path = os.path.join(model_dir, model_name)

                            if "randomforest" in model_name.lower():
                                model = PricePredictionModel(self.config, self.logger, model_name)
                            elif "gradientboosting" in model_name.lower():
                                model = PricePredictionModel(self.config, self.logger, model_name)
                            elif "lstm" in model_name.lower():
                                model = LSTMModel(self.config, self.logger, model_name)
                                model.is_classification = False
                            else:
                                continue

                            if model.load(path):
                                models['magnitude'] = model
                                self.logger.info(f"Loaded magnitude model from file: {filename}")
                        except Exception as e:
                            self.logger.warning(f"Failed to load magnitude model from file {filename}: {e}")

            # Final check if we loaded any models
            if not models:
                self.logger.error(f"No suitable models found in {model_dir} for {pair} {timeframe}")
                raise ValueError(f"No models could be loaded for {pair} {timeframe}")

            return models

        except Exception as e:
            self.logger.error(f"Error loading models: {e}", exc_info=True)
            raise

    def add_predictions(self, data: pd.DataFrame, models: Dict[str, ModelBase]) -> pd.DataFrame:
        """Add model predictions to the dataset for backtesting."""
        try:
            # Create a copy to avoid modifying the original
            backtest_data = data.copy()

            # Log column names for debugging
            self.logger.info(f"Data columns: {list(backtest_data.columns)}")

            # Check if all required OHLC columns exist
            required_columns = ['open', 'high', 'low', 'close']
            missing_ohlc = [col for col in required_columns if col not in backtest_data.columns]

            if missing_ohlc:
                self.logger.error(f"Missing required OHLC columns: {missing_ohlc}")
                raise ValueError(f"Missing required OHLC columns: {missing_ohlc}")

            # Generate predictions
            if 'direction' in models:
                try:
                    # Get features excluding OHLC columns but keep tick_volume
                    # Make sure tick_volume is included if it's in the data
                    feature_cols = models['direction'].feature_columns

                    # Check that all required features are in the data
                    missing_cols = [col for col in feature_cols if col not in backtest_data.columns]
                    if missing_cols:
                        self.logger.warning(f"Missing columns for prediction: {missing_cols}")
                        # Create default direction predictions
                        self.logger.info("Using default neutral predictions for direction")
                        backtest_data['direction_pred'] = 0
                        backtest_data['confidence'] = 0.5
                    else:
                        # Create features DataFrame with exactly the columns needed
                        features = backtest_data[feature_cols].copy()

                        # Get direction predictions
                        direction_preds = models['direction'].predict(features)

                        # Convert to integer predictions (-1, 0, 1)
                        if len(direction_preds.shape) > 1 and direction_preds.shape[1] > 1:
                            # If probabilities, get the class with highest probability
                            backtest_data['direction_pred'] = np.argmax(direction_preds,
                                                                        axis=1) - 1  # Convert 0,1,2 to -1,0,1
                            backtest_data['confidence'] = np.max(direction_preds, axis=1)
                        else:
                            # Binary model (0/1) convert to -1/1
                            if len(direction_preds) > 0:
                                backtest_data['direction_pred'] = (direction_preds > 0.5).astype(int) * 2 - 1
                                backtest_data['confidence'] = np.where(direction_preds > 0.5, direction_preds,
                                                                       1 - direction_preds)
                            else:
                                # Handle empty predictions
                                self.logger.warning("Empty direction predictions, using defaults")
                                backtest_data['direction_pred'] = 0
                                backtest_data['confidence'] = 0.5

                    self.logger.info(f"Added direction predictions to backtest data")
                except Exception as e:
                    self.logger.error(f"Failed to generate direction predictions: {e}")
                    # Add default values to allow backtest to continue
                    backtest_data['direction_pred'] = 0  # Neutral
                    backtest_data['confidence'] = 0.5

            if 'magnitude' in models:
                try:
                    feature_cols = models['magnitude'].feature_columns

                    # Check that all required features are in the data
                    missing_cols = [col for col in feature_cols if col not in backtest_data.columns]
                    if missing_cols:
                        self.logger.warning(f"Missing columns for magnitude prediction: {missing_cols}")
                        # Create default magnitude predictions
                        self.logger.info("Using default 0 predictions for magnitude")
                        backtest_data['magnitude_pred'] = 0
                    else:
                        # Create features DataFrame with exactly the columns needed
                        features = backtest_data[feature_cols].copy()

                        # Get magnitude predictions
                        magnitude_preds = models['magnitude'].predict(features)

                        # Add to dataset
                        if len(magnitude_preds) > 0:
                            backtest_data['magnitude_pred'] = magnitude_preds
                        else:
                            # Handle empty predictions
                            self.logger.warning("Empty magnitude predictions, using defaults")
                            backtest_data['magnitude_pred'] = 0

                    self.logger.info(f"Added magnitude predictions to backtest data")
                except Exception as e:
                    self.logger.error(f"Failed to generate magnitude predictions: {e}")
                    # Add default values to allow backtest to continue
                    backtest_data['magnitude_pred'] = 0

            # Ensure all required columns for strategy are present
            if 'atr' not in backtest_data.columns:
                self.logger.warning("ATR not found in data, using default values")
                backtest_data['atr'] = (backtest_data['high'] - backtest_data['low']).rolling(14).mean()

            # Rename columns to match backtesting.py requirements - using lowercase source columns
            rename_dict = {}
            if 'open' in backtest_data.columns:
                rename_dict['open'] = 'Open'
            if 'high' in backtest_data.columns:
                rename_dict['high'] = 'High'
            if 'low' in backtest_data.columns:
                rename_dict['low'] = 'Low'
            if 'close' in backtest_data.columns:
                rename_dict['close'] = 'Close'
            if 'tick_volume' in backtest_data.columns:
                rename_dict['tick_volume'] = 'Volume'

            # Only rename columns that exist
            backtest_data = backtest_data.rename(columns=rename_dict)

            # Verify required columns exist after renaming
            backtest_columns = list(backtest_data.columns)
            self.logger.info(f"Backtest data columns after renaming: {backtest_columns}")

            required_backtest_columns = ['Open', 'High', 'Low', 'Close']
            missing_required = [col for col in required_backtest_columns if col not in backtest_columns]

            if missing_required:
                self.logger.error(f"Missing required columns for Backtest: {missing_required}")
                raise ValueError(f"Missing required columns for Backtest: {missing_required}")

            # Set index to time for backtesting
            if 'time' in backtest_data.columns:
                backtest_data = backtest_data.set_index('time')
            else:
                self.logger.warning("No 'time' column found for index, using default index")

            # Drop rows with NaN values in essential columns only
            essential_cols = ['Open', 'High', 'Low', 'Close']
            if 'direction_pred' in backtest_data.columns:
                essential_cols.append('direction_pred')
            if 'magnitude_pred' in backtest_data.columns:
                essential_cols.append('magnitude_pred')
            if 'atr' in backtest_data.columns:
                essential_cols.append('atr')

            # Filter to columns that exist in the dataframe
            essential_cols = [col for col in essential_cols if col in backtest_data.columns]

            # Drop NaN rows only in essential columns
            backtest_data = backtest_data.dropna(subset=essential_cols)

            return backtest_data

        except Exception as e:
            self.logger.error(f"Error adding predictions: {e}", exc_info=True)
            raise

    def run_backtest(self, data: pd.DataFrame, pair: str, timeframe: str, params: Dict[str, Any] = None) -> Dict[
        str, Any]:
        """
        Run the backtest with the prepared data and specified parameters.
        """
        try:
            # Set default parameters if none provided
            if params is None:
                params = {
                    'tp_ratio': 2.0,
                    'sl_ratio': 1.0,
                    'min_confidence': 0.55,
                    'volatility_scale': 1.0
                }

            # Create and run backtest
            backtest = Backtest(
                data,
                ModelPredictionStrategy,
                cash=10000,
                commission=0.0002,
                margin=1.0,
                trade_on_close=False,
                exclusive_orders=True,
                hedging=False
            )

            # Run backtest with parameters
            result = backtest.run(**params)

            # Log basic results
            self.logger.info(f"Backtest completed for {pair} {timeframe}")
            self.logger.info(f"Return: {result['Return']:.2f}%, Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
            self.logger.info(f"Win Rate: {result['Win Rate']:.2f}%, Max Drawdown: {result['Max. Drawdown']:.2f}%")

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(self.results_dir, f"{pair}_{timeframe}_{timestamp}")
            os.makedirs(result_path, exist_ok=True)

            # Save stats as CSV
            stats_df = pd.DataFrame([result])
            stats_df.to_csv(os.path.join(result_path, "stats.csv"))

            # Save trade list
            trades_df = pd.DataFrame(result._trades)
            trades_df.to_csv(os.path.join(result_path, "trades.csv"))

            # Plot and save figures
            fig = backtest.plot(filename=os.path.join(result_path, "backtest_plot.html"))
            plt.savefig(os.path.join(result_path, "backtest_plot.png"))

            self.logger.info(f"Backtest results saved to {result_path}")

            return {
                'result': result,
                'stats': result._stats,
                'trades': result._trades,
                'result_path': result_path
            }

        except Exception as e:
            self.logger.error(f"Error running backtest: {e}", exc_info=True)
            raise

    def optimize_parameters(self, data: pd.DataFrame, pair: str, timeframe: str) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        """
        try:
            self.logger.info(f"Starting parameter optimization for {pair} {timeframe}")

            # Create backtest instance
            backtest = Backtest(
                data,
                ModelPredictionStrategy,
                cash=10000,
                commission=0.0002,
                margin=1.0,
                trade_on_close=False,
                exclusive_orders=True,
                hedging=False
            )

            # Define parameter ranges to optimize
            optimization_params = {
                'tp_ratio': np.linspace(1.5, 3.0, 4),
                'sl_ratio': np.linspace(0.5, 1.5, 3),
                'min_confidence': np.linspace(0.5, 0.75, 6),
                'volatility_scale': np.linspace(0.5, 2.0, 4)
            }

            # Get optimization metric from config
            optimize_metric = self.backtest_config.get('optimize_metric', 'Sharpe Ratio')
            maximize = True  # Assume we want to maximize the metric

            # Run optimization
            self.logger.info(
                f"Running optimization for {len(optimization_params['tp_ratio']) * len(optimization_params['sl_ratio']) * len(optimization_params['min_confidence']) * len(optimization_params['volatility_scale'])} parameter combinations")

            opt_result = backtest.optimize(
                tp_ratio=optimization_params['tp_ratio'],
                sl_ratio=optimization_params['sl_ratio'],
                min_confidence=optimization_params['min_confidence'],
                volatility_scale=optimization_params['volatility_scale'],
                maximize=optimize_metric if maximize else None,
                minimize=None if maximize else optimize_metric,
                constraint=lambda p: p.tp_ratio > p.sl_ratio,  # TP should be higher than SL
                max_tries=None,  # Exhaustive search
                random_state=42,
                return_heatmap=False
            )

            # Save optimization results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(self.results_dir, f"{pair}_{timeframe}_optimization_{timestamp}")
            os.makedirs(result_path, exist_ok=True)

            # Save heatmaps
            for param1, param2 in [('tp_ratio', 'sl_ratio'), ('min_confidence', 'volatility_scale')]:
                if len(optimization_params[param1]) > 1 and len(optimization_params[param2]) > 1:
                    heatmap = backtest.optimize(
                        **{param1: optimization_params[param1], param2: optimization_params[param2]},
                        maximize=optimize_metric if maximize else None,
                        minimize=None if maximize else optimize_metric,
                        return_heatmap=True
                    )
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plot_heatmaps(heatmap, ax=ax, title=f"{param1} vs {param2}")
                    plt.savefig(os.path.join(result_path, f"heatmap_{param1}_{param2}.png"))
                    plt.close()

            # Save best parameters
            best_params = opt_result._params
            with open(os.path.join(result_path, "best_params.txt"), "w") as f:
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")

            # Run backtest with best parameters
            best_result = backtest.run(**best_params)

            # Save best result plot
            fig = backtest.plot(filename=os.path.join(result_path, "optimized_backtest_plot.html"))
            plt.savefig(os.path.join(result_path, "optimized_backtest_plot.png"))

            # Save best stats as CSV
            stats_df = pd.DataFrame([best_result])
            stats_df.to_csv(os.path.join(result_path, "optimized_stats.csv"))

            self.logger.info(f"Optimization completed. Best {optimize_metric}: {best_result[optimize_metric]:.4f}")
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Optimization results saved to {result_path}")

            return {
                'best_result': best_result,
                'best_params': best_params,
                'result_path': result_path
            }

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}", exc_info=True)
            raise

    def run_full_backtest(self, pair: str, timeframe: str, optimize: bool = False) -> Dict[str, Any]:
        """
        Run a complete backtest process including data preparation, model loading, and evaluation.
        """
        try:
            self.logger.info(f"Starting full backtest for {pair} {timeframe}")

            # Step 1: Load models
            models = self.load_models(pair, timeframe)
            if not models:
                raise ValueError(f"No models available for {pair} {timeframe}")

            # Step 2: Prepare data
            data = self.prepare_data(pair, timeframe)
            if data.empty:
                raise ValueError(f"No data available for {pair} {timeframe}")

            # Step 3: Add predictions to data
            prepared_data = self.add_predictions(data, models)
            self.logger.info(f"Prepared {len(prepared_data)} data points for backtesting")

            # Step 4: Run optimization if requested
            if optimize:
                opt_results = self.optimize_parameters(prepared_data, pair, timeframe)
                best_params = opt_results['best_params']
                self.logger.info(f"Using optimized parameters: {best_params}")

                # Run backtest with optimized parameters
                results = self.run_backtest(prepared_data, pair, timeframe, params=best_params)
                results['optimization'] = opt_results

            else:
                # Run backtest with default parameters
                results = self.run_backtest(prepared_data, pair, timeframe)

            self.logger.info(f"Full backtest completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error in full backtest: {e}", exc_info=True)
            raise