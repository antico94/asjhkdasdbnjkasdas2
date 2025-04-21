import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import Dict, List, Tuple, Any, Optional, Union

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.TechnicalIndicators import TechnicalIndicators


class DataProcessor:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

        # Get GoldTradingSettings from config
        self.gold_settings = config.get('GoldTradingSettings', {})

        # Log if GoldTradingSettings is missing or empty
        if not self.gold_settings:
            self.logger.warning("GoldTradingSettings is missing or empty in config. Using default values.")
            # Set default values
            self.gold_settings = {
                'Indicators': {
                    'MovingAverages': {
                        'SMA': {'Periods': [5, 8, 13, 21, 50, 200]},
                        'EMA': {'Periods': [5, 8, 13, 21, 50, 200]},
                        'MACD': {'FastPeriod': 12, 'SlowPeriod': 26, 'SignalPeriod': 9}
                    },
                    'Volatility': {
                        'BollingerBands': {'Period': 20, 'NumStd': 2.0},
                        'ATR': {'Period': 14}
                    },
                    'Momentum': {
                        'RSI': {'Period': 14, 'OverBought': 70, 'OverSold': 30},
                        'Stochastic': {'KPeriod': 14, 'DPeriod': 3, 'SlowingPeriod': 3}
                    },
                    'PivotPoints': {'Method': 'standard'}
                },
                'FeatureEngineering': {
                    'WindowSizes': [1, 3, 5],
                    'PriceFeatures': ['close', 'high', 'low', 'open']
                },
                'MachineLearning': {
                    'Targets': {
                        'PricePrediction': {'Horizons': [1, 3, 5]},
                        'DirectionPrediction': {'Threshold': 0.001}
                    }
                }
            }
        else:
            self.logger.info("GoldTradingSettings found in config.")

        self.db_config = config.get('Database', {})
        self.indicators = TechnicalIndicators()
        self.engine = self._create_engine()

        # Log indicator settings to ensure they're being loaded correctly
        self.logger.info(f"Loaded indicator settings: {self.gold_settings.get('Indicators', {})}")

    def _create_engine(self):
        """Create SQLAlchemy engine for database connections."""
        db = self.db_config
        connection_string = (
            f"mssql+pyodbc://{db['User']}:{db['Password']}@{db['Host']},{db['Port']}/"
            f"{db['Database']}?driver=ODBC+Driver+17+for+SQL+Server"
        )
        return create_engine(connection_string)

    def get_data_from_db(self, pair: str = "XAUUSD", timeframe: str = "H1",
                         data_type: str = "training") -> pd.DataFrame:
        """Retrieve data from database."""
        try:
            table_name = f"{pair}_{timeframe.lower()}_{data_type}"

            self.logger.info(f"Retrieving {data_type} data for {pair} {timeframe}")

            query = f"SELECT * FROM {table_name} ORDER BY time"
            df = pd.read_sql(query, self.engine)

            self.logger.info(f"Successfully retrieved {len(df)} rows from {table_name}")

            # Ensure datetime column is properly formatted
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])

            return df

        except Exception as e:
            self.logger.error(f"Failed to retrieve data: {e}")
            raise

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with technical indicators."""
        try:
            if df.empty:
                self.logger.warning("Empty dataframe provided for processing")
                return df

            self.logger.info(f"Processing {len(df)} rows of raw data")

            # Make sure we have a proper copy to avoid modifying the original
            processed_df = df.copy()

            # Ensure time column is present and properly formatted
            if 'time' not in processed_df.columns:
                self.logger.error("Time column missing from input data")
                raise ValueError("Time column missing from input data")

            processed_df['time'] = pd.to_datetime(processed_df['time'])

            # Calculate all technical indicators based on configuration
            processed_df = self.calculate_indicators(processed_df)

            # Remove NaN values that come from indicators using windows
            processed_df = self.handle_missing_values(processed_df)

            self.logger.info(f"Data processing complete with {len(processed_df)} rows remaining")
            return processed_df

        except Exception as e:
            self.logger.error(f"Error in data processing: {e}")
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators based on configuration."""
        result = df.copy()

        try:
            # Get indicator settings from config
            ma_settings = self.gold_settings.get('Indicators', {}).get('MovingAverages', {})
            volatility_settings = self.gold_settings.get('Indicators', {}).get('Volatility', {})
            momentum_settings = self.gold_settings.get('Indicators', {}).get('Momentum', {})
            pivot_settings = self.gold_settings.get('Indicators', {}).get('PivotPoints', {})

            # Log the settings being used
            self.logger.info(f"MA settings: {ma_settings}")
            self.logger.info(f"Volatility settings: {volatility_settings}")
            self.logger.info(f"Momentum settings: {momentum_settings}")
            self.logger.info(f"Pivot settings: {pivot_settings}")

            # Calculate Moving Averages
            if 'SMA' in ma_settings:
                periods = ma_settings['SMA'].get('Periods', [5, 8, 13, 21, 50, 200])
                self.logger.info(f"Calculating SMA with periods: {periods}")
                result = self.indicators.calculate_sma(result, periods=periods)

            if 'EMA' in ma_settings:
                periods = ma_settings['EMA'].get('Periods', [5, 8, 13, 21, 50, 200])
                self.logger.info(f"Calculating EMA with periods: {periods}")
                result = self.indicators.calculate_ema(result, periods=periods)

            if 'MACD' in ma_settings:
                fast_period = ma_settings['MACD'].get('FastPeriod', 12)
                slow_period = ma_settings['MACD'].get('SlowPeriod', 26)
                signal_period = ma_settings['MACD'].get('SignalPeriod', 9)
                self.logger.info(
                    f"Calculating MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}")
                result = self.indicators.calculate_macd(
                    result, fast_period=fast_period, slow_period=slow_period, signal_period=signal_period
                )

            # Calculate Volatility indicators
            if 'BollingerBands' in volatility_settings:
                period = volatility_settings['BollingerBands'].get('Period', 20)
                num_std = volatility_settings['BollingerBands'].get('NumStd', 2.0)
                self.logger.info(f"Calculating Bollinger Bands with period={period}, std={num_std}")
                result = self.indicators.calculate_bollinger_bands(
                    result, period=period, num_std=num_std
                )

            if 'ATR' in volatility_settings:
                period = volatility_settings['ATR'].get('Period', 14)
                self.logger.info(f"Calculating ATR with period={period}")
                result = self.indicators.calculate_atr(result, period=period)

            # Calculate Momentum indicators
            if 'RSI' in momentum_settings:
                period = momentum_settings['RSI'].get('Period', 14)
                self.logger.info(f"Calculating RSI with period={period}")
                result = self.indicators.calculate_rsi(result, period=period)

            if 'Stochastic' in momentum_settings:
                k_period = momentum_settings['Stochastic'].get('KPeriod', 14)
                d_period = momentum_settings['Stochastic'].get('DPeriod', 3)
                self.logger.info(f"Calculating Stochastic with k_period={k_period}, d_period={d_period}")
                result = self.indicators.calculate_stochastic(
                    result, k_period=k_period, d_period=d_period
                )

            # Calculate Pivot Points if configured
            if pivot_settings:
                method = pivot_settings.get('Method', 'standard')
                self.logger.info(f"Calculating Pivot Points with method={method}")
                result = self.indicators.calculate_pivot_points(result, method=method)

            # Log columns after adding indicators
            self.logger.info(f"Columns after adding indicators: {list(result.columns)}")

            # Fallback to calculating all indicators if none were added from config
            if len(result.columns) <= len(df.columns) + 2:  # Allow for a couple of extra columns
                self.logger.warning("Few or no indicators were added from config, calculating all")
                result = self.indicators.calculate_all_indicators(result)
                self.logger.info(f"Columns after fallback: {list(result.columns)}")

            return result

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise

    def handle_missing_values(self, df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        if method == 'drop':
            original_len = len(df)
            df = df.dropna()
            dropped = original_len - len(df)
            if dropped > 0:
                self.logger.info(f"Dropped {dropped} rows with NaN values")
            return df
        elif method == 'fill':
            df = df.fillna(method='ffill')
            # Fill any remaining NaNs (at the beginning) with zeros
            df = df.fillna(0)
            self.logger.info("Filled NaN values using forward fill method")
            return df
        else:
            self.logger.warning(f"Unknown missing value handling method: {method}, using drop")
            return df.dropna()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning."""
        try:
            feature_settings = self.gold_settings.get('FeatureEngineering', {})
            result = df.copy()

            # Log before feature creation
            self.logger.info(f"Starting feature engineering with columns: {list(result.columns)}")

            # Create price-based features
            price_features = feature_settings.get('PriceFeatures', ['close'])

            # Create lagged features
            window_sizes = feature_settings.get('WindowSizes', [1, 3, 5])
            for feature in price_features:
                if feature in result.columns:
                    # Create lag features - explicitly name these for validation
                    for window in window_sizes:
                        col_name = f'{feature}_lag_{window}'
                        result[col_name] = result[feature].shift(window)
                        self.logger.debug(f"Created lag feature: {col_name}")

                    # Create return features (percent change)
                    col_name = f'{feature}_pct_change'
                    result[col_name] = result[feature].pct_change()
                    self.logger.debug(f"Created percent change feature: {col_name}")

                    for window in window_sizes:
                        col_name = f'{feature}_pct_change_{window}'
                        result[col_name] = result[feature].pct_change(window)
                        self.logger.debug(f"Created percent change with window feature: {col_name}")

                    # Create rolling statistics
                    col_name = f'{feature}_rolling_mean_5'
                    result[col_name] = result[feature].rolling(window=5).mean()
                    self.logger.debug(f"Created rolling mean feature: {col_name}")

                    col_name = f'{feature}_rolling_std_5'
                    result[col_name] = result[feature].rolling(window=5).std()
                    self.logger.debug(f"Created rolling std feature: {col_name}")

            # Create candlestick pattern features
            result['candle_body'] = abs(result['close'] - result['open'])
            result['candle_wick_upper'] = result['high'] - np.maximum(result['close'], result['open'])
            result['candle_wick_lower'] = np.minimum(result['close'], result['open']) - result['low']
            result['candle_range'] = result['high'] - result['low']
            self.logger.debug("Created candlestick features")

            # Create time-based features
            result['hour'] = pd.to_datetime(result['time']).dt.hour
            result['day_of_week'] = pd.to_datetime(result['time']).dt.dayofweek
            self.logger.debug("Created time-based features")

            # Log after feature creation before handling missing values
            self.logger.info(f"Completed feature engineering. Now have columns: {list(result.columns)}")

            # Handle missing values from feature creation
            result = self.handle_missing_values(result)

            return result

        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            raise

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for supervised learning."""
        try:
            ml_settings = self.gold_settings.get('MachineLearning', {})
            targets = ml_settings.get('Targets', {})
            result = df.copy()

            # Create price prediction targets
            if 'PricePrediction' in targets:
                horizons = targets['PricePrediction'].get('Horizons', [1, 3, 5])
                for horizon in horizons:
                    result[f'future_price_{horizon}'] = result['close'].shift(-horizon)

            # Create direction prediction targets
            if 'DirectionPrediction' in targets:
                horizons = targets['PricePrediction'].get('Horizons', [1, 3, 5])
                threshold = targets['DirectionPrediction'].get('Threshold', 0.001)

                for horizon in horizons:
                    future_return = result['close'].pct_change(-horizon)

                    # Direction: 1 for up, 0 for flat, -1 for down
                    result[f'direction_{horizon}'] = 0
                    result.loc[future_return > threshold, f'direction_{horizon}'] = 1
                    result.loc[future_return < -threshold, f'direction_{horizon}'] = -1

                    # Binary signal: 1 for up, 0 for not up
                    result[f'signal_up_{horizon}'] = (result[f'direction_{horizon}'] == 1).astype(int)

                    # Binary signal: 1 for down, 0 for not down
                    result[f'signal_down_{horizon}'] = (result[f'direction_{horizon}'] == -1).astype(int)

            return result

        except Exception as e:
            self.logger.error(f"Error creating target variables: {e}")
            raise

    def prepare_dataset(self, pair: str = "XAUUSD", timeframe: str = "H1", data_type: str = "training") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare a complete dataset for machine learning with features and targets."""
        try:
            # Get raw data
            df = self.get_data_from_db(pair, timeframe, data_type)
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Add technical indicators
            df = self.process_raw_data(df)

            # Create features
            df = self.create_features(df)

            # Create target variables
            df = self.create_target_variables(df)

            # Drop NaN values from the dataset
            df = self.handle_missing_values(df)

            # Prepare X and y
            ml_settings = self.gold_settings.get('MachineLearning', {})
            targets = ml_settings.get('Targets', {})

            # Determine target columns based on ML settings
            target_columns = []

            if 'PricePrediction' in targets:
                horizons = targets['PricePrediction'].get('Horizons', [1])
                for horizon in horizons:
                    target_columns.append(f'future_price_{horizon}')

            if 'DirectionPrediction' in targets:
                horizons = targets['PricePrediction'].get('Horizons', [1])
                for horizon in horizons:
                    target_columns.append(f'direction_{horizon}')
                    target_columns.append(f'signal_up_{horizon}')
                    target_columns.append(f'signal_down_{horizon}')

            # Remove target columns from features (but keep 'time')
            feature_columns = [col for col in df.columns if col not in target_columns]

            # Create X (features) and y (targets) dataframes
            X = df[feature_columns]  # Keep time column in X
            y = df[target_columns] if target_columns else pd.DataFrame()

            # Final check for any NaN values
            if 'time' in X.columns and X['time'].isna().any():
                self.logger.warning("Time column contains NaN values, fixing...")
                X = X.dropna(subset=['time'])
                if not y.empty:
                    y = y.loc[X.index]

            # Log the final column lists
            self.logger.info(f"Feature columns: {list(X.columns)}")
            if not y.empty:
                self.logger.info(f"Target columns: {list(y.columns)}")

            return X, y

        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise