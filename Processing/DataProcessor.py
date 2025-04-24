import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import Tuple, Dict, Any, Optional

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity
from Processing.TechnicalIndicators import TechnicalIndicators


class DataProcessor:
    def __init__(self, config: Config, logger: Logger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler

        # Get GoldTradingSettings from config - no fallbacks, must exist
        self.gold_settings = config.get('GoldTradingSettings')

        # Validate required config sections
        self._validate_config()

        self.db_config = config.get('Database')
        self.indicators = TechnicalIndicators()
        self.engine = self._create_engine()

        # Log indicator settings to ensure they're being loaded correctly
        self.logger.info(f"Loaded indicator settings: {self.gold_settings.get('Indicators', {})}")

    def _validate_config(self) -> None:
        """Validate that all required configuration sections exist"""
        context = {
            "class": self.__class__.__name__,
            "operation": "_validate_config"
        }

        if not self.gold_settings:
            self.error_handler.handle_error(
                ValueError("GoldTradingSettings missing from configuration"),
                context,
                ErrorSeverity.FATAL,
                reraise=True
            )

        # Validate essential sections
        required_sections = ['Indicators', 'FeatureEngineering']
        missing_sections = [section for section in required_sections if section not in self.gold_settings]

        if missing_sections:
            self.error_handler.handle_error(
                ValueError(f"Missing required configuration sections: {missing_sections}"),
                context,
                ErrorSeverity.FATAL,
                reraise=True
            )

        # Validate indicator subsections
        indicator_sections = ['MovingAverages', 'Volatility', 'Momentum', 'PivotPoints']
        missing_indicators = [section for section in indicator_sections
                              if section not in self.gold_settings['Indicators']]

        if missing_indicators:
            self.error_handler.handle_error(
                ValueError(f"Missing indicator configuration sections: {missing_indicators}"),
                context,
                ErrorSeverity.FATAL,
                reraise=True
            )

        # Validate feature engineering settings
        required_feature_settings = ['WindowSizes', 'PriceFeatures', 'DefaultColumn']
        missing_feature_settings = [setting for setting in required_feature_settings
                                    if setting not in self.gold_settings['FeatureEngineering']]

        if missing_feature_settings:
            self.error_handler.handle_error(
                ValueError(f"Missing feature engineering settings: {missing_feature_settings}"),
                context,
                ErrorSeverity.FATAL,
                reraise=True
            )

    @property
    def error_context(self) -> Dict[str, Any]:
        """Base context for error handling in this class"""
        return {
            "class": self.__class__.__name__,
            "db_host": self.db_config.get('Host', 'unknown'),
            "db_name": self.db_config.get('Database', 'unknown')
        }

    def _create_engine(self):
        """Create SQLAlchemy engine for database connections."""
        context = {
            **self.error_context,
            "operation": "_create_engine"
        }

        try:
            db = self.db_config
            connection_string = (
                f"mssql+pyodbc://{db['User']}:{db['Password']}@{db['Host']},{db['Port']}/"
                f"{db['Database']}?driver=ODBC+Driver+17+for+SQL+Server"
            )
            return create_engine(connection_string)
        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.FATAL,
                reraise=True
            )
            raise

    def get_data_from_db(self, pair: str = "XAUUSD", timeframe: str = "H1",
                         data_type: str = "training") -> pd.DataFrame:
        """Retrieve data from database."""
        context = {
            **self.error_context,
            "operation": "get_data_from_db",
            "pair": pair,
            "timeframe": timeframe,
            "data_type": data_type
        }

        try:
            table_name = f"{pair}_{timeframe.lower()}_{data_type}"
            context["table_name"] = table_name

            self.logger.info(f"Retrieving {data_type} data for {pair} {timeframe}")

            query = f"SELECT * FROM {table_name} ORDER BY time"
            df = pd.read_sql(query, self.engine)

            self.logger.info(f"Successfully retrieved {len(df)} rows from {table_name}")

            # Ensure datetime column is properly formatted
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])

            return df

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with technical indicators."""
        context = {
            **self.error_context,
            "operation": "process_raw_data",
            "df_shape": str(df.shape) if df is not None else "None"
        }

        try:
            if df.empty:
                self.error_handler.handle_error(
                    ValueError("Empty dataframe provided for processing"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=True
                )

            self.logger.info(f"Processing {len(df)} rows of raw data")

            # Make sure we have a proper copy to avoid modifying the original
            processed_df = df.copy()

            # Ensure time column is present and properly formatted
            if 'time' not in processed_df.columns:
                self.error_handler.handle_error(
                    ValueError("Time column missing from input data"),
                    context,
                    ErrorSeverity.HIGH,
                    reraise=True
                )

            processed_df['time'] = pd.to_datetime(processed_df['time'])

            # Calculate all technical indicators based on configuration
            processed_df = self.calculate_indicators(processed_df)

            # Remove NaN values that come from indicators using windows
            processed_df = self.handle_missing_values(processed_df)

            self.logger.info(f"Data processing complete with {len(processed_df)} rows remaining")
            return processed_df

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators based on configuration."""
        context = {
            **self.error_context,
            "operation": "calculate_indicators",
            "df_shape": str(df.shape) if df is not None else "None"
        }

        try:
            result = df.copy()

            # Get indicator settings from config
            ma_settings = self.gold_settings['Indicators']['MovingAverages']
            volatility_settings = self.gold_settings['Indicators']['Volatility']
            momentum_settings = self.gold_settings['Indicators']['Momentum']
            pivot_settings = self.gold_settings['Indicators']['PivotPoints']

            # Log the settings being used
            self.logger.info(f"MA settings: {ma_settings}")
            self.logger.info(f"Volatility settings: {volatility_settings}")
            self.logger.info(f"Momentum settings: {momentum_settings}")
            self.logger.info(f"Pivot settings: {pivot_settings}")

            # Get default column for indicators
            default_column = self.gold_settings['FeatureEngineering']['DefaultColumn']

            # Calculate Moving Averages
            if 'SMA' in ma_settings:
                periods = ma_settings['SMA']['Periods']
                self.logger.info(f"Calculating SMA with periods: {periods}")
                result = self.indicators.calculate_sma(result, column=default_column, periods=periods)

            if 'EMA' in ma_settings:
                periods = ma_settings['EMA']['Periods']
                self.logger.info(f"Calculating EMA with periods: {periods}")
                result = self.indicators.calculate_ema(result, column=default_column, periods=periods)

            if 'MACD' in ma_settings:
                fast_period = ma_settings['MACD']['FastPeriod']
                slow_period = ma_settings['MACD']['SlowPeriod']
                signal_period = ma_settings['MACD']['SignalPeriod']
                self.logger.info(
                    f"Calculating MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}")
                result = self.indicators.calculate_macd(
                    result, column=default_column, fast_period=fast_period,
                    slow_period=slow_period, signal_period=signal_period
                )

            # Calculate Volatility indicators
            if 'BollingerBands' in volatility_settings:
                period = volatility_settings['BollingerBands']['Period']
                num_std = volatility_settings['BollingerBands']['NumStd']
                self.logger.info(f"Calculating Bollinger Bands with period={period}, std={num_std}")
                result = self.indicators.calculate_bollinger_bands(
                    result, column=default_column, period=period, num_std=num_std
                )

            if 'ATR' in volatility_settings:
                period = volatility_settings['ATR']['Period']
                self.logger.info(f"Calculating ATR with period={period}")
                result = self.indicators.calculate_atr(result, period=period)

            # Calculate Momentum indicators
            if 'RSI' in momentum_settings:
                period = momentum_settings['RSI']['Period']
                self.logger.info(f"Calculating RSI with period={period}")
                result = self.indicators.calculate_rsi(result, column=default_column, period=period)

            if 'Stochastic' in momentum_settings:
                k_period = momentum_settings['Stochastic']['KPeriod']
                d_period = momentum_settings['Stochastic']['DPeriod']
                self.logger.info(f"Calculating Stochastic with k_period={k_period}, d_period={d_period}")
                result = self.indicators.calculate_stochastic(
                    result, k_period=k_period, d_period=d_period
                )

            # Calculate Pivot Points if configured
            method = pivot_settings['Method']
            self.logger.info(f"Calculating Pivot Points with method={method}")
            result = self.indicators.calculate_pivot_points(result, method=method)

            # Log columns after adding indicators
            self.logger.info(f"Columns after adding indicators: {list(result.columns)}")

            # Check if any indicators were added
            if len(result.columns) <= len(df.columns) + 2:  # Allow for a couple of extra columns
                self.error_handler.handle_error(
                    ValueError("Few or no indicators were added from configuration"),
                    context,
                    ErrorSeverity.HIGH,
                    reraise=True
                )

            return result

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        context = {
            **self.error_context,
            "operation": "handle_missing_values",
            "df_shape": str(df.shape) if df is not None else "None"
        }

        try:
            # Always drop NaN values in financial data processing
            original_len = len(df)
            df = df.dropna()
            dropped = original_len - len(df)
            if dropped > 0:
                self.logger.info(f"Dropped {dropped} rows with NaN values")
            return df
        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.MEDIUM,
                reraise=True
            )
            raise

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning."""
        context = {
            **self.error_context,
            "operation": "create_features",
            "df_shape": str(df.shape) if df is not None else "None"
        }

        try:
            feature_settings = self.gold_settings['FeatureEngineering']
            result = df.copy()

            # Log before feature creation
            self.logger.info(f"Starting feature engineering with columns: {list(result.columns)}")

            # Create price-based features
            price_features = feature_settings['PriceFeatures']

            # Create lagged features
            window_sizes = feature_settings['WindowSizes']
            for feature in price_features:
                if feature in result.columns:
                    # Create lag features - explicitly name these for validation
                    for window in window_sizes:
                        col_name = f'{feature}_lag_{window}'
                        result[col_name] = result[feature].shift(window)
                        self.logger.debug(f"Created lag feature: {col_name}")

                    # Create return features (percent change)
                    col_name = f'{feature}_pct_change'
                    # Explicitly set periods as positive to ensure we're using past data
                    result[col_name] = result[feature].pct_change(periods=1)
                    self.logger.debug(f"Created percent change feature: {col_name}")

                    for window in window_sizes:
                        col_name = f'{feature}_pct_change_{window}'
                        # Positive period to ensure we're using past data
                        result[col_name] = result[feature].pct_change(periods=window)
                        self.logger.debug(f"Created percent change with window feature: {col_name}")

                    # Create rolling statistics
                    for window in window_sizes:
                        if window > 1:  # Only create rolling stats for windows > 1
                            col_name = f'{feature}_rolling_mean_{window}'
                            result[col_name] = result[feature].rolling(window=window).mean()
                            self.logger.debug(f"Created rolling mean feature: {col_name}")

                            col_name = f'{feature}_rolling_std_{window}'
                            result[col_name] = result[feature].rolling(window=window).std()
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
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def prepare_processed_data(self, pair: str = "XAUUSD", timeframe: str = "H1",
                               data_type: str = "training") -> pd.DataFrame:
        """Prepare processed data with indicators and features."""
        context = {
            **self.error_context,
            "operation": "prepare_processed_data",
            "pair": pair,
            "timeframe": timeframe,
            "data_type": data_type
        }

        try:
            # Get raw data
            df = self.get_data_from_db(pair, timeframe, data_type)
            if df.empty:
                self.error_handler.handle_error(
                    ValueError(f"Empty dataframe returned from database for {pair} {timeframe} {data_type}"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=True
                )

            # Verify chronological ordering
            if 'time' in df.columns:
                if not df['time'].equals(df['time'].sort_values()):
                    self.error_handler.handle_error(
                        ValueError("Data not in chronological order"),
                        context,
                        ErrorSeverity.MEDIUM,
                        reraise=False
                    )
                    self.logger.warning("Data not in chronological order. Sorting by time.")
                    df = df.sort_values('time')
            else:
                self.error_handler.handle_error(
                    ValueError("Time column missing from input data"),
                    context,
                    ErrorSeverity.HIGH,
                    reraise=True
                )

            # Add technical indicators
            df = self.process_raw_data(df)

            # Create features
            df = self.create_features(df)

            # Ensure we maintain chronological order after processing
            if 'time' in df.columns:
                if not df['time'].equals(df['time'].sort_values()):
                    self.error_handler.handle_error(
                        ValueError("Order changed after processing"),
                        context,
                        ErrorSeverity.LOW,
                        reraise=False
                    )
                    self.logger.warning("Order changed after processing. Restoring chronological order.")
                    df = df.sort_values('time')

            self.logger.info(f"Processed data: {len(df)} rows with {len(df.columns)} columns")
            return df

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    # Add to Processing/DataProcessor.py

    def calculate_gold_specific_features(self, df: pd.DataFrame,
                                         correlation_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate gold-specific features like Gold/Silver ratio, USD correlation, etc."""
        context = {
            **self.error_context,
            "operation": "calculate_gold_specific_features",
            "df_shape": str(df.shape) if df is not None else "None",
            "available_corr_data": str(list(correlation_data.keys()))
        }

        try:
            result = df.copy()

            # Calculate CCI if not already present
            if 'cci' not in result.columns:
                result = self.indicators.calculate_cci(result, period=20)
                self.logger.info("Added CCI indicator to the dataset")

            # Synchronize timestamps for correlation calculations
            result_times = set(result['time'])

            # Calculate Gold/Silver ratio if both are available
            if "XAUUSD" in correlation_data and "XAGUSD" in correlation_data:
                gold_df = correlation_data["XAUUSD"]
                silver_df = correlation_data["XAGUSD"]

                # Ensure we only work with matching timestamps
                gold_df = gold_df[gold_df['time'].isin(result_times)]
                silver_df = silver_df[silver_df['time'].isin(result_times)]

                # Create a common index based on time
                common_times = set(gold_df['time']).intersection(set(silver_df['time']))
                gold_df = gold_df[gold_df['time'].isin(common_times)]
                silver_df = silver_df[silver_df['time'].isin(common_times)]

                # Ensure data is sorted by time
                gold_df = gold_df.sort_values('time')
                silver_df = silver_df.sort_values('time')

                # Calculate Gold/Silver ratio
                if not gold_df.empty and not silver_df.empty:
                    # Create a DataFrame with time and ratio columns
                    ratio_df = pd.DataFrame()
                    ratio_df['time'] = gold_df['time']
                    ratio_df['gold_silver_ratio'] = gold_df['close'] / silver_df['close']

                    # Merge ratio with the main dataframe
                    result = pd.merge(result, ratio_df, on='time', how='left')
                    self.logger.info("Added Gold/Silver ratio to the dataset")

            # Calculate USD Index correlation if available
            if "USDX" in correlation_data:
                usd_df = correlation_data["USDX"]

                # Ensure we only work with matching timestamps
                usd_df = usd_df[usd_df['time'].isin(result_times)]

                if not usd_df.empty:
                    # Create correlation features
                    usd_corr_df = pd.DataFrame()
                    usd_corr_df['time'] = usd_df['time']
                    usd_corr_df['usd_index'] = usd_df['close']

                    # Calculate USD returns
                    usd_corr_df['usd_returns'] = usd_df['close'].pct_change()

                    # Merge with the main dataframe
                    result = pd.merge(result, usd_corr_df, on='time', how='left')

                    # Calculate correlation between Gold and USD returns
                    # (using rolling window for ongoing correlation)
                    # First we need close returns for gold
                    if 'close_pct_change' not in result.columns:
                        result['close_pct_change'] = result['close'].pct_change()

                    # Now calculate rolling correlation
                    window_sizes = [5, 10, 20]
                    for window in window_sizes:
                        result[f'gold_usd_corr_{window}'] = result['close_pct_change'].rolling(window).corr(
                            result['usd_returns'])

                    self.logger.info("Added USD Index correlation features to the dataset")

            # Calculate VIX relationship features if available
            if "VIX" in correlation_data:
                vix_df = correlation_data["VIX"]

                # Ensure we only work with matching timestamps
                vix_df = vix_df[vix_df['time'].isin(result_times)]

                if not vix_df.empty:
                    # Create VIX features
                    vix_feature_df = pd.DataFrame()
                    vix_feature_df['time'] = vix_df['time']
                    vix_feature_df['vix'] = vix_df['close']

                    # Add VIX returns
                    vix_feature_df['vix_returns'] = vix_df['close'].pct_change()

                    # Merge with the main dataframe
                    result = pd.merge(result, vix_feature_df, on='time', how='left')

                    # Calculate correlation between Gold and VIX
                    if 'close_pct_change' not in result.columns:
                        result['close_pct_change'] = result['close'].pct_change()

                    # Calculate rolling correlation
                    window_sizes = [5, 10, 20]
                    for window in window_sizes:
                        result[f'gold_vix_corr_{window}'] = result['close_pct_change'].rolling(window).corr(
                            result['vix_returns'])

                    self.logger.info("Added VIX relationship features to the dataset")

            return result

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def process_raw_data(self, df: pd.DataFrame,
                         correlation_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Process raw data with technical indicators and gold-specific features."""
        context = {
            **self.error_context,
            "operation": "process_raw_data",
            "df_shape": str(df.shape) if df is not None else "None",
            "has_correlation_data": correlation_data is not None
        }

        try:
            if df.empty:
                self.error_handler.handle_error(
                    ValueError("Empty dataframe provided for processing"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=True
                )

            self.logger.info(f"Processing {len(df)} rows of raw data")

            # Make sure we have a proper copy to avoid modifying the original
            processed_df = df.copy()

            # Ensure time column is present and properly formatted
            if 'time' not in processed_df.columns:
                self.error_handler.handle_error(
                    ValueError("Time column missing from input data"),
                    context,
                    ErrorSeverity.HIGH,
                    reraise=True
                )

            processed_df['time'] = pd.to_datetime(processed_df['time'])

            # Calculate all technical indicators based on configuration
            processed_df = self.calculate_indicators(processed_df)

            # Add gold-specific features if correlation data is available
            if correlation_data:
                processed_df = self.calculate_gold_specific_features(processed_df, correlation_data)
                self.logger.info("Added gold-specific features to the dataset")

            # Remove NaN values that come from indicators using windows
            processed_df = self.handle_missing_values(processed_df)

            self.logger.info(f"Data processing complete with {len(processed_df)} rows remaining")
            return processed_df

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise