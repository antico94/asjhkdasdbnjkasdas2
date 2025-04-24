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

    def process_raw_data(self, df: pd.DataFrame,
                         correlation_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Process raw data with technical indicators."""
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

            # Note: We no longer add gold-specific features here
            # They will be added separately after storing correlation data

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

    def add_gold_features(self, df: pd.DataFrame,
                          correlation_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add gold-specific features using correlation data loaded from tables.

        This method adds gold-specific features without dropping rows that don't
        have correlation data.
        """
        context = {
            **self.error_context,
            "operation": "add_gold_features",
            "df_shape": str(df.shape) if df is not None else "None",
            "available_corr_data": str(list(correlation_data.keys()))
        }

        try:
            result = df.copy()

            # Calculate CCI if not already present
            if 'cci' not in result.columns:
                result = self.indicators.calculate_cci(result, period=20)
                self.logger.info("Added CCI indicator to the dataset")

            # Log original data time range
            self.logger.info(f"Original data: {len(result)} rows from {result['time'].min()} to {result['time'].max()}")

            # Calculate Gold/Silver ratio if both are available
            if "XAUUSD" in correlation_data and "XAGUSD" in correlation_data:
                gold_df = correlation_data["XAUUSD"].copy()
                silver_df = correlation_data["XAGUSD"].copy()

                # Log correlation data details
                self.logger.info(
                    f"Gold data: {len(gold_df)} rows from {gold_df['time'].min()} to {gold_df['time'].max()}")
                self.logger.info(
                    f"Silver data: {len(silver_df)} rows from {silver_df['time'].min()} to {silver_df['time'].max()}")

                # Calculate Gold/Silver ratio
                if not gold_df.empty and not silver_df.empty:
                    # Create a temporary dataframe with both gold and silver data
                    ratio_df = pd.merge(
                        gold_df[['time', 'close']].rename(columns={'close': 'gold_close'}),
                        silver_df[['time', 'close']].rename(columns={'close': 'silver_close'}),
                        on='time',
                        how='inner'  # Only use timestamps that exist in both datasets
                    )

                    # Calculate the ratio
                    ratio_df['gold_silver_ratio'] = ratio_df['gold_close'] / ratio_df['silver_close']

                    # Log ratio data details
                    self.logger.info(
                        f"Ratio data: {len(ratio_df)} rows from {ratio_df['time'].min()} to {ratio_df['time'].max()}")

                    # Merge with main dataframe using left join to avoid losing data
                    result = pd.merge(
                        result,
                        ratio_df[['time', 'gold_silver_ratio']],
                        on='time',
                        how='left'  # Use left join to keep all original rows
                    )

                    self.logger.info(
                        f"After ratio merge: {len(result)} rows, Nulls in ratio: {result['gold_silver_ratio'].isna().sum()}")

            # Calculate USD Index correlation if available
            if "USDX" in correlation_data:
                usd_df = correlation_data["USDX"].copy()

                if not usd_df.empty:
                    # Calculate USD returns
                    usd_df['usd_returns'] = usd_df['close'].pct_change()

                    # Create a dataframe with USD features
                    usd_feature_df = usd_df[['time', 'close', 'usd_returns']].rename(
                        columns={'close': 'usd_index'}
                    ).dropna()

                    # Log USD data details
                    self.logger.info(
                        f"USD data: {len(usd_feature_df)} rows from {usd_feature_df['time'].min()} to {usd_feature_df['time'].max()}")

                    # Merge with main dataframe using left join
                    result = pd.merge(
                        result,
                        usd_feature_df,
                        on='time',
                        how='left'  # Use left join to keep all original rows
                    )

                    self.logger.info(
                        f"After USD merge: {len(result)} rows, Nulls in USD index: {result['usd_index'].isna().sum()}")

                    # Calculate correlation between Gold and USD returns
                    # First we need close returns for gold if not already present
                    if 'close_pct_change' not in result.columns:
                        result['close_pct_change'] = result['close'].pct_change()

                    # Instead of rolling correlation which would drop rows,
                    # we'll create a feature that indicates if gold and USD are moving in the same direction
                    result['gold_usd_same_direction'] = (
                            (result['close_pct_change'] >= 0) & (result['usd_returns'] >= 0) |
                            (result['close_pct_change'] < 0) & (result['usd_returns'] < 0)
                    )

                    # For rolling correlation, we'll add it but not drop rows with NaN
                    window_sizes = [5, 10, 20]
                    for window in window_sizes:
                        # Use a separate DataFrame for correlation calculation to avoid affecting the main result
                        corr_df = pd.DataFrame({
                            'time': result['time'],
                            'gold_returns': result['close_pct_change'],
                            'usd_returns': result['usd_returns']
                        })

                        # Calculate rolling correlation
                        corr_series = corr_df['gold_returns'].rolling(window).corr(corr_df['usd_returns'])

                        # Create a temporary dataframe with time and correlation
                        temp_df = pd.DataFrame({
                            'time': corr_df['time'],
                            f'gold_usd_corr_{window}': corr_series
                        })

                        # Merge back with the result
                        result = pd.merge(
                            result,
                            temp_df,
                            on='time',
                            how='left'  # Use left join to keep all original rows
                        )

            # Calculate VIX relationship features if available
            if "VIX" in correlation_data:
                vix_df = correlation_data["VIX"].copy()

                if not vix_df.empty:
                    # Check if has_real_vix flag exists, otherwise set it to 1 (for backward compatibility)
                    if 'has_real_vix' not in vix_df.columns:
                        vix_df['has_real_vix'] = 1

                    # Calculate VIX returns
                    vix_df['vix_returns'] = vix_df['close'].pct_change()

                    # Create a dataframe with VIX features
                    vix_feature_df = vix_df[['time', 'close', 'vix_returns', 'has_real_vix']].rename(
                        columns={'close': 'vix'}
                    ).dropna(subset=['vix', 'vix_returns'])  # Only drop rows where these specific columns are NA

                    # Log VIX data details
                    self.logger.info(
                        f"VIX data: {len(vix_feature_df)} rows from {vix_feature_df['time'].min()} to {vix_feature_df['time'].max()}")

                    # Count how many records have real VIX data
                    real_vix_count = vix_feature_df['has_real_vix'].sum()
                    total_vix_count = len(vix_feature_df)
                    self.logger.info(
                        f"Real VIX data: {real_vix_count} of {total_vix_count} records ({real_vix_count / total_vix_count * 100:.1f}%)")

                    # Merge with main dataframe using left join
                    result = pd.merge(
                        result,
                        vix_feature_df,
                        on='time',
                        how='left'  # Use left join to keep all original rows
                    )

                    self.logger.info(f"After VIX merge: {len(result)} rows, Nulls in VIX: {result['vix'].isna().sum()}")

                    # Set has_real_vix to 0 where it's missing/null
                    result['has_real_vix'] = result['has_real_vix'].fillna(0).astype('int64')

                    # Create same direction indicator
                    if 'close_pct_change' not in result.columns:
                        result['close_pct_change'] = result['close'].pct_change()

                    # Calculate correlation only where VIX data exists
                    vix_exists_mask = result['vix'].notna()

                    if vix_exists_mask.any():
                        # Only compute direction for rows with VIX data
                        result.loc[vix_exists_mask, 'gold_vix_same_direction'] = (
                                (result.loc[vix_exists_mask, 'close_pct_change'] >= 0) &
                                (result.loc[vix_exists_mask, 'vix_returns'] >= 0) |
                                (result.loc[vix_exists_mask, 'close_pct_change'] < 0) &
                                (result.loc[vix_exists_mask, 'vix_returns'] < 0)
                        )

                        # For rolling correlation, add without dropping NaN values
                        window_sizes = [5, 10, 20]
                        for window in window_sizes:
                            # Create a temporary DataFrame with data only where VIX exists
                            temp_df = pd.DataFrame({
                                'time': result.loc[vix_exists_mask, 'time'],
                                'gold_returns': result.loc[vix_exists_mask, 'close_pct_change'],
                                'vix_returns': result.loc[vix_exists_mask, 'vix_returns']
                            })

                            if not temp_df.empty:
                                # Calculate rolling correlation
                                corr_series = temp_df['gold_returns'].rolling(window).corr(temp_df['vix_returns'])

                                # Create a dataframe with time and correlation
                                corr_df = pd.DataFrame({
                                    'time': temp_df['time'],
                                    f'gold_vix_corr_{window}': corr_series
                                })

                                # Merge back with the result
                                result = pd.merge(
                                    result,
                                    corr_df,
                                    on='time',
                                    how='left'  # Use left join to keep all original rows
                                )

                    # Log VIX features after adding
                    vix_cols = [col for col in result.columns if 'vix' in col.lower()]
                    self.logger.info(f"Added VIX features: {vix_cols}")

                    # Log the coverage of VIX data in the dataset
                    vix_coverage = (~result['vix'].isna()).mean() * 100
                    self.logger.info(f"VIX data coverage: {vix_coverage:.2f}% of records have VIX data")

            # Calculate the final number of nulls for correlation features
            null_count = sum(result[col].isna().sum() for col in result.columns if 'corr_' in col)
            self.logger.info(f"Total null values in correlation features: {null_count}")

            # Log the final dataset shape
            self.logger.info(
                f"Final dataset after adding gold features: {len(result)} rows with {len(result.columns)} columns")

            return result

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise