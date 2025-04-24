from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity


class TechnicalIndicators:
    """Calculate technical indicators for financial market data."""

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler

    def _get_error_context(self, operation: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build error context dictionary with class and operation info"""
        context = {
            "class": self.__class__.__name__,
            "operation": operation
        }

        if params:
            context.update(params)

        return context

    def calculate_sma(self, data: pd.DataFrame, column: str = 'close',
                      periods: List[int] = None) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        if periods is None:
            periods = [5, 8, 13, 21, 50, 200]

        context = self._get_error_context(
            "calculate_sma",
            {"column": column, "periods": str(periods), "data_shape": str(data.shape) if data is not None else "None"}
        )

        try:
            result = data.copy()

            # Validate inputs
            if column not in result.columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Column '{column}' not found in dataframe"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Column '{column}' not found in dataframe")

            for period in periods:
                result[f'sma_{period}'] = result[column].rolling(window=period).mean()
            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    def calculate_ema(self, data: pd.DataFrame, column: str = 'close',
                      periods: List[int] = None) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        if periods is None:
            periods = [5, 8, 13, 21, 50, 200]

        context = self._get_error_context(
            "calculate_ema",
            {"column": column, "periods": str(periods), "data_shape": str(data.shape) if data is not None else "None"}
        )

        try:
            result = data.copy()

            # Validate inputs
            if column not in result.columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Column '{column}' not found in dataframe"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Column '{column}' not found in dataframe")

            for period in periods:
                result[f'ema_{period}'] = result[column].ewm(span=period, adjust=False).mean()
            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    def calculate_macd(self, data: pd.DataFrame, column: str = 'close',
                       fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        context = self._get_error_context(
            "calculate_macd",
            {
                "column": column,
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "data_shape": str(data.shape) if data is not None else "None"
            }
        )

        try:
            result = data.copy()

            # Validate inputs
            if column not in result.columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Column '{column}' not found in dataframe"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Column '{column}' not found in dataframe")

            if fast_period >= slow_period:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")

            # Fast EMA
            fast_ema = result[column].ewm(span=fast_period, adjust=False).mean()
            # Slow EMA
            slow_ema = result[column].ewm(span=slow_period, adjust=False).mean()
            # MACD Line
            result['macd_line'] = fast_ema - slow_ema
            # Signal Line
            result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=False).mean()
            # MACD Histogram
            result['macd_histogram'] = result['macd_line'] - result['macd_signal']
            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    def calculate_rsi(self, data: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        context = self._get_error_context(
            "calculate_rsi",
            {
                "column": column,
                "period": period,
                "data_shape": str(data.shape) if data is not None else "None"
            }
        )

        try:
            result = data.copy()

            # Validate inputs
            if column not in result.columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Column '{column}' not found in dataframe"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Column '{column}' not found in dataframe")

            if period is None or not isinstance(period, int) or period <= 0:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Period ({period}) must be a positive integer"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Period ({period}) must be a positive integer")

            delta = result[column].diff()

            # Make two series: one for gains and one for losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gain and loss using Wilder's smoothing (typical for RSI)
            # The .ewm(alpha=1/period, adjust=False) approach is common for Wilder's smoothing
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()


            # Calculate Relative Strength (RS)
            # Handle division by zero by filling inf values with NaN
            rs = avg_gain / avg_loss
            rs = rs.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN

            # Calculate RSI
            # Handle cases where avg_loss is 0 (RS is inf or NaN)
            # If avg_loss is 0 and avg_gain > 0, RSI is 100
            # If avg_loss is 0 and avg_gain is 0, RSI is 0 (or NaN depending on convention)
            rsi = 100 - (100 / (1 + rs))

            # Explicitly handle the case where avg_loss is 0 and avg_gain > 0
            rsi[(avg_loss == 0) & (avg_gain > 0)] = 100
             # Handle the case where both are 0 (or NaN/inf propagated) - often results in NaN or 0
             # The default NaN propagation should handle most cases, but explicit handling can be added if needed
             # For now, rely on default NaN behavior or adjust based on specific RSI definition

            result['rsi'] = rsi
            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    def calculate_bollinger_bands(self, data: pd.DataFrame, column: str = 'close',
                                  period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        context = self._get_error_context(
            "calculate_bollinger_bands",
            {
                "column": column,
                "period": period,
                "num_std": num_std,
                "data_shape": str(data.shape) if data is not None else "None"
            }
        )

        try:
            result = data.copy()

            # Validate inputs
            if column not in result.columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Column '{column}' not found in dataframe"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Column '{column}' not found in dataframe")

            if period is None or not isinstance(period, int) or period <= 0:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Period ({period}) must be a positive integer"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Period ({period}) must be a positive integer")

            result['bb_middle'] = result[column].rolling(window=period).mean()
            result['bb_std'] = result[column].rolling(window=period).std()
            result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * num_std)
            result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * num_std)
            # Handle division by zero for bb_width if bb_middle is zero (unlikely for prices)
            bb_middle_safe = result['bb_middle'].replace(0, np.nan)
            result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / bb_middle_safe

            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        context = self._get_error_context(
            "calculate_atr",
            {
                "period": period,
                "data_shape": str(data.shape) if data is not None else "None"
            }
        )

        try:
            result = data.copy()

            # Validate required columns
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in result.columns]
            if missing_columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Missing required columns: {missing_columns}"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Missing required columns: {missing_columns}")

            if period is None or not isinstance(period, int) or period <= 0:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Period ({period}) must be a positive integer"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Period ({period}) must be a positive integer")

            high = result['high']
            low = result['low']
            close = result['close']

            # True Range calculation
            # Use .shift(1) for previous close
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result['atr'] = tr.ewm(alpha=1/period, adjust=False).mean() # Use EWM for ATR smoothing
            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        context = self._get_error_context(
            "calculate_stochastic",
            {
                "k_period": k_period,
                "d_period": d_period,
                "data_shape": str(data.shape) if data is not None else "None"
            }
        )

        try:
            result = data.copy()

            # Validate required columns
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in result.columns]
            if missing_columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Missing required columns: {missing_columns}"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Missing required columns: {missing_columns}")

            if k_period is None or not isinstance(k_period, int) or k_period <= 0 or \
               d_period is None or not isinstance(d_period, int) or d_period <= 0:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Periods (k_period={k_period}, d_period={d_period}) must be positive integers"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Periods (k_period={k_period}, d_period={d_period}) must be positive integers")


            # %K calculation
            lowest_low = result['low'].rolling(window=k_period).min()
            highest_high = result['high'].rolling(window=k_period).max()

            # Handle division by zero (when highest_high == lowest_low)
            denom = highest_high - lowest_low
            close_low_diff = result['close'] - lowest_low

            # Initialize %K with NaNs
            result['stoch_k'] = pd.Series(float('nan'), index=result.index)

            # Calculate %K where denominator is not zero
            mask = denom != 0
            result.loc[mask, 'stoch_k'] = 100 * (close_low_diff[mask] / denom[mask])

            # Handle the case where denom is zero (no price movement in the period)
            # Set %K to 50 when denominator is zero as a convention
            result.loc[~mask, 'stoch_k'] = 50

            # %D calculation (d_period period SMA of %K)
            result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    def calculate_pivot_points(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Calculate daily pivot points based on previous day's data.
        Methods: 'standard', 'fibonacci', 'woodie', 'camarilla'
        """
        context = self._get_error_context(
            "calculate_pivot_points",
            {
                "method": method,
                "data_shape": str(data.shape) if data is not None else "None"
            }
        )

        try:
            result = data.copy()

            # Validate required columns
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in result.columns]
            if missing_columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Missing required columns: {missing_columns}"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Missing required columns: {missing_columns}")

             # Validate method parameter
            valid_methods = ['standard', 'fibonacci', 'woodie', 'camarilla']
            if method not in valid_methods:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Invalid pivot point method: {method}. Valid options are {valid_methods}"),
                        context,
                        ErrorSeverity.MEDIUM,
                        reraise=True
                    )
                raise ValueError(f"Invalid pivot point method: {method}. Valid options are {valid_methods}")

            # Resample data to daily to get previous day's HLC
            # This requires the input data to have a datetime index or a 'time' column
            if 'time' in result.columns:
                 # Make sure 'time' is the index for resampling
                 temp_df = result.set_index('time')
            elif isinstance(result.index, pd.DatetimeIndex):
                 temp_df = result.copy()
            else:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError("Input DataFrame must have a 'time' column or DatetimeIndex for Pivot Points"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError("Input DataFrame must have a 'time' column or DatetimeIndex for Pivot Points")


            # Get the last OHLC of the previous day
            # Use .iloc[-2] to get the *second* to last row (yesterday) after grouping
            # and then get the values
            ohlc_prev_day = temp_df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna().iloc[:-1] # Drop the current incomplete day and any NaNs


            if ohlc_prev_day.empty:
                 # Not enough data for pivot points
                 result['pivot'] = np.nan
                 result['support1'] = np.nan
                 result['support2'] = np.nan
                 result['support3'] = np.nan # Fibonacci/Camarilla
                 result['support4'] = np.nan # Camarilla
                 result['resistance1'] = np.nan
                 result['resistance2'] = np.nan
                 result['resistance3'] = np.nan # Fibonacci/Camarilla
                 result['resistance4'] = np.nan # Camarilla
                 return result


            # Map daily pivots back to the original high-frequency data
            # Calculate pivots for each *day* based on the previous day's OHLC
            daily_pivots = {}
            for prev_day_time, row in ohlc_prev_day.iterrows():
                h = row['high']
                l = row['low']
                c = row['close']
                o = row['open'] # Needed for some methods (e.g., Woodie if using specific formula)

                if method == 'standard':
                    pp = (h + l + c) / 3
                    s1 = (2 * pp) - h
                    s2 = pp - (h - l)
                    r1 = (2 * pp) - l
                    r2 = pp + (h - l)
                    daily_pivots[prev_day_time + pd.Timedelta(days=1)] = {'pivot': pp, 'support1': s1, 'support2': s2, 'resistance1': r1, 'resistance2': r2}

                elif method == 'fibonacci':
                    pp = (h + l + c) / 3
                    r1 = pp + 0.382 * (h - l)
                    r2 = pp + 0.618 * (h - l)
                    r3 = pp + (h - l)
                    s1 = pp - 0.382 * (h - l)
                    s2 = pp - 0.618 * (h - l)
                    s3 = pp - (h - l)
                    daily_pivots[prev_day_time + pd.Timedelta(days=1)] = {'pivot': pp, 'support1': s1, 'support2': s2, 'support3': s3, 'resistance1': r1, 'resistance2': r2, 'resistance3': r3}

                elif method == 'woodie':
                    # Note: Woodie's standard pivot often uses Open, High, Low, Twice Close / 4
                    # Some variations exist. Using H, L, C as per your original code's structure intention
                    pp = (h + l + 2 * c) / 4
                    r1 = (2 * pp) - l
                    r2 = pp + (h - l)
                    s1 = (2 * pp) - h
                    s2 = pp - (h - l)
                     # Some Woodie definitions have additional levels (R3, S3 etc.) but using your original structure
                    daily_pivots[prev_day_time + pd.Timedelta(days=1)] = {'pivot': pp, 'support1': s1, 'support2': s2, 'resistance1': r1, 'resistance2': r2}

                elif method == 'camarilla':
                    pp = (h + l + c) / 3 # Camarilla often doesn't use PP directly, but calculate for consistency
                    range_val = h - l
                    r1 = c + range_val * 1.1 / 12
                    r2 = c + range_val * 1.1 / 6
                    r3 = c + range_val * 1.1 / 4
                    r4 = c + range_val * 1.1 / 2
                    s1 = c - range_val * 1.1 / 12
                    s2 = c - range_val * 1.1 / 6
                    s3 = c - range_val * 1.1 / 4
                    s4 = c - range_val * 1.1 / 2
                    daily_pivots[prev_day_time + pd.Timedelta(days=1)] = {'pivot': pp, 'support1': s1, 'support2': s2, 'support3': s3, 'support4': s4, 'resistance1': r1, 'resistance2': r2, 'resistance3': r3, 'resistance4': r4}


            # Convert daily pivots dictionary to DataFrame
            daily_pivots_df = pd.DataFrame.from_dict(daily_pivots, orient='index')
            daily_pivots_df.index.name = 'day' # Index is the start of the day for which pivots are valid


            # Merge daily pivots back to the original high-frequency data
            # Need to map each timestamp in result to the start of its day to merge with daily_pivots_df
            result['day'] = result['time'].dt.normalize() # Get the date part

            # Use left merge to keep all rows from result and add pivot columns
            result = pd.merge(result, daily_pivots_df, left_on='day', right_index=True, how='left')

            # Drop the temporary 'day' column
            result = result.drop(columns=['day'])


            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    # --- CORRECTED calculate_cci METHOD ---
    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        context = self._get_error_context(
            "calculate_cci",
            {
                "period": period,
                "data_shape": str(data.shape) if data is not None else "None"
            }
        )

        try:
            if data is None or data.empty:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError("Input DataFrame is None or empty"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError("Input DataFrame is None or empty")

            result = data.copy()

            # Validate required columns
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in result.columns]
            if missing_columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Missing required columns: {missing_columns}"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Missing required columns: {missing_columns}")

            if period is None or not isinstance(period, int) or period <= 0:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError(f"Period ({period}) must be a positive integer"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                raise ValueError(f"Period ({period}) must be a positive integer")

            # Calculate typical price
            tp = (result['high'] + result['low'] + result['close']) / 3

            # Calculate the simple moving average of the typical price
            tp_sma = tp.rolling(window=period).mean()

            # Calculate the mean deviation
            # Use numpy to calculate Mean Absolute Deviation within the rolling window
            # raw=True is generally more efficient when the lambda returns a scalar from numpy operations
            mean_deviation = tp.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
            )

            # Calculate CCI
            # Initialize the 'cci' column with NaN to correctly handle periods with insufficient data
            result['cci'] = np.nan
            # Create a mask where mean_deviation is not zero and not NaN
            mask = (mean_deviation.notna()) & (mean_deviation != 0)

            # Apply the CCI calculation only where the mask is True
            result.loc[mask, 'cci'] = (tp[mask] - tp_sma[mask]) / (0.015 * mean_deviation[mask])

            # Return the original dataframe with the new 'cci' column added
            return result # Return the whole df, not just [['cci']]

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

        # Assuming necessary imports (numpy, pandas, etc.) are at the top of the file
        # Assuming _get_error_context and self.error_handler are defined in the class __init__

# Assuming necessary imports (numpy, pandas, etc.) are at the top of the file
    # Assuming _get_error_context and self.error_handler are defined in the class __init__

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with default parameters."""
        context = self._get_error_context(
            "calculate_all_indicators",
            {"data_shape": str(data.shape) if data is not None else "None"}
        )

        try:
            if data is None or data.empty:
                 if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError("Input DataFrame is None or empty"),
                        context,
                        ErrorSeverity.HIGH,
                        reraise=True
                    )
                 raise ValueError("Input DataFrame is None or empty")

            result = data.copy()

            # Calculate Moving Averages by calling the methods
            # Ensure these methods exist in the TechnicalIndicators class
            # and handle their own input validation/error handling.
            result = self.calculate_sma(result, periods=[5, 8, 13, 21, 50, 200])
            result = self.calculate_ema(result, periods=[5, 8, 13, 21, 50, 200])
            result = self.calculate_macd(result, fast_period=12, slow_period=26, signal_period=9)

            # Calculate Volatility indicators by calling the methods
            # Ensure these methods exist in the TechnicalIndicators class
            result = self.calculate_bollinger_bands(result, period=20, num_std=2.0)
            result = self.calculate_atr(result, period=14)

            # Calculate Momentum indicators by calling the methods
            # Ensure these methods exist in the TechnicalIndicators class
            result = self.calculate_rsi(result, period=14)
            result = self.calculate_stochastic(result, k_period=14, d_period=3)

            # Calculate Pivot Points by calling the method
            # Ensure this method exists in the TechnicalIndicators class and
            # correctly handles the data frequency (e.g., resampling to daily).
            result = self.calculate_pivot_points(result, method="standard")

            # Calculate CCI by calling the method
            # Ensure the corrected calculate_cci method exists at the class level.
            result = self.calculate_cci(result, period=20) # <--- Call the corrected CCI method here

            # Add other indicators (implement methods for these if not already done, then call them)
            # For now, keeping direct calculations based on your previous code structure.
            # Ensure these direct calculations are free of errors and deprecated features.

            # Calculate Rate of Change (RoC) - Example direct calculation
            # Validate 'close' column exists
            if 'close' in result.columns:
                periods = [5, 10, 20]
                for period in periods:
                    if period is not None and isinstance(period, int) and period > 0:
                         result[f'roc_{period}'] = (result['close'] / result['close'].shift(period) - 1) * 100
                    else:
                         # Replace print with self.logger.warning if logger is available
                         print(f"Warning: Invalid period ({period}) for RoC calculation.")
            else:
                # Replace print with self.logger.warning if logger is available
                print(f"Warning: Cannot calculate RoC: Missing 'close' column.")


            # Calculate Average Directional Index (ADX) - Example direct calculation
            # Validate required columns
            required_adx = ['high', 'low', 'close']
            if all(col in result.columns for col in required_adx):
                period = 14 # ADX period
                # Ensure period is valid
                if period is not None and isinstance(period, int) and period > 0:
                    plus_dm = result['high'].diff()
                    # Correct calculation for minus_dm absolute difference from previous low
                    # Handle potential NaNs from shift
                    low_shifted = result['low'].shift(1)
                    minus_dm = (result['low'] - low_shifted).abs()

                    # Filter out negative differences
                    plus_dm[plus_dm < 0] = 0
                    minus_dm[minus_dm < 0] = 0

                    tr1 = (result['high'] - result['low']).abs()
                    # Handle potential NaNs from shift
                    close_shifted = result['close'].shift(1)
                    tr2 = (result['high'] - close_shifted).abs()
                    tr3 = (result['low'] - close_shifted).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                    # Use EWM smoothing for ATR within ADX calculation as per standard ADX
                    # Handle potential division by zero if atr_adx is 0
                    atr_adx = tr.ewm(alpha=1/period, adjust=False).mean()
                    atr_adx_safe = atr_adx.replace(0, np.nan) # Replace 0 with NaN for safe division


                    # Smoothed DM
                    plus_smooth_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
                    minus_smooth_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()


                    # Handle division by zero for DI calculations
                    # Initialize with NaN and fill where denom is not NaN and > 0
                    plus_di = pd.Series(np.nan, index=result.index)
                    minus_di = pd.Series(np.nan, index=result.index)

                    mask_di = atr_adx_safe.notna() # Valid where smoothed ATR is not NaN
                    plus_di[mask_di] = 100 * (plus_smooth_dm[mask_di] / atr_adx_safe[mask_di])
                    minus_di[mask_di] = 100 * (minus_smooth_dm[mask_di] / atr_adx_safe[mask_di])


                    # Handle division by zero for DX calculation
                    # Initialize with NaN
                    dx = pd.Series(np.nan, index=result.index)
                    di_sum = plus_di.abs() + minus_di.abs()

                    # Valid where sum is not NaN and > 0
                    mask_dx = (di_sum > 0) & (di_sum.notna())
                    dx[mask_dx] = 100 * ((plus_di[mask_dx] - minus_di[mask_dx]).abs() / di_sum[mask_dx])

                    # ADX is smoothed DX
                    # Use EWM for final ADX smoothing
                    result['adx'] = dx.ewm(alpha=1/period, adjust=False).mean()
                    result['plus_di'] = plus_di
                    result['minus_di'] = minus_di
                else:
                     # Replace print with self.logger.warning if logger is available
                     print(f"Warning: Invalid period ({period}) for ADX calculation.")
            else:
                # Replace print with self.logger.warning if logger is available
                print(f"Warning: Cannot calculate ADX: Missing required columns {required_adx}.")


            # Calculate Money Flow Index (MFI) - Example direct calculation
            # Validate required columns
            required_mfi = ['high', 'low', 'close', 'tick_volume']
            if all(col in result.columns for col in required_mfi):
                period = 14 # MFI period
                 # Ensure period is valid
                if period is not None and isinstance(period, int) and period > 0:
                    # Ensure tick_volume is numeric
                    if not pd.api.types.is_numeric_dtype(result['tick_volume']):
                         # Replace print with self.logger.warning/error if logger is available
                         print("Error: 'tick_volume' is not numeric. Cannot calculate MFI.")
                         # Skip MFI calculation
                    else:
                        typical_price = (result['high'] + result['low'] + result['close']) / 3
                        # Handle potential NaNs in typical_price or tick_volume
                        money_flow = typical_price * result['tick_volume']

                        # Calculate Price Change for Direction - use .diff(1)
                        price_change = typical_price.diff(periods=1)

                        positive_flow = pd.Series(0.0, index=result.index) # Use float
                        negative_flow = pd.Series(0.0, index=result.index) # Use float

                        # Determine positive and negative flow based on price change
                        # Use .loc for assignment based on boolean indexing
                        positive_flow.loc[price_change > 0] = money_flow[price_change > 0]
                        negative_flow.loc[price_change < 0] = money_flow[price_change < 0]

                        # Calculate sums of positive and negative money flow over the period
                        # Use .rolling().sum()
                        positive_mf = positive_flow.rolling(window=period).sum()
                        negative_mf = negative_flow.rolling(window=period).sum()

                        # Calculate Money Ratio and MFI
                        # Handle division by zero and edge cases
                        money_ratio = pd.Series(np.nan, index=result.index)
                        # Valid ratio calculation where negative_mf is not zero and not NaN
                        mask_valid_ratio = (negative_mf != 0) & (negative_mf.notna())
                        money_ratio[mask_valid_ratio] = positive_mf[mask_valid_ratio] / negative_mf[mask_valid_ratio]


                        mfi = pd.Series(np.nan, index=result.index)
                        # Standard calculation where money_ratio is valid and not NaN
                        mask_mfi = money_ratio.notna()
                        mfi[mask_mfi] = 100 - (100 / (1 + money_ratio[mask_mfi]))

                        # Special case: negative_mf == 0 and positive_mf > 0 --> MFI = 100
                        mask_mfi_100 = (negative_mf == 0) & (positive_mf > 0)
                        mfi[mask_mfi_100] = 100

                        # Special case: negative_mf == 0 and positive_mf == 0 --> MFI = 50 or NaN
                        # Set to 50 where both rolled sums are 0
                        mask_mfi_50 = (negative_mf == 0) & (positive_mf == 0)
                        mfi[mask_mfi_50] = 50

                        result['mfi'] = mfi

                else:
                     # Replace print with self.logger.warning if logger is available
                     print(f"Warning: Invalid period ({period}) for MFI calculation.")
            else:
                # Replace print with self.logger.warning if logger is available
                print(f"Warning: Cannot calculate MFI: Missing required columns {required_mfi}.")


            # Calculate Williams %R - Example direct calculation
            # Validate required columns
            required_wpr = ['high', 'low', 'close']
            if all(col in result.columns for col in required_wpr):
                periods = [14, 28]
                for period in periods:
                    if period is not None and isinstance(period, int) and period > 0:
                        highest_high = result['high'].rolling(window=period).max()
                        lowest_low = result['low'].rolling(window=period).min()

                        # Handle division by zero
                        range_val = highest_high - lowest_low

                        # Initialize with NaN
                        williams_r = pd.Series(np.nan, index=result.index)

                        # Calculate where range is not zero and not NaN
                        mask = (range_val != 0) & (range_val.notna())
                        williams_r[mask] = -100 * (highest_high[mask] - result['close'][mask]) / range_val[mask]

                         # If range is 0 or NaN, it remains NaN due to initialization/mask

                        result[f'williams_r_{period}'] = williams_r
                    else:
                         # Replace print with self.logger.warning if logger is available
                         print(f"Warning: Invalid period ({period}) for Williams %R.")
            else:
                 # Replace print with self.logger.warning if logger is available
                 print(f"Warning: Cannot calculate Williams %R: Missing required columns {required_wpr}.")


            # Calculate Ichimoku Cloud elements - Example direct calculation
            # Validate required columns
            required_ichimoku = ['high', 'low', 'close']
            if all(col in result.columns for col in required_ichimoku):
                tenkan_period = 9
                kijun_period = 26
                senkou_b_period = 52
                chikou_period = 26 # Lagging span period (shifts back)

                # Ensure periods are valid
                if (tenkan_period is not None and isinstance(tenkan_period, int) and tenkan_period > 0 and
                    kijun_period is not None and isinstance(kijun_period, int) and kijun_period > 0 and
                    senkou_b_period is not None and isinstance(senkou_b_period, int) and senkou_b_period > 0 and
                    chikou_period is not None and isinstance(chikou_period, int) and chikou_period > 0):

                    # Tenkan-sen (Conversion Line)
                    result['ichimoku_tenkan_sen'] = (result['high'].rolling(window=tenkan_period).max() +
                                                     result['low'].rolling(window=tenkan_period).min()) / 2

                    # Kijun-sen (Base Line)
                    result['ichimoku_kijun_sen'] = (result['high'].rolling(window=kijun_period).max() +
                                                    result['low'].rolling(window=kijun_period).min()) / 2

                    # Senkou Span A (Leading Span A) - Shift forward kijun_period
                    # Handle potential division by zero if both tenkan and kijun sums are 0
                    span_a_base = (result['ichimoku_tenkan_sen'] + result['ichimoku_kijun_sen']) / 2
                    result['ichimoku_senkou_span_a'] = span_a_base.shift(kijun_period)

                    # Senkou Span B (Leading Span B) - Shift forward senkou_b_period (standard)
                    result['ichimoku_senkou_span_b'] = ((result['high'].rolling(window=senkou_b_period).max() +
                                                         result['low'].rolling(window=senkou_b_period).min()) / 2).shift(senkou_b_period)

                    # Chikou Span (Lagging Span) - Shift back chikou_period
                    result['ichimoku_chikou_span'] = result['close'].shift(-chikou_period) # Negative shift goes backwards in index
                else:
                     # Replace print with self.logger.warning if logger is available
                     print(f"Warning: Invalid period(s) for Ichimoku calculation.")
            else:
                 # Replace print with self.logger.warning if logger is available
                 print(f"Warning: Cannot calculate Ichimoku: Missing required columns {required_ichimoku}.")


            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise