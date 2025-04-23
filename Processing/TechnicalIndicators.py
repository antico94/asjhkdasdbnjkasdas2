from typing import List, Dict, Any, Optional
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

            if period <= 0:
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

            # First value is sum of gains or losses
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # Check for division by zero
            if (avg_loss == 0).any():
                # Handle the case where avg_loss is zero (RSI = 100)
                rs = pd.Series(index=avg_gain.index)
                for i in range(len(avg_gain)):
                    if avg_loss.iloc[i] == 0:
                        if avg_gain.iloc[i] == 0:
                            rs.iloc[i] = 0  # Both are zero, default to 0
                        else:
                            rs.iloc[i] = float('inf')  # Loss is zero, gain is non-zero
                    else:
                        rs.iloc[i] = avg_gain.iloc[i] / avg_loss.iloc[i]
            else:
                rs = avg_gain / avg_loss

            result['rsi'] = 100 - (100 / (1 + rs))
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

            if period <= 0:
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
            result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
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

            if period <= 0:
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
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result['atr'] = tr.rolling(window=period).mean()
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

            if k_period <= 0 or d_period <= 0:
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

            # Check for division by zero (when highest_high == lowest_low)
            denom = highest_high - lowest_low
            close_low_diff = result['close'] - lowest_low

            # Initialize %K with NaNs
            result['stoch_k'] = pd.Series(float('nan'), index=result.index)

            # Set %K to 50 when denominator is zero (no price movement in the period)
            # Otherwise calculate normally
            mask = denom != 0
            result.loc[mask, 'stoch_k'] = 100 * (close_low_diff[mask] / denom[mask])
            result.loc[~mask, 'stoch_k'] = 50  # Set to middle value when no movement

            # %D calculation (3-period SMA of %K)
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

            # Get high, low, close for pivot calculation (previous day)
            high = result['high'].shift(1)
            low = result['low'].shift(1)
            close = result['close'].shift(1)

            if method == 'standard':
                # Standard pivot points
                pp = (high + low + close) / 3
                s1 = (2 * pp) - high
                s2 = pp - (high - low)
                r1 = (2 * pp) - low
                r2 = pp + (high - low)

                result['pivot'] = pp
                result['support1'] = s1
                result['support2'] = s2
                result['resistance1'] = r1
                result['resistance2'] = r2

            elif method == 'fibonacci':
                # Fibonacci pivot points
                pp = (high + low + close) / 3
                r1 = pp + 0.382 * (high - low)
                r2 = pp + 0.618 * (high - low)
                r3 = pp + (high - low)
                s1 = pp - 0.382 * (high - low)
                s2 = pp - 0.618 * (high - low)
                s3 = pp - (high - low)

                result['pivot'] = pp
                result['support1'] = s1
                result['support2'] = s2
                result['support3'] = s3
                result['resistance1'] = r1
                result['resistance2'] = r2
                result['resistance3'] = r3

            elif method == 'woodie':
                # Woodie pivot points
                pp = (high + low + 2 * close) / 4
                r1 = (2 * pp) - low
                r2 = pp + (high - low)
                s1 = (2 * pp) - high
                s2 = pp - (high - low)

                result['pivot'] = pp
                result['support1'] = s1
                result['support2'] = s2
                result['resistance1'] = r1
                result['resistance2'] = r2

            elif method == 'camarilla':
                # Camarilla pivot points
                pp = (high + low + close) / 3
                range_val = high - low

                r1 = close + range_val * 1.1 / 12
                r2 = close + range_val * 1.1 / 6
                r3 = close + range_val * 1.1 / 4
                r4 = close + range_val * 1.1 / 2

                s1 = close - range_val * 1.1 / 12
                s2 = close - range_val * 1.1 / 6
                s3 = close - range_val * 1.1 / 4
                s4 = close - range_val * 1.1 / 2

                result['pivot'] = pp
                result['support1'] = s1
                result['support2'] = s2
                result['support3'] = s3
                result['support4'] = s4
                result['resistance1'] = r1
                result['resistance2'] = r2
                result['resistance3'] = r3
                result['resistance4'] = r4

            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with default parameters."""
        context = self._get_error_context(
            "calculate_all_indicators",
            {"data_shape": str(data.shape) if data is not None else "None"}
        )

        try:
            result = data.copy()

            # Calculate Moving Averages
            result = self.calculate_sma(result, periods=[5, 8, 13, 21, 50, 200])
            result = self.calculate_ema(result, periods=[5, 8, 13, 21, 50, 200])
            result = self.calculate_macd(result, fast_period=12, slow_period=26, signal_period=9)

            # Calculate Volatility indicators
            result = self.calculate_bollinger_bands(result, period=20, num_std=2.0)
            result = self.calculate_atr(result, period=14)

            # Calculate Momentum indicators
            result = self.calculate_rsi(result, period=14)
            result = self.calculate_stochastic(result, k_period=14, d_period=3)

            # Calculate Pivot Points
            result = self.calculate_pivot_points(result, method="standard")

            # Additional indicators that might be useful

            # Calculate Rate of Change (RoC)
            periods = [5, 10, 20]
            for period in periods:
                result[f'roc_{period}'] = (result['close'] / result['close'].shift(period) - 1) * 100

            # Calculate Average Directional Index (ADX)
            period = 14
            plus_dm = result['high'].diff()
            minus_dm = result['low'].diff(-1).abs()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            tr1 = (result['high'] - result['low']).abs()
            tr2 = (result['high'] - result['close'].shift()).abs()
            tr3 = (result['low'] - result['close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()

            # Handle division by zero for DI calculations
            plus_di = pd.Series(0, index=result.index)
            minus_di = pd.Series(0, index=result.index)

            mask = atr > 0
            plus_di[mask] = 100 * (plus_dm[mask].rolling(window=period).mean() / atr[mask])
            minus_di[mask] = 100 * (minus_dm[mask].rolling(window=period).mean() / atr[mask])

            # Handle division by zero for DX calculation
            dx = pd.Series(0, index=result.index)
            di_sum = plus_di.abs() + minus_di.abs()

            mask = di_sum > 0
            dx[mask] = 100 * ((plus_di[mask] - minus_di[mask]).abs() / di_sum[mask])

            result['adx'] = dx.rolling(window=period).mean()
            result['plus_di'] = plus_di
            result['minus_di'] = minus_di

            # Calculate Money Flow Index (MFI)
            period = 14

            if 'tick_volume' not in result.columns:
                if self.error_handler:
                    self.error_handler.handle_error(
                        ValueError("Missing 'tick_volume' column required for MFI calculation"),
                        context,
                        ErrorSeverity.MEDIUM,
                        reraise=False
                    )
                # Skip MFI calculation if volume data not available
            else:
                typical_price = (result['high'] + result['low'] + result['close']) / 3
                money_flow = typical_price * result['tick_volume']
                positive_flow = pd.Series(0, index=result.index)
                negative_flow = pd.Series(0, index=result.index)

                # Determine positive and negative flow
                mask_pos = typical_price > typical_price.shift(1)
                mask_neg = typical_price < typical_price.shift(1)

                positive_flow[mask_pos] = money_flow[mask_pos]
                negative_flow[mask_neg] = money_flow[mask_neg]

                positive_mf = positive_flow.rolling(window=period).sum()
                negative_mf = negative_flow.rolling(window=period).sum()

                # Handle division by zero
                mfi = pd.Series(50, index=result.index)  # Default to neutral when no negative flow

                mask = negative_mf > 0
                money_ratio = positive_mf[mask] / negative_mf[mask]
                mfi[mask] = 100 - (100 / (1 + money_ratio))

                # Special case - when positive_mf > 0 and negative_mf = 0
                mask = (negative_mf == 0) & (positive_mf > 0)
                mfi[mask] = 100

                # Special case - when both are zero
                mask = (negative_mf == 0) & (positive_mf == 0)
                mfi[mask] = 50

                result['mfi'] = mfi

            # Calculate Commodity Channel Index (CCI)
            period = 20
            tp = (result['high'] + result['low'] + result['close']) / 3
            tp_ma = tp.rolling(window=period).mean()
            tp_dev = (tp - tp_ma).abs().rolling(window=period).mean()

            # Handle division by zero
            cci = pd.Series(0, index=result.index)
            mask = tp_dev > 0
            cci[mask] = (tp[mask] - tp_ma[mask]) / (0.015 * tp_dev[mask])

            result['cci'] = cci

            # Calculate Williams %R
            periods = [14, 28]
            for period in periods:
                highest_high = result['high'].rolling(window=period).max()
                lowest_low = result['low'].rolling(window=period).min()

                # Handle division by zero
                williams_r = pd.Series(-50, index=result.index)
                range_val = highest_high - lowest_low

                mask = range_val > 0
                williams_r[mask] = -100 * (highest_high[mask] - result['close'][mask]) / range_val[mask]

                result[f'williams_r_{period}'] = williams_r

            # Calculate Ichimoku Cloud elements
            tenkan_period = 9
            kijun_period = 26
            senkou_b_period = 52

            # Tenkan-sen (Conversion Line)
            result['ichimoku_tenkan_sen'] = (result['high'].rolling(window=tenkan_period).max() +
                                             result['low'].rolling(window=tenkan_period).min()) / 2

            # Kijun-sen (Base Line)
            result['ichimoku_kijun_sen'] = (result['high'].rolling(window=kijun_period).max() +
                                            result['low'].rolling(window=kijun_period).min()) / 2

            # Senkou Span A (Leading Span A)
            result['ichimoku_senkou_span_a'] = (
                        (result['ichimoku_tenkan_sen'] + result['ichimoku_kijun_sen']) / 2).shift(
                kijun_period)

            # Senkou Span B (Leading Span B)
            result['ichimoku_senkou_span_b'] = ((result['high'].rolling(window=senkou_b_period).max() +
                                                 result['low'].rolling(window=senkou_b_period).min()) / 2).shift(
                kijun_period)

            # Chikou Span (Lagging Span)
            result['ichimoku_chikou_span'] = result['close'].shift(-kijun_period)

            return result

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.HIGH, reraise=True
                )
            raise