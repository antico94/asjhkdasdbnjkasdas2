from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity


class ExternalVIXFetcher:
    """Fetches VIX data from Yahoo Finance when not available in MT5."""

    def __init__(self, config: Config, logger: Logger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.db_config = config.get('Database', {})
        self.engine = self._create_engine()

    @property
    def error_context(self) -> Dict[str, Any]:
        """Base context for error handling in this class."""
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

    def fetch_vix_data(self, from_date: datetime, to_date: datetime, timeframe: str = 'H1') -> pd.DataFrame:
        """Fetch VIX data from Yahoo Finance for the last 730 days (limit for hourly data)."""
        context = {
            **self.error_context,
            "operation": "fetch_vix_data",
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "timeframe": timeframe
        }

        try:
            # Adjust the start date to respect Yahoo Finance's 730-day limit
            max_days_limit = 730
            current_date = datetime.now()
            earliest_possible_date = current_date - timedelta(days=max_days_limit)

            # Use either the requested from_date or the earliest possible date, whichever is more recent
            adjusted_from_date = max(from_date, earliest_possible_date)

            self.logger.info(f"Original date range: {from_date} to {to_date}")
            self.logger.info(f"Adjusted date range for Yahoo Finance limit: {adjusted_from_date} to {to_date}")
            print(f"Fetching VIX data from Yahoo Finance (last {max_days_limit} days)...")
            print(f"Date range: {adjusted_from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")

            # Import yfinance here to avoid making it a global dependency
            try:
                import yfinance as yf
            except ImportError:
                self.error_handler.handle_error(
                    ImportError("yfinance package is required to fetch VIX data. Install with 'pip install yfinance'"),
                    context,
                    ErrorSeverity.HIGH,
                    reraise=False
                )
                print("Error: yfinance package not found. Installing...")
                import sys
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
                import yfinance as yf
                print("yfinance package installed successfully.")

            # Add a small buffer to the start date to ensure we have enough data for calculations
            fetch_start = adjusted_from_date - timedelta(days=1)

            # Convert timeframe to yfinance interval format
            if timeframe.upper() == 'H1':
                interval = '1h'
            elif timeframe.upper() == 'H4':
                interval = '4h'
            elif timeframe.upper() == 'D1':
                interval = '1d'
            else:
                # Default to daily if timeframe is not recognized
                self.logger.warning(f"Unrecognized timeframe {timeframe}, defaulting to daily data")
                interval = '1d'

            self.logger.info(f"Using yfinance interval: {interval}")

            # Fetch VIX data (^VIX is the Yahoo Finance ticker for VIX)
            vix_data = yf.download('^VIX', start=fetch_start, end=to_date, interval=interval)

            if vix_data.empty:
                self.error_handler.handle_error(
                    ValueError(f"No VIX data received from Yahoo Finance for the specified period"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=False
                )
                print("No VIX data available from Yahoo Finance.")
                return pd.DataFrame()

            # Reset index to make 'Date' a column
            vix_data = vix_data.reset_index()

            # Rename columns to match MT5 format
            vix_data = vix_data.rename(columns={
                'Date': 'time',
                'Datetime': 'time',  # In case hourly data uses 'Datetime'
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'tick_volume'
            })

            # Ensure datetime is handled consistently
            if 'time' in vix_data.columns:
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(vix_data['time']):
                    vix_data['time'] = pd.to_datetime(vix_data['time'])

            # Add feature flag to indicate this is actual hourly VIX data
            vix_data['has_real_vix'] = 1

            # Filter to our target date range - use adjusted_from_date to match what we requested
            vix_data = vix_data[(vix_data['time'] >= adjusted_from_date) & (vix_data['time'] <= to_date)]

            # Add missing columns to match MT5 format
            vix_data['spread'] = 0  # Default value
            if 'tick_volume' not in vix_data.columns or vix_data['tick_volume'].isnull().all():
                vix_data['tick_volume'] = 0  # Default if volume data is unavailable

            # Convert volume to integer type
            vix_data['tick_volume'] = vix_data['tick_volume'].fillna(0).astype('int64')
            vix_data['spread'] = vix_data['spread'].fillna(0).astype('int64')

            # Set real_volume to same as tick_volume for consistency
            vix_data['real_volume'] = vix_data['tick_volume']

            # Log data statistics
            self.logger.info(
                f"Downloaded {len(vix_data)} rows of VIX data from {vix_data['time'].min()} to {vix_data['time'].max()}")

            # Check hour distribution for debugging
            hour_counts = vix_data['time'].dt.hour.value_counts().sort_index()
            self.logger.info(f"VIX hour distribution: {hour_counts.to_dict()}")

            # Log original vs. adjusted date ranges
            coverage_days = (vix_data['time'].max() - vix_data['time'].min()).days
            total_requested_days = (to_date - from_date).days
            coverage_percentage = (coverage_days / total_requested_days) * 100 if total_requested_days > 0 else 0

            self.logger.info(
                f"Requested {total_requested_days} days, got {coverage_days} days ({coverage_percentage:.1f}%)")
            print(f"Retrieved VIX data for {coverage_days} days ({coverage_percentage:.1f}% of requested period)")

            return vix_data

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=False
            )
            print(f"Error fetching VIX data: {str(e)}")
            return pd.DataFrame()

    def save_vix_data(self, vix_df: pd.DataFrame, timeframe: str = 'H1') -> bool:
        """Save VIX data to the correlation data table."""
        context = {
            **self.error_context,
            "operation": "save_vix_data",
            "timeframe": timeframe,
            "data_shape": str(vix_df.shape) if vix_df is not None else "None"
        }

        try:
            if vix_df.empty:
                self.error_handler.handle_error(
                    ValueError("Empty dataframe for VIX"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=False
                )
                return False

            # Create table name for VIX data
            table_name = f"corr_VIX_{timeframe.lower()}"
            context["table_name"] = table_name

            self.logger.info(f"Saving VIX data to {table_name}")

            # Drop existing table if it exists
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()

            # Define table schema and create it
            with self.engine.connect() as conn:
                # Create the VIX table with appropriate columns, including the has_real_vix flag
                conn.execute(text(f"""
                CREATE TABLE {table_name} (
                    time DATETIME PRIMARY KEY,
                    open FLOAT NOT NULL,
                    high FLOAT NOT NULL,
                    low FLOAT NOT NULL,
                    close FLOAT NOT NULL,
                    tick_volume INT NOT NULL,
                    spread INT NOT NULL,
                    real_volume INT NULL,
                    has_real_vix INT NOT NULL
                )
                """))
                conn.commit()

            # Write data to database
            vix_df.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                chunksize=1000  # Process in chunks for better performance
            )

            self.logger.info(f"Successfully saved {len(vix_df)} rows to {table_name}")
            return True

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=False
            )
            return False