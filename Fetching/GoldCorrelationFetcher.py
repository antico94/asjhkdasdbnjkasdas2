from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, Table, Column, DateTime, Float, Integer, MetaData, text
from datetime import datetime

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity


class GoldCorrelationFetcher:
    """Handles fetching and storage of gold-specific correlation data."""

    def __init__(self, config: Config, logger: Logger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.db_config = config.get('Database', {})
        self.engine = self._create_engine()
        self.metadata = MetaData()

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

    def save_correlation_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """Save correlation data to a dedicated table."""
        context = {
            **self.error_context,
            "operation": "save_correlation_data",
            "symbol": symbol,
            "timeframe": timeframe,
            "data_shape": str(data.shape) if data is not None else "None"
        }

        try:
            if data.empty:
                self.error_handler.handle_error(
                    ValueError(f"Empty dataframe for {symbol}"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=True
                )

            # Create table name for correlation data
            table_name = f"corr_{symbol}_{timeframe.lower()}"
            context["table_name"] = table_name

            self.logger.info(f"Saving correlation data to {table_name}")

            # Remove existing table if it exists
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()

            # Ensure datetime column is properly formatted
            if 'time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['time']):
                data['time'] = pd.to_datetime(data['time'])

            # Create table schema based on DataFrame columns
            self._create_price_data_table(table_name)

            # Write data to database
            data.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                chunksize=1000  # Process in chunks for better performance
            )

            self.logger.info(f"Successfully saved {len(data)} rows to {table_name}")
            return True

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=False
            )
            return False

    def _create_price_data_table(self, table_name: str) -> None:
        """Create a table for price data with OHLCV columns."""
        context = {
            **self.error_context,
            "operation": "_create_price_data_table",
            "table_name": table_name
        }

        try:
            price_data_table = Table(
                table_name,
                self.metadata,
                Column("time", DateTime, primary_key=True),
                Column("open", Float, nullable=False),
                Column("high", Float, nullable=False),
                Column("low", Float, nullable=False),
                Column("close", Float, nullable=False),
                Column("tick_volume", Integer, nullable=False),
                Column("spread", Integer, nullable=False),
                Column("real_volume", Integer, nullable=True)
            )

            # Create the table
            self.metadata.create_all(self.engine, tables=[price_data_table])

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def load_correlation_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load correlation data from database."""
        context = {
            **self.error_context,
            "operation": "load_correlation_data",
            "symbol": symbol,
            "timeframe": timeframe
        }

        try:
            # Create table name for correlation data
            table_name = f"corr_{symbol}_{timeframe.lower()}"
            context["table_name"] = table_name

            self.logger.info(f"Loading correlation data from {table_name}")

            # Query to get all data
            query = f"SELECT * FROM {table_name} ORDER BY time"
            df = pd.read_sql(query, self.engine)

            if df.empty:
                self.logger.warning(f"No correlation data found in {table_name}")
                return pd.DataFrame()

            # Ensure datetime column is properly formatted
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])

            self.logger.info(f"Loaded {len(df)} rows from {table_name}")
            return df

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.MEDIUM,
                reraise=False
            )
            return pd.DataFrame()