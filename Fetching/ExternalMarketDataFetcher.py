from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, DateTime, Float, MetaData
from sqlalchemy.exc import SQLAlchemyError

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity


class ExternalMarketDataFetcher:
    def __init__(self, config: Config, logger: Logger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.db_config = config.get('Database', {})
        self.engine = self._create_engine()
        self.metadata = MetaData()
        self.connected = False

    @property
    def error_context(self) -> Dict[str, Any]:
        """Base context for error handling in this class"""
        return {
            "class": self.__class__.__name__,
            "connected": self.connected
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

    def fetch_gold_silver_ratio(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch Gold/Silver ratio data from external source"""
        context = {
            **self.error_context,
            "operation": "fetch_gold_silver_ratio",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }

        try:
            self.logger.info(f"Fetching Gold/Silver ratio from {start_date} to {end_date}")

            # Implementation will depend on your data source
            # This should connect to your data provider API or database

            # Return empty dataframe if implementation is not complete
            self.error_handler.handle_error(
                NotImplementedError("Gold/Silver ratio fetching not implemented"),
                context,
                ErrorSeverity.HIGH,
                reraise=True
            )

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def fetch_usd_index(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch US Dollar Index data"""
        context = {
            **self.error_context,
            "operation": "fetch_usd_index",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }

        try:
            self.logger.info(f"Fetching USD Index from {start_date} to {end_date}")

            # Implementation will depend on your data source

            self.error_handler.handle_error(
                NotImplementedError("USD Index fetching not implemented"),
                context,
                ErrorSeverity.HIGH,
                reraise=True
            )

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def fetch_vix(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch VIX (market volatility) data"""
        context = {
            **self.error_context,
            "operation": "fetch_vix",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }

        try:
            self.logger.info(f"Fetching VIX from {start_date} to {end_date}")

            # Implementation will depend on your data source

            self.error_handler.handle_error(
                NotImplementedError("VIX fetching not implemented"),
                context,
                ErrorSeverity.HIGH,
                reraise=True
            )

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def store_external_data(self, data_type: str, df: pd.DataFrame) -> bool:
        """Store external market data in database"""
        context = {
            **self.error_context,
            "operation": "store_external_data",
            "data_type": data_type,
            "df_shape": str(df.shape) if df is not None else "None"
        }

        try:
            if df.empty:
                self.error_handler.handle_error(
                    ValueError(f"Empty dataframe for {data_type}"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=True
                )

            table_name = f"External_{data_type}"

            # Drop existing table if it exists
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()

            # Define table structure based on data type
            if data_type == "GoldSilverRatio":
                self._create_ratio_table(table_name)
            elif data_type in ["USDIndex", "VIX"]:
                self._create_index_table(table_name)
            else:
                self.error_handler.handle_error(
                    ValueError(f"Unknown data type: {data_type}"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=True
                )

            # Insert data
            df.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                chunksize=1000  # Process in chunks for better performance
            )

            self.logger.info(f"Inserted {len(df)} rows into {table_name}")
            return True

        except SQLAlchemyError as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=False
            )
            return False
        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=False
            )
            return False

    def _create_ratio_table(self, table_name: str) -> None:
        context = {
            **self.error_context,
            "operation": "_create_ratio_table",
            "table_name": table_name
        }

        try:
            # Define table using SQLAlchemy
            ratio_table = Table(
                table_name,
                self.metadata,
                Column("time", DateTime, primary_key=True),
                Column("ratio", Float, nullable=False)
            )

            # Create the table
            self.metadata.create_all(self.engine, tables=[ratio_table])

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def _create_index_table(self, table_name: str) -> None:
        context = {
            **self.error_context,
            "operation": "_create_index_table",
            "table_name": table_name
        }

        try:
            # Define table using SQLAlchemy
            index_table = Table(
                table_name,
                self.metadata,
                Column("time", DateTime, primary_key=True),
                Column("open", Float, nullable=False),
                Column("high", Float, nullable=False),
                Column("low", Float, nullable=False),
                Column("close", Float, nullable=False),
                Column("volume", Float, nullable=True)
            )

            # Create the table
            self.metadata.create_all(self.engine, tables=[index_table])

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise