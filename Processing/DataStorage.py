from typing import Tuple, Dict, Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity


class DataStorage:
    def __init__(self, config: Config, logger: Logger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler
        self.db_config = config.get('Database', {})
        self.engine = self._create_engine()

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

    def save_processed_data(self, X: pd.DataFrame, y: pd.DataFrame,
                            pair: str, timeframe: str, dataset_type: str) -> bool:
        """Save processed data to database tables with indicators and features."""
        context = {
            **self.error_context,
            "operation": "save_processed_data",
            "pair": pair,
            "timeframe": timeframe,
            "dataset_type": dataset_type,
            "X_shape": str(X.shape) if X is not None else "None",
            "y_shape": str(y.shape) if y is not None else "None"
        }

        try:
            # Ensure we have data to save
            if X.empty:
                self.error_handler.handle_error(
                    ValueError(f"No data to save for {pair}_{timeframe}_{dataset_type}"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=False
                )
                return False

            # Verify time column exists and has no NaN values
            if 'time' not in X.columns:
                self.error_handler.handle_error(
                    ValueError("Time column missing from features dataframe"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=False
                )
                return False

            if X['time'].isna().any():
                self.error_handler.handle_error(
                    ValueError("Time column contains NaN values"),
                    context,
                    ErrorSeverity.MEDIUM,
                    reraise=False
                )
                return False

            # Create table name for the processed data
            table_name = f"{pair}_{timeframe}_{dataset_type}_processed"
            context["table_name"] = table_name

            self.logger.info(f"Saving processed data to {table_name}")

            # Create the processed data table
            self._create_processed_table(table_name)

            # Prepare a single DataFrame with all data
            combined_df = self._prepare_combined_dataframe(X, y)

            # Write data directly using pandas to_sql
            combined_df.to_sql(
                table_name,
                self.engine,
                if_exists='replace',
                index=False,
                chunksize=1000  # Process in chunks for better performance
            )

            self.logger.info(f"Successfully saved {len(X)} rows to {table_name}")
            return True

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=False
            )
            return False

    def _create_processed_table(self, table_name: str) -> None:
        """Create a table for processed data with features and targets."""
        context = {
            **self.error_context,
            "operation": "_create_processed_table",
            "table_name": table_name
        }

        try:
            # Drop existing table if it exists
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()
        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.MEDIUM,
                reraise=True
            )
            raise

    def _prepare_combined_dataframe(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Prepare a combined DataFrame with features and targets."""
        context = {
            **self.error_context,
            "operation": "_prepare_combined_dataframe",
            "X_shape": str(X.shape),
            "y_shape": str(y.shape) if y is not None else "None"
        }

        try:
            # Start with a copy of X
            combined_df = X.copy()

            # Rename feature columns (except time)
            for col in combined_df.columns:
                if col != 'time':
                    combined_df = combined_df.rename(columns={col: f'feature_{col}'})

            # Add target columns if y is not empty
            if not y.empty:
                for col in y.columns:
                    combined_df[f'target_{col}'] = y[col].values

            return combined_df
        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def load_processed_data(self, pair: str, timeframe: str, dataset_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed data from database."""
        context = {
            **self.error_context,
            "operation": "load_processed_data",
            "pair": pair,
            "timeframe": timeframe,
            "dataset_type": dataset_type
        }

        try:
            # Create table name
            table_name = f"{pair}_{timeframe}_{dataset_type}_processed"
            context["table_name"] = table_name

            # Query to get all data
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)

            if df.empty:
                self.logger.warning(f"No data found in {table_name}")
                return pd.DataFrame(), pd.DataFrame()

            # Separate columns into time, features, and targets
            time_col = df['time'] if 'time' in df.columns else None
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            target_cols = [col for col in df.columns if col.startswith('target_')]

            # Extract features
            X = pd.DataFrame()
            if time_col is not None:
                X['time'] = time_col

            for col in feature_cols:
                feature_name = col.replace('feature_', '')
                X[feature_name] = df[col]

            # Extract targets
            y = pd.DataFrame()
            for col in target_cols:
                target_name = col.replace('target_', '')
                y[target_name] = df[col]

            self.logger.info(f"Loaded {len(df)} rows from {table_name}")
            return X, y

        except Exception as e:
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=False
            )
            return pd.DataFrame(), pd.DataFrame()