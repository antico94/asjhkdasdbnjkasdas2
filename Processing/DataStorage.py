from typing import Tuple

import pandas as pd
from sqlalchemy import create_engine, text

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger


class DataStorage:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.db_config = config.get('Database', {})
        self.engine = self._create_engine()

    def _create_engine(self):
        """Create SQLAlchemy engine for database connections."""
        db = self.db_config
        connection_string = (
            f"mssql+pyodbc://{db['User']}:{db['Password']}@{db['Host']},{db['Port']}/"
            f"{db['Database']}?driver=ODBC+Driver+17+for+SQL+Server"
        )
        return create_engine(connection_string)

    def save_processed_data(self, X: pd.DataFrame, y: pd.DataFrame,
                            pair: str, timeframe: str, dataset_type: str) -> bool:
        """Save processed data to database tables with indicators and features."""
        try:
            # Ensure we have data to save
            if X.empty:
                self.logger.warning(f"No data to save for {pair}_{timeframe}_{dataset_type}")
                return False

            # Verify time column exists and has no NaN values
            if 'time' not in X.columns:
                self.logger.error("Time column missing from features dataframe")
                return False

            if X['time'].isna().any():
                self.logger.error("Time column contains NaN values")
                return False

            # Create table name for the processed data
            table_name = f"{pair}_{timeframe}_{dataset_type}_processed"

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
            self.logger.error(f"Failed to save processed data: {e}")
            return False

    def _create_processed_table(self, table_name: str) -> None:
        """Create a table for processed data with features and targets."""
        # Drop existing table if it exists
        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()

    def _prepare_combined_dataframe(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Prepare a combined DataFrame with features and targets."""
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

    def load_processed_data(self, pair: str, timeframe: str, dataset_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed data from database."""
        try:
            # Create table name
            table_name = f"{pair}_{timeframe}_{dataset_type}_processed"

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
            self.logger.error(f"Failed to load processed data: {e}")
            return pd.DataFrame(), pd.DataFrame()
