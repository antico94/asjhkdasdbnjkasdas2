from typing import List, Dict, Optional, Tuple
import pandas as pd
from sqlalchemy import text, create_engine # Assuming create_engine might be needed if not already imported elsewhere
from sqlalchemy.engine import Engine # Type hinting for engine

# Assuming these imports are correct relative to your project structure
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.DataStorage import DataStorage


class FeatureService:
    def __init__(self, config: Config, logger: Logger, data_storage: DataStorage):
        self.config = config
        self.logger = logger
        self.data_storage = data_storage
        self.feature_table = "SelectedFeatures"
        # It's generally better practice to create the engine once if possible,
        # or manage its lifecycle carefully. Accessing a "private" method
        # like _create_engine might be okay depending on DataStorage's design,
        # but consider if DataStorage should expose the engine directly or
        # provide connection methods.
        self.engine: Engine = self.data_storage._create_engine() # Added type hint

    def get_selected_features(
            self, pair: str, timeframe: str, model_type: str = "direction"
    ) -> List[str]:
        """Retrieves a list of selected feature names for a given pair, timeframe, and model type."""
        try:
            # Using f-string for table name is generally okay if it's controlled internally,
            # but be cautious about SQL injection if table names ever come from external input.
            # Parameter binding is correctly used for WHERE clause values.
            query = text(
                f"""
                SELECT feature_name
                FROM {self.feature_table}
                WHERE symbol = :symbol
                AND timeframe = :timeframe
                AND model_type = :model_type
                ORDER BY importance DESC
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(
                    query, {"symbol": pair, "timeframe": timeframe, "model_type": model_type}
                )
                # Use .scalars().all() for fetching a single column directly into a list
                features = result.scalars().all()

            if not features:
                self.logger.warning(
                    f"No selected features found for {pair} {timeframe} {model_type}"
                )
                return [] # Return empty list consistently

            self.logger.info(
                f"Retrieved {len(features)} selected features for {pair} {timeframe} {model_type}"
            )
            return features

        except Exception as e:
            # Log the full exception for better debugging
            self.logger.error(f"Failed to retrieve selected features for {pair} {timeframe} {model_type}: {e}", exc_info=True)
            return [] # Return empty list on error

    def filter_features(
            self, df: pd.DataFrame, pair: str, timeframe: str, model_type: str = "direction"
    ) -> pd.DataFrame:
        """Filters a DataFrame to include only selected features for a given context."""
        selected_features = self.get_selected_features(pair, timeframe, model_type)

        if not selected_features:
            self.logger.warning(
                f"No selected features found for {pair} {timeframe} {model_type}. Using all available features in the DataFrame."
            )
            # Return the original DataFrame if no specific features are selected
            return df

        # Ensure 'time' column is included if present in the original DataFrame,
        # regardless of whether it was explicitly selected.
        cols_to_keep = []
        if "time" in df.columns:
            cols_to_keep.append("time")

        # Identify which selected features are actually present in the DataFrame
        available_selected_features = [f for f in selected_features if f in df.columns and f not in cols_to_keep]
        cols_to_keep.extend(available_selected_features)

        # Log any selected features that are missing from the input DataFrame
        missing_features = set(selected_features) - set(available_selected_features)
        # Don't count 'time' as missing if it wasn't in selected_features but is in df.columns
        if "time" in df.columns and "time" not in selected_features:
             missing_features.discard("time")

        if missing_features:
            self.logger.warning(
                f"Selected features not found in the provided DataFrame for {pair} {timeframe} {model_type}: {missing_features}"
            )

        # Check if any columns remain after filtering
        if not cols_to_keep:
            self.logger.error(
                f"No selected features (or 'time' column) are present in the DataFrame for {pair} {timeframe} {model_type}. Returning original DataFrame."
            )
            return df # Or potentially raise an error depending on desired behavior

        # Check if the columns to keep are identical to the original columns
        if set(cols_to_keep) == set(df.columns):
             self.logger.info(f"All columns in the DataFrame were selected for {pair} {timeframe} {model_type}.")
             return df # Return original df if no filtering actually happened

        self.logger.info(
            f"Filtering DataFrame to {len(cols_to_keep)} features for {pair} {timeframe} {model_type}."
        )
        return df[cols_to_keep]

    def get_feature_importance(
            self, pair: str, timeframe: str, model_type: str = "direction", top_n: int = 10
    ) -> Dict[str, float]:
        """Retrieves the top N features and their importance scores."""
        try:
            # --- Refactored Query ---
            # Use TOP (:param) syntax for SQL Server instead of LIMIT :param
            query = text(
                f"""
                SELECT TOP (:top_n) feature_name, importance
                FROM {self.feature_table}
                WHERE symbol = :symbol
                AND timeframe = :timeframe
                AND model_type = :model_type
                ORDER BY importance DESC
            """
            )
            # --- End Refactored Query ---

            with self.engine.connect() as conn:
                result = conn.execute(
                    query,
                    {"symbol": pair, "timeframe": timeframe, "model_type": model_type, "top_n": top_n}
                )
                # Directly create the dictionary from the result rows
                importance_dict = {row.feature_name: row.importance for row in result}

            if not importance_dict:
                 self.logger.warning(f"No feature importance data found for {pair} {timeframe} {model_type}")
                 return {}

            self.logger.info(f"Retrieved top {len(importance_dict)} feature importances for {pair} {timeframe} {model_type}")
            return importance_dict

        except Exception as e:
            self.logger.error(f"Failed to retrieve feature importance for {pair} {timeframe} {model_type}: {e}", exc_info=True)
            return {} # Return empty dict on error

    def check_feature_analysis_exists(
            self, pair: str, timeframe: str, model_type: str = "direction"
    ) -> bool:
        """Checks if any feature analysis results exist for the given criteria."""
        try:
            query = text(
                f"""
                SELECT COUNT(*)
                FROM {self.feature_table}
                WHERE symbol = :symbol
                AND timeframe = :timeframe
                AND model_type = :model_type
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(
                    query, {"symbol": pair, "timeframe": timeframe, "model_type": model_type}
                )
                count = result.scalar_one_or_none() # Use scalar_one_or_none for safety

            # Return True if count is greater than 0, False otherwise (including None)
            return (count or 0) > 0

        except Exception as e:
            self.logger.error(f"Failed to check if feature analysis exists for {pair} {timeframe} {model_type}: {e}", exc_info=True)
            return False # Return False on error

    def prepare_features_for_model(
            self,
            pair: str,
            timeframe: str,
            dataset_type: str,
            model_type: str = "direction"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads processed data and applies feature selection if available."""
        try:
            # Load processed data
            # Consider adding error handling specifically around data loading
            X, y = self.data_storage.load_processed_data(pair, timeframe, dataset_type)

            if X is None or y is None or X.empty or y.empty:
                self.logger.error(f"No data loaded for {pair} {timeframe} {dataset_type}. Cannot prepare features.")
                # Return empty DataFrames consistently
                return pd.DataFrame(), pd.DataFrame()

            # Check if feature analysis exists before attempting to filter
            if self.check_feature_analysis_exists(pair, timeframe, model_type):
                self.logger.info(
                    f"Feature analysis found for {pair} {timeframe} {model_type}. Applying feature selection."
                )
                X_filtered = self.filter_features(X.copy(), pair, timeframe, model_type) # Use copy to avoid modifying original X if filter_features changes inplace

                # Check if filtering resulted in an empty DataFrame (e.g., no columns matched)
                if X_filtered.empty and not X.empty:
                     self.logger.error(f"Filtering removed all columns for {pair} {timeframe} {model_type}. Returning original features.")
                     # Decide strategy: return original X or empty? Returning original for now.
                     return X, y
                elif X_filtered.shape[1] < X.shape[1]:
                     self.logger.info(f"Applied feature selection. Using {len(X_filtered.columns)} features.")

                return X_filtered, y
            else:
                self.logger.warning(
                    f"No feature analysis found for {pair} {timeframe} {model_type}. Using all {len(X.columns)} available features."
                )
                return X, y # Return the originally loaded data

        except Exception as e:
            self.logger.error(f"Failed to prepare features for model ({pair} {timeframe} {dataset_type} {model_type}): {e}", exc_info=True)
            # Return empty DataFrames on unexpected errors
            return pd.DataFrame(), pd.DataFrame()