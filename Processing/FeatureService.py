from typing import List, Dict, Optional, Tuple
import pandas as pd
from sqlalchemy import text

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.DataStorage import DataStorage


class FeatureService:
    def __init__(self, config: Config, logger: Logger, data_storage: DataStorage):
        self.config = config
        self.logger = logger
        self.data_storage = data_storage
        self.feature_table = "SelectedFeatures"
        self.engine = self.data_storage._create_engine()

    def get_selected_features(
            self, pair: str, timeframe: str, model_type: str = "direction"
    ) -> List[str]:
        try:
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
                features = [row[0] for row in result]

            if not features:
                self.logger.warning(
                    f"No selected features found for {pair} {timeframe} {model_type}"
                )
                return []

            self.logger.info(
                f"Retrieved {len(features)} selected features for {pair} {timeframe} {model_type}"
            )
            return features

        except Exception as e:
            self.logger.error(f"Failed to retrieve selected features: {e}")
            return []

    def filter_features(
            self, df: pd.DataFrame, pair: str, timeframe: str, model_type: str = "direction"
    ) -> pd.DataFrame:
        selected_features = self.get_selected_features(pair, timeframe, model_type)

        if not selected_features:
            self.logger.warning(
                f"No selected features found for {pair} {timeframe} {model_type}. Using all features."
            )
            return df

        # Make sure to keep time column if it exists
        if "time" in df.columns and "time" not in selected_features:
            selected_features = ["time"] + selected_features

        # Filter dataframe to only include selected features
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = set(selected_features) - set(available_features)

        if missing_features:
            self.logger.warning(
                f"Some selected features are not in the dataframe: {missing_features}"
            )

        if not available_features:
            self.logger.error(
                f"None of the selected features are in the dataframe. Using all features."
            )
            return df

        return df[available_features]

    def get_feature_importance(
            self, pair: str, timeframe: str, model_type: str = "direction", top_n: int = 10
    ) -> Dict[str, float]:
        try:
            query = text(
                f"""
                SELECT feature_name, importance 
                FROM {self.feature_table}
                WHERE symbol = :symbol 
                AND timeframe = :timeframe 
                AND model_type = :model_type
                ORDER BY importance DESC
                LIMIT :top_n
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(
                    query,
                    {"symbol": pair, "timeframe": timeframe, "model_type": model_type, "top_n": top_n}
                )
                importance_dict = {row[0]: row[1] for row in result}

            return importance_dict

        except Exception as e:
            self.logger.error(f"Failed to retrieve feature importance: {e}")
            return {}

    def check_feature_analysis_exists(
            self, pair: str, timeframe: str, model_type: str = "direction"
    ) -> bool:
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
                count = result.scalar()

            return count > 0

        except Exception as e:
            self.logger.error(f"Failed to check if feature analysis exists: {e}")
            return False

    def prepare_features_for_model(
            self,
            pair: str,
            timeframe: str,
            dataset_type: str,
            model_type: str = "direction"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            # Load processed data
            X, y = self.data_storage.load_processed_data(pair, timeframe, dataset_type)

            if X.empty or y.empty:
                self.logger.error(f"No data found for {pair} {timeframe} {dataset_type}")
                return pd.DataFrame(), pd.DataFrame()

            # Apply feature selection if analysis exists
            if self.check_feature_analysis_exists(pair, timeframe, model_type):
                X = self.filter_features(X, pair, timeframe, model_type)
                self.logger.info(
                    f"Applied feature selection for {pair} {timeframe} {model_type}. Using {len(X.columns)} features."
                )
            else:
                self.logger.warning(
                    f"No feature analysis found for {pair} {timeframe} {model_type}. Using all features."
                )

            return X, y

        except Exception as e:
            self.logger.error(f"Failed to prepare features for model: {e}")
            return pd.DataFrame(), pd.DataFrame()