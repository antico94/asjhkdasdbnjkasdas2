import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
import os


class DataPreprocessor:
    def __init__(self, config, logger, data_storage, feature_service=None, path_resolver=None):
        self.config = config
        self.logger = logger
        self.data_storage = data_storage
        self.feature_service = feature_service
        self.path_resolver = path_resolver
        self.feature_importance = None
        self.selected_features = None
        self.scalers = {}
        self.pair = None
        self.timeframe = None

    def set_pair_timeframe(self, pair: str, timeframe: str) -> None:
        self.pair = pair
        self.timeframe = timeframe
        self.logger.info(f"Set pair to {pair} and timeframe to {timeframe}")

    def load_feature_importance(self, model_type: str = "direction") -> bool:
        try:
            if self.feature_service is None:
                self.logger.warning("No feature service provided, using default features")
                return False

            if not self.pair or not self.timeframe:
                self.logger.warning("Pair or timeframe not set, using default features")
                return False

            # Load feature importance from feature service
            importance_dict = self.feature_service.get_feature_importance(
                self.pair, self.timeframe, model_type
            )

            if not importance_dict:
                self.logger.warning(f"No feature importance found for {self.pair} {self.timeframe} {model_type}")
                return False

            self.feature_importance = importance_dict

            # Get selected features
            self.selected_features = self.feature_service.get_selected_features(
                self.pair, self.timeframe, model_type
            )

            self.logger.info(f"Loaded feature importance for {len(importance_dict)} features")
            self.logger.info(f"Found {len(self.selected_features)} selected features for {model_type}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading feature importance: {e}")
            return False

    def get_selected_features(self, model_type: str = "direction") -> List[str]:
        if not self.selected_features:
            if self.feature_service and self.pair and self.timeframe:
                features = self.feature_service.get_selected_features(
                    self.pair, self.timeframe, model_type
                )
                if not features:
                    self.logger.error(f"No features found for {self.pair} {self.timeframe} {model_type}")
                    raise ValueError(f"No features available for {self.pair} {self.timeframe} {model_type}")
                return features
            else:
                self.logger.error("Feature service not configured or pair/timeframe not set")
                raise ValueError("Feature selection prerequisites not met")

        return self.selected_features

    def scale_features(self, X: pd.DataFrame, feature_importance: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        try:
            # Create a copy to avoid modifying the original
            scaled_data = X.copy()

            # Initialize time_col variable
            time_col = None

            # Remove 'time' column if present, as it's not a feature for scaling
            if 'time' in scaled_data.columns:
                time_col = scaled_data['time'].copy()
                scaled_data = scaled_data.drop('time', axis=1)

            # Fit scalers for each feature if not already fit
            for column in scaled_data.columns:
                if column not in self.scalers:
                    self.scalers[column] = StandardScaler()
                    self.scalers[column].fit(scaled_data[[column]])

                # Scale the column
                scaled_data[column] = self.scalers[column].transform(scaled_data[[column]])

            # Apply feature importance weights if provided
            if feature_importance:
                for column in scaled_data.columns:
                    if column in feature_importance:
                        scaled_data[column] = scaled_data[column] * feature_importance[column]

            # Restore time column if it was present
            if time_col is not None:
                scaled_data['time'] = time_col

            return scaled_data

        except Exception as e:
            self.logger.error(f"Error scaling features: {e}")
            raise

    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Convert DataFrame to numpy first if it's still a DataFrame
            if isinstance(data, pd.DataFrame):
                # Extract the time column if present
                if 'time' in data.columns:
                    times = data['time'].values
                    features_data = data.drop('time', axis=1).values
                else:
                    times = np.arange(len(data))
                    features_data = data.values
            else:
                # If already a numpy array, create a simple time index
                features_data = data
                times = np.arange(len(data))

            n_samples = len(features_data) - sequence_length + 1
            n_features = features_data.shape[1]

            # Initialize the output array
            X_sequences = np.zeros((n_samples, sequence_length, n_features))
            time_points = np.zeros(n_samples, dtype=object)

            # Create sequences
            for i in range(n_samples):
                X_sequences[i] = features_data[i:i + sequence_length]
                time_points[i] = times[i + sequence_length - 1]  # Use the last time in the sequence

            self.logger.info(f"Created {n_samples} sequences with shape {X_sequences.shape}")
            return X_sequences, time_points

        except Exception as e:
            self.logger.error(f"Error creating sequences: {e}")
            raise

    def prepare_multi_target(self, y_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        try:
            targets = {}

            # Direction prediction (binary classification)
            if 'direction_1' in y_data.columns:
                # Convert -1, 0, 1 to binary 0/1 for 'up' direction
                targets['direction'] = (y_data['direction_1'] > 0).astype(int).values

            # Magnitude prediction (regression)
            if 'future_price_1' in y_data.columns:
                # Calculate magnitude as percentage change
                # Create a numpy array to store magnitudes
                magnitude = np.zeros(len(y_data))
                # Safe approach to calculate percentage changes
                for i in range(1, len(y_data)):
                    prev_price = y_data.iloc[i - 1]['future_price_1']
                    curr_price = y_data.iloc[i]['future_price_1']
                    if prev_price != 0:  # Avoid division by zero
                        magnitude[i] = (curr_price - prev_price) / prev_price * 100

                targets['magnitude'] = magnitude

            # If volatility target is desired but not directly available,
            # we can approximate it from other columns
            if 'atr' in y_data.columns and 'future_price_1' in y_data.columns:
                # Use ATR as volatility proxy, normalized by price
                # Make sure we don't divide by zero
                prices = y_data['future_price_1'].values
                atrs = y_data['atr'].values

                # Initialize volatility array
                volatility = np.zeros(len(prices))

                # Calculate volatility where prices are non-zero
                non_zero_mask = prices != 0
                volatility[non_zero_mask] = atrs[non_zero_mask] / prices[non_zero_mask] * 100

                targets['volatility'] = volatility

            self.logger.info(f"Prepared multi-target output with keys: {list(targets.keys())}")
            return targets

        except Exception as e:
            self.logger.error(f"Error preparing multi-target data: {e}")
            raise

    def split_data_temporal(self, X: np.ndarray, y: Dict[str, np.ndarray],
                            train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, Any]:
        try:
            n_samples = len(X)

            # Calculate split indices
            train_idx = int(n_samples * train_ratio)
            val_idx = train_idx + int(n_samples * val_ratio)

            # Split feature data
            X_train = X[:train_idx]
            X_val = X[train_idx:val_idx]
            X_test = X[val_idx:]

            # Split each target
            y_train = {}
            y_val = {}
            y_test = {}

            for target_name, target_data in y.items():
                y_train[target_name] = target_data[:train_idx]
                y_val[target_name] = target_data[train_idx:val_idx]
                y_test[target_name] = target_data[val_idx:]

            # Create output dictionary
            split_data = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }

            self.logger.info(f"Split data temporally: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            return split_data

        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise

    def prepare_dataset(self, pair: str = None, timeframe: str = None,
                        dataset_type: str = "training", sequence_length: int = 24,
                        model_type: str = "direction") -> Dict[str, Any]:
        try:
            # Use provided parameters or instance variables
            pair = pair or self.pair
            timeframe = timeframe or self.timeframe

            if not pair or not timeframe:
                self.logger.error("Pair and timeframe must be provided")
                return {}

            # Set pair and timeframe for future reference
            self.set_pair_timeframe(pair, timeframe)

            # Load processed data from storage
            X, y = self.data_storage.load_processed_data(pair, timeframe, dataset_type)

            if X.empty or y.empty:
                self.logger.error(f"No data found for {pair} {timeframe} {dataset_type}")
                return {}

            # Get selected features based on model type
            if self.feature_importance is None:
                self.load_feature_importance(model_type)

            selected_features = self.get_selected_features(model_type)

            # Extract time column before filtering features
            time_col = None
            if 'time' in X.columns:
                time_col = X['time']
                X = X.drop('time', axis=1)

            # Filter to use only selected features
            available_features = [f for f in selected_features if f in X.columns]
            if not available_features:
                self.logger.warning("None of the selected features found in data, using all features")
                available_features = X.columns

            X_filtered = X[available_features]

            # Add time back
            if time_col is not None:
                X_filtered['time'] = time_col

            # Scale features (with importance weighting if available)
            X_scaled = self.scale_features(X_filtered, self.feature_importance)

            # Create sequences
            X_sequences, sequence_times = self.create_sequences(X_scaled, sequence_length)

            y_aligned = y.iloc[sequence_length - 1:].reset_index(drop=True)
            y_targets = self.prepare_multi_target(y_aligned)

            # Split data temporally
            dataset = self.split_data_temporal(X_sequences, y_targets)

            # Add metadata
            dataset['feature_names'] = available_features
            dataset['times'] = sequence_times
            dataset['pair'] = pair
            dataset['timeframe'] = timeframe

            self.logger.info(f"Prepared complete dataset for {pair} {timeframe}, shape: {X_sequences.shape}")
            return dataset

        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise