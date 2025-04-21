import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
import os
import json


class DataPreprocessor:
    """Prepares data for machine learning models, focusing on LSTM requirements."""

    def __init__(self, config, logger, data_storage, path_resolver=None):
        self.config = config
        self.logger = logger
        self.data_storage = data_storage
        self.path_resolver = path_resolver
        self.feature_importance = None
        self.selected_features = None
        self.scalers = {}

    def load_feature_importance(self, file_path: str = None) -> bool:
        """Load feature importance information from analysis results."""
        try:
            if file_path is None:
                relative_path = "Analisys/FeatureAnalysis/feature_analysis_report.txt"
                if self.path_resolver:
                    file_path = self.path_resolver.resolve_path(relative_path)
                else:
                    file_path = relative_path

            if not os.path.exists(file_path):
                self.logger.warning(f"Feature importance file not found: {file_path}")
                return False

            # Parse the feature analysis report to extract importance values
            importance_dict = {}
            with open(file_path, 'r') as f:
                content = f.read()

                # Find the section with feature importance
                importance_section = content.split("TOP 20 FEATURES BY IMPORTANCE:")[1].split("SELECTED FEATURES")[0]

                # Parse each line to get feature name and importance value
                for line in importance_section.strip().split("\n")[1:]:  # Skip the header line
                    if not line or "----" in line:
                        continue
                    parts = line.split(": ")
                    if len(parts) == 2:
                        feature = parts[0].strip()
                        importance = float(parts[1].strip())
                        importance_dict[feature] = importance

            self.feature_importance = importance_dict

            # Extract selected features after redundancy removal
            selected_section = content.split("SELECTED FEATURES AFTER REDUNDANCY REMOVAL:")[1].split("HIGHLY CORRELATED")[0]
            selected_features = []
            for line in selected_section.strip().split("\n")[1:]:
                if line and "----" not in line:
                    selected_features.append(line.strip())

            self.selected_features = selected_features

            self.logger.info(f"Loaded feature importance for {len(importance_dict)} features")
            self.logger.info(f"Found {len(selected_features)} selected features after redundancy removal")
            return True

        except Exception as e:
            self.logger.error(f"Error loading feature importance: {e}")
            return False

    def get_selected_features(self) -> List[str]:
        """Get list of selected features based on feature analysis."""
        if not self.selected_features:
            # If no feature analysis results loaded, use a default set of known important features
            self.logger.warning("No feature analysis results loaded, using default important features")
            return ["atr", "rsi", "macd_histogram", "stoch_k", "candle_wick_upper",
                    "candle_wick_lower", "close_pct_change_3", "close_pct_change_5"]

        return self.selected_features

    def scale_features(self, X: pd.DataFrame, feature_importance: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Scale features using StandardScaler and optionally weight by importance."""
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
        """Create sequences for LSTM model from dataframe.

        Args:
            data: DataFrame containing features
            sequence_length: Number of time steps in each sequence

        Returns:
            Tuple of arrays (X_sequences, times) where:
            - X_sequences has shape (n_samples, sequence_length, n_features)
            - times has the corresponding timestamp for each sequence
        """
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
        """Prepare multiple target variables for multi-output model.

        Args:
            y_data: DataFrame containing target columns

        Returns:
            Dict with keys for different target types and corresponding arrays
        """
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
        """Split data for training, keeping temporal order.

        Args:
            X: Feature sequences
            y: Dictionary of target variables
            train_ratio: Percentage for training set
            val_ratio: Percentage for validation set

        Returns:
            Dictionary with train/val/test splits for features and targets
        """
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

    def prepare_dataset(self, pair: str = "XAUUSD", timeframe: str = "H1",
                        dataset_type: str = "training", sequence_length: int = 24) -> Dict[str, Any]:
        """Prepare complete dataset for model training.

        Args:
            pair: Currency pair to process
            timeframe: Timeframe to use
            dataset_type: Dataset type (training, validation, testing)
            sequence_length: Length of sequences for LSTM

        Returns:
            Dictionary with complete prepared dataset
        """
        try:
            # Load processed data from storage
            X, y = self.data_storage.load_processed_data(pair, timeframe, dataset_type)

            if X.empty or y.empty:
                self.logger.error(f"No data found for {pair} {timeframe} {dataset_type}")
                return {}

            # Get selected features if feature importance was loaded
            if self.feature_importance is None:
                self.load_feature_importance()

            selected_features = self.get_selected_features()

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

            self.logger.info(f"Prepared complete dataset for {pair} {timeframe}, shape: {X_sequences.shape}")
            return dataset

        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise
