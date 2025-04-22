from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from tensorflow import keras
import logging

from Models.ModelFactory import ModelFactory
from Models.ModelBase import ModelBase

class ModelTrainer:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, model_factory: ModelFactory, data_storage):
        self.config = config
        self.logger = logger
        self.model_factory = model_factory
        self.data_storage = data_storage
        self.feature_service = model_factory.feature_service

    def prepare_data(self, pair: str, timeframe: str, target: str) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            # Load training and validation data
            X_train, y_train = self.data_storage.load_processed_data(pair, timeframe, "training")
            X_val, y_val = self.data_storage.load_processed_data(pair, timeframe, "validation")

            # Check if data was loaded successfully
            if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
                self.logger.error(f"Failed to load data for {pair} {timeframe}")
                raise ValueError(f"Failed to load data for {pair} {timeframe}")

            # Check if target exists in y_train
            if target not in y_train.columns:
                self.logger.error(f"Target {target} not found in training data")
                raise ValueError(f"Target {target} not found in training data")

            # Store original target data for validation
            self.target_name = target

            # Handle direction targets properly
            is_direction = "direction" in target or "signal" in target
            model_type = "direction" if is_direction else "magnitude"

            # Log information about the target
            self.logger.info(f"Preparing data for {model_type} model with target {target}")
            if is_direction and y_train[target].min() < 0:
                self.logger.info(
                    f"Direction target contains values: min={y_train[target].min()}, max={y_train[target].max()}")

            # Remove 'time' column from training data
            if 'time' in X_train.columns:
                time_train = X_train['time']
                time_val = X_val['time']
                X_train = X_train.drop('time', axis=1)
                X_val = X_val.drop('time', axis=1)

            # Use feature selection from feature service if available
            use_feature_selection = self.config.get('TrainingSettings', {}).get('use_feature_selection', True)

            if use_feature_selection:
                # Check if feature selection exists
                has_features = self.feature_service.check_feature_analysis_exists(pair, timeframe, model_type)

                if has_features:
                    # Get selected features
                    selected_features = self.feature_service.get_selected_features(pair, timeframe, model_type)

                    if selected_features:
                        # Filter features
                        X_train = X_train[selected_features]
                        X_val = X_val[selected_features]
                        self.logger.info(f"Filtered to {len(selected_features)} selected features for {model_type}")
                    else:
                        self.logger.warning(f"No selected features found for {pair} {timeframe} {model_type}")
                else:
                    self.logger.warning(f"No feature selection found for {pair} {timeframe} {model_type}")

            # Keep only the target column in y
            y_train = y_train[[target]]
            y_val = y_val[[target]]

            return X_train, y_train, X_val, y_val

        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise

    def train_model(self, model_type: str, pair: str, timeframe: str, target: str) -> ModelBase:
        try:
            self.logger.info(f"Training {model_type} model for {pair} {timeframe} with target {target}")

            # Prepare data
            X_train, y_train, X_val, y_val = self.prepare_data(pair, timeframe, target)

            # Create model
            model = self.model_factory.create_model(model_type, pair, timeframe, target)

            # Build model
            input_shape = (X_train.shape[1],)
            model.build(input_shape)

            # Train model
            validation_data = (X_val, y_val[target].values)
            history = model.train(X_train, y_train, validation_data)

            # Evaluate model
            metrics = model.evaluate(X_val, y_val)

            # Save model
            self.model_factory.save_model(model)

            self.logger.info(f"Successfully trained and saved {model_type} model for {pair} {timeframe} {target}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to train model: {e}")
            raise

    def train_all_models(self, pair: str, timeframe: str) -> Dict[str, ModelBase]:
        try:
            models = {}

            # Get target settings from config
            ml_settings = self.config.get('MachineLearning', {})
            targets = ml_settings.get('Targets', {})
            model_types = ml_settings.get('Models', ["RandomForest", "GradientBoosting", "LSTM"])

            # Train models for each target
            if 'PricePrediction' in targets:
                horizons = targets['PricePrediction'].get('Horizons', [1, 3, 5])

                for horizon in horizons:
                    target = f"future_price_{horizon}"
                    for model_type in model_types:
                        model_key = f"{model_type.lower()}_{target}"
                        try:
                            models[model_key] = self.train_model(model_type, pair, timeframe, target)
                        except Exception as e:
                            self.logger.error(f"Failed to train {model_type} for {target}: {e}")

            if 'DirectionPrediction' in targets:
                horizons = targets['PricePrediction'].get('Horizons', [1, 3, 5])

                for horizon in horizons:
                    target = f"direction_{horizon}"
                    for model_type in model_types:
                        model_key = f"{model_type.lower()}_{target}"
                        try:
                            models[model_key] = self.train_model(model_type, pair, timeframe, target)
                        except Exception as e:
                            self.logger.error(f"Failed to train {model_type} for {target}: {e}")

                    # Also train signal models
                    target = f"signal_up_{horizon}"
                    for model_type in model_types:
                        model_key = f"{model_type.lower()}_{target}"
                        try:
                            models[model_key] = self.train_model(model_type, pair, timeframe, target)
                        except Exception as e:
                            self.logger.error(f"Failed to train {model_type} for {target}: {e}")

            self.logger.info(f"Successfully trained {len(models)} models for {pair} {timeframe}")
            return models

        except Exception as e:
            self.logger.error(f"Failed to train all models: {e}")
            return {}