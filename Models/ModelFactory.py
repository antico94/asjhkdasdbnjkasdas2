from typing import Dict, Any, Optional
import os
import logging

from Models.DirectionClassificationModel import DirectionClassificationModel
from Models.LTSMModel import LSTMModel
from Models.ModelBase import ModelBase
from Models.PricePredictionModel import PricePredictionModel


class ModelFactory:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, feature_service, path_resolver):
        self.config = config
        self.logger = logger
        self.feature_service = feature_service
        self.path_resolver = path_resolver
        self.model_base_dir = "TrainedModels"

        # Use get() method to check if TrainingSettings exists
        training_settings = self.config.get('TrainingSettings')
        if training_settings is None:
            # Log a warning but don't try to modify the config object
            self.logger.warning("TrainingSettings not found in config, using defaults")
            self.training_config = {
                'DefaultPair': 'XAUUSD',
                'DefaultTimeframe': 'H1',
                'ModelSelection': 'Both',
                'use_feature_selection': True
            }
        else:
            self.training_config = training_settings

    def create_model(self, model_type: str, pair: str, timeframe: str, target: str) -> ModelBase:
        try:
            self.logger.info(f"Creating model of type {model_type} for {pair} {timeframe} with target {target}")

            # Determine if this is a classification or regression task
            is_classification = "direction" in target or "signal" in target
            model_subtype = "classification" if is_classification else "regression"

            if model_type.lower() == "randomforest":
                name = f"{pair}_{timeframe}_{target}_randomforest"
                if is_classification:
                    return DirectionClassificationModel(self.config, self.logger, name)
                else:
                    return PricePredictionModel(self.config, self.logger, name)

            elif model_type.lower() == "gradientboosting":
                name = f"{pair}_{timeframe}_{target}_gradientboosting"
                if is_classification:
                    return DirectionClassificationModel(self.config, self.logger, name)
                else:
                    return PricePredictionModel(self.config, self.logger, name)

            elif model_type.lower() == "lstm":
                name = f"{pair}_{timeframe}_{target}_lstm"
                model = LSTMModel(self.config, self.logger, name)
                model.is_classification = is_classification
                return model

            else:
                self.logger.error(f"Unknown model type: {model_type}")
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            raise

    def get_model_path(self, model: ModelBase) -> str:
        # Create path based on model name
        return self.path_resolver.resolve_path(os.path.join(self.model_base_dir, f"{model.name}.h5"))

    def save_model(self, model: ModelBase) -> bool:
        path = self.get_model_path(model)
        return model.save(path)

    def load_model(self, model_type: str, pair: str, timeframe: str, target: str) -> Optional[ModelBase]:
        try:
            # Create a new model instance
            model = self.create_model(model_type, pair, timeframe, target)

            # Get path and load model
            path = self.get_model_path(model)
            if os.path.exists(path):
                if model.load(path):
                    self.logger.info(f"Successfully loaded model {model.name}")
                    return model

            self.logger.warning(f"No saved model found at {path}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None