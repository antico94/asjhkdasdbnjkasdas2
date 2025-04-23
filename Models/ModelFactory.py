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

        # Ensure model directory exists
        os.makedirs(self.path_resolver.resolve_path(self.model_base_dir), exist_ok=True)

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
        """Create a path based on model name, without file extension"""
        # Create a base directory path without any file extension
        # Extensions are added by the model save methods
        model_dir = self.path_resolver.resolve_path(self.model_base_dir)
        return os.path.join(model_dir, model.name)

    def save_model(self, model: ModelBase) -> bool:
        try:
            path = self.get_model_path(model)
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            success = model.save(path)
            if success:
                self.logger.info(f"Successfully saved model {model.name} to {path}")
            else:
                self.logger.error(f"Failed to save model {model.name} to {path}")
            return success
        except Exception as e:
            self.logger.error(f"Error saving model {model.name}: {e}", exc_info=True)
            return False

    def load_model(self, model_type: str, pair: str, timeframe: str, target: str) -> Optional[ModelBase]:
        try:
            # Create a new model instance
            model = self.create_model(model_type, pair, timeframe, target)

            # Get base path and log it
            base_path = self.get_model_path(model)
            self.logger.info(f"Attempting to load model from base path: {base_path}")

            # Check if the directory exists
            if not os.path.exists(os.path.dirname(base_path)):
                self.logger.error(f"Directory does not exist: {os.path.dirname(base_path)}")
                return None

            # Get all files in the directory to help debugging
            all_files = os.listdir(os.path.dirname(base_path))
            self.logger.info(f"Files in directory: {all_files}")

            # Check for different possible file variations
            model_found = False

            # Try the exact base path
            if os.path.exists(base_path):
                if model.load(base_path):
                    self.logger.info(f"Successfully loaded model {model.name} from {base_path}")
                    return model

            # Try with .keras extension
            keras_path = f"{base_path}.keras"
            if os.path.exists(keras_path):
                if model.load(base_path):  # Note: model.load will handle the extension
                    self.logger.info(f"Successfully loaded model {model.name} from {keras_path}")
                    return model

            # Try with .h5 extension
            h5_path = f"{base_path}.h5"
            if os.path.exists(h5_path):
                if model.load(base_path):  # Note: model.load will handle the extension
                    self.logger.info(f"Successfully loaded model {model.name} from {h5_path}")
                    return model

            # Search for any file that starts with the model name
            model_prefix = os.path.basename(base_path)
            matching_files = [f for f in all_files if f.startswith(model_prefix) and
                              (f.endswith('.keras') or f.endswith('.h5'))]

            if matching_files:
                self.logger.info(f"Found matching files: {matching_files}")
                # Try loading with the first matching file
                if model.load(base_path):
                    self.logger.info(f"Successfully loaded model {model.name} using matching file")
                    return model

            # Last resort - check for _weights files for DirectionClassificationModel
            if isinstance(model, DirectionClassificationModel):
                weights_paths = [
                    f"{base_path}_weights.keras",
                    f"{base_path}_weights.h5",
                    f"{base_path}_weights.weights.h5"  # This seems to be the actual format used
                ]

                for weights_path in weights_paths:
                    if os.path.exists(weights_path):
                        self.logger.info(f"Found weights file: {weights_path}")
                        # For direction models, the load method should handle the weights file format
                        if model.load(base_path):
                            self.logger.info(f"Successfully loaded model {model.name} from weights file")
                            return model

            self.logger.warning(f"No model files found matching {base_path} with common extensions")
            return None

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)
            return None