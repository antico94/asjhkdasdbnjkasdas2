from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Models.LSTMModel import LSTMModel, AttentionLayer
from Training.DataPreprocessor import DataPreprocessor
from Training.ModelTrainer import ModelTrainer
import os
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

from Utilities.PathResolver import PathResolver


class ModelFactory:
    """Factory for creating model-related components."""

    def __init__(self, config: Config, logger: Logger, path_resolver: Optional['PathResolver'] = None):
        self.config = config
        self.logger = logger
        self.path_resolver = path_resolver

    def create_data_preprocessor(self, data_storage) -> DataPreprocessor:
        """Create a data preprocessor instance."""
        path_resolver = self.path_resolver
        return DataPreprocessor(self.config, self.logger, data_storage, path_resolver)

    def create_model_trainer(self, data_preprocessor: DataPreprocessor,
                             model: Optional[LSTMModel] = None) -> ModelTrainer:
        """Create a model trainer instance."""
        return ModelTrainer(self.config, self.logger, data_preprocessor, model)

    def create_lstm_model(self, input_shape: Tuple[int, int], n_features: int) -> LSTMModel:
        """Create an LSTM model instance."""
        return LSTMModel(self.config, input_shape, n_features)

    def load_model(self, model_path: str) -> LSTMModel:
        """Load a trained model from file."""
        try:
            self.logger.info(f"Loading model from {model_path}")

            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load Keras model to get input shape, registering the custom layer
            keras_model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            input_shape = keras_model.input_shape[1:]  # Remove batch dimension
            n_features = input_shape[1]

            # Create LSTMModel with appropriate dimensions
            model = LSTMModel(self.config, input_shape, n_features)

            # Load weights (AttentionLayer is already registered)
            model.load_model(model_path)

            self.logger.info(f"Model loaded successfully with input shape {input_shape}")
            return model

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def create_ensemble_model(self, n_models: int = 5) -> Dict[str, Any]:
        """Create an ensemble of models (placeholder for now)."""
        self.logger.info(f"Creating ensemble model with {n_models} models")

        # Placeholder implementation
        ensemble = {
            'n_models': n_models,
            'models': []
        }

        return ensemble
