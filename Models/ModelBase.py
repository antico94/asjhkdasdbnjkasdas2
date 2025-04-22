from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
from tensorflow import keras
import logging
import json
from datetime import datetime


class ModelBase(ABC):
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, name: str = "model"):
        self.config = config
        self.logger = logger
        self.name = name
        self.model = None
        self.history = None
        self.metrics = {}
        self.feature_columns = []

    @abstractmethod
    def build(self, input_shape: Tuple) -> None:
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame, validation_data: Optional[Tuple] = None) -> Dict[
        str, List[float]]:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def save(self, path: str) -> bool:
        try:
            if self.model is None:
                self.logger.error(f"Cannot save {self.name}: model not built")
                return False

            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save TensorFlow model
            self.model.save(path)

            # Save feature columns and metadata
            metadata = {
                "feature_columns": self.feature_columns,
                "metrics": self.metrics,
                "name": self.name,
                "saved_at": datetime.now().isoformat()
            }

            with open(f"{path}_metadata.json", "w") as f:
                json.dump(metadata, f)

            self.logger.info(f"Model {self.name} saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model {self.name}: {e}")
            return False

    def load(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                self.logger.error(f"Model path not found: {path}")
                return False

            # Load TensorFlow model
            self.model = keras.models.load_model(path)

            # Load metadata
            metadata_path = f"{path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                self.feature_columns = metadata.get("feature_columns", [])
                self.metrics = metadata.get("metrics", {})
                self.name = metadata.get("name", self.name)

            self.logger.info(f"Model {self.name} loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model {self.name}: {e}")
            return False

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            self.logger.error(f"Cannot evaluate {self.name}: model not built")
            return {}

        try:
            metrics = self.model.evaluate(X, y, verbose=0)
            metrics_dict = dict(zip(self.model.metrics_names, metrics))
            self.metrics.update(metrics_dict)
            self.logger.info(f"Model {self.name} evaluation: {metrics_dict}")
            return metrics_dict
        except Exception as e:
            self.logger.error(f"Failed to evaluate model {self.name}: {e}")
            return {}