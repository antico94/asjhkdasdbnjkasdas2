import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

from Models.ModelBase import ModelBase


class LSTMModel(ModelBase):
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, name: str = "lstm_model"):
        super().__init__(config, logger, name)
        self.model_config = config.get("MachineLearning", {}).get("Models", {}).get("LSTM", {})
        self.sequence_length = self.model_config.get("sequence_length", 10)
        self.is_classification = False

    def build(self, input_shape: Tuple) -> None:
        try:
            # Get hyperparameters from config
            lstm_units = self.model_config.get("units", [64, 32])
            dropout_rate = self.model_config.get("dropout", 0.2)
            recurrent_dropout = self.model_config.get("recurrent_dropout", 0.2)
            learning_rate = self.model_config.get("learning_rate", 0.001)

            # Build model
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(input_shape=input_shape))

            # Add LSTM layers
            return_sequences = len(lstm_units) > 1
            model.add(keras.layers.LSTM(
                lstm_units[0],
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout
            ))

            # Add any additional LSTM layers
            for i, units in enumerate(lstm_units[1:]):
                return_sequences = i < len(lstm_units) - 2
                model.add(keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout
                ))

            # Add Dense layers
            model.add(keras.layers.Dense(32, activation="relu"))
            model.add(keras.layers.Dropout(dropout_rate))

            # Output layer
            if self.is_classification:
                model.add(keras.layers.Dense(1, activation="sigmoid"))
                loss = "binary_crossentropy"
                metrics = ["accuracy", keras.metrics.AUC(name="auc")]
            else:
                model.add(keras.layers.Dense(1))
                loss = "mse"
                metrics = ["mae", keras.metrics.RootMeanSquaredError(name="rmse")]

            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )

            self.model = model
            self.logger.info(f"Built LSTM model with layers: {lstm_units}")
        except Exception as e:
            self.logger.error(f"Failed to build model {self.name}: {e}")
            raise

    def prepare_sequences(self, X: pd.DataFrame) -> np.ndarray:
        # Convert DataFrame to numpy for sequence creation
        data = X.values
        n_samples = len(data) - self.sequence_length + 1

        # Create sequences
        X_seq = np.zeros((n_samples, self.sequence_length, data.shape[1]))
        for i in range(n_samples):
            X_seq[i] = data[i:i + self.sequence_length]

        return X_seq

    def train(self, X: pd.DataFrame, y: pd.DataFrame, validation_data: Optional[Tuple] = None) -> Dict[
        str, List[float]]:
        if self.model is None:
            self.logger.error(f"Cannot train {self.name}: model not built")
            return {}

        try:
            # Store feature columns
            self.feature_columns = list(X.columns)

            # Extract target column
            target_col = y.columns[0]

            # Identify task type from target
            if "direction" in target_col or "signal" in target_col:
                self.is_classification = True

            # Prepare sequences
            X_seq = self.prepare_sequences(X)

            # Adjust target to match sequence length
            y_train = y[target_col].values[self.sequence_length - 1:]

            # Prepare validation data if provided
            val_data = None
            if validation_data:
                X_val, y_val = validation_data
                X_val_seq = self.prepare_sequences(X_val)
                y_val_seq = y_val[target_col].values[self.sequence_length - 1:]
                val_data = (X_val_seq, y_val_seq)

            # Training parameters
            batch_size = self.model_config.get("batch_size", 32)
            epochs = self.model_config.get("epochs", 100)
            patience = self.model_config.get("early_stopping_patience", 15)

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss" if val_data else "loss",
                    patience=patience,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss" if val_data else "loss",
                    factor=0.5,
                    patience=patience // 2,
                    min_lr=1e-6
                )
            ]

            # Train model
            history = self.model.fit(
                X_seq, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )

            self.history = history.history
            self.logger.info(f"Trained {self.name} for {len(history.history['loss'])} epochs")
            return self.history
        except Exception as e:
            self.logger.error(f"Failed to train model {self.name}: {e}")
            return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self.logger.error(f"Cannot predict with {self.name}: model not built")
            return np.array([])

        try:
            # Check if the model has the expected features
            missing_cols = [col for col in self.feature_columns if col not in X.columns]
            if missing_cols:
                self.logger.warning(f"Missing expected columns in prediction data: {missing_cols}")

            # Reorder columns to match training data
            X_pred = X[self.feature_columns].copy()

            # Prepare sequences
            X_seq = self.prepare_sequences(X_pred)

            # Generate predictions
            predictions = self.model.predict(X_seq)
            self.logger.info(f"Generated predictions with {self.name} for {len(X_seq)} sequences")
            return predictions
        except Exception as e:
            self.logger.error(f"Failed to generate predictions with {self.name}: {e}")
            return np.array([])

    def save(self, path: str) -> bool:
        try:
            if self.model is None:
                self.logger.error(f"Cannot save {self.name}: model not built")
                return False

            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save TensorFlow model with proper extension
            model_path = f"{path}.keras"
            self.model.save(model_path)

            # Save feature columns and metadata
            metadata = {
                "feature_columns": self.feature_columns,
                "metrics": self.metrics,
                "name": self.name,
                "is_classification": self.is_classification,
                "sequence_length": self.sequence_length,
                "saved_at": datetime.now().isoformat()
            }

            with open(f"{path}_metadata.json", "w") as f:
                json.dump(metadata, f)

            self.logger.info(f"Model {self.name} saved to {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model {self.name}: {e}")
            return False

    def load(self, path: str) -> bool:
        try:
            # Try different possible file extensions
            model_path = f"{path}.keras"
            if not os.path.exists(model_path):
                model_path = f"{path}.h5"
                if not os.path.exists(model_path):
                    # Try without extension (original path)
                    if not os.path.exists(path):
                        self.logger.error(f"Model path not found: {path}")
                        return False
                    model_path = path

            # Load TensorFlow model
            self.model = keras.models.load_model(model_path)

            # Load metadata
            metadata_path = f"{path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                self.feature_columns = metadata.get("feature_columns", [])
                self.metrics = metadata.get("metrics", {})
                self.name = metadata.get("name", self.name)
                self.is_classification = metadata.get("is_classification", False)
                self.sequence_length = metadata.get("sequence_length", 10)

            self.logger.info(f"Model {self.name} loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model {self.name}: {e}")
            return False