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