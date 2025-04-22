import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

from Models.ModelBase import ModelBase


class PricePredictionModel(ModelBase):
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, name: str = "price_model"):
        super().__init__(config, logger, name)
        self.model_config = config.get("MachineLearning", {}).get("Models", {}).get("GradientBoosting", {})

    def build(self, input_shape: Tuple) -> None:
        try:
            # Get hyperparameters from config
            n_hidden = self.model_config.get("hidden_units", [128, 64, 32])
            dropout_rate = self.model_config.get("dropout", 0.3)
            learning_rate = self.model_config.get("learning_rate", 0.001)

            # Build model
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(input_shape=input_shape))

            # Add hidden layers
            for units in n_hidden:
                model.add(keras.layers.Dense(units, activation="relu"))
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Dropout(dropout_rate))

            # Output layer (single value regression)
            model.add(keras.layers.Dense(1))

            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")]
            )

            self.model = model
            self.logger.info(f"Built price prediction model with {sum(n_hidden)} hidden units")
        except Exception as e:
            self.logger.error(f"Failed to build model {self.name}: {e}")
            raise

    def train(self, X: pd.DataFrame, y: pd.DataFrame, validation_data: Optional[Tuple] = None) -> Dict[str, List[float]]:
        if self.model is None:
            self.logger.error(f"Cannot train {self.name}: model not built")
            return {}

        try:
            # Store feature columns
            self.feature_columns = list(X.columns)

            # Extract target column
            target_col = y.columns[0]
            y_train = y[target_col].values

            # Training parameters
            batch_size = self.model_config.get("batch_size", 32)
            epochs = self.model_config.get("epochs", 100)
            patience = self.model_config.get("early_stopping_patience", 15)

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss" if validation_data else "loss",
                    patience=patience,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss" if validation_data else "loss",
                    factor=0.5,
                    patience=patience // 2,
                    min_lr=1e-6
                )
            ]

            # Train model
            history = self.model.fit(
                X, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
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

            # Generate predictions
            predictions = self.model.predict(X_pred)
            self.logger.info(f"Generated predictions with {self.name} for {len(X_pred)} samples")
            return predictions
        except Exception as e:
            self.logger.error(f"Failed to generate predictions with {self.name}: {e}")
            return np.array([])