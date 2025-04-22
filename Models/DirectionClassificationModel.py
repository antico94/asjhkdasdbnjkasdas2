from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from tensorflow import keras
import logging
from Models.ModelBase import ModelBase


class DirectionClassificationModel(ModelBase):
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, name: str = "direction_model"):
        super().__init__(config, logger, name)
        self.model_config = config.get("MachineLearning", {}).get("Models", {}).get("RandomForest", {})
        self.class_weights = None

    def build(self, input_shape: Tuple) -> None:
        try:
            # Get hyperparameters from config
            n_hidden = self.model_config.get("hidden_units", [64, 32])
            dropout_rate = self.model_config.get("dropout", 0.2)
            learning_rate = self.model_config.get("learning_rate", 0.001)

            # Build model
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(input_shape=input_shape))

            # Add hidden layers
            for units in n_hidden:
                model.add(keras.layers.Dense(units, activation="relu"))
                model.add(keras.layers.Dropout(dropout_rate))

            # Output layer (binary classification)
            model.add(keras.layers.Dense(1, activation="sigmoid"))

            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(name="auc")]
            )

            self.model = model
            self.logger.info(f"Built direction classification model with {sum(n_hidden)} hidden units")
        except Exception as e:
            self.logger.error(f"Failed to build model {self.name}: {e}")
            raise

    def train(self, X: pd.DataFrame, y: pd.DataFrame, validation_data: Optional[Tuple] = None) -> Dict[
        str, List[float]]:
        if self.model is None:
            self.logger.error(f"Cannot train {self.name}: model not built")
            return {}

        try:
            # Store feature columns
            self.feature_columns = list(X.columns)

            # Extract target column (should be a binary target)
            target_col = y.columns[0]
            y_train = y[target_col].values

            # Compute class weights for imbalanced data
            class_counts = np.bincount(y_train.astype(int))
            total = len(y_train)
            self.class_weights = {
                0: total / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
                1: total / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
            }

            # Training parameters
            batch_size = self.model_config.get("batch_size", 32)
            epochs = self.model_config.get("epochs", 50)
            patience = self.model_config.get("early_stopping_patience", 10)

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
                class_weight=self.class_weights,
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