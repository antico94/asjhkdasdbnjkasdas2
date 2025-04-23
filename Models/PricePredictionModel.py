import numpy as np
import pandas as pd
from tensorflow import keras
import logging
from Models.ModelBase import ModelBase
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error


class PricePredictionModel(ModelBase):
    def __init__(self, config: dict, logger: logging.Logger, name: str = "price_model"):
        super().__init__(config, logger, name)
        # Get configs from proper location - prioritize training settings
        training_settings = config.get('TrainingSettings', {}).get('ModelConfig', {}).get('GradientBoosting', {})
        if not training_settings:
            training_settings = config.get("MachineLearning", {}).get("Models", {}).get("GradientBoosting", {})
        self.model_config = training_settings
        self.scaler = StandardScaler()
        self.target_scaler = None  # Optional scaler for target normalization
        self.feature_means = None
        self.feature_stds = None
        self.normalize_target = False  # Set to True to normalize target values

    def build(self, input_shape: tuple) -> None:
        try:
            # Get hyperparameters from config
            n_hidden = self.model_config.get("hidden_units", [128, 64, 32])
            dropout_rate = self.model_config.get("dropout", 0.3)
            learning_rate = self.model_config.get("learning_rate", 0.001)
            l2_reg = self.model_config.get("l2_regularization", 0.001)

            # Build model with functional API
            inputs = keras.layers.Input(shape=input_shape)

            # Start with normalized inputs
            x = inputs

            # Add hidden layers with residual connections
            prev_layer = None
            for i, units in enumerate(n_hidden):
                # Main dense layer
                layer_x = keras.layers.Dense(
                    units=units,
                    kernel_initializer=keras.initializers.HeNormal(),
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name=f"dense_{i}"
                )(x)
                layer_x = keras.layers.BatchNormalization(name=f"bn_{i}")(layer_x)
                layer_x = keras.layers.LeakyReLU(negative_slope=0.1, name=f"leaky_relu_{i}")(layer_x)

                # Add residual connection if possible
                if prev_layer is not None and prev_layer.shape[-1] == units:
                    layer_x = keras.layers.Add()([layer_x, prev_layer])

                # Apply dropout after activation
                layer_x = keras.layers.Dropout(dropout_rate)(layer_x)

                # Update current layer
                x = layer_x
                prev_layer = x

            # Output layer (regression - no activation)
            outputs = keras.layers.Dense(1, name="price_prediction")(x)

            # Assemble model
            model = keras.Model(inputs=inputs, outputs=outputs)

            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss="mse",  # Mean Squared Error
                metrics=[
                    keras.metrics.MeanAbsoluteError(name="mae"),
                    keras.metrics.RootMeanSquaredError(name="rmse")
                ]
            )

            self.model = model
            self.logger.info(f"Built price prediction model with {sum(n_hidden)} neurons in hidden layers")
            self.model.summary(print_fn=lambda x: self.logger.debug(x))

        except Exception as e:
            self.logger.error(f"Failed to build model {self.name}: {e}")
            raise

    def train(self, X: pd.DataFrame, y: pd.DataFrame, validation_data=None) -> dict:
        if self.model is None:
            self.logger.error(f"Cannot train {self.name}: model not built")
            return {}

        try:
            # Store feature columns
            self.feature_columns = list(X.columns)

            # Extract target column
            target_col = y.columns[0]
            y_train = y[target_col].values.reshape(-1, 1)

            # Get mean of target for percentage calculations
            self.target_mean = float(np.mean(y_train))

            # Scale features
            X_train = X.values.astype(np.float32)
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Store scaler parameters
            self.feature_means = self.scaler.mean_
            self.feature_stds = self.scaler.scale_

            # Optionally scale target (can help with numerical stability)
            if self.normalize_target:
                self.target_scaler = StandardScaler()
                y_train_scaled = self.target_scaler.fit_transform(y_train).flatten()
            else:
                y_train_scaled = y_train.flatten()

            # Prepare validation data
            val_data = None
            if validation_data:
                X_val, y_val = validation_data
                X_val = X_val.values.astype(np.float32)
                X_val_scaled = self.scaler.transform(X_val)

                # Handle y_val which could be a DataFrame or ndarray
                if isinstance(y_val, pd.DataFrame):
                    if target_col in y_val.columns:
                        y_val = y_val[target_col].values.reshape(-1, 1)
                    else:
                        # If target_col is not in y_val, assume it's already the right column
                        y_val = y_val.values.reshape(-1, 1)
                else:
                    # If y_val is already a numpy array, just reshape it
                    y_val = np.asarray(y_val).reshape(-1, 1)

                # Apply scaling if needed
                if self.normalize_target:
                    y_val_scaled = self.target_scaler.transform(y_val).flatten()
                else:
                    y_val_scaled = y_val.flatten()

                val_data = (X_val_scaled, y_val_scaled)

            # Training parameters
            batch_size = self.model_config.get("batch_size", 64)
            epochs = self.model_config.get("epochs", 100)
            patience = self.model_config.get("early_stopping_patience", 15)

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss" if val_data else "loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss" if val_data else "loss",
                    factor=0.5,
                    patience=patience // 3,
                    min_lr=1e-6,
                    verbose=1
                ),
                # Add TensorBoard callback
                keras.callbacks.TensorBoard(
                    log_dir=f'./logs/{self.name}',
                    histogram_freq=1
                )
            ]

            # Add ModelCheckpoint callback if saving is enabled
            save_checkpoints = self.model_config.get("save_checkpoints", True)
            if save_checkpoints:
                callbacks.append(
                    keras.callbacks.ModelCheckpoint(
                        filepath=f"{self.name}_best.keras",
                        save_best_only=True,
                        monitor="val_loss" if val_data else "loss",
                        verbose=1
                    )
                )

            # Train model
            history = self.model.fit(
                X_train_scaled, y_train_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )

            self.history = history.history

            # Load the best model if checkpoints were saved
            if save_checkpoints:
                try:
                    best_model = keras.models.load_model(f"{self.name}_best.keras")
                    self.model = best_model
                    self.logger.info("Loaded best model from checkpoint")
                except Exception as e:
                    self.logger.warning(f"Could not load best model, using final model: {e}")

            # Log training summary
            val_metrics = {k: v[-1] for k, v in history.history.items() if k.startswith('val_')}
            self.logger.info(f"Training completed. Final validation metrics: {val_metrics}")

            return self.history

        except Exception as e:
            self.logger.error(f"Failed to train model {self.name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self.logger.error(f"Cannot predict with {self.name}: model not built")
            return np.array([])

        try:
            # Check if the model has the expected features
            missing_cols = [col for col in self.feature_columns if col not in X.columns]
            if missing_cols:
                self.logger.warning(f"Missing expected columns in prediction data: {missing_cols}")
                raise ValueError(f"Missing columns: {missing_cols}")

            # Reorder columns to match training data
            X_pred = X[self.feature_columns].values.astype(np.float32)

            # Apply normalization
            if self.feature_means is not None and self.feature_stds is not None:
                X_pred_scaled = (X_pred - self.feature_means) / self.feature_stds
            else:
                X_pred_scaled = self.scaler.transform(X_pred)

            # Generate predictions
            predictions = self.model.predict(X_pred_scaled)

            # Inverse transform if target was normalized
            if self.normalize_target and self.target_scaler is not None:
                predictions = self.target_scaler.inverse_transform(predictions)

            self.logger.info(f"Generated predictions with {self.name} for {len(X_pred)} samples")
            return predictions

        except Exception as e:
            self.logger.error(f"Failed to generate predictions with {self.name}: {e}")
            return np.array([])

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
        """Custom evaluation method with comprehensive regression metrics"""
        if self.model is None:
            self.logger.error(f"Cannot evaluate {self.name}: model not built")
            return {}

        try:
            # Prepare X data
            X_eval = X[self.feature_columns].values.astype(np.float32)

            # Apply normalization
            if self.feature_means is not None and self.feature_stds is not None:
                X_eval_scaled = (X_eval - self.feature_means) / self.feature_stds
            else:
                X_eval_scaled = self.scaler.transform(X_eval)

            # Get target values
            target_col = y.columns[0]
            y_true = y[target_col].values

            # Scale target if needed
            if self.normalize_target and self.target_scaler is not None:
                y_eval_scaled = self.target_scaler.transform(y_true.reshape(-1, 1)).flatten()
            else:
                y_eval_scaled = y_true

            # Standard model evaluation
            metrics = self.model.evaluate(X_eval_scaled, y_eval_scaled, verbose=0)
            metrics_dict = dict(zip(self.model.metrics_names, metrics))

            # Make raw predictions for additional metrics
            y_pred_scaled = self.model.predict(X_eval_scaled).flatten()

            # Convert predictions back to original scale if needed
            if self.normalize_target and self.target_scaler is not None:
                y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            else:
                y_pred = y_pred_scaled

            # Calculate additional metrics
            # 1. R-squared (coefficient of determination)
            r2 = r2_score(y_true, y_pred)
            metrics_dict['r2_score'] = float(r2)

            # 2. Mean Absolute Percentage Error (MAPE)
            try:
                mape = mean_absolute_percentage_error(y_true, y_pred)
                metrics_dict['mape'] = float(mape)
            except:
                # MAPE can fail if true values contain zeros
                metrics_dict['mape'] = float('nan')

            # 3. Root Mean Squared Error (RMSE)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics_dict['rmse'] = rmse

            # 4. RMSE as percentage of mean
            if hasattr(self, 'target_mean') and self.target_mean and self.target_mean != 0:
                rmse_pct = (rmse / abs(self.target_mean)) * 100
                metrics_dict['rmse_pct_of_mean'] = float(rmse_pct)

            # 5. Mean Absolute Error (MAE)
            mae = float(np.mean(np.abs(y_true - y_pred)))
            metrics_dict['mae'] = mae

            # 6. MAE as percentage of mean
            if hasattr(self, 'target_mean') and self.target_mean and self.target_mean != 0:
                mae_pct = (mae / abs(self.target_mean)) * 100
                metrics_dict['mae_pct_of_mean'] = float(mae_pct)

            # 7. Direction Accuracy (for price prediction)
            # Calculate if the direction of change is correctly predicted
            actual_changes = np.diff(y_true)
            predicted_changes = np.diff(y_pred)

            # Find non-zero changes to avoid division by zero
            non_zero_idxs = actual_changes != 0
            if np.any(non_zero_idxs):
                direction_accuracy = np.mean(
                    np.sign(actual_changes[non_zero_idxs]) == np.sign(predicted_changes[non_zero_idxs])
                )
                metrics_dict['direction_accuracy'] = float(direction_accuracy)

            # 8. Maximum Error
            max_error = float(np.max(np.abs(y_true - y_pred)))
            metrics_dict['max_error'] = max_error

            # 9. 90th percentile error
            p90_error = float(np.percentile(np.abs(y_true - y_pred), 90))
            metrics_dict['p90_error'] = p90_error

            # Update model metrics
            self.metrics.update(metrics_dict)
            self.logger.info(f"Model {self.name} evaluation: {metrics_dict}")
            return metrics_dict

        except Exception as e:
            self.logger.error(f"Failed to evaluate model {self.name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return basic metrics to prevent errors
            basic_metrics = {'loss': 0.0, 'mae': 0.0, 'rmse': 0.0}
            self.metrics.update(basic_metrics)
            return basic_metrics

    def save(self, path: str) -> bool:
        """Override save method to include scaler parameters"""
        try:
            if self.model is None:
                self.logger.error(f"Cannot save {self.name}: model not built")
                return False

            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save TensorFlow model - ensure proper extension
            model_path = f"{path}.keras"
            self.model.save(model_path)

            # Save feature columns, scaler parameters and metadata
            import json
            metadata = {
                "feature_columns": self.feature_columns,
                "metrics": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                            for k, v in self.metrics.items()},
                "name": self.name,
                "normalize_target": self.normalize_target,
                "target_mean": float(self.target_mean) if hasattr(self, 'target_mean') else None,
                "saved_at": pd.Timestamp.now().isoformat()
            }

            # Save feature scaler parameters
            if self.feature_means is not None and self.feature_stds is not None:
                metadata["feature_means"] = self.feature_means.tolist()
                metadata["feature_stds"] = self.feature_stds.tolist()

            # Save target scaler parameters if used
            if self.normalize_target and self.target_scaler is not None:
                metadata["target_mean"] = self.target_scaler.mean_.tolist()
                metadata["target_std"] = self.target_scaler.scale_.tolist()

            with open(f"{path}_metadata.json", "w") as f:
                json.dump(metadata, f)

            self.logger.info(f"Model {self.name} saved to {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model {self.name}: {e}")
            return False

    def load(self, path: str) -> bool:
        """Override load method to restore scaler parameters"""
        try:
            import os
            import json

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
                self.normalize_target = metadata.get("normalize_target", False)

                if "target_mean" in metadata:
                    self.target_mean = metadata["target_mean"]

                # Load feature scaler parameters
                if "feature_means" in metadata and "feature_stds" in metadata:
                    self.feature_means = np.array(metadata["feature_means"])
                    self.feature_stds = np.array(metadata["feature_stds"])

                # Load target scaler if needed
                if self.normalize_target and "target_mean" in metadata and "target_std" in metadata:
                    self.target_scaler = StandardScaler()
                    self.target_scaler.mean_ = np.array(metadata["target_mean"])
                    self.target_scaler.scale_ = np.array(metadata["target_std"])

            self.logger.info(f"Model {self.name} loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model {self.name}: {e}")
            return False

    def load(self, path: str) -> bool:
        """Override load method to restore scaler parameters"""
        try:
            import os
            import json

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
                self.normalize_target = metadata.get("normalize_target", False)

                if "target_mean" in metadata:
                    self.target_mean = metadata["target_mean"]

                # Load feature scaler parameters
                if "feature_means" in metadata and "feature_stds" in metadata:
                    self.feature_means = np.array(metadata["feature_means"])
                    self.feature_stds = np.array(metadata["feature_stds"])

                # Load target scaler if needed
                if self.normalize_target and "target_mean" in metadata and "target_std" in metadata:
                    self.target_scaler = StandardScaler()
                    self.target_scaler.mean_ = np.array(metadata["target_mean"])
                    self.target_scaler.scale_ = np.array(metadata["target_std"])

            self.logger.info(f"Model {self.name} loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model {self.name}: {e}")
            return False