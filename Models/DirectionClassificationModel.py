import json
import os

import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import logging
from sklearn.utils import class_weight
from Models.ModelBase import ModelBase
from sklearn.preprocessing import StandardScaler


class DirectionClassificationModel(ModelBase):
    def __init__(self, config: dict, logger: logging.Logger, name: str = "direction_model"):
        super().__init__(config, logger, name)
        # Get configs from proper location - prioritize training settings
        training_settings = config.get('TrainingSettings', {}).get('ModelConfig', {}).get('RandomForest', {})
        if not training_settings:
            training_settings = config.get("MachineLearning", {}).get("Models", {}).get("RandomForest", {})
        self.model_config = training_settings
        self.class_weights = None
        self.is_multiclass = True  # Default to multiclass for direction (-1, 0, 1)
        self.scaler = StandardScaler()  # Use scikit-learn scaler for more robust normalization
        self.feature_means = None  # Store feature means for prediction
        self.feature_stds = None  # Store feature stds for prediction

    def build(self, input_shape: tuple) -> None:
        try:
            # Get hyperparameters from config
            n_hidden = self.model_config.get("hidden_units", [256, 128, 64, 32])
            dropout_rate = self.model_config.get("dropout", 0.4)
            learning_rate = self.model_config.get("learning_rate", 0.001)
            l2_reg = self.model_config.get("l2_regularization", 0.0005)

            # Determine if we're doing binary or multiclass classification
            n_classes = 3 if self.is_multiclass else 1

            # Build model with functional API for more flexibility
            inputs = keras.layers.Input(shape=input_shape)

            # Create hidden layers with residual connections for better gradient flow
            x = inputs
            prev_outputs = []  # Store previous layer outputs for skip connections

            # Add more capacity and better regularization
            for i, units in enumerate(n_hidden):
                # Main path
                layer_x = keras.layers.Dense(
                    units=units,
                    kernel_initializer=keras.initializers.HeNormal(),
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name=f"dense_{i}"
                )(x)
                layer_x = keras.layers.BatchNormalization(name=f"bn_{i}")(layer_x)
                layer_x = keras.layers.LeakyReLU(alpha=0.1, name=f"leaky_relu_{i}")(layer_x)

                # Skip connections (residual-style) - connect to all previous layers of same dimension
                if prev_outputs and prev_outputs[-1].shape[-1] == units:
                    layer_x = keras.layers.Add(name=f"residual_{i}")([layer_x, prev_outputs[-1]])

                # Apply dropout after activation
                layer_x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")(layer_x)

                # Update current layer output
                x = layer_x
                prev_outputs.append(x)

                # Add a cross layer occasionally
                if i > 0 and i % 2 == 0 and i < len(n_hidden) - 1:
                    x = keras.layers.Dense(
                        units=units // 2,
                        kernel_initializer=keras.initializers.HeNormal(),
                        name=f"cross_dense_{i}"
                    )(x)
                    x = keras.layers.LeakyReLU(alpha=0.1, name=f"cross_relu_{i}")(x)

            # Add a pre-output layer with higher regularization
            x = keras.layers.Dense(
                units=32,
                kernel_regularizer=keras.regularizers.l2(l2_reg * 2),
                name="pre_output"
            )(x)
            x = keras.layers.BatchNormalization(name="pre_output_bn")(x)
            x = keras.layers.LeakyReLU(alpha=0.1, name="pre_output_relu")(x)

            # Output layer
            if self.is_multiclass:
                # For 3-class classification (direction: -1, 0, 1)
                outputs = keras.layers.Dense(n_classes, activation="softmax", name="direction_prediction")(x)
                loss = "sparse_categorical_crossentropy"
                metrics = [
                    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                    keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top1_accuracy")
                ]
            else:
                # For binary classification (direction: 0, 1)
                outputs = keras.layers.Dense(1, activation="sigmoid", name="direction_prediction")(x)
                loss = "binary_crossentropy"
                metrics = [
                    "accuracy",
                    keras.metrics.AUC(name="auc")
                ]

            # Assemble the model
            model = keras.Model(inputs=inputs, outputs=outputs)

            # Create optimizer with fixed learning rate
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )

            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )

            # Store model
            self.model = model
            self.logger.info(
                f"Built direction classification model with {sum(n_hidden)} total neurons in hidden layers")
            self.logger.info(f"Model type: {'multiclass (3 classes)' if self.is_multiclass else 'binary'}")
            self.model.summary(print_fn=lambda x: self.logger.debug(x))

        except Exception as e:
            self.logger.error(f"Failed to build model {self.name}: {e}")
            raise

    def prepare_direction_data(self, y_data):
        """Prepare direction data based on model type (binary or multiclass)"""
        # Convert to numpy array if DataFrame
        if isinstance(y_data, pd.DataFrame):
            col = y_data.columns[0]
            y_data = y_data[col].values

        if self.is_multiclass:
            # For multiclass, map -1, 0, 1 to 0, 1, 2 for proper indexing
            # First make sure values are one of the expected three
            unique_values = np.unique(y_data)
            self.logger.info(f"Original direction values: {unique_values}")

            # Map -1 → 0, 0 → 1, 1 → 2
            y_processed = np.zeros_like(y_data, dtype=np.int32)
            y_processed[y_data == -1] = 0
            y_processed[y_data == 0] = 1
            y_processed[y_data == 1] = 2

            # Verify conversion
            self.logger.info(
                f"Converted direction values to class indices: {np.unique(y_processed, return_counts=True)}")
            return y_processed
        else:
            # For binary, just convert to positive or not (> 0 → 1, ≤ 0 → 0)
            y_processed = (y_data > 0).astype(np.int32)
            self.logger.info(f"Converted direction values to binary: {np.unique(y_processed, return_counts=True)}")
            return y_processed

    def train(self, X: pd.DataFrame, y: pd.DataFrame, validation_data=None) -> dict:
        if self.model is None:
            self.logger.error(f"Cannot train {self.name}: model not built")
            return {}

        try:
            # Store feature columns
            self.feature_columns = list(X.columns)

            # Extract target data
            target_col = y.columns[0]

            # Determine if we're dealing with multiclass direction
            if "direction" in target_col:
                unique_values = np.unique(y[target_col].values)
                self.is_multiclass = len(unique_values) > 2 or np.any(unique_values < 0)
                self.logger.info(f"Direction target has unique values: {unique_values}")
                self.logger.info(f"Using {'multiclass' if self.is_multiclass else 'binary'} classification")

                # If we detect 3 classes but model was built for binary (or vice versa), rebuild it
                if hasattr(self.model, 'output_shape'):
                    if self.model.output_shape[-1] == 1 and self.is_multiclass:
                        self.logger.info("Rebuilding model for multiclass classification")
                        self.build(input_shape=(X.shape[1],))
                    elif self.model.output_shape[-1] > 1 and not self.is_multiclass:
                        self.logger.info("Rebuilding model for binary classification")
                        self.build(input_shape=(X.shape[1],))

            # Prepare X data - use scikit-learn StandardScaler for more robust normalization
            X_train = X.values.astype(np.float32)
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Store scaler parameters for prediction
            self.feature_means = self.scaler.mean_
            self.feature_stds = self.scaler.scale_

            # Prepare y data
            y_train = self.prepare_direction_data(y)

            # Prepare validation data
            val_data = None
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val = X_val.values.astype(np.float32)
                X_val_scaled = self.scaler.transform(X_val)
                y_val = self.prepare_direction_data(y_val)
                val_data = (X_val_scaled, y_val)

                # Log class distribution in train and validation sets
                train_dist = np.bincount(y_train)
                val_dist = np.bincount(y_val)
                self.logger.info(f"Training class distribution: {train_dist}")
                self.logger.info(f"Validation class distribution: {val_dist}")

            # Compute class weights to handle imbalance
            if self.is_multiclass:
                # For multiclass
                class_counts = np.bincount(y_train)
                if len(class_counts) < 3:  # Ensure we have all three classes
                    class_counts = np.pad(class_counts, (0, 3 - len(class_counts)))
                total = len(y_train)
                # Square root scaling for less aggressive weighting
                weights = np.sqrt(total / (3 * class_counts))
                # Replace inf/nan with 1.0
                weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
                # Cap weights to avoid extreme values
                weights = np.clip(weights, 0.5, 5.0)
                self.class_weights = {i: weights[i] for i in range(3)}
            else:
                # For binary
                classes = np.unique(y_train)
                if len(classes) == 1:
                    self.logger.warning("Only one class found in training data")
                    self.class_weights = {int(classes[0]): 1.0}
                    # Add missing class with default weight
                    missing_class = 1 if classes[0] == 0 else 0
                    self.class_weights = {missing_class: 1.0}
                else:
                    try:
                        class_weights_array = class_weight.compute_class_weight(
                            class_weight='balanced',
                            classes=classes,
                            y=y_train
                        )
                        # Apply square root to make weights less extreme
                        class_weights_array = np.sqrt(class_weights_array)
                        # Cap weights
                        class_weights_array = np.clip(class_weights_array, 0.5, 5.0)
                        self.class_weights = {c: w for c, w in zip(classes, class_weights_array)}
                    except Exception as e:
                        self.logger.warning(f"Error computing class weights: {e}")
                        self.class_weights = {0: 1.0, 1: 1.0}

            self.logger.info(f"Class weights: {self.class_weights}")

            # Training parameters
            batch_size = self.model_config.get("batch_size", 128)
            epochs = self.model_config.get("epochs", 200)
            patience = self.model_config.get("early_stopping_patience", 25)

            # Create callbacks
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
                    patience=patience // 4,
                    min_lr=1e-6,
                    verbose=1
                ),
                # Add TensorBoard callback for better monitoring
                keras.callbacks.TensorBoard(
                    log_dir=f'./logs/{self.name}',
                    histogram_freq=1,
                    write_graph=True
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

            # Train model with fit
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                callbacks=callbacks,
                class_weight=self.class_weights,
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

            # Reorder columns to match training data and convert to numpy array
            X_pred = X[self.feature_columns].values.astype(np.float32)

            # Apply saved normalization
            if self.feature_means is not None and self.feature_stds is not None:
                # Apply stored scaler params explicitly
                X_pred_scaled = (X_pred - self.feature_means) / self.feature_stds
            else:
                # Fallback to scaler object if available
                X_pred_scaled = self.scaler.transform(X_pred)

            # Generate raw predictions
            raw_preds = self.model.predict(X_pred_scaled)

            # Process predictions based on model type
            if self.is_multiclass:
                # For multiclass, return class with highest probability
                class_preds = np.argmax(raw_preds, axis=1)
                # Map back from 0,1,2 to -1,0,1
                direction_preds = np.zeros_like(class_preds, dtype=np.int32)
                direction_preds[class_preds == 0] = -1
                direction_preds[class_preds == 1] = 0
                direction_preds[class_preds == 2] = 1
                return direction_preds
            else:
                # For binary, return probability (for threshold tuning) or rounded binary prediction
                return raw_preds  # Return probabilities

        except Exception as e:
            self.logger.error(f"Failed to generate predictions with {self.name}: {e}")
            return np.array([])

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
        """Custom evaluation method to replace the one in ModelBase"""
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

            # Prepare y data
            y_eval = self.prepare_direction_data(y)

            # Standard model evaluation for metrics
            metrics = self.model.evaluate(X_eval_scaled, y_eval, verbose=0)
            metrics_dict = dict(zip(self.model.metrics_names, metrics))

            # Add additional metrics for direction prediction
            if self.is_multiclass:
                # Get predictions
                y_pred_probs = self.model.predict(X_eval_scaled)
                y_pred = np.argmax(y_pred_probs, axis=1)

                # Calculate accuracy per class
                for class_idx in range(3):
                    class_mask = (y_eval == class_idx)
                    if np.any(class_mask):
                        class_acc = np.mean(y_pred[class_mask] == class_idx)
                        metrics_dict[f'class_{class_idx}_accuracy'] = float(class_acc)

                # Calculate overall balanced accuracy
                class_accs = [
                    metrics_dict.get(f'class_{i}_accuracy', 0.0)
                    for i in range(3)
                    if metrics_dict.get(f'class_{i}_accuracy') is not None
                ]

                if class_accs:
                    metrics_dict['balanced_accuracy'] = float(np.mean(class_accs))
                else:
                    metrics_dict['balanced_accuracy'] = 0.0

                # Map predictions back to -1, 0, 1 for custom metrics
                dir_values = np.zeros_like(y_pred, dtype=np.int32)
                dir_values[y_pred == 0] = -1
                dir_values[y_pred == 1] = 0
                dir_values[y_pred == 2] = 1

                # Map true values back to -1, 0, 1
                true_dir = np.zeros_like(y_eval, dtype=np.int32)
                true_dir[y_eval == 0] = -1
                true_dir[y_eval == 1] = 0
                true_dir[y_eval == 2] = 1

                # Direction agreement rate (when prediction has same sign as actual)
                # Handle zeros properly by considering them correct only when matching exactly
                sign_true = np.sign(true_dir)
                sign_pred = np.sign(dir_values)

                # Correct for zeros: a zero prediction agrees with a zero actual, not with non-zero actuals
                agreement = (sign_true == sign_pred) | ((sign_true == 0) & (sign_pred == 0))
                sign_agreement = float(np.mean(agreement))

                metrics_dict['direction_agreement'] = sign_agreement

                # Add confusion matrix statistics
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_eval, y_pred)
                if cm.shape == (3, 3):
                    # Calculate direction metrics based on confusion matrix
                    up_to_down = cm[2, 0]  # True up predicted as down
                    down_to_up = cm[0, 2]  # True down predicted as up
                    # Trading cost metrics
                    metrics_dict['critical_error_rate'] = float((up_to_down + down_to_up) / len(y_eval))

                # Add F1 score for each class
                from sklearn.metrics import f1_score
                f1_macro = f1_score(y_eval, y_pred, average='macro')
                metrics_dict['f1_macro'] = float(f1_macro)

            else:
                # Binary classification
                y_pred_probs = self.model.predict(X_eval_scaled)
                y_pred = (y_pred_probs > 0.5).astype(int).flatten()

                # Accuracy per class
                pos_mask = (y_eval == 1)
                neg_mask = (y_eval == 0)

                pos_acc = float(np.mean(y_pred[pos_mask] == 1) if np.any(pos_mask) else 0.0)
                neg_acc = float(np.mean(y_pred[neg_mask] == 0) if np.any(neg_mask) else 0.0)

                metrics_dict['pos_accuracy'] = pos_acc
                metrics_dict['neg_accuracy'] = neg_acc
                metrics_dict['balanced_accuracy'] = (pos_acc + neg_acc) / 2.0

            # Update model metrics
            self.metrics.update(metrics_dict)
            self.logger.info(f"Model {self.name} evaluation: {metrics_dict}")
            return metrics_dict

        except Exception as e:
            self.logger.error(f"Failed to evaluate model {self.name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return basic metrics to prevent errors
            basic_metrics = {'loss': 0.0, 'accuracy': 0.0, 'balanced_accuracy': 0.0}
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

            # Save TensorFlow model
            self.model.save(path)

            # Save feature columns, scaler parameters and metadata
            import json
            metadata = {
                "feature_columns": self.feature_columns,
                "metrics": {k: float(v) if isinstance(v, np.float32) else v for k, v in self.metrics.items()},
                "name": self.name,
                "is_multiclass": self.is_multiclass,
                "saved_at": pd.Timestamp.now().isoformat()
            }

            # Save scaler parameters if available
            if self.feature_means is not None and self.feature_stds is not None:
                metadata["feature_means"] = self.feature_means.tolist()
                metadata["feature_stds"] = self.feature_stds.tolist()

            with open(f"{path}_metadata.json", "w") as f:
                json.dump(metadata, f)

            self.logger.info(f"Model {self.name} saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model {self.name}: {e}")
            return False

    def load(self, path: str) -> bool:
        """Override load method to restore scaler parameters"""
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
                self.is_multiclass = metadata.get("is_multiclass", True)

                # Load scaler parameters if available
                if "feature_means" in metadata and "feature_stds" in metadata:
                    self.feature_means = np.array(metadata["feature_means"])
                    self.feature_stds = np.array(metadata["feature_stds"])

            self.logger.info(f"Model {self.name} loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model {self.name}: {e}")
            return False