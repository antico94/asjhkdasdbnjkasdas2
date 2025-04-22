import json
import os
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils import class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, log_loss, balanced_accuracy_score, roc_auc_score,
    accuracy_score
)

# Assuming ModelBase definition exists
try:
    from Models.ModelBase import ModelBase
except ImportError:
    print("Warning: ModelBase not found. Using a placeholder.")
    class ModelBase:
        def __init__(self, config: dict, logger: logging.Logger, name: str):
            self.config = config
            self.logger = logger
            self.name = name
            self.model = None
            self.history = None


class DirectionClassificationModel(ModelBase):
    CLASS_DOWN = 0
    CLASS_NEUTRAL = 1
    CLASS_UP = 2
    DIRECTION_CLASS_MAPPING = {-1: CLASS_DOWN, 0: CLASS_NEUTRAL, 1: CLASS_UP}
    INVERSE_MAPPING = {v: k for k, v in DIRECTION_CLASS_MAPPING.items()}
    CLASS_NAMES_MULTI = ["DOWN", "NEUTRAL", "UP"]
    CLASS_NAMES_BINARY = ["NOT_UP", "UP"]

    def __init__(self, config: dict, logger: logging.Logger, name: str = "direction_model"):
        super().__init__(config, logger, name)
        model_specific_config = config.get('TrainingSettings', {}).get('ModelConfig', {}).get(self.__class__.__name__, {})
        if not model_specific_config:
            model_specific_config = config.get("MachineLearning", {}).get("Models", {}).get(self.__class__.__name__, {})
        self.model_config = model_specific_config or {}
        self.class_weights = None
        self.is_multiclass = True
        self.scaler = StandardScaler()
        self.feature_means = None
        self.feature_stds = None
        self.class_distribution = None
        self.feature_columns = []
        self.history = {} # Initialize history here
        self.logger.info(f"Initialized {self.name} with config: {self.model_config}")
        tf.random.set_seed(self.model_config.get("seed", 42))
        np.random.seed(self.model_config.get("seed", 42))
        random.seed(self.model_config.get("seed", 42))


    def build(self, input_shape: tuple) -> None:
        """ Builds the Keras model architecture. (Implementation same as before) """
        try:
            n_hidden = self.model_config.get("hidden_units", [256, 128, 64, 32])
            dropout_rate = self.model_config.get("dropout", 0.3)
            learning_rate = self.model_config.get("learning_rate", 0.001)
            l2_reg = self.model_config.get("l2_regularization", 0.001)
            use_attention_config = self.model_config.get("use_attention", True)

            inputs = keras.layers.Input(shape=input_shape, name="input_features")
            x = inputs
            prev_outputs = []

            for i, units in enumerate(n_hidden):
                layer_x = keras.layers.Dense(
                    units=units, kernel_initializer=keras.initializers.HeNormal(),
                    kernel_regularizer=keras.regularizers.l2(l2_reg), name=f"dense_{i}"
                )(x)
                layer_x = keras.layers.BatchNormalization(name=f"bn_{i}")(layer_x)
                layer_x = keras.layers.LeakyReLU(negative_slope=0.1, name=f"leaky_relu_{i}")(layer_x) # Fixed alpha

                if prev_outputs and prev_outputs[-1].shape[-1] == units:
                    layer_x = keras.layers.Add(name=f"residual_{i}")([layer_x, prev_outputs[-1]])

                layer_x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")(layer_x)
                x = layer_x
                prev_outputs.append(x)

                if i > 0 and i % 2 == 0 and i < len(n_hidden) - 1:
                    cross_units = max(16, units // 2)
                    x = keras.layers.Dense(units=cross_units, kernel_initializer=keras.initializers.HeNormal(), name=f"cross_dense_{i}")(x)
                    x = keras.layers.LeakyReLU(negative_slope=0.1, name=f"cross_relu_{i}")(x) # Fixed alpha

            if self.is_multiclass and use_attention_config:
                self.logger.debug("Adding Attention mechanism to the model.")
                attention_dense_units = max(16, n_hidden[-1] // 2)
                attention = keras.layers.Dense(units=attention_dense_units, activation='tanh', name="attention_dense")(x)
                attention_weights = keras.layers.Dense(units=x.shape[-1], activation='sigmoid', name="attention_weights")(attention)
                x = keras.layers.Multiply(name="attention_multiply")([x, attention_weights])

            pre_output_units = max(16, n_hidden[-1] // 2)
            x = keras.layers.Dense(units=pre_output_units, kernel_regularizer=keras.regularizers.l2(l2_reg * 1.5), name="pre_output")(x)
            x = keras.layers.BatchNormalization(name="pre_output_bn")(x)
            x = keras.layers.LeakyReLU(negative_slope=0.1, name="pre_output_relu")(x) # Fixed alpha

            # --- Output Layer Definition (Activation only) ---
            if self.is_multiclass:
                n_classes = 3
                outputs = keras.layers.Dense(n_classes, activation="softmax", name="direction_prediction")(x)
                log_message = (f"Multiclass model build target (3 classes): {self.CLASS_DOWN}=DOWN(-1), "
                               f"{self.CLASS_NEUTRAL}=NEUTRAL(0), {self.CLASS_UP}=UP(1)")
            else: # Binary classification
                n_classes = 1
                outputs = keras.layers.Dense(n_classes, activation="sigmoid", name="direction_prediction")(x)
                log_message = f"Binary model build target: 0={self.CLASS_NAMES_BINARY[0]}(<=0), 1={self.CLASS_NAMES_BINARY[1]}(>0)"

            # --- Model Assembly Only (Compilation happens separately or after loading weights) ---
            model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            self.model = model # Assign the uncompiled model structure

            self.logger.info(f"Built model structure {self.name}.")
            self.logger.info(log_message)
            self.model.summary(print_fn=lambda msg: self.logger.debug(msg))

        except Exception as e:
            self.logger.error(f"Failed to build model structure {self.name}: {e}", exc_info=True)
            self.model = None
            raise

    # --- prepare_direction_data (Keep as it was in the full file example) ---
    def prepare_direction_data(self, y_data):
        """ Prepares direction labels and logs distribution. (Implementation same as before) """
        if isinstance(y_data, pd.DataFrame):
            y_values = y_data.iloc[:, 0].values
        elif isinstance(y_data, pd.Series):
             y_values = y_data.values
        elif isinstance(y_data, np.ndarray):
             y_values = y_data.flatten()
        else:
             raise TypeError("y_data must be a pandas DataFrame, Series, or numpy array.")

        if y_values.size == 0:
            self.logger.warning("Input y_data is empty for prepare_direction_data.")
            return np.array([], dtype=np.int32)

        if self.is_multiclass:
            unique_values = np.unique(y_values)
            self.logger.debug(f"Original direction values (multiclass): {unique_values}")
            y_processed = np.vectorize(self.DIRECTION_CLASS_MAPPING.get)(y_values)
            class_counts = np.bincount(y_processed, minlength=3)
            total_samples = len(y_processed)
            class_distribution = { name: f"{class_counts[i]} ({class_counts[i] / total_samples:.1%})" for i, name in enumerate(self.CLASS_NAMES_MULTI)}
            self.class_distribution = class_distribution
            self.logger.info(f"Multiclass label distribution: {class_distribution}")
            return y_processed.astype(np.int32)
        else: # Binary
            unique_values = np.unique(y_values)
            self.logger.debug(f"Original direction values (binary): {unique_values}")
            y_processed = (y_values > 0).astype(np.int32)
            class_counts = np.bincount(y_processed, minlength=2)
            total_samples = len(y_processed)
            class_distribution = { name: f"{class_counts[i]} ({class_counts[i] / total_samples:.1%})" for i, name in enumerate(self.CLASS_NAMES_BINARY)}
            self.class_distribution = class_distribution
            self.logger.info(f"Binary label distribution: {class_distribution}")
            return y_processed


    # --- focal_loss (Defined as static method with decorator) ---
    @staticmethod
    @register_keras_serializable(package='Custom', name='FocalLoss')
    def focal_loss(gamma=2.0, alpha=0.25):
        """ Focal Loss implementation (details same as before) """
        def sparse_categorical_focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            y_pred = tf.cast(y_pred, tf.float32)
            if len(y_true.shape) > 1 and y_true.shape[-1] == 1:
                y_true = tf.squeeze(y_true, axis=-1)
            num_classes = tf.shape(y_pred)[1]
            y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
            ce = -tf.math.log(p_t + epsilon)
            focal_modulator = tf.pow(1.0 - p_t, gamma)
            loss = focal_modulator * ce
            if alpha is not None:
                alpha_tensor = tf.constant(alpha, dtype=tf.float32)
                alpha_t = tf.gather(alpha_tensor, y_true)
                loss = alpha_t * loss
            return tf.reduce_mean(loss)
        alpha_str = str(alpha).replace('.', '_').replace('[', '').replace(']', '').replace(', ', '_')
        sparse_categorical_focal_loss.__name__ = f'focal_loss_g{gamma}_a{alpha_str}'
        return sparse_categorical_focal_loss


    # --- _create_lr_schedule (Keep as it was in the full file example) ---
    def _create_lr_schedule(self, epochs):
        """ Creates a learning rate schedule with warmup and decay. (Implementation same as before) """
        def lr_scheduler(epoch, lr): # Keras passes current lr
            initial_lr = self.model_config.get("learning_rate", 0.001)
            max_lr = initial_lr * self.model_config.get("lr_schedule_max_factor", 5)
            min_lr = initial_lr / self.model_config.get("lr_schedule_min_factor", 10)
            warmup_epochs = max(1, int(epochs * self.model_config.get("lr_schedule_warmup_ratio", 0.1)))
            total_cycle_epochs = epochs - warmup_epochs
            step_size = max(1, int(total_cycle_epochs * self.model_config.get("lr_schedule_cycle_ratio", 0.4)))
            decay_start_ratio = self.model_config.get("lr_schedule_decay_start_ratio", 0.8)
            decay_rate = self.model_config.get("lr_schedule_decay_rate", 0.95)

            if epoch < warmup_epochs:
                new_lr = min_lr + (initial_lr - min_lr) * (epoch / warmup_epochs)
            else:
                 cycle_epoch = epoch - warmup_epochs
                 cycle = np.floor(1 + cycle_epoch / (2 * step_size))
                 x = abs(cycle_epoch / step_size - 2 * cycle + 1)
                 new_lr = initial_lr + (max_lr - initial_lr) * max(0, (1 - x))
                 if epoch > epochs * decay_start_ratio:
                     decay_factor = decay_rate ** (epoch - epochs * decay_start_ratio)
                     new_lr = max(min_lr, new_lr * decay_factor)
            return float(new_lr)
        return lr_scheduler


    # --- _create_balanced_dataset (Keep as it was in the full file example) ---
    def _create_balanced_dataset(self, X, y, batch_size=32):
        """ Creates a balanced TensorFlow dataset. (Implementation same as before) """
        if not self.is_multiclass:
             self.logger.warning("Balanced dataset requested for binary; returning standard shuffled dataset.")
             return tf.data.Dataset.from_tensor_slices((X, y)).shuffle(buffer_size=len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        num_classes = 3
        datasets_per_class = []
        counts_per_class = []

        for i in range(num_classes):
            class_indices = np.where(y == i)[0]
            count = len(class_indices)
            counts_per_class.append(count)
            if count == 0:
                self.logger.warning(f"No samples for class {self.CLASS_NAMES_MULTI[i]} in balanced dataset creation.")
                continue
            class_X = X[class_indices]
            class_y = y[class_indices]
            datasets_per_class.append(tf.data.Dataset.from_tensor_slices((class_X, class_y)).repeat().shuffle(buffer_size=max(1000, count)))

        if not datasets_per_class:
             self.logger.error("No data available for any class. Cannot create balanced dataset.")
             return None

        num_active_classes = len(datasets_per_class)
        sampling_weights = [1.0 / num_active_classes] * num_active_classes

        try:
            balanced_ds = tf.data.Dataset.sample_from_datasets(datasets_per_class, weights=sampling_weights, stop_on_empty_dataset=False)
            balanced_ds = balanced_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            self.logger.info(f"Created balanced dataset sampling from {num_active_classes} classes.")
            return balanced_ds
        except Exception as ds_err:
             self.logger.error(f"Failed to create balanced dataset: {ds_err}", exc_info=True)
             return None


    # --- predict method (Keep as it was in the full file example) ---
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """ Generates predictions for new data. (Implementation same as before) """
        if self.model is None or not self.feature_columns:
            self.logger.error(f"Cannot predict with {self.name}: Model not built or trained properly.")
            return np.array([])
        if not self.model.built:
            self.logger.error(f"Cannot predict with {self.name}: Model structure exists but has not been built (run build() or train()).")
            return np.array([])

        self.logger.debug(f"Starting prediction for {X.shape[0]} samples.")
        try:
            missing_cols = [col for col in self.feature_columns if col not in X.columns]
            if missing_cols: raise ValueError(f"Missing columns required for prediction: {missing_cols}")
            if X.shape[1] > len(self.feature_columns):
                 extra_cols = [col for col in X.columns if col not in self.feature_columns]
                 self.logger.warning(f"Input data has extra columns: {extra_cols}. Ignoring them.")

            X_pred_np = X[self.feature_columns].values.astype(np.float32)

            if self.feature_means is not None and self.feature_stds is not None:
                 safe_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
                 X_pred_scaled = (X_pred_np - self.feature_means) / safe_stds
            else:
                 self.logger.warning("Scaler parameters not found. Using scaler object transform.")
                 X_pred_scaled = self.scaler.transform(X_pred_np)

            raw_preds = self.model.predict(X_pred_scaled)

            if self.is_multiclass:
                class_preds = np.argmax(raw_preds, axis=1)
                direction_preds = np.vectorize(self.INVERSE_MAPPING.get)(class_preds)
                self.logger.debug("Generated multiclass direction predictions {-1, 0, 1}.")
                return direction_preds
            else:
                binary_probs = raw_preds.squeeze(axis=-1)
                self.logger.debug("Generated binary probabilities {0.0 to 1.0}.")
                return binary_probs
        except Exception as e:
            self.logger.error(f"Prediction failed for {self.name}: {e}", exc_info=True)
            return np.array([])


    # --- FULLY REVISED train METHOD ---
    def train(self, X: pd.DataFrame, y: pd.DataFrame, validation_data=None) -> dict:
        """
        Trains the direction classification model. Handles model building,
        compilation, data prep, fitting, and checkpointing.
        """
        self.logger.info(f"--- Starting Training for {self.name} ---")
        self.history = {} # Initialize history attribute at the beginning

        if X.empty or y.empty:
            self.logger.error("Training aborted: Input features (X) or labels (y) are empty.")
            return {"error": "Input data is empty."}

        # --- Feature and Target Type Verification & Model Build/Rebuild ---
        try:
            if not self.feature_columns:
                 self.feature_columns = list(X.columns)
                 self.logger.info(f"Set feature columns during training: {self.feature_columns}")
            elif set(self.feature_columns) != set(X.columns):
                 self.logger.warning("Input features differ from stored. Updating feature columns.")
                 self.feature_columns = list(X.columns)
                 # Force rebuild if feature count changes the expected input shape
                 if self.model and hasattr(self.model, 'input_shape') and self.model.input_shape[1] != len(self.feature_columns):
                      self.logger.info("Input feature count changed. Forcing model rebuild.")
                      self.model = None # Mark for rebuild

            target_col = y.columns[0] if isinstance(y, pd.DataFrame) else y.name
            new_is_multiclass = self.is_multiclass

            if "direction" in target_col.lower():
                unique_values = np.unique(y.iloc[:, 0].values if isinstance(y, pd.DataFrame) else y.values)
                new_is_multiclass = len(unique_values) > 2 or np.any(unique_values < 0)
                self.logger.info(f"Detected target type: {'Multiclass' if new_is_multiclass else 'Binary'} based on unique values: {unique_values}")
            else:
                 self.logger.warning(f"Target column '{target_col}' may not indicate direction. Assuming 'is_multiclass'={self.is_multiclass}.")

            needs_rebuild = False
            if self.model is None:
                needs_rebuild = True
            elif self.is_multiclass != new_is_multiclass:
                needs_rebuild = True
                self.logger.info(f"Switching model type.")
            elif hasattr(self.model, 'output_shape') and self.model.output_shape is not None:
                expected_output_size = 3 if new_is_multiclass else 1
                if self.model.output_shape[-1] != expected_output_size:
                    needs_rebuild = True
            elif hasattr(self.model, 'input_shape') and self.model.input_shape is not None:
                 if self.model.input_shape[-1] != X.shape[1]:
                     needs_rebuild = True

            if needs_rebuild:
                self.logger.info("Rebuilding model structure...")
                self.is_multiclass = new_is_multiclass
                self.build(input_shape=(X.shape[1],))
                if self.model is None: raise RuntimeError("Model build failed during training.")

        except Exception as build_err:
            self.logger.error(f"Model build/rebuild failed: {build_err}", exc_info=True)
            return {"error": f"Model build/rebuild failed: {build_err}"}

        # --- Data Preparation ---
        try:
            X_train_np = X[self.feature_columns].values.astype(np.float32)
            X_train_scaled = self.scaler.fit_transform(X_train_np)
            self.feature_means = self.scaler.mean_
            self.feature_stds = self.scaler.scale_
            if self.feature_stds is not None and np.any(self.feature_stds == 0):
                self.logger.warning("Scaler detected std dev of zero. Replacing with 1.0.")
                self.feature_stds[self.feature_stds == 0] = 1.0

            y_train = self.prepare_direction_data(y)
            total_train = len(y_train)
            if total_train == 0: raise ValueError("Training labels became empty after preparation.")

            val_data_prepared = None
            if validation_data is not None and isinstance(validation_data, (list, tuple)) and len(validation_data) == 2:
                X_val, y_val = validation_data
                if set(self.feature_columns) != set(X_val.columns):
                    raise ValueError("Validation features mismatch training features.")
                X_val_np = X_val[self.feature_columns].values.astype(np.float32)
                X_val_scaled = self.scaler.transform(X_val_np)
                y_val_prepared = self.prepare_direction_data(y_val)
                val_data_prepared = (X_val_scaled, y_val_prepared)
                self.logger.info("Validation data prepared.")
            elif validation_data is not None:
                self.logger.warning("Validation data ignored: incorrect format.")

        except Exception as prep_err:
             self.logger.error(f"Data preparation failed: {prep_err}", exc_info=True)
             return {"error": f"Data preparation failed: {prep_err}"}

        # --- Determine Loss, Metrics, Optimizer and Compile ---
        try:
            use_focal_loss = self.model_config.get("use_focal_loss", True)
            learning_rate = self.model_config.get("learning_rate", 0.001) # Base LR for optimizer
            loss_to_compile = None
            metrics_to_compile = []
            self.class_weights = None # Reset class weights

            if self.is_multiclass:
                metrics_to_compile = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                class_counts_train = np.bincount(y_train, minlength=3)
                if use_focal_loss and total_train > 0 and not np.any(class_counts_train == 0):
                    beta = 0.999
                    effective_num = 1.0 - np.power(beta, class_counts_train)
                    alpha_weights = (1.0 - beta) / effective_num
                    alpha_weights /= np.sum(alpha_weights)
                    alpha_weights = np.clip(alpha_weights, 0.1, 0.9)
                    focal_alpha = list(alpha_weights)
                    loss_to_compile = self.focal_loss(gamma=2.0, alpha=focal_alpha)
                    self.logger.info(f"Using Focal Loss with dynamic alpha: {focal_alpha}")
                else:
                    if use_focal_loss: self.logger.warning("Cannot use Focal Loss (missing classes/data).")
                    loss_to_compile = "sparse_categorical_crossentropy"
                    # Calculate class weights if not using focal loss
                    if total_train > 0 and not np.any(class_counts_train == 0):
                         weight_values = class_weight.compute_class_weight('balanced', classes=np.arange(3), y=y_train)
                         weight_values = np.clip(weight_values, 0.2, 5.0)
                         self.class_weights = dict(enumerate(weight_values))
                         weights_log = {self.CLASS_NAMES_MULTI[i]: f"{w:.2f}" for i, w in self.class_weights.items()}
                         self.logger.info(f"Using Class Weights: {weights_log}")
                    else:
                         self.logger.warning("Cannot compute class weights (missing classes/data). Using equal weights.")
                         self.class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
            else: # Binary
                loss_to_compile = "binary_crossentropy"
                metrics_to_compile = [keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")]
                class_counts_train = np.bincount(y_train, minlength=2)
                if total_train > 0 and not np.any(class_counts_train == 0):
                     weight_values = class_weight.compute_class_weight('balanced', classes=np.arange(2), y=y_train)
                     weight_values = np.clip(weight_values, 0.2, 5.0)
                     self.class_weights = dict(enumerate(weight_values))
                     weights_log = {self.CLASS_NAMES_BINARY[i]: f"{w:.2f}" for i, w in self.class_weights.items()}
                     self.logger.info(f"Using Class Weights: {weights_log}")
                else:
                     self.logger.warning("Cannot compute class weights (missing classes/data). Using equal weights.")
                     self.class_weights = {0: 1.0, 1: 1.0}

            optimizer = keras.optimizers.Adam(learning_rate=learning_rate) # LR schedule handles rate changes later

            # Compile the model before fitting
            self.model.compile(optimizer=optimizer, loss=loss_to_compile, metrics=metrics_to_compile)
            self.logger.info("Model compiled successfully.")

        except Exception as compile_err:
             self.logger.error(f"Model compilation failed: {compile_err}", exc_info=True)
             return {"error": f"Model compilation failed: {compile_err}"}


        # --- Callbacks Setup ---
        try:
            epochs = self.model_config.get("epochs", 50)
            patience = self.model_config.get("early_stopping_patience", 15)
            batch_size = self.model_config.get("batch_size", 128)
            save_checkpoints = self.model_config.get("save_checkpoints", True)
            log_dir = f'./logs/{self.name}'
            os.makedirs(log_dir, exist_ok=True)
            checkpoint_filepath = os.path.join(log_dir, f"{self.name}_best_weights.weights.h5")

            callbacks = [
                 keras.callbacks.EarlyStopping(
                     monitor="val_loss" if val_data_prepared else "loss", patience=patience,
                     restore_best_weights=True, verbose=1, mode='min'
                 ),
                 keras.callbacks.LearningRateScheduler(self._create_lr_schedule(epochs), verbose=0),
                 keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            ]
            if save_checkpoints:
                 callbacks.append(keras.callbacks.ModelCheckpoint(
                     filepath=checkpoint_filepath, save_best_only=True, save_weights_only=True,
                     monitor="val_loss" if val_data_prepared else "loss", verbose=1, mode='min'
                 ))
        except Exception as cb_err:
             self.logger.error(f"Callback setup failed: {cb_err}", exc_info=True)
             return {"error": f"Callback setup failed: {cb_err}"}


        # --- Model Fitting ---
        self.logger.info(f"Starting model fitting... Epochs: {epochs}, Batch Size: {batch_size}")
        fit_args = {"epochs": epochs, "callbacks": callbacks, "verbose": 1}
        if val_data_prepared: fit_args["validation_data"] = val_data_prepared

        use_balanced_sampling_config = self.model_config.get("use_balanced_batch_sampling", False)
        min_c, max_c = np.min(class_counts_train), np.max(class_counts_train)
        imbalance_ratio_train = max_c / min_c if min_c > 0 else float('inf')

        # Local variable for history within the try block for fitting
        history_obj = None
        try:
            if self.is_multiclass and use_balanced_sampling_config and imbalance_ratio_train > 5.0 :
                self.logger.info(f"Using balanced batch sampling (imbalance ratio: {imbalance_ratio_train:.1f})")
                train_dataset = self._create_balanced_dataset(X_train_scaled, y_train, batch_size=batch_size)
                if not train_dataset: raise RuntimeError("Failed to create balanced dataset.")
                fit_args["steps_per_epoch"] = max(1, total_train // batch_size)
                if val_data_prepared:
                    val_dataset = tf.data.Dataset.from_tensor_slices(val_data_prepared).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                    fit_args["validation_data"] = val_dataset
                history_obj = self.model.fit(train_dataset, **fit_args) # No class_weight with sampling
            else:
                self.logger.info("Using standard training data feed.")
                fit_args["x"] = X_train_scaled
                fit_args["y"] = y_train
                fit_args["batch_size"] = batch_size
                # Apply class weights ONLY if they were calculated (i.e., not using focal loss)
                if self.class_weights:
                     fit_args["class_weight"] = self.class_weights
                     self.logger.info("Applying class weights.")
                else:
                     self.logger.info("Not applying class weights (using focal loss or weights unavailable).")
                history_obj = self.model.fit(**fit_args)

            # Assign to self.history ONLY if fit was successful
            if history_obj:
                 self.history = history_obj.history
                 self.logger.info("Model fitting finished successfully.")

        except Exception as fit_err:
             self.logger.error(f"Model fitting failed: {fit_err}", exc_info=True)
             # self.history remains {}, return error
             return {"error": f"Model fitting failed: {fit_err}"}

        # --- Post-Training Logging (if fit succeeded) ---
        if save_checkpoints:
             if os.path.exists(checkpoint_filepath): self.logger.info(f"Best weights saved: {checkpoint_filepath}")
             else: self.logger.warning("Checkpoint file not found despite restore_best_weights=True.")

        if self.history: # Check if history dict is populated
            final_metrics = {k: f"{v[-1]:.4f}" for k, v in self.history.items()}
            self.logger.info(f"Training completed. Final epoch metrics: {final_metrics}")
            if val_data_prepared and 'val_loss' in self.history:
                best_val_loss_epoch = np.argmin(self.history['val_loss'])
                best_val_metrics = {k: f"{v[best_val_loss_epoch]:.4f}" for k, v in self.history.items() if k.startswith('val_')}
                self.logger.info(f"Best validation metrics (epoch {best_val_loss_epoch + 1}): {best_val_metrics}")
        else:
            self.logger.warning("Training finished, but no history object was generated.")

        self.logger.info(f"--- Training Finished for {self.name} ---")
        return self.history


    # --- FULLY REVISED evaluate METHOD ---
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
        """
        Provides comprehensive evaluation metrics. Ensures data shapes are correct.
        """
        if self.model is None or not self.feature_columns:
            self.logger.error(f"Cannot evaluate {self.name}: Model not built or trained properly.")
            return {"error": "Model not ready for evaluation."}
        if not self.model.built:
             self.logger.error(f"Cannot evaluate {self.name}: Model structure exists but is not built.")
             return {"error": "Model not built."}
        # Check if the model is compiled (necessary for evaluate)
        if getattr(self.model, "optimizer", None) is None:
            self.logger.error(f"Cannot evaluate {self.name}: Model is not compiled.")
            return {"error": "Model not compiled."}


        self.logger.info(f"--- Starting Evaluation for {self.name} ---")
        results = {}

        try:
            # --- Data Preparation ---
            self.logger.debug("Preparing evaluation data...")
            missing_cols = [col for col in self.feature_columns if col not in X.columns]
            if missing_cols: raise ValueError(f"Missing evaluation columns: {missing_cols}")

            X_eval_np = X[self.feature_columns].values.astype(np.float32)
            if self.feature_means is not None and self.feature_stds is not None:
                safe_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
                X_eval_scaled = (X_eval_np - self.feature_means) / safe_stds
            else:
                X_eval_scaled = self.scaler.transform(X_eval_np)

            # Prepare true labels and ensure they are 1D: (n_samples,)
            y_true_mapped = self.prepare_direction_data(y)
            if y_true_mapped.ndim > 1: y_true_mapped = y_true_mapped.squeeze()
            if y_true_mapped.size != X_eval_scaled.shape[0]:
                 raise ValueError(f"Shape mismatch between features ({X_eval_scaled.shape[0]}) and labels ({y_true_mapped.size}) after prep.")
            if y_true_mapped.size == 0: raise ValueError("Prepared true labels are empty.")
            self.logger.debug(f"Prepared evaluation labels shape: {y_true_mapped.shape}")

            # --- Get Predictions ---
            self.logger.debug("Generating predictions for evaluation...")
            y_pred_probs = self.model.predict(X_eval_scaled)
            self.logger.debug("Predictions generated.")


            # --- Keras Compiled Metrics ---
            self.logger.debug("Calculating Keras base metrics...")
            try:
                 # Ensure y_true_mapped is 1D as expected by sparse/binary metrics usually
                 keras_metrics = self.model.evaluate(X_eval_scaled, y_true_mapped, verbose=0)
                 keras_metrics_dict = dict(zip(self.model.metrics_names, keras_metrics))
                 results["keras_metrics"] = {k: float(v) for k, v in keras_metrics_dict.items()}
                 self.logger.info(f"Keras Base Metrics: {results['keras_metrics']}")
            except Exception as keras_eval_err:
                 self.logger.error(f"Keras model.evaluate() failed: {keras_eval_err}", exc_info=True)
                 results["keras_metrics"] = {"error": str(keras_eval_err)}


            # --- Detailed Scikit-learn Metrics ---
            self.logger.debug("Calculating detailed classification metrics...")
            eps = 1e-15
            y_pred_probs_clipped = np.clip(y_pred_probs, eps, 1 - eps)

            if self.is_multiclass:
                 n_classes = 3
                 class_names = self.CLASS_NAMES_MULTI
                 labels = [self.CLASS_DOWN, self.CLASS_NEUTRAL, self.CLASS_UP]
                 y_pred_mapped = np.argmax(y_pred_probs, axis=1) # Shape: (n_samples,)

                 results['log_loss'] = float(log_loss(y_true_mapped, y_pred_probs_clipped, labels=labels))
                 cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels)
                 report_dict = classification_report(y_true_mapped, y_pred_mapped, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
                 report_str = classification_report(y_true_mapped, y_pred_mapped, labels=labels, target_names=class_names, zero_division=0)
                 results['mcc'] = float(matthews_corrcoef(y_true_mapped, y_pred_mapped))
                 results['cohen_kappa'] = float(cohen_kappa_score(y_true_mapped, y_pred_mapped, labels=labels))
                 results['balanced_accuracy'] = float(balanced_accuracy_score(y_true_mapped, y_pred_mapped))
                 results['overall_accuracy'] = float(accuracy_score(y_true_mapped, y_pred_mapped))
                 results['confusion_matrix'] = cm.tolist()
                 results['classification_report'] = report_dict
                 # ... (logging for multiclass results - same as before) ...
                 self.logger.info(f"Log Loss: {results['log_loss']:.4f}")
                 self.logger.info(f"Confusion Matrix (Rows: True, Cols: Pred):\nLabels: {class_names}\n{cm}")
                 self.logger.info(f"Classification Report:\n{report_str}")
                 self.logger.info(f"MCC: {results['mcc']:.4f}, Kappa: {results['cohen_kappa']:.4f}, Balanced Acc: {results['balanced_accuracy']:.4f}, Overall Acc: {results['overall_accuracy']:.4f}")


                 # --- Multiclass AUC ---
                 try:
                     y_true_binarized = label_binarize(y_true_mapped, classes=labels)
                     if y_true_binarized.shape[1] > 1 and y_true_binarized.shape[1] == y_pred_probs_clipped.shape[1] :
                          auc_ovr_weighted = roc_auc_score(y_true_binarized, y_pred_probs_clipped, average='weighted', multi_class='ovr')
                          results['auc_weighted_ovr'] = float(auc_ovr_weighted)
                          self.logger.info(f"AUC (Weighted One-vs-Rest): {results['auc_weighted_ovr']:.4f}")
                     else:
                          self.logger.warning(f"Skipping Multiclass AUC: Not enough classes in true labels ({y_true_binarized.shape[1]}) or mismatch with predictions ({y_pred_probs_clipped.shape[1]}).")
                          results['auc_weighted_ovr'] = None
                 except Exception as auc_err:
                     self.logger.warning(f"Could not calculate Multiclass AUC: {auc_err}", exc_info=False) # Less verbose exc_info
                     results['auc_weighted_ovr'] = None

            else: # Binary classification
                 class_names = self.CLASS_NAMES_BINARY
                 labels = [0, 1]
                 # Ensure probs are 1D: (n_samples,)
                 y_pred_probs_binary = y_pred_probs_clipped.squeeze(axis=-1)
                 if y_pred_probs_binary.ndim == 0 and y_true_mapped.ndim == 0 and y_true_mapped.size == 1: # Handle single prediction case
                       y_pred_probs_binary = np.array([y_pred_probs_binary])
                       y_true_mapped = np.array([y_true_mapped]) # Ensure they are arrays for sklearn funcs
                 elif y_pred_probs_binary.ndim != 1:
                       raise ValueError(f"Binary probabilities have unexpected shape: {y_pred_probs.shape}")

                 y_pred_mapped = (y_pred_probs_binary > 0.5).astype(int) # Shape: (n_samples,)

                 results['log_loss'] = float(log_loss(y_true_mapped, y_pred_probs_binary, labels=labels))
                 cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels)
                 report_dict = classification_report(y_true_mapped, y_pred_mapped, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
                 report_str = classification_report(y_true_mapped, y_pred_mapped, labels=labels, target_names=class_names, zero_division=0)
                 results['mcc'] = float(matthews_corrcoef(y_true_mapped, y_pred_mapped))
                 results['cohen_kappa'] = float(cohen_kappa_score(y_true_mapped, y_pred_mapped, labels=labels))
                 results['balanced_accuracy'] = float(balanced_accuracy_score(y_true_mapped, y_pred_mapped))
                 results['overall_accuracy'] = float(accuracy_score(y_true_mapped, y_pred_mapped))
                 results['confusion_matrix'] = cm.tolist()
                 results['classification_report'] = report_dict
                 # ... (logging for binary results - same as before) ...
                 self.logger.info(f"Log Loss: {results['log_loss']:.4f}")
                 self.logger.info(f"Confusion Matrix (Rows: True, Cols: Pred):\nLabels: {class_names}\n{cm}")
                 self.logger.info(f"Classification Report:\n{report_str}")
                 self.logger.info(f"MCC: {results['mcc']:.4f}, Kappa: {results['cohen_kappa']:.4f}, Balanced Acc: {results['balanced_accuracy']:.4f}, Overall Acc: {results['overall_accuracy']:.4f}")


                 # --- Binary AUC ---
                 try:
                    if len(np.unique(y_true_mapped)) > 1:
                        auc_score = roc_auc_score(y_true_mapped, y_pred_probs_binary)
                        results['auc'] = float(auc_score)
                        keras_auc = results.get("keras_metrics", {}).get("auc")
                        auc_log = f"AUC (Binary): {results['auc']:.4f}"
                        if keras_auc is not None: auc_log += f" (Keras reported: {keras_auc:.4f})"
                        self.logger.info(auc_log)
                    else:
                        self.logger.warning("Skipping Binary AUC: Only one class present in true labels.")
                        results['auc'] = None
                 except Exception as auc_err:
                    self.logger.warning(f"Could not calculate Binary AUC: {auc_err}", exc_info=False)
                    results['auc'] = None

            self.logger.info(f"--- Evaluation Complete for {self.name} ---")
            return results

        except Exception as e:
            self.logger.error(f"Failed during evaluation of model {self.name}: {e}", exc_info=True)
            return {"error": f"Evaluation failed: {e}"}


    # --- FULLY REVISED save METHOD (Weights Only) ---
    def save(self, path: str) -> None:
        """
        Saves the model's weights and necessary metadata (scaler, features, etc.).
        Uses model.save_weights() for robustness.
        """
        if self.model is None:
            self.logger.error(f"Cannot save {self.name}: No model object found.")
            return
        if not self.model.built:
             self.logger.error(f"Cannot save {self.name}: Model is not built.")
             return
        if not self.feature_columns:
            self.logger.warning(f"Saving model {self.name} without stored feature columns.")
        # Check if scaler params exist, required for consistent loading
        if self.feature_means is None or self.feature_stds is None:
             self.logger.warning(f"Saving model {self.name} without scaler parameters (means/stds). Reloading might require retraining the scaler.")


        try:
            os.makedirs(path, exist_ok=True)
            self.logger.info(f"Saving model '{self.name}' weights and metadata to: {path}")

            # 1. Save Model Weights
            weights_filename = f"{self.name}_weights.weights.h5" # Specific extension
            weights_path = os.path.join(path, weights_filename)
            self.model.save_weights(weights_path) # Use save_weights
            self.logger.info(f"Model weights saved to {weights_path}")

            # 2. Save Metadata
            metadata = {
                "model_class": self.__class__.__name__,
                "name": self.name,
                "is_multiclass": self.is_multiclass,
                "feature_columns": self.feature_columns,
                # Store input shape HINT used for build (e.g., number of features)
                "input_shape_hint": self.model.input_shape[1:] if hasattr(self.model, 'input_shape') and self.model.input_shape else (len(self.feature_columns),) if self.feature_columns else None,
                "scaler_means": self.feature_means.tolist() if self.feature_means is not None else None,
                "scaler_stds": self.feature_stds.tolist() if self.feature_stds is not None else None,
                "class_distribution_train": self.class_distribution,
                "weights_filename": weights_filename, # Link to weights file
                # Store the configuration used for building architecture if needed
                "model_config_build": self.model_config # Save config relevant to build
            }
            metadata_filename = f"{self.name}_metadata.json"
            metadata_path = os.path.join(path, metadata_filename)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            self.logger.info(f"Model metadata saved to {metadata_path}")

        except Exception as e:
            self.logger.error(f"Failed to save model weights/metadata for {self.name} to {path}: {e}", exc_info=True)
            raise


    # --- FULLY REVISED load METHOD (Rebuild + Weights + Compile) ---
    @classmethod
    def load(cls, path: str, config: dict, logger: logging.Logger) -> 'DirectionClassificationModel':
        """
        Loads a model by rebuilding architecture, loading weights, and compiling.
        Relies on metadata saved alongside weights.
        """
        logger.info(f"--- Loading Model (Rebuild+Weights) from: {path} ---")
        try:
            # 1. Find and load metadata
            metadata_path = None
            expected_metadata_suffix = "_metadata.json"
            for fname in os.listdir(path):
                if fname.endswith(expected_metadata_suffix):
                    metadata_path = os.path.join(path, fname)
                    logger.info(f"Found metadata file: {metadata_path}")
                    break
            if not metadata_path:
                raise FileNotFoundError(f"Metadata JSON file (*{expected_metadata_suffix}) not found in {path}")

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info("Metadata loaded successfully.")

            # --- Instantiate the class ---
            model_name = metadata.get("name", "loaded_model")
            # Use the original config potentially updated with saved build config
            # This allows loading architecture independent of current run's config
            instance_config = config # Start with current config
            saved_model_config = metadata.get("model_config_build")
            if saved_model_config:
                 # Need a deep merge strategy if only partially overriding
                 logger.info("Using saved model config for build parameters.")
                 # Simple override for now:
                 instance_config['TrainingSettings'] = instance_config.get('TrainingSettings', {})
                 instance_config['TrainingSettings']['ModelConfig'] = instance_config['TrainingSettings'].get('ModelConfig', {})
                 instance_config['TrainingSettings']['ModelConfig'][cls.__name__] = saved_model_config

            instance = cls(config=instance_config, logger=logger, name=model_name)

            # Restore state from metadata BEFORE building
            instance.is_multiclass = metadata.get("is_multiclass")
            if instance.is_multiclass is None: raise ValueError("Metadata missing 'is_multiclass' flag.")
            instance.feature_columns = metadata.get("feature_columns", [])
            input_shape_hint = metadata.get("input_shape_hint")
            if not instance.feature_columns and not input_shape_hint:
                 raise ValueError("Metadata lacks feature_columns and input_shape_hint; cannot determine input shape.")
            if not input_shape_hint: input_shape_hint = (len(instance.feature_columns),)

            # Restore scaler
            means = metadata.get("scaler_means")
            stds = metadata.get("scaler_stds")
            if means is not None and stds is not None:
                instance.feature_means = np.array(means)
                instance.feature_stds = np.array(stds)
                instance.scaler = StandardScaler()
                instance.scaler.mean_ = instance.feature_means
                instance.scaler.scale_ = instance.feature_stds
                if instance.feature_columns: instance.scaler.n_features_in_ = len(instance.feature_columns)
                logger.info("StandardScaler parameters restored.")
            else:
                logger.warning("Scaler parameters not found in metadata.")
                instance.scaler = StandardScaler()

            # --- 2. Rebuild Model Architecture ---
            logger.info(f"Rebuilding model structure with input shape hint: {input_shape_hint}")
            instance.build(input_shape=input_shape_hint)
            if instance.model is None:
                 raise RuntimeError("Model build failed during load.")

            # --- 3. Load Weights ---
            weights_filename = metadata.get("weights_filename")
            if not weights_filename: raise ValueError("Metadata missing 'weights_filename'.")
            weights_path = os.path.join(path, weights_filename)
            if not os.path.exists(weights_path): raise FileNotFoundError(f"Weights file not found: {weights_path}")

            instance.model.load_weights(weights_path)
            logger.info(f"Model weights loaded successfully from {weights_path}")

            # --- 4. Compile Model ---
            logger.info("Compiling loaded model...")
            # Determine loss and metrics based on loaded state (is_multiclass)
            # Note: Focal loss alpha cannot be restored dynamically here, compile with standard loss
            # Or retrieve alpha from metadata if stored during save (more complex)
            compile_loss = None
            compile_metrics = []
            if instance.is_multiclass:
                 # Cannot reliably restore dynamic focal loss alpha without saving it. Use standard.
                 compile_loss = "sparse_categorical_crossentropy"
                 compile_metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                 logger.warning("Compiling loaded multiclass model with standard SparseCategoricalCrossentropy (Focal Loss alpha state not saved).")
            else:
                 compile_loss = "binary_crossentropy"
                 compile_metrics = [keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")]

            # Use a base learning rate, scheduler is not restored here
            compile_lr = instance.model_config.get("learning_rate", 0.001)
            optimizer = keras.optimizers.Adam(learning_rate=compile_lr)

            instance.model.compile(optimizer=optimizer, loss=compile_loss, metrics=compile_metrics)
            logger.info("Loaded model compiled successfully.")

            logger.info(f"--- Model '{instance.name}' Loaded Successfully (Rebuild+Weights) ---")
            return instance

        except FileNotFoundError as fnf_err:
             logger.error(f"Failed to load model from {path}: {fnf_err}", exc_info=True)
             raise fnf_err
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}", exc_info=True)
            raise Exception(f"Failed to load model from {path}: {e}")
