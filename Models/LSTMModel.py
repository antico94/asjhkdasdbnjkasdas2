import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model, register_keras_serializable
import numpy as np
import os
from typing import Dict, List, Any, Tuple

@register_keras_serializable(package='Custom', name='AttentionLayer')
class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention mechanism to focus on important time steps."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Alignment scores
        e = tf.tanh(tf.matmul(x, self.W) + self.b)

        # Get attention weights
        a = tf.nn.softmax(e, axis=1)

        # Apply attention weights to input
        context = x * a
        context = tf.reduce_sum(context, axis=1)

        return context

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        # no additional args to add here, but if you add hyperparameters in __init__,
        # include them in this dict
        return config


class LSTMModel:
    def __init__(self, config, input_shape: Tuple[int, int], n_features: int, model_type: str = "combined"):
        self.config = config
        self.sequence_length = input_shape[0]
        self.n_features = n_features
        self.model = None
        self.history = None
        self.model_type = model_type  # "direction", "magnitude", "volatility", or "combined"

        # Get model configuration from config
        ml_config = self.config.get('GoldTradingSettings', {}).get('MachineLearning', {})
        lstm_config = ml_config.get('Hyperparameters', {}).get('LSTM', {})

        # Model parameters with defaults
        self.units = lstm_config.get('units', 50)
        self.dropout = lstm_config.get('dropout', 0.2)
        self.lr = lstm_config.get('learning_rate', 0.001)
        self.batch_size = lstm_config.get('batch_size', 32)
        self.epochs = lstm_config.get('epochs', 50)

    def build_model(self) -> None:
        """Build the LSTM model architecture based on model_type."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))

        # LSTM layers
        x = Bidirectional(LSTM(self.units, return_sequences=True))(inputs)
        x = Dropout(self.dropout)(x)
        x = Bidirectional(LSTM(self.units, return_sequences=True))(x)
        x = Dropout(self.dropout)(x)

        # Attention mechanism
        x = AttentionLayer()(x)
        x = Dropout(self.dropout)(x)

        # Common dense layers
        x = Dense(32, activation='relu')(x)

        # Output layer(s) based on model type
        outputs = []
        if self.model_type == "direction" or self.model_type == "combined":
            direction_output = Dense(1, activation='sigmoid', name='direction')(x)
            outputs.append(direction_output)

        if self.model_type == "magnitude" or self.model_type == "combined":
            magnitude_output = Dense(1, activation='linear', name='magnitude')(x)
            outputs.append(magnitude_output)

        if self.model_type == "volatility" or self.model_type == "combined":
            volatility_output = Dense(1, activation='relu', name='volatility')(x)
            outputs.append(volatility_output)

        # Create model with appropriate outputs
        self.model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        self.compile_model()

    def compile_model(self) -> None:
        """Compile the model with appropriate loss functions and metrics."""
        # Define losses, metrics and weights based on model type
        losses = {}
        metrics = {}
        loss_weights = {}

        if self.model_type == "direction" or self.model_type == "combined":
            losses['direction'] = 'binary_crossentropy'
            metrics['direction'] = ['accuracy']
            loss_weights['direction'] = 1.0

        if self.model_type == "magnitude" or self.model_type == "combined":
            losses['magnitude'] = 'mse'
            metrics['magnitude'] = ['mae']
            loss_weights['magnitude'] = 0.5 if self.model_type == "combined" else 1.0

        if self.model_type == "volatility" or self.model_type == "combined":
            losses['volatility'] = 'mse'
            metrics['volatility'] = ['mae']
            loss_weights['volatility'] = 0.3 if self.model_type == "combined" else 1.0

        # Compile model with Adam optimizer and learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )

    def fit(self, dataset: Dict[str, Any], epochs: int = None, batch_size: int = None,
            callbacks: List = None) -> Dict[str, Any]:
        """Train the model."""
        if self.model is None:
            self.build_model()

        # Use parameters or defaults
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size

        # Prepare target data format for model based on model type
        y_train = {}
        y_val = {}

        if self.model_type == "direction" or self.model_type == "combined":
            y_train['direction'] = dataset['y_train']['direction']
            y_val['direction'] = dataset['y_val']['direction']

        if self.model_type == "magnitude" or self.model_type == "combined":
            y_train['magnitude'] = dataset['y_train']['magnitude'].reshape(-1, 1)
            y_val['magnitude'] = dataset['y_val']['magnitude'].reshape(-1, 1)

        if self.model_type == "volatility" or self.model_type == "combined":
            if 'volatility' in dataset['y_train']:
                y_train['volatility'] = dataset['y_train']['volatility'].reshape(-1, 1)
                y_val['volatility'] = dataset['y_val']['volatility'].reshape(-1, 1)
            else:
                # Use zeros if volatility not available
                y_train['volatility'] = np.zeros((len(dataset['y_train']['direction']), 1))
                y_val['volatility'] = np.zeros((len(dataset['y_val']['direction']), 1))

        # Define default callbacks if none provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor=f"val_{'direction_accuracy' if 'direction' in y_train else 'loss'}",
                    patience=10,
                    restore_best_weights=True,
                    mode='max' if 'direction' in y_train else 'min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]

        # Train the model
        self.history = self.model.fit(
            dataset['X_train'],
            y_train,
            validation_data=(dataset['X_val'], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history.history

    def predict(self, X_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() or load_model() first.")

        # Make predictions
        predictions = self.model.predict(X_data)

        # Return results as dictionary based on model type
        result = {}

        # Handle single vs multiple outputs
        if self.model_type == "combined":
            # Combined model with multiple outputs
            result['direction'] = predictions[0].flatten() if self.model_type in ["direction", "combined"] else None
            result['magnitude'] = predictions[1].flatten() if self.model_type in ["magnitude", "combined"] else None
            result['volatility'] = predictions[2].flatten() if self.model_type in ["volatility", "combined"] else None
        elif isinstance(predictions, list) and len(predictions) > 1:
            # Multiple outputs but not combined model type
            if self.model_type == "direction":
                result['direction'] = predictions[0].flatten()
            elif self.model_type == "magnitude":
                result['magnitude'] = predictions[0].flatten()
            elif self.model_type == "volatility":
                result['volatility'] = predictions[0].flatten()
        else:
            # Single output
            result[self.model_type] = predictions.flatten() if not isinstance(predictions, list) else predictions[
                0].flatten()

        return result

    def load_model(self, path: str) -> None:
        """Load model from file."""

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load model with custom objects so AttentionLayer is recognized
        self.model = load_model(
            path,
            compile=True
        )
