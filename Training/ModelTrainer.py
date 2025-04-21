import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

from Models.LSTMModel import LSTMModel
from Training.DataPreprocessor import DataPreprocessor


class LossHistory(tf.keras.callbacks.Callback):
    """Custom callback to track training metrics and trading performance."""

    def __init__(self, validation_data: Tuple, output_dir: str = 'model_training'):
        super().__init__()
        self.validation_data = validation_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize history tracking
        self.training_history = {
            'loss': [],
            'direction_accuracy': [],
            'magnitude_mae': [],
            'volatility_mae': [],
            'val_loss': [],
            'val_direction_accuracy': [],
            'val_magnitude_mae': [],
            'val_volatility_mae': [],
            'trading_metrics': []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Store standard metrics
        for key in logs:
            if key in self.training_history:
                self.training_history[key].append(logs[key])

        # Calculate trading-specific metrics every 5 epochs
        if epoch % 5 == 0:
            trading_metrics = self.calculate_trading_metrics()
            self.training_history['trading_metrics'].append(trading_metrics)

            # Print trading metrics
            print(f"\nTrading metrics at epoch {epoch}:")
            print(f"Win rate: {trading_metrics['win_rate']:.2f}, "
                  f"Profit factor: {trading_metrics['profit_factor']:.2f}, "
                  f"Expected return: {trading_metrics['expected_return']:.4f}")

    def calculate_trading_metrics(self) -> Dict[str, float]:
        """Calculate trading-specific performance metrics on validation data."""
        # Get validation data
        X_val, y_val_dict = self.validation_data

        # Make predictions
        predictions = self.model.predict(X_val)

        # Get direction predictions (binary)
        direction_pred = (predictions[0] > 0.5).astype(int).flatten()
        direction_true = y_val_dict['direction'].astype(int)

        # Get magnitude predictions
        magnitude_pred = predictions[1].flatten()
        magnitude_true = y_val_dict['magnitude']

        # Calculate metrics

        # 1. Win rate (accuracy for direction)
        win_rate = np.mean(direction_pred == direction_true)

        # 2. Profit factor (sum of winning trades / sum of losing trades)
        # Convert predictions to PnL values
        pnl = []
        for i in range(len(direction_pred)):
            # If prediction matches reality, win the magnitude, otherwise lose it
            if direction_pred[i] == direction_true[i]:
                pnl.append(abs(magnitude_true[i]))  # Win the magnitude
            else:
                pnl.append(-abs(magnitude_true[i]))  # Lose the magnitude

        pnl = np.array(pnl)
        winning_trades = pnl[pnl > 0]
        losing_trades = pnl[pnl < 0]

        # Avoid division by zero
        profit_factor = (np.sum(winning_trades) / abs(np.sum(losing_trades))) if len(losing_trades) > 0 and np.sum(
            losing_trades) != 0 else 0

        # 3. Expected return per trade
        expected_return = np.mean(pnl) if len(pnl) > 0 else 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expected_return': expected_return
        }

    def plot_history(self, save_path: str = None) -> None:
        """Plot training history and metrics."""
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # 1. Loss plot
        axes[0].plot(self.training_history['loss'], label='Train Loss')
        axes[0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()

        # 2. Accuracy plot
        axes[1].plot(self.training_history['direction_accuracy'], label='Train Accuracy')
        axes[1].plot(self.training_history['val_direction_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Direction Prediction Accuracy')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

        # 3. Trading metrics
        if self.training_history['trading_metrics']:
            epochs = list(range(0, len(self.training_history['loss']), 5))[
                     :len(self.training_history['trading_metrics'])]
            win_rates = [tm['win_rate'] for tm in self.training_history['trading_metrics']]
            profit_factors = [tm['profit_factor'] for tm in self.training_history['trading_metrics']]
            expected_returns = [tm['expected_return'] for tm in self.training_history['trading_metrics']]

            ax3 = axes[2]
            ax3.plot(epochs, win_rates, 'g-', label='Win Rate')
            ax3.set_ylabel('Win Rate')
            ax3.set_xlabel('Epoch')
            ax3.legend(loc='upper left')

            ax4 = ax3.twinx()
            ax4.plot(epochs, profit_factors, 'r-', label='Profit Factor')
            ax4.set_ylabel('Profit Factor')
            ax4.legend(loc='upper right')

            ax3.set_title('Trading Performance Metrics')

        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.output_dir, 'training_history.png'))

        plt.close()


class ModelTrainer:
    """Handles the complete model training workflow."""

    def __init__(self, config, logger, data_preprocessor: DataPreprocessor, model: LSTMModel = None):
        """Initialize the model trainer.

        Args:
            config: Application configuration
            logger: Logger instance
            data_preprocessor: Preprocessor for data preparation
            model: Optional pre-built LSTM model
        """
        self.config = config
        self.logger = logger
        self.data_preprocessor = data_preprocessor
        self.model = model

        # Setup output directory for models and visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"ModelTraining_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Training history
        self.history = None
        self.dataset = None

    def prepare_training_data(self, pair: str = "XAUUSD", timeframe: str = "H1",
                              dataset_type: str = "training", sequence_length: int = 24) -> Dict[str, Any]:
        """Prepare data for model training.

        Args:
            pair: Currency pair
            timeframe: Timeframe (M15, H1, etc.)
            dataset_type: Dataset type (training, validation, testing)
            sequence_length: Sequence length for LSTM

        Returns:
            Dictionary with prepared dataset
        """
        try:
            self.logger.info(f"Preparing training data for {pair} {timeframe}")

            # Use data preprocessor to prepare the dataset
            dataset = self.data_preprocessor.prepare_dataset(
                pair=pair,
                timeframe=timeframe,
                dataset_type=dataset_type,
                sequence_length=sequence_length
            )

            if not dataset:
                self.logger.error("Failed to prepare dataset")
                return {}

            self.dataset = dataset

            # Log dataset information
            self.logger.info(f"Dataset prepared with {len(dataset['X_train'])} training samples")
            self.logger.info(f"Features used: {dataset['feature_names']}")

            return dataset

        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise

    def train_model(self, epochs: int = 100, batch_size: int = 32,
                    continued_training: bool = False) -> Dict[str, Any]:
        """Train the LSTM model.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            continued_training: Whether to continue training an existing model

        Returns:
            Training history
        """
        try:
            if self.dataset is None:
                self.logger.error("No dataset prepared. Call prepare_training_data first.")
                return {}

            # Initialize model if not provided
            if self.model is None or not continued_training:
                # Get shape information
                sequence_length = self.dataset['X_train'].shape[1]
                n_features = self.dataset['X_train'].shape[2]

                self.logger.info(f"Building new model with sequence_length={sequence_length}, features={n_features}")
                self.model = LSTMModel(self.config, (sequence_length, n_features), n_features)
                self.model.build_model()

            # Format validation data for callbacks
            X_val = self.dataset['X_val']
            y_val = self.dataset['y_val']

            # Setup callbacks
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_model.h5'),
                monitor='val_direction_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_direction_accuracy',
                patience=15,
                restore_best_weights=True,
                mode='max'
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1
            )

            trading_metrics = LossHistory(
                validation_data=(X_val, y_val),
                output_dir=self.output_dir
            )

            callbacks = [
                model_checkpoint,
                early_stopping,
                reduce_lr,
                trading_metrics
            ]

            # Train the model
            self.logger.info(f"Starting model training with {epochs} epochs, batch_size={batch_size}")

            history = self.model.fit(
                self.dataset,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )

            self.history = history

            # Save the final model
            self.model.save_model(os.path.join(self.output_dir, 'final_model.h5'))

            # Plot training history
            trading_metrics.plot_history()

            self.logger.info("Model training completed")
            return history

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise

    def evaluate_model(self, test_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_data: Optional test data to use instead of the test split from dataset

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if self.model is None:
                self.logger.error("No model available. Train or load a model first.")
                return {}

            # Use test data from dataset if not provided
            if test_data is None:
                if self.dataset is None:
                    self.logger.error("No dataset prepared. Call prepare_training_data first.")
                    return {}

                test_data = {
                    'X_test': self.dataset['X_test'],
                    'y_test': self.dataset['y_test']
                }

            # Evaluate the model
            self.logger.info("Evaluating model on test data")
            metrics = self.model.evaluate(test_data['X_test'], test_data['y_test'])

            # Create mapping from Keras metric names to expected names
            keras_metrics = dict(zip(self.model.model.metrics_names, metrics))

            # Extract specific metrics we need for display
            result_metrics = {'loss': keras_metrics.get('loss', 0.0),
                              'direction_accuracy': keras_metrics.get('direction_accuracy', 0.0),
                              'magnitude_mae': keras_metrics.get('magnitude_mae', 0.0),
                              'volatility_mae': keras_metrics.get('volatility_mae', 0.0)}

            # Calculate trading-specific metrics
            pred = self.model.predict(test_data['X_test'])

            # Direction accuracy
            direction_pred = (pred['direction'] > 0.5).astype(int)
            direction_true = test_data['y_test']['direction'].astype(int)
            win_rate = np.mean(direction_pred == direction_true)

            # Create PnL array
            pnl = []
            for i in range(len(direction_pred)):
                if direction_pred[i] == direction_true[i]:
                    pnl.append(abs(test_data['y_test']['magnitude'][i]))
                else:
                    pnl.append(-abs(test_data['y_test']['magnitude'][i]))

            pnl = np.array(pnl)
            winning_trades = pnl[pnl > 0]
            losing_trades = pnl[pnl < 0]

            # Avoid division by zero
            profit_factor = (np.sum(winning_trades) / abs(np.sum(losing_trades))) if len(losing_trades) > 0 and np.sum(
                losing_trades) != 0 else 0

            # Expected return per trade
            expected_return = np.mean(pnl) if len(pnl) > 0 else 0

            # Add trading metrics to the evaluation results
            result_metrics.update({
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expected_return': expected_return
            })

            # Generate evaluation plots
            self._plot_confusion_matrix(direction_true, direction_pred)
            self._plot_pnl_distribution(pnl)

            self.logger.info(
                f"Model evaluation completed with win rate: {win_rate:.2f}, profit factor: {profit_factor:.2f}")
            return result_metrics

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix for direction prediction."""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Direction')
        plt.xlabel('Predicted Direction')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

    def _plot_pnl_distribution(self, pnl: np.ndarray) -> None:
        """Plot distribution of profit/loss."""
        plt.figure(figsize=(10, 6))
        plt.hist(pnl, bins=20, alpha=0.7, color='blue')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Profit/Loss Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_dir, 'pnl_distribution.png'))
        plt.close()

        # Plot cumulative PnL
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pnl), 'g-')
        plt.title('Cumulative Profit/Loss')
        plt.xlabel('Trade #')
        plt.ylabel('Cumulative PnL')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'cumulative_pnl.png'))
        plt.close()

    def save_model(self, path: str = None) -> None:
        """Save the model to file."""
        if self.model is None:
            self.logger.error("No model to save")
            return

        if path is None:
            path = os.path.join(self.output_dir, 'final_model.h5')

        self.model.save_model(path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a model from file."""
        try:
            if not os.path.exists(path):
                self.logger.error(f"Model file not found: {path}")
                return

            # Get the input shape from saved model
            temp_model = tf.keras.models.load_model(path, compile=False)
            input_shape = temp_model.input_shape[1:]  # Remove batch dimension
            n_features = input_shape[1]

            # Create a new LSTM model with the correct shape
            self.model = LSTMModel(self.config, input_shape, n_features)

            # Load the saved weights
            self.model.load_model(path)

            self.logger.info(f"Model loaded from {path} with input shape {input_shape}")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
