import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from datetime import datetime

from Models.LSTMModel import LSTMModel
from Training.DataPreprocessor import DataPreprocessor


class LossHistory(tf.keras.callbacks.Callback):
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
    def __init__(self, config, logger, data_preprocessor: DataPreprocessor, model: LSTMModel = None):
        self.config = config
        self.logger = logger
        self.data_preprocessor = data_preprocessor
        self.model = model

        # Setup base output directory for models
        self.base_output_dir = "TrainedModels"
        os.makedirs(self.base_output_dir, exist_ok=True)

        # Instance-specific output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = None

        # Default pair and timeframe
        self.pair = None
        self.timeframe = None
        self.model_type = None

        # Training history
        self.history = None
        self.dataset = None

    def set_model_type(self, model_type: str) -> None:
        self.model_type = model_type
        self.logger.info(f"Set model type to {model_type}")

    def setup_output_dir(self) -> None:
        if not self.pair or not self.timeframe or not self.model_type:
            self.logger.error("Pair, timeframe, and model type must be set before creating output directory")
            raise ValueError("Missing required parameters: pair, timeframe, and model type must be set")

        self.output_dir = os.path.join(
            self.base_output_dir,
            f"{self.pair}_{self.timeframe}_{self.model_type}_{self.timestamp}"
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir}")

    def prepare_training_data(self, pair: str = None, timeframe: str = None,
                              dataset_type: str = "training", sequence_length: int = 24,
                              model_type: str = None) -> Dict[str, Any]:
        try:
            # Set pair and timeframe
            self.pair = pair or self.pair
            self.timeframe = timeframe or self.timeframe
            self.model_type = model_type or self.model_type or "direction"

            if not self.pair or not self.timeframe:
                self.logger.error("Pair and timeframe must be provided")
                raise ValueError("Missing required parameters: pair and timeframe")

            # Create output directory
            self.setup_output_dir()

            self.logger.info(f"Preparing training data for {self.pair} {self.timeframe}")

            # Use data preprocessor to prepare the dataset
            dataset = self.data_preprocessor.prepare_dataset(
                pair=self.pair,
                timeframe=self.timeframe,
                dataset_type=dataset_type,
                sequence_length=sequence_length,
                model_type=self.model_type
            )

            if not dataset:
                self.logger.error("Failed to prepare dataset")
                raise ValueError(f"Failed to prepare dataset for {self.pair} {self.timeframe} {self.model_type}")

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
        try:
            if self.dataset is None:
                self.logger.error("No dataset prepared. Call prepare_training_data first.")
                raise ValueError("No dataset available. Call prepare_training_data first.")

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
            self.save_model()

            # Plot training history
            trading_metrics.plot_history()

            self.logger.info("Model training completed")
            return history

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise

    def evaluate_model(self, test_data: Dict[str, Any] = None) -> Dict[str, float]:
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

            # Get predictions
            self.logger.info("Evaluating model on test data")
            predictions = self.model.predict(test_data['X_test'])

            # Import ModelEvaluator if not already available
            from path.to.model_evaluator import ModelEvaluator

            # Create evaluator
            evaluator = ModelEvaluator(self.logger, self.output_dir)

            # Evaluate based on model type
            metrics = evaluator.evaluate(
                self.model_type,
                predictions,
                test_data['y_test']
            )

            # Optionally calculate trading performance metrics
            if self.config.get('EvaluationSettings', {}).get('IncludeTradingMetrics', True):
                trading_metrics = evaluator.evaluate_trading_performance(
                    predictions,
                    test_data['y_test'],
                    os.path.join(self.output_dir, 'trading_performance.png')
                )

                # Add trading metrics to the evaluation results
                metrics.update({
                    'trading_win_rate': trading_metrics.get('win_rate', 0),
                    'trading_profit_factor': trading_metrics.get('profit_factor', 0),
                    'trading_expected_return': trading_metrics.get('expected_return', 0)
                })

            # Log evaluation results
            model_type_display = self.model_type.capitalize()
            primary_metric = metrics.get('accuracy', metrics.get('r2', 0))
            self.logger.info(
                f"{model_type_display} model evaluation complete with primary metric: {primary_metric:.4f}"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
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
        if self.model is None:
            self.logger.error("No model to save")
            return

        if not self.output_dir:
            self.setup_output_dir()

        if path is None:
            model_name = f"final_model.h5"
            if self.pair and self.timeframe and self.model_type:
                model_name = f"{self.pair}_{self.timeframe}_{self.model_type}_model.h5"
            path = os.path.join(self.output_dir, model_name)

        self.model.save_model(path)
        self.logger.info(f"Model saved to {path}")

        # Save model metadata
        self._save_model_metadata()

    def _save_model_metadata(self) -> None:
        metadata = {
            "pair": self.pair,
            "timeframe": self.timeframe,
            "model_type": self.model_type,
            "timestamp": self.timestamp,
            "features": self.dataset.get("feature_names", []),
            "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save metadata to JSON file
        import json
        metadata_path = os.path.join(self.output_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"Model metadata saved to {metadata_path}")

    def load_model(self, path: str) -> None:
        try:
            if not os.path.exists(path):
                self.logger.error(f"Model file not found: {path}")
                return

            # Extract model information from path if possible
            model_dir = os.path.dirname(path)
            model_filename = os.path.basename(path)

            # Try to extract pair, timeframe and model_type from directory name
            dir_name = os.path.basename(model_dir)
            parts = dir_name.split('_')
            if len(parts) >= 3:
                self.pair = parts[0]
                self.timeframe = parts[1]
                self.model_type = parts[2]
                self.logger.info(f"Extracted model info: {self.pair} {self.timeframe} {self.model_type}")

            # Get the input shape from saved model
            temp_model = tf.keras.models.load_model(path, compile=False,
                                                    custom_objects={'AttentionLayer': LSTMModel.AttentionLayer})
            input_shape = temp_model.input_shape[1:]  # Remove batch dimension
            n_features = input_shape[1]

            # Create a new LSTM model with the correct shape
            self.model = LSTMModel(self.config, input_shape, n_features)

            # Load the saved weights
            self.model.load_model(path)

            # Load metadata if available
            metadata_path = os.path.join(model_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                self.pair = metadata.get("pair", self.pair)
                self.timeframe = metadata.get("timeframe", self.timeframe)
                self.model_type = metadata.get("model_type", self.model_type)

                self.logger.info(f"Loaded model metadata: {self.pair} {self.timeframe} {self.model_type}")

            self.logger.info(f"Model loaded from {path} with input shape {input_shape}")
            self.setup_output_dir()  # Set up new output directory for this instance

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise