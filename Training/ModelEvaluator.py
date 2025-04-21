from typing import Dict, Any, Tuple, List, Optional, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)


class ModelEvaluator:
    def __init__(self, logger, output_dir: str = None):
        self.logger = logger
        self.output_dir = output_dir or "ModelEvaluation"
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self, model_type: str, predictions: Dict[str, np.ndarray],
                 true_values: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance based on its type"""
        if model_type == "direction":
            return self.evaluate_direction_model(predictions.get('direction'),
                                                 true_values.get('direction'))
        elif model_type == "magnitude":
            return self.evaluate_magnitude_model(predictions.get('magnitude'),
                                                 true_values.get('magnitude'))
        elif model_type == "combined":
            metrics = {}
            # Evaluate direction component
            if 'direction' in predictions and 'direction' in true_values:
                direction_metrics = self.evaluate_direction_model(
                    predictions['direction'], true_values['direction'],
                    prefix="direction_"
                )
                metrics.update(direction_metrics)

            # Evaluate magnitude component
            if 'magnitude' in predictions and 'magnitude' in true_values:
                magnitude_metrics = self.evaluate_magnitude_model(
                    predictions['magnitude'], true_values['magnitude'],
                    prefix="magnitude_"
                )
                metrics.update(magnitude_metrics)

            return metrics
        else:
            self.logger.error(f"Unknown model type: {model_type}")
            return {}

    def evaluate_direction_model(self, predictions: np.ndarray, true_values: np.ndarray,
                                 prefix: str = "") -> Dict[str, float]:
        """Evaluate a direction prediction model (classification)"""
        if predictions is None or true_values is None:
            self.logger.error("Missing predictions or true values for direction evaluation")
            return {}

        # Ensure we have binary predictions (0 or 1)
        pred_binary = (predictions > 0.5).astype(int)
        true_binary = true_values.astype(int)

        metrics = {
            f"{prefix}accuracy": accuracy_score(true_binary, pred_binary),
            f"{prefix}precision": precision_score(true_binary, pred_binary, zero_division=0),
            f"{prefix}recall": recall_score(true_binary, pred_binary, zero_division=0),
            f"{prefix}f1_score": f1_score(true_binary, pred_binary, zero_division=0),
        }

        # Plot confusion matrix
        if self.output_dir:
            self._plot_confusion_matrix(true_binary, pred_binary,
                                        os.path.join(self.output_dir, f"{prefix}confusion_matrix.png"))

        return metrics

    def evaluate_magnitude_model(self, predictions: np.ndarray, true_values: np.ndarray,
                                 prefix: str = "") -> Dict[str, float]:
        """Evaluate a magnitude prediction model (regression)"""
        if predictions is None or true_values is None:
            self.logger.error("Missing predictions or true values for magnitude evaluation")
            return {}

        # Ensure arrays are flattened
        pred_flat = predictions.flatten()
        true_flat = true_values.flatten()

        # Calculate regression metrics
        metrics = {
            f"{prefix}mae": mean_absolute_error(true_flat, pred_flat),
            f"{prefix}rmse": np.sqrt(mean_squared_error(true_flat, pred_flat)),
            f"{prefix}r2": r2_score(true_flat, pred_flat),
        }

        # Plot prediction vs actual
        if self.output_dir:
            self._plot_prediction_scatter(true_flat, pred_flat,
                                          os.path.join(self.output_dir, f"{prefix}prediction_scatter.png"))
            self._plot_error_distribution(true_flat - pred_flat,
                                          os.path.join(self.output_dir, f"{prefix}error_distribution.png"))

        return metrics

    def evaluate_trading_performance(self, predictions: Dict[str, np.ndarray],
                                     true_values: Dict[str, np.ndarray],
                                     output_path: str = None) -> Dict[str, float]:
        """Optional: Evaluate trading performance metrics"""
        if 'direction' not in predictions or 'magnitude' not in true_values:
            self.logger.error("Missing required predictions for trading evaluation")
            return {}

        # Convert to binary predictions
        direction_pred = (predictions['direction'] > 0.5).astype(int)
        direction_true = true_values['direction'].astype(int)

        # Win rate (accuracy for direction)
        win_rate = accuracy_score(direction_true, direction_pred)

        # Calculate PnL based on direction prediction and actual magnitude
        pnl = []
        for i in range(len(direction_pred)):
            if direction_pred[i] == direction_true[i]:
                # Win: gain the actual magnitude
                pnl.append(abs(true_values['magnitude'][i]))
            else:
                # Loss: lose the actual magnitude
                pnl.append(-abs(true_values['magnitude'][i]))

        pnl = np.array(pnl)
        winning_trades = pnl[pnl > 0]
        losing_trades = pnl[pnl < 0]

        # Profit factor (total wins / total losses)
        # Avoid division by zero
        profit_factor = 0
        if len(losing_trades) > 0 and np.sum(np.abs(losing_trades)) > 0:
            profit_factor = np.sum(winning_trades) / np.sum(np.abs(losing_trades))

        # Expected return per trade
        expected_return = np.mean(pnl) if len(pnl) > 0 else 0

        # Plot PnL analysis
        if output_path or self.output_dir:
            save_path = output_path or os.path.join(self.output_dir, "trading_pnl.png")
            self._plot_trading_pnl(pnl, save_path)

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expected_return": expected_return,
            "total_trades": len(pnl),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
        }

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
        """Plot confusion matrix for classification results"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()

    def _plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
        """Plot scatter of predicted vs actual values"""
        plt.figure(figsize=(8, 8))

        # Plot scatter
        plt.scatter(y_true, y_pred, alpha=0.5)

        # Plot perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    def _plot_error_distribution(self, errors: np.ndarray, save_path: str) -> None:
        """Plot distribution of prediction errors"""
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    def _plot_trading_pnl(self, pnl: np.ndarray, save_path: str) -> None:
        """Plot trading PnL analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot 1: PnL distribution
        sns.histplot(pnl, kde=True, ax=ax1)
        ax1.axvline(x=0, color='r', linestyle='--')
        ax1.set_title('Trade PnL Distribution')
        ax1.set_xlabel('Profit/Loss')
        ax1.set_ylabel('Frequency')

        # Plot 2: Cumulative PnL
        ax2.plot(np.cumsum(pnl), 'g-')
        ax2.set_title('Cumulative Profit/Loss')
        ax2.set_xlabel('Trade #')
        ax2.set_ylabel('Cumulative PnL')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()