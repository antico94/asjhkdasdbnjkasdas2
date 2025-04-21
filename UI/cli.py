import questionary
from questionary import Choice
from UI.Constants import AppMode
from Utilities.ConfigurationUtils import Config
from Configuration.Constants import TimeFrames, CurrencyPairs
from typing import Dict, List, Optional, Any
import os
from datetime import datetime, timedelta


class TradingBotCLI:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.main_choices = [
            Choice("Fetch Data", AppMode.FETCH_DATA.value),
            Choice("Process Data", AppMode.PROCESS_DATA.value),
            Choice("Analyze Features", AppMode.ANALYZE_FEATURES.value),
            Choice("Train Model", AppMode.TRAIN_MODEL.value),
            Choice("Backtest Strategy", AppMode.BACKTEST.value),
            Choice("Live Trading", AppMode.LIVE_TRADING.value),
            Choice("Exit", "exit")
        ]

    def main_menu(self) -> str:
        return questionary.select(
            'Select an action:',
            choices=self.main_choices
        ).ask() or 'exit'

    def display_fetch_config(self):
        fetch_config = self.config.get('FetchingSettings')
        pair_code = fetch_config.get('DefaultPair', 'XAUUSD')

        # Create a formatted display of current configuration
        config_display = [
            "Current Fetching Configuration:",
            f"• Currency Pair: {CurrencyPairs.display_name(pair_code)}",
            f"• Time Period: {fetch_config.get('DefaultTimeperiod', 2001)} days",
            f"• Timeframe: {fetch_config.get('DefaultTimeframe', 'H1')}",
            f"• Splitting Ratio: {fetch_config.get('SplittingRatio', {}).get('Training', 70)}% training, "
            f"{fetch_config.get('SplittingRatio', {}).get('Validation', 15)}% validation, "
            f"{fetch_config.get('SplittingRatio', {}).get('Testing', 15)}% testing"
        ]

        for line in config_display:
            print(line)

    def fetch_data_menu(self) -> str:
        self.display_fetch_config()
        print()  # Add an empty line for better readability

        choices = [
            Choice("Fetch data with current configuration", "fetch_current"),
            Choice("Change configuration", "change_config"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select an option:',
            choices=choices
        ).ask() or 'back'

    def change_config_menu(self) -> Optional[Dict[str, Any]]:
        fetch_config = self.config.get('FetchingSettings', {})

        # 1. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None

        # 2. Input days for time period
        default_days = fetch_config.get('DefaultTimeperiod', 2001)
        days = questionary.text(
            f'Enter number of days (default: {default_days}):',
            default=str(default_days),
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        if not days:
            return None

        # 3. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None

        # Return the selected configuration
        return {
            'pair': selected_pair,
            'days': int(days),
            'timeframe': timeframe
        }

    def process_data_menu(self) -> str:
        """Menu for data processing options."""
        print("Data Processing Options:")
        print("This will add technical indicators and prepare data for machine learning.")
        print()

        choices = [
            Choice("Process all datasets (training, validation, testing)", "process_all"),
            Choice("Process specific dataset", "process_specific"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select a processing option:',
            choices=choices
        ).ask() or 'back'

    def select_dataset_menu(self) -> Optional[Dict[str, str]]:
        """Menu for selecting specific dataset to process."""
        # 1. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None

        # 2. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None

        # 3. Select dataset type
        dataset_types = ["training", "validation", "testing"]
        dataset_type = questionary.select(
            'Select dataset type:',
            choices=dataset_types
        ).ask()

        if not dataset_type:
            return None

        return {
            'pair': selected_pair,
            'timeframe': timeframe,
            'dataset_type': dataset_type
        }

    def analyze_features_menu(self) -> str:
        """Menu for feature analysis options."""
        print("Feature Analysis Options:")
        print("This will analyze feature importance and select optimal features for ML models.")
        print()

        choices = [
            Choice("Run feature analysis with default settings", "run_analysis"),
            Choice("Run analysis with custom settings", "custom_analysis"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select an analysis option:',
            choices=choices
        ).ask() or 'back'

    def feature_analysis_config_menu(self) -> Optional[Dict[str, Any]]:
        """Menu for configuring feature analysis."""
        # 1. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None

        # 2. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None

        # 3. Select target variable
        target_options = [
            Choice("Price Prediction (1 period ahead)", "future_price_1"),
            Choice("Price Prediction (3 periods ahead)", "future_price_3"),
            Choice("Price Prediction (5 periods ahead)", "future_price_5"),
            Choice("Direction Prediction (1 period ahead)", "direction_1"),
            Choice("Direction Prediction (3 periods ahead)", "direction_3"),
            Choice("Direction Prediction (5 periods ahead)", "direction_5")
        ]

        target = questionary.select(
            'Select target variable for feature analysis:',
            choices=target_options
        ).ask()

        if not target:
            return None

        return {
            'pair': selected_pair,
            'timeframe': timeframe,
            'target': target
        }

    def train_model_menu(self) -> str:
        """Menu for model training options."""
        print("Model Training Options:")
        print("This will train LSTM models for price prediction based on processed data.")
        print()

        choices = [
            Choice("Train new model", "train_new"),
            Choice("Continue training existing model", "continue_training"),
            Choice("Test model performance", "test_model"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select a training option:',
            choices=choices
        ).ask() or 'back'

    def model_config_menu(self) -> Optional[Dict[str, Any]]:
        """Menu for configuring model training."""
        # 1. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None

        # 2. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None

        # 3. Select model type
        model_types = [
            Choice("Direction prediction", "direction"),
            Choice("Price magnitude prediction", "magnitude")
        ]
        model_type = questionary.select(
            'Select model type:',
            choices=model_types
        ).ask()

        if not model_type:
            return None

        # 4. Set training parameters
        epochs = questionary.text(
            'Number of epochs (default: 50):',
            default="50",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        batch_size = questionary.text(
            'Batch size (default: 32):',
            default="32",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        return {
            'pair': selected_pair,
            'timeframe': timeframe,
            'model_type': model_type,
            'epochs': int(epochs),
            'batch_size': int(batch_size)
        }

    def select_model_menu(self) -> Optional[str]:
        """Menu for selecting an existing model."""
        models_dir = "TrainedModels"

        if not os.path.exists(models_dir):
            print("No trained models found.")
            return None

        # Get directories representing model runs
        model_dirs = [d for d in os.listdir(models_dir)
                      if os.path.isdir(os.path.join(models_dir, d))]

        if not model_dirs:
            print("No trained models found.")
            return None

        # Sort by most recent
        model_dirs.sort(reverse=True)

        # Create choices with some metadata
        choices = []
        for model_dir in model_dirs:
            parts = model_dir.split('_')
            if len(parts) >= 4:
                pair = parts[0]
                timeframe = parts[1]
                model_type = parts[2]
                timestamp = '_'.join(parts[3:])
                display = f"{CurrencyPairs.display_name(pair)} - {timeframe} - {model_type} ({timestamp})"

                # Find model file
                model_files = [f for f in os.listdir(os.path.join(models_dir, model_dir))
                               if f.endswith('.h5')]
                if model_files:
                    model_path = os.path.join(models_dir, model_dir, model_files[0])
                    choices.append(Choice(display, model_path))

        choices.append(Choice("Go back", "back"))

        return questionary.select(
            'Select a model:',
            choices=choices
        ).ask() or 'back'