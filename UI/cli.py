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
        # Display current configuration
        train_config = self.config.get('TrainingSettings', {})
        pair_code = train_config.get('DefaultPair', 'XAUUSD')
        timeframe = train_config.get('DefaultTimeframe', 'H1')
        model_selection = train_config.get('ModelSelection', 'Both')

        # Create a formatted display of current configuration
        config_display = [
            "Current Training Configuration:",
            f"• Currency Pair: {CurrencyPairs.display_name(pair_code)}",
            f"• Timeframe: {timeframe}",
            f"• Models: {model_selection}"
        ]

        for line in config_display:
            print(line)
        print()  # Add empty line for better readability

        choices = [
            Choice("Start Training with current configuration", "train_current"),
            Choice("Change configuration", "change_config"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select an option:',
            choices=choices
        ).ask() or 'back'

    def change_training_config_menu(self) -> Optional[Dict[str, Any]]:
        """Menu for changing the training configuration."""
        train_config = self.config.get('TrainingSettings', {})
        result = {}

        # 1. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]

        # If there's only one currency pair, use it without asking
        if len(pairs) == 1:
            result['pair'] = pairs[0]
            print(f"Using currency pair: {CurrencyPairs.display_name(pairs[0])}")
        else:
            pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]
            selected_pair = questionary.select(
                'Select currency pair:',
                choices=pair_choices
            ).ask()

            if not selected_pair:
                return None
            result['pair'] = selected_pair

        # 2. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]

        # If there's only one timeframe, use it without asking
        if len(timeframes) == 1:
            result['timeframe'] = timeframes[0]
            print(f"Using timeframe: {timeframes[0]}")
        else:
            timeframe = questionary.select(
                'Select timeframe:',
                choices=timeframes
            ).ask()

            if not timeframe:
                return None
            result['timeframe'] = timeframe

        # 3. Select model type(s)
        model_choices = [
            Choice("Direction Model (Classification)", "direction"),
            Choice("Magnitude Model (Regression)", "magnitude"),
            Choice("Both Models", "both")
        ]

        selected_models = questionary.select(
            'Select model type to train:',
            choices=model_choices
        ).ask()

        if not selected_models:
            return None
        result['models'] = selected_models

        return result

    def backtest_menu(self) -> str:
        """Menu for backtesting options."""
        print("Backtesting Options:")
        print("This will test your trained models on unseen data to evaluate trading performance.")
        print()

        # Get backtest settings from config if available
        backtest_config = self.config.get('BacktestSettings', {})
        pair = backtest_config.get('DefaultPair', 'XAUUSD')
        timeframe = backtest_config.get('DefaultTimeframe', 'H1')

        # Show current config
        config_display = [
            "Current Backtest Configuration:",
            f"• Currency Pair: {CurrencyPairs.display_name(pair)}",
            f"• Timeframe: {timeframe}",
        ]

        for line in config_display:
            print(line)
        print()  # Add empty line for better readability

        choices = [
            Choice("Run backtest with current settings", "run_backtest"),
            Choice("Run backtest with optimization", "run_optimized"),
            Choice("Change backtest configuration", "change_config"),
            Choice("View previous backtest results", "view_results"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select a backtest option:',
            choices=choices
        ).ask() or 'back'

    def change_backtest_config_menu(self) -> Optional[Dict[str, Any]]:
        """Menu for changing the backtest configuration."""
        result = {}

        # 1. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None
        result['pair'] = selected_pair

        # 2. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None
        result['timeframe'] = timeframe

        # 3. Ask about parameter optimization
        result['optimize'] = questionary.confirm(
            'Do you want to perform parameter optimization?',
            default=False
        ).ask()

        if result['optimize'] is None:  # User pressed Ctrl+C
            return None

        return result

    def backtest_parameters_menu(self) -> Optional[Dict[str, Any]]:
        """Menu for configuring backtest strategy parameters."""
        result = {}

        # 1. Set take-profit ratio
        tp_ratio = questionary.text(
            'Set take-profit ratio (default: 2.0):',
            default="2.0",
            validate=lambda text: text.replace('.', '', 1).isdigit() and float(text) > 0
        ).ask()

        if not tp_ratio:
            return None
        result['tp_ratio'] = float(tp_ratio)

        # 2. Set stop-loss ratio
        sl_ratio = questionary.text(
            'Set stop-loss ratio (default: 1.0):',
            default="1.0",
            validate=lambda text: text.replace('.', '', 1).isdigit() and float(text) > 0
        ).ask()

        if not sl_ratio:
            return None
        result['sl_ratio'] = float(sl_ratio)

        # 3. Set minimum confidence threshold
        min_confidence = questionary.text(
            'Set minimum confidence threshold (default: 0.55):',
            default="0.55",
            validate=lambda text: text.replace('.', '', 1).isdigit() and 0 <= float(text) <= 1
        ).ask()

        if not min_confidence:
            return None
        result['min_confidence'] = float(min_confidence)

        # 4. Set volatility scaling factor
        volatility_scale = questionary.text(
            'Set volatility scaling factor (default: 1.0):',
            default="1.0",
            validate=lambda text: text.replace('.', '', 1).isdigit() and float(text) > 0
        ).ask()

        if not volatility_scale:
            return None
        result['volatility_scale'] = float(volatility_scale)

        return result

    def select_backtest_results_menu(self, results_list: List[str]) -> Optional[str]:
        """Menu for selecting backtest results to view."""
        if not results_list:
            print("No backtest results found.")
            return None

        choices = [Choice(result, result) for result in results_list]
        choices.append(Choice("Go back", "back"))

        selected = questionary.select(
            'Select backtest result to view:',
            choices=choices
        ).ask()

        if not selected or selected == "back":
            return None

        return selected