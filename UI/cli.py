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
        print("This will train ML models to predict market movements.")
        print()

        choices = [
            Choice("Train new model", "train_new"),
            Choice("Continue training existing model", "continue_training"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select a training option:',
            choices=choices
        ).ask() or 'back'

    def model_training_config_menu(self) -> Optional[Dict[str, Any]]:
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

        # 3. Configure sequence length
        sequence_length = questionary.text(
            'Enter sequence length (number of previous bars to use, default: 24):',
            default="24",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        if not sequence_length:
            return None

        # 4. Configure training parameters
        epochs = questionary.text(
            'Enter number of training epochs (default: 100):',
            default="100",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        batch_size = questionary.text(
            'Enter batch size (default: 32):',
            default="32",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        return {
            'pair': selected_pair,
            'timeframe': timeframe,
            'sequence_length': int(sequence_length),
            'epochs': int(epochs),
            'batch_size': int(batch_size)
        }

    def continue_training_menu(self) -> Optional[Dict[str, Any]]:
        """Menu for continuing model training."""
        # 1. Select model path
        model_dir = "TrainedModels"
        model_files = []

        # Check if directory exists
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.h5'):
                        model_files.append(os.path.join(root, file))

        if not model_files:
            print("No trained models found. Please train a model first.")
            return None

        model_path = questionary.select(
            'Select model to continue training:',
            choices=model_files
        ).ask()

        if not model_path:
            return None

        # 2. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None

        # 3. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None

        # 4. Configure sequence length
        sequence_length = questionary.text(
            'Enter sequence length (must match original model, default: 24):',
            default="24",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        if not sequence_length:
            return None

        # 5. Configure training parameters
        epochs = questionary.text(
            'Enter number of additional training epochs (default: 50):',
            default="50",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        batch_size = questionary.text(
            'Enter batch size (default: 32):',
            default="32",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        return {
            'model_path': model_path,
            'pair': selected_pair,
            'timeframe': timeframe,
            'sequence_length': int(sequence_length),
            'epochs': int(epochs),
            'batch_size': int(batch_size)
        }

    def backtest_menu(self) -> str:
        """Simplified menu for backtesting options."""
        choices = [
            Choice("Run Backtest with Current Configuration", "run_current"),
            Choice("Change Configuration", "change_config"),
            Choice("View Backtest Results", "view_results"),
            Choice("Go Back", "back")
        ]

        return questionary.select(
            'Select a backtesting option:',
            choices=choices
        ).ask() or 'back'

    def backtest_config_menu(self) -> Optional[Dict[str, Any]]:
        """Menu for configuring backtests."""
        # 1. Select model for backtesting
        model_dir = "TrainedModels"
        model_files = []

        # Check if directory exists
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.h5'):
                        model_files.append(os.path.join(root, file))

        if not model_files:
            print("No trained models found. Please train a model first.")
            return None

        model_path = questionary.select(
            'Select model for backtesting:',
            choices=model_files
        ).ask()

        if not model_path:
            return None

        # 2. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None

        # 3. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None

        # 4. Configure date range
        # Default to last 6 months
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        start_date = questionary.text(
            'Enter start date (YYYY-MM-DD):',
            default=start_date
        ).ask()

        if not start_date:
            return None

        end_date = questionary.text(
            'Enter end date (YYYY-MM-DD):',
            default=end_date
        ).ask()

        if not end_date:
            return None

        return {
            'model_path': model_path,
            'pair': selected_pair,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date
        }

    def select_backtest_results_menu(self) -> Optional[str]:
        """Menu for selecting backtest results to view."""
        results_dir = "BacktestResults"
        result_dirs = []

        # Check if directory exists
        if os.path.exists(results_dir):
            result_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

        if not result_dirs:
            print("No backtest results found. Please run a backtest first.")
            return None

        result_path = questionary.select(
            'Select backtest results to view:',
            choices=result_dirs
        ).ask()

        if not result_path:
            return None

        return os.path.join(results_dir, result_path)

    def live_trading_menu(self) -> str:
        """Menu for live trading options."""
        print("Live Trading Options:")
        print("This will execute trades on your MetaTrader account based on ML predictions.")
        print()

        choices = [
            Choice("Start trading", "start_trading"),
            Choice("Monitor active trades", "monitor_trades"),
            Choice("Stop trading", "stop_trading"),
            Choice("Go back", "back")
        ]

        return questionary.select(
            'Select a live trading option:',
            choices=choices
        ).ask() or 'back'

    def trading_config_menu(self) -> Optional[Dict[str, Any]]:
        """Menu for configuring live trading."""
        # Display warning
        print("\n⚠️ WARNING: Live trading will execute real trades on your MetaTrader account!")
        confirm = questionary.confirm("Are you sure you want to continue?").ask()

        if not confirm:
            return None

        # 1. Select model for trading
        model_dir = "TrainedModels"
        model_files = []

        # Check if directory exists
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.h5'):
                        model_files.append(os.path.join(root, file))

        if not model_files:
            print("No trained models found. Please train a model first.")
            return None

        model_path = questionary.select(
            'Select model for trading:',
            choices=model_files
        ).ask()

        if not model_path:
            return None

        # 2. Select currency pair
        pairs = [CurrencyPairs.XAUUSD, CurrencyPairs.USDJPY, CurrencyPairs.EURUSD, CurrencyPairs.GBPUSD]
        pair_choices = [Choice(CurrencyPairs.display_name(p), p) for p in pairs]

        selected_pair = questionary.select(
            'Select currency pair:',
            choices=pair_choices
        ).ask()

        if not selected_pair:
            return None

        # 3. Select timeframe
        timeframes = [tf.value for tf in TimeFrames]
        timeframe = questionary.select(
            'Select timeframe:',
            choices=timeframes
        ).ask()

        if not timeframe:
            return None

        # 4. Configure update interval (in minutes)
        update_interval = questionary.text(
            'Enter update interval in minutes (default: 15):',
            default="15",
            validate=lambda text: text.isdigit() and int(text) > 0
        ).ask()

        if not update_interval:
            return None

        # Final confirmation
        print("\nTrading Configuration Summary:")
        print(f"• Model: {model_path}")
        print(f"• Currency Pair: {CurrencyPairs.display_name(selected_pair)}")
        print(f"• Timeframe: {timeframe}")
        print(f"• Update Interval: {update_interval} minutes")

        final_confirm = questionary.confirm("Start trading with these settings?").ask()

        if not final_confirm:
            return None

        return {
            'model_path': model_path,
            'pair': selected_pair,
            'timeframe': timeframe,
            'update_interval': int(update_interval)
        }