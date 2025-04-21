from enum import Enum

class AppMode(Enum):
    FETCH_DATA = "fetch_data"
    PROCESS_DATA = "process_data"
    VALIDATE_DATA = "validate_data"
    ANALYZE_FEATURES = "analyze_features"
    TRAIN_MODEL = "train_model"
    BACKTEST = "backtest"
    LIVE_TRADING = "live_trading"
