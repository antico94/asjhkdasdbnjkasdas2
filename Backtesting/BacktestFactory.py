from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Backtesting.Backtest import BacktestManager


class BacktestFactory:
    def __init__(self, config: Config, logger: Logger, model_factory, data_storage, path_resolver):
        self.config = config
        self.logger = logger
        self.model_factory = model_factory
        self.data_storage = data_storage
        self.path_resolver = path_resolver

    def create_backtest_manager(self) -> BacktestManager:
        return BacktestManager(
            self.config,
            self.logger,
            self.model_factory,
            self.data_storage,
            self.path_resolver
        )