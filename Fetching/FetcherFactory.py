from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Fetching.FetchData import MT5DataFetcher


class FetcherFactory:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def create_mt5_fetcher(self) -> MT5DataFetcher:
        return MT5DataFetcher(self.config, self.logger)
