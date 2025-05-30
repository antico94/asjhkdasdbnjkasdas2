from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity
from Fetching.FetchData import MT5DataFetcher
from typing import Dict, Any


class FetcherFactory:
    def __init__(self, config: Config, logger: Logger, error_handler: ErrorHandler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler

    @property
    def error_context(self) -> Dict[str, Any]:
        """Base context for error handling in this class"""
        return {
            "class": self.__class__.__name__,
        }

    def create_mt5_fetcher(self) -> MT5DataFetcher:
        try:
            return MT5DataFetcher(self.config, self.logger, self.error_handler)
        except Exception as e:
            context = {
                **self.error_context,
                "operation": "create_mt5_fetcher"
            }
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise