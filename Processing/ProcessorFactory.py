from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity
from Processing.DataProcessor import DataProcessor
from Processing.DataStorage import DataStorage
from typing import Dict, Any


class ProcessorFactory:
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

    def create_data_processor(self) -> DataProcessor:
        try:
            return DataProcessor(self.config, self.logger, self.error_handler)
        except Exception as e:
            context = {
                **self.error_context,
                "operation": "create_data_processor"
            }
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def create_data_storage(self) -> DataStorage:
        try:
            return DataStorage(self.config, self.logger, self.error_handler)
        except Exception as e:
            context = {
                **self.error_context,
                "operation": "create_data_storage"
            }
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise

    def create_feature_service(self):
        try:
            processor = self.create_data_processor()
            return processor  # This would be replaced with a dedicated feature service class in a real implementation
        except Exception as e:
            context = {
                **self.error_context,
                "operation": "create_feature_service"
            }
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise