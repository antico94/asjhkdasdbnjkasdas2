import os
from typing import Dict, Any
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity


class PathResolver:
    def __init__(self, config, error_handler: ErrorHandler):
        self.config = config
        self.error_handler = error_handler
        self.project_root = self._find_project_root()

    @property
    def error_context(self) -> Dict[str, Any]:
        """Base context for error handling in this class"""
        return {
            "class": self.__class__.__name__,
            "project_root": self.project_root if hasattr(self, 'project_root') else 'unknown'
        }

    def resolve_path(self, relative_path):
        """Convert a relative path to an absolute path from project root"""
        try:
            return os.path.join(self.project_root, relative_path)
        except Exception as e:
            context = {
                **self.error_context,
                "operation": "resolve_path",
                "relative_path": relative_path
            }
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.MEDIUM,
                reraise=True
            )
            raise

    def _find_project_root(self):
        """Find the project root directory"""
        try:
            return os.getcwd()
        except Exception as e:
            context = {
                "class": self.__class__.__name__,
                "operation": "_find_project_root"
            }
            self.error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=True
            )
            raise