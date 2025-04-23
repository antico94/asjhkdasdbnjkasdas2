import yaml
from pathlib import Path
import os
from typing import Any, Dict, Optional
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity


class Config:
    def __init__(self, file_path: str = 'Configuration/Configuration.yaml',
                 error_handler: Optional[ErrorHandler] = None) -> None:
        # Store error handler if provided (will be None during initial container setup)
        self.error_handler = error_handler

        # Find the project root directory
        self.project_root = self._find_project_root()

        # Create an absolute path to the config file
        config_path = self.project_root / file_path

        try:
            if not config_path.exists():
                error_msg = f"Configuration file not found at {config_path}"
                if self.error_handler:
                    self.error_handler.handle_error(
                        FileNotFoundError(error_msg),
                        self._get_error_context("__init__", {"file_path": str(file_path)}),
                        ErrorSeverity.FATAL,
                        reraise=True
                    )
                raise FileNotFoundError(error_msg)

            self._config = yaml.safe_load(config_path.read_text())

        except (yaml.YAMLError, IOError) as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("__init__", {"file_path": str(file_path)}),
                    ErrorSeverity.FATAL,
                    reraise=True
                )
            raise

    def _get_error_context(self, operation: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build error context dictionary with class and operation info"""
        context = {
            "class": self.__class__.__name__,
            "operation": operation,
        }

        if hasattr(self, 'project_root'):
            context["project_root"] = str(self.project_root)

        if params:
            context.update(params)

        return context

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._config.get(key, default)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("get", {"key": key}),
                    ErrorSeverity.HIGH,
                    reraise=True
                )
            raise

    def __getitem__(self, key: str) -> Any:
        try:
            return self._config[key]
        except KeyError as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("__getitem__", {"key": key}),
                    ErrorSeverity.HIGH,
                    reraise=True
                )
            raise

    def get_nested(self, *keys: str) -> Any:
        try:
            value = self._config
            for i, key in enumerate(keys):
                try:
                    value = value[key]
                except (KeyError, TypeError) as e:
                    # More specific error message about which level of nesting failed
                    if self.error_handler:
                        context = self._get_error_context(
                            "get_nested",
                            {
                                "keys": str(keys),
                                "failed_at": i,
                                "failed_key": key,
                                "parent_value_type": type(value).__name__
                            }
                        )
                        self.error_handler.handle_error(
                            KeyError(f"Failed to access key '{key}' at position {i} in nested keys {keys}"),
                            context,
                            ErrorSeverity.HIGH,
                            reraise=True
                        )
                    raise KeyError(f"Failed to access key '{key}' at position {i} in nested keys {keys}")
            return value
        except Exception as e:
            if self.error_handler and not isinstance(e, KeyError):
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("get_nested", {"keys": str(keys)}),
                    ErrorSeverity.HIGH,
                    reraise=True
                )
            raise

    def get_sql_connection_string(self) -> str:
        """Build and return a SQL Server connection string from config."""
        try:
            db_config = self.get('Database', {})
            host = db_config.get('Host', 'localhost')
            port = db_config.get('Port', 1433)
            user = db_config.get('User', 'sa')
            password = db_config.get('Password', '')
            database = db_config.get('Database', 'TestDB')

            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={host},{port};"
                f"DATABASE={database};"
                f"UID={user};PWD={password}"
            )

            return conn_str
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("get_sql_connection_string"),
                    ErrorSeverity.HIGH,
                    reraise=True
                )
            raise

    def _find_project_root(self) -> Path:
        """Find the project root directory containing the Configuration folder."""
        try:
            current_dir = Path(os.getcwd())

            if (current_dir / 'Configuration').exists():
                return current_dir

            while True:
                if (current_dir / 'Configuration').exists():
                    return current_dir

                parent_dir = current_dir.parent
                if parent_dir == current_dir:
                    # Try to find the configuration in the script directory
                    script_dir = Path(__file__).resolve().parent.parent
                    if (script_dir / 'Configuration').exists():
                        return script_dir

                    error_msg = (
                        "Could not find project root containing Configuration directory. "
                        "Make sure the Configuration directory exists in the project."
                    )

                    if self.error_handler:
                        self.error_handler.handle_error(
                            FileNotFoundError(error_msg),
                            self._get_error_context("_find_project_root"),
                            ErrorSeverity.FATAL,
                            reraise=True
                        )
                    raise FileNotFoundError(error_msg)

                current_dir = parent_dir

        except Exception as e:
            if self.error_handler and not isinstance(e, FileNotFoundError):
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("_find_project_root"),
                    ErrorSeverity.FATAL,
                    reraise=True
                )
            raise