import logging
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, MetaData, text
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity


class DatabaseLogHandler(logging.Handler):
    def __init__(self, connection_string: str, table_name: str = "Logs", error_handler: Optional[ErrorHandler] = None):
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self.error_handler = error_handler

        try:
            self.engine = create_engine(connection_string)
            self._ensure_table_exists()
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("__init__"),
                    ErrorSeverity.HIGH,
                    reraise=False
                )
            print(f"Failed to initialize database log handler: {e}")

    def _get_error_context(self, operation: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build error context dictionary with class and operation info"""
        context = {
            "class": self.__class__.__name__,
            "operation": operation,
            "table_name": self.table_name,
        }

        if params:
            context.update(params)

        return context

    def _ensure_table_exists(self) -> None:
        try:
            # Check if table exists
            with self.engine.connect() as conn:
                # Create the table if it doesn't exist
                conn.execute(text(f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{self.table_name}' AND xtype='U')
                CREATE TABLE {self.table_name} (
                    LogID INT IDENTITY(1,1) PRIMARY KEY,
                    Timestamp DATETIME NOT NULL,
                    Level VARCHAR(20) NOT NULL,
                    Message NVARCHAR(4000) NOT NULL
                )
                """))
                conn.commit()

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("_ensure_table_exists"),
                    ErrorSeverity.HIGH,
                    reraise=False
                )
            print(f"Failed to create log table: {e}")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self.engine.connect() as conn:
                timestamp = datetime.fromtimestamp(record.created)
                level = record.levelname

                # Get the raw message without additional formatting
                message = record.getMessage()

                # Truncate message if it's too long for the database column
                if len(message) > 4000:
                    message = message[:3997] + "..."

                # Use parameterized query to avoid SQL injection
                insert_sql = text(
                    f"INSERT INTO {self.table_name} (Timestamp, Level, Message) VALUES (:timestamp, :level, :message)")

                conn.execute(insert_sql, {"timestamp": timestamp, "level": level, "message": message})
                conn.commit()

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("emit", {"record_level": record.levelname}),
                    ErrorSeverity.MEDIUM,
                    reraise=False
                )
            # Always print as a fallback
            print(f"Failed to write log to database: {e}")
            print(f"Log: [{record.levelname}] {record.getMessage()}")

    def clear_old_logs(self) -> None:
        try:
            with self.engine.connect() as conn:
                # Keep last 1000 logs to avoid table growing too large
                clear_sql = text(f"""
                    WITH OldLogs AS (
                        SELECT Timestamp,
                               ROW_NUMBER() OVER (ORDER BY Timestamp DESC) AS RowNum
                        FROM {self.table_name}
                    )
                    DELETE FROM {self.table_name}
                    WHERE Timestamp IN (
                        SELECT Timestamp FROM OldLogs WHERE RowNum > 1000
                    )
                """)

                conn.execute(clear_sql)
                conn.commit()

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("clear_old_logs"),
                    ErrorSeverity.LOW,
                    reraise=False
                )
            print(f"Failed to clear old logs: {e}")


class Logger:
    def __init__(
            self,
            name: str = 'app',
            level: int = logging.INFO,
            use_console: bool = True,
            console_level: Optional[int] = None,
            db_config: Optional[Dict[str, Any]] = None,
            error_handler: Optional[ErrorHandler] = None
    ) -> None:
        self.name = name
        self.level = level
        self.error_handler = error_handler

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers = []  # Clear any existing handlers

        # Configure console logging if requested
        if use_console:
            try:
                console_handler = logging.StreamHandler()
                # If console_level is provided, use it; otherwise use the global level
                handler_level = console_level if console_level is not None else level
                console_handler.setLevel(handler_level)
                formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
                console_handler.setFormatter(formatter)
                self._logger.addHandler(console_handler)
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_error(
                        e,
                        self._get_error_context("__init__", {"handler": "console"}),
                        ErrorSeverity.MEDIUM,
                        reraise=False
                    )
                print(f"Failed to initialize console logger: {e}")

        # Configure database logging if requested
        self.db_handler = None
        if db_config:
            try:
                connection_string = self._build_connection_string(db_config)
                self.db_handler = DatabaseLogHandler(connection_string, "Logs", error_handler)
                # Use a simple formatter that doesn't duplicate information
                db_formatter = logging.Formatter('%(message)s')
                self.db_handler.setFormatter(db_formatter)
                # Database handler always uses the global log level
                self.db_handler.setLevel(level)
                self._logger.addHandler(self.db_handler)

                # Clear old logs
                self.db_handler.clear_old_logs()
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_error(
                        e,
                        self._get_error_context("__init__", {"handler": "database"}),
                        ErrorSeverity.MEDIUM,
                        reraise=False
                    )
                print(f"Failed to initialize database logging: {e}")

    def _get_error_context(self, operation: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build error context dictionary with class and operation info"""
        context = {
            "class": self.__class__.__name__,
            "operation": operation,
            "logger_name": self.name,
            "level": logging.getLevelName(self.level)
        }

        if params:
            context.update(params)

        return context

    def _build_connection_string(self, config: Dict[str, Any]) -> str:
        try:
            return (f"mssql+pyodbc://{config['User']}:{config['Password']}@{config['Host']},{config['Port']}/"
                    f"{config['Database']}?driver=ODBC+Driver+17+for+SQL+Server")
        except KeyError as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("_build_connection_string", {"missing_key": str(e)}),
                    ErrorSeverity.HIGH,
                    reraise=True
                )
            raise
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("_build_connection_string"),
                    ErrorSeverity.HIGH,
                    reraise=True
                )
            raise

    def debug(self, msg: str, *args, **kwargs) -> None:
        try:
            self._logger.debug(msg, *args, **kwargs)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("debug", {"message": msg[:100] + "..." if len(msg) > 100 else msg}),
                    ErrorSeverity.LOW,
                    reraise=False
                )
            # Fallback to print in case logging itself fails
            print(f"[DEBUG] {msg}")

    def info(self, msg: str, *args, **kwargs) -> None:
        try:
            self._logger.info(msg, *args, **kwargs)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("info", {"message": msg[:100] + "..." if len(msg) > 100 else msg}),
                    ErrorSeverity.LOW,
                    reraise=False
                )
            print(f"[INFO] {msg}")

    def warning(self, msg: str, *args, **kwargs) -> None:
        try:
            self._logger.warning(msg, *args, **kwargs)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("warning", {"message": msg[:100] + "..." if len(msg) > 100 else msg}),
                    ErrorSeverity.LOW,
                    reraise=False
                )
            print(f"[WARNING] {msg}")

    def error(self, msg: str, *args, **kwargs) -> None:
        try:
            self._logger.error(msg, *args, **kwargs)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("error", {"message": msg[:100] + "..." if len(msg) > 100 else msg}),
                    ErrorSeverity.LOW,
                    reraise=False
                )
            print(f"[ERROR] {msg}")

    def critical(self, msg: str, *args, **kwargs) -> None:
        try:
            self._logger.critical(msg, *args, **kwargs)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    self._get_error_context("critical", {"message": msg[:100] + "..." if len(msg) > 100 else msg}),
                    ErrorSeverity.LOW,
                    reraise=False
                )
            print(f"[CRITICAL] {msg}")

    def get(self) -> logging.Logger:
        return self._logger