import logging
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, MetaData, text


class DatabaseLogHandler(logging.Handler):
    def __init__(self, connection_string: str, table_name: str = "Logs"):
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self.engine = create_engine(connection_string)
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        try:
            metadata = MetaData()
            metadata.create_all(self.engine)

        except Exception as e:
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
            print(f"Failed to clear old logs: {e}")


class Logger:
    def __init__(
            self,
            name: str = 'app',
            level: int = logging.INFO,
            use_console: bool = True,
            console_level: int = None,
            db_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers = []  # Clear any existing handlers

        # Configure console logging if requested
        if use_console:
            console_handler = logging.StreamHandler()
            # If console_level is provided, use it; otherwise use the global level
            handler_level = console_level if console_level is not None else level
            console_handler.setLevel(handler_level)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # Configure database logging if requested
        self.db_handler = None
        if db_config:
            try:
                connection_string = self._build_connection_string(db_config)
                self.db_handler = DatabaseLogHandler(connection_string)
                # Use a simple formatter that doesn't duplicate information
                db_formatter = logging.Formatter('%(message)s')
                self.db_handler.setFormatter(db_formatter)
                # Database handler always uses the global log level
                self.db_handler.setLevel(level)
                self._logger.addHandler(self.db_handler)

                # Clear old logs
                self.db_handler.clear_old_logs()
            except Exception as e:
                print(f"Failed to initialize database logging: {e}")

    def _build_connection_string(self, config: Dict[str, Any]) -> str:
        return (f"mssql+pyodbc://{config['User']}:{config['Password']}@{config['Host']},{config['Port']}/"
                f"{config['Database']}?driver=ODBC+Driver+17+for+SQL+Server")

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def get(self) -> logging.Logger:
        return self._logger
