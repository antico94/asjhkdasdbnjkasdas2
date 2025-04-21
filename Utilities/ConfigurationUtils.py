import yaml
from pathlib import Path
import os
from typing import Any


class Config:
    def __init__(self, file_path: str = 'Configuration/Configuration.yaml') -> None:
        # Find the project root directory
        project_root = self._find_project_root()

        # Create an absolute path to the config file
        config_path = project_root / file_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        self._config = yaml.safe_load(config_path.read_text())

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get_nested(self, *keys: str) -> Any:
        value = self._config
        for key in keys:
            value = value[key]
        return value

    def get_sql_connection_string(self) -> str:
        """Build and return a SQL Server connection string from config."""
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

    def _find_project_root(self) -> Path:
        """Find the project root directory containing the Configuration folder."""
        current_dir = Path(os.getcwd())

        if (current_dir / 'Configuration').exists():
            return current_dir

        while True:
            if (current_dir / 'Configuration').exists():
                return current_dir

            parent_dir = current_dir.parent
            if parent_dir == current_dir:
                script_dir = Path(__file__).resolve().parent.parent
                if (script_dir / 'Configuration').exists():
                    return script_dir

                raise FileNotFoundError(
                    "Could not find project root containing Configuration directory. "
                    "Make sure the Configuration directory exists in the project."
                )

            current_dir = parent_dir
