import MetaTrader5 as mt5
import pandas as pd
import pyodbc
from datetime import datetime, timedelta
from typing import Dict
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Configuration.Constants import TimeFrames, CurrencyPairs


class MT5DataFetcher:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.mt5_config = config.get('MetaTrader5', {})
        self.fetch_config = config.get('FetchingSettings', {})
        self.db_config = config.get('Database', {})
        self.connected = False

    def connect_to_mt5(self) -> bool:
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            return False

        # Connect with the provided credentials
        login = self.mt5_config.get('Login')
        password = self.mt5_config.get('Password')
        server = self.mt5_config.get('Server')

        # Check API documentation requirements
        try:
            if not mt5.login(login, password=password, server=server):
                error = mt5.last_error()
                self.logger.error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False
        except Exception as e:
            self.logger.error(f"MT5 login exception: {e}")
            mt5.shutdown()
            return False

        self.connected = True
        self.logger.info(f"Connected to MT5 server: {server}")
        return True

    def disconnect_from_mt5(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")

    def fetch_data(self, pair: str = None, days: int = None, timeframe: str = None) -> bool:
        # Use provided parameters or defaults from config
        pair = pair or self.fetch_config.get('DefaultPair', CurrencyPairs.XAUUSD)
        days = days or self.fetch_config.get('DefaultTimeperiod', 2001)
        timeframe = timeframe or self.fetch_config.get('DefaultTimeframe', TimeFrames.H1.value)

        self.logger.info(f"Fetching data for {pair}, {days} days, timeframe {timeframe}")
        print(f"Fetching {CurrencyPairs.display_name(pair)} data for {days} days with {timeframe} timeframe...")

        # Convert timeframe string to MT5 timeframe
        mt5_timeframe = self._get_mt5_timeframe(timeframe)
        if not mt5_timeframe:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            print(f"Error: Invalid timeframe {timeframe}")
            return False

        try:
            # Connect to MT5
            print("Connecting to MT5...")
            if not self.connect_to_mt5():
                return False

            # Calculate time range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)

            # Fetch rates
            self.logger.info(f"Fetching from {from_date} to {to_date}")
            print(f"Downloading market data from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}...")
            rates = mt5.copy_rates_range(pair, mt5_timeframe, from_date, to_date)

            if rates is None or len(rates) == 0:
                self.logger.error(f"No data received for {pair} with timeframe {timeframe}")
                print(f"Error: No data received from MT5")
                return False

            # Convert to DataFrame
            print(f"Processing {len(rates)} data points...")
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Sort by time ascending
            df = df.sort_values('time')

            # Split data according to ratio
            print("Splitting data into training, validation, and testing sets...")
            splits = self._split_data(df)

            # Store in database
            print("Storing data in database...")
            if self._store_data_in_db(pair, timeframe, splits):
                self.logger.info(f"Successfully fetched and stored data for {pair} {timeframe}")
                return True
            else:
                self.logger.error(f"Failed to store data for {pair} {timeframe}")
                return False

        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            print(f"Error: {str(e)}")
            return False
        finally:
            print("Disconnecting from MT5...")
            self.disconnect_from_mt5()

    def _get_mt5_timeframe(self, timeframe: str) -> int:
        timeframe_map = {
            TimeFrames.H1.value: mt5.TIMEFRAME_H1,
            TimeFrames.H4.value: mt5.TIMEFRAME_H4,
            TimeFrames.D1.value: mt5.TIMEFRAME_D1
        }
        return timeframe_map.get(timeframe.upper())

    def _split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Get splitting ratios
        ratios = self.fetch_config.get('SplittingRatio', {})
        training_ratio = ratios.get('Training', 70) / 100
        validation_ratio = ratios.get('Validation', 15) / 100

        # Calculate split indices
        total_rows = len(df)
        training_end = int(total_rows * training_ratio)
        validation_end = training_end + int(total_rows * validation_ratio)

        # Split dataframe
        training_df = df.iloc[:training_end]
        validation_df = df.iloc[training_end:validation_end]
        testing_df = df.iloc[validation_end:]

        self.logger.info(f"Data split: Training: {len(training_df)}, "
                         f"Validation: {len(validation_df)}, Testing: {len(testing_df)}")

        return {
            'training': training_df,
            'validation': validation_df,
            'testing': testing_df
        }

    def _build_connection_string(self) -> str:
        return (f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.db_config['Host']},"
                f"{self.db_config['Port']};DATABASE={self.db_config['Database']};"
                f"UID={self.db_config['User']};PWD={self.db_config['Password']}")

    def _store_data_in_db(self, pair: str, timeframe: str, data_splits: Dict[str, pd.DataFrame]) -> bool:
        try:
            conn_str = self._build_connection_string()
            print(f"Connecting to database...")
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()

                # Clean up existing tables first
                print(f"Removing existing data tables if present...")
                self._clean_existing_tables(cursor, pair, timeframe)

                # Create and populate tables for each split
                total_rows = 0
                for split_name, df in data_splits.items():
                    table_name = f"{pair}_{timeframe.lower()}_{split_name}"

                    # Create table
                    print(f"Creating table {table_name}...")
                    self._create_table(cursor, table_name)

                    # Insert data
                    print(f"Inserting {len(df)} rows into {table_name}...")
                    row_count = self._insert_data(cursor, table_name, df)
                    self.logger.info(f"Inserted {row_count} rows into {table_name}")
                    total_rows += row_count

                conn.commit()
                print(f"Database operation complete. {total_rows} total rows inserted.")
                return True

        except Exception as e:
            self.logger.error(f"Database error: {e}")
            print(f"Database error: {str(e)}")
            return False

    def _clean_existing_tables(self, cursor, pair: str, timeframe: str) -> None:
        split_types = ['training', 'validation', 'testing']
        for split in split_types:
            table_name = f"{pair}_{timeframe.lower()}_{split}"
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.logger.info(f"Dropped existing table {table_name}")
            except Exception as e:
                self.logger.warning(f"Failed to drop table {table_name}: {e}")

    def _create_table(self, cursor, table_name: str) -> None:
        cursor.execute(f"""
        CREATE TABLE {table_name} (
            [time] DATETIME NOT NULL PRIMARY KEY,
            [open] FLOAT NOT NULL,
            [high] FLOAT NOT NULL,
            [low] FLOAT NOT NULL,
            [close] FLOAT NOT NULL,
            tick_volume INT NOT NULL,
            spread INT NOT NULL,
            real_volume INT
        )
        """)

    def _insert_data(self, cursor, table_name: str, df: pd.DataFrame) -> int:
        row_count = 0
        for _, row in df.iterrows():
            cursor.execute(f"""
            INSERT INTO {table_name} ([time], [open], [high], [low], [close], tick_volume, spread, real_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['time'], row['open'], row['high'], row['low'],
                row['close'], row['tick_volume'], row['spread'],
                row.get('real_volume', 0)
            ))
            row_count += 1
        return row_count
