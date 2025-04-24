from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity
from Fetching.FetcherFactory import FetcherFactory
from Processing.ProcessorFactory import ProcessorFactory
from UI.cli import TradingBotCLI
import pandas as pd


def handle_fetch_data(cli: TradingBotCLI,
                      logger: Logger,
                      error_handler: ErrorHandler,
                      fetcher_factory: FetcherFactory) -> None:
    """Handle fetch data flow"""
    context = {
        "function": "handle_fetch_data",
        "module": __name__
    }

    try:
        while True:
            fetch_action = cli.fetch_data_menu()

            if fetch_action == "back":
                logger.info("Returning to main menu")
                break

            elif fetch_action == "fetch_current":
                logger.info("Fetching data with current configuration")
                print("Starting data fetch with current configuration...")

                fetcher = fetcher_factory.create_mt5_fetcher()
                success = fetcher.fetch_data()

                if success:
                    logger.info("Data fetching completed successfully")
                    print("✓ Data fetching completed successfully")
                else:
                    logger.error("Data fetching failed")
                    print("✗ Data fetching failed")

            elif fetch_action == "change_config":
                logger.info("Changing fetching configuration")
                new_config = cli.change_config_menu()

                if new_config:
                    logger.info(f"New configuration: {new_config}")
                    print(f"Starting data fetch with new configuration...")

                    fetcher = fetcher_factory.create_mt5_fetcher()
                    success = fetcher.fetch_data(
                        pair=new_config['pair'],
                        days=new_config['days'],
                        timeframe=new_config['timeframe']
                    )

                    if success:
                        logger.info("Data fetching with new config completed successfully")
                        print("✓ Data fetching completed successfully")
                    else:
                        logger.error("Data fetching with new config failed")
                        print("✗ Data fetching failed")
                else:
                    logger.info("Configuration change cancelled")
                    print("Configuration change cancelled")

    except Exception as e:
        error_handler.handle_error(
            exception=e,
            context=context,
            severity=ErrorSeverity.HIGH,
            reraise=False
        )
        print("An error occurred during data fetching. Returning to main menu.")


def handle_process_data(cli: TradingBotCLI,
                        logger: Logger,
                        error_handler: ErrorHandler,
                        processor_factory: ProcessorFactory,
                        fetcher_factory: FetcherFactory) -> None:
    """Handle data processing flow"""
    context = {
        "function": "handle_process_data",
        "module": __name__
    }

    try:
        while True:
            process_action = cli.process_data_menu()

            if process_action == "back":
                logger.info("Returning to main menu")
                break

            elif process_action == "process_all":
                logger.info("Processing all datasets")
                print("Starting to process all datasets...")

                processor = processor_factory.create_data_processor()
                storage = processor_factory.create_data_storage()

                # Process training, validation, and testing datasets for XAUUSD
                process_datasets(processor, storage, logger, error_handler, "XAUUSD", "H1",
                                 ["training", "validation", "testing"], fetcher_factory=fetcher_factory)

            elif process_action == "process_specific":
                logger.info("Processing specific dataset")
                dataset_config = cli.select_dataset_menu()

                if dataset_config:
                    logger.info(f"Selected dataset: {dataset_config}")
                    print(f"Processing {dataset_config['dataset_type']} data for "
                          f"{dataset_config['pair']} {dataset_config['timeframe']}...")

                    processor = processor_factory.create_data_processor()
                    storage = processor_factory.create_data_storage()

                    process_datasets(
                        processor,
                        storage,
                        logger,
                        error_handler,
                        dataset_config['pair'],
                        dataset_config['timeframe'],
                        [dataset_config['dataset_type']],
                        fetcher_factory=fetcher_factory
                    )
                else:
                    logger.info("Dataset selection cancelled")
                    print("Dataset selection cancelled")

    except Exception as e:
        error_handler.handle_error(
            exception=e,
            context=context,
            severity=ErrorSeverity.HIGH,
            reraise=False
        )
        print("An error occurred during data processing. Returning to main menu.")



def process_datasets(processor, storage, logger, error_handler, pair, timeframe, dataset_types, fetcher_factory: FetcherFactory):
    """Process multiple datasets and save to database"""
    context = {
        "function": "process_datasets",
        "module": __name__,
        "pair": pair,
        "timeframe": timeframe,
        "dataset_types": str(dataset_types)
    }

    # Only fetch correlation data for XAUUSD
    correlation_data = {}
    if pair == "XAUUSD":
        try:
            print("Fetching correlation data for gold-specific features...")
            fetcher = fetcher_factory.create_mt5_fetcher()
            correlation_data = fetcher.fetch_gold_silver_data(timeframe=timeframe)

            if not correlation_data:
                logger.warning("No correlation data available for gold-specific features")
                print("No correlation data available. Processing will continue with standard features only.")
            else:
                logger.info(f"Fetched correlation data: {list(correlation_data.keys())}")
                print(f"Successfully fetched correlation data for: {', '.join(correlation_data.keys())}")
        except Exception as e:
            logger.warning(f"Failed to fetch correlation data: {e}")
            print("Failed to fetch correlation data. Processing will continue with standard features only.")

    for dataset_type in dataset_types:
        context["dataset_type"] = dataset_type

        try:
            print(f"Processing {dataset_type} data...")

            # Get raw data
            raw_data = processor.get_data_from_db(pair, timeframe, dataset_type)

            if raw_data.empty:
                logger.warning(f"No data found for {pair} {timeframe} {dataset_type}")
                print(f"✗ No data found for {dataset_type}")
                continue

            # Process the data with gold-specific features if applicable
            processed_data = processor.process_raw_data(raw_data, correlation_data if pair == "XAUUSD" else None)

            # Create features
            processed_data = processor.create_features(processed_data)

            # Log information about the processed data
            logger.info(f"Processed {len(processed_data)} rows for {pair} {timeframe} {dataset_type}")
            logger.info(f"Features: {list(processed_data.columns)}")

            # Create dummy empty DataFrame for y to maintain compatibility with storage function
            dummy_y = pd.DataFrame()

            # Save processed data to database
            print(f"Saving processed data to database...")
            table_name = f"{pair}_{timeframe}_{dataset_type}_processed"
            context["table_name"] = table_name

            db_success = storage.save_processed_data(processed_data, dummy_y, pair, timeframe, dataset_type)

            if db_success:
                print(f"✓ Successfully saved {len(processed_data)} rows to {table_name}")
                print(f"  - Dataset includes {len(processed_data.columns)} features")
            else:
                print(f"✗ Failed to save data to database")

        except Exception as e:
            error_handler.handle_error(
                exception=e,
                context=context,
                severity=ErrorSeverity.HIGH,
                reraise=False
            )
            print(f"✗ Error processing {dataset_type} data: {str(e)}")