from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity
from Fetching.FetcherFactory import FetcherFactory
from Processing.ProcessorFactory import ProcessorFactory
from UI.cli import TradingBotCLI


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
                        processor_factory: ProcessorFactory) -> None:
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
                                 ["training", "validation", "testing"])

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
                        [dataset_config['dataset_type']]
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


def process_datasets(processor, storage, logger, error_handler, pair, timeframe, dataset_types):
    """Process multiple datasets and save to database"""
    context = {
        "function": "process_datasets",
        "module": __name__,
        "pair": pair,
        "timeframe": timeframe,
        "dataset_types": str(dataset_types)
    }

    for dataset_type in dataset_types:
        context["dataset_type"] = dataset_type

        try:
            print(f"Processing {dataset_type} data...")
            X, y = processor.prepare_dataset(pair, timeframe, dataset_type)

            if X.empty:
                logger.warning(f"No data found for {pair} {timeframe} {dataset_type}")
                print(f"✗ No data found for {dataset_type}")
                continue

            # Log information about the processed data
            logger.info(f"Processed {len(X)} rows for {pair} {timeframe} {dataset_type}")
            logger.info(f"Features: {list(X.columns)}")
            if not y.empty:
                logger.info(f"Targets: {list(y.columns)}")

            # Save processed data to database
            print(f"Saving processed data to database...")
            table_name = f"{pair}_{timeframe}_{dataset_type}_processed"
            context["table_name"] = table_name

            db_success = storage.save_processed_data(X, y, pair, timeframe, dataset_type)

            if db_success:
                print(f"✓ Successfully saved {len(X)} rows to {table_name}")
                print(f"  - Dataset includes {len(X.columns)} features")
                if not y.empty:
                    print(f"  - Created {len(y.columns)} target variables")
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