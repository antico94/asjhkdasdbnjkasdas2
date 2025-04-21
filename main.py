import logging
import os
from datetime import datetime, timedelta

import questionary
from dependency_injector.wiring import inject, Provide
from questionary import Choice

from Configuration.Constants import CurrencyPairs
from Utilities.Container import Container
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Fetching.FetcherFactory import FetcherFactory
from Processing.ProcessorFactory import ProcessorFactory
from Processing.FeatureService import FeatureService
from UI.cli import TradingBotCLI
from UI.Constants import AppMode
from Models.ModelFactory import ModelFactory
from Training.ModelTrainer import ModelTrainer


@inject
def main(
        config: Config = Provide[Container.config],
        logger: Logger = Provide[Container.logger],
        fetcher_factory: FetcherFactory = Provide[Container.fetcher_factory],
        processor_factory: ProcessorFactory = Provide[Container.processor_factory],
        feature_service: FeatureService = Provide[Container.feature_service],
        model_factory: ModelFactory = Provide[Container.model_factory],
) -> None:
    logger.info('Application started')
    # Log configuration values
    mt5_login = config.get_nested('MetaTrader5', 'Login')
    logger.debug(f'MT5 login: {mt5_login}')
    mt5_server = config.get_nested('MetaTrader5', 'Server')
    mt5_timeout = config.get_nested('MetaTrader5', 'Timeout')

    # Log database connection info
    db_host = config.get_nested('Database', 'Host')
    db_port = config.get_nested('Database', 'Port')
    db_user = config.get_nested('Database', 'User')
    db_name = config.get_nested('Database', 'Database')

    logger.info(f'MT5 Server: {mt5_server} with timeout {mt5_timeout}')
    logger.info(f'Database: {db_user}@{db_host}:{db_port}/{db_name}')

    # Run CLI
    cli = TradingBotCLI(config)

    while True:
        action = cli.main_menu()

        if action == "exit":
            logger.info('Application exiting')
            break
        elif action == AppMode.FETCH_DATA.value:
            handle_fetch_data(cli, logger, fetcher_factory)
        elif action == AppMode.PROCESS_DATA.value:
            handle_process_data(cli, logger, processor_factory)
        elif action == AppMode.ANALYZE_FEATURES.value:
            handle_analyze_features(cli, logger, processor_factory, feature_service)
        elif action == AppMode.TRAIN_MODEL.value:
            handle_train_model(cli, logger, processor_factory, model_factory)
        else:
            logger.info(f'Selected action: {action}')

    logger.info('Application finished')


def handle_fetch_data(cli: TradingBotCLI, logger: Logger, fetcher_factory: FetcherFactory) -> None:
    """Handle fetch data flow"""
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


def handle_process_data(cli: TradingBotCLI, logger: Logger, processor_factory: ProcessorFactory) -> None:
    """Handle data processing flow"""
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
            process_datasets(processor, storage, logger, "XAUUSD", "H1", ["training", "validation", "testing"])

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
                    dataset_config['pair'],
                    dataset_config['timeframe'],
                    [dataset_config['dataset_type']]
                )
            else:
                logger.info("Dataset selection cancelled")
                print("Dataset selection cancelled")


def handle_analyze_features(
        cli: TradingBotCLI,
        logger: Logger,
        processor_factory: ProcessorFactory,
        feature_service: FeatureService
) -> None:
    """Handle feature analysis flow"""
    while True:
        analyze_action = cli.analyze_features_menu()

        if analyze_action == "back":
            logger.info("Returning to main menu")
            break

        elif analyze_action == "run_analysis":
            logger.info("Running comprehensive feature analysis")
            print("Starting comprehensive feature analysis...")

            feature_analyzer = processor_factory.create_feature_analyzer()

            # Run dual analysis (both direction and magnitude)
            try:
                results = feature_analyzer.run_dual_analysis(
                    pair="XAUUSD",
                    timeframe="H1",
                    dataset_type="training",
                    direction_target="direction_1",
                    magnitude_target="future_price_1"
                )

                if results and (results.get("direction") or results.get("magnitude")):
                    print(f"✓ Feature analysis completed successfully")
                    print(f"  - Selected {len(results.get('direction', []))} features for direction model")
                    print(f"  - Selected {len(results.get('magnitude', []))} features for magnitude model")
                    print(f"  - Results saved to database and FeatureAnalysis directory")

                    # Display top features
                    display_top_features(feature_service, "XAUUSD", "H1")
                else:
                    print("✗ Feature analysis failed to select features")
            except Exception as e:
                logger.error(f"Error running feature analysis: {e}")
                print(f"✗ Error running feature analysis: {str(e)}")

        elif analyze_action == "custom_analysis":
            logger.info("Running custom feature analysis")
            analysis_config = cli.feature_analysis_config_menu()

            if analysis_config:
                logger.info(f"Selected analysis config: {analysis_config}")
                print(f"Running feature analysis for {analysis_config['pair']} {analysis_config['timeframe']} "
                      f"with target {analysis_config['target']}...")

                feature_analyzer = processor_factory.create_feature_analyzer()

                # Determine if this is a direction or magnitude target
                is_classification = "direction" in analysis_config['target'] or "signal" in analysis_config['target']
                model_type = "direction" if is_classification else "magnitude"

                try:
                    # Run single target analysis
                    selected_features, importance_df = feature_analyzer.run_complete_analysis(
                        pair=analysis_config['pair'],
                        timeframe=analysis_config['timeframe'],
                        dataset_type="training",
                        target_col=analysis_config['target'],
                        is_classification=is_classification
                    )

                    if selected_features:
                        print(f"✓ Feature analysis completed successfully")
                        print(f"  - Selected {len(selected_features)} optimal features for {model_type} model")
                        print(f"  - Results saved to database and FeatureAnalysis directory")

                        # Display top features
                        display_top_features(
                            feature_service,
                            analysis_config['pair'],
                            analysis_config['timeframe'],
                            model_type
                        )
                    else:
                        print("✗ Feature analysis failed to select features")
                except Exception as e:
                    logger.error(f"Error running feature analysis: {e}")
                    print(f"✗ Error running feature analysis: {str(e)}")
            else:
                logger.info("Feature analysis configuration cancelled")
                print("Feature analysis configuration cancelled")


def handle_train_model(
        cli: TradingBotCLI,
        logger: Logger,
        processor_factory: ProcessorFactory,
        model_factory: ModelFactory
) -> None:
    """Handle model training flow"""
    while True:
        train_action = cli.train_model_menu()

        if train_action == "back":
            logger.info("Returning to main menu")
            break

        elif train_action == "train_new":
            logger.info("Training new model")
            model_config = cli.model_config_menu()

            if model_config:
                logger.info(f"Selected model config: {model_config}")
                print(f"Training new model for {model_config['pair']} {model_config['timeframe']} "
                      f"({model_config['model_type']})...")

                try:
                    # Create data storage
                    data_storage = processor_factory.create_data_storage()

                    # Create training pipeline
                    training_pipeline = model_factory.create_training_pipeline(
                        data_storage,
                        pair=model_config['pair'],
                        timeframe=model_config['timeframe'],
                        model_type=model_config['model_type']
                    )

                    if not training_pipeline:
                        logger.error("Failed to create training pipeline")
                        print("✗ Failed to create training pipeline")
                        continue

                    # Get trainer from pipeline
                    trainer = training_pipeline.get('trainer')

                    # Prepare dataset
                    print("Preparing training data...")
                    dataset = trainer.prepare_training_data(
                        pair=model_config['pair'],
                        timeframe=model_config['timeframe'],
                        model_type=model_config['model_type']
                    )

                    if not dataset:
                        logger.error("Failed to prepare training data")
                        print("✗ Failed to prepare training data")
                        continue

                    # Train model
                    print(
                        f"Training model with {model_config['epochs']} epochs, batch size {model_config['batch_size']}...")
                    history = trainer.train_model(
                        epochs=model_config['epochs'],
                        batch_size=model_config['batch_size']
                    )

                    if history:
                        print("✓ Model training completed successfully")
                        print(f"  - Model saved to {trainer.output_dir}")

                        # Evaluate model
                        print("Evaluating model performance...")
                        metrics = trainer.evaluate_model()

                        if metrics:
                            print("Model evaluation results:")
                            print(f"  - Direction accuracy: {metrics.get('direction_accuracy', 0):.4f}")
                            print(f"  - Win rate: {metrics.get('win_rate', 0):.4f}")
                            print(f"  - Profit factor: {metrics.get('profit_factor', 0):.4f}")
                            print(f"  - Expected return: {metrics.get('expected_return', 0):.4f}")
                    else:
                        print("✗ Model training failed")

                except Exception as e:
                    logger.error(f"Error training model: {e}")
                    print(f"✗ Error training model: {str(e)}")
            else:
                logger.info("Model configuration cancelled")
                print("Model configuration cancelled")

        elif train_action == "continue_training":
            logger.info("Continuing training for existing model")
            model_path = cli.select_model_menu()

            if model_path and model_path != "back":
                print(f"Loading model from {model_path}...")

                try:
                    # Load the model
                    model = model_factory.load_model(model_path)

                    if not model:
                        logger.error("Failed to load model")
                        print("✗ Failed to load model")
                        continue

                    # Get model metadata
                    model_dir = os.path.dirname(model_path)
                    metadata_path = os.path.join(model_dir, "model_metadata.json")

                    import json
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    pair = metadata.get("pair")
                    timeframe = metadata.get("timeframe")
                    model_type = metadata.get("model_type")

                    print(f"Model loaded: {pair} {timeframe} {model_type}")

                    # Get training parameters
                    epochs = questionary.text(
                        'Additional epochs to train (default: 20):',
                        default="20",
                        validate=lambda text: text.isdigit() and int(text) > 0
                    ).ask()

                    batch_size = questionary.text(
                        'Batch size (default: 32):',
                        default="32",
                        validate=lambda text: text.isdigit() and int(text) > 0
                    ).ask()

                    # Create data storage
                    data_storage = processor_factory.create_data_storage()

                    # Create data preprocessor
                    preprocessor = model_factory.create_data_preprocessor(data_storage)
                    preprocessor.set_pair_timeframe(pair, timeframe)

                    # Create trainer with the loaded model
                    trainer = model_factory.create_model_trainer(preprocessor, model)
                    trainer.set_model_type(model_type)

                    # Prepare dataset
                    print("Preparing training data...")
                    dataset = trainer.prepare_training_data(
                        pair=pair,
                        timeframe=timeframe,
                        model_type=model_type
                    )

                    if not dataset:
                        logger.error("Failed to prepare training data")
                        print("✗ Failed to prepare training data")
                        continue

                    # Continue training
                    print(f"Continuing training with {epochs} additional epochs...")
                    history = trainer.train_model(
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        continued_training=True
                    )

                    if history:
                        print("✓ Model training completed successfully")
                        print(f"  - Updated model saved to {trainer.output_dir}")

                        # Evaluate model
                        print("Evaluating model performance...")
                        metrics = trainer.evaluate_model()

                        if metrics:
                            print("Model evaluation results:")
                            print(f"  - Direction accuracy: {metrics.get('direction_accuracy', 0):.4f}")
                            print(f"  - Win rate: {metrics.get('win_rate', 0):.4f}")
                            print(f"  - Profit factor: {metrics.get('profit_factor', 0):.4f}")
                            print(f"  - Expected return: {metrics.get('expected_return', 0):.4f}")
                    else:
                        print("✗ Model training failed")

                except Exception as e:
                    logger.error(f"Error continuing model training: {e}")
                    print(f"✗ Error continuing model training: {str(e)}")
            else:
                logger.info("Model selection cancelled")
                print("Model selection cancelled")

        elif train_action == "test_model":
            logger.info("Testing model performance")
            model_path = cli.select_model_menu()

            if model_path and model_path != "back":
                print(f"Loading model from {model_path}...")

                try:
                    # Load the model
                    model = model_factory.load_model(model_path)

                    if not model:
                        logger.error("Failed to load model")
                        print("✗ Failed to load model")
                        continue

                    # Get model metadata
                    model_dir = os.path.dirname(model_path)
                    metadata_path = os.path.join(model_dir, "model_metadata.json")

                    import json
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    pair = metadata.get("pair")
                    timeframe = metadata.get("timeframe")
                    model_type = metadata.get("model_type")

                    print(f"Model loaded: {pair} {timeframe} {model_type}")

                    # Create data storage
                    data_storage = processor_factory.create_data_storage()

                    # Create data preprocessor
                    preprocessor = model_factory.create_data_preprocessor(data_storage)
                    preprocessor.set_pair_timeframe(pair, timeframe)

                    # Create trainer with the loaded model
                    trainer = model_factory.create_model_trainer(preprocessor, model)
                    trainer.set_model_type(model_type)

                    # Test dataset options
                    dataset_types = ["training", "validation", "testing"]
                    dataset_type = questionary.select(
                        'Select dataset for model evaluation:',
                        choices=dataset_types
                    ).ask()

                    if not dataset_type:
                        continue

                    # Prepare dataset
                    print(f"Loading {dataset_type} data for model evaluation...")
                    dataset = trainer.prepare_training_data(
                        pair=pair,
                        timeframe=timeframe,
                        dataset_type=dataset_type,
                        model_type=model_type
                    )

                    if not dataset:
                        logger.error(f"Failed to prepare {dataset_type} data")
                        print(f"✗ Failed to prepare {dataset_type} data")
                        continue

                    # Evaluate model
                    print("Evaluating model performance...")
                    metrics = trainer.evaluate_model()

                    if metrics:
                        print(f"Model evaluation results on {dataset_type} data:")
                        print(f"  - Direction accuracy: {metrics.get('direction_accuracy', 0):.4f}")
                        print(f"  - Win rate: {metrics.get('win_rate', 0):.4f}")
                        print(f"  - Profit factor: {metrics.get('profit_factor', 0):.4f}")
                        print(f"  - Expected return: {metrics.get('expected_return', 0):.4f}")
                        print(f"  - Evaluation plots saved to {trainer.output_dir}")
                    else:
                        print("✗ Model evaluation failed")

                except Exception as e:
                    logger.error(f"Error evaluating model: {e}")
                    print(f"✗ Error evaluating model: {str(e)}")
            else:
                logger.info("Model selection cancelled")
                print("Model selection cancelled")


def display_top_features(
        feature_service: FeatureService,
        pair: str,
        timeframe: str,
        model_type: str = "direction"
) -> None:
    """Display top features for a given model type"""
    importance_dict = feature_service.get_feature_importance(
        pair, timeframe, model_type, top_n=10
    )

    if not importance_dict:
        print(f"No feature importance data available for {pair} {timeframe} {model_type}")
        return

    print(f"\n═════════════ TOP FEATURES ({model_type.upper()}) ═════════════")
    for i, (feature, importance) in enumerate(importance_dict.items(), 1):
        print(f"{i}. {feature}: {importance:.4f}")
    print("═══════════════════════════════════════════")


def process_datasets(processor, storage, logger, pair, timeframe, dataset_types):
    """Process multiple datasets and save to database"""
    for dataset_type in dataset_types:
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
            db_success = storage.save_processed_data(X, y, pair, timeframe, dataset_type)

            if db_success:
                print(f"✓ Successfully saved {len(X)} rows to {table_name}")
                print(f"  - Dataset includes {len(X.columns)} features")
                if not y.empty:
                    print(f"  - Created {len(y.columns)} target variables")
            else:
                print(f"✗ Failed to save data to database")

        except Exception as e:
            logger.error(f"Error processing {dataset_type} data: {e}")
            print(f"✗ Error processing {dataset_type} data: {str(e)}")


if __name__ == '__main__':
    container = Container()
    container.wire(modules=[__name__])
    main()