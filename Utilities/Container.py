from dependency_injector import containers, providers
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.ProcessorFactory import ProcessorFactory
from Fetching.FetcherFactory import FetcherFactory
from Utilities.PathResolver import PathResolver
import logging


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=[__name__])

    config = providers.Singleton(Config, file_path='Configuration/Configuration.yaml')

    # Logger with warnings and errors only for console, full logging to database
    logger = providers.Singleton(
        Logger,
        name='MT5App',
        level=logging.INFO,
        use_console=True,
        console_level=logging.WARNING,
        db_config=providers.Callable(
            lambda c: c['Database'],
            c=config
        )
    )

    # Path resolver
    path_resolver = providers.Singleton(
        PathResolver,
        config=config
    )

    # Fetcher factory
    fetcher_factory = providers.Singleton(
        FetcherFactory,
        config=config,
        logger=logger
    )

    # Data processor factory
    processor_factory = providers.Singleton(
        ProcessorFactory,
        config=config,
        logger=logger
    )

    # Feature service
    feature_service = providers.Callable(
        lambda factory: factory.create_feature_service(),
        factory=processor_factory
    )

    # Data storage for convenience
    data_storage = providers.Callable(
        lambda factory: factory.create_data_storage(),
        factory=processor_factory
    )
