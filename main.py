from dependency_injector.wiring import inject, Provide

from UI.Handlers import handle_fetch_data, handle_process_data
from Utilities.Container import Container
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Utilities.ErrorHandler import ErrorHandler, ErrorSeverity
from Fetching.FetcherFactory import FetcherFactory
from Processing.ProcessorFactory import ProcessorFactory
from UI.cli import TradingBotCLI
from UI.Constants import AppMode


@inject
def main(
        config: Config = Provide[Container.config],
        logger: Logger = Provide[Container.logger],
        error_handler: ErrorHandler = Provide[Container.error_handler],
        fetcher_factory: FetcherFactory = Provide[Container.fetcher_factory],
        processor_factory: ProcessorFactory = Provide[Container.processor_factory],
) -> None:
    context = {
        "function": "main",
        "module": __name__
    }

    try:
        logger.info('Application started')
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
                handle_fetch_data(cli, logger, error_handler, fetcher_factory)
            elif action == AppMode.PROCESS_DATA.value:
                handle_process_data(cli, logger, error_handler, processor_factory)
            else:
                logger.info(f'Selected action: {action}')

        logger.info('Application finished')

    except Exception as e:
        error_handler.handle_error(
            exception=e,
            context=context,
            severity=ErrorSeverity.FATAL,
            reraise=False
        )
        print(f"Critical error occurred. The application must exit. See logs for details.")
        return


if __name__ == '__main__':
    container = Container()
    container.wire(modules=[__name__])
    main()
