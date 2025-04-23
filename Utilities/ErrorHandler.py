from enum import Enum
from typing import Callable, Optional, Type, Dict, Any, TypeVar, List, Union
import logging
import traceback
import sys
import functools

T = TypeVar('T')


class ErrorSeverity(Enum):
    LOW = "LOW"  # Non-critical errors that don't affect functionality
    MEDIUM = "MEDIUM"  # Affects some functionality but system can continue
    HIGH = "HIGH"  # Critical errors that may require intervention
    FATAL = "FATAL"  # System cannot continue operation


class ErrorHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def handle_error(self,
                     exception: Exception,
                     context: Dict[str, Any],
                     severity: ErrorSeverity,
                     reraise: bool = True) -> None:
        """
        Centralized error handling with consistent logging and context
        """
        # Get detailed exception info
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # Build error message with context
        error_msg = f"Error: {type(exception).__name__}: {str(exception)}"
        context_msg = ", ".join([f"{k}={v}" for k, v in context.items()])

        # Log based on severity
        if severity == ErrorSeverity.LOW:
            self.logger.warning(f"{error_msg} | Context: {context_msg}")
            self.logger.debug(f"Stack trace: {stack_trace}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.error(f"{error_msg} | Context: {context_msg}")
            self.logger.debug(f"Stack trace: {stack_trace}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"{error_msg} | Context: {context_msg}\n{stack_trace}")
        elif severity == ErrorSeverity.FATAL:
            self.logger.critical(f"{error_msg} | Context: {context_msg}\n{stack_trace}")

        # Reraise if specified
        if reraise:
            raise

    def error_boundary(self,
                       severity: ErrorSeverity,
                       reraise: bool = True,
                       success_value: Optional[T] = None) -> Callable:
        """
        Decorator for standardized error handling in methods
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Build context from function and args
                    context = {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args": str(args) if args else "none",
                        "kwargs": str(kwargs) if kwargs else "none"
                    }

                    self.handle_error(
                        exception=e,
                        context=context,
                        severity=severity,
                        reraise=reraise
                    )

                    if not reraise:
                        return success_value
                    raise

            return wrapper

        return decorator