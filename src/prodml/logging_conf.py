import contextvars
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from pythonjsonlogger import json

F = TypeVar("F", bound=Callable[..., Any])

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id",
    default="-",
)


class JsonFormatter(json.JsonFormatter):
    """Format log records as structured JSON."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(
            log_record,
            record,
            message_dict,
        )

        log_record["timestamp"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(record.created),
        )

        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["correlation_id"] = correlation_id_var.get()


def configure_logging() -> None:
    """Configure application logging with a JSON formatter."""

    handler = logging.StreamHandler()

    formatter = JsonFormatter(
        "%(timestamp)s %(level)s %(logger)s " "%(message)s %(correlation_id)s"
    )

    handler.setFormatter(formatter)

    root_logger = logging.getLogger()

    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def timed(func: F) -> F:
    """Log the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start

            logging.getLogger(func.__module__).info(
                "function_timing",
                extra={
                    "function": func.__name__,
                    "duration_ms": round(elapsed * 1000, 3),
                },
            )

    return cast(F, wrapper)
