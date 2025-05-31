import logging
import sys
import structlog
from .config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    log_level = settings.log_level.upper()

    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stdout,
    )

    # Configure structlog for prettier and more consistent logs
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
        ],
    )

    # Create a logger instance for the application
    logger = structlog.get_logger("codeviz")
    logger.info("Logging configured", level=log_level)

    return logger