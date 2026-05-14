import logging
import sys

from loguru import logger


def setup_logging(debug_mode: bool = False):
    logger.remove()

    console_level = "DEBUG" if debug_mode else "INFO"
    logger.add(
        sys.stderr,
        level=console_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra}</cyan> - <level>{message}</level>",
        enqueue=True,
        colorize=True,
    )

    logger.add(
        "logs/transcriber_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{message}",
        serialize=True,
        enqueue=True,
        rotation="128 MB",
        retention="14 days",  # Храним логи неделю, потом удаляем
    )

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    return logger
