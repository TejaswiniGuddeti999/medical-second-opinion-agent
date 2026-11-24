import sys
import structlog
import logging
from app.config import get_settings

# Windows defaults to cp1252 when output is redirected; LLMs emit characters
# like '‑' (non-breaking hyphen) that crash a log call. Force UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

settings = get_settings()

def setup_logging() -> None:
    logging.basicConfig(
        format = "%(message)s",
        level = getattr(logging, settings.log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()   # pretty in dev; swap for JSONRenderer in prod
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)

    