"""
Logging System (T067)

Structured logging with context tracking for research reproducibility.

Features:
- Structured JSON logs for programmatic analysis
- Context tracking (experiment_id, run_id, context_condition)
- Separate log files for errors vs debug info
- Performance metrics (execution time, memory usage)
- Configurable log levels
"""

import logging
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager
import time


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs as JSON with standard fields:
    - timestamp: ISO 8601 datetime
    - level: Log level (DEBUG, INFO, WARNING, ERROR)
    - logger: Logger name
    - message: Log message
    - context: Additional context fields (experiment_id, run_id, etc.)
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context if available
        if hasattr(record, "context"):
            log_data["context"] = record.context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "context",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.

    Format: [TIMESTAMP] LEVEL - message [context]
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        level = record.levelname
        message = record.getMessage()

        # Add context if available
        context_str = ""
        if hasattr(record, "context"):
            context_items = [f"{k}={v}" for k, v in record.context.items()]
            context_str = f" [{', '.join(context_items)}]"

        log_line = f"[{timestamp}] {level:8s} - {message}{context_str}"

        # Add exception if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)

        return log_line


class ContextLogger:
    """
    Logger with automatic context injection.

    Maintains context fields (experiment_id, run_id, etc.) that are
    automatically added to all log messages.

    Example:
        >>> logger = ContextLogger("oversee.executor")
        >>> logger.set_context(experiment_id="exp_001", run_id="run_123")
        >>> logger.info("Starting execution")
        # Logs: {"message": "Starting execution", "context": {"experiment_id": "exp_001", "run_id": "run_123"}}
    """

    def __init__(self, name: str):
        """
        Initialize context logger.

        Args:
            name: Logger name (e.g., "oversee.metrics")
        """
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """
        Set context fields for all subsequent log messages.

        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context fields."""
        self.context.clear()

    def _log(self, level: int, message: str, **kwargs) -> None:
        """
        Internal log method with context injection.

        Args:
            level: Log level (logging.DEBUG, INFO, etc.)
            message: Log message
            **kwargs: Additional fields to add to log
        """
        extra = {"context": {**self.context, **kwargs}}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        extra = {"context": {**self.context, **kwargs}}
        self.logger.exception(message, extra=extra)

    @contextmanager
    def timer(self, operation: str):
        """
        Context manager to log operation duration.

        Example:
            >>> with logger.timer("metric_computation"):
            ...     compute_metrics()
            # Logs: "metric_computation completed in 1.23s"
        """
        start_time = time.time()
        self.debug(f"{operation} started")

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.info(f"{operation} completed", duration_seconds=duration)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
    structured: bool = False,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        log_dir: Directory to write log files (optional)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        console: Enable console output
        structured: Use structured JSON format for console

    Creates log files:
        - {log_dir}/oversee.log: All logs (JSON format)
        - {log_dir}/oversee_errors.log: Errors only (JSON format)
    """
    # Convert level string to logging constant
    log_level = getattr(logging, level.upper())

    # Get root logger for oversee package
    root_logger = logging.getLogger("oversee")
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(HumanReadableFormatter())

        root_logger.addHandler(console_handler)

    # File handlers (if log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # All logs file (JSON format)
        all_logs_handler = logging.FileHandler(log_dir / "oversee.log")
        all_logs_handler.setLevel(logging.DEBUG)  # Capture everything
        all_logs_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(all_logs_handler)

        # Errors file (JSON format)
        error_handler = logging.FileHandler(log_dir / "oversee_errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)


def get_logger(name: str) -> ContextLogger:
    """
    Get a context logger for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        ContextLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return ContextLogger(f"oversee.{name}")
