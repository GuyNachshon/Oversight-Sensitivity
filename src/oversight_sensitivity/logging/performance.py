"""
Performance Logging Utilities (T067)

Track execution time, memory usage, and model statistics for debugging and optimization.
"""

import time
import psutil
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps

from .logger import get_logger


logger = get_logger(__name__)


def log_memory_usage(label: str = "current") -> Dict[str, float]:
    """
    Log current memory usage.

    Args:
        label: Label for this measurement

    Returns:
        Dictionary with memory statistics (MB)
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    memory_percent = process.memory_percent()

    logger.debug(
        f"Memory usage: {label}",
        memory_mb=round(memory_mb, 2),
        memory_percent=round(memory_percent, 2),
    )

    return {
        "memory_mb": memory_mb,
        "memory_percent": memory_percent,
    }


def log_execution_stats(
    run_id: str,
    num_tokens: int,
    num_layers: int,
    duration_seconds: float,
    context: str,
) -> None:
    """
    Log execution statistics for a single run.

    Args:
        run_id: Run identifier
        num_tokens: Number of tokens generated
        num_layers: Number of layers tracked
        duration_seconds: Execution time
        context: Context condition
    """
    tokens_per_second = num_tokens / duration_seconds if duration_seconds > 0 else 0

    logger.info(
        "Execution completed",
        run_id=run_id,
        context=context,
        num_tokens=num_tokens,
        num_layers=num_layers,
        duration_seconds=round(duration_seconds, 3),
        tokens_per_second=round(tokens_per_second, 2),
    )


@contextmanager
def track_performance(operation_name: str, **context_fields):
    """
    Context manager to track operation performance.

    Logs start time, end time, duration, and memory delta.

    Args:
        operation_name: Name of operation being tracked
        **context_fields: Additional context fields to include

    Example:
        >>> with track_performance("batch_execution", experiment_id="exp_001"):
        ...     run_batch()
    """
    logger.info(f"{operation_name} starting", **context_fields)

    start_time = time.time()
    start_memory = log_memory_usage(f"{operation_name}_start")

    try:
        yield
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"{operation_name} failed",
            duration_seconds=round(duration, 3),
            error=str(e),
            **context_fields,
        )
        raise
    else:
        duration = time.time() - start_time
        end_memory = log_memory_usage(f"{operation_name}_end")
        memory_delta = end_memory["memory_mb"] - start_memory["memory_mb"]

        logger.info(
            f"{operation_name} completed",
            duration_seconds=round(duration, 3),
            memory_delta_mb=round(memory_delta, 2),
            **context_fields,
        )


def performance_monitor(func):
    """
    Decorator to automatically track function performance.

    Example:
        >>> @performance_monitor
        ... def compute_metrics(data):
        ...     return metrics
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        operation_name = f"{func.__module__}.{func.__name__}"

        with track_performance(operation_name):
            return func(*args, **kwargs)

    return wrapper
