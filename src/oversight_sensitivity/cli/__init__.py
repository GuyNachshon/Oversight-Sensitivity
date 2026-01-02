"""
CLI Package

Command-line interface for oversight sensitivity measurement.
"""

import sys
import os
from pathlib import Path

# Import logging setup
from ..logging import setup_logging


def main():
    """Main CLI entry point."""
    # Setup logging early
    # Check for log level from environment variable
    log_level = os.environ.get("OVERSEE_LOG_LEVEL", "INFO")
    log_dir = os.environ.get("OVERSEE_LOG_DIR", None)

    # For debug/verbose modes, check command line args
    if "--debug" in sys.argv or "-v" in sys.argv:
        log_level = "DEBUG"
        sys.argv = [arg for arg in sys.argv if arg not in ["--debug", "-v"]]

    # Setup logging
    setup_logging(
        log_dir=Path(log_dir) if log_dir else None,
        level=log_level,
        console=True,
        structured=False,  # Human-readable by default
    )

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "--version":
        from .. import __version__
        print(f"oversee v{__version__}")
        sys.exit(0)

    if command == "--help" or command == "-h":
        print_usage()
        sys.exit(0)

    if command == "run-single":
        from .run_single import run_single
        run_single(sys.argv[2:])
    elif command == "run-batch":
        from .run_batch import run_batch
        run_batch(sys.argv[2:])
    elif command == "compute-metrics":
        from .compute_metrics import run_compute_metrics
        run_compute_metrics(sys.argv[2:])
    elif command == "analyze":
        from .analyze import run_analyze
        run_analyze(sys.argv[2:])
    elif command == "visualize":
        from .visualize import run_visualize
        run_visualize(sys.argv[2:])
    elif command == "compare-baselines":
        from .compare_baselines import run_compare_baselines
        run_compare_baselines(sys.argv[2:])
    elif command == "model-info":
        from .model_info import run_model_info
        run_model_info(sys.argv[2:])
    elif command == "select-layers":
        from .select_layers import run_select_layers
        run_select_layers(sys.argv[2:])
    elif command == "validate-config":
        from .validate_config import run_validate_config
        run_validate_config(sys.argv[2:])
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


def print_usage():
    """Print CLI usage information."""
    print("""
Oversight Sensitivity Measurement CLI

Usage:
    oversee <command> [options]

Commands:
    run-single        Execute single prompt with context condition
    run-batch         Execute batch of prompts with progress tracking
    compute-metrics   Compute CCI/EHL/TP/OSS from execution runs
    analyze           Compute bootstrap CIs and effect sizes
    visualize         Generate publication-ready plots from metrics
    compare-baselines Compare experimental contexts vs baseline controls
    select-layers     Smart layer selection via profiling (RECOMMENDED)
    model-info        Get model info and suggested layer indices
    validate-config   Validate experiment configuration
    --version         Show version
    --help            Show this help

Global Options:
    --debug, -v           Enable debug logging
    --help, -h            Show this help
    --version             Show version

Environment Variables:
    OVERSEE_LOG_LEVEL     Log level (DEBUG, INFO, WARNING, ERROR) [default: INFO]
    OVERSEE_LOG_DIR       Directory for log files [optional]

Examples:
    oversee run-single --config experiments/configs/exp_001.json --prompt-id p001 --prompt-text "What is 2+2?" --context N
    oversee --debug run-batch --config experiments/configs/exp_001.json
    oversee compute-metrics --experiment-dir results/raw/exp_001/ --output results/metrics/metrics.json --context-pairs A_vs_N
    oversee analyze --metrics results/metrics/metrics.json --output results/analysis/stats.json
    oversee visualize --metrics results/metrics/metrics.json --analysis results/analysis/stats.json --output-dir plots/
    oversee validate-config --config experiments/configs/exp_001.json

    # Enable debug logging and save logs to file
    OVERSEE_LOG_LEVEL=DEBUG OVERSEE_LOG_DIR=logs/ oversee run-batch --config exp.json

For detailed command help:
    oversee <command> --help
    """)


if __name__ == "__main__":
    main()
