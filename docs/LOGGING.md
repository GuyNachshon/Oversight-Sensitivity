# Logging Guide

The oversight sensitivity measurement system includes comprehensive structured logging for debugging, performance monitoring, and research audit trails.

## Features

- **Structured JSON logs** for programmatic analysis
- **Context tracking** (experiment_id, run_id, context_condition) automatically injected
- **Performance metrics** (execution time, memory usage, tokens/second)
- **Multiple log levels** (DEBUG, INFO, WARNING, ERROR)
- **Separate log files** for errors vs all logs
- **Human-readable console output** with optional debug mode

## Quick Start

### Console Logging (Default)

By default, logs are printed to console in human-readable format:

```bash
# INFO level (default)
oversee run-single --config exp.json --prompt "test" --context N

# DEBUG level (verbose)
oversee --debug run-single --config exp.json --prompt "test" --context N
```

### File Logging

Enable file logging by setting the `OVERSEE_LOG_DIR` environment variable:

```bash
# Save all logs to logs/ directory
OVERSEE_LOG_DIR=logs/ oversee run-batch --config exp.json
```

This creates:
- `logs/oversee.log` - All logs in JSON format
- `logs/oversee_errors.log` - Errors only in JSON format

### Log Levels

Control verbosity with `OVERSEE_LOG_LEVEL`:

```bash
# DEBUG: All messages (very verbose)
OVERSEE_LOG_LEVEL=DEBUG oversee run-batch --config exp.json

# INFO: Standard operational messages (default)
OVERSEE_LOG_LEVEL=INFO oversee run-batch --config exp.json

# WARNING: Warnings and errors only
OVERSEE_LOG_LEVEL=WARNING oversee run-batch --config exp.json

# ERROR: Errors only
OVERSEE_LOG_LEVEL=ERROR oversee run-batch --config exp.json
```

## Log Format

### Console Output (Human-Readable)

```
[19:29:53] INFO     - Execution completed [run_id=exp_001_p001_N_a1b2c3d4, context=N, num_tokens=45, duration_seconds=1.234]
[19:29:54] WARNING  - Low memory available [memory_mb=512.5, memory_percent=85.2]
[19:29:55] ERROR    - Execution failed [run_id=exp_001_p002_A_e5f6g7h8, error=CUDA out of memory]
```

### File Output (Structured JSON)

```json
{
  "timestamp": "2026-01-01T19:29:53.944492Z",
  "level": "INFO",
  "logger": "oversee.inference.executor",
  "message": "Execution completed",
  "context": {
    "experiment_id": "exp_001",
    "run_id": "exp_001_p001_N_a1b2c3d4",
    "context": "N",
    "num_tokens": 45,
    "duration_seconds": 1.234
  }
}
```

## Automatic Context Tracking

The logging system automatically tracks context fields throughout execution:

```
Initializing executor
  └─ experiment_id: exp_001

Starting execution
  ├─ experiment_id: exp_001
  ├─ run_id: exp_001_p001_N_a1b2c3d4
  ├─ prompt_id: p001
  └─ context: N

Execution completed
  ├─ experiment_id: exp_001
  ├─ run_id: exp_001_p001_N_a1b2c3d4
  ├─ num_tokens: 45
  ├─ num_layers: 3
  ├─ duration_seconds: 1.234
  └─ tokens_per_second: 36.46
```

## Performance Monitoring

The system automatically logs:

### Execution Metrics
- Tokens per second
- Generation duration
- Number of tokens generated
- Number of layers tracked

### Memory Usage
- Memory consumption (MB and %)
- Memory delta during operations
- Logged at key checkpoints (post-model-load, post-execution)

Example output:
```
[19:29:53] DEBUG - Memory usage: post_model_load [memory_mb=2048.5, memory_percent=12.8]
[19:29:54] INFO  - Execution completed [duration_seconds=1.234, tokens_per_second=36.46]
```

## Using Logging in Code

### Basic Logging

```python
from oversight_sensitivity.logging import get_logger

logger = get_logger(__name__)

logger.info("Operation starting")
logger.debug("Detailed debug info", variable_value=123)
logger.warning("Potential issue detected", details="...")
logger.error("Operation failed", error_message="...")
```

### Context Injection

```python
logger = get_logger(__name__)
logger.set_context(experiment_id="exp_001", batch_size=10)

# All subsequent logs will include experiment_id and batch_size
logger.info("Processing batch")
# Logs: [experiment_id=exp_001, batch_size=10]

# Clear context when done
logger.clear_context()
```

### Performance Tracking

```python
from oversight_sensitivity.logging import track_performance

# Context manager
with track_performance("metric_computation", num_prompts=60):
    compute_metrics()
# Automatically logs start, end, duration, and memory delta

# Decorator
from oversight_sensitivity.logging.performance import performance_monitor

@performance_monitor
def expensive_operation():
    # Function execution is automatically tracked
    pass
```

## Log Analysis

### Parsing JSON Logs

```bash
# Extract all errors
jq 'select(.level == "ERROR")' logs/oversee.log

# Find slow executions (>5 seconds)
jq 'select(.context.duration_seconds > 5)' logs/oversee.log

# Group by context condition
jq 'group_by(.context.context) | map({context: .[0].context.context, count: length})' logs/oversee.log

# Calculate average tokens/second
jq '[.context.tokens_per_second] | add / length' logs/oversee.log
```

### Python Analysis

```python
import json

# Load all logs
with open("logs/oversee.log") as f:
    logs = [json.loads(line) for line in f]

# Find all executions
executions = [
    log for log in logs
    if log["message"] == "Execution completed"
]

# Calculate statistics
import numpy as np
durations = [e["context"]["duration_seconds"] for e in executions]
print(f"Mean: {np.mean(durations):.2f}s")
print(f"Median: {np.median(durations):.2f}s")
print(f"Max: {np.max(durations):.2f}s")
```

## Troubleshooting

### Common Issues

**Logs not appearing:**
- Check log level is appropriate (DEBUG shows everything, ERROR only errors)
- Verify `OVERSEE_LOG_DIR` path exists and is writable
- Check console output is enabled

**Too verbose:**
- Set `OVERSEE_LOG_LEVEL=INFO` or `OVERSEE_LOG_LEVEL=WARNING`
- Remove `--debug` flag

**Missing context fields:**
- Context is set at logger initialization
- Check logger.set_context() was called
- Verify context fields match expected keys

## Best Practices

1. **Use appropriate log levels:**
   - DEBUG: Detailed diagnostic information
   - INFO: Confirmation that things are working
   - WARNING: Something unexpected happened
   - ERROR: Serious problem, operation failed

2. **Include context:**
   - Always set experiment_id, run_id at logger init
   - Add operation-specific context (batch_size, num_prompts)
   - Use structured key=value pairs, not string formatting

3. **Log at key checkpoints:**
   - Start/end of operations
   - Before/after expensive computations
   - When state changes (model loaded, hooks registered)

4. **Avoid sensitive data:**
   - Don't log full prompts (may contain PII)
   - Don't log model outputs verbatim
   - Use prompt_id/run_id for tracking instead

5. **Monitor performance:**
   - Use `track_performance()` for operations >100ms
   - Log memory usage for operations loading models/data
   - Track tokens/second to identify bottlenecks

## Integration with Research Workflow

### Experiment Tracking

Logs provide a complete audit trail:
- When experiments ran (timestamps)
- What configs were used (experiment_id)
- How long each step took (duration_seconds)
- What resources were consumed (memory_mb, tokens_per_second)

### Debugging Failed Runs

```bash
# Find all failed runs
jq 'select(.level == "ERROR")' logs/oversee.log

# Get context for specific run
jq 'select(.context.run_id == "exp_001_p001_N_a1b2c3d4")' logs/oversee.log

# Trace execution timeline
jq 'select(.context.run_id == "exp_001_p001_N_a1b2c3d4") | {time: .timestamp, message: .message}' logs/oversee.log
```

### Performance Analysis

```bash
# Find slowest executions
jq 'select(.message == "Execution completed") | {run_id: .context.run_id, duration: .context.duration_seconds}' logs/oversee.log | jq -s 'sort_by(.duration) | reverse | .[0:10]'

# Memory high-water marks
jq 'select(.message | contains("Memory usage")) | .context.memory_mb' logs/oversee.log | jq -s 'max'

# Tokens/second by context
jq 'select(.message == "Execution completed") | {context: .context.context, tps: .context.tokens_per_second}' logs/oversee.log | jq -s 'group_by(.context) | map({context: .[0].context, mean_tps: (map(.tps) | add / length)})'
```

## Examples

### Full Research Pipeline with Logging

```bash
#!/bin/bash
# Setup logging
export OVERSEE_LOG_DIR=logs/exp_001
export OVERSEE_LOG_LEVEL=INFO

# Run experiment
oversee run-batch \
  --config experiments/configs/exp_001.json \
  --contexts N A ARD KW OO R

# Analyze logs
echo "=== Execution Summary ==="
jq 'select(.message == "Execution completed") | .context' logs/exp_001/oversee.log | jq -s 'length' | xargs echo "Total runs:"

echo "=== Performance Stats ==="
jq 'select(.message == "Execution completed") | .context.duration_seconds' logs/exp_001/oversee.log | jq -s 'add / length' | xargs printf "Mean duration: %.2f seconds\n"

echo "=== Errors ==="
jq 'select(.level == "ERROR")' logs/exp_001/oversee_errors.log | jq -s 'length' | xargs echo "Total errors:"
```

### Debug Mode Investigation

```bash
# Enable full debug logging
OVERSEE_LOG_LEVEL=DEBUG OVERSEE_LOG_DIR=debug_logs/ \
  oversee --debug run-single \
    --config exp.json \
    --prompt "test prompt" \
    --context N

# Review detailed logs
less debug_logs/oversee.log
```
