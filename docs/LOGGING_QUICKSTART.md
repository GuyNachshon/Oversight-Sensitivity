# Logging Quick Start Guide

Get started with the logging system in 5 minutes.

## 1. Basic Console Logging (Default)

Just run any command - logging is automatic:

```bash
# Logs appear on console at INFO level
oversee run-single --config experiments/configs/exp_001.json \
  --prompt "What is 2+2?" \
  --prompt-id test_001 \
  --context N
```

**Output:**
```
[19:44:02] INFO - Initializing executor [experiment_id=exp_001, model=TinyLlama]
[19:44:03] INFO - Starting execution [run_id=exp_001_test_001_N_abc123, context=N]
[19:44:05] INFO - Execution completed [duration_seconds=1.234, tokens_per_second=36.46]
```

## 2. Debug Mode (Verbose)

Add `--debug` flag for detailed logs:

```bash
oversee --debug run-single --config exp.json \
  --prompt "test" \
  --prompt-id test_001 \
  --context N
```

**Output:**
```
[19:44:02] INFO  - Initializing executor [experiment_id=exp_001]
[19:44:02] DEBUG - Random seed set [seed=42]
[19:44:02] INFO  - Loading model and tokenizer
[19:44:03] DEBUG - Memory usage: post_model_load [memory_mb=2048.5, memory_percent=12.8]
[19:44:03] DEBUG - Layer indices validated [layer_indices=[0, 11, 21], total_layers=22]
[19:44:03] DEBUG - Hooks registered [num_layers=3]
[19:44:03] INFO  - Executor initialized successfully
[19:44:03] INFO  - Starting execution [run_id=..., prompt_id=test_001, context=N]
[19:44:03] DEBUG - Applied context template [context=N, prompt_length=15]
[19:44:03] DEBUG - Tokenized input [num_input_tokens=25]
[19:44:03] DEBUG - Starting generation
[19:44:05] DEBUG - Collecting statistics [num_layers=3]
[19:44:05] INFO  - Execution completed [num_tokens=45, tokens_per_second=36.46]
```

## 3. Save Logs to File

Set `OVERSEE_LOG_DIR` to save structured JSON logs:

```bash
OVERSEE_LOG_DIR=logs/ oversee run-batch --config exp.json
```

**Creates:**
- `logs/oversee.log` - All logs (JSON format)
- `logs/oversee_errors.log` - Errors only

**Analyze with jq:**
```bash
# View all logs
jq . logs/oversee.log

# Find errors
jq 'select(.level == "ERROR")' logs/oversee.log

# Extract execution times
jq 'select(.message | contains("completed")) | .context.duration_seconds' logs/oversee.log
```

## 4. Environment Variables

Control logging via environment:

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR)
OVERSEE_LOG_LEVEL=DEBUG oversee run-single ...

# Save logs and set level
OVERSEE_LOG_LEVEL=INFO OVERSEE_LOG_DIR=logs/ oversee run-batch ...

# Errors only
OVERSEE_LOG_LEVEL=ERROR oversee run-batch ...
```

## 5. Common Scenarios

### Scenario A: Development/Testing
```bash
# Verbose console output
oversee --debug run-single --config exp.json --prompt "test" --context N
```

### Scenario B: Production Batch Run
```bash
# Save to file, INFO level
OVERSEE_LOG_LEVEL=INFO OVERSEE_LOG_DIR=experiments/logs/exp_001/ \
  oversee run-batch --config experiments/configs/exp_001.json
```

### Scenario C: Debugging Failed Run
```bash
# Maximum verbosity + file logging
OVERSEE_LOG_LEVEL=DEBUG OVERSEE_LOG_DIR=debug_logs/ \
  oversee --debug run-single --config exp.json --prompt "..." --context A

# Then check errors
jq 'select(.level == "ERROR")' debug_logs/oversee_errors.log

# Or view all logs
less debug_logs/oversee.log
```

### Scenario D: Performance Analysis
```bash
# Run with logging
OVERSEE_LOG_DIR=perf_logs/ oversee run-batch --config exp.json

# Analyze performance
jq 'select(.message | contains("completed")) | {run_id: .context.run_id, duration: .context.duration_seconds, tps: .context.tokens_per_second}' perf_logs/oversee.log | jq -s 'sort_by(.duration) | reverse'
```

## 6. Log Analysis Recipes

### Find slowest runs
```bash
jq 'select(.message == "Execution completed")' logs/oversee.log | \
  jq -s 'sort_by(.context.duration_seconds) | reverse | .[0:10]'
```

### Calculate statistics
```bash
# Mean tokens/second
jq 'select(.message == "Execution completed") | .context.tokens_per_second' logs/oversee.log | \
  jq -s 'add / length'

# Max memory usage
jq 'select(.message | contains("Memory usage")) | .context.memory_mb' logs/oversee.log | \
  jq -s 'max'
```

### Group by context condition
```bash
jq 'select(.message == "Execution completed") | {context: .context.context, duration: .context.duration_seconds}' logs/oversee.log | \
  jq -s 'group_by(.context) | map({context: .[0].context, mean_duration: (map(.duration) | add / length)})'
```

### Extract timeline for specific run
```bash
RUN_ID="exp_001_p001_N_abc123"
jq "select(.context.run_id == \"$RUN_ID\") | {time: .timestamp, event: .message}" logs/oversee.log
```

## 7. Python Analysis

```python
import json
from pathlib import Path

# Load logs
log_file = Path("logs/oversee.log")
with open(log_file) as f:
    logs = [json.loads(line) for line in f]

# Find all execution completions
executions = [
    log for log in logs
    if log["message"] == "Execution completed"
]

# Extract durations
durations = [e["context"]["duration_seconds"] for e in executions]

# Calculate statistics
import numpy as np
print(f"Mean: {np.mean(durations):.2f}s")
print(f"Median: {np.median(durations):.2f}s")
print(f"Std: {np.std(durations):.2f}s")
print(f"Min: {np.min(durations):.2f}s")
print(f"Max: {np.max(durations):.2f}s")

# Group by context
from collections import defaultdict
by_context = defaultdict(list)
for e in executions:
    ctx = e["context"]["context"]
    by_context[ctx].append(e["context"]["duration_seconds"])

for ctx, durs in by_context.items():
    print(f"\n{ctx}: {np.mean(durs):.2f}s (n={len(durs)})")
```

## 8. Troubleshooting

### Logs not appearing?
- Check log level: `OVERSEE_LOG_LEVEL=DEBUG oversee ...`
- Verify console output is enabled (it is by default)
- Check if redirecting stderr: `2>&1 | tee output.log`

### Too many logs?
- Reduce verbosity: `OVERSEE_LOG_LEVEL=WARNING oversee ...`
- Remove `--debug` flag
- Filter output: `oversee run-batch ... 2>&1 | grep -E "(ERROR|WARNING)"`

### Logs filling disk?
- Logs aren't rotated automatically
- Use external log rotation: `logrotate`
- Or clean old logs: `find logs/ -name "*.log" -mtime +7 -delete`

### Want structured console output?
```python
from oversight_sensitivity.logging import setup_logging
setup_logging(level="INFO", console=True, structured=True)
```

## Quick Reference

| Feature | Command/Config |
|---------|----------------|
| Debug mode | `oversee --debug ...` |
| Set log level | `OVERSEE_LOG_LEVEL=DEBUG oversee ...` |
| Save to file | `OVERSEE_LOG_DIR=logs/ oversee ...` |
| View logs | `jq . logs/oversee.log` |
| Find errors | `jq 'select(.level == "ERROR")' logs/oversee.log` |
| Analyze performance | See recipes above |

## Next Steps

- Read full guide: `docs/LOGGING.md`
- View test results: `docs/LOGGING_TEST_RESULTS.md`
- Check examples in `src/oversight_sensitivity/logging/`
