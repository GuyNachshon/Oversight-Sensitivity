"""
Single Prompt Executor (T025)

Execute single prompt with context template and collect statistics.
"""

import torch
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
import time

from ..experiments.config import ExperimentConfig
from ..experiments.run import ExecutionRun
from ..experiments.validation import validate_config, validate_layer_indices_for_model
from .model_loader import ModelLoader
from .reproducibility import set_seed, get_environment_info
from .contexts import ContextTemplateManager
from .hooks import HookManager
from ..statistics.collector import StatisticsCollector
from ..logging import get_logger, log_memory_usage, log_execution_stats

logger = get_logger(__name__)


class PromptExecutor:
    """Execute prompts with different context conditions."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize executor with experiment configuration.

        Args:
            config: Validated experiment configuration
        """
        self.config = config

        # Set context for all logs
        logger.set_context(experiment_id=config.experiment_id)
        logger.info("Initializing executor", model=config.model_identifier)

        # Validate config
        warnings = validate_config(config)
        if warnings:
            for warning in warnings:
                logger.warning("Configuration warning", warning=warning)

        # Set reproducibility
        set_seed(config.random_seed)
        logger.debug("Random seed set", seed=config.random_seed)

        # Load model and tokenizer
        logger.info("Loading model and tokenizer")
        self.model, self.tokenizer = ModelLoader.load_model_and_tokenizer(
            config.model_identifier
        )
        log_memory_usage("post_model_load")

        # Validate layer indices against model
        num_layers = ModelLoader.get_num_layers(self.model)
        validate_layer_indices_for_model(config.layer_indices, num_layers)
        logger.debug("Layer indices validated", layer_indices=config.layer_indices, total_layers=num_layers)

        # Initialize hook manager
        self.hook_manager = HookManager(self.model, config.layer_indices)
        self.hook_manager.register_hooks()
        logger.debug("Hooks registered", num_layers=len(config.layer_indices))

        # Get environment info
        self.env_info = get_environment_info()

        logger.info("Executor initialized successfully")

    def execute_single_prompt(
        self, prompt_id: str, prompt_text: str, context_condition: str
    ) -> ExecutionRun:
        """
        Execute single prompt with specified context condition.

        Args:
            prompt_id: Unique identifier for the prompt
            prompt_text: The prompt text
            context_condition: Context template to use ("N", "A", or "ARD")

        Returns:
            ExecutionRun with generated text and statistics
        """
        # Generate unique run ID
        run_id = f"{self.config.experiment_id}_{prompt_id}_{context_condition}_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Starting execution",
            run_id=run_id,
            prompt_id=prompt_id,
            context=context_condition,
        )

        # Get context template
        full_prompt = ContextTemplateManager.apply_template(context_condition, prompt_text)
        logger.debug("Applied context template", context=context_condition, prompt_length=len(prompt_text))

        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        logger.debug("Tokenized input", num_input_tokens=input_ids.shape[1])

        # Clear previous activations
        self.hook_manager.clear()

        # Generate
        start_time = datetime.utcnow()
        start_perf = time.perf_counter()

        logger.debug("Starting generation")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.generation_config.max_new_tokens,
                temperature=self.config.generation_config.temperature,
                top_k=self.config.generation_config.top_k,
                top_p=self.config.generation_config.top_p,
                do_sample=self.config.generation_config.do_sample,
                use_cache=self.config.generation_config.use_cache,
                return_dict_in_generate=True,
                output_scores=True,  # Get logits
            )

        end_time = datetime.utcnow()
        duration = time.perf_counter() - start_perf

        # Extract generated text
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Get logits (scores from generation)
        logits = torch.stack(outputs.scores, dim=1)  # [batch_size, seq_len, vocab_size]

        # Get captured activations
        hidden_states = self.hook_manager.get_activations()
        attention_weights = self.hook_manager.get_attention_weights()

        # Collect statistics
        logger.debug("Collecting statistics", num_layers=len(self.config.layer_indices))
        statistics = StatisticsCollector.collect_statistics(
            run_id, logits, hidden_states, attention_weights
        )

        # Organize statistics by layer
        statistics_by_layer = {}
        for stat in statistics:
            layer_key = str(stat.layer_index)
            if layer_key not in statistics_by_layer:
                statistics_by_layer[layer_key] = []
            statistics_by_layer[layer_key].append(stat.to_dict())

        # Count tokens generated (excluding input tokens)
        num_tokens_generated = len(outputs.scores)

        # Log execution stats
        log_execution_stats(
            run_id=run_id,
            num_tokens=num_tokens_generated,
            num_layers=len(self.config.layer_indices),
            duration_seconds=duration,
            context=context_condition,
        )

        # Create execution run
        run = ExecutionRun(
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            prompt_id=prompt_id,
            context_condition=context_condition,
            generated_text=generated_text,
            num_tokens_generated=num_tokens_generated,
            inference_time_seconds=(end_time - start_time).total_seconds(),
            pytorch_version=self.env_info["pytorch_version"],
            transformers_version=self.env_info["transformers_version"],
            cuda_version=self.env_info["cuda_version"],
            gpu_model=self.env_info["gpu_model"],
            timestamp=start_time,
            status="success",
            statistics_by_layer=statistics_by_layer,
        )

        logger.info("Execution completed successfully", run_id=run_id)
        return run

    def save_run(self, run: ExecutionRun, output_dir: Optional[Path] = None) -> Path:
        """
        Save execution run to JSON file.

        Args:
            run: ExecutionRun to save
            output_dir: Directory to save to (defaults to config.output_directory)

        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path(self.config.output_directory)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Filename: {experiment_id}_{prompt_id}_{context}_{timestamp}.json
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{run.experiment_id}_{run.prompt_id}_{run.context_condition}_{timestamp}.json"
        filepath = output_dir / filename

        run.to_json(filepath)

        return filepath

    def cleanup(self) -> None:
        """Remove hooks and free resources."""
        self.hook_manager.remove_hooks()
