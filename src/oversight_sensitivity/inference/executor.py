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
        self,
        prompt_id: str,
        prompt_text: str,
        context_condition: str,
        io_variant_index: int = None,
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

        # Build chat messages with context as system role
        # For IO context, select variant if not specified
        if context_condition == "IO" and io_variant_index is None:
            import random
            io_variant_index = random.randint(0, ContextTemplateManager.get_io_variant_count() - 1)

        messages = ContextTemplateManager.build_messages(
            context_condition, prompt_text, io_variant_index
        )
        logger.debug(
            "Built chat messages",
            context=context_condition,
            io_variant=io_variant_index if context_condition == "IO" else None,
            num_messages=len(messages),
        )

        # Apply chat template and tokenize
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
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
                attention_mask=attention_mask,
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
            io_variant_index=io_variant_index if context_condition == "IO" else None,
        )

        logger.info("Execution completed successfully", run_id=run_id)
        return run

    def execute_reprime_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        stage1_tokens: int = 32,
        stage1_context: str = "N",
        stage2_context: str = "EO",
    ) -> ExecutionRun:
        """
        Two-stage re-prime generation: Start under one context, switch mid-stream.

        Args:
            prompt_id: Unique identifier for the prompt
            prompt_text: The prompt text
            stage1_tokens: Number of tokens to generate in stage 1
            stage1_context: Context for stage 1 (default: N)
            stage2_context: Context for stage 2 (default: EO)

        Returns:
            ExecutionRun with combined statistics, phase markers included
        """
        run_id = f"{self.config.experiment_id}_{prompt_id}_REPRIME_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Starting REPRIME execution",
            run_id=run_id,
            prompt_id=prompt_id,
            stage1_context=stage1_context,
            stage2_context=stage2_context,
            stage1_tokens=stage1_tokens,
        )

        # === STAGE 1: Generate under stage1_context ===
        messages_s1 = ContextTemplateManager.build_messages(stage1_context, prompt_text)
        formatted_s1 = self.tokenizer.apply_chat_template(
            messages_s1, tokenize=False, add_generation_prompt=True
        )
        inputs_s1 = self.tokenizer(formatted_s1, return_tensors="pt")
        input_ids_s1 = inputs_s1["input_ids"].to(self.model.device)
        attention_mask_s1 = inputs_s1["attention_mask"].to(self.model.device)

        self.hook_manager.clear()

        start_time = datetime.utcnow()
        start_perf = time.perf_counter()

        with torch.no_grad():
            outputs_s1 = self.model.generate(
                input_ids_s1,
                attention_mask=attention_mask_s1,
                max_new_tokens=stage1_tokens,
                temperature=self.config.generation_config.temperature,
                top_k=self.config.generation_config.top_k,
                top_p=self.config.generation_config.top_p,
                do_sample=self.config.generation_config.do_sample,
                use_cache=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Collect stage 1 statistics
        logits_s1 = torch.stack(outputs_s1.scores, dim=1)
        hidden_states_s1 = self.hook_manager.get_activations()
        attention_weights_s1 = self.hook_manager.get_attention_weights()

        statistics_s1 = StatisticsCollector.collect_statistics(
            run_id, logits_s1, hidden_states_s1, attention_weights_s1
        )

        # Extract generated tokens from stage 1
        generated_ids_s1 = outputs_s1.sequences[0][input_ids_s1.shape[1]:]
        generated_text_s1 = self.tokenizer.decode(generated_ids_s1, skip_special_tokens=True)

        logger.debug(
            "Stage 1 complete",
            tokens_generated=len(generated_ids_s1),
            text_preview=generated_text_s1[:50],
        )

        # === STAGE 2: Continue under stage2_context ===
        # Build new prompt with stage2 context + original prompt + stage1 output as assistant prefix
        messages_s2 = ContextTemplateManager.build_messages(stage2_context, prompt_text)
        # Add the stage 1 output as assistant's partial response
        messages_s2.append({"role": "assistant", "content": generated_text_s1})

        formatted_s2 = self.tokenizer.apply_chat_template(
            messages_s2, tokenize=False, add_generation_prompt=False  # Already have assistant content
        )
        inputs_s2 = self.tokenizer(formatted_s2, return_tensors="pt")
        input_ids_s2 = inputs_s2["input_ids"].to(self.model.device)
        attention_mask_s2 = inputs_s2["attention_mask"].to(self.model.device)

        self.hook_manager.clear()

        remaining_tokens = self.config.generation_config.max_new_tokens - stage1_tokens
        if remaining_tokens <= 0:
            remaining_tokens = stage1_tokens  # At least generate some in stage 2

        with torch.no_grad():
            outputs_s2 = self.model.generate(
                input_ids_s2,
                attention_mask=attention_mask_s2,
                max_new_tokens=remaining_tokens,
                temperature=self.config.generation_config.temperature,
                top_k=self.config.generation_config.top_k,
                top_p=self.config.generation_config.top_p,
                do_sample=self.config.generation_config.do_sample,
                use_cache=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        end_time = datetime.utcnow()
        duration = time.perf_counter() - start_perf

        # Collect stage 2 statistics
        logits_s2 = torch.stack(outputs_s2.scores, dim=1) if outputs_s2.scores else None
        hidden_states_s2 = self.hook_manager.get_activations()
        attention_weights_s2 = self.hook_manager.get_attention_weights()

        statistics_s2 = []
        if logits_s2 is not None:
            statistics_s2 = StatisticsCollector.collect_statistics(
                run_id, logits_s2, hidden_states_s2, attention_weights_s2
            )

        # Extract stage 2 generated text
        generated_ids_s2 = outputs_s2.sequences[0][input_ids_s2.shape[1]:]
        generated_text_s2 = self.tokenizer.decode(generated_ids_s2, skip_special_tokens=True)

        # Combine statistics with phase markers
        statistics_by_layer = {}
        for stat in statistics_s1:
            layer_key = str(stat.layer_index)
            if layer_key not in statistics_by_layer:
                statistics_by_layer[layer_key] = []
            stat_dict = stat.to_dict()
            stat_dict["phase"] = 1
            stat_dict["phase_token"] = stat.token_position
            statistics_by_layer[layer_key].append(stat_dict)

        for stat in statistics_s2:
            layer_key = str(stat.layer_index)
            if layer_key not in statistics_by_layer:
                statistics_by_layer[layer_key] = []
            stat_dict = stat.to_dict()
            stat_dict["phase"] = 2
            stat_dict["phase_token"] = stat.token_position
            # Offset token_position by stage1 tokens for global position
            stat_dict["token_position"] = stat.token_position + stage1_tokens
            statistics_by_layer[layer_key].append(stat_dict)

        total_tokens = len(generated_ids_s1) + len(generated_ids_s2)
        full_generated_text = generated_text_s1 + generated_text_s2

        log_execution_stats(
            run_id=run_id,
            num_tokens=total_tokens,
            num_layers=len(self.config.layer_indices),
            duration_seconds=duration,
            context="REPRIME",
        )

        run = ExecutionRun(
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            prompt_id=prompt_id,
            context_condition="REPRIME",
            generated_text=full_generated_text,
            num_tokens_generated=total_tokens,
            inference_time_seconds=(end_time - start_time).total_seconds(),
            pytorch_version=self.env_info["pytorch_version"],
            transformers_version=self.env_info["transformers_version"],
            cuda_version=self.env_info["cuda_version"],
            gpu_model=self.env_info["gpu_model"],
            timestamp=start_time,
            status="success",
            statistics_by_layer=statistics_by_layer,
            reprime_phase=2,  # Indicates this is a reprime run
            reprime_transition_token=stage1_tokens,
        )

        logger.info("REPRIME execution completed", run_id=run_id, total_tokens=total_tokens)
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
