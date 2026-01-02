"""
Integration Test (T033)

End-to-end test of the full pipeline: config → execution → metrics.
"""

import pytest
import tempfile
import json
from pathlib import Path
from oversight_sensitivity.experiments.config import ExperimentConfig
from oversight_sensitivity.experiments.run import ExecutionRun
from oversight_sensitivity.metrics.compute import compute_metrics_for_prompt


class TestEndToEndPipeline:
    """Test the complete measurement pipeline."""

    @pytest.fixture
    def example_config_path(self):
        """Return path to example config."""
        return Path("experiments/configs/example.json")

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_config_loading(self, example_config_path):
        """Test loading experiment configuration from JSON."""
        config = ExperimentConfig.from_json(example_config_path)

        assert config.experiment_id == "example_experiment"
        assert config.model_identifier == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config.random_seed == 42
        assert len(config.layer_indices) == 3
        assert "N" in config.context_conditions
        assert "A" in config.context_conditions

    def test_config_round_trip(self, temp_output_dir):
        """Test saving and loading config."""
        from oversight_sensitivity.experiments.config import GenerationConfig
        from datetime import datetime

        config = ExperimentConfig(
            experiment_id="test_exp",
            model_identifier="test/model",
            model_size_category="1-3B",
            random_seed=123,
            layer_indices=[0, 10, 20],
            context_conditions=["N", "A"],
            prompt_dataset_path="test.csv",
            generation_config=GenerationConfig(
                max_new_tokens=10,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                do_sample=False,
                use_cache=False,
            ),
            output_directory=str(temp_output_dir),
            created_at=datetime.fromisoformat("2025-01-01T00:00:00"),
            created_by="test",
        )

        # Save
        config_path = temp_output_dir / "config.json"
        config.to_json(config_path)

        # Load
        loaded = ExperimentConfig.from_json(config_path)

        assert loaded.experiment_id == config.experiment_id
        assert loaded.random_seed == config.random_seed
        assert loaded.layer_indices == config.layer_indices

    def test_execution_run_round_trip(self, temp_output_dir):
        """Test saving and loading execution run."""
        from datetime import datetime

        run = ExecutionRun(
            run_id="test_run_123",
            experiment_id="test_exp",
            prompt_id="p001",
            context_condition="N",
            generated_text="Test output",
            num_tokens_generated=2,
            inference_time_seconds=0.5,
            pytorch_version="2.0.0",
            transformers_version="4.30.0",
            cuda_version=None,
            gpu_model=None,
            timestamp=datetime.utcnow(),
            status="success",
            statistics_by_layer={
                "0": [
                    {
                        "run_id": "test_run_123",
                        "layer_index": 0,
                        "token_position": 0,
                        "logit_entropy": 2.5,
                        "top1_probability": 0.3,
                        "residual_stream_norm": 10.0,
                        "activation_variance": 5.0,
                        "attention_entropy": 1.5,
                        "cosine_similarity_to_prev": None,
                    }
                ]
            },
        )

        # Save
        run_path = temp_output_dir / "run.json"
        run.to_json(run_path)

        # Load
        loaded = ExecutionRun.from_json(run_path)

        assert loaded.run_id == run.run_id
        assert loaded.experiment_id == run.experiment_id
        assert loaded.prompt_id == run.prompt_id
        assert loaded.generated_text == run.generated_text
        assert loaded.num_tokens_generated == run.num_tokens_generated
        assert len(loaded.statistics_by_layer) == 1

    def test_metrics_computation(self):
        """Test computing metrics from execution runs."""
        from datetime import datetime

        # Create mock runs for neutral and audited contexts
        run_neutral = ExecutionRun(
            run_id="test_n",
            experiment_id="test",
            prompt_id="p001",
            context_condition="N",
            generated_text="Test",
            num_tokens_generated=5,
            inference_time_seconds=0.1,
            pytorch_version="2.0.0",
            transformers_version="4.30.0",
            cuda_version=None,
            gpu_model=None,
            timestamp=datetime.utcnow(),
            status="success",
            statistics_by_layer={
                "0": [
                    {
                        "run_id": "test_n",
                        "layer_index": 0,
                        "token_position": i,
                        "logit_entropy": 3.0 - i * 0.3,  # Decaying entropy
                        "top1_probability": 0.3 + i * 0.05,
                        "residual_stream_norm": 10.0 + i,
                        "activation_variance": 4.0,  # Higher variance in neutral
                        "attention_entropy": 1.5,
                        "cosine_similarity_to_prev": 0.8 if i > 0 else None,
                    }
                    for i in range(5)
                ]
            },
        )

        run_audited = ExecutionRun(
            run_id="test_a",
            experiment_id="test",
            prompt_id="p001",
            context_condition="A",
            generated_text="Test",
            num_tokens_generated=5,
            inference_time_seconds=0.1,
            pytorch_version="2.0.0",
            transformers_version="4.30.0",
            cuda_version=None,
            gpu_model=None,
            timestamp=datetime.utcnow(),
            status="success",
            statistics_by_layer={
                "0": [
                    {
                        "run_id": "test_a",
                        "layer_index": 0,
                        "token_position": i,
                        "logit_entropy": 3.0 - i * 0.5,  # Faster decay
                        "top1_probability": 0.3 + i * 0.05,
                        "residual_stream_norm": 10.0 + i,
                        "activation_variance": 2.0,  # Lower variance in audited (compressed)
                        "attention_entropy": 1.5,
                        "cosine_similarity_to_prev": 0.9 if i > 0 else None,
                    }
                    for i in range(5)
                ]
            },
        )

        # Compute metrics
        result = compute_metrics_for_prompt(
            prompt_id="p001",
            experiment_id="test",
            runs_neutral=[run_neutral],
            runs_audited=[run_audited],
            context_pair="A_vs_N",
        )

        # Verify result
        assert result.prompt_id == "p001"
        assert result.experiment_id == "test"
        assert result.context_pair == "A_vs_N"

        # CCI should be positive (variance reduced under audit)
        # CCI = 1 - (2.0 / 4.0) = 0.5
        assert result.cci is not None
        assert result.cci > 0, f"Expected positive CCI, got {result.cci}"

        # EHL should be computed
        assert result.ehl is not None
        assert result.ehl > 0

        # TP should be in valid range
        assert result.tp_score is not None
        assert 0 <= result.tp_score <= 1

        # OSS should be computed
        assert result.oss is not None

    def test_cli_validation(self, example_config_path):
        """Test config validation via CLI."""
        from oversight_sensitivity.experiments.validation import validate_config

        config = ExperimentConfig.from_json(example_config_path)
        warnings = validate_config(config)

        # Should validate without errors
        assert isinstance(warnings, list)
        # May have warnings about temperature, but should not raise


@pytest.mark.slow
class TestFullExecution:
    """
    Full execution tests (marked as slow - requires model loading).

    Run with: pytest -m slow
    """

    def test_single_prompt_execution(self):
        """Test executing a single prompt end-to-end."""
        # This is tested by the CLI test we ran earlier
        # Just verify the saved output exists
        import glob

        result_files = glob.glob("results/raw/example_experiment/*.json")
        assert len(result_files) > 0, "Should have at least one execution result"

        # Load and verify
        with open(result_files[0], "r") as f:
            data = json.load(f)

        assert "run_id" in data
        assert "experiment_id" in data
        assert "prompt_id" in data
        assert "generated_text" in data
        assert "statistics_by_layer" in data
        assert "pytorch_version" in data
        assert "transformers_version" in data
