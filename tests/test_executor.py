"""
Unit Tests for Executor and Statistics Collector (T032)

Test prompt execution and statistics collection.
"""

import pytest
import torch
import numpy as np
from oversight_sensitivity.experiments.config import ExperimentConfig, GenerationConfig
from oversight_sensitivity.statistics.collector import StatisticsCollector
from oversight_sensitivity.statistics.stats import ExecutionStatistics


class TestStatisticsCollector:
    """Test statistics collection from model outputs."""

    def test_collector_basic(self):
        """Test basic statistics collection."""
        # Create minimal test data
        batch_size = 1
        seq_len = 3
        vocab_size = 100
        hidden_dim = 64

        # Mock logits
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Mock hidden states for 3 layers
        hidden_states = {
            0: torch.randn(batch_size, seq_len, hidden_dim),
            11: torch.randn(batch_size, seq_len, hidden_dim),
            21: torch.randn(batch_size, seq_len, hidden_dim),
        }

        # Mock attention weights (num_heads=4)
        num_heads = 4
        attention_weights = {
            0: torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1),
            11: torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1),
            21: torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1),
        }

        # Collect statistics
        stats = StatisticsCollector.collect_statistics(
            run_id="test_run",
            logits=logits,
            hidden_states=hidden_states,
            attention_weights=attention_weights,
        )

        # Verify output
        assert len(stats) == 3 * 3, "Should have 3 layers × 3 tokens = 9 stats"

        # Check each stat is valid
        for stat in stats:
            assert isinstance(stat, ExecutionStatistics)
            assert stat.run_id == "test_run"
            assert stat.layer_index in [0, 11, 21]
            assert 0 <= stat.token_position < seq_len
            assert stat.logit_entropy >= 0
            assert 0 <= stat.top1_probability <= 1
            assert stat.residual_stream_norm >= 0
            assert stat.activation_variance >= 0
            assert stat.attention_entropy >= 0

    def test_collector_cosine_similarity(self):
        """Test cosine similarity computation."""
        batch_size = 1
        seq_len = 3
        vocab_size = 100
        hidden_dim = 64

        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create hidden states where tokens are identical (should give cosine = 1.0)
        base_hidden = torch.randn(1, 1, hidden_dim)
        hidden_states = {
            0: base_hidden.expand(batch_size, seq_len, hidden_dim).clone(),
        }

        attention_weights = {}

        stats = StatisticsCollector.collect_statistics(
            run_id="test_run",
            logits=logits,
            hidden_states=hidden_states,
            attention_weights=attention_weights,
        )

        # Token 0 should have None cosine similarity
        token_0_stats = [s for s in stats if s.token_position == 0]
        assert len(token_0_stats) == 1
        assert token_0_stats[0].cosine_similarity_to_prev is None

        # Tokens 1+ should have cosine ≈ 1.0 (identical vectors)
        for stat in stats:
            if stat.token_position > 0:
                assert stat.cosine_similarity_to_prev is not None
                assert abs(stat.cosine_similarity_to_prev - 1.0) < 0.1


class TestExperimentConfig:
    """Test experiment configuration validation."""

    def test_config_validation_layer_indices(self):
        """Test that config validates layer indices."""
        config = ExperimentConfig(
            experiment_id="test",
            model_identifier="test/model",
            model_size_category="1-3B",
            random_seed=42,
            layer_indices=[0, 11, 21],  # Valid: 3 layers
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
            output_directory="test_output",
            created_at="2025-01-01T00:00:00Z",
            created_by="test",
        )

        assert len(config.layer_indices) == 3

    def test_config_requires_neutral_and_audited(self):
        """Test that config requires both N and A contexts."""
        with pytest.raises(ValueError, match="context_conditions must contain at least N and A"):
            ExperimentConfig(
                experiment_id="test",
                model_identifier="test/model",
                model_size_category="1-3B",
                random_seed=42,
                layer_indices=[0, 11, 21],
                context_conditions=["N"],  # Missing A
                prompt_dataset_path="test.csv",
                generation_config=GenerationConfig(
                    max_new_tokens=10,
                    temperature=0.0,
                    top_k=1,
                    top_p=1.0,
                    do_sample=False,
                    use_cache=False,
                ),
                output_directory="test_output",
                created_at="2025-01-01T00:00:00Z",
                created_by="test",
            )

    def test_config_requires_three_layers(self):
        """Test that config requires exactly 3 layer indices."""
        with pytest.raises(ValueError, match="layer_indices must have exactly 3 values"):
            ExperimentConfig(
                experiment_id="test",
                model_identifier="test/model",
                model_size_category="1-3B",
                random_seed=42,
                layer_indices=[0, 11],  # Only 2 layers
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
                output_directory="test_output",
                created_at="2025-01-01T00:00:00Z",
                created_by="test",
            )
