"""
Integration Test for ARD Intervention (T039)

Test that ARD context shifts metrics toward neutral (reversibility hypothesis).
"""

import pytest
from datetime import datetime
from oversight_sensitivity.experiments.run import ExecutionRun
from oversight_sensitivity.metrics.compute import compute_metrics_for_prompt
from oversight_sensitivity.analysis.reversibility import compute_reversibility


class TestARDIntervention:
    """Test ARD intervention analysis."""

    def _create_run(self, context: str, activation_variance: float, entropy_pattern: list) -> ExecutionRun:
        """Helper to create execution run with specified parameters."""
        return ExecutionRun(
            run_id=f"test_{context}",
            experiment_id="test_ard",
            prompt_id="p001",
            context_condition=context,
            generated_text="Test output",
            num_tokens_generated=len(entropy_pattern),
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
                        "run_id": f"test_{context}",
                        "layer_index": 0,
                        "token_position": i,
                        "logit_entropy": entropy_pattern[i],
                        "top1_probability": 0.3 + i * 0.05,
                        "residual_stream_norm": 10.0 + i,
                        "activation_variance": activation_variance,
                        "attention_entropy": 1.5,
                        "cosine_similarity_to_prev": 0.8 if i > 0 else None,
                    }
                    for i in range(len(entropy_pattern))
                ]
            },
        )

    def test_ard_reverses_compression(self):
        """
        Test that ARD reverses oversight-induced compression.

        Scenario:
        - Neutral (N): variance = 4.0
        - Audited (A): variance = 2.0 (compressed)
        - ARD: variance = 3.0 (partially reversed toward N)

        Expected:
        - A_vs_N: CCI > 0 (compression detected)
        - ARD_vs_A: CCI < 0 (expansion from A)
        - Reversibility: ~50% (moved halfway back to neutral)
        """
        # Create runs with different compression levels
        entropy_n = [3.0, 2.5, 2.0, 1.5, 1.0]
        entropy_a = [3.0, 2.0, 1.5, 1.0, 0.5]  # Faster decay (more compressed)
        entropy_ard = [3.0, 2.3, 1.8, 1.3, 0.8]  # Between N and A

        run_n = self._create_run("N", activation_variance=4.0, entropy_pattern=entropy_n)
        run_a = self._create_run("A", activation_variance=2.0, entropy_pattern=entropy_a)
        run_ard = self._create_run("ARD", activation_variance=3.0, entropy_pattern=entropy_ard)

        # Compute metrics for A vs N
        metrics_a_vs_n = compute_metrics_for_prompt(
            prompt_id="p001",
            experiment_id="test_ard",
            runs_neutral=[run_n],
            runs_audited=[run_a],
            context_pair="A_vs_N",
        )

        # Compute metrics for ARD vs A
        metrics_ard_vs_a = compute_metrics_for_prompt(
            prompt_id="p001",
            experiment_id="test_ard",
            runs_neutral=[run_a],  # A is now the baseline
            runs_audited=[run_ard],  # ARD is the "audited" context
            context_pair="ARD_vs_A",
        )

        # Compute metrics for ARD vs N (net effect)
        metrics_ard_vs_n = compute_metrics_for_prompt(
            prompt_id="p001",
            experiment_id="test_ard",
            runs_neutral=[run_n],
            runs_audited=[run_ard],
            context_pair="ARD_vs_N",
        )

        # Verify A_vs_N shows compression (positive CCI)
        assert metrics_a_vs_n.cci is not None
        assert metrics_a_vs_n.cci > 0, f"Expected A to compress vs N, got CCI={metrics_a_vs_n.cci}"

        # Verify ARD_vs_A shows expansion (negative CCI)
        # ARD has higher variance than A, so CCI should be negative
        assert metrics_ard_vs_a.cci is not None
        assert metrics_ard_vs_a.cci < 0, f"Expected ARD to expand vs A, got CCI={metrics_ard_vs_a.cci}"

        # Verify ARD_vs_N shows partial compression (between 0 and A_vs_N)
        assert metrics_ard_vs_n.cci is not None
        assert 0 < metrics_ard_vs_n.cci < metrics_a_vs_n.cci, (
            f"Expected ARD between N and A, got ARD_vs_N CCI={metrics_ard_vs_n.cci}, "
            f"A_vs_N CCI={metrics_a_vs_n.cci}"
        )

        # Compute reversibility
        reversibility = compute_reversibility(metrics_a_vs_n, metrics_ard_vs_a, metrics_ard_vs_n)

        assert reversibility["cci_reversibility_percent"] is not None
        # Should be around 50% (ARD moved halfway back)
        # CCI_A_vs_N = 1 - (2/4) = 0.5
        # CCI_ARD_vs_A = 1 - (3/2) = -0.5
        # Reversibility = -(-0.5 / 0.5) * 100 = 100%
        assert 80 <= reversibility["cci_reversibility_percent"] <= 120, (
            f"Expected ~100% reversibility, got {reversibility['cci_reversibility_percent']}%"
        )

    def test_context_template_differences(self):
        """Test that N, A, and ARD templates are distinct."""
        from oversight_sensitivity.inference.contexts import ContextTemplateManager

        template_n = ContextTemplateManager.get_template("N")
        template_a = ContextTemplateManager.get_template("A")
        template_ard = ContextTemplateManager.get_template("ARD")

        # All should be non-empty
        assert len(template_n) > 0
        assert len(template_a) > 0
        assert len(template_ard) > 0

        # All should be distinct
        assert template_n != template_a
        assert template_a != template_ard
        assert template_n != template_ard

        # ARD should be longer than A (includes reasoning discipline)
        assert len(template_ard) > len(template_a)

        # ARD should mention reasoning/discipline
        assert any(word in template_ard.lower() for word in ["reason", "discipline", "correctness", "check"])

    def test_reversibility_edge_cases(self):
        """Test reversibility computation edge cases."""
        from oversight_sensitivity.metrics.results import MetricResults

        # Case 1: No deformation (A = N)
        metrics_no_change = MetricResults(
            metric_id="test1",
            experiment_id="test",
            prompt_id="p001",
            context_pair="A_vs_N",
            cci=0.0,  # No compression
            ehl=10.0,
            tp_score=0.5,
            oss=0.0,
        )

        metrics_ard_change = MetricResults(
            metric_id="test2",
            experiment_id="test",
            prompt_id="p001",
            context_pair="ARD_vs_A",
            cci=-0.1,  # Small expansion
            ehl=11.0,
            tp_score=0.6,
            oss=0.1,
        )

        reversibility = compute_reversibility(metrics_no_change, metrics_ard_change)

        # Should return None when no deformation to reverse
        assert reversibility["cci_reversibility_percent"] is None

        # Case 2: ARD amplifies compression (unusual)
        metrics_compression = MetricResults(
            metric_id="test3",
            experiment_id="test",
            prompt_id="p001",
            context_pair="A_vs_N",
            cci=0.5,  # Compression
            ehl=10.0,
            tp_score=0.5,
            oss=1.0,
        )

        metrics_more_compression = MetricResults(
            metric_id="test4",
            experiment_id="test",
            prompt_id="p001",
            context_pair="ARD_vs_A",
            cci=0.2,  # More compression (same direction)
            ehl=9.0,
            tp_score=0.4,
            oss=0.5,
        )

        reversibility2 = compute_reversibility(metrics_compression, metrics_more_compression)

        # Negative reversibility (ARD made it worse)
        assert reversibility2["cci_reversibility_percent"] is not None
        assert reversibility2["cci_reversibility_percent"] < 0
