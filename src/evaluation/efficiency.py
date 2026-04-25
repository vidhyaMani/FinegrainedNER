"""Efficiency metrics for NER model evaluation.

Computes:
- Parameter count for each model
- Training time (from logs or measured)
- Inference latency (avg ms per query)
- Peak memory usage (GPU/CPU)
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in a PyTorch model.

    Args:
        model: PyTorch model.

    Returns:
        Dict with total_params, trainable_params, non_trainable_params.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
    }


def measure_inference_latency(
    model,
    test_samples: List[List[str]],
    num_warmup: int = 10,
    num_runs: int = 100,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Measure inference latency for a model.

    Args:
        model: Model with a predict() method.
        test_samples: List of token sequences for testing.
        num_warmup: Number of warmup runs (not counted).
        num_runs: Number of runs to average over.
        batch_size: Batch size for inference.

    Returns:
        Dict with avg_latency_ms, std_latency_ms, throughput_queries_per_sec.
    """
    import numpy as np

    # Use a subset of test samples
    samples = test_samples[:min(len(test_samples), num_runs + num_warmup)]

    # Warmup
    for i in range(min(num_warmup, len(samples))):
        _ = model.predict([samples[i]])

    # Measure
    latencies = []
    for i in range(num_warmup, min(num_warmup + num_runs, len(samples))):
        start = time.perf_counter()
        _ = model.predict([samples[i]])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    throughput = 1000 / avg_latency if avg_latency > 0 else 0  # queries/sec

    return {
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": float(np.min(latencies)) if len(latencies) > 0 else 0,
        "max_latency_ms": float(np.max(latencies)) if len(latencies) > 0 else 0,
        "throughput_queries_per_sec": throughput,
        "num_samples_measured": len(latencies),
    }


def measure_peak_memory(
    model,
    test_samples: List[List[str]],
    device: str = "auto",
) -> Dict[str, float]:
    """Measure peak memory usage during inference.

    Args:
        model: Model with a predict() method.
        test_samples: List of token sequences for testing.
        device: Device to use ('cuda', 'cpu', or 'auto').

    Returns:
        Dict with peak_memory_mb, allocated_memory_mb.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear cache
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Run inference
    samples = test_samples[:min(100, len(test_samples))]
    _ = model.predict(samples)

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    else:
        # For CPU, we can't easily measure memory; return -1
        peak_memory = -1
        allocated_memory = -1

    return {
        "peak_memory_mb": peak_memory,
        "allocated_memory_mb": allocated_memory,
        "device": device,
    }


def parse_training_time_from_log(log_path: Path) -> Optional[float]:
    """Parse training time from a training log file.

    Args:
        log_path: Path to training log file.

    Returns:
        Training time in seconds, or None if not found.
    """
    if not log_path.exists():
        return None

    import re

    with open(log_path, "r") as f:
        content = f.read()

    # Try different patterns
    patterns = [
        r"Training.*completed in (\d+\.?\d*)\s*seconds",
        r"Total training time:\s*(\d+\.?\d*)\s*s",
        r"train_runtime['\"]?\s*[:=]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*seconds.*total",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def collect_efficiency_metrics(
    model,
    model_name: str,
    test_samples: List[List[str]],
    log_dir: Optional[Path] = None,
    num_latency_runs: int = 100,
) -> Dict[str, Any]:
    """Collect all efficiency metrics for a model.

    Args:
        model: Model with a predict() method and internal torch model.
        model_name: Name of the model (for log file lookup).
        test_samples: List of token sequences for testing.
        log_dir: Directory containing training logs.
        num_latency_runs: Number of runs for latency measurement.

    Returns:
        Dict with all efficiency metrics.
    """
    metrics: Dict[str, Any] = {"model": model_name}

    # 1. Parameter count
    try:
        if hasattr(model, "model"):
            param_metrics = count_parameters(model.model)
        elif hasattr(model, "_model"):
            param_metrics = count_parameters(model._model)
        else:
            param_metrics = {"total_params": -1, "trainable_params": -1}
        metrics.update(param_metrics)
    except Exception as e:
        logger.warning(f"Failed to count parameters for {model_name}: {e}")
        metrics["total_params"] = -1
        metrics["trainable_params"] = -1

    # 2. Inference latency
    try:
        latency_metrics = measure_inference_latency(
            model, test_samples, num_runs=num_latency_runs
        )
        metrics.update(latency_metrics)
    except Exception as e:
        logger.warning(f"Failed to measure latency for {model_name}: {e}")
        metrics["avg_latency_ms"] = -1

    # 3. Peak memory
    try:
        memory_metrics = measure_peak_memory(model, test_samples)
        metrics.update(memory_metrics)
    except Exception as e:
        logger.warning(f"Failed to measure memory for {model_name}: {e}")
        metrics["peak_memory_mb"] = -1

    # 4. Training time from logs
    if log_dir:
        log_patterns = [
            f"train_{model_name}.log",
            f"{model_name}_train.log",
            f"train_{model_name.replace('_', '')}.log",
        ]
        for pattern in log_patterns:
            log_path = log_dir / pattern
            training_time = parse_training_time_from_log(log_path)
            if training_time is not None:
                metrics["training_time_seconds"] = training_time
                metrics["training_time_minutes"] = training_time / 60
                break
        else:
            metrics["training_time_seconds"] = -1

    return metrics


def export_efficiency_csv(
    efficiency_results: List[Dict[str, Any]],
    output_path: str,
):
    """Export efficiency metrics to CSV.

    Args:
        efficiency_results: List of efficiency metric dicts.
        output_path: Path to output CSV file.
    """
    import pandas as pd

    df = pd.DataFrame(efficiency_results)

    # Reorder columns
    column_order = [
        "model",
        "total_params",
        "total_params_millions",
        "trainable_params",
        "trainable_params_millions",
        "training_time_seconds",
        "training_time_minutes",
        "avg_latency_ms",
        "std_latency_ms",
        "throughput_queries_per_sec",
        "peak_memory_mb",
    ]
    existing_cols = [c for c in column_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in column_order]
    df = df[existing_cols + other_cols]

    df.to_csv(output_path, index=False)
    logger.info(f"Saved efficiency metrics to {output_path}")


