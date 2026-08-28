"""
Metrics recording, persistence, and comparison utilities for PS09 TinyQuant.
Manages raw metrics and computed performance comparisons without hardcoded values.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def get_file_size_bytes(file_path: str) -> int:
    """
    Returns exact file size in bytes.

    Args:
        file_path: Path to the target file.

    Returns:
        File size in bytes as an integer.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.stat().st_size


def get_file_size_kb(file_path: str) -> float:
    """
    Returns exact file size in kilobytes (KB).

    Args:
        file_path: Path to the target file.

    Returns:
        File size in KB rounded to 2 decimal places.
    """
    return round(get_file_size_bytes(file_path) / 1024.0, 2)


def save_metrics_json(metrics: Dict[str, Any], file_path: str = "results/metrics.json") -> str:
    """
    Saves a dictionary of metrics to a JSON file.

    Args:
        metrics: Dictionary of evaluated metrics.
        file_path: Destination path for the metrics JSON.

    Returns:
        The file path where metrics were saved.
    """
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return str(out_path)


def load_metrics_json(file_path: str = "results/metrics.json") -> Dict[str, Any]:
    """
    Loads metrics from a JSON file.

    Args:
        file_path: Path to the metrics JSON file.

    Returns:
        Parsed dictionary of metrics.
    """
    out_path = Path(file_path)
    if not out_path.exists():
        return {}
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_metrics(new_metrics: Dict[str, Any], file_path: str = "results/metrics.json") -> Dict[str, Any]:
    """
    Updates the existing metrics file with new values.

    Args:
        new_metrics: Dictionary containing new or updated key-value pairs.
        file_path: Path to the metrics JSON file.

    Returns:
        The combined metrics dictionary.
    """
    current_metrics = load_metrics_json(file_path)
    current_metrics.update(new_metrics)
    save_metrics_json(current_metrics, file_path)
    return current_metrics
