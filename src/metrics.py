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


def evaluate_memory_budget(
    available_flash_kb: float,
    available_ram_kb: float,
    model_flash_bytes: int = 13824,
    estimated_arena_bytes: int = 14336,
) -> Dict[str, Any]:
    """
    Evaluates whether the INT8 TinyML model fits within a given microcontroller Flash and RAM budget.

    Args:
        available_flash_kb: Target MCU Flash / ROM budget in kilobytes.
        available_ram_kb: Target MCU RAM / SRAM budget in kilobytes.
        model_flash_bytes: Verified INT8 model binary footprint in bytes (default 13,824).
        estimated_arena_bytes: Estimated TFLite Micro tensor arena footprint in bytes (default 14,336).

    Returns:
        Dictionary containing utilization metrics, headroom, status booleans, and explanation strings.
    """
    # Guard against negative values
    avail_flash = max(0.0, float(available_flash_kb))
    avail_ram = max(0.0, float(available_ram_kb))

    model_flash_kb = round(model_flash_bytes / 1024.0, 2)
    estimated_arena_kb = round(estimated_arena_bytes / 1024.0, 2)

    # Flash evaluation
    if avail_flash > 0:
        flash_usage_pct = round((model_flash_kb / avail_flash) * 100.0, 1)
        flash_fits = model_flash_kb <= avail_flash
    else:
        flash_usage_pct = 0.0
        flash_fits = False

    flash_headroom_kb = round(max(0.0, avail_flash - model_flash_kb), 2)
    flash_shortfall_kb = round(max(0.0, model_flash_kb - avail_flash), 2)

    # RAM evaluation
    if avail_ram > 0:
        ram_usage_pct = round((estimated_arena_kb / avail_ram) * 100.0, 1)
        ram_fits = estimated_arena_kb <= avail_ram
    else:
        ram_usage_pct = 0.0
        ram_fits = False

    ram_headroom_kb = round(max(0.0, avail_ram - estimated_arena_kb), 2)
    ram_shortfall_kb = round(max(0.0, estimated_arena_kb - avail_ram), 2)

    fits_overall = flash_fits and ram_fits

    # Explanations
    reasons = []
    if not flash_fits:
        if avail_flash == 0:
            reasons.append("Available Flash budget is 0 KB.")
        else:
            reasons.append(f"Model Flash ({model_flash_kb} KB) exceeds available Flash ({avail_flash} KB) by {flash_shortfall_kb} KB.")
    if not ram_fits:
        if avail_ram == 0:
            reasons.append("Available RAM budget is 0 KB.")
        else:
            reasons.append(f"Estimated Tensor Arena ({estimated_arena_kb} KB) exceeds available RAM ({avail_ram} KB) by {ram_shortfall_kb} KB.")

    if fits_overall:
        summary_status = "✅ FITS"
        explanation = f"Model fits comfortably with {flash_headroom_kb} KB Flash headroom and {ram_headroom_kb} KB RAM headroom remaining."
    else:
        summary_status = "❌ DOES NOT FIT"
        explanation = " ".join(reasons)

    return {
        "available_flash_kb": avail_flash,
        "available_ram_kb": avail_ram,
        "model_flash_bytes": model_flash_bytes,
        "model_flash_kb": model_flash_kb,
        "estimated_arena_bytes": estimated_arena_bytes,
        "estimated_arena_kb": estimated_arena_kb,
        "flash_usage_pct": flash_usage_pct,
        "ram_usage_pct": ram_usage_pct,
        "flash_fits": flash_fits,
        "ram_fits": ram_fits,
        "fits_overall": fits_overall,
        "flash_headroom_kb": flash_headroom_kb,
        "ram_headroom_kb": ram_headroom_kb,
        "flash_shortfall_kb": flash_shortfall_kb,
        "ram_shortfall_kb": ram_shortfall_kb,
        "summary_status": summary_status,
        "explanation": explanation,
    }


def simulate_mcu_resources(
    available_flash_kb: float,
    available_ram_kb: float,
    model_flash_bytes: int = 13824,
    estimated_arena_bytes: int = 14336,
) -> Dict[str, Any]:
    """
    Simulates static MCU resource compatibility for the INT8 TinyML model against target Flash and SRAM constraints.

    Args:
        available_flash_kb: Target MCU Flash / ROM budget in kilobytes.
        available_ram_kb: Target MCU RAM / SRAM budget in kilobytes.
        model_flash_bytes: Verified INT8 model binary footprint in bytes (default 13,824).
        estimated_arena_bytes: Estimated TFLite Micro tensor arena footprint in bytes (default 14,336).

    Returns:
        Dictionary containing simulation results, utilization, headroom/shortfall, status labels, and pass/fail states.
    """
    res = evaluate_memory_budget(
        available_flash_kb=available_flash_kb,
        available_ram_kb=available_ram_kb,
        model_flash_bytes=model_flash_bytes,
        estimated_arena_bytes=estimated_arena_bytes,
    )

    if res["flash_fits"] and res["ram_fits"]:
        status_category = "BOTH_PASS"
        status_title = "✓ MODEL FITS"
        status_message = "This model is compatible with the selected static Flash/RAM budget."
    elif not res["flash_fits"] and res["ram_fits"]:
        status_category = "FLASH_FAIL"
        status_title = "⚠ MODEL DOES NOT FIT"
        status_message = "Insufficient Flash capacity."
    elif res["flash_fits"] and not res["ram_fits"]:
        status_category = "RAM_FAIL"
        status_title = "⚠ MODEL DOES NOT FIT"
        status_message = "Insufficient SRAM for the estimated Tensor Arena."
    else:
        status_category = "BOTH_FAIL"
        status_title = "✕ MODEL DOES NOT FIT"
        status_message = "The selected resource budget is insufficient."

    res.update({
        "status_category": status_category,
        "status_title": status_title,
        "status_message": status_message,
        "flash_status_str": "PASS" if res["flash_fits"] else "FAIL",
        "ram_status_str": "PASS" if res["ram_fits"] else "FAIL",
    })
    return res


