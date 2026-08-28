"""
Unit tests for the TinyML Memory Budget Checker (Feature 1).
Tests budget calculations, edge cases, utilization percentages, headroom/shortfall math,
and regression safety against existing model binaries and metrics.
"""
import json
import os
import unittest
from pathlib import Path

from src.metrics import evaluate_memory_budget, get_file_size_bytes


class TestMemoryBudgetChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.int8_model_path = "models/model_int8.tflite"
        cls.analysis_path = "tinyml/model_analysis.json"

        # Load ground truth values from static analysis
        with open(cls.analysis_path, "r", encoding="utf-8") as f:
            cls.analysis_data = json.load(f)

        cls.verified_flash_bytes = cls.analysis_data["verified"]["flash_storage_bytes"]
        cls.estimated_arena_bytes = cls.analysis_data["estimated"]["estimated_tensor_arena_bytes"]

    def test_verified_constants(self):
        # Ensure INT8 model binary matches 13,824 bytes
        actual_bytes = get_file_size_bytes(self.int8_model_path)
        self.assertEqual(actual_bytes, 13824)
        self.assertEqual(self.verified_flash_bytes, 13824)
        self.assertEqual(self.estimated_arena_bytes, 14336)

    def test_sufficient_budget_standard_32kb(self):
        # 32 KB Flash, 32 KB RAM
        res = evaluate_memory_budget(
            available_flash_kb=32.0,
            available_ram_kb=32.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )

        self.assertTrue(res["flash_fits"])
        self.assertTrue(res["ram_fits"])
        self.assertTrue(res["fits_overall"])
        self.assertEqual(res["summary_status"], "✅ FITS")

        # Flash usage: 13.5 / 32 = 42.1875% -> 42.2%
        self.assertAlmostEqual(res["flash_usage_pct"], 42.2, delta=0.2)
        # RAM usage: 14.0 / 32 = 43.75% -> 43.8%
        self.assertAlmostEqual(res["ram_usage_pct"], 43.8, delta=0.2)

        self.assertGreater(res["flash_headroom_kb"], 0.0)
        self.assertGreater(res["ram_headroom_kb"], 0.0)
        self.assertEqual(res["flash_shortfall_kb"], 0.0)
        self.assertEqual(res["ram_shortfall_kb"], 0.0)

    def test_sufficient_budget_tight_16kb(self):
        # 16 KB Flash, 16 KB RAM
        res = evaluate_memory_budget(
            available_flash_kb=16.0,
            available_ram_kb=16.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )

        self.assertTrue(res["flash_fits"])
        self.assertTrue(res["ram_fits"])
        self.assertTrue(res["fits_overall"])
        self.assertEqual(res["summary_status"], "✅ FITS")
        # Flash: 13.5 / 16 = 84.4%
        self.assertAlmostEqual(res["flash_usage_pct"], 84.4, delta=0.2)
        # RAM: 14.0 / 16 = 87.5%
        self.assertAlmostEqual(res["ram_usage_pct"], 87.5, delta=0.2)

    def test_insufficient_flash_case(self):
        # 10 KB Flash, 32 KB RAM
        res = evaluate_memory_budget(
            available_flash_kb=10.0,
            available_ram_kb=32.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )

        self.assertFalse(res["flash_fits"])
        self.assertTrue(res["ram_fits"])
        self.assertFalse(res["fits_overall"])
        self.assertEqual(res["summary_status"], "❌ DOES NOT FIT")
        self.assertGreater(res["flash_shortfall_kb"], 0.0)
        self.assertEqual(res["flash_headroom_kb"], 0.0)
        self.assertIn("exceeds available Flash", res["explanation"])

    def test_insufficient_ram_case(self):
        # 32 KB Flash, 10 KB RAM
        res = evaluate_memory_budget(
            available_flash_kb=32.0,
            available_ram_kb=10.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )

        self.assertTrue(res["flash_fits"])
        self.assertFalse(res["ram_fits"])
        self.assertFalse(res["fits_overall"])
        self.assertEqual(res["summary_status"], "❌ DOES NOT FIT")
        self.assertGreater(res["ram_shortfall_kb"], 0.0)
        self.assertEqual(res["ram_headroom_kb"], 0.0)
        self.assertIn("exceeds available RAM", res["explanation"])

    def test_insufficient_both_flash_and_ram(self):
        # 8 KB Flash, 8 KB RAM
        res = evaluate_memory_budget(
            available_flash_kb=8.0,
            available_ram_kb=8.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )

        self.assertFalse(res["flash_fits"])
        self.assertFalse(res["ram_fits"])
        self.assertFalse(res["fits_overall"])
        self.assertEqual(res["summary_status"], "❌ DOES NOT FIT")
        self.assertIn("exceeds available Flash", res["explanation"])
        self.assertIn("exceeds available RAM", res["explanation"])

    def test_edge_cases_zero_and_negative(self):
        # Zero budget handling without division by zero crash
        res_zero = evaluate_memory_budget(
            available_flash_kb=0.0,
            available_ram_kb=0.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertFalse(res_zero["flash_fits"])
        self.assertFalse(res_zero["ram_fits"])
        self.assertFalse(res_zero["fits_overall"])

        # Negative input should be clamped to 0.0 without errors
        res_neg = evaluate_memory_budget(
            available_flash_kb=-16.0,
            available_ram_kb=-16.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertEqual(res_neg["available_flash_kb"], 0.0)
        self.assertEqual(res_neg["available_ram_kb"], 0.0)
        self.assertFalse(res_neg["fits_overall"])


if __name__ == "__main__":
    unittest.main()
