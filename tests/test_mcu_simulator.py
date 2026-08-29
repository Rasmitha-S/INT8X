"""
Unit tests for What-If MCU Simulator for TinyML Deployment.
Tests preset configurations, custom MCU budgets, utilization calculations,
headroom/shortfall reporting, and pass/fail state transitions.
"""
import json
import unittest

from src.metrics import evaluate_memory_budget, simulate_mcu_resources, get_file_size_bytes


class TestWhatIfMCUSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.int8_model_path = "models/model_int8.tflite"
        cls.analysis_path = "tinyml/model_analysis.json"

        with open(cls.analysis_path, "r", encoding="utf-8") as f:
            cls.analysis_data = json.load(f)

        cls.verified_flash_bytes = cls.analysis_data["verified"]["flash_storage_bytes"]
        cls.estimated_arena_bytes = cls.analysis_data["estimated"]["estimated_tensor_arena_bytes"]

    def test_32kb_flash_32kb_ram_fits(self):
        # 32 KB Flash, 32 KB RAM -> FITS
        res = simulate_mcu_resources(
            available_flash_kb=32.0,
            available_ram_kb=32.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertTrue(res["flash_fits"])
        self.assertTrue(res["ram_fits"])
        self.assertTrue(res["fits_overall"])
        self.assertEqual(res["status_category"], "BOTH_PASS")
        self.assertEqual(res["status_title"], "✓ MODEL FITS")
        self.assertEqual(res["flash_status_str"], "PASS")
        self.assertEqual(res["ram_status_str"], "PASS")
        self.assertAlmostEqual(res["flash_usage_pct"], 42.2, delta=0.2)
        self.assertAlmostEqual(res["ram_usage_pct"], 43.8, delta=0.2)
        self.assertGreater(res["flash_headroom_kb"], 0.0)
        self.assertGreater(res["ram_headroom_kb"], 0.0)

    def test_16kb_flash_16kb_ram_fits(self):
        # 16 KB Flash, 16 KB RAM -> FITS (Tight)
        res = simulate_mcu_resources(
            available_flash_kb=16.0,
            available_ram_kb=16.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertTrue(res["flash_fits"])
        self.assertTrue(res["ram_fits"])
        self.assertTrue(res["fits_overall"])
        self.assertEqual(res["status_category"], "BOTH_PASS")
        self.assertAlmostEqual(res["flash_usage_pct"], 84.4, delta=0.2)
        self.assertAlmostEqual(res["ram_usage_pct"], 87.5, delta=0.2)

    def test_10kb_flash_32kb_ram_flash_fail(self):
        # 10 KB Flash, 32 KB RAM -> Flash FAIL
        res = simulate_mcu_resources(
            available_flash_kb=10.0,
            available_ram_kb=32.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertFalse(res["flash_fits"])
        self.assertTrue(res["ram_fits"])
        self.assertFalse(res["fits_overall"])
        self.assertEqual(res["status_category"], "FLASH_FAIL")
        self.assertEqual(res["status_title"], "⚠ MODEL DOES NOT FIT")
        self.assertEqual(res["flash_status_str"], "FAIL")
        self.assertEqual(res["ram_status_str"], "PASS")
        self.assertGreater(res["flash_shortfall_kb"], 0.0)
        self.assertEqual(res["ram_shortfall_kb"], 0.0)
        self.assertIn("Insufficient Flash capacity", res["status_message"])

    def test_32kb_flash_10kb_ram_ram_fail(self):
        # 32 KB Flash, 10 KB RAM -> RAM FAIL
        res = simulate_mcu_resources(
            available_flash_kb=32.0,
            available_ram_kb=10.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertTrue(res["flash_fits"])
        self.assertFalse(res["ram_fits"])
        self.assertFalse(res["fits_overall"])
        self.assertEqual(res["status_category"], "RAM_FAIL")
        self.assertEqual(res["status_title"], "⚠ MODEL DOES NOT FIT")
        self.assertEqual(res["flash_status_str"], "PASS")
        self.assertEqual(res["ram_status_str"], "FAIL")
        self.assertGreater(res["ram_shortfall_kb"], 0.0)
        self.assertEqual(res["flash_shortfall_kb"], 0.0)
        self.assertIn("Insufficient SRAM for the estimated Tensor Arena", res["status_message"])

    def test_8kb_flash_8kb_ram_both_fail(self):
        # 8 KB Flash, 8 KB RAM -> Both FAIL
        res = simulate_mcu_resources(
            available_flash_kb=8.0,
            available_ram_kb=8.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertFalse(res["flash_fits"])
        self.assertFalse(res["ram_fits"])
        self.assertFalse(res["fits_overall"])
        self.assertEqual(res["status_category"], "BOTH_FAIL")
        self.assertEqual(res["status_title"], "✕ MODEL DOES NOT FIT")
        self.assertEqual(res["flash_status_str"], "FAIL")
        self.assertEqual(res["ram_status_str"], "FAIL")
        self.assertGreater(res["flash_shortfall_kb"], 0.0)
        self.assertGreater(res["ram_shortfall_kb"], 0.0)

    def test_custom_large_budget_fits(self):
        # 1024 KB Flash, 256 KB RAM (e.g. Arduino Nano 33 BLE)
        res = simulate_mcu_resources(
            available_flash_kb=1024.0,
            available_ram_kb=256.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertTrue(res["fits_overall"])
        self.assertEqual(res["status_category"], "BOTH_PASS")
        self.assertLess(res["flash_usage_pct"], 2.0)
        self.assertLess(res["ram_usage_pct"], 6.0)

    def test_zero_and_negative_input_safety(self):
        # 0 KB input
        res_zero = simulate_mcu_resources(
            available_flash_kb=0.0,
            available_ram_kb=0.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertFalse(res_zero["fits_overall"])
        self.assertEqual(res_zero["status_category"], "BOTH_FAIL")

        # Negative input (should be clamped to 0.0 safely)
        res_neg = simulate_mcu_resources(
            available_flash_kb=-32.0,
            available_ram_kb=-32.0,
            model_flash_bytes=self.verified_flash_bytes,
            estimated_arena_bytes=self.estimated_arena_bytes,
        )
        self.assertEqual(res_neg["available_flash_kb"], 0.0)
        self.assertEqual(res_neg["available_ram_kb"], 0.0)
        self.assertFalse(res_neg["fits_overall"])
        self.assertEqual(res_neg["status_category"], "BOTH_FAIL")


if __name__ == "__main__":
    unittest.main()
