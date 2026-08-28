"""
Unit tests for TinyML preparation and verification (Phase 8).
Verifies C-array generation, binary-to-C-array byte-level parity, static model analysis,
operator support validation, and categorical separation of Verified / Estimated / Not Verified metrics.
"""
import json
import os
import re
import unittest
from pathlib import Path

import tensorflow as tf

from src.metrics import get_file_size_bytes


class TestTinyMLPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tflite_path = "models/model_int8.tflite"
        cls.header_path = "tinyml/model_data.h"
        cls.source_path = "tinyml/model_data.cc"
        cls.analysis_path = "tinyml/model_analysis.json"

    def test_tinyml_files_exist(self):
        self.assertTrue(os.path.exists(self.tflite_path))
        self.assertTrue(os.path.exists(self.header_path))
        self.assertTrue(os.path.exists(self.source_path))
        self.assertTrue(os.path.exists(self.analysis_path))

    def test_c_array_length_and_binary_integrity(self):
        # 1. Read original binary bytes
        with open(self.tflite_path, "rb") as f:
            original_bytes = f.read()
        original_len = len(original_bytes)
        self.assertEqual(original_len, 13824)

        # 2. Parse hex values from generated .cc source file
        with open(self.source_path, "r", encoding="utf-8") as f:
            cc_content = f.read()

        hex_matches = re.findall(r"0x([0-9a-fA-F]{2})", cc_content)
        parsed_bytes = bytes(int(h, 16) for h in hex_matches)

        # 3. Verify length match
        self.assertEqual(len(parsed_bytes), original_len, "C array length must match .tflite file size")

        # 4. Verify byte-for-byte equality
        self.assertEqual(parsed_bytes, original_bytes, "C array content must be byte-for-byte identical to .tflite binary")

    def test_c_header_structure(self):
        with open(self.header_path, "r", encoding="utf-8") as f:
            header_content = f.read()

        self.assertIn("#ifndef MODEL_DATA_H_", header_content)
        self.assertIn("extern const unsigned char g_model_int8_tflite[];", header_content)
        self.assertIn("extern const unsigned int g_model_int8_tflite_len;", header_content)

    def test_model_analysis_structure(self):
        with open(self.analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)

        self.assertIn("verified", analysis)
        self.assertIn("estimated", analysis)
        self.assertIn("not_verified", analysis)

        verified = analysis["verified"]
        estimated = analysis["estimated"]
        not_verified = analysis["not_verified"]

        # Verified checks
        self.assertEqual(verified["flash_storage_bytes"], 13824)
        self.assertEqual(verified["input_tensor"]["dtype"], "int8")
        self.assertEqual(verified["output_tensor"]["dtype"], "int8")
        self.assertEqual(verified["input_tensor"]["shape"], [1, 28, 28, 1])
        self.assertEqual(verified["output_tensor"]["shape"], [1, 10])
        self.assertTrue(verified["all_ops_supported_in_tflm"])

        # Estimated checks
        self.assertGreater(estimated["estimated_tensor_arena_bytes"], 0)
        self.assertIn("estimation_methodology", estimated)

        # Not verified checks
        self.assertFalse(not_verified["physical_mcu_deployment"])
        self.assertIsNone(not_verified["physical_mcu_cycle_count"])


if __name__ == "__main__":
    unittest.main()
