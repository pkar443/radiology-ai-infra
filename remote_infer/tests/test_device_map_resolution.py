from __future__ import annotations

import unittest

from remote_infer.model_loader import resolve_model_device_map


class ResolveModelDeviceMapTests(unittest.TestCase):
    def test_defaults_to_single_gpu_map_when_cuda_available(self) -> None:
        self.assertEqual(resolve_model_device_map("single", cuda_available=True), {"": 0})

    def test_accepts_auto_when_cuda_available(self) -> None:
        self.assertEqual(resolve_model_device_map("auto", cuda_available=True), "auto")

    def test_falls_back_to_cpu_when_cuda_unavailable(self) -> None:
        self.assertEqual(resolve_model_device_map("single", cuda_available=False), "cpu")
        self.assertEqual(resolve_model_device_map("auto", cuda_available=False), "cpu")

    def test_rejects_unknown_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported MEDGEMMA_DEVICE_MAP"):
            resolve_model_device_map("balanced", cuda_available=True)


if __name__ == "__main__":
    unittest.main()
