from __future__ import annotations

import base64
import io
import unittest
from types import SimpleNamespace

from PIL import Image

from remote_infer.utils import (
    decode_image_data_url,
    decode_image_report_slices,
    strip_prompt_echo_from_generated_text,
)


def _build_image_data_url(*, mode: str = "RGB", image_format: str = "JPEG") -> str:
    color = 180 if mode == "L" else (32, 128, 224)
    image = Image.new(mode, (4, 4), color)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


class ImageUtilsTests(unittest.TestCase):
    def test_decode_image_data_url_loads_image_and_converts_to_rgb(self) -> None:
        image = decode_image_data_url(_build_image_data_url(mode="L", image_format="JPEG"))

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (4, 4))

    def test_decode_image_data_url_rejects_invalid_base64(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            decode_image_data_url("data:image/jpeg;base64,abc")

    def test_decode_image_report_slices_preserves_input_order(self) -> None:
        slice_items = [
            SimpleNamespace(
                slice_index=121,
                relative_position="n+1",
                anchor_label="A01 n+1",
                sop_instance_uid="1.2.3.121",
                anchor_id="A01",
                center_slice_index=120,
                image_data_url=_build_image_data_url(),
            ),
            SimpleNamespace(
                slice_index=119,
                relative_position="n-1",
                anchor_label="A01 n-1",
                sop_instance_uid="1.2.3.119",
                anchor_id="A01",
                center_slice_index=120,
                image_data_url=_build_image_data_url(),
            ),
        ]

        decoded = decode_image_report_slices(slice_items)

        self.assertEqual([slice_item.slice_index for slice_item in decoded], [121, 119])
        self.assertEqual([slice_item.relative_position for slice_item in decoded], ["n+1", "n-1"])
        self.assertEqual([slice_item.anchor_label for slice_item in decoded], ["A01 n+1", "A01 n-1"])
        self.assertTrue(all(isinstance(slice_item.image, str) for slice_item in decoded))
        self.assertTrue(all(slice_item.image.startswith("data:image/jpeg;base64,") for slice_item in decoded))
        self.assertTrue(all(getattr(slice_item.validated_image, "mode", None) == "RGB" for slice_item in decoded))

    def test_decode_image_report_slices_includes_slice_context_in_errors(self) -> None:
        slice_items = [
            SimpleNamespace(
                slice_index=120,
                relative_position="n",
                anchor_label="A01 n",
                image_data_url="data:image/jpeg;base64,abc",
            )
        ]

        with self.assertRaisesRegex(ValueError, r"Slice 120 \(A01 n\) image_data_url is invalid"):
            decode_image_report_slices(slice_items)

    def test_strip_prompt_echo_removes_prompt_prefix(self) -> None:
        result = strip_prompt_echo_from_generated_text(
            "Prompt text here\n\nFindings:\nNo acute abnormality.",
            "Prompt text here",
        )

        self.assertTrue(result.prompt_echo_removed)
        self.assertEqual(result.text, "Findings:\nNo acute abnormality.")


if __name__ == "__main__":
    unittest.main()
