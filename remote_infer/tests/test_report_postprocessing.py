from __future__ import annotations

import unittest
from unittest.mock import patch

from remote_infer.utils import postprocess_generated_report_text


class ReportPostprocessingTests(unittest.TestCase):
    def test_non_empty_decoded_continuation_survives_postprocessing(self) -> None:
        result = postprocess_generated_report_text(
            visible_text="",
            raw_text="<pad>Visible fallback text<pad>",
        )

        self.assertEqual(result.normalized_report["report_text"], "<pad>Visible fallback text<pad>")
        self.assertEqual(result.normalized_report["findings"], "<pad>Visible fallback text<pad>")
        self.assertEqual(result.normalized_report["technique"], "")
        self.assertEqual(result.normalized_report["impression"], "")
        self.assertEqual(result.normalization_mode, "report_text_only")
        self.assertEqual(result.text_source, "raw_cleaned_fallback")
        self.assertFalse(result.preserved_raw_text)

    def test_preserves_raw_text_when_cleanup_empties_usable_text(self) -> None:
        with patch("remote_infer.utils.clean_generated_text", return_value=""):
            result = postprocess_generated_report_text(
                visible_text="",
                raw_text="Plain English report text.",
            )

        self.assertEqual(result.normalized_report["report_text"], "Plain English report text.")
        self.assertEqual(result.normalized_report["findings"], "Plain English report text.")
        self.assertEqual(result.normalization_mode, "raw_preserved_after_cleanup")
        self.assertEqual(result.text_source, "raw_raw_preserved")
        self.assertTrue(result.preserved_raw_text)

    def test_preserves_report_text_even_when_section_parsing_fails(self) -> None:
        with patch("remote_infer.utils.normalize_report_sections", side_effect=ValueError("parse failed")):
            result = postprocess_generated_report_text(
                visible_text="Free text without reliable sections.",
                raw_text="",
            )

        self.assertEqual(result.normalized_report["report_text"], "Free text without reliable sections.")
        self.assertEqual(result.normalized_report["findings"], "Free text without reliable sections.")
        self.assertEqual(result.normalized_report["technique"], "")
        self.assertEqual(result.normalized_report["impression"], "")
        self.assertEqual(result.normalization_mode, "report_text_only_after_normalization_exception")

    def test_rejects_unusable_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty after decoding"):
            postprocess_generated_report_text(
                visible_text="",
                raw_text="<pad><eos>",
            )


if __name__ == "__main__":
    unittest.main()
