from __future__ import annotations

import unittest

from remote_infer.utils import clean_generated_text, normalize_report_sections


class NormalizeReportSectionsTests(unittest.TestCase):
    def test_clean_generated_text_only_normalizes_whitespace(self) -> None:
        raw_text = "\n\nVisible text\n\n\nMore text\n"

        cleaned = clean_generated_text(raw_text)

        self.assertEqual(cleaned, "Visible text\n\nMore text")

    def test_parses_fully_labeled_report(self) -> None:
        raw_text = (
            "Technique:\n"
            "CT abdomen and pelvis without contrast\n\n"
            "Findings:\n"
            "Mild hepatic steatosis. No bowel obstruction.\n\n"
            "Impression:\n"
            "No acute abdominopelvic abnormality.\n"
        )

        normalized = normalize_report_sections(raw_text)

        self.assertEqual(normalized["technique"], "CT abdomen and pelvis without contrast")
        self.assertEqual(normalized["findings"], "Mild hepatic steatosis. No bowel obstruction.")
        self.assertEqual(normalized["impression"], "No acute abdominopelvic abnormality.")
        self.assertEqual(normalized["normalization_mode"], "parsed_sections")
        self.assertEqual(
            normalized["report_text"],
            (
                "Technique:\n"
                "CT abdomen and pelvis without contrast\n\n"
                "Findings:\n"
                "Mild hepatic steatosis. No bowel obstruction.\n\n"
                "Impression:\n"
                "No acute abdominopelvic abnormality."
            ),
        )

    def test_parses_findings_and_impression_without_technique(self) -> None:
        raw_text = (
            "Findings:\n"
            "No focal liver lesion. No ascites.\n\n"
            "Impression:\n"
            "No acute intra-abdominal abnormality.\n"
        )

        normalized = normalize_report_sections(raw_text)

        self.assertEqual(normalized["technique"], "")
        self.assertEqual(normalized["findings"], "No focal liver lesion. No ascites.")
        self.assertEqual(normalized["impression"], "No acute intra-abdominal abnormality.")
        self.assertEqual(normalized["normalization_mode"], "parsed_sections")
        self.assertEqual(normalized["report_text"], raw_text.strip())

    def test_parses_findings_only_text(self) -> None:
        raw_text = "Findings:\nMild hepatic steatosis. No bowel obstruction.\n"

        normalized = normalize_report_sections(raw_text)

        self.assertEqual(normalized["technique"], "")
        self.assertEqual(normalized["findings"], "Mild hepatic steatosis. No bowel obstruction.")
        self.assertEqual(normalized["impression"], "")
        self.assertEqual(normalized["normalization_mode"], "parsed_sections")
        self.assertEqual(normalized["report_text"], raw_text.strip())

    def test_parses_impression_only_text(self) -> None:
        raw_text = "Impression:\nNo acute abdominal abnormality.\n"

        normalized = normalize_report_sections(raw_text)

        self.assertEqual(normalized["technique"], "")
        self.assertEqual(normalized["findings"], "")
        self.assertEqual(normalized["impression"], "No acute abdominal abnormality.")
        self.assertEqual(normalized["normalization_mode"], "parsed_sections")
        self.assertEqual(normalized["report_text"], raw_text.strip())

    def test_falls_back_to_plain_text_as_findings(self) -> None:
        raw_text = "Mild hepatic steatosis. No bowel obstruction. No ascites."

        normalized = normalize_report_sections(raw_text)

        self.assertEqual(normalized["technique"], "")
        self.assertEqual(normalized["findings"], raw_text)
        self.assertEqual(normalized["impression"], "")
        self.assertEqual(normalized["normalization_mode"], "report_text_only")
        self.assertEqual(normalized["report_text"], raw_text)

    def test_handles_mixed_case_headers_and_spacing(self) -> None:
        raw_text = (
            "technique :  \n"
            "CT abdomen pelvis\n\n"
            "FINDINGS: Mild hepatic steatosis.\n"
            "No bowel obstruction.\n\n"
            "impression : No acute abnormality.\n"
        )

        normalized = normalize_report_sections(raw_text)

        self.assertEqual(normalized["technique"], "CT abdomen pelvis")
        self.assertEqual(normalized["findings"], "Mild hepatic steatosis.\nNo bowel obstruction.")
        self.assertEqual(normalized["impression"], "No acute abnormality.")
        self.assertEqual(normalized["normalization_mode"], "parsed_sections")

    def test_whitespace_only_output_still_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Generated report text is empty after cleanup."):
            normalize_report_sections("   \n\t   ")

    def test_header_only_output_is_preserved(self) -> None:
        normalized = normalize_report_sections("Findings:\n\nImpression:\n")

        self.assertEqual(normalized["report_text"], "Findings:\n\nImpression:")
        self.assertEqual(normalized["technique"], "")
        self.assertEqual(normalized["findings"], "")
        self.assertEqual(normalized["impression"], "")
        self.assertEqual(normalized["normalization_mode"], "parsed_empty_sections")


if __name__ == "__main__":
    unittest.main()
