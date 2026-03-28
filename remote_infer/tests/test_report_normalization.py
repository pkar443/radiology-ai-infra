from __future__ import annotations

import unittest

from remote_infer.utils import normalize_report_sections


class NormalizeReportSectionsTests(unittest.TestCase):
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
        self.assertTrue(normalized["report_text"].startswith("Technique:\n\nFindings:\n"))

    def test_falls_back_to_plain_text_as_findings(self) -> None:
        raw_text = "Mild hepatic steatosis. No bowel obstruction. No ascites."

        normalized = normalize_report_sections(raw_text)

        self.assertEqual(normalized["technique"], "")
        self.assertEqual(normalized["findings"], raw_text)
        self.assertEqual(normalized["impression"], "")
        self.assertEqual(
            normalized["report_text"],
            (
                "Technique:\n\n"
                "Findings:\n"
                "Mild hepatic steatosis. No bowel obstruction. No ascites.\n\n"
                "Impression:"
            ),
        )

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


if __name__ == "__main__":
    unittest.main()
