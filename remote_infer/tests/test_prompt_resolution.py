from __future__ import annotations

import unittest

from remote_infer.utils import resolve_report_prompt


class ResolveReportPromptTests(unittest.TestCase):
    def test_prefers_direct_prompt(self) -> None:
        prompt = resolve_report_prompt(
            prompt="  Use this prompt directly.  ",
            study_id="study-1",
            modality="CT",
            body_part="Abdomen",
            clinical_context="Pain.",
            findings_input="No acute abnormality.",
        )

        self.assertEqual(prompt, "Use this prompt directly.")

    def test_builds_minimal_legacy_passthrough(self) -> None:
        prompt = resolve_report_prompt(
            prompt=None,
            study_id="study-1",
            modality="CT",
            body_part="Abdomen",
            clinical_context="Pain.",
            findings_input="No acute abnormality.",
        )

        self.assertEqual(
            prompt,
            (
                "Study ID: study-1\n"
                "Modality: CT\n"
                "Body part: Abdomen\n"
                "Clinical context: Pain.\n"
                "Findings input:\n"
                "No acute abnormality."
            ),
        )

    def test_rejects_empty_prompt_and_empty_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "Either prompt or report input fields must be provided."):
            resolve_report_prompt(
                prompt="   ",
                study_id=None,
                modality=None,
                body_part=None,
                clinical_context=None,
                findings_input=None,
            )


if __name__ == "__main__":
    unittest.main()
