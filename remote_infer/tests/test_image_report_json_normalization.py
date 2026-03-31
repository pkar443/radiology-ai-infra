from __future__ import annotations

import json
import unittest

from remote_infer.utils import (
    AnchorContext,
    normalize_image_report_json_response,
    postprocess_generated_image_report_text,
)


def _anchor_contexts() -> list[AnchorContext]:
    return [
        AnchorContext(
            anchor_id="A01",
            anchor_label="A01 n",
            center_slice_index=120,
            all_slice_labels=("A01 n-1", "A01 n", "A01 n+1"),
        ),
        AnchorContext(
            anchor_id="A02",
            anchor_label="A02 n",
            center_slice_index=150,
            all_slice_labels=("A02 n-1", "A02 n", "A02 n+1"),
        ),
    ]


class ImageReportJsonNormalizationTests(unittest.TestCase):
    def test_parses_valid_structured_json_output(self) -> None:
        raw_text = json.dumps(
            {
                "report_text": "Technique:\nCT selected anchor review\n\nFindings:\nNo focal acute abnormality.\n\nImpression:\nNo acute abnormality.",
                "technique": "CT selected anchor review",
                "findings": "No focal acute abnormality.",
                "impression": "No acute abnormality.",
                "explanation_summary": "The selected anchor groups do not show a focal acute abnormality.",
                "structured_findings": [
                    {
                        "id": "finding-1",
                        "organ": "abdomen",
                        "label": "No focal acute abnormality",
                        "summary": "No focal acute abnormality is visible in the selected anchor groups.",
                        "explanation": "The reviewed local triplets do not show a focal acute abnormality.",
                        "anchor_slice_index": 120,
                        "anchor_label": "A01 n",
                        "supporting_anchors": ["A01 n-1", "A01 n+1"],
                        "confidence": "low",
                        "evidence": "No focal acute abnormality is visible in the provided local triplet around A01.",
                        "abnormal": False,
                    }
                ],
                "limitations": "Selected anchor-group review only.",
            }
        )

        normalized = normalize_image_report_json_response(raw_text, _anchor_contexts())

        self.assertEqual(normalized["technique"], "CT selected anchor review")
        self.assertEqual(normalized["structured_findings"][0]["anchor_label"], "A01 n")
        self.assertEqual(normalized["normalization_mode"], "json_structured")

    def test_rejects_structured_findings_with_unknown_anchor(self) -> None:
        raw_text = json.dumps(
            {
                "report_text": "Technique:\n\nFindings:\nText\n\nImpression:\nText",
                "technique": "",
                "findings": "Text",
                "impression": "Text",
                "explanation_summary": "Text",
                "structured_findings": [
                    {
                        "id": "finding-1",
                        "organ": "abdomen",
                        "label": "Bad anchor",
                        "summary": "Bad anchor",
                        "explanation": "Bad anchor",
                        "anchor_slice_index": 999,
                        "anchor_label": "A99 n",
                        "supporting_anchors": ["A99 n"],
                        "confidence": "low",
                        "evidence": "Bad anchor",
                        "abnormal": True,
                    }
                ],
                "limitations": "",
            }
        )

        with self.assertRaisesRegex(ValueError, "anchor_slice_index"):
            normalize_image_report_json_response(raw_text, _anchor_contexts())

    def test_falls_back_to_section_text_when_json_is_unavailable(self) -> None:
        result = postprocess_generated_image_report_text(
            visible_text=(
                "Findings:\nNo focal acute abnormality on reviewed slices.\n\n"
                "Impression:\nNo acute abnormality."
            ),
            raw_text="",
            anchor_contexts=_anchor_contexts(),
        )

        self.assertEqual(result.normalized_report["findings"], "No focal acute abnormality on reviewed slices.")
        self.assertEqual(result.normalized_report["impression"], "No acute abnormality.")
        self.assertEqual(result.normalized_report["structured_findings"][0]["anchor_label"], "A01 n")
        self.assertEqual(result.normalization_mode, "section_text_fallback")


if __name__ == "__main__":
    unittest.main()
