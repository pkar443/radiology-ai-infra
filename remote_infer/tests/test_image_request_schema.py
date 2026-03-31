from __future__ import annotations

import unittest

try:
    from pydantic import ValidationError
    from remote_infer.schemas import ImageReportInferRequest
except ModuleNotFoundError:  # pragma: no cover - depends on local test env
    ValidationError = None
    ImageReportInferRequest = None


def _build_image_data_url() -> str:
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0zQAAAAASUVORK5CYII="
    )


def _build_anchor_group(anchor_id: str, center_slice_index: int) -> dict[str, object]:
    triplet = [center_slice_index - 1, center_slice_index, center_slice_index + 1]
    return {
        "anchor_id": anchor_id,
        "anchor_label": f"{anchor_id} n",
        "center_slice_index": center_slice_index,
        "center_sop_instance_uid": f"1.2.3.{center_slice_index}",
        "slice_indices": triplet,
        "slices": [
            {
                "slice_index": triplet[0],
                "relative_position": "n-1",
                "anchor_label": f"{anchor_id} n-1",
                "sop_instance_uid": f"1.2.3.{triplet[0]}",
                "image_data_url": _build_image_data_url(),
            },
            {
                "slice_index": triplet[1],
                "relative_position": "n",
                "anchor_label": f"{anchor_id} n",
                "sop_instance_uid": f"1.2.3.{triplet[1]}",
                "image_data_url": _build_image_data_url(),
            },
            {
                "slice_index": triplet[2],
                "relative_position": "n+1",
                "anchor_label": f"{anchor_id} n+1",
                "sop_instance_uid": f"1.2.3.{triplet[2]}",
                "image_data_url": _build_image_data_url(),
            },
        ],
    }


def _flatten_anchor_groups(anchor_groups: list[dict[str, object]]) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for anchor_group in anchor_groups:
        for slice_item in anchor_group["slices"]:
            flattened.append(
                {
                    **slice_item,
                    "anchor_id": anchor_group["anchor_id"],
                    "center_slice_index": anchor_group["center_slice_index"],
                }
            )
    return flattened


def _build_payload(anchor_groups: list[dict[str, object]] | None = None) -> dict[str, object]:
    anchor_groups = anchor_groups or [_build_anchor_group("A01", 120)]
    return {
        "request_id": "req-123",
        "study_id": "study-123",
        "series_uid": "1.2.840.0.1",
        "modality": "CT",
        "body_part": "Abdomen",
        "clinical_context": "Right upper quadrant pain",
        "instruction": "Instruction: Review the slices in anchor-group order.",
        "query": "Return one JSON object only.",
        "selection_strategy": "deterministic-uniform-non-overlapping-triplets",
        "anchor_group_count": len(anchor_groups),
        "anchor_groups": anchor_groups,
        "slices": _flatten_anchor_groups(anchor_groups),
    }


@unittest.skipIf(ImageReportInferRequest is None or ValidationError is None, "pydantic is not installed")
class ImageReportInferRequestTests(unittest.TestCase):
    def test_accepts_single_anchor_group_payload(self) -> None:
        payload = ImageReportInferRequest(**_build_payload())

        self.assertEqual(payload.anchor_group_count, 1)
        self.assertEqual(len(payload.anchor_groups), 1)
        self.assertEqual(len(payload.slices), 3)

    def test_accepts_multiple_anchor_groups(self) -> None:
        payload = ImageReportInferRequest(
            **_build_payload(
                [
                    _build_anchor_group("A01", 120),
                    _build_anchor_group("A02", 150),
                ]
            )
        )

        self.assertEqual(payload.anchor_group_count, 2)
        self.assertEqual([group.anchor_id for group in payload.anchor_groups], ["A01", "A02"])
        self.assertEqual([slice_item.anchor_label for slice_item in payload.slices[:4]], ["A01 n-1", "A01 n", "A01 n+1", "A02 n-1"])

    def test_rejects_anchor_group_count_mismatch(self) -> None:
        payload = _build_payload()
        payload["anchor_group_count"] = 2

        with self.assertRaisesRegex(ValidationError, "anchor_group_count must match"):
            ImageReportInferRequest(**payload)

    def test_rejects_flattened_slices_that_do_not_match_anchor_groups(self) -> None:
        payload = _build_payload()
        payload["slices"][0]["anchor_label"] = "A99 n-1"

        with self.assertRaisesRegex(ValidationError, "flattened compatibility copy"):
            ImageReportInferRequest(**payload)

    def test_rejects_more_than_six_anchor_groups(self) -> None:
        anchor_groups = [_build_anchor_group(f"A{index:02d}", 120 + (index * 3)) for index in range(1, 8)]

        with self.assertRaises(ValidationError):
            ImageReportInferRequest(**_build_payload(anchor_groups))


if __name__ == "__main__":
    unittest.main()
