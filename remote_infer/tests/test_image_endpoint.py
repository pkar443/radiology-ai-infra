from __future__ import annotations

import base64
import io
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from remote_infer import app as remote_app
from remote_infer.model_loader import GenerationResult


def _build_image_data_url(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (8, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_anchor_group(anchor_id: str, center_slice_index: int, base_color: tuple[int, int, int]) -> dict[str, object]:
    triplet = [center_slice_index - 1, center_slice_index, center_slice_index + 1]
    colors = [
        base_color,
        tuple(min(channel + 16, 255) for channel in base_color),
        tuple(min(channel + 32, 255) for channel in base_color),
    ]
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
                "image_data_url": _build_image_data_url(colors[0]),
            },
            {
                "slice_index": triplet[1],
                "relative_position": "n",
                "anchor_label": f"{anchor_id} n",
                "sop_instance_uid": f"1.2.3.{triplet[1]}",
                "image_data_url": _build_image_data_url(colors[1]),
            },
            {
                "slice_index": triplet[2],
                "relative_position": "n+1",
                "anchor_label": f"{anchor_id} n+1",
                "sop_instance_uid": f"1.2.3.{triplet[2]}",
                "image_data_url": _build_image_data_url(colors[2]),
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


def _build_payload() -> dict[str, object]:
    anchor_groups = [
        _build_anchor_group("A01", 120, (32, 64, 96)),
        _build_anchor_group("A02", 150, (64, 96, 128)),
    ]
    return {
        "request_id": "api-image-test-1",
        "study_id": "3",
        "series_uid": "1.2.3.4",
        "modality": "CT",
        "body_part": "Abdomen",
        "clinical_context": "Pain",
        "instruction": "Instruction: Review the provided anchor groups in order.",
        "query": "Return one JSON object only.",
        "selection_strategy": "deterministic-uniform-non-overlapping-triplets",
        "anchor_group_count": len(anchor_groups),
        "anchor_groups": anchor_groups,
        "slices": _flatten_anchor_groups(anchor_groups),
    }


class ImageEndpointTests(unittest.TestCase):
    def test_infer_image_report_receives_anchor_group_payload_and_returns_structured_response(self) -> None:
        payload = _build_payload()
        captured: dict[str, object] = {}

        def fake_generate_image_report(messages: list[dict[str, object]], generation_config: object) -> GenerationResult:
            captured["messages"] = messages
            captured["generation_config"] = generation_config
            structured = {
                "report_text": (
                    "Technique:\nSelected CT anchor-group review of provided slices.\n\n"
                    "Findings:\nNo focal acute abnormality is visible in the provided local slice groups.\n\n"
                    "Impression:\nNo acute abnormality identified in the reviewed anchor groups."
                ),
                "technique": "Selected CT anchor-group review of provided slices.",
                "findings": "No focal acute abnormality is visible in the provided local slice groups.",
                "impression": "No acute abnormality identified in the reviewed anchor groups.",
                "explanation_summary": "The reviewed anchor groups do not show a focal acute abnormality in this limited explainable CT context set.",
                "structured_findings": [
                    {
                        "id": "finding-1",
                        "organ": "abdomen",
                        "label": "No focal acute abnormality in reviewed anchor groups",
                        "summary": "No focal acute abnormality is visible in the reviewed local slice groups.",
                        "explanation": "The center slice A01 n is used as the primary anchor, with adjacent slices used as local confirmation only.",
                        "anchor_slice_index": 120,
                        "anchor_label": "A01 n",
                        "supporting_anchors": ["A01 n-1", "A01 n+1"],
                        "confidence": "low",
                        "evidence": "No focal acute abnormality is visible across the provided local triplet around A01.",
                        "abnormal": False,
                    }
                ],
                "limitations": "This is a selected anchor-group review, not a full-volume interpretation.",
            }
            structured_text = json.dumps(structured)
            return GenerationResult(
                raw_text=structured_text,
                text=structured_text,
                inference_time_ms=321,
                model_id="test-model",
                device="cuda:0",
                load_state="loaded",
                input_ids_length=100,
                prompt_token_count=100,
                generated_token_count=30,
                generated_sequence_length=130,
                continuation_token_count=30,
                generated_token_ids_head=(1, 2, 3),
                special_only_continuation=False,
                full_text=structured_text,
                decoded_input_text="",
                prompt_echo_removed=False,
                prompt_echo_offset=None,
            )

        with (
            patch.object(remote_app.service, "load_model", return_value=None),
            patch.object(remote_app.service, "generate_image_report", side_effect=fake_generate_image_report),
        ):
            with TestClient(remote_app.app) as client:
                response = client.post("/infer-image-report", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["request_id"], "api-image-test-1")
        self.assertEqual(body["technique"], "Selected CT anchor-group review of provided slices.")
        self.assertEqual(body["findings"], "No focal acute abnormality is visible in the provided local slice groups.")
        self.assertEqual(body["impression"], "No acute abnormality identified in the reviewed anchor groups.")
        self.assertEqual(body["explanation_summary"], "The reviewed anchor groups do not show a focal acute abnormality in this limited explainable CT context set.")
        self.assertEqual(body["structured_findings"][0]["anchor_label"], "A01 n")
        self.assertEqual(body["model_id"], "test-model")

        messages = captured["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": payload["instruction"]})
        self.assertEqual(content[1], {"type": "image", "image": payload["anchor_groups"][0]["slices"][0]["image_data_url"]})
        self.assertEqual(content[2], {"type": "text", "text": "SLICE A01 n-1 (index 119)"})
        self.assertEqual(content[3], {"type": "image", "image": payload["anchor_groups"][0]["slices"][1]["image_data_url"]})
        self.assertEqual(content[4], {"type": "text", "text": "SLICE A01 n (index 120)"})
        self.assertEqual(content[8], {"type": "text", "text": "SLICE A02 n-1 (index 149)"})
        self.assertEqual(content[-1], {"type": "text", "text": payload["query"]})


if __name__ == "__main__":
    unittest.main()
