from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from remote_infer import app as remote_app
from remote_infer.model_loader import GenerationResult
from remote_infer.schemas import ImageReportInferRequest


def _build_hydrated_payload() -> ImageReportInferRequest:
    return ImageReportInferRequest(
        request_id="orthanc-ref-1",
        study_id="study-1",
        series_uid="2.16.840.1",
        modality="CT",
        body_part="Abdomen",
        clinical_context="Abdominal pain",
        instruction="Instruction: Review the hydrated Orthanc slices in order.",
        query="Return one JSON object only.",
        selection_strategy="deterministic-uniform-non-overlapping-triplets",
        anchor_group_count=1,
        anchor_groups=[
            {
                "anchor_id": "A01",
                "anchor_label": "A01 n",
                "center_slice_index": 10,
                "center_sop_instance_uid": "1.2.3.10",
                "slice_indices": [9, 10, 11],
                "slices": [
                    {
                        "slice_index": 9,
                        "relative_position": "n-1",
                        "anchor_label": "A01 n-1",
                        "sop_instance_uid": "1.2.3.9",
                        "image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0zQAAAAASUVORK5CYII=",
                    },
                    {
                        "slice_index": 10,
                        "relative_position": "n",
                        "anchor_label": "A01 n",
                        "sop_instance_uid": "1.2.3.10",
                        "image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0zQAAAAASUVORK5CYII=",
                    },
                    {
                        "slice_index": 11,
                        "relative_position": "n+1",
                        "anchor_label": "A01 n+1",
                        "sop_instance_uid": "1.2.3.11",
                        "image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0zQAAAAASUVORK5CYII=",
                    },
                ],
            }
        ],
        slices=[
            {
                "slice_index": 9,
                "relative_position": "n-1",
                "anchor_label": "A01 n-1",
                "sop_instance_uid": "1.2.3.9",
                "image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0zQAAAAASUVORK5CYII=",
                "anchor_id": "A01",
                "center_slice_index": 10,
            },
            {
                "slice_index": 10,
                "relative_position": "n",
                "anchor_label": "A01 n",
                "sop_instance_uid": "1.2.3.10",
                "image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0zQAAAAASUVORK5CYII=",
                "anchor_id": "A01",
                "center_slice_index": 10,
            },
            {
                "slice_index": 11,
                "relative_position": "n+1",
                "anchor_label": "A01 n+1",
                "sop_instance_uid": "1.2.3.11",
                "image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0zQAAAAASUVORK5CYII=",
                "anchor_id": "A01",
                "center_slice_index": 10,
            },
        ],
    )


class OrthancReferenceEndpointTests(unittest.TestCase):
    def test_infer_image_report_accepts_orthanc_reference_payload(self) -> None:
        captured: dict[str, object] = {}

        def fake_hydrate(payload, *, default_base_url, username, password):
            captured["payload"] = payload
            captured["default_base_url"] = default_base_url
            captured["username"] = username
            captured["password"] = password
            return _build_hydrated_payload(), {"orthanc_series_id": payload.orthanc_series_id}

        def fake_generate_image_report(messages, generation_config):
            structured = {
                "report_text": "Technique:\nCT abdomen.\n\nFindings:\nNo acute abnormality.\n\nImpression:\nNo acute abnormality.",
                "technique": "CT abdomen.",
                "findings": "No acute abnormality.",
                "impression": "No acute abnormality.",
                "explanation_summary": "No acute abnormality on the hydrated Orthanc slice review.",
                "structured_findings": [
                    {
                        "id": "finding-1",
                        "organ": "abdomen",
                        "label": "No acute abnormality",
                        "summary": "No acute abnormality.",
                        "explanation": "The hydrated Orthanc preview slices do not show a focal acute abnormality.",
                        "anchor_slice_index": 10,
                        "anchor_label": "A01 n",
                        "supporting_anchors": ["A01 n-1", "A01 n+1"],
                        "confidence": "low",
                        "evidence": "No focal acute abnormality on the provided local triplet.",
                        "abnormal": False,
                    }
                ],
                "limitations": "This is a selected anchor-group review, not a full-volume interpretation.",
            }
            structured_text = json.dumps(structured)
            return GenerationResult(
                raw_text=structured_text,
                text=structured_text,
                inference_time_ms=111,
                model_id="test-model",
                device="cuda:0",
                load_state="loaded",
                input_ids_length=10,
                prompt_token_count=10,
                generated_token_count=5,
                generated_sequence_length=15,
                continuation_token_count=5,
                generated_token_ids_head=(1, 2, 3),
                special_only_continuation=False,
                full_text=structured_text,
                decoded_input_text="",
                prompt_echo_removed=False,
                prompt_echo_offset=None,
            )

        payload = {
            "request_id": "orthanc-ref-1",
            "study_id": "study-1",
            "study_instance_uid": "1.2.840.1",
            "series_instance_uid": "2.16.840.1",
            "orthanc_study_id": "orthanc-study-1",
            "orthanc_series_id": "orthanc-series-1",
            "modality": "CT",
            "body_part": "Abdomen",
            "clinical_context": "Abdominal pain",
            "instruction": "Instruction: Review the fetched Orthanc slices in order.",
            "query": "Return one JSON object only.",
            "selection_strategy": "orthanc-reference-by-uid",
            "questionnaire_context": {"body_part": "Abdomen"},
            "study_metadata": {
                "study_id": "study-1",
                "body_part": "Abdomen",
                "orthanc_public_base_url": "http://130.216.239.131:8042",
                "orthanc_remote_access_allowed": True,
            },
            "request_source": "dashboard_auto_report",
        }

        with (
            patch.object(remote_app.service, "load_model", return_value=None),
            patch.object(remote_app, "build_image_request_from_orthanc_reference", side_effect=fake_hydrate),
            patch.object(remote_app.service, "generate_image_report", side_effect=fake_generate_image_report),
        ):
            with TestClient(remote_app.app) as client:
                response = client.post("/infer-image-report", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["technique"], "CT abdomen.")
        self.assertEqual(captured["default_base_url"], remote_app.settings.orthanc_base_url)
        self.assertEqual(captured["username"], remote_app.settings.orthanc_username)
        self.assertEqual(captured["password"], remote_app.settings.orthanc_password)


if __name__ == "__main__":
    unittest.main()
