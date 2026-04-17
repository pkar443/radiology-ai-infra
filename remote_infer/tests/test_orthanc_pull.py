from __future__ import annotations

import unittest
from unittest.mock import patch

from remote_infer.orthanc_pull import build_image_request_from_orthanc_reference
from remote_infer.schemas import OrthancReferenceInferRequest


class OrthancPullTests(unittest.TestCase):
    def test_build_image_request_from_orthanc_reference_hydrates_anchor_groups(self) -> None:
        payload = OrthancReferenceInferRequest(
            request_id="orthanc-ref-2",
            study_id="study-2",
            study_instance_uid="1.2.840.2",
            series_instance_uid="2.16.840.2",
            orthanc_study_id="orthanc-study-2",
            orthanc_series_id="orthanc-series-2",
            modality="CT",
            body_part="Abdomen",
            clinical_context="Pain",
            instruction="Instruction: Review the fetched Orthanc slices in order.",
            query="Return one JSON object only.",
            selection_strategy="orthanc-reference-by-uid",
            study_metadata={
                "study_id": "study-2",
                "body_part": "Abdomen",
                "orthanc_public_base_url": "http://130.216.239.131:8042",
                "orthanc_remote_access_allowed": True,
            },
        )

        def fake_request_json(url: str, username: str, password: str):
            if url.endswith("/series/orthanc-series-2"):
                return {"Instances": ["inst-1", "inst-2", "inst-3", "inst-4", "inst-5"]}
            instance_id = url.split("/")[-2]
            mapping = {
                "inst-1": {"SOPInstanceUID": "1.2.3.1", "InstanceNumber": "1"},
                "inst-2": {"SOPInstanceUID": "1.2.3.2", "InstanceNumber": "2"},
                "inst-3": {"SOPInstanceUID": "1.2.3.3", "InstanceNumber": "3"},
                "inst-4": {"SOPInstanceUID": "1.2.3.4", "InstanceNumber": "4"},
                "inst-5": {"SOPInstanceUID": "1.2.3.5", "InstanceNumber": "5"},
            }
            return mapping[instance_id]

        with (
            patch("remote_infer.orthanc_pull._request_json", side_effect=fake_request_json),
            patch("remote_infer.orthanc_pull._request_bytes", return_value=b"preview-bytes"),
        ):
            hydrated, context = build_image_request_from_orthanc_reference(
                payload,
                default_base_url="http://130.216.239.131:8042",
                username="orthanc",
                password="orthanc",
            )

        self.assertEqual(hydrated.selection_strategy, "deterministic-uniform-non-overlapping-triplets")
        self.assertEqual(hydrated.anchor_group_count, 1)
        self.assertEqual(hydrated.anchor_groups[0].center_slice_index, 2)
        self.assertEqual(hydrated.anchor_groups[0].slices[1].sop_instance_uid, "1.2.3.3")
        self.assertEqual(hydrated.slices[0].anchor_id, "A01")
        self.assertEqual(context["orthanc_series_id"], "orthanc-series-2")
        self.assertEqual(context["resolved_slice_count"], 3)


if __name__ == "__main__":
    unittest.main()
