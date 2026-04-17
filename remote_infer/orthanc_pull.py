from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .model_loader import InvalidPayloadError
from .schemas import ImageReportInferRequest, OrthancReferenceInferRequest


@dataclass(frozen=True)
class OrthancSliceRef:
    slice_index: int
    orthanc_instance_id: str
    sop_instance_uid: str
    instance_number: int | None
    position_along_normal: float | None


def _parse_number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_int(value: object) -> int | None:
    parsed = _parse_number(value)
    if parsed is None:
        return None
    return int(parsed)


def _parse_float_list(value: object) -> list[float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).replace("\\", ",").split(",")
    floats: list[float] = []
    for part in parts:
        parsed = _parse_number(part)
        if parsed is None:
            return None
        floats.append(parsed)
    return floats or None


def _cross_product(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _position_along_normal(
    image_position_patient: list[float] | None,
    image_orientation_patient: list[float] | None,
) -> float | None:
    if not image_position_patient or not image_orientation_patient or len(image_position_patient) != 3 or len(image_orientation_patient) != 6:
        return None
    normal = _cross_product(image_orientation_patient[:3], image_orientation_patient[3:])
    return sum(image_position_patient[idx] * normal[idx] for idx in range(3))


def _normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise InvalidPayloadError("Orthanc base URL is not configured.")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InvalidPayloadError("Orthanc base URL must be a valid http(s) URL.")
    return normalized


def _basic_auth_header(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def _request_bytes(url: str, username: str, password: str, *, accept: str) -> bytes:
    request = Request(
        url,
        headers={
            "Authorization": _basic_auth_header(username, password),
            "Accept": accept,
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        message = detail or exc.reason or "HTTP error"
        raise InvalidPayloadError(f"Orthanc request failed for {url}: {exc.code} {message}") from exc
    except URLError as exc:
        raise InvalidPayloadError(f"Orthanc is unreachable at {url}: {exc.reason}") from exc


def _request_json(url: str, username: str, password: str) -> dict[str, Any]:
    payload = _request_bytes(url, username, password, accept="application/json")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidPayloadError(f"Orthanc returned invalid JSON for {url}.") from exc
    if not isinstance(decoded, dict):
        raise InvalidPayloadError(f"Orthanc returned an unexpected JSON shape for {url}.")
    return decoded


def _request_json_list(url: str, username: str, password: str) -> list[Any]:
    payload = _request_bytes(url, username, password, accept="application/json")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidPayloadError(f"Orthanc returned invalid JSON for {url}.") from exc
    if not isinstance(decoded, list):
        raise InvalidPayloadError(f"Orthanc returned an unexpected JSON shape for {url}.")
    return decoded


def _preview_to_data_url(preview_bytes: bytes) -> str:
    encoded = base64.b64encode(preview_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _sort_series_instances(
    base_url: str,
    username: str,
    password: str,
    orthanc_series_id: str,
) -> list[OrthancSliceRef]:
    series = _request_json(f"{base_url}/series/{orthanc_series_id}", username, password)
    instance_ids = series.get("Instances") or []
    if not isinstance(instance_ids, list) or not instance_ids:
        raise InvalidPayloadError("Orthanc series does not contain any instances.")

    provisional: list[OrthancSliceRef] = []
    for fallback_index, instance_id in enumerate(instance_ids):
        if not isinstance(instance_id, str) or not instance_id.strip():
            continue
        tags = _request_json(f"{base_url}/instances/{instance_id}/simplified-tags", username, password)
        sop_instance_uid = str(tags.get("SOPInstanceUID") or "").strip()
        if not sop_instance_uid:
            raise InvalidPayloadError(f"Orthanc instance {instance_id} is missing SOPInstanceUID.")
        orientation = _parse_float_list(tags.get("ImageOrientationPatient"))
        position = _parse_float_list(tags.get("ImagePositionPatient"))
        provisional.append(
            OrthancSliceRef(
                slice_index=fallback_index,
                orthanc_instance_id=instance_id,
                sop_instance_uid=sop_instance_uid,
                instance_number=_parse_int(tags.get("InstanceNumber")),
                position_along_normal=_position_along_normal(position, orientation),
            )
        )

    ordered = sorted(
        provisional,
        key=lambda item: (
            item.position_along_normal is None,
            item.position_along_normal if item.position_along_normal is not None else 0.0,
            item.instance_number if item.instance_number is not None else item.slice_index,
            item.slice_index,
        ),
    )
    return [
        OrthancSliceRef(
            slice_index=index,
            orthanc_instance_id=item.orthanc_instance_id,
            sop_instance_uid=item.sop_instance_uid,
            instance_number=item.instance_number,
            position_along_normal=item.position_along_normal,
        )
        for index, item in enumerate(ordered)
    ]


def _select_reference_center_indices(total_slices: int, target_groups: int = 6) -> list[int]:
    if total_slices <= 0:
        return []
    if total_slices <= 3:
        return [total_slices // 2]
    max_groups = max(1, total_slices // 3)
    group_count = min(target_groups, max_groups)
    if group_count == 1:
        center = min(max(1, total_slices // 2), total_slices - 2)
        return [center]
    start = 1
    end = total_slices - 2
    step = (end - start) / (group_count - 1)
    centers: list[int] = []
    for idx in range(group_count):
        center = int(round(start + idx * step))
        center = min(max(1, center), total_slices - 2)
        if centers and center <= centers[-1] + 1:
            center = min(total_slices - 2, centers[-1] + 2)
        if centers and center <= centers[-1]:
            continue
        centers.append(center)
    return centers or [min(max(1, total_slices // 2), total_slices - 2)]


def build_image_request_from_orthanc_reference(
    payload: OrthancReferenceInferRequest,
    *,
    default_base_url: str,
    username: str,
    password: str,
) -> tuple[ImageReportInferRequest, dict[str, Any]]:
    metadata = payload.study_metadata
    base_url = _normalize_base_url(
        (metadata.orthanc_public_base_url if metadata and metadata.orthanc_public_base_url else default_base_url)
    )
    if not username or not password:
        raise InvalidPayloadError("Orthanc credentials are not configured on Hades.")

    ordered_slices = _sort_series_instances(base_url, username, password, payload.orthanc_series_id)
    center_indices = _select_reference_center_indices(len(ordered_slices))

    anchor_groups: list[dict[str, Any]] = []
    flat_slices: list[dict[str, Any]] = []
    for anchor_number, center_index in enumerate(center_indices, start=1):
        anchor_id = f"A{anchor_number:02d}"
        group_slices: list[dict[str, Any]] = []
        for offset, relative_position in [(-1, "n-1"), (0, "n"), (1, "n+1")]:
            slice_ref = ordered_slices[center_index + offset]
            preview_bytes = _request_bytes(
                f"{base_url}/instances/{slice_ref.orthanc_instance_id}/preview",
                username,
                password,
                accept="image/png",
            )
            slice_payload = {
                "slice_index": slice_ref.slice_index,
                "relative_position": relative_position,
                "anchor_label": f"{anchor_id} {relative_position}",
                "sop_instance_uid": slice_ref.sop_instance_uid,
                "image_data_url": _preview_to_data_url(preview_bytes),
            }
            group_slices.append(slice_payload)
            flat_slices.append(
                {
                    **slice_payload,
                    "anchor_id": anchor_id,
                    "center_slice_index": ordered_slices[center_index].slice_index,
                }
            )

        anchor_groups.append(
            {
                "anchor_id": anchor_id,
                "anchor_label": f"{anchor_id} n",
                "center_slice_index": ordered_slices[center_index].slice_index,
                "center_sop_instance_uid": ordered_slices[center_index].sop_instance_uid,
                "slice_indices": [ordered_slices[center_index + offset].slice_index for offset in (-1, 0, 1)],
                "slices": group_slices,
            }
        )

    image_request = ImageReportInferRequest(
        request_id=payload.request_id,
        study_id=payload.study_id or (metadata.study_id if metadata and metadata.study_id else None),
        series_uid=payload.series_instance_uid,
        modality=payload.modality or "CT",
        body_part=payload.body_part or (metadata.body_part if metadata and metadata.body_part else "Unknown"),
        clinical_context=payload.clinical_context,
        instruction=payload.instruction,
        query=payload.query,
        selection_strategy="deterministic-uniform-non-overlapping-triplets",
        anchor_group_count=len(anchor_groups),
        anchor_groups=anchor_groups,
        slices=flat_slices,
    )
    return image_request, {
        "orthanc_base_url": base_url,
        "orthanc_study_id": payload.orthanc_study_id,
        "orthanc_series_id": payload.orthanc_series_id,
        "study_instance_uid": payload.study_instance_uid,
        "series_instance_uid": payload.series_instance_uid,
        "resolved_anchor_group_count": len(anchor_groups),
        "resolved_slice_count": len(flat_slices),
    }
