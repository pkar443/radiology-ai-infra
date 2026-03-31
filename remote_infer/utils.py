from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import io
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("remote_infer.utils")
_SECTION_PATTERN = re.compile(r"(?im)^\s*(technique|findings|impression)\s*:\s*")
_SPECIAL_TOKEN_PATTERN = re.compile(
    r"(?:<pad>|<bos>|<eos>|<unk>|<start_of_turn>|<end_of_turn>|"
    r"<image_soft_token>|<start_of_image>|<audio_soft_token>|<start_of_audio>)"
)
_REPEATED_BLANK_LINES_PATTERN = re.compile(r"\n(?:[ \t]*\n){2,}")
_IMAGE_DATA_URL_PATTERN = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<data>[a-zA-Z0-9+/=\s]+)$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ReportPostprocessResult:
    normalized_report: dict[str, Any]
    normalization_mode: str
    text_source: str
    text_before_cleanup: str
    text_after_cleanup: str
    preserved_raw_text: bool


@dataclass(frozen=True)
class DecodedImageSlice:
    slice_index: int
    relative_position: str
    image: Any
    anchor_label: str | None = None
    sop_instance_uid: str | None = None
    anchor_id: str | None = None
    center_slice_index: int | None = None
    validated_image: Any | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class PromptEchoCleanupResult:
    text: str
    prompt_echo_removed: bool
    matched_offset: int | None


@dataclass(frozen=True)
class ImageSlicePayload:
    slice_index: int
    relative_position: str
    anchor_label: str
    sop_instance_uid: str
    image_data_url: str
    anchor_id: str
    center_slice_index: int


@dataclass(frozen=True)
class AnchorContext:
    anchor_id: str
    anchor_label: str
    center_slice_index: int
    all_slice_labels: tuple[str, ...]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={json.dumps(value, ensure_ascii=True)}")
    logger.log(level, " ".join(parts))


def bootstrap_env_from_file(env_path: Path) -> None:
    if not env_path.is_file():
        return

    try:
        completed = subprocess.run(
            [
                "bash",
                "-lc",
                'set -a; source "$1" >/dev/null 2>&1; env -0',
                "bash",
                str(env_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "env_bootstrap_failed",
            env_file=str(env_path),
            returncode=exc.returncode,
        )
        return

    for chunk in completed.stdout.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        key, value = chunk.split(b"=", 1)
        os.environ.setdefault(key.decode("utf-8"), value.decode("utf-8"))


def ensure_workspace_token(hf_home: str) -> str | None:
    hf_home_path = Path(hf_home).expanduser()
    hf_home_path.mkdir(parents=True, exist_ok=True)

    target_token_path = hf_home_path / "token"
    if target_token_path.is_file() and target_token_path.stat().st_size > 0:
        return str(target_token_path)

    default_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if default_token_path.is_file() and default_token_path.stat().st_size > 0:
        shutil.copy2(default_token_path, target_token_path)
        target_token_path.chmod(0o600)
        return str(target_token_path)

    return None


def ensure_request_id(request_id: str | None) -> str:
    if request_id and request_id.strip():
        return request_id.strip()
    return uuid.uuid4().hex


def resolve_report_prompt(
    prompt: str | None,
    study_id: str | None,
    modality: str | None,
    body_part: str | None,
    clinical_context: str | None,
    findings_input: str | None,
) -> str:
    if prompt and prompt.strip():
        return prompt.strip()

    parts: list[str] = []
    if study_id and study_id.strip():
        parts.append(f"Study ID: {study_id.strip()}")
    if modality and modality.strip():
        parts.append(f"Modality: {modality.strip()}")
    if body_part and body_part.strip():
        parts.append(f"Body part: {body_part.strip()}")
    if clinical_context and clinical_context.strip():
        parts.append(f"Clinical context: {clinical_context.strip()}")
    if findings_input and findings_input.strip():
        parts.append("Findings input:")
        parts.append(findings_input.strip())

    combined_prompt = "\n".join(parts).strip()
    if not combined_prompt:
        raise ValueError("Either prompt or report input fields must be provided.")

    return combined_prompt


def preview_text(text: str, limit: int = 240) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if len(normalized) <= limit:
        return normalized
    remaining = len(normalized) - limit
    return f"{normalized[:limit]}... [truncated {remaining} chars]"


def build_medgemma_text_messages(prompt: str) -> list[dict[str, object]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def _apply_chat_template(
    chat_io: object,
    messages: list[dict[str, object]],
    *,
    tokenize: bool,
    return_dict: bool = False,
    return_tensors: str | None = None,
    continue_final_message: bool | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": tokenize,
    }
    if return_dict:
        kwargs["return_dict"] = True
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    if continue_final_message is not None:
        kwargs["continue_final_message"] = continue_final_message

    try:
        return chat_io.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "continue_final_message" not in kwargs:
            raise
        kwargs.pop("continue_final_message")
        return chat_io.apply_chat_template(messages, **kwargs)


def prepare_medgemma_text_inputs(text_io: object, prompt: str) -> dict[str, object]:
    messages = build_medgemma_text_messages(prompt)

    if hasattr(text_io, "apply_chat_template"):
        try:
            tokenized = _apply_chat_template(
                text_io,
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                continue_final_message=False,
            )
            return dict(tokenized)
        except Exception:
            prompt = _apply_chat_template(
                text_io,
                messages,
                tokenize=False,
                continue_final_message=False,
            )

    return dict(text_io(prompt, return_tensors="pt"))


def build_medgemma_image_report_content(
    instruction: str,
    slices: Sequence[DecodedImageSlice],
    query: str,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "text", "text": instruction}]
    for slice_item in slices:
        slice_label = slice_item.anchor_label or slice_item.relative_position
        content.append({"type": "image", "image": slice_item.image})
        content.append(
            {
                "type": "text",
                "text": f"SLICE {slice_label} (index {slice_item.slice_index})",
            }
        )
    content.append({"type": "text", "text": query})
    return content


def build_medgemma_image_report_messages(
    instruction: str,
    slices: Sequence[DecodedImageSlice],
    query: str,
) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": build_medgemma_image_report_content(
                instruction=instruction,
                slices=slices,
                query=query,
            ),
        }
    ]


def prepare_medgemma_multimodal_inputs(
    processor: object,
    messages: list[dict[str, object]],
) -> Any:
    if not hasattr(processor, "apply_chat_template"):
        raise TypeError("Processor does not support apply_chat_template for multimodal inference.")

    return _apply_chat_template(
        processor,
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        continue_final_message=False,
    )


def flatten_anchor_group_slices(anchor_groups: Sequence[object]) -> list[ImageSlicePayload]:
    flattened: list[ImageSlicePayload] = []
    for anchor_group in anchor_groups:
        for slice_item in getattr(anchor_group, "slices"):
            flattened.append(
                ImageSlicePayload(
                    slice_index=getattr(slice_item, "slice_index"),
                    relative_position=getattr(slice_item, "relative_position"),
                    anchor_label=getattr(slice_item, "anchor_label"),
                    sop_instance_uid=getattr(slice_item, "sop_instance_uid"),
                    image_data_url=getattr(slice_item, "image_data_url"),
                    anchor_id=getattr(anchor_group, "anchor_id"),
                    center_slice_index=getattr(anchor_group, "center_slice_index"),
                )
            )
    return flattened


def build_anchor_context(anchor_groups: Sequence[object]) -> list[AnchorContext]:
    contexts: list[AnchorContext] = []
    for anchor_group in anchor_groups:
        contexts.append(
            AnchorContext(
                anchor_id=getattr(anchor_group, "anchor_id"),
                anchor_label=getattr(anchor_group, "anchor_label"),
                center_slice_index=getattr(anchor_group, "center_slice_index"),
                all_slice_labels=tuple(getattr(slice_item, "anchor_label") for slice_item in getattr(anchor_group, "slices")),
            )
        )
    return contexts


def normalize_image_data_url(image_data_url: str) -> tuple[str, str]:
    data_url = image_data_url.strip()
    match = _IMAGE_DATA_URL_PATTERN.match(data_url)
    if not match:
        raise ValueError("Expected image data URL in the form data:image/<type>;base64,<data>.")

    mime_type = match.group("mime").lower()
    encoded_data = re.sub(r"\s+", "", match.group("data"))
    return f"data:{mime_type};base64,{encoded_data}", mime_type


def decode_image_data_url(image_data_url: str) -> Any:
    normalized_data_url, _ = normalize_image_data_url(image_data_url)
    encoded_data = normalized_data_url.split(",", 1)[1]
    encoded_data = re.sub(r"\s+", "", encoded_data)
    try:
        image_bytes = base64.b64decode(encoded_data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Image data URL contained invalid base64 content.") from exc

    try:
        from PIL import Image, UnidentifiedImageError
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for image-aware inference support.") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Decoded image bytes were not recognized as a valid image.") from exc
    except OSError as exc:
        raise ValueError(f"Decoded image bytes could not be opened: {exc}") from exc

    return rgb_image


def decode_image_report_slices(slice_items: Sequence[object]) -> list[DecodedImageSlice]:
    decoded_slices: list[DecodedImageSlice] = []
    for slice_item in slice_items:
        slice_index = getattr(slice_item, "slice_index")
        relative_position = getattr(slice_item, "relative_position")
        anchor_label = getattr(slice_item, "anchor_label", None)
        sop_instance_uid = getattr(slice_item, "sop_instance_uid", None)
        anchor_id = getattr(slice_item, "anchor_id", None)
        center_slice_index = getattr(slice_item, "center_slice_index", None)
        image_data_url = getattr(slice_item, "image_data_url")
        try:
            normalized_data_url, mime_type = normalize_image_data_url(image_data_url)
            decoded_image = decode_image_data_url(image_data_url)
        except ValueError as exc:
            raise ValueError(
                f"Slice {slice_index} ({anchor_label or relative_position}) image_data_url is invalid: {exc}"
            ) from exc

        decoded_slices.append(
            DecodedImageSlice(
                slice_index=slice_index,
                relative_position=relative_position,
                image=normalized_data_url,
                anchor_label=anchor_label,
                sop_instance_uid=sop_instance_uid,
                anchor_id=anchor_id,
                center_slice_index=center_slice_index,
                validated_image=decoded_image,
                mime_type=mime_type,
            )
        )

    return decoded_slices


def describe_slice_order(slices: Sequence[object]) -> list[str]:
    descriptions: list[str] = []
    for slice_item in slices:
        anchor_label = getattr(slice_item, "anchor_label", None)
        label = anchor_label or getattr(slice_item, "relative_position")
        descriptions.append(f"{label}:{getattr(slice_item, 'slice_index')}")
    return descriptions


def summarize_api_slice_payloads(slices: Sequence[object]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for slice_item in slices:
        image_data_url = getattr(slice_item, "image_data_url")
        try:
            normalized_data_url, mime_type = normalize_image_data_url(image_data_url)
            prefix = normalized_data_url.split(",", 1)[0]
        except ValueError:
            mime_type = "invalid"
            prefix = preview_text(str(image_data_url), limit=32)

        summaries.append(
            {
                "slice_index": getattr(slice_item, "slice_index"),
                "relative_position": getattr(slice_item, "relative_position"),
                "anchor_label": getattr(slice_item, "anchor_label", None),
                "anchor_id": getattr(slice_item, "anchor_id", None),
                "center_slice_index": getattr(slice_item, "center_slice_index", None),
                "sop_instance_uid": getattr(slice_item, "sop_instance_uid", None),
                "mime_type": mime_type,
                "data_url_prefix": prefix,
                "data_url_chars": len(str(image_data_url)),
            }
        )
    return summaries


def summarize_validated_slices(slices: Sequence[DecodedImageSlice]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for slice_item in slices:
        validated_image = slice_item.validated_image
        image_size = getattr(validated_image, "size", None)
        summaries.append(
            {
                "slice_index": slice_item.slice_index,
                "relative_position": slice_item.relative_position,
                "anchor_label": slice_item.anchor_label,
                "anchor_id": slice_item.anchor_id,
                "center_slice_index": slice_item.center_slice_index,
                "sop_instance_uid": slice_item.sop_instance_uid,
                "mime_type": slice_item.mime_type,
                "image_mode": getattr(validated_image, "mode", None),
                "image_size": list(image_size) if isinstance(image_size, tuple) else image_size,
                "message_image_type": type(slice_item.image).__name__,
            }
        )
    return summaries


def strip_prompt_echo_from_generated_text(
    generated_text: str,
    decoded_input_text: str,
) -> PromptEchoCleanupResult:
    candidate = generated_text.replace("\r\n", "\n").replace("\r", "\n")
    prompt_text = decoded_input_text.replace("\r\n", "\n").replace("\r", "\n")

    if not prompt_text:
        return PromptEchoCleanupResult(text=candidate, prompt_echo_removed=False, matched_offset=None)

    matched_offset = candidate.find(prompt_text)
    if 0 <= matched_offset <= 2:
        return PromptEchoCleanupResult(
            text=candidate[matched_offset + len(prompt_text) :].lstrip(),
            prompt_echo_removed=True,
            matched_offset=matched_offset,
        )

    stripped_prompt = prompt_text.strip()
    if stripped_prompt:
        stripped_candidate = candidate.lstrip()
        leading_gap = len(candidate) - len(stripped_candidate)
        matched_offset = stripped_candidate.find(stripped_prompt)
        if 0 <= matched_offset <= 2 and leading_gap <= 2:
            return PromptEchoCleanupResult(
                text=stripped_candidate[matched_offset + len(stripped_prompt) :].lstrip(),
                prompt_echo_removed=True,
                matched_offset=leading_gap + matched_offset,
            )

    return PromptEchoCleanupResult(text=candidate, prompt_echo_removed=False, matched_offset=None)


def clean_generated_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = _REPEATED_BLANK_LINES_PATTERN.sub("\n\n", cleaned)
    return cleaned


def _normalize_section_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _strip_section_headers(text: str) -> str:
    content_lines: list[str] = []
    for line in text.splitlines():
        match = _SECTION_PATTERN.match(line)
        if match:
            remainder = line[match.end() :].strip()
            if remainder:
                content_lines.append(remainder)
            continue
        stripped = line.strip()
        if stripped:
            content_lines.append(stripped)
    return "\n".join(content_lines).strip()


def _has_usable_report_content(text: str) -> bool:
    meaningful_text = _SPECIAL_TOKEN_PATTERN.sub("", text).strip()
    return bool(meaningful_text)


def build_report_text_only_result(report_text: str, normalization_mode: str) -> dict[str, str]:
    return {
        "report_text": report_text,
        "technique": "",
        "findings": report_text,
        "impression": "",
        "normalization_mode": normalization_mode,
    }


def normalize_report_sections(raw_text: str) -> dict[str, str]:
    cleaned = clean_generated_text(raw_text)
    if not cleaned:
        raise ValueError("Generated report text is empty after cleanup.")
    if not _has_usable_report_content(cleaned):
        raise ValueError("Generated report text was empty after removing known special tokens.")

    matches = list(_SECTION_PATTERN.finditer(cleaned))
    if not matches:
        return build_report_text_only_result(cleaned, normalization_mode="report_text_only")

    sections = {"technique": "", "findings": "", "impression": ""}
    for index, match in enumerate(matches):
        section_name = match.group(1).lower()
        section_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        section_body = _normalize_section_text(cleaned[section_start:section_end])
        sections[section_name] = section_body

    return {
        "report_text": cleaned,
        "technique": sections["technique"],
        "findings": sections["findings"],
        "impression": sections["impression"],
        "normalization_mode": "parsed_sections" if any(sections.values()) else "parsed_empty_sections",
    }


def _normalize_string_field(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return clean_generated_text(value)
    return clean_generated_text(str(value))


def _compose_report_text(technique: str, findings: str, impression: str) -> str:
    return "\n\n".join(
        [
            f"Technique:\n{technique}".rstrip(),
            f"Findings:\n{findings}".rstrip(),
            f"Impression:\n{impression}".rstrip(),
        ]
    ).rstrip()


def _extract_json_object_text(text: str) -> str:
    candidate = clean_generated_text(text)
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Generated image report did not contain a JSON object.")
    return candidate[start : end + 1]


def _infer_abnormal_flag(text: str) -> bool:
    normalized = text.lower()
    negative_markers = (
        "no acute",
        "no focal abnormality",
        "no abnormality",
        "unremarkable",
        "normal",
        "without acute abnormality",
    )
    return not any(marker in normalized for marker in negative_markers)


def _primary_anchor_context(anchor_contexts: Sequence[AnchorContext]) -> AnchorContext:
    if not anchor_contexts:
        raise ValueError("At least one anchor context is required for image report normalization.")
    return anchor_contexts[0]


def _build_fallback_structured_finding(
    report_text: str,
    anchor_contexts: Sequence[AnchorContext],
    *,
    abnormal: bool | None = None,
    evidence: str,
) -> list[dict[str, Any]]:
    primary_anchor = _primary_anchor_context(anchor_contexts)
    summary_source = clean_generated_text(report_text) or "CT draft finding"
    summary_line = summary_source.splitlines()[0].strip() if summary_source else "CT draft finding"
    summary_line = summary_line[:220] or "CT draft finding"
    abnormal_flag = _infer_abnormal_flag(summary_source) if abnormal is None else abnormal
    default_label = (
        "No focal acute abnormality in provided slice subset"
        if not abnormal_flag
        else "CT draft finding"
    )

    return [
        {
            "id": "finding-1",
            "organ": "unspecified",
            "label": default_label,
            "summary": summary_line,
            "explanation": summary_source,
            "anchor_slice_index": primary_anchor.center_slice_index,
            "anchor_label": primary_anchor.anchor_label,
            "supporting_anchors": list(primary_anchor.all_slice_labels),
            "confidence": "low",
            "evidence": evidence,
            "abnormal": abnormal_flag,
        }
    ]


def _augment_report_with_structured_fields(
    normalized_report: dict[str, str],
    anchor_contexts: Sequence[AnchorContext],
    *,
    normalization_mode: str,
    limitations: str,
) -> dict[str, Any]:
    report_text = normalized_report["report_text"]
    findings = normalized_report["findings"]
    impression = normalized_report["impression"]
    explanation_summary = impression or findings or report_text

    return {
        "report_text": report_text,
        "technique": normalized_report["technique"],
        "findings": findings,
        "impression": impression,
        "explanation_summary": explanation_summary,
        "structured_findings": _build_fallback_structured_finding(
            impression or findings or report_text,
            anchor_contexts,
            abnormal=_infer_abnormal_flag(impression or findings or report_text),
            evidence="Structured finding synthesized from non-JSON model output anchored to the first provided anchor group.",
        ),
        "limitations": limitations,
        "normalization_mode": normalization_mode,
    }


def _normalize_structured_findings(
    value: object,
    anchor_contexts: Sequence[AnchorContext],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("structured_findings must be a non-empty list.")

    allowed_center_slice_indices = {context.center_slice_index for context in anchor_contexts}
    allowed_center_labels = {context.anchor_label for context in anchor_contexts}
    allowed_slice_labels = {
        slice_label
        for context in anchor_contexts
        for slice_label in context.all_slice_labels
    }

    normalized_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(value, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError("Each structured finding must be a JSON object.")

        anchor_slice_index = raw_item.get("anchor_slice_index")
        if not isinstance(anchor_slice_index, int) or anchor_slice_index not in allowed_center_slice_indices:
            raise ValueError("anchor_slice_index must be one of the provided center slice indices.")

        anchor_label = _normalize_string_field(raw_item.get("anchor_label"))
        if anchor_label not in allowed_center_labels:
            raise ValueError("anchor_label must be one of the provided center anchor labels.")

        supporting_anchors_raw = raw_item.get("supporting_anchors")
        if not isinstance(supporting_anchors_raw, list) or not supporting_anchors_raw:
            supporting_anchors = [anchor_label]
        else:
            supporting_anchors = [_normalize_string_field(item) for item in supporting_anchors_raw if _normalize_string_field(item)]
        if not supporting_anchors:
            supporting_anchors = [anchor_label]
        if any(anchor not in allowed_slice_labels for anchor in supporting_anchors):
            raise ValueError("supporting_anchors must use only provided anchor labels.")

        confidence = _normalize_string_field(raw_item.get("confidence"), default="low").lower()
        if confidence not in {"low", "medium", "high"}:
            raise ValueError("confidence must be one of low, medium, or high.")

        abnormal_value = raw_item.get("abnormal")
        abnormal = abnormal_value if isinstance(abnormal_value, bool) else _infer_abnormal_flag(
            " ".join(
                filter(
                    None,
                    [
                        _normalize_string_field(raw_item.get("summary")),
                        _normalize_string_field(raw_item.get("explanation")),
                        _normalize_string_field(raw_item.get("evidence")),
                    ],
                )
            )
        )

        normalized_items.append(
            {
                "id": _normalize_string_field(raw_item.get("id"), default=f"finding-{index}") or f"finding-{index}",
                "organ": _normalize_string_field(raw_item.get("organ"), default="unspecified") or "unspecified",
                "label": _normalize_string_field(raw_item.get("label"), default=f"Finding {index}") or f"Finding {index}",
                "summary": _normalize_string_field(raw_item.get("summary"), default=_normalize_string_field(raw_item.get("label"), default="CT draft finding")),
                "explanation": _normalize_string_field(raw_item.get("explanation"), default=_normalize_string_field(raw_item.get("summary"), default="CT draft finding")),
                "anchor_slice_index": anchor_slice_index,
                "anchor_label": anchor_label,
                "supporting_anchors": supporting_anchors,
                "confidence": confidence,
                "evidence": _normalize_string_field(raw_item.get("evidence"), default="Evidence was provided in the structured model output."),
                "abnormal": abnormal,
            }
        )

    return normalized_items


def normalize_image_report_json_response(
    raw_text: str,
    anchor_contexts: Sequence[AnchorContext],
) -> dict[str, Any]:
    cleaned = clean_generated_text(raw_text)
    if not cleaned:
        raise ValueError("Generated image report text is empty after cleanup.")
    if not _has_usable_report_content(cleaned):
        raise ValueError("Generated image report text was empty after removing known special tokens.")

    json_text = _extract_json_object_text(cleaned)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Generated image report JSON must decode to an object.")

    technique = _normalize_string_field(parsed.get("technique"))
    findings = _normalize_string_field(parsed.get("findings"))
    impression = _normalize_string_field(parsed.get("impression"))
    report_text = _normalize_string_field(parsed.get("report_text"))
    if not report_text:
        report_text = _compose_report_text(technique, findings, impression)

    explanation_summary = _normalize_string_field(parsed.get("explanation_summary"), default=impression or findings or report_text)
    limitations = _normalize_string_field(parsed.get("limitations"))
    structured_findings = _normalize_structured_findings(parsed.get("structured_findings"), anchor_contexts)

    return {
        "report_text": report_text,
        "technique": technique,
        "findings": findings,
        "impression": impression,
        "explanation_summary": explanation_summary,
        "structured_findings": structured_findings,
        "limitations": limitations,
        "normalization_mode": "json_structured",
    }


def postprocess_generated_image_report_text(
    visible_text: str,
    raw_text: str,
    anchor_contexts: Sequence[AnchorContext],
) -> ReportPostprocessResult:
    visible_before_cleanup = visible_text
    raw_before_cleanup = raw_text

    visible_after_cleanup = clean_generated_text(visible_before_cleanup) if visible_before_cleanup else ""
    raw_after_cleanup = clean_generated_text(raw_before_cleanup) if raw_before_cleanup else ""

    selected_source = "empty"
    text_before_cleanup = ""
    text_after_cleanup = ""

    if visible_after_cleanup:
        selected_source = "visible_cleaned"
        text_before_cleanup = visible_before_cleanup
        text_after_cleanup = visible_after_cleanup
    elif raw_after_cleanup and _has_usable_report_content(raw_after_cleanup):
        selected_source = "raw_cleaned_fallback"
        text_before_cleanup = raw_before_cleanup
        text_after_cleanup = raw_after_cleanup
    elif raw_before_cleanup and _has_usable_report_content(raw_before_cleanup):
        selected_source = "raw_raw_preserved"
        text_before_cleanup = raw_before_cleanup
        text_after_cleanup = raw_before_cleanup
    else:
        raise ValueError("Generated report continuation was empty after decoding.")

    preserved_raw_text = selected_source.endswith("_raw_preserved")
    if preserved_raw_text:
        normalized_report = _augment_report_with_structured_fields(
            build_report_text_only_result(
                text_after_cleanup,
                normalization_mode="raw_preserved_after_cleanup",
            ),
            anchor_contexts,
            normalization_mode="raw_preserved_after_cleanup",
            limitations="Structured JSON was unavailable; raw text was preserved.",
        )
        return ReportPostprocessResult(
            normalized_report=normalized_report,
            normalization_mode=normalized_report["normalization_mode"],
            text_source=selected_source,
            text_before_cleanup=text_before_cleanup,
            text_after_cleanup=text_after_cleanup,
            preserved_raw_text=True,
        )

    try:
        normalized_report = normalize_image_report_json_response(text_after_cleanup, anchor_contexts)
        normalization_mode = normalized_report["normalization_mode"]
    except Exception:
        try:
            normalized_sections = normalize_report_sections(text_after_cleanup)
            normalized_report = _augment_report_with_structured_fields(
                normalized_sections,
                anchor_contexts,
                normalization_mode="section_text_fallback",
                limitations="Structured JSON was unavailable; structured findings were synthesized from section text.",
            )
            normalization_mode = normalized_report["normalization_mode"]
        except Exception:
            normalized_report = _augment_report_with_structured_fields(
                build_report_text_only_result(
                    text_after_cleanup,
                    normalization_mode="report_text_only_after_normalization_exception",
                ),
                anchor_contexts,
                normalization_mode="report_text_only_after_normalization_exception",
                limitations="Structured JSON and section parsing were unavailable; structured findings were synthesized from plain text.",
            )
            normalization_mode = normalized_report["normalization_mode"]

    return ReportPostprocessResult(
        normalized_report=normalized_report,
        normalization_mode=normalization_mode,
        text_source=selected_source,
        text_before_cleanup=text_before_cleanup,
        text_after_cleanup=text_after_cleanup,
        preserved_raw_text=False,
    )


def postprocess_generated_report_text(visible_text: str, raw_text: str) -> ReportPostprocessResult:
    visible_before_cleanup = visible_text
    raw_before_cleanup = raw_text

    visible_after_cleanup = clean_generated_text(visible_before_cleanup) if visible_before_cleanup else ""
    raw_after_cleanup = clean_generated_text(raw_before_cleanup) if raw_before_cleanup else ""

    selected_source = "empty"
    text_before_cleanup = ""
    text_after_cleanup = ""

    if visible_after_cleanup:
        selected_source = "visible_cleaned"
        text_before_cleanup = visible_before_cleanup
        text_after_cleanup = visible_after_cleanup
    elif raw_after_cleanup and _has_usable_report_content(raw_after_cleanup):
        selected_source = "raw_cleaned_fallback"
        text_before_cleanup = raw_before_cleanup
        text_after_cleanup = raw_after_cleanup
    elif raw_before_cleanup and _has_usable_report_content(raw_before_cleanup):
        selected_source = "raw_raw_preserved"
        text_before_cleanup = raw_before_cleanup
        text_after_cleanup = raw_before_cleanup
    else:
        raise ValueError("Generated report continuation was empty after decoding.")

    preserved_raw_text = selected_source.endswith("_raw_preserved")
    if preserved_raw_text:
        normalized_report = build_report_text_only_result(
            text_after_cleanup,
            normalization_mode="raw_preserved_after_cleanup",
        )
        return ReportPostprocessResult(
            normalized_report=normalized_report,
            normalization_mode=normalized_report["normalization_mode"],
            text_source=selected_source,
            text_before_cleanup=text_before_cleanup,
            text_after_cleanup=text_after_cleanup,
            preserved_raw_text=True,
        )

    try:
        normalized_report = normalize_report_sections(text_after_cleanup)
        normalization_mode = normalized_report["normalization_mode"]
    except Exception:
        normalized_report = build_report_text_only_result(
            text_after_cleanup,
            normalization_mode="report_text_only_after_normalization_exception",
        )
        normalization_mode = normalized_report["normalization_mode"]

    return ReportPostprocessResult(
        normalized_report=normalized_report,
        normalization_mode=normalization_mode,
        text_source=selected_source,
        text_before_cleanup=text_before_cleanup,
        text_after_cleanup=text_after_cleanup,
        preserved_raw_text=False,
    )
