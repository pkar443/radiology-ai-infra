from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


NonEmptyTrimmedStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RequestIdStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
StudyIdStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
SeriesUidStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
ModalityStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
BodyPartStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
ClinicalContextStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
InstructionStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
QueryStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]
RelativePositionStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
AnchorIdStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
AnchorLabelStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
SopInstanceUidStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
SelectionStrategyStr = Literal["deterministic-uniform-non-overlapping-triplets"]
ConfidenceStr = Literal["low", "medium", "high"]
OrganStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
FindingIdStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
FindingLabelStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]
FindingSummaryStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
FindingExplanationStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
EvidenceStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    device: str
    gpu_count: int
    load_error: str | None = None


class TextInferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    max_new_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    do_sample: bool | None = None
    request_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_sampling(self) -> "TextInferRequest":
        if self.do_sample and self.temperature is not None and self.temperature <= 0:
            raise ValueError("temperature must be greater than 0 when do_sample is true")
        return self


class TextInferResponse(BaseModel):
    request_id: str
    model_id: str
    device: str
    load_state: str
    inference_time_ms: int
    generated_text: str


class ReportInferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(default=None, min_length=1, max_length=24000)
    study_id: str | None = Field(default=None, max_length=128)
    modality: str | None = Field(default=None, min_length=1, max_length=64)
    body_part: str | None = Field(default=None, min_length=1, max_length=128)
    clinical_context: str | None = Field(default=None, max_length=2000)
    findings_input: str | None = Field(default=None, min_length=1, max_length=12000)
    request_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_prompt_or_inputs(self) -> "ReportInferRequest":
        if self.prompt and self.prompt.strip():
            return self

        text_fields = [
            self.study_id,
            self.modality,
            self.body_part,
            self.clinical_context,
            self.findings_input,
        ]
        if any(value and value.strip() for value in text_fields):
            return self

        raise ValueError("Provide either prompt or report input fields.")


class ReportInferResponse(BaseModel):
    request_id: str
    report_text: str
    technique: str
    findings: str
    impression: str
    model_id: str
    device: str
    inference_time_ms: int


class ImageReportSliceBaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_index: int
    relative_position: RelativePositionStr
    anchor_label: AnchorLabelStr
    sop_instance_uid: SopInstanceUidStr
    image_data_url: NonEmptyTrimmedStr


class ImageReportAnchorSliceRequest(ImageReportSliceBaseRequest):
    pass


class ImageReportFlatSliceRequest(ImageReportSliceBaseRequest):
    anchor_id: AnchorIdStr
    center_slice_index: int


class ImageReportAnchorGroupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_id: AnchorIdStr
    anchor_label: AnchorLabelStr
    center_slice_index: int
    center_sop_instance_uid: SopInstanceUidStr
    slice_indices: list[int] = Field(min_length=3, max_length=3)
    slices: list[ImageReportAnchorSliceRequest] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_group_structure(self) -> "ImageReportAnchorGroupRequest":
        expected_center_label = f"{self.anchor_id} n"
        if self.anchor_label != expected_center_label:
            raise ValueError(f"anchor_label must equal {expected_center_label!r}.")

        expected_positions = ["n-1", "n", "n+1"]
        actual_positions = [slice_item.relative_position for slice_item in self.slices]
        if actual_positions != expected_positions:
            raise ValueError("Each anchor group must contain slices ordered as n-1, n, n+1.")

        actual_slice_indices = [slice_item.slice_index for slice_item in self.slices]
        if self.slice_indices != actual_slice_indices:
            raise ValueError("slice_indices must match the ordered slice_index values in slices.")

        for slice_item in self.slices:
            expected_label = f"{self.anchor_id} {slice_item.relative_position}"
            if slice_item.anchor_label != expected_label:
                raise ValueError(f"Slice anchor_label must equal {expected_label!r}.")

        center_slice = self.slices[1]
        if center_slice.slice_index != self.center_slice_index:
            raise ValueError("center_slice_index must match the n slice_index.")
        if center_slice.anchor_label != self.anchor_label:
            raise ValueError("anchor_label must match the n slice anchor_label.")
        if center_slice.sop_instance_uid != self.center_sop_instance_uid:
            raise ValueError("center_sop_instance_uid must match the n slice sop_instance_uid.")

        return self


class ImageReportInferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: RequestIdStr | None = None
    study_id: StudyIdStr | None = None
    series_uid: SeriesUidStr | None = None
    modality: Literal["CT"]
    body_part: BodyPartStr
    clinical_context: ClinicalContextStr | None = None
    instruction: InstructionStr
    query: QueryStr
    selection_strategy: SelectionStrategyStr
    anchor_group_count: int = Field(ge=1, le=6)
    anchor_groups: list[ImageReportAnchorGroupRequest] = Field(min_length=1, max_length=6)
    slices: list[ImageReportFlatSliceRequest] = Field(min_length=1, max_length=18)

    @field_validator("request_id", "study_id", "series_uid", "clinical_context", mode="before")
    @classmethod
    def normalize_optional_strings(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_anchor_groups_and_flattened_slices(self) -> "ImageReportInferRequest":
        if self.anchor_group_count != len(self.anchor_groups):
            raise ValueError("anchor_group_count must match the number of anchor_groups.")

        expected_flattened: list[tuple[object, ...]] = []
        for anchor_group in self.anchor_groups:
            for slice_item in anchor_group.slices:
                expected_flattened.append(
                    (
                        slice_item.slice_index,
                        slice_item.relative_position,
                        slice_item.anchor_label,
                        slice_item.sop_instance_uid,
                        slice_item.image_data_url,
                        anchor_group.anchor_id,
                        anchor_group.center_slice_index,
                    )
                )

        provided_flattened = [
            (
                slice_item.slice_index,
                slice_item.relative_position,
                slice_item.anchor_label,
                slice_item.sop_instance_uid,
                slice_item.image_data_url,
                slice_item.anchor_id,
                slice_item.center_slice_index,
            )
            for slice_item in self.slices
        ]

        if provided_flattened != expected_flattened:
            raise ValueError("slices must be the ordered flattened compatibility copy of anchor_groups.")

        return self


class ImageStructuredFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: FindingIdStr
    organ: OrganStr
    label: FindingLabelStr
    summary: FindingSummaryStr
    explanation: FindingExplanationStr
    anchor_slice_index: int
    anchor_label: AnchorLabelStr
    supporting_anchors: list[AnchorLabelStr] = Field(min_length=1, max_length=18)
    confidence: ConfidenceStr
    evidence: EvidenceStr
    abnormal: bool


class ImageReportInferResponse(ReportInferResponse):
    explanation_summary: str
    structured_findings: list[ImageStructuredFinding] = Field(min_length=1)
    limitations: str
