from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
