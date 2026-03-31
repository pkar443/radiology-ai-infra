from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .auth import verify_bearer_token
from .config import get_settings
from .model_loader import (
    GenerationConfig,
    InferenceExecutionError,
    InferenceOOMError,
    InvalidPayloadError,
    ModelAccessError,
    ModelNotLoadedError,
    RemoteInferError,
    get_model_service,
)
from .schemas import (
    HealthResponse,
    ImageReportInferRequest,
    ImageReportInferResponse,
    ReportInferRequest,
    ReportInferResponse,
    TextInferRequest,
    TextInferResponse,
)
from .utils import (
    build_anchor_context,
    build_medgemma_image_report_messages,
    decode_image_report_slices,
    describe_slice_order,
    ensure_request_id,
    flatten_anchor_group_slices,
    log_event,
    postprocess_generated_image_report_text,
    postprocess_generated_report_text,
    preview_text,
    resolve_report_prompt,
    summarize_api_slice_payloads,
    summarize_validated_slices,
    setup_logging,
)


setup_logging()
LOGGER = logging.getLogger("remote_infer.app")

settings = get_settings()
service = get_model_service()


@asynccontextmanager
async def lifespan(_: FastAPI):
    log_event(
        LOGGER,
        logging.INFO,
        "startup_begin",
        model_source=settings.model_source,
        host=settings.remote_infer_host,
        port=settings.remote_infer_port,
        debug_enabled=settings.remote_infer_debug,
    )
    try:
        service.load_model()
    except RemoteInferError as exc:
        log_event(
            LOGGER,
            logging.ERROR,
            "startup_model_load_failed",
            error=str(exc),
        )
    except Exception as exc:
        log_event(
            LOGGER,
            logging.ERROR,
            "startup_unexpected_error",
            error=str(exc),
        )

    yield

    log_event(LOGGER, logging.INFO, "shutdown")


app = FastAPI(
    title="MedGemma Remote Inference",
    version="0.1.0",
    lifespan=lifespan,
)


def _preview_limit() -> int:
    return 1200 if settings.remote_infer_debug else 240


def _error_response(status_code: int, error: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "message": message},
    )


@app.exception_handler(ModelNotLoadedError)
async def handle_model_not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    log_event(LOGGER, logging.ERROR, "model_not_loaded", path=request.url.path, error=str(exc))
    return _error_response(exc.status_code, exc.error_code, str(exc))


@app.exception_handler(ModelAccessError)
async def handle_model_access_error(request: Request, exc: ModelAccessError) -> JSONResponse:
    log_event(LOGGER, logging.ERROR, "model_access_error", path=request.url.path, error=str(exc))
    return _error_response(exc.status_code, exc.error_code, str(exc))


@app.exception_handler(InvalidPayloadError)
async def handle_invalid_payload(request: Request, exc: InvalidPayloadError) -> JSONResponse:
    log_event(LOGGER, logging.WARNING, "invalid_payload", path=request.url.path, error=str(exc))
    return _error_response(exc.status_code, exc.error_code, str(exc))


@app.exception_handler(InferenceOOMError)
async def handle_oom(request: Request, exc: InferenceOOMError) -> JSONResponse:
    log_event(LOGGER, logging.ERROR, "cuda_oom", path=request.url.path, error=str(exc))
    return _error_response(exc.status_code, exc.error_code, str(exc))


@app.exception_handler(InferenceExecutionError)
async def handle_inference_error(request: Request, exc: InferenceExecutionError) -> JSONResponse:
    log_event(LOGGER, logging.ERROR, "inference_error", path=request.url.path, error=str(exc))
    return _error_response(exc.status_code, exc.error_code, str(exc))


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    log_event(
        LOGGER,
        logging.WARNING,
        "request_validation_failed",
        path=request.url.path,
        error=str(exc),
    )
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_payload", "message": "Request validation failed", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    log_event(
        LOGGER,
        logging.ERROR,
        "unexpected_exception",
        path=request.url.path,
        error=str(exc),
    )
    return _error_response(500, "unexpected_error", "Unexpected server error")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(**service.health_snapshot())


@app.post("/infer-text", response_model=TextInferResponse, dependencies=[Depends(verify_bearer_token)])
async def infer_text(payload: TextInferRequest) -> TextInferResponse:
    request_id = ensure_request_id(payload.request_id)

    do_sample = payload.do_sample if payload.do_sample is not None else settings.medgemma_default_temperature > 0
    temperature = payload.temperature if payload.temperature is not None else settings.medgemma_default_temperature
    top_p = payload.top_p if payload.top_p is not None else settings.medgemma_default_top_p
    max_new_tokens = payload.max_new_tokens if payload.max_new_tokens is not None else settings.medgemma_max_new_tokens

    log_event(
        LOGGER,
        logging.INFO,
        "infer_text_request",
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    result = service.generate_text(
        payload.prompt,
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        ),
    )

    log_event(
        LOGGER,
        logging.INFO,
        "infer_text_success",
        request_id=request_id,
        inference_time_ms=result.inference_time_ms,
    )

    return TextInferResponse(
        request_id=request_id,
        model_id=result.model_id,
        device=result.device,
        load_state=result.load_state,
        inference_time_ms=result.inference_time_ms,
        generated_text=result.text,
    )


@app.post("/infer-report-test", response_model=ReportInferResponse, dependencies=[Depends(verify_bearer_token)])
async def infer_report_test(payload: ReportInferRequest) -> ReportInferResponse:
    request_id = ensure_request_id(payload.request_id or payload.study_id)
    try:
        prompt = resolve_report_prompt(
            prompt=payload.prompt,
            study_id=payload.study_id,
            modality=payload.modality,
            body_part=payload.body_part,
            clinical_context=payload.clinical_context,
            findings_input=payload.findings_input,
        )
    except ValueError as exc:
        raise InferenceExecutionError(f"Report prompt resolution failed: {exc}") from exc

    generation_config = GenerationConfig(
        max_new_tokens=settings.medgemma_report_max_new_tokens,
        temperature=settings.medgemma_report_temperature,
        top_p=settings.medgemma_report_top_p,
        do_sample=settings.medgemma_report_do_sample,
    )

    log_event(
        LOGGER,
        logging.INFO,
        "infer_report_request",
        request_id=request_id,
        study_id=payload.study_id,
        modality=payload.modality,
        body_part=payload.body_part,
        prompt_source="direct" if payload.prompt and payload.prompt.strip() else "legacy_fields",
        prompt_chars=len(prompt),
        prompt_preview=preview_text(prompt, limit=_preview_limit()),
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        do_sample=generation_config.do_sample,
        debug_enabled=settings.remote_infer_debug,
    )

    result = service.generate_text(prompt, generation_config=generation_config)
    log_event(
        LOGGER,
        logging.INFO,
        "infer_report_raw_output",
        request_id=request_id,
        input_ids_length=result.input_ids_length,
        prompt_token_count=result.prompt_token_count,
        generated_sequence_length=result.generated_sequence_length,
        generated_token_count=result.generated_token_count,
        continuation_token_count=result.continuation_token_count,
        special_only_continuation=result.special_only_continuation,
        generated_token_ids_head=list(result.generated_token_ids_head),
        continuation_token_ids_head=list(result.generated_token_ids_head),
        raw_output_chars=len(result.raw_text),
        raw_output_preview=preview_text(result.raw_text, limit=_preview_limit()),
        visible_output_chars=len(result.text),
        visible_output_preview=preview_text(result.text, limit=_preview_limit()),
        decoded_continuation_raw_preview=preview_text(result.raw_text, limit=_preview_limit()),
        decoded_continuation_visible_preview=preview_text(result.text, limit=_preview_limit()),
    )

    try:
        postprocessed = postprocess_generated_report_text(
            visible_text=result.text,
            raw_text=result.raw_text,
        )
    except ValueError as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "infer_report_unusable_output",
            request_id=request_id,
            reason=str(exc),
            input_ids_length=result.input_ids_length,
            prompt_token_count=result.prompt_token_count,
            generated_sequence_length=result.generated_sequence_length,
            generated_token_count=result.generated_token_count,
            continuation_token_count=result.continuation_token_count,
            special_only_continuation=result.special_only_continuation,
            generated_token_ids_head=list(result.generated_token_ids_head),
            continuation_token_ids_head=list(result.generated_token_ids_head),
            raw_output_preview=preview_text(result.raw_text, limit=_preview_limit()),
            visible_output_preview=preview_text(result.text, limit=_preview_limit()),
        )
        raise InferenceExecutionError(f"Generated report was unusable: {exc}") from exc

    normalized_report = postprocessed.normalized_report
    normalization_mode = postprocessed.normalization_mode
    log_event(
        LOGGER,
        logging.INFO,
        "infer_report_cleaned_output",
        request_id=request_id,
        text_source=postprocessed.text_source,
        preserved_raw_text=postprocessed.preserved_raw_text,
        text_before_cleanup_chars=len(postprocessed.text_before_cleanup),
        text_before_cleanup_preview=preview_text(postprocessed.text_before_cleanup, limit=_preview_limit()),
        text_after_cleanup_chars=len(postprocessed.text_after_cleanup),
        text_after_cleanup_preview=preview_text(postprocessed.text_after_cleanup, limit=_preview_limit()),
    )
    if normalization_mode != "parsed_sections":
        log_event(
            LOGGER,
            logging.WARNING,
            "infer_report_normalization_fallback",
            request_id=request_id,
            normalization_mode=normalization_mode,
            text_source=postprocessed.text_source,
            preserved_raw_text=postprocessed.preserved_raw_text,
        )

    log_event(
        LOGGER,
        logging.INFO,
        "infer_report_normalized",
        request_id=request_id,
        normalization_mode=normalization_mode,
        technique_chars=len(normalized_report["technique"]),
        findings_chars=len(normalized_report["findings"]),
        impression_chars=len(normalized_report["impression"]),
        technique_preview=preview_text(normalized_report["technique"], limit=_preview_limit()),
        findings_preview=preview_text(normalized_report["findings"], limit=_preview_limit()),
        impression_preview=preview_text(normalized_report["impression"], limit=_preview_limit()),
    )

    log_event(
        LOGGER,
        logging.INFO,
        "infer_report_success",
        request_id=request_id,
        inference_time_ms=result.inference_time_ms,
        normalization_mode=normalization_mode,
        technique_present=bool(normalized_report["technique"]),
        findings_present=bool(normalized_report["findings"]),
        impression_present=bool(normalized_report["impression"]),
    )

    return ReportInferResponse(
        request_id=request_id,
        report_text=normalized_report["report_text"],
        technique=normalized_report["technique"],
        findings=normalized_report["findings"],
        impression=normalized_report["impression"],
        model_id=result.model_id,
        device=result.device,
        inference_time_ms=result.inference_time_ms,
    )


@app.post("/infer-image-report", response_model=ImageReportInferResponse, dependencies=[Depends(verify_bearer_token)])
async def infer_image_report(payload: ImageReportInferRequest) -> ImageReportInferResponse:
    request_id = ensure_request_id(payload.request_id or payload.study_id)
    generation_config = GenerationConfig(
        max_new_tokens=settings.medgemma_image_report_max_new_tokens,
        temperature=settings.medgemma_image_report_temperature,
        top_p=settings.medgemma_image_report_top_p,
        do_sample=settings.medgemma_image_report_do_sample,
    )

    flattened_anchor_slices = flatten_anchor_group_slices(payload.anchor_groups)
    anchor_context = build_anchor_context(payload.anchor_groups)
    ordered_slices = describe_slice_order(flattened_anchor_slices)
    api_slice_summaries = summarize_api_slice_payloads(flattened_anchor_slices)
    log_event(
        LOGGER,
        logging.INFO,
        "infer_image_report_request",
        request_id=request_id,
        study_id=payload.study_id,
        series_uid=payload.series_uid,
        modality=payload.modality,
        body_part=payload.body_part,
        selection_strategy=payload.selection_strategy,
        anchor_group_count=payload.anchor_group_count,
        compatibility_slice_count=len(payload.slices),
        slice_count=len(flattened_anchor_slices),
        ordered_slices=ordered_slices,
        payload_summary={
            "study_id": payload.study_id,
            "series_uid": payload.series_uid,
            "modality": payload.modality,
            "body_part": payload.body_part,
            "selection_strategy": payload.selection_strategy,
            "anchor_group_count": payload.anchor_group_count,
            "slice_count": len(flattened_anchor_slices),
        },
        anchor_groups=[
            {
                "anchor_id": anchor.anchor_id,
                "anchor_label": anchor.anchor_label,
                "center_slice_index": anchor.center_slice_index,
            }
            for anchor in anchor_context
        ],
        api_slice_summaries=api_slice_summaries,
        clinical_context_preview=preview_text(payload.clinical_context or "", limit=_preview_limit()),
        instruction_chars=len(payload.instruction),
        instruction_preview=preview_text(payload.instruction, limit=_preview_limit()),
        query_chars=len(payload.query),
        query_preview=preview_text(payload.query, limit=_preview_limit()),
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        do_sample=generation_config.do_sample,
        debug_enabled=settings.remote_infer_debug,
    )

    try:
        decoded_slices = decode_image_report_slices(flattened_anchor_slices)
        messages = build_medgemma_image_report_messages(
            instruction=payload.instruction,
            slices=decoded_slices,
            query=payload.query,
        )
    except ValueError as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "infer_image_report_invalid_payload",
            request_id=request_id,
            slice_count=len(flattened_anchor_slices),
            ordered_slices=ordered_slices,
            error=str(exc),
        )
        raise InvalidPayloadError(str(exc)) from exc

    message_content = messages[0]["content"]
    validated_slice_summaries = summarize_validated_slices(decoded_slices)
    log_event(
        LOGGER,
        logging.INFO,
        "infer_image_report_message_prepared",
        request_id=request_id,
        message_count=len(messages),
        content_item_count=len(message_content),
        content_types=[item["type"] for item in message_content],
        ordered_slices=ordered_slices,
        validated_slice_summaries=validated_slice_summaries,
        first_message_item_preview=preview_text(str(message_content[0]), limit=_preview_limit()) if message_content else "",
        last_message_item_preview=preview_text(str(message_content[-1]), limit=_preview_limit()) if message_content else "",
    )

    result = service.generate_image_report(messages, generation_config=generation_config)
    log_event(
        LOGGER,
        logging.INFO,
        "infer_image_report_raw_output",
        request_id=request_id,
        input_ids_length=result.input_ids_length,
        prompt_token_count=result.prompt_token_count,
        generated_sequence_length=result.generated_sequence_length,
        generated_token_count=result.generated_token_count,
        continuation_token_count=result.continuation_token_count,
        special_only_continuation=result.special_only_continuation,
        generated_token_ids_head=list(result.generated_token_ids_head),
        prompt_echo_removed=result.prompt_echo_removed,
        prompt_echo_offset=result.prompt_echo_offset,
        raw_output_chars=len(result.full_text or result.raw_text),
        raw_output_preview=preview_text(result.full_text or result.raw_text, limit=_preview_limit()),
        cleaned_output_chars=len(result.text),
        cleaned_output_preview=preview_text(result.text, limit=_preview_limit()),
    )

    try:
        postprocessed = postprocess_generated_image_report_text(
            visible_text=result.text,
            raw_text=result.raw_text,
            anchor_contexts=anchor_context,
        )
    except ValueError as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "infer_image_report_unusable_output",
            request_id=request_id,
            reason=str(exc),
            generated_token_ids_head=list(result.generated_token_ids_head),
            prompt_echo_removed=result.prompt_echo_removed,
            raw_output_preview=preview_text(result.full_text or result.raw_text, limit=_preview_limit()),
            cleaned_output_preview=preview_text(result.text, limit=_preview_limit()),
        )
        raise InferenceExecutionError(f"Generated image-aware report was unusable: {exc}") from exc

    normalized_report = postprocessed.normalized_report
    normalization_mode = postprocessed.normalization_mode
    log_event(
        LOGGER,
        logging.INFO,
        "infer_image_report_cleaned_output",
        request_id=request_id,
        text_source=postprocessed.text_source,
        preserved_raw_text=postprocessed.preserved_raw_text,
        text_before_cleanup_chars=len(postprocessed.text_before_cleanup),
        text_before_cleanup_preview=preview_text(postprocessed.text_before_cleanup, limit=_preview_limit()),
        text_after_cleanup_chars=len(postprocessed.text_after_cleanup),
        text_after_cleanup_preview=preview_text(postprocessed.text_after_cleanup, limit=_preview_limit()),
    )
    if normalization_mode not in {"parsed_sections", "json_structured"}:
        log_event(
            LOGGER,
            logging.WARNING,
            "infer_image_report_normalization_fallback",
            request_id=request_id,
            normalization_mode=normalization_mode,
            text_source=postprocessed.text_source,
            preserved_raw_text=postprocessed.preserved_raw_text,
        )

    log_event(
        LOGGER,
        logging.INFO,
        "infer_image_report_normalized",
        request_id=request_id,
        normalization_mode=normalization_mode,
        technique_chars=len(normalized_report["technique"]),
        findings_chars=len(normalized_report["findings"]),
        impression_chars=len(normalized_report["impression"]),
        explanation_summary_chars=len(normalized_report["explanation_summary"]),
        structured_findings_count=len(normalized_report["structured_findings"]),
        technique_preview=preview_text(normalized_report["technique"], limit=_preview_limit()),
        findings_preview=preview_text(normalized_report["findings"], limit=_preview_limit()),
        impression_preview=preview_text(normalized_report["impression"], limit=_preview_limit()),
        explanation_summary_preview=preview_text(normalized_report["explanation_summary"], limit=_preview_limit()),
    )

    log_event(
        LOGGER,
        logging.INFO,
        "infer_image_report_success",
        request_id=request_id,
        inference_time_ms=result.inference_time_ms,
        normalization_mode=normalization_mode,
        technique_present=bool(normalized_report["technique"]),
        findings_present=bool(normalized_report["findings"]),
        impression_present=bool(normalized_report["impression"]),
        structured_findings_count=len(normalized_report["structured_findings"]),
        response_report_preview=preview_text(normalized_report["report_text"], limit=_preview_limit()),
    )

    return ImageReportInferResponse(
        request_id=request_id,
        report_text=normalized_report["report_text"],
        technique=normalized_report["technique"],
        findings=normalized_report["findings"],
        impression=normalized_report["impression"],
        explanation_summary=normalized_report["explanation_summary"],
        structured_findings=normalized_report["structured_findings"],
        limitations=normalized_report["limitations"],
        model_id=result.model_id,
        device=result.device,
        inference_time_ms=result.inference_time_ms,
    )
