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
    ModelAccessError,
    ModelNotLoadedError,
    RemoteInferError,
    get_model_service,
)
from .schemas import (
    HealthResponse,
    ReportInferRequest,
    ReportInferResponse,
    TextInferRequest,
    TextInferResponse,
)
from .utils import build_report_prompt, ensure_request_id, log_event, setup_logging


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
    prompt = build_report_prompt(
        modality=payload.modality,
        body_part=payload.body_part,
        findings_input=payload.findings_input,
        clinical_context=payload.clinical_context,
    )

    do_sample = settings.medgemma_default_temperature > 0

    log_event(
        LOGGER,
        logging.INFO,
        "infer_report_request",
        request_id=request_id,
        study_id=payload.study_id,
        modality=payload.modality,
        body_part=payload.body_part,
    )

    result = service.generate_text(
        prompt,
        generation_config=GenerationConfig(
            max_new_tokens=settings.medgemma_max_new_tokens,
            temperature=settings.medgemma_default_temperature,
            top_p=settings.medgemma_default_top_p,
            do_sample=do_sample,
        ),
    )

    log_event(
        LOGGER,
        logging.INFO,
        "infer_report_success",
        request_id=request_id,
        inference_time_ms=result.inference_time_ms,
    )

    return ReportInferResponse(
        request_id=request_id,
        report_text=result.text,
        model_id=result.model_id,
        device=result.device,
        inference_time_ms=result.inference_time_ms,
    )
