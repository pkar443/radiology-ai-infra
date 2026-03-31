from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .config import Settings, get_settings
from .utils import (
    ensure_workspace_token,
    log_event,
    prepare_medgemma_multimodal_inputs,
    prepare_medgemma_text_inputs,
    preview_text,
    strip_prompt_echo_from_generated_text,
)

try:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except Exception:  # pragma: no cover - compatibility fallback
    GatedRepoError = HfHubHTTPError = RepositoryNotFoundError = Exception


LOGGER = logging.getLogger("remote_infer.model_loader")


class RemoteInferError(RuntimeError):
    status_code = 500
    error_code = "remote_infer_error"


class ModelNotLoadedError(RemoteInferError):
    status_code = 503
    error_code = "model_not_loaded"


class ModelAccessError(RemoteInferError):
    status_code = 503
    error_code = "model_access_error"


class InvalidPayloadError(RemoteInferError):
    status_code = 422
    error_code = "invalid_payload"


class InferenceOOMError(RemoteInferError):
    status_code = 507
    error_code = "cuda_oom"


class InferenceExecutionError(RemoteInferError):
    status_code = 500
    error_code = "inference_error"


def resolve_model_device_map(strategy: str, *, cuda_available: bool) -> str | dict[str, int]:
    normalized = (strategy or "single").strip().lower()

    if not cuda_available:
        return "cpu"

    if normalized in {"single", "single-gpu", "single_gpu", "first"}:
        return {"": 0}

    if normalized == "auto":
        return "auto"

    raise ValueError(
        f"Unsupported MEDGEMMA_DEVICE_MAP value {strategy!r}. Use 'single' or 'auto'."
    )


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    text: str
    inference_time_ms: int
    model_id: str
    device: str
    load_state: str
    input_ids_length: int
    prompt_token_count: int
    generated_token_count: int
    generated_sequence_length: int
    continuation_token_count: int
    generated_token_ids_head: tuple[int, ...]
    special_only_continuation: bool
    full_text: str = ""
    decoded_input_text: str = ""
    prompt_echo_removed: bool = False
    prompt_echo_offset: int | None = None


class MedGemmaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.processor: Any | None = None
        self.text_io: Any | None = None
        self.processor_label = "unloaded"
        self.text_io_label = "unloaded"
        self.device = "unloaded"
        self.dtype_name = "unknown"
        self.model_loaded = False
        self.load_error: str | None = None
        self._lock = threading.RLock()

    def _resolved_model_source(self) -> str:
        if self.settings.medgemma_model_path:
            model_path = Path(self.settings.medgemma_model_path).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            return str(model_path)
        return self.settings.medgemma_model_id

    def _preferred_dtype(self) -> torch.dtype:
        if not torch.cuda.is_available():
            return torch.float32

        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return torch.bfloat16 if bf16_supported else torch.float16

    def _resolved_device_map(self) -> str | dict[str, int]:
        return resolve_model_device_map(
            self.settings.medgemma_device_map,
            cuda_available=torch.cuda.is_available(),
        )

    def _load_processor(self, model_source: str) -> Any:
        try:
            return AutoProcessor.from_pretrained(model_source, use_fast=True)
        except Exception:
            return AutoProcessor.from_pretrained(model_source, use_fast=False)

    def _record_load_failure(self, error: str) -> None:
        self.model_loaded = False
        self.load_error = error
        self.model = None
        self.processor = None
        self.text_io = None
        self.processor_label = "unloaded"
        self.text_io_label = "unloaded"
        self.device = "unloaded"

    def load_model(self) -> None:
        with self._lock:
            if self.model_loaded:
                return

            model_source = self._resolved_model_source()
            ensure_workspace_token(self.settings.hf_home)
            dtype = self._preferred_dtype()
            device_map = self._resolved_device_map()

            try:
                processor = self._load_processor(model_source)
                text_io = processor if hasattr(processor, "apply_chat_template") else AutoTokenizer.from_pretrained(
                    model_source,
                    use_fast=False,
                )
                model = AutoModelForImageTextToText.from_pretrained(
                    model_source,
                    device_map=device_map,
                    dtype=dtype,
                    offload_buffers=True,
                )
            except ValueError as exc:
                self._record_load_failure(str(exc))
                raise ModelAccessError(self.load_error) from exc
            except FileNotFoundError as exc:
                self._record_load_failure(f"Local model path not found: {exc}")
                raise ModelAccessError(self.load_error) from exc
            except RepositoryNotFoundError as exc:
                self._record_load_failure(f"Model repository not found: {exc}")
                raise ModelAccessError(self.load_error) from exc
            except GatedRepoError as exc:
                self._record_load_failure(
                    "Access to the MedGemma repository is gated. Confirm Hugging Face access and token setup."
                )
                raise ModelAccessError(self.load_error) from exc
            except HfHubHTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code in (401, 403):
                    self._record_load_failure(
                        "Authentication or authorization failed while loading MedGemma from Hugging Face."
                    )
                    raise ModelAccessError(self.load_error) from exc
                self._record_load_failure(f"Hugging Face Hub error while loading model: {exc}")
                raise ModelAccessError(self.load_error) from exc
            except OSError as exc:
                message = str(exc)
                if "401" in message or "403" in message:
                    self._record_load_failure(
                        "Authentication or authorization failed while loading MedGemma from Hugging Face."
                    )
                    raise ModelAccessError(self.load_error) from exc
                self._record_load_failure(f"Model load OS error: {message}")
                raise ModelAccessError(self.load_error) from exc
            except Exception as exc:
                self._record_load_failure(f"Unexpected model load failure: {exc}")
                raise ModelAccessError(self.load_error) from exc

            self.processor = processor
            self.processor_label = type(processor).__name__
            self.text_io = text_io
            self.text_io_label = type(text_io).__name__
            self.model = model
            self.device = str(next(model.parameters()).device)
            self.dtype_name = str(dtype).replace("torch.", "")
            self.model_loaded = True
            self.load_error = None

            log_event(
                LOGGER,
                logging.INFO,
                "model_loaded",
                model_source=model_source,
                device=self.device,
                dtype=self.dtype_name,
                device_map_strategy=self.settings.medgemma_device_map,
                resolved_device_map=device_map,
                hf_device_map=getattr(model, "hf_device_map", None),
                processor=self.processor_label,
                text_io=self.text_io_label,
                supports_chat_template=hasattr(text_io, "apply_chat_template"),
                gpu_count=torch.cuda.device_count(),
            )

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": self.model_loaded,
            "model_id": self._safe_model_source(),
            "device": self.device,
            "gpu_count": torch.cuda.device_count(),
            "load_error": self.load_error,
        }

    def _safe_model_source(self) -> str:
        try:
            return self._resolved_model_source()
        except Exception:
            return self.settings.model_source

    def ensure_loaded(self) -> None:
        if not self.model_loaded or self.model is None or self.text_io is None:
            detail = self.load_error or "Model is not loaded. Restart the service and inspect startup logs."
            raise ModelNotLoadedError(detail)

    def ensure_multimodal_ready(self) -> None:
        self.ensure_loaded()
        if self.processor is None:
            detail = self.load_error or "The MedGemma processor is not loaded."
            raise ModelNotLoadedError(detail)

    def _prepare_text_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        assert self.text_io is not None
        prepared_inputs = prepare_medgemma_text_inputs(self.text_io, prompt)
        return {name: tensor for name, tensor in prepared_inputs.items()}

    def _prepare_multimodal_inputs(self, messages: list[dict[str, object]]) -> Any:
        assert self.processor is not None
        return prepare_medgemma_multimodal_inputs(self.processor, messages)

    def _decode_token_ids(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert self.text_io is not None

        decode_targets: list[Any] = [self.text_io]
        tokenizer = getattr(self.text_io, "tokenizer", None)
        if tokenizer is not None:
            decode_targets.append(tokenizer)

        first_result: str | None = None
        for target in decode_targets:
            if hasattr(target, "decode"):
                try:
                    decoded = target.decode(
                        token_ids,
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    decoded = target.decode(token_ids, skip_special_tokens=skip_special_tokens)
                if first_result is None:
                    first_result = decoded
                if decoded:
                    return decoded

            if hasattr(target, "batch_decode"):
                try:
                    decoded_batch = target.batch_decode(
                        [token_ids],
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    decoded_batch = target.batch_decode([token_ids], skip_special_tokens=skip_special_tokens)
                decoded = decoded_batch[0] if decoded_batch else ""
                if first_result is None:
                    first_result = decoded
                if decoded:
                    return decoded

        return first_result or ""

    def _special_token_ids(self) -> set[int]:
        assert self.text_io is not None

        special_ids = getattr(self.text_io, "all_special_ids", None)
        if special_ids is not None:
            return {int(token_id) for token_id in special_ids}

        tokenizer = getattr(self.text_io, "tokenizer", None)
        special_ids = getattr(tokenizer, "all_special_ids", None)
        if special_ids is not None:
            return {int(token_id) for token_id in special_ids}

        return set()

    def _resolve_pad_token_id(self) -> int | None:
        decode_targets: list[Any] = []

        if self.processor is not None:
            decode_targets.append(self.processor)
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is not None:
                decode_targets.append(tokenizer)

        if self.text_io is not None:
            decode_targets.append(self.text_io)
            tokenizer = getattr(self.text_io, "tokenizer", None)
            if tokenizer is not None:
                decode_targets.append(tokenizer)

        for target in decode_targets:
            pad_token_id = getattr(target, "pad_token_id", None)
            if isinstance(pad_token_id, int):
                return pad_token_id

        for target in decode_targets:
            eos_token_id = getattr(target, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                return eos_token_id

        return None

    def _move_inputs_to_target_device(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.model is not None
        target_device = next(self.model.parameters()).device
        target_dtype = next(self.model.parameters()).dtype

        moved_inputs: dict[str, torch.Tensor] = {}
        for name, tensor in inputs.items():
            if torch.is_floating_point(tensor):
                moved_inputs[name] = tensor.to(device=target_device, dtype=target_dtype)
            else:
                moved_inputs[name] = tensor.to(device=target_device)
        return moved_inputs

    def _move_multimodal_inputs_to_target_device(self, inputs: Any) -> Any:
        assert self.model is not None
        target_device = next(self.model.parameters()).device
        target_dtype = next(self.model.parameters()).dtype

        if hasattr(inputs, "to"):
            try:
                return inputs.to(target_device, dtype=target_dtype)
            except TypeError:
                return inputs.to(target_device)

        return self._move_inputs_to_target_device(dict(inputs))

    def _post_process_image_text(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert self.processor is not None

        try:
            decoded = self.processor.post_process_image_text_to_text(
                token_ids.detach().cpu(),
                skip_special_tokens=skip_special_tokens,
            )
        except TypeError:
            decoded = self.processor.post_process_image_text_to_text(token_ids.detach().cpu())

        if isinstance(decoded, list):
            return decoded[0] if decoded else ""
        return str(decoded)

    def generate_text(self, prompt: str, generation_config: GenerationConfig) -> GenerationResult:
        with self._lock:
            self.ensure_loaded()
            assert self.model is not None

            try:
                inputs = self._prepare_text_inputs(prompt)
                inputs = self._move_inputs_to_target_device(inputs)

                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": generation_config.max_new_tokens,
                    "do_sample": generation_config.do_sample,
                }
                pad_token_id = self._resolve_pad_token_id()
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = pad_token_id

                if generation_config.do_sample:
                    generate_kwargs["temperature"] = generation_config.temperature
                    generate_kwargs["top_p"] = generation_config.top_p

                input_ids_length = int(inputs["input_ids"].shape[-1])
                prompt_token_count = input_ids_length
                log_event(
                    LOGGER,
                    logging.INFO,
                    "generation_inputs_prepared",
                    input_ids_length=input_ids_length,
                    prompt_token_count=prompt_token_count,
                    text_io=self.text_io_label,
                    supports_chat_template=hasattr(self.text_io, "apply_chat_template"),
                    max_new_tokens=generation_config.max_new_tokens,
                    do_sample=generation_config.do_sample,
                )

                infer_start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generate_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time_ms = int((time.perf_counter() - infer_start) * 1000)

                generated_sequence_length = int(output_ids.shape[-1])
                generated_tokens = output_ids[0][prompt_token_count:]
                continuation_token_ids = [int(token_id) for token_id in generated_tokens.detach().cpu().tolist()]
                visible_generated_text: str
                raw_generated_text: str
                raw_generated_text = self._decode_token_ids(
                    continuation_token_ids,
                    skip_special_tokens=False,
                )
                visible_generated_text = self._decode_token_ids(
                    continuation_token_ids,
                    skip_special_tokens=True,
                )

                continuation_token_count = len(continuation_token_ids)
                special_token_ids = self._special_token_ids()
                special_only_continuation = bool(continuation_token_ids) and all(
                    token_id in special_token_ids for token_id in continuation_token_ids
                )
                log_event(
                    LOGGER,
                    logging.INFO,
                    "generation_continuation_decoded",
                    input_ids_length=input_ids_length,
                    prompt_token_count=prompt_token_count,
                    generated_token_count=continuation_token_count,
                    generated_sequence_length=generated_sequence_length,
                    continuation_token_count=continuation_token_count,
                    continuation_token_ids_head=tuple(continuation_token_ids[:50]),
                    special_only_continuation=special_only_continuation,
                    decoded_continuation_raw_preview=preview_text(raw_generated_text, limit=600),
                    decoded_continuation_visible_preview=preview_text(visible_generated_text, limit=600),
                )

                return GenerationResult(
                    raw_text=raw_generated_text,
                    text=visible_generated_text,
                    inference_time_ms=inference_time_ms,
                    model_id=self._safe_model_source(),
                    device=self.device,
                    load_state="loaded",
                    input_ids_length=input_ids_length,
                    prompt_token_count=prompt_token_count,
                    generated_token_count=continuation_token_count,
                    generated_sequence_length=generated_sequence_length,
                    continuation_token_count=continuation_token_count,
                    generated_token_ids_head=tuple(continuation_token_ids[:50]),
                    special_only_continuation=special_only_continuation,
                    full_text=visible_generated_text,
                    decoded_input_text=prompt,
                )
            except torch.cuda.OutOfMemoryError as exc:
                raise InferenceOOMError("CUDA out of memory during MedGemma inference.") from exc
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise InferenceOOMError("CUDA out of memory during MedGemma inference.") from exc
                raise InferenceExecutionError(f"Runtime error during generation: {exc}") from exc
            except HfHubHTTPError as exc:
                raise ModelAccessError(f"Hugging Face access error during inference: {exc}") from exc
            except Exception as exc:
                raise InferenceExecutionError(f"Unexpected inference error: {exc}") from exc

    def generate_image_report(
        self,
        messages: list[dict[str, object]],
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        with self._lock:
            self.ensure_multimodal_ready()
            assert self.model is not None

            try:
                inputs = self._prepare_multimodal_inputs(messages)
                inputs = self._move_multimodal_inputs_to_target_device(inputs)

                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": generation_config.max_new_tokens,
                    "do_sample": generation_config.do_sample,
                }
                pad_token_id = self._resolve_pad_token_id()
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = pad_token_id

                if generation_config.do_sample:
                    generate_kwargs["temperature"] = generation_config.temperature
                    generate_kwargs["top_p"] = generation_config.top_p

                input_ids_length = int(inputs["input_ids"].shape[-1])
                prompt_token_count = input_ids_length
                log_event(
                    LOGGER,
                    logging.INFO,
                    "multimodal_generation_inputs_prepared",
                    input_ids_length=input_ids_length,
                    prompt_token_count=prompt_token_count,
                    processor=self.processor_label,
                    prepared_input_keys=sorted(inputs.keys()),
                    input_ids_dtype=str(inputs["input_ids"].dtype),
                    pixel_values_present="pixel_values" in inputs,
                    pixel_values_shape=tuple(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
                    pixel_values_dtype=str(inputs["pixel_values"].dtype) if "pixel_values" in inputs else None,
                    max_new_tokens=generation_config.max_new_tokens,
                    do_sample=generation_config.do_sample,
                )

                infer_start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generate_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time_ms = int((time.perf_counter() - infer_start) * 1000)

                generated_sequence_length = int(output_ids.shape[-1])
                generated_tokens = output_ids[0][prompt_token_count:]
                continuation_token_ids = [int(token_id) for token_id in generated_tokens.detach().cpu().tolist()]
                continuation_token_count = len(continuation_token_ids)
                special_token_ids = self._special_token_ids()
                special_only_continuation = bool(continuation_token_ids) and all(
                    token_id in special_token_ids for token_id in continuation_token_ids
                )

                decoded_generated_text = self._post_process_image_text(output_ids, skip_special_tokens=True)
                decoded_input_text = self._post_process_image_text(inputs["input_ids"], skip_special_tokens=True)
                echo_cleanup = strip_prompt_echo_from_generated_text(decoded_generated_text, decoded_input_text)
                cleaned_generated_text = echo_cleanup.text

                log_event(
                    LOGGER,
                    logging.INFO,
                    "multimodal_generation_continuation_decoded",
                    input_ids_length=input_ids_length,
                    prompt_token_count=prompt_token_count,
                    generated_token_count=continuation_token_count,
                    generated_sequence_length=generated_sequence_length,
                    continuation_token_count=continuation_token_count,
                    continuation_token_ids_head=tuple(continuation_token_ids[:50]),
                    special_only_continuation=special_only_continuation,
                    prompt_echo_removed=echo_cleanup.prompt_echo_removed,
                    prompt_echo_offset=echo_cleanup.matched_offset,
                    decoded_input_preview=preview_text(decoded_input_text, limit=600),
                    decoded_output_preview=preview_text(decoded_generated_text, limit=600),
                    cleaned_output_preview=preview_text(cleaned_generated_text, limit=600),
                )

                return GenerationResult(
                    raw_text=cleaned_generated_text,
                    text=cleaned_generated_text,
                    inference_time_ms=inference_time_ms,
                    model_id=self._safe_model_source(),
                    device=self.device,
                    load_state="loaded",
                    input_ids_length=input_ids_length,
                    prompt_token_count=prompt_token_count,
                    generated_token_count=continuation_token_count,
                    generated_sequence_length=generated_sequence_length,
                    continuation_token_count=continuation_token_count,
                    generated_token_ids_head=tuple(continuation_token_ids[:50]),
                    special_only_continuation=special_only_continuation,
                    full_text=decoded_generated_text,
                    decoded_input_text=decoded_input_text,
                    prompt_echo_removed=echo_cleanup.prompt_echo_removed,
                    prompt_echo_offset=echo_cleanup.matched_offset,
                )
            except torch.cuda.OutOfMemoryError as exc:
                raise InferenceOOMError("CUDA out of memory during MedGemma multimodal inference.") from exc
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise InferenceOOMError("CUDA out of memory during MedGemma multimodal inference.") from exc
                raise InferenceExecutionError(f"Runtime error during multimodal generation: {exc}") from exc
            except HfHubHTTPError as exc:
                raise ModelAccessError(f"Hugging Face access error during multimodal inference: {exc}") from exc
            except Exception as exc:
                raise InferenceExecutionError(f"Unexpected multimodal inference error: {exc}") from exc


_SERVICE = MedGemmaService(get_settings())


def get_model_service() -> MedGemmaService:
    return _SERVICE
