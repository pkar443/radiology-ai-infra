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
from .utils import ensure_workspace_token, log_event

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


class InferenceOOMError(RemoteInferError):
    status_code = 507
    error_code = "cuda_oom"


class InferenceExecutionError(RemoteInferError):
    status_code = 500
    error_code = "inference_error"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


@dataclass(frozen=True)
class GenerationResult:
    text: str
    inference_time_ms: int
    model_id: str
    device: str
    load_state: str


class MedGemmaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.text_io: Any | None = None
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

    def _load_text_io(self, model_source: str) -> Any:
        try:
            processor = AutoProcessor.from_pretrained(model_source, use_fast=False)
            if hasattr(processor, "apply_chat_template"):
                return processor
        except Exception:
            pass

        return AutoTokenizer.from_pretrained(model_source)

    def _record_load_failure(self, error: str) -> None:
        self.model_loaded = False
        self.load_error = error
        self.model = None
        self.text_io = None
        self.device = "unloaded"

    def load_model(self) -> None:
        with self._lock:
            if self.model_loaded:
                return

            model_source = self._resolved_model_source()
            ensure_workspace_token(self.settings.hf_home)
            dtype = self._preferred_dtype()

            try:
                text_io = self._load_text_io(model_source)
                model = AutoModelForImageTextToText.from_pretrained(
                    model_source,
                    device_map="auto",
                    dtype=dtype,
                )
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

            self.text_io = text_io
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

    def _prepare_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        assert self.text_io is not None

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if hasattr(self.text_io, "apply_chat_template"):
            try:
                tokenized = self.text_io.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                return dict(tokenized)
            except Exception:
                prompt = self.text_io.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        return dict(self.text_io(prompt, return_tensors="pt"))

    def generate_text(self, prompt: str, generation_config: GenerationConfig) -> GenerationResult:
        with self._lock:
            self.ensure_loaded()
            assert self.model is not None

            try:
                inputs = self._prepare_inputs(prompt)
                target_device = next(self.model.parameters()).device
                inputs = {name: tensor.to(target_device) for name, tensor in inputs.items()}

                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": generation_config.max_new_tokens,
                    "do_sample": generation_config.do_sample,
                }

                if generation_config.do_sample:
                    generate_kwargs["temperature"] = generation_config.temperature
                    generate_kwargs["top_p"] = generation_config.top_p

                prompt_token_count = inputs["input_ids"].shape[-1]

                infer_start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generate_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time_ms = int((time.perf_counter() - infer_start) * 1000)

                generated_tokens = output_ids[0][prompt_token_count:]
                if hasattr(self.text_io, "decode"):
                    generated_text = self.text_io.decode(generated_tokens, skip_special_tokens=True).strip()
                elif hasattr(self.text_io, "tokenizer") and hasattr(self.text_io.tokenizer, "decode"):
                    generated_text = self.text_io.tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True,
                    ).strip()
                else:
                    raise InferenceExecutionError("Loaded MedGemma text IO object cannot decode generated tokens.")

                return GenerationResult(
                    text=generated_text,
                    inference_time_ms=inference_time_ms,
                    model_id=self._safe_model_source(),
                    device=self.device,
                    load_state="loaded",
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


_SERVICE = MedGemmaService(get_settings())


def get_model_service() -> MedGemmaService:
    return _SERVICE
