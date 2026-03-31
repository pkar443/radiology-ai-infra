#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

try:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except Exception:  # pragma: no cover - fallback for older hub builds
    GatedRepoError = HfHubHTTPError = RepositoryNotFoundError = Exception


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def resolve_model_name() -> str:
    model_path = os.environ.get("MEDGEMMA_MODEL_PATH", "").strip()
    model_id = os.environ.get("MEDGEMMA_MODEL_ID", "").strip()

    if model_path:
        expanded_path = Path(model_path).expanduser()
        if not expanded_path.exists():
            raise FileNotFoundError(expanded_path)
        return str(expanded_path)
    if model_id:
        return model_id
    raise ValueError("Set MEDGEMMA_MODEL_ID or MEDGEMMA_MODEL_PATH before running this script.")


def resolve_device_map() -> tuple[str | dict[str, int], str]:
    normalized = os.environ.get("MEDGEMMA_DEVICE_MAP", "single").strip().lower() or "single"

    if not torch.cuda.is_available():
        return "cpu", "cpu"

    if normalized in {"single", "single-gpu", "single_gpu", "first"}:
        return {"": 0}, "single"

    if normalized == "auto":
        return "auto", "auto"

    raise ValueError(f"Unsupported MEDGEMMA_DEVICE_MAP value {normalized!r}. Use 'single' or 'auto'.")


def build_messages() -> list[dict]:
    prompt_text = (
        "You are reviewing a chest radiograph report draft.\n"
        "Summarize the likely impression in 2 concise sentences.\n\n"
        "Findings: Mild bibasilar linear atelectatic change. No focal consolidation, pleural effusion, "
        "or pneumothorax. Cardiomediastinal silhouette is within normal size limits.\n"
        "Question: What is the impression?"
    )

    return [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]


def prepare_inputs(model_name: str) -> tuple[object, dict[str, torch.Tensor]]:
    messages = build_messages()

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        if hasattr(processor, "apply_chat_template"):
            tokenized = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return processor, dict(tokenized)
    except Exception:
        processor = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_text = messages[0]["content"][0]["text"]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return tokenizer, dict(tokenizer(prompt_text, return_tensors="pt"))


def main() -> int:
    model_name = resolve_model_name()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_label = "cuda" if torch.cuda.is_available() else "cpu"
    device_map, device_map_label = resolve_device_map()

    print(f"model source: {model_name}")
    print(f"device: {device_label}")
    print(f"dtype: {dtype}")
    print(f"device_map: {device_map_label}")

    try:
        load_start = time.perf_counter()
        text_io, inputs = prepare_inputs(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            offload_buffers=True,
        )
        load_seconds = time.perf_counter() - load_start

        target_device = next(model.parameters()).device
        inputs = {name: tensor.to(target_device) for name, tensor in inputs.items()}

        infer_start = time.perf_counter()
        with torch.inference_mode():
            pad_token_id = getattr(text_io, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(text_io, "eos_token_id", None)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_seconds = time.perf_counter() - infer_start

        prompt_token_count = inputs["input_ids"].shape[-1]
        generated_tokens = output_ids[0][prompt_token_count:]
        generated_text = text_io.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"load_time_seconds: {load_seconds:.4f}")
        print(f"inference_time_seconds: {infer_seconds:.4f}")
        print("generated_text:")
        print(generated_text if generated_text else "<empty generation>")
        return 0

    except ValueError as exc:
        eprint(f"Configuration error: {exc}")
        return 2
    except FileNotFoundError as exc:
        eprint(f"Local model path not found: {exc}")
        return 3
    except RepositoryNotFoundError as exc:
        eprint(f"Model repository not found: {exc}")
        return 4
    except GatedRepoError as exc:
        eprint("Access to the model is gated. Log in with `hf auth login` and accept the model terms on Hugging Face.")
        eprint(str(exc))
        return 5
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            eprint("Authentication or authorization failed. Run `hf auth login` and confirm access to the MedGemma model.")
            eprint(str(exc))
            return 6
        eprint(f"Hugging Face Hub error: {exc}")
        return 7
    except torch.cuda.OutOfMemoryError as exc:
        eprint("CUDA out of memory while loading or running MedGemma.")
        eprint(str(exc))
        return 8
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            eprint("CUDA out of memory while loading or running MedGemma.")
            eprint(str(exc))
            return 8
        eprint(f"Runtime error: {exc}")
        return 9
    except OSError as exc:
        message = str(exc)
        if "401" in message or "403" in message:
            eprint("Authentication or authorization failed. Run `hf auth login` and confirm access to the MedGemma model.")
            eprint(message)
            return 6
        if "not found" in message.lower():
            eprint(f"Model or required files were not found: {message}")
            return 3
        eprint(f"OS error while loading MedGemma: {message}")
        return 10
    except Exception as exc:
        eprint(f"Unexpected error: {exc}")
        return 11


if __name__ == "__main__":
    raise SystemExit(main())
