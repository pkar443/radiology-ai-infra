from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/f/f0/Normal_contrast_enhanced_abdominal_CT.jpg"
DEFAULT_INSTRUCTION = (
    "You are reviewing a single CT image. Look carefully at the image provided below."
)
DEFAULT_QUERY = (
    "Describe what you can see in plain text. Keep the answer concise. "
    "Do not use markdown. Do not add section headers."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw MedGemma image-text test using one downloaded CT JPEG."
    )
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device-map", default=None)
    return parser.parse_args()


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return torch.bfloat16 if bf16_supported else torch.float16


def resolve_cache_dir(cli_cache_dir: str | None) -> Path:
    if cli_cache_dir:
        return Path(cli_cache_dir).expanduser()

    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "hades_setup" / ".cache" / "raw_image_test"


def resolve_device_map(strategy: str | None) -> tuple[str | dict[str, int], str]:
    normalized = (strategy or os.environ.get("MEDGEMMA_DEVICE_MAP", "single")).strip().lower() or "single"

    if not torch.cuda.is_available():
        return "cpu", "cpu"

    if normalized in {"single", "single-gpu", "single_gpu", "first"}:
        return {"": 0}, "single"

    if normalized == "auto":
        return "auto", "auto"

    raise ValueError(f"Unsupported device map strategy {normalized!r}. Use 'single' or 'auto'.")


def download_image(image_url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = image_url.rstrip("/").split("/")[-1] or "ct_test_image.jpg"
    image_path = cache_dir / filename
    if image_path.exists() and image_path.stat().st_size > 0:
        return image_path

    request = urllib.request.Request(
        image_url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Codex-MedGemma-Test/1.0",
        },
    )
    with urllib.request.urlopen(request) as response:
        image_bytes = response.read()
    image_path.write_bytes(image_bytes)
    return image_path


def image_file_to_data_url(image_path: Path) -> tuple[str, tuple[int, int], str]:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        image_size = rgb_image.size
        with io.BytesIO() as buffer:
            rgb_image.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}", image_size, "RGB"


def main() -> int:
    args = parse_args()
    model_source = os.environ.get("MEDGEMMA_MODEL_PATH", "").strip() or os.environ.get(
        "MEDGEMMA_MODEL_ID",
        "google/medgemma-1.5-4b-it",
    ).strip()

    dtype = preferred_dtype()
    device_map, device_map_label = resolve_device_map(args.device_map)
    cache_dir = resolve_cache_dir(args.cache_dir)
    image_path = download_image(args.image_url, cache_dir)
    image_data_url, image_size, image_mode = image_file_to_data_url(image_path)

    print("=== Raw MedGemma Image Test ===")
    print(f"model_source: {model_source}")
    print(f"image_url: {args.image_url}")
    print(f"cached_image_path: {image_path}")
    print(f"image_size: {image_size}")
    print(f"image_mode: {image_mode}")
    print(f"image_data_url_prefix: {image_data_url[:32]}...")
    print(f"instruction: {args.instruction}")
    print(f"query: {args.query}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"device_map: {device_map_label}")
    print()

    processor = AutoProcessor.from_pretrained(model_source, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        device_map=device_map,
        dtype=dtype,
        offload_buffers=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.instruction},
                {"type": "image", "image": image_data_url},
                {"type": "text", "text": args.query},
            ],
        }
    ]
    display_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.instruction},
                {"type": "image", "image": f"{image_data_url[:32]}..."},
                {"type": "text", "text": args.query},
            ],
        }
    ]

    print("=== Notebook-Style Messages ===")
    print(display_messages)
    print()

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )

    print("=== Prepared Inputs ===")
    print(f"keys: {sorted(inputs.keys())}")
    for key, value in inputs.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else "<no-shape>"
        dtype_name = str(getattr(value, "dtype", type(value).__name__))
        print(f"{key}: shape={shape} dtype={dtype_name}")
    print()

    try:
        moved_inputs = inputs.to(model.device, dtype=dtype)
    except TypeError:
        moved_inputs = inputs.to(model.device)

    pad_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    start = time.perf_counter()
    with torch.inference_mode():
        generate_kwargs = {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        }
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        generated_sequence = model.generate(**moved_inputs, **generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    continuation_ids = generated_sequence[0][moved_inputs["input_ids"].shape[-1] :].detach().cpu().tolist()
    raw_full_output = processor.post_process_image_text_to_text(
        generated_sequence.detach().cpu(),
        skip_special_tokens=False,
    )[0]
    visible_full_output = processor.post_process_image_text_to_text(
        generated_sequence.detach().cpu(),
        skip_special_tokens=True,
    )[0]

    tokenizer = getattr(processor, "tokenizer", None)
    raw_continuation = ""
    visible_continuation = ""
    if tokenizer is not None:
        raw_continuation = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        visible_continuation = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    print("=== Generation Result ===")
    print(f"inference_time_ms: {elapsed_ms}")
    print(f"generated_sequence_shape: {tuple(generated_sequence.shape)}")
    print(f"continuation_token_count: {len(continuation_ids)}")
    print(f"continuation_token_ids_head: {continuation_ids[:80]}")
    print()

    print("=== Raw Full Output (No Hades Post-Processing) ===")
    print(raw_full_output)
    print()

    print("=== Visible Full Output (No Hades Post-Processing) ===")
    print(visible_full_output)
    print()

    print("=== Raw Continuation Decode (Tokenizer, No Hades Post-Processing) ===")
    print(raw_continuation)
    print()

    print("=== Visible Continuation Decode (Tokenizer, No Hades Post-Processing) ===")
    print(visible_continuation)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
