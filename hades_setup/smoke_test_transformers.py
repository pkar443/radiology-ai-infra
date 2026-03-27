#!/usr/bin/env python3
import os
import platform
import time

import torch
from transformers import pipeline, set_seed


def main() -> None:
    model_id = os.environ.get("SMOKE_TEST_MODEL_ID", "distilgpt2")
    prompt = os.environ.get(
        "SMOKE_TEST_PROMPT",
        "Radiology note summary: No acute cardiopulmonary abnormality. Impression:",
    )

    device = 0 if torch.cuda.is_available() else -1

    print("=== Transformers Smoke Test ===")
    print(f"python version: {platform.python_version()}")
    print(f"torch version: {torch.__version__}")
    print(f"transformers device: {'cuda:0' if device == 0 else 'cpu'}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"gpu[0]: {torch.cuda.get_device_name(0)}")
    print(f"model_id: {model_id}")
    print("note: first run will download the model weights into the Hugging Face cache.")

    set_seed(42)

    load_start = time.perf_counter()
    generator = pipeline(
        "text-generation",
        model=model_id,
        device=device,
    )
    load_elapsed = time.perf_counter() - load_start

    run_start = time.perf_counter()
    outputs = generator(
        prompt,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    run_elapsed = time.perf_counter() - run_start

    print(f"load_seconds: {load_elapsed:.4f}")
    print(f"generation_seconds: {run_elapsed:.4f}")
    print("generated_text:")
    print(outputs[0]["generated_text"])


if __name__ == "__main__":
    main()
