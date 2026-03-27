#!/usr/bin/env python3
import platform
import time

import torch


def main() -> None:
    print("=== Torch Smoke Test ===")
    print(f"python version: {platform.python_version()}")
    print(f"torch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    for idx in range(torch.cuda.device_count()):
        print(f"gpu[{idx}]: {torch.cuda.get_device_name(idx)}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"selected device: {device}")

    size = 2048 if device.type == "cuda" else 1024
    start = time.perf_counter()
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)
    c = a @ b

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    print(f"matmul shape: {tuple(c.shape)}")
    print(f"elapsed seconds: {elapsed:.4f}")


if __name__ == "__main__":
    main()
