from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("remote_infer.utils")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={json.dumps(value, ensure_ascii=True)}")
    logger.log(level, " ".join(parts))


def bootstrap_env_from_file(env_path: Path) -> None:
    if not env_path.is_file():
        return

    try:
        completed = subprocess.run(
            [
                "bash",
                "-lc",
                'set -a; source "$1" >/dev/null 2>&1; env -0',
                "bash",
                str(env_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "env_bootstrap_failed",
            env_file=str(env_path),
            returncode=exc.returncode,
        )
        return

    for chunk in completed.stdout.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        key, value = chunk.split(b"=", 1)
        os.environ.setdefault(key.decode("utf-8"), value.decode("utf-8"))


def ensure_workspace_token(hf_home: str) -> str | None:
    hf_home_path = Path(hf_home).expanduser()
    hf_home_path.mkdir(parents=True, exist_ok=True)

    target_token_path = hf_home_path / "token"
    if target_token_path.is_file() and target_token_path.stat().st_size > 0:
        return str(target_token_path)

    default_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if default_token_path.is_file() and default_token_path.stat().st_size > 0:
        shutil.copy2(default_token_path, target_token_path)
        target_token_path.chmod(0o600)
        return str(target_token_path)

    return None


def ensure_request_id(request_id: str | None) -> str:
    if request_id and request_id.strip():
        return request_id.strip()
    return uuid.uuid4().hex


def build_report_prompt(
    modality: str,
    body_part: str,
    findings_input: str,
    clinical_context: str | None,
) -> str:
    clinical_text = clinical_context.strip() if clinical_context else "Not provided."

    return (
        "You are drafting a concise radiology report.\n"
        "Use only the information provided below.\n"
        "Do not invent patient demographics, history, or findings.\n"
        "Return plain text only.\n"
        "Write the report with exactly these sections:\n"
        "Findings:\n"
        "Impression:\n\n"
        f"Modality: {modality}\n"
        f"Body part: {body_part}\n"
        f"Clinical context: {clinical_text}\n"
        f"Source findings: {findings_input.strip()}\n"
    )
