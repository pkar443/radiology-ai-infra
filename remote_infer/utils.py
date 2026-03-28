from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("remote_infer.utils")
_SECTION_PATTERN = re.compile(r"(?im)^\s*(technique|findings|impression)\s*:\s*")


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


def resolve_report_prompt(
    prompt: str | None,
    study_id: str | None,
    modality: str | None,
    body_part: str | None,
    clinical_context: str | None,
    findings_input: str | None,
) -> str:
    if prompt and prompt.strip():
        return prompt.strip()

    parts: list[str] = []
    if study_id and study_id.strip():
        parts.append(f"Study ID: {study_id.strip()}")
    if modality and modality.strip():
        parts.append(f"Modality: {modality.strip()}")
    if body_part and body_part.strip():
        parts.append(f"Body part: {body_part.strip()}")
    if clinical_context and clinical_context.strip():
        parts.append(f"Clinical context: {clinical_context.strip()}")
    if findings_input and findings_input.strip():
        parts.append("Findings input:")
        parts.append(findings_input.strip())

    combined_prompt = "\n".join(parts).strip()
    if not combined_prompt:
        raise ValueError("Either prompt or report input fields must be provided.")

    return combined_prompt


def _clean_report_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()

    return cleaned


def _normalize_section_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _format_report_text(technique: str, findings: str, impression: str) -> str:
    sections = [
        ("Technique", technique.strip()),
        ("Findings", findings.strip()),
        ("Impression", impression.strip()),
    ]

    formatted_sections = []
    for label, content in sections:
        if content:
            formatted_sections.append(f"{label}:\n{content}")
        else:
            formatted_sections.append(f"{label}:")

    return "\n\n".join(formatted_sections)


def normalize_report_sections(raw_text: str) -> dict[str, str]:
    cleaned = _clean_report_text(raw_text)
    if not cleaned:
        raise ValueError("Generated report text is empty.")

    matches = list(_SECTION_PATTERN.finditer(cleaned))
    if not matches:
        findings = _normalize_section_text(cleaned)
        return {
            "report_text": _format_report_text("", findings, ""),
            "technique": "",
            "findings": findings,
            "impression": "",
        }

    sections = {"technique": "", "findings": "", "impression": ""}
    for index, match in enumerate(matches):
        section_name = match.group(1).lower()
        section_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        section_body = _normalize_section_text(cleaned[section_start:section_end])
        sections[section_name] = section_body

    if not sections["findings"] and not sections["impression"]:
        findings = _normalize_section_text(cleaned)
        return {
            "report_text": _format_report_text("", findings, ""),
            "technique": "",
            "findings": findings,
            "impression": "",
        }

    return {
        "report_text": _format_report_text(
            sections["technique"],
            sections["findings"],
            sections["impression"],
        ),
        "technique": sections["technique"],
        "findings": sections["findings"],
        "impression": sections["impression"],
    }
