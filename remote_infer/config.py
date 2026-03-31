from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .utils import bootstrap_env_from_file


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / "hades_setup" / ".env"


def _read_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _read_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def _read_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


bootstrap_env_from_file(DEFAULT_ENV_FILE)


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    env_file: Path
    medgemma_model_id: str
    medgemma_model_path: str
    medgemma_device_map: str
    hf_home: str
    huggingface_hub_cache: str
    cuda_visible_devices: str
    remote_infer_host: str
    remote_infer_port: int
    remote_infer_auth_token: str
    remote_infer_debug: bool
    medgemma_max_new_tokens: int
    medgemma_default_temperature: float
    medgemma_default_top_p: float
    medgemma_report_max_new_tokens: int
    medgemma_report_do_sample: bool
    medgemma_report_temperature: float
    medgemma_report_top_p: float
    medgemma_image_report_max_new_tokens: int
    medgemma_image_report_do_sample: bool
    medgemma_image_report_temperature: float
    medgemma_image_report_top_p: float

    @property
    def model_source(self) -> str:
        if self.medgemma_model_path:
            return self.medgemma_model_path
        return self.medgemma_model_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    repo_root = REPO_ROOT
    hf_home = os.environ.get("HF_HOME", str(repo_root / "hf_cache")).strip()
    huggingface_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home).strip()

    return Settings(
        repo_root=repo_root,
        env_file=DEFAULT_ENV_FILE,
        medgemma_model_id=os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it").strip(),
        medgemma_model_path=os.environ.get("MEDGEMMA_MODEL_PATH", "").strip(),
        medgemma_device_map=os.environ.get("MEDGEMMA_DEVICE_MAP", "single").strip() or "single",
        hf_home=hf_home,
        huggingface_hub_cache=huggingface_hub_cache,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "").strip(),
        remote_infer_host=os.environ.get("REMOTE_INFER_HOST", "127.0.0.1").strip() or "127.0.0.1",
        remote_infer_port=_read_int("REMOTE_INFER_PORT", 8009),
        remote_infer_auth_token=os.environ.get("REMOTE_INFER_AUTH_TOKEN", "").strip(),
        remote_infer_debug=_read_bool("REMOTE_INFER_DEBUG", False),
        medgemma_max_new_tokens=_read_int("MEDGEMMA_MAX_NEW_TOKENS", 180),
        medgemma_default_temperature=_read_float("MEDGEMMA_DEFAULT_TEMPERATURE", 0.0),
        medgemma_default_top_p=_read_float("MEDGEMMA_DEFAULT_TOP_P", 1.0),
        medgemma_report_max_new_tokens=_read_int("MEDGEMMA_REPORT_MAX_NEW_TOKENS", 240),
        medgemma_report_do_sample=_read_bool("MEDGEMMA_REPORT_DO_SAMPLE", False),
        medgemma_report_temperature=_read_float("MEDGEMMA_REPORT_TEMPERATURE", 0.0),
        medgemma_report_top_p=_read_float("MEDGEMMA_REPORT_TOP_P", 1.0),
        medgemma_image_report_max_new_tokens=_read_int("MEDGEMMA_IMAGE_REPORT_MAX_NEW_TOKENS", 900),
        medgemma_image_report_do_sample=_read_bool("MEDGEMMA_IMAGE_REPORT_DO_SAMPLE", False),
        medgemma_image_report_temperature=_read_float("MEDGEMMA_IMAGE_REPORT_TEMPERATURE", 0.0),
        medgemma_image_report_top_p=_read_float("MEDGEMMA_IMAGE_REPORT_TOP_P", 1.0),
    )
