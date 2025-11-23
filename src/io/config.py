from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable
import yaml


def load_yaml(path: str | bytes) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def resolve_grid(cfg: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Iterable[Any]]:
    return {k: cfg[k] for k in keys if k in cfg}
