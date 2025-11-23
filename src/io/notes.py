from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn

from .paths import ensure_parent, out_path


NOTES_FILE = out_path('NOTLAR.md')


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_run_metadata(cmd: str, args: Any, seeds: List[int] | None = None, extras: Dict[str, Any] | None = None) -> None:
    """Persist run metadata (environment, command arguments, seeds)."""
    args_dict: Dict[str, Any] = {}
    if hasattr(args, '__dict__'):
        for key, val in vars(args).items():
            if key == 'func':
                continue
            args_dict[key] = _json_safe(val)
    meta = {
        'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'command': cmd,
        'argv': sys.argv,
        'args': args_dict,
        'seeds': list(seeds or []),
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'sklearn': sklearn.__version__,
        },
    }
    if extras:
        meta['extras'] = extras
    meta_path = out_path(f'reproducibility/{cmd}.json')
    ensure_parent(meta_path)
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def append_note(title: str, body: str, files: List[str] | None = None) -> None:
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')
    files = files or []
    ensure_parent(NOTES_FILE)
    with open(NOTES_FILE, 'a') as fh:
        fh.write(f"\n\n## {title} — {ts}\n")
        fh.write(body.strip() + "\n")
        if files:
            fh.write("\nİlgili Dosyalar:\n")
            for path in files:
                fh.write(f"- `{path}`\n")
