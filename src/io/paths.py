from pathlib import Path


def project_root() -> Path:
    # src/io/paths.py -> src/io -> src -> root
    return Path(__file__).resolve().parents[2]


ROOT = project_root()


def out_path(name: str) -> Path:
    """Return a path in project root for main.tex compatibility."""
    return ROOT / name


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
