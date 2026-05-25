import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
ENV_FILE = Path(os.environ.get("ENV_FILE", ROOT / ".env"))


def load_env_file(path: Path) -> None:
    for key, value in read_env_file(path).items():
        os.environ.setdefault(key, value)


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def require_env(name: str, fallback: str = "") -> str:
    value = os.environ.get(name, fallback).strip()
    if not value:
        print(f"ERROR: {name} is empty. Set it in {ENV_FILE} or export {name}.", file=sys.stderr)
        sys.exit(2)
    return value
