from __future__ import annotations

from pathlib import Path
import sys


def _exe_dir() -> Path:
    """Return the directory where the running executable/module lives.

    __main__ 中已将 cwd 切到可执行目录，但这里仍然稳妥获取。
    """
    try:
        return Path(sys.argv[0]).resolve().parent
    except Exception:
        return Path.cwd()


def _pkg_root() -> Path:
    """Return project root path: <repo>"""
    # 从 divere/__init__.py 获取项目根目录
    import divere
    return Path(divere.__file__).resolve().parent.parent


def resolve_data_path(*parts: str) -> Path:
    """Resolve a data file path with unified search order.

    Search priority:
      1) Executable dir (binary adjacent): ./<parts>
      2) Package data: divere/<parts>
      3) Current working directory (fallback): ./<parts>

    Raises FileNotFoundError if none exists.
    """
    candidates = [
        _exe_dir().joinpath(*parts),
        _pkg_root().joinpath(*parts),
        Path.cwd().joinpath(*parts),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Data path not found: " + "/".join(parts))


def get_data_dir(name: str) -> Path:
    """Return a data directory (assets/config/models) with unified search order.

    The directory is not created here; it must exist in one of the candidate locations.
    """
    if name not in {"assets", "config", "models"}:
        raise ValueError(f"Unsupported data dir: {name}")
    return resolve_data_path(name)


