"""Portable paths for shared dashboard releases, including older Mac/Windows exports."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

# Only known project directories can be relocated from an older absolute path.
# Never follow an arbitrary external path or guess from a result's basename.
LEGACY_PREFIXES = (
    ("data", "processed"), ("data", "reference"), ("data", "raw"), ("data", "sample"),
    ("artifacts",), ("config",), ("docs", "results"),
)


def resolve_project_path(root: Path, value: str | Path) -> Path:
    """Resolve under root; relocate recognized legacy paths without modifying the source."""
    root = root.resolve()
    text = str(value)
    windows = PureWindowsPath(text)
    saved = windows if windows.drive or "\\" in text else PurePosixPath(text)
    if not text or ".." in saved.parts:
        raise ValueError("결과 파일이 프로젝트 범위를 벗어났습니다.")
    native = Path(text)
    if native.is_absolute() and native.is_relative_to(root):
        candidate = native
    elif saved.is_absolute():
        suffixes = {
            saved.parts[i:]
            for i in range(1, len(saved.parts))
            for prefix in LEGACY_PREFIXES
            if saved.parts[i:i + len(prefix)] == prefix
        }
        if len(suffixes) != 1:
            raise ValueError("프로젝트 범위의 공유 결과 경로를 확인하세요.")
        candidate = root.joinpath(*next(iter(suffixes)))
    elif saved.drive or saved.root:
        raise ValueError("드라이브 기준 상대경로 대신 프로젝트 상대경로를 사용하세요.")
    else:
        candidate = root.joinpath(*saved.parts)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("결과 파일이 프로젝트 범위를 벗어났습니다.")
    return resolved


def relative_project_path(root: Path, value: str | Path) -> str:
    return resolve_project_path(root, value).relative_to(root.resolve()).as_posix()


def portable_release(root: Path, release: dict) -> dict:
    """Convert location fields only; preserve identities and hashes of result files."""
    result = dict(release)
    for key in ("backtest", "plan"):
        if result.get(key):
            result[key] = relative_project_path(root, result[key])
    for key in ("inputs", "source_paths"):
        if result.get(key):
            result[key] = {k: relative_project_path(root, v) for k, v in result[key].items()}
    if result.get("output_hashes"):
        result["output_hashes"] = {
            relative_project_path(root, path): digest
            for path, digest in result["output_hashes"].items()
        }
    return result


def portable_status(root: Path, state: dict) -> dict:
    result = dict(state)
    if result.get("active"):
        result["active"] = portable_release(root, result["active"])
    if result.get("releases"):
        result["releases"] = {
            name: portable_release(root, release) for name, release in result["releases"].items()
        }
    if "stages" in result:
        result["stages"] = [
            {**stage, "missing": [relative_project_path(root, p) for p in stage["missing"]]}
            if "missing" in stage else dict(stage)
            for stage in result["stages"]
        ]
    return result
