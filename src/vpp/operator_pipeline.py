"""담당: 대시보드 통합 — 팀 Drive 에서 받은 예측 결과 파일 → MILP → 화면.

팀원이 모델을 바꾸면 결과 파일(발전량 parquet, SMP csv)을 Drive 로 공유하고 각자 같은 경로에 둔다.
이 모듈은 그 파일과 ESS·REC 설정, 최적화 코드의 **내용**이 바뀌었을 때만 비교 백테스트와 48시간
운영계획을 다시 풀고, 검산을 통과한 결과를 화면의 공통 결과로 올린다. 모델 학습·추론은 하지 않는다
(담당자가 돌려 공유한다). Git 커밋·푸시, 다운로드도 하지 않는다.

    python -m vpp.operator_pipeline      # 한 번 실행. 대시보드가 열려 있으면 30초마다 변경 확인
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/operator_pipeline.yaml"
INPUT_KEYS = ("solar", "wind", "smp", "plants", "ess", "rec")  # 공유 결과 파일 + 설정
ENGINE_FILES = (  # 이 코드가 바뀌면 결과가 달라지므로 다시 푼다 (팀원 모델 코드는 보지 않음)
    "optimization.py", "baselines.py", "evaluation.py", "validation.py", "rec.py",
    "forecast_inputs.py", "forecast_rolling.py", "forecast_backtest.py", "operating_plan.py",
    "operator_pipeline.py",
)


def _prepare_windows_lock(stream: TextIO) -> None:
    """Windows byte-range locks require an existing byte to lock."""
    if os.fstat(stream.fileno()).st_size == 0:
        stream.write("\0")
        stream.flush()
    stream.seek(0)


def _lock_shared(stream: TextIO) -> None:
    if os.name == "nt":
        _prepare_windows_lock(stream)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(stream, fcntl.LOCK_SH)


def _lock_exclusive(stream: TextIO, *, blocking: bool) -> None:
    if os.name == "nt":
        _prepare_windows_lock(stream)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        try:
            msvcrt.locking(stream.fileno(), mode, 1)
        except OSError as exc:
            raise BlockingIOError from exc
    else:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(stream, flags)


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def config(root: Path = ROOT) -> dict:
    cfg = yaml.safe_load(
        (root / "config/operator_pipeline.yaml").read_text(encoding="utf-8")
    )
    for name in (*INPUT_KEYS, "output"):
        if cfg.get(name):
            cfg[name] = str((root / cfg[name]).resolve())
    return cfg


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _rec_table(root: Path, rec_path: str | None) -> Path | None:
    """REC 설정이 월별 단가표를 가리키면 그 경로 (표 내용이 바뀌어도 다시 풀어야 한다)."""
    if not rec_path or not Path(rec_path).exists():
        return None
    raw = yaml.safe_load(Path(rec_path).read_text(encoding="utf-8")) or {}
    if not raw.get("price_table"):
        return None
    table = Path(raw["price_table"])
    if table.is_absolute():
        return table
    return root / table if (root / table).exists() else Path(rec_path).parent / table


def revision(root: Path, cfg: dict) -> str:
    """공유 파일·설정·최적화 코드의 내용 지문. 파일명이 같아도 내용이 바뀌면 달라진다."""
    h = hashlib.sha256(json.dumps({k: cfg.get(k) for k in INPUT_KEYS}, sort_keys=True).encode())
    for name in ENGINE_FILES:
        path = root / "src/vpp" / name
        h.update(name.encode())
        h.update(path.read_bytes() if path.exists() else b"missing")
    for key in INPUT_KEYS:
        path = Path(cfg[key]) if cfg.get(key) else None
        h.update((digest(path) if path and path.exists() else f"missing:{key}").encode())
    table = _rec_table(root, cfg.get("rec"))
    if table is not None:
        h.update((digest(table) if table.exists() else f"missing:{table}").encode())
    return h.hexdigest()


def status(root: Path = ROOT) -> dict:
    return read_json(Path(config(root)["output"]) / "status.json")


def requirements(cfg: dict) -> list[dict]:
    """화면에 보여 줄 '같은 경로에 있어야 하는 공유 파일' 목록."""
    rows = (
        ("태양광 예측", "solar", "발전량 담당이 Drive 로 공유한 parquet"),
        ("풍력 예측", "wind", "발전량 담당이 Drive 로 공유한 parquet"),
        ("SMP 예측", "smp", "SMP 담당이 Drive 로 공유한 csv"),
        ("발전소 기준정보", "plants", "Git 포함"),
        ("ESS 설정", "ess", "config/"),
        ("REC 설정", "rec", "config/ (비우면 REC 미적용)"),
    )
    out = []
    for name, key, purpose in rows:
        path = cfg.get(key)
        ready = Path(path).is_file() if path else key == "rec"
        out.append({"name": name, "ready": ready, "path": path or "(미지정)", "purpose": purpose})
    return out


def refresh_dispatch(root: Path, cfg: dict, sources: dict, *, label: str) -> dict:
    """Snapshot inputs; deduplicate by content+settings+solver code; publish verified results."""
    from vpp import forecast_inputs as fi
    from vpp.forecast_backtest import run_forecast_backtest
    from vpp.operating_plan import run_operating_plan
    from vpp.optimization import EssParams

    out = Path(cfg["output"])
    paths = {k: Path(v) for k, v in sources.items()}
    for key in ("plants", "ess", "rec"):
        if cfg.get(key):
            paths[key] = Path(cfg[key])
    code = hashlib.sha256()
    for p in sorted((root / "src/vpp").glob("*.py")):
        code.update(p.read_bytes())
    from vpp.rec import load_rec, rec_metadata

    rec = load_rec(cfg.get("rec"))
    if rec is not None and rec.table_source:
        paths["rec_table"] = Path(rec.table_source)
    hashes = {k: digest(p) for k, p in paths.items()}
    rec_info = rec_metadata(rec)
    signature = {"files": hashes, "engine": code.hexdigest(), "rec": rec_info}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    done = out / "releases" / f"{key}.json"
    previous = read_json(done)
    if (
        previous
        and previous.get("output_hashes")
        and all(
            Path(p).exists() and digest(Path(p)) == expected
            for p, expected in previous["output_hashes"].items()
        )
    ):
        return previous
    snap = out / "snapshots" / key
    snap.mkdir(parents=True, exist_ok=True)
    copies = {}
    for name, p in paths.items():
        # REC YAML relative references must stay relative to its original location.
        if name == "rec":
            copies[name] = p
            continue
        target = snap / f"{name}{p.suffix}"
        shutil.copyfile(p, target)
        if digest(target) != hashes[name] or digest(p) != hashes[name]:
            raise RuntimeError("복사 중 입력이 바뀌었습니다. 완전히 저장된 뒤 다시 실행합니다.")
        copies[name] = target
    pred = fi.load_generation_predictions(copies["solar"], copies["wind"], copies["plants"])
    portfolio = fi.aggregate_portfolio(pred, fi.expected_assets(copies["plants"]))
    smp = fi.load_smp_forecast(copies["smp"])
    start, end = min(portfolio.planning_day), max(portfolio.planning_day)
    windows, excluded = fi.build_plan_windows(portfolio, smp, start, end)
    if not windows:
        raise ValueError("발전량·SMP가 겹치는 48시간 예측 창이 없습니다.")
    common = dict(
        solar_path=copies["solar"],
        wind_path=copies["wind"],
        smp_path=copies["smp"],
        plants_path=copies["plants"],
        ess_path=copies["ess"],
        rec_path=cfg.get("rec"),
    )
    start_energy = None
    if True:  # 실측이 없어도 예측 계획(롤링)은 만든다. 비교 시나리오는 실측이 있을 때만 붙는다.
        backtest = run_forecast_backtest(
            windows[0].planning_day,
            windows[-1].planning_day,
            **common,
            out_dir=out / "backtests",
            run_id=key,
            overwrite=True,
        )
        validation = read_json(backtest / "validation.json")
        if not validation.get("comparison_all_passed") or not validation.get(
            "plan_checks_all_passed"
        ):
            raise ValueError("새 운영 결과의 제약·비교 검산 실패. 이전 결과를 대체하지 않습니다.")
        import pandas as pd

        hourly = pd.read_csv(backtest / "hourly_forecast.csv")
        dates = pd.to_datetime(hourly.ts, utc=True).dt.tz_convert("Asia/Seoul").dt.date
        last_day = windows[-1].planning_day
        day_rows = hourly.loc[dates == last_day]
        if len(day_rows):
            row = day_rows.iloc[0]
            ess = EssParams.from_yaml(copies["ess"])
            start_energy = (
                row.energy_end_mwh
                - ess.charge_efficiency * row.charge_mwh
                + row.discharge_mwh / ess.discharge_efficiency
            )
    plan = run_operating_plan(
        windows[-1].planning_day,
        **common,
        start_energy_mwh=start_energy,
        out_dir=out / "plans",
        run_id=key,
        overwrite=True,
    )
    if not read_json(plan / "validation.json").get("all_passed"):
        raise ValueError("새 48시간 운영계획의 제약 검산이 실패했습니다.")
    for name, path in paths.items():
        if digest(path) != hashes[name]:
            raise RuntimeError("계산 중 입력이 바뀌어 새 결과의 게시를 보류했습니다.")
    if rec_metadata(load_rec(cfg.get("rec"))) != rec_info:
        raise RuntimeError("계산 중 REC 단가표가 바뀌었습니다.")
    result = {
        "id": key,
        "created_at": now(),
        "label": label,
        "inputs": {k: str(v) for k, v in copies.items()},
        "hashes": hashes,
        "source_paths": {k: str(v) for k, v in paths.items()},
        "forecast_columns": sorted(pred.forecast_source_column.unique()),
        "start": str(windows[0].planning_day),
        "end": str(windows[-1].planning_day),
        "days": len(windows),
        "excluded_days": len(excluded),
        "backtest": str(backtest) if backtest else None,
        "plan": str(plan),
        "realized_profit": None,
        # 실측이 없는 기간은 예측 계획만 만든다 (비교 시나리오 없음). 화면이 이 값으로 분기한다.
        "actual_available": bool(read_json(backtest / "summary.json").get("actual_available"))
        if backtest else False,
        "generation_availability_verified": False,
        "start_soc_note": "ESS 설정 시작 잔량",
    }
    result["start_soc_note"] = (
        "비교 계획의 시작 잔량 (실제 실행 SOC 아님)"
        if start_energy is not None
        else "ESS 설정 시작 잔량 (이전 실행 SOC 없음)"
    )
    result["output_hashes"] = {
        str(p): digest(p)
        for folder in (plan, backtest)
        if folder is not None
        for p in folder.iterdir()
        if p.suffix in (".csv", ".json")
    }
    atomic_json(done, result)
    return result


def run(root: Path = ROOT) -> dict:
    cfg = config(root)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    # A UI launcher publishes queued state before the child starts writing state.
    with (out / "launch.lock").open("a") as launch_lock:
        _lock_shared(launch_lock)
    with (out / "worker.lock").open("a") as lock:
        try:
            _lock_exclusive(lock, blocking=False)
        except BlockingIOError:
            return status(root)
        fingerprint = revision(root, cfg)
        old = status(root)
        state = {
            "revision": fingerprint,
            "pid": os.getpid(),
            "state": "running",
            "started_at": now(),
            "stages": [],
            "active": old.get("active"),
        }

        def save():
            atomic_json(out / "status.json", state)

        save()

        try:
            sources = {k: Path(cfg[k]) for k in ("solar", "wind", "smp")}
            label = "Drive 공유 예측 파일"
            if all(p.exists() for p in sources.values()):
                state["stages"].append({"name": "MILP 재계산·검산", "state": "running"})
                save()
                release = refresh_dispatch(root, cfg, sources, label=label)
                state["stages"][-1]["state"] = "complete"
                state["active"] = release
            else:
                state["stages"].append(
                    {
                        "name": "MILP 재계산·검산",
                        "state": "blocked",
                        "missing": [str(p) for p in sources.values() if not p.exists()],
                    }
                )
            state["state"] = ("complete" if state["stages"][-1]["state"] == "complete"
                              else "needs_data")
        except Exception as exc:
            state["state"] = "failed"
            state["error"] = str(exc)
            if state["stages"] and state["stages"][-1]["state"] == "running":
                state["stages"][-1]["state"] = "failed"
        state["finished_at"] = now()
        save()
        return state


def launch(root: Path = ROOT, *, force=False) -> dict:
    """One background worker; code/data changes create a new attempt automatically."""
    cfg = config(root)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "launch.lock").open("a") as lock:
        _lock_exclusive(lock, blocking=True)
        state = status(root)
        if state.get("state") in ("queued", "running"):
            try:
                os.kill(state["pid"], 0)
                return state
            except (OSError, KeyError):
                state.update(
                    state="failed",
                    error="이전 실행 프로세스가 종료됐습니다. 전체 실행으로 재시도하세요.",
                )
                atomic_json(out / "status.json", state)
        rev = revision(root, cfg)
        if state.get("revision") == rev and not force:
            return state
        args = [sys.executable, "-B", "-m", "vpp.operator_pipeline", "--root", str(root)]
        with (out / "worker.log").open("a") as log:
            proc = subprocess.Popen(
                args,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        queued = {**state, "state": "queued", "pid": proc.pid, "revision": rev, "stages": []}
        queued.pop("error", None)
        atomic_json(out / "status.json", queued)
        return queued


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
