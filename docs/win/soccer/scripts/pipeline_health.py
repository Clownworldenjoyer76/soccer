#!/usr/bin/env python3
"""Soccer pipeline health reporter.

Writes:
    docs/win/soccer/pipeline_health.json
    docs/win/soccer/errors/pipeline_health.txt
    frontend/data/pipeline_health/soccer.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path("docs/win/soccer")
OUTPUT = BASE / "pipeline_health.json"
LOG = BASE / "errors/pipeline_health.txt"
FRONTEND_OUTPUT = Path("frontend/data/pipeline_health/soccer.json")
NY = ZoneInfo("America/New_York")
SPORT_KEY = "soccer"
SPORT_LABEL = "Soccer"
JOB_STATUS_ENV = "SOCCER_PIPELINE_JOB_STATUS"
RUN_DATE_ENV = "SOCCER_PIPELINE_RUN_DATE"

BAD_MARKERS = (
    "STATUS: FAILED",
    "STATUS: PARTIAL",
    "STATUS: COMPLETED WITH ERRORS",
    "FATAL ERROR",
    "TRACEBACK (MOST RECENT CALL LAST)",
)

STAGES = [
    ("log", "ESPN Endpoint Diagnostic", "errors/00_intake/espn_endpoint_diagnostic.txt"),
    ("log", "Name Normalization", "errors/00_intake/name_normalization.txt"),
    ("log", "Soccer Cleaner", "errors/00_intake/soccer_cleaner.txt"),
    ("log", "EPL ML Inference", "errors/01_merge/epl_ml_infer.txt"),
    ("log", "Bundesliga ML Inference", "errors/01_merge/bundesliga_ml_infer.txt"),
    ("log", "La Liga ML Inference", "errors/01_merge/laliga_ml_infer.txt"),
    ("log", "Serie A ML Inference", "errors/01_merge/seriea_ml_infer.txt"),
    ("log", "Ligue 1 ML Inference", "errors/01_merge/ligue1_ml_infer.txt"),
    ("log", "MLS ML Inference", "errors/01_merge/mls_ml_infer.txt"),
    ("output", "Games Master", "00_intake/games/{date}_soccer_games.csv"),
    ("output", "Normalized Predictions", "00_intake/predictions/normalized/{date}_*.csv"),
    ("output", "Normalized Sportsbook", "00_intake/sportsbook/normalized/{date}_*.csv"),
    ("output", "Merge", "01_merge/**/*{date}*.csv"),
    ("output", "Juice", "02_juice/**/*{date}*.csv"),
    ("output", "Edges", "03_edges/**/*{date}*.csv"),
    ("output", "Select", "04_select/*{date}*.csv"),
    ("output", "Locked Select", "04_select/locked/*{date}*.csv"),
]


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def clean_id(value) -> str:
    text = clean(value)
    if not text:
        return ""
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def count_rows(paths: list[Path]) -> int:
    return sum(len(read_rows(path)) for path in paths)


def unique_game_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            gid = clean_id(row.get("game_id") or row.get("id") or row.get("event_id"))
            if gid:
                ids.add(gid)
    return ids


def resolve_date(now_ny: datetime) -> str:
    raw = clean(os.getenv(RUN_DATE_ENV))
    if raw:
        value = raw.replace("-", "_")
        if re.fullmatch(r"\d{4}_\d{2}_\d{2}", value):
            return value
    return now_ny.strftime("%Y_%m_%d")


def last_status(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    statuses = [line.strip() for line in text.splitlines() if "STATUS:" in line]
    if statuses:
        return statuses[-1]
    upper = text.upper()
    if any(marker in upper for marker in BAD_MARKERS):
        return "STATUS: FAILED"
    return "STATUS: LOG PRESENT"


def stage_status(date: str) -> list[dict]:
    rows: list[dict] = []
    for stage in STAGES:
        kind = stage[0]
        name = stage[1]
        spec = stage[2]
        if kind == "log":
            path = BASE / spec
            rows.append({
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "status": last_status(path) or "STATUS: NO LOG",
            })
        else:
            pattern = spec.format(date=date)
            matches = sorted(BASE.glob(pattern))
            rows.append({
                "name": name,
                "path": str(BASE / pattern),
                "exists": bool(matches),
                "status": "STATUS: OUTPUT PRESENT" if matches else "STATUS: NO OUTPUT",
                "matched_files": [str(p) for p in matches[:25]],
            })
    return rows


def build_counts(date: str) -> tuple[dict, dict, list[str], list[str]]:
    games = [BASE / f"00_intake/games/{date}_soccer_games.csv"]
    predictions = sorted((BASE / "00_intake/predictions/normalized").glob(f"{date}_*.csv"))
    sportsbook = sorted((BASE / "00_intake/sportsbook/normalized").glob(f"{date}_*.csv"))
    merged = sorted((BASE / "01_merge").glob(f"**/*{date}*.csv"))
    edges = sorted((BASE / "03_edges").glob(f"**/*{date}*.csv"))
    selected = sorted(p for p in (BASE / "04_select").glob(f"*{date}*.csv") if p.is_file())
    locked = sorted((BASE / "04_select/locked").glob(f"*{date}*.csv"))

    game_ids = unique_game_ids(games)
    pred_ids = unique_game_ids(predictions)
    book_ids = unique_game_ids(sportsbook)
    merged_ids = unique_game_ids(merged)

    warnings: list[str] = []
    fatals: list[str] = []
    if game_ids and pred_ids and game_ids - pred_ids:
        warnings.append(f"Soccer: {len(game_ids - pred_ids)} game(s) missing normalized predictions")
    if game_ids and book_ids and game_ids - book_ids:
        warnings.append(f"Soccer: {len(game_ids - book_ids)} game(s) missing normalized sportsbook data")
    if book_ids and merged_ids and book_ids - merged_ids:
        warnings.append(f"Soccer: {len(book_ids - merged_ids)} sportsbook game(s) did not reach merge outputs")

    counts = {
        "scheduled_games": len(game_ids) if game_ids else count_rows(games),
        "prediction_games": len(pred_ids) if pred_ids else count_rows(predictions),
        "sportsbook_games": len(book_ids) if book_ids else count_rows(sportsbook),
        "merged_games": len(merged_ids) if merged_ids else count_rows(merged),
        "edge_rows": count_rows(edges),
        "selected_bets": count_rows(selected),
        "locked_bets": count_rows(locked),
    }
    paths = {
        "games": [str(p) for p in games],
        "predictions": [str(p) for p in predictions],
        "sportsbook": [str(p) for p in sportsbook],
        "merged": [str(p) for p in merged],
        "edges": [str(p) for p in edges],
        "selected": [str(p) for p in selected],
        "locked": [str(p) for p in locked],
    }
    return counts, paths, warnings, fatals


def write_log(payload: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"=== {SPORT_LABEL} PIPELINE HEALTH {payload['generated_at_utc']} ===",
        f"status={payload['status']}",
        f"workflow_status={payload['workflow']['status']}",
        f"report_date={payload['report_date']}",
        f"fatal_errors={len(payload.get('fatal_errors', []))}",
        f"warnings={len(payload.get('warnings', []))}",
    ]
    for item in payload.get("fatal_errors", []):
        lines.append(f"FATAL: {item}")
    for item in payload.get("warnings", []):
        lines.append(f"WARNING: {item}")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    now_utc = datetime.now(UTC)
    now_ny = datetime.now(NY)
    date = resolve_date(now_ny)
    job_status = clean(os.getenv(JOB_STATUS_ENV)) or "unknown"

    counts, paths, warnings, fatals = build_counts(date)
    stages = stage_status(date)

    if job_status.lower() != "success":
        fatals.append(f"{SPORT_LABEL} workflow status is {job_status.upper()}")

    failed_logs = [
        row for row in stages
        if any(marker in clean(row.get("status")).upper() for marker in BAD_MARKERS)
    ]
    if failed_logs and job_status.lower() == "success":
        fatals.append(
            f"{SPORT_LABEL} has failed stage log(s): "
            + "; ".join(clean(row.get("name")) for row in failed_logs)
        )

    status = "failed" if fatals else ("warning" if warnings else "healthy")
    payload = {
        "schema_version": 1,
        "sport": SPORT_KEY,
        "generated_at_utc": now_utc.isoformat(),
        "game_date_new_york": now_ny.strftime("%Y_%m_%d"),
        "report_date": date,
        "status": status,
        "fatal_errors": fatals,
        "warnings": warnings,
        "counts": counts,
        "stage_status": stages,
        "paths": paths,
        "workflow": {
            "name": "Soccer Pipeline",
            "status": job_status.lower(),
            "run_id": clean(os.getenv("GITHUB_RUN_ID")),
            "run_attempt": clean(os.getenv("GITHUB_RUN_ATTEMPT")),
            "sha": clean(os.getenv("GITHUB_SHA")),
            "ref_name": clean(os.getenv("GITHUB_REF_NAME")),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    FRONTEND_OUTPUT.write_text(text, encoding="utf-8")
    write_log(payload)

    print(f"{SPORT_LABEL} pipeline health: {status}")
    print(f"output: {OUTPUT}")
    print(f"frontend: {FRONTEND_OUTPUT}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for fatal in fatals:
        print(f"FATAL: {fatal}")
    return 1 if fatals else 0


if __name__ == "__main__":
    raise SystemExit(main())
