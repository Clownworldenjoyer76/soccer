#!/usr/bin/env python3
# docs/win/soccer/scripts/05_final_scores/06_soccer_results_dashboard.py
#
# Builds the Soccer dashboard data (read live by the React Soccer page) from the reporting artifacts
# already produced by:
#   03_soccer_results_reports.py
#   05_soccer_model_reports.py
#
# Inputs:
#   docs/win/soccer/05_final_scores/{league}_market_tally.csv
#   docs/win/soccer/05_final_scores/reports/{league}/...
#   docs/win/soccer/05_final_scores/model_evaluation/SOCCER_model_metrics.csv
#   docs/win/soccer/05_final_scores/model_evaluation/SOCCER_model_calibration.csv
#   docs/win/soccer/05_final_scores/model_evaluation/SOCCER_xg_metrics.csv
#
# Output:
#   frontend/data/dashboards/soccer.json  ({"generated_at": <UTC ISO>, "data": {"all": ..., <league>: ...}})
#
# Log:
#   docs/win/soccer/05_final_scores/errors/06_soccer_results_dashboard.txt

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import json
import math
import os
import traceback

import pandas as pd


BASE = Path("docs/win/soccer/05_final_scores")
REPORT_DIR = BASE / "reports"
MODEL_DIR = BASE / "model_evaluation"
OUTPUT_FILE = Path("frontend/data/dashboards/soccer.json")
ERROR_DIR = BASE / "errors"
LOG_FILE = ERROR_DIR / "06_soccer_results_dashboard.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = [
    ("mls", "MLS"),
    ("epl", "EPL"),
    ("laliga", "La Liga"),
    ("ligue1", "Ligue 1"),
    ("seriea", "Serie A"),
    ("bundesliga", "Bundesliga"),
]

MARKETS = [
    {
        "key": "match_odds",
        "display": "Match Odds",
        "folder": "match_odds",
        "file_prefix": "match_odds",
        "side_suffix": "home_draw_away",
        "dimensions": ["ev", "kelly", "month", "odds", "model_prob", "win_prob"],
    },
    {
        "key": "btts",
        "display": "BTTS",
        "folder": "btts",
        "file_prefix": "btts",
        "side_suffix": "yes_no",
        "dimensions": ["ev", "kelly", "month", "odds", "model_prob"],
    },
    {
        "key": "total25",
        "display": "Total 2.5",
        "folder": "total_25",
        "file_prefix": "total_25",
        "side_suffix": "over_under",
        "dimensions": ["ev", "kelly", "month", "odds", "model_prob"],
    },
    {
        "key": "total35",
        "display": "Total 3.5",
        "folder": "total_35",
        "file_prefix": "total_35",
        "side_suffix": "over_under",
        "dimensions": ["ev", "kelly", "month", "odds", "model_prob"],
    },
]

MARKET_LABELS = {
    "match_odds": "Match Odds",
    "btts": "BTTS",
    "total25": "Total 2.5",
    "total35": "Total 3.5",
}

RUN_STARTED = datetime.now(UTC)
WARNING_COUNT = 0
ERROR_COUNT = 0
INPUT_FILE_COUNT = 0
INPUT_ROW_COUNT = 0
OUTPUT_FILE_COUNT = 0
OUTPUT_ROW_COUNT = 0
INPUT_FILES_SEEN: set[str] = set()

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== 06_soccer_results_dashboard ===\n")
    f.write(f"START_TIMESTAMP_UTC: {RUN_STARTED.isoformat()}\n")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def log(level: str, message: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"{_now()} | {level} | {message}\n")


def warn(message: str) -> None:
    global WARNING_COUNT
    WARNING_COUNT += 1
    log("WARNING", message)


def error(message: str) -> None:
    global ERROR_COUNT
    ERROR_COUNT += 1
    log("ERROR", message)


def log_input(path: Path, rows: int, exists: bool) -> None:
    global INPUT_FILE_COUNT, INPUT_ROW_COUNT
    key = str(path)
    if key in INPUT_FILES_SEEN:
        return
    INPUT_FILES_SEEN.add(key)
    INPUT_FILE_COUNT += 1
    INPUT_ROW_COUNT += rows
    log("INFO", f"INPUT | file={path} | exists={int(exists)} | rows={rows}")


def log_output(path: Path, rows: int, bytes_written: int) -> None:
    global OUTPUT_FILE_COUNT, OUTPUT_ROW_COUNT
    OUTPUT_FILE_COUNT += 1
    OUTPUT_ROW_COUNT += rows
    log("INFO", f"OUTPUT | file={path} | rows={rows} | bytes={bytes_written}")


def finish(status: str) -> None:
    ended = datetime.now(UTC)
    with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"INPUT_SUMMARY | files={INPUT_FILE_COUNT} | rows={INPUT_ROW_COUNT}\n")
        log_handle.write(f"OUTPUT_SUMMARY | files={OUTPUT_FILE_COUNT} | rows={OUTPUT_ROW_COUNT}\n")
        log_handle.write(f"WARNING_COUNT: {WARNING_COUNT}\n")
        log_handle.write(f"ERROR_COUNT: {ERROR_COUNT}\n")
        log_handle.write(f"END_TIMESTAMP_UTC: {ended.isoformat()}\n")
        log_handle.write(f"STATUS: {status}\n")


def safe_read(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        log_input(path, 0, False)
        if required:
            warn(f"Required dashboard input missing: {path}")
        else:
            log("INFO", f"Optional dashboard input missing: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        log_input(path, len(df), True)
        return df
    except Exception as exc:
        log_input(path, 0, True)
        warn(f"Unable to read dashboard input {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def json_safe_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def df_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    records = []
    for row in df.to_dict(orient="records"):
        records.append({k: json_safe_value(v) for k, v in row.items()})
    return records


def _tally_counts(
    frame: pd.DataFrame,
) -> tuple[int, int, int, int]:
    wins = (
        int(frame["Win"].sum())
        if "Win" in frame.columns
        else 0
    )
    losses = (
        int(frame["Loss"].sum())
        if "Loss" in frame.columns
        else 0
    )
    pushes = (
        int(frame["Push"].sum())
        if "Push" in frame.columns
        else 0
    )
    total = wins + losses + pushes

    return wins, losses, pushes, total


def aggregate_tally(
    tally: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    if tally.empty:
        return {}, []

    work = tally.copy()

    for col in [
        "Win",
        "Loss",
        "Push",
        "Total",
        "Sample_Count",
    ]:
        if col in work.columns:
            work[col] = (
                pd.to_numeric(
                    work[col],
                    errors="coerce",
                )
                .fillna(0)
            )

    (
        wins,
        losses,
        pushes,
        total,
    ) = _tally_counts(work)

    win_pct = (
        wins / (wins + losses)
        if (wins + losses)
        else None
    )

    headline = {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total": total,
        "win_pct": win_pct,
    }

    by_market = []

    if "market" in work.columns:
        for market, part in work.groupby(
            "market",
            dropna=False,
        ):
            (
                market_wins,
                market_losses,
                market_pushes,
                market_total,
            ) = _tally_counts(part)

            by_market.append(
                {
                    "market": str(market),
                    "market_display": MARKET_LABELS.get(
                        str(market),
                        str(market),
                    ),
                    "Win": market_wins,
                    "Loss": market_losses,
                    "Push": market_pushes,
                    "Total": market_total,
                    "Sample_Count": market_total,
                    "Win_Pct": (
                        market_wins
                        / (
                            market_wins
                            + market_losses
                        )
                        if (
                            market_wins
                            + market_losses
                        )
                        else None
                    ),
                }
            )

    return headline, by_market


def load_model_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = safe_read(MODEL_DIR / "SOCCER_model_metrics.csv")
    calibration = safe_read(MODEL_DIR / "SOCCER_model_calibration.csv")
    xg = safe_read(MODEL_DIR / "SOCCER_xg_metrics.csv")
    return metrics, calibration, xg


def _league_scope_mask(
    df: pd.DataFrame,
) -> pd.Series:
    if "scope" not in df.columns:
        return pd.Series(
            True,
            index=df.index,
        )

    return (
        df["scope"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("league")
    )


def filter_league_rows(
    df: pd.DataFrame,
    league: str,
) -> pd.DataFrame:
    if (
        df.empty
        or "league" not in df.columns
    ):
        return pd.DataFrame()

    scope_mask = _league_scope_mask(df)

    league_mask = (
        df["league"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(league)
    )

    return df.loc[
        scope_mask & league_mask
    ].copy()


def collect_league_data(
    league: str,
    display: str,
    model_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    xg_metrics: pd.DataFrame,
) -> dict:
    tally = safe_read(BASE / f"{league}_market_tally.csv", required=True)
    headline, by_market = aggregate_tally(tally)

    data = {
        "league": league,
        "display": display,
        "headline": headline,
        "tally": df_to_records(tally),
        "by_market": by_market,
        "model_metrics": df_to_records(filter_league_rows(model_metrics, league)),
        "calibration": df_to_records(filter_league_rows(calibration, league)),
        "xg_metrics": df_to_records(filter_league_rows(xg_metrics, league)),
        "markets": {},
    }

    for spec in MARKETS:
        market_dir = REPORT_DIR / league / spec["folder"]
        market_data = {
            "display": spec["display"],
            "by": {},
            "by_side": {},
        }

        for dim in spec["dimensions"]:
            base_name = f"{league}_{spec['file_prefix']}_by_{dim}"
            overall = market_dir / f"{base_name}.csv"
            by_side = market_dir / f"{base_name}_{spec['side_suffix']}_summary.csv"

            overall_df = safe_read(overall)
            side_df = safe_read(by_side)

            if not overall_df.empty or not side_df.empty:
                market_data["by"][dim] = df_to_records(overall_df)
                market_data["by_side"][dim] = df_to_records(side_df)

        data["markets"][spec["key"]] = market_data

    return data




def filter_all_league_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if (
        df.empty
        or "league" not in df.columns
    ):
        return pd.DataFrame()

    return df.loc[
        _league_scope_mask(df)
    ].copy()


def combine_result_rows(rows: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, dict] = {}

    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys)
        if key not in grouped:
            grouped[key] = {
                group_key: row.get(group_key)
                for group_key in group_keys
            }
            grouped[key].update(
                {
                    "Win": 0,
                    "Loss": 0,
                    "Push": 0,
                    "Total": 0,
                    "Sample_Count": 0,
                    "Win_Pct": None,
                }
            )

        target = grouped[key]
        for col in ["Win", "Loss", "Push"]:
            try:
                target[col] += int(float(row.get(col) or 0))
            except (TypeError, ValueError):
                pass

    combined = []
    for target in grouped.values():
        wins = target["Win"]
        losses = target["Loss"]
        pushes = target["Push"]
        total = wins + losses + pushes
        target["Total"] = total
        target["Sample_Count"] = total
        target["Win_Pct"] = wins / (wins + losses) if (wins + losses) else None
        combined.append(target)

    return combined


def collect_all_data(
    league_payloads: dict[str, dict],
    model_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    xg_metrics: pd.DataFrame,
) -> dict:
    tally = safe_read(BASE / "all_soccer_market_tally.csv", required=True)
    headline, by_market = aggregate_tally(tally)

    data = {
        "league": "all",
        "display": "All",
        "is_all": True,
        "headline": headline,
        "tally": df_to_records(tally),
        "by_market": by_market,
        "model_metrics": df_to_records(filter_all_league_rows(model_metrics)),
        "calibration": df_to_records(filter_all_league_rows(calibration)),
        "xg_metrics": df_to_records(filter_all_league_rows(xg_metrics)),
        "markets": {},
    }

    for spec in MARKETS:
        market_data = {
            "display": spec["display"],
            "by": {},
            "by_side": {},
        }

        for dim in spec["dimensions"]:
            overall_rows = []
            side_rows = []

            for payload in league_payloads.values():
                source_market = (payload.get("markets") or {}).get(spec["key"], {})
                overall_rows.extend((source_market.get("by") or {}).get(dim, []))
                side_rows.extend((source_market.get("by_side") or {}).get(dim, []))

            combined_overall = combine_result_rows(overall_rows, ["bucket"])
            combined_side = combine_result_rows(side_rows, ["bucket", "side"])

            if combined_overall or combined_side:
                market_data["by"][dim] = combined_overall
                market_data["by_side"][dim] = combined_side

        data["markets"][spec["key"]] = market_data

    return data


def build_payload() -> dict:
    metrics, calibration, xg = load_model_tables()

    league_payloads = {
        league: collect_league_data(
            league,
            display,
            metrics,
            calibration,
            xg,
        )
        for league, display in LEAGUES
    }

    return {
        "all": collect_all_data(
            league_payloads,
            metrics,
            calibration,
            xg,
        ),
        **league_payloads,
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data) -> int:
    document = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": json_safe(data),
    }
    content = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(content, encoding="utf-8")
    os.replace(staging, path)
    return len(content.encode("utf-8"))


def run() -> None:
    bytes_written = write_json(OUTPUT_FILE, build_payload())
    log_output(OUTPUT_FILE, 1, bytes_written)
    log("INFO", f"dashboard data -> {OUTPUT_FILE}")


def main() -> None:
    status = "FAILED"
    try:
        run()
        status = "SUCCESS"
    except Exception as exc:
        error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        trace = traceback.format_exc()
        with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
            log_handle.write(trace)
            if not trace.endswith("\n"):
                log_handle.write("\n")
        raise
    finally:
        finish(status)


if __name__ == "__main__":
    main()

