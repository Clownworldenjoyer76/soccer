#!/usr/bin/env python3
# docs/win/soccer/scripts/04_select/soccer_select_bets.py
#
# Reads Stage-3 EV/Kelly outputs and applies per-league x per-market x per-side
# filters from markets.yaml.
#
# League/market ml_filter blocks are applied before side filters. For EPL these
# consume the trained predictability and skip-probability outputs. Missing or
# invalid configured ML filter inputs are hard errors; they are never silently
# ignored.
#
# Historical rebuildable selections remain restricted to the fixed tuning
# window declared in markets.yaml. Current evaluation-day selections are
# additionally written once to 04_select/locked and locked files remain
# immutable.

import hashlib
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE / "03_edges"
OUTPUT_DIR = BASE / "04_select"
LOCKED_DIR = OUTPUT_DIR / "locked"
CONFIG_PATH = BASE / "config" / "markets.yaml"
ERROR_DIR = BASE / "errors" / "04_select"
LOG_FILE = ERROR_DIR / "select_bets.txt"

OUTPUT_DIR.mkdir(exist_ok=True)
LOCKED_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

MARKET_FROM_SUFFIX = {
    "_match_odds": "match_odds",
    "_btts": "btts",
    "_total_25": "total25",
    "_total_35": "total35",
}
LEAGUES = ["bundesliga", "seriea", "laliga", "ligue1", "epl", "mls"]
MATCH_ODDS_SOURCE_KEYS = (
    "ev",
    "kelly",
    "fair_odds",
    "edge",
    "selection_filter",
)
SUPPORTED_MATCH_ODDS_SOURCES = {"raw", "adjusted", "engine"}

LOCK_BASE_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
    "market",
    "side",
    "odds",
    "american_odds",
    "ev",
    "kelly",
    "model_prob",
    "edge",
    "model_prob_source",
    "model_prob_underlying_source",
    "ev_prob",
    "ev_prob_source",
    "kelly_prob",
    "kelly_prob_source",
    "fair_odds_prob",
    "fair_odds_prob_source",
    "fair_odds",
    "edge_prob",
    "edge_prob_source",
    "edge_fair_odds",
    "ml_predictability",
    "ml_predictability_source",
    "ml_skip_prob",
    "ml_skip_prob_source",
]

LOCK_METADATA_COLUMNS = [
    "selection_config_sha256",
    "selection_period",
    "tuning_start",
    "tuning_end",
    "evaluation_start",
    "locked_run_date",
    "locked_at_utc",
]

DEBUG_COUNTS: dict = defaultdict(int)


def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary, per_market, per_date, per_league):
    lines = [
        "",
        "=" * 70,
        f"SUMMARY  {_now()}",
        "=" * 70,
        f"  files_processed : {summary['files_processed']}",
        f"  total_bets      : {summary['total_bets']}",
        f"  dates_written   : {summary['dates_written']}",
        f"  skipped_files   : {summary['skipped']}",
        f"  locked_created  : {summary['locked_created']}",
        f"  locked_preserved: {summary['locked_preserved']}",
        f"  errors          : {summary['errors']}",
        "",
        "--- By Market ---",
        f"  {'market':<15} {'bets':>6}",
    ]
    for m, c in sorted(per_market.items()):
        lines.append(f"  {m:<15} {c:>6}")
    lines += ["", "--- By League ---", f"  {'league':<15} {'bets':>6}"]
    for lg, c in sorted(per_league.items()):
        lines.append(f"  {lg:<15} {c:>6}")
    lines += ["", "--- By Date ---", f"  {'date':<14} {'bets':>6} {'file'}"]
    for date, info in sorted(per_date.items()):
        lines.append(f"  {date:<14} {info['bets']:>6}  {info['file']}")
    lines += ["", "--- Filter Reject Counts ---"]
    for k, v in sorted(DEBUG_COUNTS.items()):
        lines.append(f"  {k:<40} : {v}")
    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 70]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _normalize_policy_date(value, label: str) -> str:
    normalized = str(value or "").strip().replace("-", "_")
    try:
        datetime.strptime(normalized, "%Y_%m_%d")
    except ValueError as e:
        raise ValueError(
            f"backtest_policy.soccer.{label} must be YYYY-MM-DD or YYYY_MM_DD"
        ) from e
    return normalized


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = data["markets"]["soccer"]
        if not isinstance(config, dict):
            raise ValueError("markets.soccer must be a mapping")

        try:
            source_config = data["probability_sources"]["soccer"]["match_odds"]
        except (TypeError, KeyError) as e:
            raise ValueError(
                "markets.yaml missing probability_sources.soccer.match_odds"
            ) from e
        if not isinstance(source_config, dict):
            raise ValueError("probability_sources.soccer.match_odds must be a mapping")
        missing = [key for key in MATCH_ODDS_SOURCE_KEYS if key not in source_config]
        if missing:
            raise ValueError(
                "probability_sources.soccer.match_odds missing keys: "
                f"{missing}"
            )
        normalized_sources = {}
        for key in MATCH_ODDS_SOURCE_KEYS:
            source = str(source_config[key]).strip().lower()
            if source not in SUPPORTED_MATCH_ODDS_SOURCES:
                raise ValueError(
                    f"Unsupported 1X2 probability source for {key}: {source!r}. "
                    f"Supported: {sorted(SUPPORTED_MATCH_ODDS_SOURCES)}"
                )
            normalized_sources[key] = source

        try:
            raw_policy = data["backtest_policy"]["soccer"]
        except (TypeError, KeyError) as e:
            raise ValueError("markets.yaml missing backtest_policy.soccer") from e
        if not isinstance(raw_policy, dict):
            raise ValueError("backtest_policy.soccer must be a mapping")
        policy = {
            "tuning_start": _normalize_policy_date(raw_policy.get("tuning_start"), "tuning_start"),
            "tuning_end": _normalize_policy_date(raw_policy.get("tuning_end"), "tuning_end"),
            "evaluation_start": _normalize_policy_date(raw_policy.get("evaluation_start"), "evaluation_start"),
        }
        tuning_start = datetime.strptime(policy["tuning_start"], "%Y_%m_%d")
        tuning_end = datetime.strptime(policy["tuning_end"], "%Y_%m_%d")
        evaluation_start = datetime.strptime(policy["evaluation_start"], "%Y_%m_%d")
        if tuning_start > tuning_end:
            raise ValueError("backtest tuning_start must be <= tuning_end")
        if evaluation_start <= tuning_end:
            raise ValueError("backtest evaluation_start must be later than tuning_end")
        return config, normalized_sources, policy
    except Exception as e:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== soccer select_bets RUN {_now()} ===\n")
        _log(f"CONFIG LOAD FAILED | {CONFIG_PATH} | {e}\n{traceback.format_exc()}", "ERROR")
        raise


CONFIG, MATCH_ODDS_PROBABILITY_SOURCES, BACKTEST_POLICY = load_config()


def fv(x):
    try:
        if x is None or pd.isna(x):
            return None
        v = float(x)
        return v if pd.notna(v) else None
    except Exception:
        return None


def decimal_to_american(decimal_odds):
    d = fv(decimal_odds)
    if d is None or d <= 1:
        return None
    if d >= 2:
        return round((d - 1) * 100)
    return round(-100 / (d - 1))


def probability_column(side: str, source: str) -> str:
    if source == "raw":
        return f"{side}_prob"
    if source == "adjusted":
        return f"juiced_{side}_prob"
    if source == "engine":
        return f"engine_{side}_prob"
    raise ValueError(f"Unsupported probability source: {source!r}")


def normalize_bands(raw_bands, label=""):
    if raw_bands is None or raw_bands == []:
        return []
    if not isinstance(raw_bands, list):
        raise ValueError(f"{label} must be a list of [lo, hi] bands")
    bands = []
    for i, band in enumerate(raw_bands):
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError(f"{label}[{i}] must be [lo, hi], got {band!r}")
        lo = fv(band[0])
        hi = fv(band[1])
        if lo is None or hi is None:
            raise ValueError(f"{label}[{i}] has non-numeric bounds: {band!r}")
        if lo > hi:
            raise ValueError(f"{label}[{i}] lower bound > upper bound: {band!r}")
        bands.append((lo, hi))
    return bands


def in_any_band(value, bands):
    v = fv(value)
    return v is not None and bool(bands) and any(lo <= v <= hi for lo, hi in bands)


def parse_date(s):
    try:
        return datetime.strptime(str(s).strip().replace("-", "_"), "%Y_%m_%d")
    except Exception:
        return None


def selection_period(game_date: str) -> str:
    dt = parse_date(game_date)
    if dt is None:
        return "invalid"
    tuning_start = parse_date(BACKTEST_POLICY["tuning_start"])
    tuning_end = parse_date(BACKTEST_POLICY["tuning_end"])
    evaluation_start = parse_date(BACKTEST_POLICY["evaluation_start"])
    if tuning_start <= dt <= tuning_end:
        return "tuning"
    if dt >= evaluation_start:
        return "evaluation"
    if dt < tuning_start:
        return "pre_tuning"
    return "gap"


def historical_rebuild_allowed(game_date: str, active_lock_date: str) -> bool:
    period = selection_period(game_date)
    if period == "tuning":
        return True
    if period == "evaluation" and game_date == active_lock_date:
        return True
    return False


def date_ok(game_date, months, exclude_dow):
    if not months and not exclude_dow:
        return True
    dt = parse_date(game_date) if isinstance(game_date, str) else None
    if dt is None:
        return True
    if months and dt.month not in months:
        DEBUG_COUNTS["fail_month"] += 1
        return False
    if exclude_dow and dt.weekday() in exclude_dow:
        DEBUG_COUNTS["fail_dow"] += 1
        return False
    return True


def passes_filters(values: dict, scfg: dict, game_date: str) -> bool:
    filter_map = [
        ("odds_bands", "odds", "fail_odds"),
        ("american_odds_bands", "american_odds", "fail_american_odds"),
        ("ev_bands", "ev", "fail_ev"),
        ("kelly_bands", "kelly", "fail_kelly"),
        ("model_prob_bands", "model_prob", "fail_model_prob"),
        ("edge_bands", "edge", "fail_edge"),
    ]
    for band_key, value_key, fail_key in filter_map:
        bands = normalize_bands(scfg.get(band_key), band_key)
        if not bands:
            continue
        if not in_any_band(values.get(value_key), bands):
            DEBUG_COUNTS[fail_key] += 1
            return False
    return date_ok(
        game_date,
        scfg.get("months", []) or [],
        scfg.get("exclude_days_of_week", []) or [],
    )


def validate_ml_filter_config(ml_cfg: dict, league: str, market: str):
    required = (
        "predictability_column",
        "min_predictability",
        "skip_prob_column",
        "reject_skip_at_or_above",
    )
    missing = [key for key in required if key not in ml_cfg]
    if missing:
        raise ValueError(
            f"ML filter config missing keys for league={league} market={market}: {missing}"
        )
    min_predictability = fv(ml_cfg["min_predictability"])
    reject_skip = fv(ml_cfg["reject_skip_at_or_above"])
    if min_predictability is None or not 0.0 <= min_predictability <= 1.0:
        raise ValueError(
            f"Invalid min_predictability for league={league} market={market}"
        )
    if reject_skip is None or not 0.0 <= reject_skip <= 1.0:
        raise ValueError(
            f"Invalid reject_skip_at_or_above for league={league} market={market}"
        )
    return min_predictability, reject_skip


def passes_ml_filter(row, league: str, market: str, cfg: dict):
    ml_cfg = cfg.get("ml_filter")
    if not ml_cfg or not ml_cfg.get("enabled", False):
        return True, {
            "ml_predictability": None,
            "ml_predictability_source": None,
            "ml_skip_prob": None,
            "ml_skip_prob_source": None,
        }
    if not isinstance(ml_cfg, dict):
        raise ValueError(f"ml_filter must be a mapping: league={league} market={market}")

    min_predictability, reject_skip = validate_ml_filter_config(ml_cfg, league, market)
    predictability_column = str(ml_cfg["predictability_column"]).strip()
    skip_column = str(ml_cfg["skip_prob_column"]).strip()
    predictability = fv(row.get(predictability_column))
    skip_prob = fv(row.get(skip_column))
    if predictability is None or not 0.0 <= predictability <= 1.0:
        raise ValueError(
            f"Missing/invalid ML predictability | league={league} market={market} "
            f"column={predictability_column} value={row.get(predictability_column)!r}"
        )
    if skip_prob is None or not 0.0 <= skip_prob <= 1.0:
        raise ValueError(
            f"Missing/invalid ML skip probability | league={league} market={market} "
            f"column={skip_column} value={row.get(skip_column)!r}"
        )

    metadata = {
        "ml_predictability": predictability,
        "ml_predictability_source": predictability_column,
        "ml_skip_prob": skip_prob,
        "ml_skip_prob_source": skip_column,
    }

    if predictability < min_predictability:
        DEBUG_COUNTS[f"fail_ml_predictability_{league}_{market}"] += 1
        return False, metadata
    if skip_prob >= reject_skip:
        DEBUG_COUNTS[f"fail_ml_skip_{league}_{market}"] += 1
        return False, metadata
    return True, metadata


def pick(qualifying, preference):
    if not qualifying:
        return None
    metric = preference.get("metric", "ev")
    direction = preference.get("direction", "max")
    def key(c):
        v = c.get(metric)
        if v is None:
            return float("-inf") if direction == "max" else float("inf")
        return v
    return max(qualifying, key=key) if direction == "max" else min(qualifying, key=key)


def market_cfg(league, market_type):
    try:
        return CONFIG[league.lower()][market_type]
    except KeyError as e:
        raise KeyError(f"No config: league={league!r} market_type={market_type!r}") from e


def make_values(odds, ev, kelly, model_prob, edge):
    return {
        "odds": odds,
        "american_odds": decimal_to_american(odds),
        "ev": ev,
        "kelly": kelly,
        "model_prob": model_prob,
        "edge": edge,
    }


def make_side(side, odds, ev, kelly, model_prob, edge, **extra):
    return {
        "side": side,
        "odds": odds,
        "american_odds": decimal_to_american(odds),
        "ev": ev,
        "kelly": kelly,
        "model_prob": model_prob,
        "edge": edge,
        **extra,
    }


def validate_match_odds_provenance(row, side: str) -> None:
    checks = {
        "ev": f"{side}_ev_prob_source",
        "kelly": f"{side}_kelly_prob_source",
        "fair_odds": f"{side}_fair_odds_prob_source",
        "edge": f"{side}_edge_prob_source",
        "selection_filter": f"{side}_selection_prob_source",
    }
    for metric, source_field in checks.items():
        expected = probability_column(side, MATCH_ODDS_PROBABILITY_SOURCES[metric])
        actual = row.get(source_field)
        if actual is None or pd.isna(actual):
            raise ValueError(
                f"Missing 1X2 provenance field {source_field!r}; rerun build_edges.py before selection"
            )
        actual = str(actual).strip()
        if actual != expected:
            raise ValueError(
                f"Stale/inconsistent 1X2 provenance for side={side} metric={metric}: "
                f"markets.yaml expects {expected!r}, stage-3 row says {actual!r}. "
                "Rerun build_edges.py before selection."
            )


def clear_old_outputs() -> None:
    deleted = 0
    for old_file in sorted(OUTPUT_DIR.glob("*_soccer_bets.csv")):
        old_file.unlink()
        deleted += 1
        _log(f"DELETED OLD SELECT FILE: {old_file}")
    DEBUG_COUNTS["deleted_old_select_files"] += deleted
    _log(f"Old select files deleted: {deleted}")
    _log(f"Locked directory preserved: {LOCKED_DIR}")


def current_ny_date() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y_%m_%d")


def resolve_lock_date() -> str:
    raw = os.environ.get("RUN_DATE", "").strip()
    if raw:
        normalized = raw.replace("-", "_")
        try:
            datetime.strptime(normalized, "%Y_%m_%d")
        except ValueError as e:
            raise ValueError(
                f"Invalid RUN_DATE {raw!r}; expected YYYY-MM-DD or YYYY_MM_DD"
            ) from e
        return normalized
    return current_ny_date()


def config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def lock_daily_picks(df_all, seen_input_dates, summary):
    lock_date = resolve_lock_date()
    locked_path = LOCKED_DIR / f"{lock_date}_soccer_bets.csv"
    _log(f"LOCK DATE: {lock_date}")
    today = current_ny_date()
    if lock_date != today:
        _log(
            f"LOCK SKIPPED | run date {lock_date} is not current New York date {today}; "
            "historical backfill is not allowed",
            "WARN",
        )
        DEBUG_COUNTS["lock_skipped_historical_run_date"] += 1
        return
    if selection_period(lock_date) != "evaluation":
        _log(
            f"LOCK SKIPPED | run date {lock_date} is not inside evaluation period "
            f"starting {BACKTEST_POLICY['evaluation_start']}",
            "WARN",
        )
        DEBUG_COUNTS["lock_skipped_not_evaluation_period"] += 1
        return
    if lock_date not in seen_input_dates:
        _log(f"LOCK SKIPPED | no stage-3 input files found for run date {lock_date}", "WARN")
        DEBUG_COUNTS["lock_skipped_no_run_date_input"] += 1
        return
    if locked_path.exists():
        summary["locked_preserved"] += 1
        DEBUG_COUNTS["locked_existing_preserved"] += 1
        _log(f"LOCK PRESERVED | existing immutable file not overwritten: {locked_path}")
        return

    if df_all.empty or "match_date" not in df_all.columns:
        locked = pd.DataFrame(columns=LOCK_BASE_COLUMNS)
    else:
        match_dates = df_all["match_date"].astype(str).str.strip()
        locked = df_all.loc[match_dates == lock_date].copy()
    for col in LOCK_BASE_COLUMNS:
        if col not in locked.columns:
            locked[col] = pd.NA

    sha = config_sha256()
    locked["selection_config_sha256"] = sha
    locked["selection_period"] = "evaluation"
    locked["tuning_start"] = BACKTEST_POLICY["tuning_start"]
    locked["tuning_end"] = BACKTEST_POLICY["tuning_end"]
    locked["evaluation_start"] = BACKTEST_POLICY["evaluation_start"]
    locked["locked_run_date"] = lock_date
    locked["locked_at_utc"] = _now()
    ordered = LOCK_BASE_COLUMNS + LOCK_METADATA_COLUMNS
    extra = [c for c in locked.columns if c not in ordered]
    locked = locked[ordered + extra]
    locked.to_csv(locked_path, index=False)

    summary["locked_created"] += 1
    DEBUG_COUNTS["locked_files_created"] += 1
    DEBUG_COUNTS["locked_rows_created"] += len(locked)
    _log(
        f"LOCK CREATED | {locked_path} | rows={len(locked)} | markets_sha256={sha} | "
        f"tuning={BACKTEST_POLICY['tuning_start']}..{BACKTEST_POLICY['tuning_end']} | "
        f"evaluation_start={BACKTEST_POLICY['evaluation_start']}"
    )


def parse_filename(name: str):
    stem = name[:-4] if name.endswith(".csv") else name
    market_type = None
    league_part = None
    for suffix, mt in MARKET_FROM_SUFFIX.items():
        if stem.endswith(suffix):
            market_type = mt
            league_part = stem[:-len(suffix)]
            break
    if market_type is None:
        return None, None, None
    m = re.match(r"^(\d{4}_\d{2}_\d{2})_(.+)$", league_part)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), market_type


def build_match_odds_sides(row, game_date, cfg):
    sides = []
    for side in ("home", "draw", "away"):
        scfg = cfg.get(side)
        if not scfg or not scfg.get("enabled", True):
            continue
        validate_match_odds_provenance(row, side)
        odds = fv(row.get(f"dk_{side}_decimal"))
        ev = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        model_prob = fv(row.get(f"{side}_selection_prob"))
        edge = fv(row.get(f"{side}_edge"))
        values = make_values(odds, ev, kelly, model_prob, edge)
        if passes_filters(values, scfg, game_date):
            sides.append(
                make_side(
                    side,
                    odds,
                    ev,
                    kelly,
                    model_prob,
                    edge,
                    model_prob_source=str(row.get(f"{side}_selection_prob_source")).strip(),
                    model_prob_underlying_source=str(
                        row.get(
                            f"{side}_selection_prob_underlying_source",
                            row.get(f"engine_{side}_prob_source", f"engine_{side}_prob"),
                        )
                    ).strip(),
                    ev_prob=fv(row.get(f"{side}_ev_prob")),
                    ev_prob_source=str(row.get(f"{side}_ev_prob_source")).strip(),
                    kelly_prob=fv(row.get(f"{side}_kelly_prob")),
                    kelly_prob_source=str(row.get(f"{side}_kelly_prob_source")).strip(),
                    fair_odds_prob=fv(row.get(f"{side}_fair_odds_prob")),
                    fair_odds_prob_source=str(row.get(f"{side}_fair_odds_prob_source")).strip(),
                    fair_odds=fv(row.get(f"{side}_fair_decimal")),
                    edge_prob=fv(row.get(f"{side}_edge_prob")),
                    edge_prob_source=str(row.get(f"{side}_edge_prob_source")).strip(),
                    edge_fair_odds=fv(row.get(f"{side}_edge_fair_decimal")),
                )
            )
        else:
            DEBUG_COUNTS["rejected_match_odds"] += 1
    return sides


def build_btts_sides(row, game_date, cfg):
    sides = []
    for side in ("yes", "no"):
        scfg = cfg.get(side)
        if not scfg or not scfg.get("enabled", True):
            continue
        odds = fv(row.get(f"btts_{side}"))
        ev = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        mp = fv(row.get(f"engine_btts_{side}_prob"))
        edge = fv(row.get(f"{side}_edge"))
        values = make_values(odds, ev, kelly, mp, edge)
        if passes_filters(values, scfg, game_date):
            sides.append(
                make_side(
                    side,
                    odds,
                    ev,
                    kelly,
                    mp,
                    edge,
                    model_prob_source=str(
                        row.get(f"{side}_model_prob_source", f"engine_btts_{side}_prob")
                    ).strip(),
                    model_prob_underlying_source=str(
                        row.get(f"engine_btts_{side}_prob_source", f"engine_btts_{side}_prob")
                    ).strip(),
                )
            )
        else:
            DEBUG_COUNTS["rejected_btts"] += 1
    return sides


def build_totals_sides(row, game_date, cfg, line_tag):
    sides = []
    for side in ("over", "under"):
        scfg = cfg.get(side)
        if not scfg or not scfg.get("enabled", True):
            continue
        odds = fv(row.get(f"dk_{side}{line_tag}_decimal"))
        ev = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        mp = fv(row.get(f"engine_{side}_prob"))
        edge = fv(row.get(f"{side}_edge"))
        values = make_values(odds, ev, kelly, mp, edge)
        if passes_filters(values, scfg, game_date):
            sides.append(
                make_side(
                    side,
                    odds,
                    ev,
                    kelly,
                    mp,
                    edge,
                    model_prob_source=str(
                        row.get(f"{side}_model_prob_source", f"engine_{side}_prob")
                    ).strip(),
                    model_prob_underlying_source=str(
                        row.get(f"engine_{side}_prob_source", f"engine_{side}_prob")
                    ).strip(),
                )
            )
        else:
            DEBUG_COUNTS[f"rejected_total{line_tag}"] += 1
    return sides


def base_row(row):
    return {
        "game_id": row.get("game_id"),
        "sport": row.get("sport"),
        "league": row.get("league"),
        "match_date": row.get("match_date"),
        "match_time": row.get("match_time"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
    }


def process_file(file: Path):
    date, league, market_type = parse_filename(file.name)
    if market_type is None:
        _log(f"SKIP unrecognized: {file.name}", "WARN")
        return [], "skip"
    league_key = (league or "").lower().strip()
    if league_key not in CONFIG:
        _log(f"SKIP league not in config: {file.name} (league={league_key!r})", "WARN")
        return [], "skip"

    cfg = market_cfg(league_key, market_type)
    if not cfg.get("enabled", True):
        _log(f"DISABLED in config: league={league_key} market={market_type}")
        return [], "disabled"

    df = pd.read_csv(file)
    if df.empty:
        _log(f"EMPTY: {file.name}", "WARN")
        return [], "empty"

    selection_mode = cfg.get("selection_mode", "all_qualifying")
    preference = cfg.get("pick_preference", {"metric": "ev", "direction": "max"})
    _log(
        f"--- FILE: {file.name} league={league_key} market={market_type} "
        f"rows={len(df)} mode={selection_mode}"
    )

    out_rows = []
    for _, row in df.iterrows():
        raw_game_date = row.get("match_date")
        if raw_game_date is None or pd.isna(raw_game_date) or str(raw_game_date).strip() == "":
            game_date = date
        else:
            game_date = str(raw_game_date).strip().replace("-", "_")

        ml_pass, ml_meta = passes_ml_filter(row, league_key, market_type, cfg)
        if not ml_pass:
            continue

        if market_type == "match_odds":
            sides = build_match_odds_sides(row, game_date, cfg)
        elif market_type == "btts":
            sides = build_btts_sides(row, game_date, cfg)
        elif market_type == "total25":
            sides = build_totals_sides(row, game_date, cfg, "25")
        elif market_type == "total35":
            sides = build_totals_sides(row, game_date, cfg, "35")
        else:
            sides = []

        if not sides:
            continue
        picks = sides if selection_mode == "all_qualifying" else [pick(sides, preference)]
        picks = [p for p in picks if p]

        for sel in picks:
            sel_ev = fv(sel.get("ev"))
            if sel_ev is None:
                DEBUG_COUNTS["blocked_missing_ev"] += 1
                _log(
                    f"BLOCKED MISSING EV | file={file.name} league={league_key} "
                    f"market={market_type} side={sel.get('side')} ev={sel.get('ev')}",
                    "WARN",
                )
                continue
            if sel_ev < 0:
                DEBUG_COUNTS["blocked_negative_ev"] += 1
                _log(
                    f"BLOCKED NEGATIVE EV | file={file.name} league={league_key} "
                    f"market={market_type} side={sel.get('side')} ev={sel_ev}",
                    "WARN",
                )
                continue

            DEBUG_COUNTS["selected"] += 1
            out_row = {
                **base_row(row),
                "match_date": game_date,
                "market": market_type,
                "side": sel["side"],
                "odds": sel["odds"],
                "american_odds": sel["american_odds"],
                "ev": sel_ev,
                "kelly": sel["kelly"],
                "model_prob": sel["model_prob"],
                "edge": sel["edge"],
                "model_prob_source": sel.get("model_prob_source"),
                "model_prob_underlying_source": sel.get("model_prob_underlying_source"),
                "ml_predictability": ml_meta["ml_predictability"],
                "ml_predictability_source": ml_meta["ml_predictability_source"],
                "ml_skip_prob": ml_meta["ml_skip_prob"],
                "ml_skip_prob_source": ml_meta["ml_skip_prob_source"],
                "selection_period": selection_period(game_date),
            }
            if market_type == "match_odds":
                out_row.update(
                    {
                        "ev_prob": sel.get("ev_prob"),
                        "ev_prob_source": sel.get("ev_prob_source"),
                        "kelly_prob": sel.get("kelly_prob"),
                        "kelly_prob_source": sel.get("kelly_prob_source"),
                        "fair_odds_prob": sel.get("fair_odds_prob"),
                        "fair_odds_prob_source": sel.get("fair_odds_prob_source"),
                        "fair_odds": sel.get("fair_odds"),
                        "edge_prob": sel.get("edge_prob"),
                        "edge_prob_source": sel.get("edge_prob_source"),
                        "edge_fair_odds": sel.get("edge_fair_odds"),
                    }
                )
            out_rows.append(out_row)

    _log(f"{file.name} | {len(out_rows)} selected from {len(df)} rows")
    return out_rows, "ok"


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== soccer select_bets RUN {_now()} ===\n")

    clear_old_outputs()
    summary = {
        "files_processed": 0,
        "total_bets": 0,
        "dates_written": 0,
        "skipped": 0,
        "locked_created": 0,
        "locked_preserved": 0,
        "errors": 0,
    }
    per_market = {}
    per_date = {}
    per_league = {}
    all_bets = []
    seen_input_dates = set()

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")
    _log(f"LOCKED_DIR: {LOCKED_DIR}")
    _log(f"CONFIG    : {CONFIG_PATH}")
    _log(
        "BACKTEST POLICY | "
        f"tuning={BACKTEST_POLICY['tuning_start']}..{BACKTEST_POLICY['tuning_end']} | "
        f"evaluation_start={BACKTEST_POLICY['evaluation_start']}"
    )
    _log(
        "MATCH_ODDS PROBABILITY SOURCES | "
        + " | ".join(
            f"{key}={MATCH_ODDS_PROBABILITY_SOURCES[key]}"
            for key in MATCH_ODDS_SOURCE_KEYS
        )
    )

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    _log(f"Files found: {len(input_files)}")
    active_lock_date = resolve_lock_date()

    try:
        for file in input_files:
            file_date, _, market_type = parse_filename(file.name)
            if file_date and market_type:
                seen_input_dates.add(file_date)
                if not historical_rebuild_allowed(file_date, active_lock_date):
                    period = selection_period(file_date)
                    _log(
                        "SKIP OUTSIDE REBUILD WINDOW | "
                        f"file={file.name} | date={file_date} | period={period} | "
                        f"tuning={BACKTEST_POLICY['tuning_start']}..{BACKTEST_POLICY['tuning_end']} | "
                        f"current_eval_date={active_lock_date}",
                        "WARN",
                    )
                    DEBUG_COUNTS[f"skipped_period_{period}"] += 1
                    summary["skipped"] += 1
                    continue

            try:
                rows, status = process_file(file)
                if status in ("skip", "empty", "disabled"):
                    summary["skipped"] += 1
                    continue
                summary["files_processed"] += 1
                all_bets.extend(rows)
                for b in rows:
                    mkt = b.get("market", "unknown")
                    per_market[mkt] = per_market.get(mkt, 0) + 1
                    lg = str(b.get("league", "unknown")).lower()
                    per_league[lg] = per_league.get(lg, 0) + 1
            except KeyError as e:
                _log(f"{file.name} CONFIG ERROR: {e}", "ERROR")
                summary["errors"] += 1
            except Exception as e:
                _log(f"{file.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                summary["errors"] += 1

        if all_bets:
            df_all = pd.DataFrame(all_bets)
            summary["total_bets"] = len(df_all)
            for date, group in df_all.groupby("match_date"):
                out_path = OUTPUT_DIR / f"{date}_soccer_bets.csv"
                group.to_csv(out_path, index=False)
                summary["dates_written"] += 1
                per_date[str(date)] = {"bets": len(group), "file": out_path.name}
                _log(f"WROTE: {out_path} ({len(group)} bets)")
        else:
            df_all = pd.DataFrame(columns=LOCK_BASE_COLUMNS)
            _log("No bets selected across all files", "WARN")

        lock_daily_picks(df_all, seen_input_dates, summary)
    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1
        _write_summary(summary, per_market, per_date, per_league)
        sys.exit(1)

    _write_summary(summary, per_market, per_date, per_league)
    print("soccer select_bets complete.")


if __name__ == "__main__":
    main()
