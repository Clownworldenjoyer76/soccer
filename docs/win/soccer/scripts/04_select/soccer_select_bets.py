#!/usr/bin/env python3
# docs/win/soccer/scripts/04_select/soccer_select_bets.py
#
# Reads stage-3 EV/Kelly outputs and applies per-league x per-market x per-side
# filters from markets.yaml. Picks bet(s) per game according to the configured
# selection_mode and pick_preference.
#
# Input layout:
#   docs/win/soccer/03_edges/{date}_{league}_match_odds.csv
#   docs/win/soccer/03_edges/{date}_{league}_btts.csv
#   docs/win/soccer/03_edges/{date}_{league}_total_25.csv
#   docs/win/soccer/03_edges/{date}_{league}_total_35.csv
#
# Output:
#   docs/win/soccer/04_select/{date}_soccer_bets.csv
#
# Output columns:
#   game_id, sport, league, match_date, match_time,
#   home_team, away_team, market, side,
#   odds, american_odds, ev, kelly, model_prob, edge
#
# Filters per side:
#   odds_bands              decimal odds
#   american_odds_bands     American odds converted from decimal odds inside this script
#   ev_bands                decimal EV
#   kelly_bands             decimal Kelly fraction
#   model_prob_bands        decimal probability, from engine_*_prob
#   edge_bands              decimal edge, from *_edge
#
# Empty filter lists are ignored.
#
# Date filters per side:
#   months                list of ints 1-12; empty = all months allowed
#   exclude_days_of_week  list of ints 0=Mon ... 6=Sun
#
# Per-market:
#   enabled
#   selection_mode: pick_one | all_qualifying
#   pick_preference: { metric: ev|kelly|model_prob|edge|odds|american_odds, direction: max|min }

import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[2]

INPUT_DIR   = BASE / "03_edges"
OUTPUT_DIR  = BASE / "04_select"
CONFIG_PATH = BASE / "config" / "markets.yaml"

ERROR_DIR = BASE / "errors" / "04_select"
LOG_FILE  = ERROR_DIR / "select_bets.txt"

OUTPUT_DIR.mkdir(exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

MARKET_FROM_SUFFIX = {
    "_match_odds": "match_odds",
    "_btts":       "btts",
    "_total_25":   "total25",
    "_total_35":   "total35",
}

LEAGUES = ["bundesliga", "seriea", "laliga", "ligue1", "epl", "mls"]

DEBUG_COUNTS: dict = defaultdict(int)


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_market: dict, per_date: dict, per_league: dict) -> None:
    lines = [
        "",
        "=" * 70,
        f"SUMMARY  {_now()}",
        "=" * 70,
        f"  files_processed : {summary['files_processed']}",
        f"  total_bets      : {summary['total_bets']}",
        f"  dates_written   : {summary['dates_written']}",
        f"  skipped_files   : {summary['skipped']}",
        f"  errors          : {summary['errors']}",
        "",
        "--- By Market ---",
        f"  {'market':<15} {'bets':>6}",
    ]

    for m, c in sorted(per_market.items()):
        lines.append(f"  {m:<15} {c:>6}")

    lines += [
        "",
        "--- By League ---",
        f"  {'league':<15} {'bets':>6}",
    ]

    for lg, c in sorted(per_league.items()):
        lines.append(f"  {lg:<15} {c:>6}")

    lines += [
        "",
        "--- By Date ---",
        f"  {'date':<14} {'bets':>6} {'file'}",
    ]

    for date, info in sorted(per_date.items()):
        lines.append(f"  {date:<14} {info['bets']:>6}  {info['file']}")

    lines += ["", "--- Filter Reject Counts ---"]

    for k, v in sorted(DEBUG_COUNTS.items()):
        lines.append(f"  {k:<36} : {v}")

    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 70]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# CONFIG
# =========================

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = data["markets"]["soccer"]

        if not isinstance(config, dict):
            raise ValueError("markets.soccer must be a mapping")

        return config

    except Exception as e:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== soccer select_bets RUN {_now()} ===\n")
        _log(f"CONFIG LOAD FAILED | {CONFIG_PATH} | {e}\n{traceback.format_exc()}", "ERROR")
        raise


CONFIG = load_config()


# =========================
# HELPERS
# =========================

def fv(x):
    """Float-or-None from any cell."""
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def decimal_to_american(decimal_odds):
    """
    Convert decimal odds to American odds.

    Examples:
      1.50 -> -200
      2.00 -> +100
      2.50 -> +150
    """
    d = fv(decimal_odds)

    if d is None or d <= 1:
        return None

    if d >= 2:
        return round((d - 1) * 100)

    return round(-100 / (d - 1))


def normalize_bands(raw_bands, label: str = "") -> list:
    """
    Normalize YAML bands into numeric [lo, hi] pairs.

    Empty / missing bands return [] and are ignored by passes_filters().
    Invalid bands raise ValueError so bad YAML does not silently pass.
    """
    if raw_bands is None:
        return []

    if raw_bands == []:
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
    """True if value falls inside any [lo, hi] band inclusive."""
    if value is None or not bands:
        return False

    v = fv(value)

    if v is None:
        return False

    return any(lo <= v <= hi for lo, hi in bands)


def parse_date(s):
    try:
        return datetime.strptime(s, "%Y_%m_%d")
    except Exception:
        return None


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

    if not date_ok(
        game_date,
        scfg.get("months", []) or [],
        scfg.get("exclude_days_of_week", []) or [],
    ):
        return False

    return True


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
    american_odds = decimal_to_american(odds)

    return {
        "odds": odds,
        "american_odds": american_odds,
        "ev": ev,
        "kelly": kelly,
        "model_prob": model_prob,
        "edge": edge,
    }


def make_side(side, odds, ev, kelly, model_prob, edge):
    american_odds = decimal_to_american(odds)

    return {
        "side": side,
        "odds": odds,
        "american_odds": american_odds,
        "ev": ev,
        "kelly": kelly,
        "model_prob": model_prob,
        "edge": edge,
    }


def clear_old_outputs() -> None:
    deleted = 0

    for old_file in sorted(OUTPUT_DIR.glob("*_soccer_bets.csv")):
        old_file.unlink()
        deleted += 1
        _log(f"DELETED OLD SELECT FILE: {old_file}")

    DEBUG_COUNTS["deleted_old_select_files"] += deleted
    _log(f"Old select files deleted: {deleted}")


# =========================
# FILENAME PARSING
# =========================

def parse_filename(name: str):
    """
    Returns (date, league, market_type) or (None, None, None) if not recognized.
    File pattern: {YYYY_MM_DD}_{league}_{market_suffix}.csv
    """
    stem = name[:-4] if name.endswith(".csv") else name

    market_type = None
    league_part = None

    for suffix, mt in MARKET_FROM_SUFFIX.items():
        if stem.endswith(suffix):
            market_type = mt
            league_part = stem[: -len(suffix)]
            break

    if market_type is None:
        return None, None, None

    m = re.match(r"^(\d{4}_\d{2}_\d{2})_(.+)$", league_part)

    if not m:
        return None, None, None

    return m.group(1), m.group(2), market_type


# =========================
# MARKET SIDE BUILDERS
# =========================

def build_match_odds_sides(row, game_date, cfg):
    sides = []

    for side in ("home", "draw", "away"):
        scfg = cfg.get(side)

        if not scfg or not scfg.get("enabled", True):
            continue

        odds  = fv(row.get(f"dk_{side}_decimal"))
        ev    = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        mp    = fv(row.get(f"engine_{side}_prob"))
        edge  = fv(row.get(f"{side}_edge"))

        values = make_values(odds, ev, kelly, mp, edge)

        if passes_filters(values, scfg, game_date):
            sides.append(make_side(side, odds, ev, kelly, mp, edge))
        else:
            DEBUG_COUNTS["rejected_match_odds"] += 1

    return sides


def build_btts_sides(row, game_date, cfg):
    sides = []

    for side in ("yes", "no"):
        scfg = cfg.get(side)

        if not scfg or not scfg.get("enabled", True):
            continue

        odds  = fv(row.get(f"btts_{side}"))
        ev    = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        mp    = fv(row.get(f"engine_btts_{side}_prob"))
        edge  = fv(row.get(f"{side}_edge"))

        values = make_values(odds, ev, kelly, mp, edge)

        if passes_filters(values, scfg, game_date):
            sides.append(make_side(side, odds, ev, kelly, mp, edge))
        else:
            DEBUG_COUNTS["rejected_btts"] += 1

    return sides


def build_totals_sides(row, game_date, cfg, line_tag):
    sides = []

    for side in ("over", "under"):
        scfg = cfg.get(side)

        if not scfg or not scfg.get("enabled", True):
            continue

        odds  = fv(row.get(f"dk_{side}{line_tag}_decimal"))
        ev    = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        mp    = fv(row.get(f"engine_{side}_prob"))
        edge  = fv(row.get(f"{side}_edge"))

        values = make_values(odds, ev, kelly, mp, edge)

        if passes_filters(values, scfg, game_date):
            sides.append(make_side(side, odds, ev, kelly, mp, edge))
        else:
            DEBUG_COUNTS[f"rejected_total{line_tag}"] += 1

    return sides


def base_row(row):
    return {
        "game_id":    row.get("game_id"),
        "sport":      row.get("sport"),
        "league":     row.get("league"),
        "match_date": row.get("match_date"),
        "match_time": row.get("match_time"),
        "home_team":  row.get("home_team"),
        "away_team":  row.get("away_team"),
    }


# =========================
# FILE PROCESSOR
# =========================

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
    preference     = cfg.get("pick_preference", {"metric": "ev", "direction": "max"})

    _log(
        f"--- FILE: {file.name} league={league_key} market={market_type} "
        f"rows={len(df)} mode={selection_mode}"
    )

    out_rows = []

    for _, row in df.iterrows():
        game_date = row.get("match_date") or date

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

        if selection_mode == "all_qualifying":
            picks = sides
        else:
            p = pick(sides, preference)
            picks = [p] if p else []

        for sel in picks:
            sel_ev = fv(sel.get("ev"))

            if sel_ev is None:
                DEBUG_COUNTS["blocked_missing_ev"] += 1
                _log(
                    f"BLOCKED MISSING EV | file={file.name} "
                    f"league={league_key} market={market_type} side={sel.get('side')} ev={sel.get('ev')}",
                    "WARN",
                )
                continue

            if sel_ev < 0:
                DEBUG_COUNTS["blocked_negative_ev"] += 1
                _log(
                    f"BLOCKED NEGATIVE EV | file={file.name} "
                    f"league={league_key} market={market_type} side={sel.get('side')} ev={sel_ev}",
                    "WARN",
                )
                continue

            DEBUG_COUNTS["selected"] += 1

            out_rows.append({
                **base_row(row),
                "market":        market_type,
                "side":          sel["side"],
                "odds":          sel["odds"],
                "american_odds": sel["american_odds"],
                "ev":            sel_ev,
                "kelly":         sel["kelly"],
                "model_prob":    sel["model_prob"],
                "edge":          sel["edge"],
            })

    _log(f"{file.name} | {len(out_rows)} selected from {len(df)} rows")
    return out_rows, "ok"


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== soccer select_bets RUN {_now()} ===\n")

    clear_old_outputs()

    summary = {
        "files_processed": 0,
        "total_bets": 0,
        "dates_written": 0,
        "skipped": 0,
        "errors": 0,
    }

    per_market: dict = {}
    per_date: dict = {}
    per_league: dict = {}
    all_bets: list = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")
    _log(f"CONFIG    : {CONFIG_PATH}")

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    _log(f"Files found: {len(input_files)}")

    try:
        for file in input_files:
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

        if not all_bets:
            _log("No bets selected across all files", "WARN")
            _write_summary(summary, per_market, per_date, per_league)
            return

        df_all = pd.DataFrame(all_bets)
        summary["total_bets"] = len(df_all)

        for date, group in df_all.groupby("match_date"):
            out_path = OUTPUT_DIR / f"{date}_soccer_bets.csv"
            group.to_csv(out_path, index=False)

            summary["dates_written"] += 1
            per_date[str(date)] = {
                "bets": len(group),
                "file": out_path.name,
            }

            _log(f"WROTE: {out_path} ({len(group)} bets)")

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1
        _write_summary(summary, per_market, per_date, per_league)
        sys.exit(1)

    _write_summary(summary, per_market, per_date, per_league)
    print("soccer select_bets complete.")


if __name__ == "__main__":
    main()
