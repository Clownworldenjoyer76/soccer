#!/usr/bin/env python3
# docs/win/soccer/scripts/05_final_scores/02_soccer_results_analyze.py
#
# Reads the master graded file produced by 01_soccer_results_grade.py and
# enriches each row with bucket columns used by the reporting stage.
#
# Adds:
#   ev_bucket
#   kelly_bucket
#   odds_bucket        American odds buckets, converted from decimal odds
#   month_bucket
#   selected_win_prob
#   win_prob_bucket
#
# Output:
#   docs/win/soccer/05_final_scores/intermediate/work_soccer.csv

from __future__ import annotations

import math
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd


# =========================
# PATHS
# =========================

GRADED_DIR        = Path("docs/win/soccer/05_final_scores/results/graded")
EDGES_DIR         = Path("docs/win/soccer/03_edges")
INTERMEDIATE_DIR  = Path("docs/win/soccer/05_final_scores/intermediate")
ERROR_DIR         = Path("docs/win/soccer/05_final_scores/errors")

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG    = ERROR_DIR / "soccer_results_analyze_errors.txt"
SUMMARY_LOG  = ERROR_DIR / "soccer_results_analyze_summary.txt"

MASTER_FILE  = GRADED_DIR / "SOCCER_final.csv"
WORK_FILE    = INTERMEDIATE_DIR / "work_soccer.csv"


# =========================
# REQUIRED HEADERS
# =========================

REQUIRED_COLUMNS = [
    "game_id", "sport", "league", "match_date", "match_time",
    "home_team", "away_team",
    "market", "side", "odds", "ev", "kelly",
    "game_date", "league_lower", "market_type", "take_bet",
    "odds_american", "edge_pct",
    "home_score", "away_score", "bet_result",
]


# =========================
# LOGGING
# =========================

def reset_logs() -> None:
    ERROR_LOG.write_text("", encoding="utf-8")
    SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg: str) -> None:
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def log_summary(msg: str) -> None:
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def clear_output_files() -> None:
    deleted = 0

    if WORK_FILE.exists():
        WORK_FILE.unlink()
        deleted += 1
        log_summary(f"DELETED OLD OUTPUT | {WORK_FILE}")

    log_summary(f"OLD OUTPUT FILES DELETED | count={deleted}")


# =========================
# IO HELPERS
# =========================

def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            log_error(f"FILE MISSING | {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)

        if df.empty:
            log_error(f"FILE EMPTY | {path}")
            return pd.DataFrame()

        return df

    except Exception as e:
        log_error(f"READ ERROR | {path} | {e}")
        log_error(traceback.format_exc())
        return pd.DataFrame()


def validate_headers(df: pd.DataFrame, required: list[str], path: Path) -> bool:
    missing = [c for c in required if c not in df.columns]

    if missing:
        log_error(f"MISSING HEADERS | {path}")
        log_error(f"Missing: {missing}")
        log_error(f"Available: {list(df.columns)}")
        return False

    return True


# =========================
# BUCKETS
# =========================

def step_bucket(value, step: float, decimals: int) -> tuple[str, float | None]:
    if value is None or pd.isna(value):
        return "missing", None

    try:
        v = float(value)
    except Exception:
        return "missing", None

    floor_val = math.floor(v / step) * step
    lo = round(floor_val, decimals)
    hi = round(floor_val + step, decimals)

    fmt = f"{{:.{decimals}f}}"

    return f"{fmt.format(lo)}_to_{fmt.format(hi)}", lo


def ev_bucket(v):
    return step_bucket(v, 0.05, 2)


def kelly_bucket(v):
    return step_bucket(v, 0.05, 2)


def win_prob_bucket(v):
    return step_bucket(v, 0.10, 1)


def decimal_to_american(dec) -> float | None:
    """Convert decimal odds to American odds."""
    if dec is None or pd.isna(dec):
        return None

    try:
        d = float(dec)
    except Exception:
        return None

    if d <= 1.0:
        return None

    if d >= 2.0:
        return (d - 1.0) * 100.0

    return -100.0 / (d - 1.0)


def odds_bucket_from_american(american) -> tuple[str, float | None]:
    if american is None or pd.isna(american):
        return "missing", None

    try:
        a = float(american)
    except Exception:
        return "missing", None

    if a <= -300:
        return "-300_or_lower", -10000.0
    if -300 < a <= -250:
        return "-299_to_-250", -299.0
    if -250 < a <= -200:
        return "-249_to_-200", -249.0
    if -200 < a <= -150:
        return "-199_to_-150", -199.0
    if -150 < a <= -100:
        return "-149_to_-100", -149.0
    if 100 <= a < 150:
        return "+100_to_+149", 100.0
    if 150 <= a < 200:
        return "+150_to_+199", 150.0
    if 200 <= a < 250:
        return "+200_to_+249", 200.0
    if 250 <= a < 300:
        return "+250_to_+299", 250.0
    if 300 <= a:
        return "+300_or_higher", 300.0

    return "missing", None


def month_bucket(match_date) -> tuple[str, int | None]:
    if match_date is None or pd.isna(match_date):
        return "missing", None

    s = str(match_date).strip()
    parts = s.split("_")

    if len(parts) >= 2 and parts[1].isdigit():
        try:
            mm = int(parts[1])

            if 1 <= mm <= 12:
                return f"{mm:02d}", mm

        except Exception:
            pass

    return "missing", None


# =========================
# WIN PROB JOIN
# =========================

_edges_cache: dict[tuple[str, str], pd.DataFrame] = {}


def load_match_odds_edges(game_date: str, league_lower: str) -> pd.DataFrame:
    key = (game_date, league_lower)

    if key in _edges_cache:
        return _edges_cache[key]

    path = EDGES_DIR / f"{game_date}_{league_lower}_match_odds.csv"
    df = safe_read_csv(path)

    if df.empty:
        _edges_cache[key] = df
        return df

    needed = ["game_id", "engine_home_prob", "engine_draw_prob", "engine_away_prob"]
    have = [c for c in needed if c in df.columns]

    if "game_id" not in have:
        log_error(f"NO game_id IN EDGES | {path}")
        _edges_cache[key] = pd.DataFrame()
        return _edges_cache[key]

    sub = df[have].copy()
    sub["game_id"] = sub["game_id"].astype(str).str.strip()
    sub = sub.drop_duplicates(subset=["game_id"], keep="first")

    _edges_cache[key] = sub

    return sub


def attach_win_prob(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["selected_win_prob"] = pd.NA

    is_match_odds = df["market_type"].astype(str).str.lower().eq("match_odds")

    if not is_match_odds.any():
        return df

    df["game_id"] = df["game_id"].astype(str).str.strip()

    for (game_date, league_lower), grp in df[is_match_odds].groupby(
        ["game_date", "league_lower"],
        dropna=False,
    ):
        edges = load_match_odds_edges(str(game_date), str(league_lower).lower())

        if edges.empty:
            continue

        merged = grp.merge(
            edges[["game_id", "engine_home_prob", "engine_draw_prob", "engine_away_prob"]],
            on="game_id",
            how="left",
            suffixes=("", "_edge"),
        )

        prob_vals = []

        for _, r in merged.iterrows():
            s = str(r["side"]).lower().strip()
            col = {
                "home": "engine_home_prob",
                "draw": "engine_draw_prob",
                "away": "engine_away_prob",
            }.get(s)

            if col is None:
                prob_vals.append(pd.NA)
                continue

            v = r.get(col)
            prob_vals.append(v if v is not None and not pd.isna(v) else pd.NA)

        df.loc[grp.index, "selected_win_prob"] = pd.Series(prob_vals, index=grp.index)

    return df


# =========================
# PREPARE
# =========================

def prepare() -> None:
    df = safe_read_csv(MASTER_FILE)

    if df.empty:
        log_summary("NO WORK FILE WRITTEN | master file missing, empty, or unreadable")
        return

    if not validate_headers(df, REQUIRED_COLUMNS, MASTER_FILE):
        log_summary("NO WORK FILE WRITTEN | master file failed required-header validation")
        return

    text_cols = [
        "game_id", "sport", "league", "match_date", "match_time",
        "home_team", "away_team", "market", "side",
        "game_date", "league_lower", "market_type", "take_bet", "bet_result",
    ]

    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()

    df["selected_ev"] = pd.to_numeric(df["ev"], errors="coerce")
    df["selected_kelly"] = pd.to_numeric(df["kelly"], errors="coerce")
    df["selected_odds_decimal"] = pd.to_numeric(df["odds"], errors="coerce")

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # Prefer real odds_american from the grade step.
    # Fallback converts from decimal odds if missing/bad.
    df["selected_odds_american"] = pd.to_numeric(df["odds_american"], errors="coerce")

    missing_american = df["selected_odds_american"].isna()
    df.loc[missing_american, "selected_odds_american"] = (
        df.loc[missing_american, "selected_odds_decimal"].apply(decimal_to_american)
    )

    ev_b = df["selected_ev"].apply(ev_bucket)
    kel_b = df["selected_kelly"].apply(kelly_bucket)
    odd_b = df["selected_odds_american"].apply(odds_bucket_from_american)
    mon_b = df["match_date"].apply(month_bucket)

    df["ev_bucket"] = ev_b.apply(lambda t: t[0])
    df["ev_sort"] = ev_b.apply(lambda t: t[1])

    df["kelly_bucket"] = kel_b.apply(lambda t: t[0])
    df["kelly_sort"] = kel_b.apply(lambda t: t[1])

    df["odds_bucket"] = odd_b.apply(lambda t: t[0])
    df["odds_sort"] = odd_b.apply(lambda t: t[1])

    df["month_bucket"] = mon_b.apply(lambda t: t[0])
    df["month_sort"] = mon_b.apply(lambda t: t[1])

    df = attach_win_prob(df)

    wp_num = pd.to_numeric(df["selected_win_prob"], errors="coerce")
    wp_b = wp_num.apply(win_prob_bucket)

    df["win_prob_bucket"] = wp_b.apply(lambda t: t[0])
    df["win_prob_sort"] = wp_b.apply(lambda t: t[1])

    df.to_csv(WORK_FILE, index=False)

    log_summary(f"WORK FILE WRITTEN | {WORK_FILE} | rows={len(df)}")
    log_summary(f"market_type counts: {df['market_type'].value_counts(dropna=False).to_dict()}")
    log_summary(f"bet_result counts: {df['bet_result'].value_counts(dropna=False).to_dict()}")
    log_summary(f"ev_bucket nunique: {df['ev_bucket'].nunique()}")
    log_summary(f"kelly_bucket nunique: {df['kelly_bucket'].nunique()}")
    log_summary(f"odds_bucket counts: {df['odds_bucket'].value_counts(dropna=False).to_dict()}")
    log_summary(f"month_bucket counts: {df['month_bucket'].value_counts(dropna=False).to_dict()}")
    log_summary(
        f"win_prob_bucket counts (match_odds only): "
        f"{df.loc[df['market_type'] == 'match_odds', 'win_prob_bucket'].value_counts(dropna=False).to_dict()}"
    )


# =========================
# MAIN
# =========================

def main() -> None:
    reset_logs()
    log_summary(f"=== START 02_soccer_results_analyze.py {datetime.now().isoformat()} ===")
    clear_output_files()
    prepare()
    log_summary(f"=== END 02_soccer_results_analyze.py {datetime.now().isoformat()} ===")
    print("Soccer analytics preparation complete.")


if __name__ == "__main__":
    main()
