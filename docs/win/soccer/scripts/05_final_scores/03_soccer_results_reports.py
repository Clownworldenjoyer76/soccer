#!/usr/bin/env python3
# docs/win/soccer/scripts/05_final_scores/03_soccer_results_reports.py
#
# Reads the enriched intermediate file produced by 02_soccer_results_analyze.py
# and writes:
#
# 1) Tally files
#      docs/win/soccer/05_final_scores/all_soccer_market_tally.csv
#          headers: market, market_type, Win, Loss, Push, Total, Win_Pct
#          (one row per (market, side) across all leagues)
#
#      docs/win/soccer/05_final_scores/{league}_market_tally.csv
#          headers: market, market_type, Win, Loss, Push, Total, Win_Pct
#          (one row per (market, side) within that league)
#
# 2) Per-league per-market bucket reports
#      docs/win/soccer/05_final_scores/reports/{league}/{market_folder}/
#
#    For btts / total_25 / total_35 (8 files each):
#      {league}_{market_folder}_by_ev.csv
#      {league}_{market_folder}_by_ev_{sides}_summary.csv
#      {league}_{market_folder}_by_kelly.csv
#      {league}_{market_folder}_by_kelly_{sides}_summary.csv
#      {league}_{market_folder}_by_month.csv
#      {league}_{market_folder}_by_month_{sides}_summary.csv
#      {league}_{market_folder}_by_odds.csv
#      {league}_{market_folder}_by_odds_{sides}_summary.csv
#
#    For match_odds (10 files), additionally:
#      {league}_match_odds_by_win_prob.csv
#      {league}_match_odds_by_win_prob_home_draw_away_summary.csv
#
#    by_*.csv         => combined across sides; headers: bucket, Win, Loss, Push, Total, Win_Pct
#    *_summary.csv    => per side per bucket; headers: bucket, side, Win, Loss, Push, Total, Win_Pct
#
# Win_Pct = Win / (Win + Loss), excluding pushes. Total = Win + Loss + Push.
# Rows with bet_result not in {Win, Loss, Push} are excluded from counts.

from datetime import datetime
from pathlib import Path
import shutil
import traceback

import pandas as pd


# =========================
# PATHS
# =========================

INTERMEDIATE = Path("docs/win/soccer/05_final_scores/intermediate/work_soccer.csv")
FINAL_DIR    = Path("docs/win/soccer/05_final_scores")
REPORTS_DIR  = FINAL_DIR / "reports"
ERROR_DIR    = FINAL_DIR / "errors"

ALL_TALLY    = FINAL_DIR / "all_soccer_market_tally.csv"
ERROR_LOG    = ERROR_DIR / "soccer_results_reports_errors.txt"
SUMMARY_LOG  = ERROR_DIR / "soccer_results_reports_summary.txt"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CONFIG
# =========================

# market_type value -> (folder name on disk, side-label suffix for summary files)
MARKET_LAYOUT = {
    "match_odds": ("match_odds", "home_draw_away"),
    "btts":       ("btts",       "yes_no"),
    "total25":    ("total_25",   "over_under"),
    "total35":    ("total_35",   "over_under"),
}

# Bucket types to generate per market.
# Each entry: (bucket_col, sort_col, by_label, market_types_allowed)
BUCKETS = [
    ("ev_bucket",       "ev_sort",       "ev",       None),
    ("kelly_bucket",    "kelly_sort",    "kelly",    None),
    ("month_bucket",    "month_sort",    "month",    None),
    ("odds_bucket",     "odds_sort",     "odds",     None),
    ("win_prob_bucket", "win_prob_sort", "win_prob", {"match_odds"}),
]

VALID_RESULTS = {"Win", "Loss", "Push"}

LEAGUE_TALLY_FILES = [
    FINAL_DIR / "epl_market_tally.csv",
    FINAL_DIR / "bundesliga_market_tally.csv",
    FINAL_DIR / "laliga_market_tally.csv",
    FINAL_DIR / "ligue1_market_tally.csv",
    FINAL_DIR / "seriea_market_tally.csv",
    FINAL_DIR / "mls_market_tally.csv",
]


# =========================
# LOGGING
# =========================

def reset_logs() -> None:
    SUMMARY_LOG.write_text("", encoding="utf-8")
    ERROR_LOG.write_text("", encoding="utf-8")


def log_error(msg: str) -> None:
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def log_summary(msg: str) -> None:
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def clear_output_files() -> None:
    deleted_files = 0
    deleted_dirs = 0

    if ALL_TALLY.exists():
        ALL_TALLY.unlink()
        deleted_files += 1
        log_summary(f"DELETED OLD OUTPUT | {ALL_TALLY}")

    for path in LEAGUE_TALLY_FILES:
        if path.exists():
            path.unlink()
            deleted_files += 1
            log_summary(f"DELETED OLD OUTPUT | {path}")

    for path in sorted(FINAL_DIR.glob("*_market_tally.csv")):
        if path.exists():
            path.unlink()
            deleted_files += 1
            log_summary(f"DELETED OLD OUTPUT | {path}")

    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
        deleted_dirs += 1
        log_summary(f"DELETED OLD REPORTS DIR | {REPORTS_DIR}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log_summary(
        f"OLD REPORT OUTPUTS DELETED | files={deleted_files} dirs={deleted_dirs}"
    )


# =========================
# IO HELPERS
# =========================

def safe_read_intermediate(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            log_error(f"INTERMEDIATE FILE MISSING | {path} — run 02 first")
            return pd.DataFrame()

        df = pd.read_csv(
            path,
            dtype={
                "month_bucket":    str,
                "ev_bucket":       str,
                "kelly_bucket":    str,
                "odds_bucket":     str,
                "win_prob_bucket": str,
            },
        )

        if df.empty:
            log_error(f"INTERMEDIATE FILE EMPTY | {path}")
            return pd.DataFrame()

        return df

    except Exception as e:
        log_error(f"READ ERROR | {path} | {e}")
        log_error(traceback.format_exc())
        return pd.DataFrame()


# =========================
# AGG HELPERS
# =========================

def summarize(sub: pd.DataFrame) -> dict:
    res = sub["bet_result"].astype(str)
    w = int((res == "Win").sum())
    l = int((res == "Loss").sum())
    p = int((res == "Push").sum())
    total = w + l + p
    pct = round(w / (w + l), 4) if (w + l) > 0 else 0.0
    return {"Win": w, "Loss": l, "Push": p, "Total": total, "Win_Pct": pct}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log_summary(f"WROTE {path} ({len(df)} rows)")


def filter_graded(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a valid bet_result (Win/Loss/Push)."""
    if "bet_result" not in df.columns:
        log_error("MISSING COLUMN | bet_result")
        return pd.DataFrame()

    return df[df["bet_result"].astype(str).isin(VALID_RESULTS)].copy()


# =========================
# TALLY FILES
# =========================

def build_all_tally(df: pd.DataFrame) -> None:
    """One row per (market, side) across all leagues."""
    rows = []

    for (market, side), sub in df.groupby(["market_type", "side"], dropna=False):
        s = summarize(sub)
        rows.append({"market": market, "market_type": side, **s})

    out = pd.DataFrame(
        rows,
        columns=["market", "market_type", "Win", "Loss", "Push", "Total", "Win_Pct"],
    )

    if not out.empty:
        out = out.sort_values(["market", "market_type"]).reset_index(drop=True)

    write_csv(out, ALL_TALLY)


def build_league_tally(df: pd.DataFrame, league: str) -> None:
    rows = []

    for (market, side), sub in df.groupby(["market_type", "side"], dropna=False):
        s = summarize(sub)
        rows.append({"market": market, "market_type": side, **s})

    out = pd.DataFrame(
        rows,
        columns=["market", "market_type", "Win", "Loss", "Push", "Total", "Win_Pct"],
    )

    if not out.empty:
        out = out.sort_values(["market", "market_type"]).reset_index(drop=True)

    path = FINAL_DIR / f"{league}_market_tally.csv"
    write_csv(out, path)


# =========================
# BUCKET REPORTS
# =========================

def by_bucket(df: pd.DataFrame, bucket_col: str, sort_col: str) -> pd.DataFrame:
    """Combined across sides: bucket -> W/L/P/T/Win_Pct."""
    rows = []

    for bucket, sub in df.groupby(bucket_col, dropna=False):
        sort_val = sub[sort_col].dropna().iloc[0] if sub[sort_col].notna().any() else None
        s = summarize(sub)
        rows.append({"bucket": bucket, "_sort": sort_val, **s})

    if not rows:
        return pd.DataFrame(columns=["bucket", "Win", "Loss", "Push", "Total", "Win_Pct"])

    out = pd.DataFrame(rows)
    out["_sort"] = pd.to_numeric(out["_sort"], errors="coerce")
    out = out.sort_values(["_sort", "bucket"], na_position="last").reset_index(drop=True)
    out = out.drop(columns=["_sort"])

    return out[["bucket", "Win", "Loss", "Push", "Total", "Win_Pct"]]


def by_bucket_by_side(df: pd.DataFrame, bucket_col: str, sort_col: str) -> pd.DataFrame:
    """Per-side breakdown: (bucket, side) -> W/L/P/T/Win_Pct."""
    rows = []

    for (bucket, side), sub in df.groupby([bucket_col, "side"], dropna=False):
        sort_val = sub[sort_col].dropna().iloc[0] if sub[sort_col].notna().any() else None
        s = summarize(sub)
        rows.append({"bucket": bucket, "side": side, "_sort": sort_val, **s})

    if not rows:
        return pd.DataFrame(columns=["bucket", "side", "Win", "Loss", "Push", "Total", "Win_Pct"])

    out = pd.DataFrame(rows)
    out["_sort"] = pd.to_numeric(out["_sort"], errors="coerce")
    out = out.sort_values(["_sort", "bucket", "side"], na_position="last").reset_index(drop=True)
    out = out.drop(columns=["_sort"])

    return out[["bucket", "side", "Win", "Loss", "Push", "Total", "Win_Pct"]]


def build_market_reports(df: pd.DataFrame, league: str, market_type: str) -> None:
    if market_type not in MARKET_LAYOUT:
        log_error(f"UNKNOWN market_type | {market_type}")
        return

    folder_name, sides_label = MARKET_LAYOUT[market_type]
    out_dir = REPORTS_DIR / league / folder_name

    sub = df[df["market_type"].astype(str) == market_type]

    if sub.empty:
        log_summary(f"NO ROWS | league={league} market={market_type} — skipping reports")
        return

    for bucket_col, sort_col, by_label, allowed in BUCKETS:
        if allowed and market_type not in allowed:
            continue

        if bucket_col not in sub.columns:
            log_error(f"MISSING COLUMN | {bucket_col} (league={league} market={market_type})")
            continue

        if sort_col not in sub.columns:
            log_error(f"MISSING COLUMN | {sort_col} (league={league} market={market_type})")
            continue

        combined = by_bucket(sub, bucket_col, sort_col)
        write_csv(combined, out_dir / f"{league}_{folder_name}_by_{by_label}.csv")

        bysd = by_bucket_by_side(sub, bucket_col, sort_col)
        write_csv(bysd, out_dir / f"{league}_{folder_name}_by_{by_label}_{sides_label}_summary.csv")


# =========================
# MAIN
# =========================

def main() -> None:
    reset_logs()
    log_summary(f"=== START 03_soccer_results_reports.py {datetime.now().isoformat()} ===")

    clear_output_files()

    raw = safe_read_intermediate(INTERMEDIATE)

    if raw.empty:
        log_error("NO REPORTS WRITTEN | intermediate file missing, empty, unreadable, or invalid")
        log_summary(f"=== END 03_soccer_results_reports.py {datetime.now().isoformat()} ===")
        print("ERROR: no intermediate rows to report on.")
        return

    df = filter_graded(raw)

    if df.empty:
        log_error("NO ROWS WITH VALID bet_result (Win/Loss/Push)")
        log_summary("NO REPORTS WRITTEN | old report outputs were already cleared")
        log_summary(f"=== END 03_soccer_results_reports.py {datetime.now().isoformat()} ===")
        print("ERROR: no graded rows to report on.")
        return

    required_cols = ["league_lower", "market_type", "side"]

    missing_required = [c for c in required_cols if c not in df.columns]

    if missing_required:
        log_error(f"MISSING REQUIRED REPORT COLUMNS | {missing_required}")
        log_summary("NO REPORTS WRITTEN | old report outputs were already cleared")
        log_summary(f"=== END 03_soccer_results_reports.py {datetime.now().isoformat()} ===")
        print("ERROR: required report columns missing.")
        return

    df["league_lower"] = df["league_lower"].astype(str).str.lower().str.strip()
    df["market_type"] = df["market_type"].astype(str).str.lower().str.strip()
    df["side"] = df["side"].astype(str).str.lower().str.strip()

    log_summary(f"Rows loaded (graded only): {len(df)}")
    log_summary(f"market_type counts: {df['market_type'].value_counts().to_dict()}")
    log_summary(f"leagues: {df['league_lower'].value_counts().to_dict()}")

    build_all_tally(df)

    for league, league_df in df.groupby("league_lower"):
        build_league_tally(league_df, league)

    for (league, market_type), grp in df.groupby(["league_lower", "market_type"]):
        build_market_reports(grp, league, market_type)

    log_summary(f"=== END 03_soccer_results_reports.py {datetime.now().isoformat()} ===")
    print("Soccer reports generated.")


if __name__ == "__main__":
    main()
