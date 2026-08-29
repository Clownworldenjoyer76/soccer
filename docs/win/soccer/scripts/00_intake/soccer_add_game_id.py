#!/usr/bin/env python3
# docs/win/soccer/scripts/01_intake/soccer_add_game_id.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback

import pandas as pd


# =========================
# PATHS
# =========================

DIRTY_SCORES_ROOT = Path("docs/win/soccer/05_final_scores/results/final_scores_dirty")
SELECT_DIR = Path("docs/win/soccer/04_select")
OUTPUT_ROOT = Path("docs/win/soccer/05_final_scores/results/final_scores")

ERROR_DIR = Path("docs/win/soccer/05_final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = ERROR_DIR / "01_soccer_add_game_id_errors.txt"
SUMMARY_LOG = ERROR_DIR / "01_soccer_add_game_id_summary.txt"


# =========================
# REQUIRED HEADERS
# =========================

FINAL_SCORES_REQUIRED = [
    "sport",
    "league",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]

SELECT_BETS_REQUIRED = [
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
    "ev",
    "kelly",
]

OUTPUT_COLUMNS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "match_time",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
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


# =========================
# HELPERS
# =========================

def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)

        if df.empty:
            log_error(f"EMPTY FILE | {path}")
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


def extract_date_from_dirty_score_file(path: Path) -> str:
    """
    Expected:
        docs/win/soccer/05_final_scores/results/final_scores_dirty/{league}/{date}_{league}.csv

    Example:
        2026_04_10_epl.csv -> 2026_04_10
    """
    league = path.parent.name
    suffix = f"_{league}"

    stem = path.stem

    if stem.endswith(suffix):
        return stem[: -len(suffix)]

    # fallback: first 10 chars for YYYY_MM_DD
    return stem[:10]


def select_file_for_date(date_str: str) -> Path:
    return SELECT_DIR / f"{date_str}_soccer_bets.csv"


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["league", "match_date", "home_team", "away_team"]:
        df[col] = df[col].astype(str).str.strip()

    df["league_key"] = df["league"].astype(str).str.lower().str.strip()
    df["match_date_key"] = df["match_date"].astype(str).str.strip()
    df["home_team_key"] = df["home_team"].astype(str).str.lower().str.strip()
    df["away_team_key"] = df["away_team"].astype(str).str.lower().str.strip()

    return df


def build_game_id_map(select_file: Path) -> pd.DataFrame:
    select_df = safe_read_csv(select_file)

    if select_df.empty:
        return pd.DataFrame()

    if not validate_headers(select_df, SELECT_BETS_REQUIRED, select_file):
        return pd.DataFrame()

    select_df = normalize_key_columns(select_df)

    game_id_map = select_df[
        [
            "league_key",
            "match_date_key",
            "home_team_key",
            "away_team_key",
            "game_id",
        ]
    ].copy()

    before = len(game_id_map)

    game_id_map = game_id_map.drop_duplicates()

    conflict_check = (
        game_id_map
        .groupby(
            [
                "league_key",
                "match_date_key",
                "home_team_key",
                "away_team_key",
            ],
            dropna=False,
        )["game_id"]
        .nunique()
        .reset_index(name="game_id_count")
    )

    conflicts = conflict_check[conflict_check["game_id_count"] > 1]

    if not conflicts.empty:
        log_error(f"GAME_ID CONFLICTS | {select_file}")
        log_error(conflicts.to_string(index=False))
        return pd.DataFrame()

    game_id_map = game_id_map.drop_duplicates(
        subset=[
            "league_key",
            "match_date_key",
            "home_team_key",
            "away_team_key",
        ]
    )

    log_summary(
        f"GAME_ID MAP | {select_file.name} | raw_rows={before} unique_games={len(game_id_map)}"
    )

    return game_id_map


def add_game_id_to_score_file(score_file: Path) -> pd.DataFrame:
    league = score_file.parent.name
    date_str = extract_date_from_dirty_score_file(score_file)
    select_file = select_file_for_date(date_str)

    log_summary(f"PROCESSING | score_file={score_file} | select_file={select_file}")

    scores_df = safe_read_csv(score_file)

    if scores_df.empty:
        return pd.DataFrame()

    if not validate_headers(scores_df, FINAL_SCORES_REQUIRED, score_file):
        return pd.DataFrame()

    game_id_map = build_game_id_map(select_file)

    if game_id_map.empty:
        log_error(f"NO GAME_ID MAP AVAILABLE | score_file={score_file}")
        return pd.DataFrame()

    scores_df = normalize_key_columns(scores_df)

    merged = scores_df.merge(
        game_id_map,
        on=[
            "league_key",
            "match_date_key",
            "home_team_key",
            "away_team_key",
        ],
        how="left",
    )

    missing = merged["game_id"].isna().sum()
    attached = len(merged) - missing

    log_summary(
        f"MATCH RESULT | {score_file.name} | rows={len(merged)} attached={attached} missing={missing}"
    )

    if missing:
        missing_rows = merged[merged["game_id"].isna()][
            [
                "league",
                "match_date",
                "home_team",
                "away_team",
            ]
        ].drop_duplicates()

        log_error(f"MISSING GAME_ID | {score_file}")
        log_error(missing_rows.to_string(index=False))

    merged["game_date"] = merged["match_date"]

    output_df = merged[
        [
            "sport",
            "league",
            "game_id",
            "game_date",
            "match_time",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    output_dir = OUTPUT_ROOT / league
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / score_file.name
    output_df.to_csv(output_file, index=False)

    log_summary(f"WROTE | {output_file}")

    return output_df


# =========================
# PROCESS
# =========================

def process() -> None:
    score_files = sorted(DIRTY_SCORES_ROOT.glob("*/*.csv"))

    log_summary(f"Dirty final score files found: {len(score_files)}")

    if not score_files:
        log_error(f"NO DIRTY FINAL SCORE FILES FOUND | {DIRTY_SCORES_ROOT}")
        return

    written = 0

    for score_file in score_files:
        result = add_game_id_to_score_file(score_file)

        if not result.empty:
            written += 1

    log_summary(f"FILES WRITTEN: {written}")


# =========================
# MAIN
# =========================

def main() -> None:
    reset_logs()
    log_summary(f"=== START 01_soccer_add_game_id.py {datetime.now().isoformat()} ===")
    process()
    log_summary(f"=== END 01_soccer_add_game_id.py {datetime.now().isoformat()} ===")
    print("Soccer game_id attachment complete.")


if __name__ == "__main__":
    main()
