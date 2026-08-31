#!/usr/bin/env python3
# docs/win/soccer/scripts/05_final_scores/01_soccer_results_grade.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback

import pandas as pd


# =========================
# PATHS
# =========================

SELECT_DIR = Path("docs/win/soccer/04_select")
LOCKED_SELECT_DIR = SELECT_DIR / "locked"
FINAL_SCORES_DIR = Path("docs/win/soccer/05_final_scores/results/final_scores")

OUTPUT_DIR = Path("docs/win/soccer/05_final_scores/results/graded")
LOCKED_OUTPUT_DIR = Path("docs/win/soccer/05_final_scores/results/graded_locked")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCKED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_DIR = Path("docs/win/soccer/05_final_scores/errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = ERROR_DIR / "soccer_results_grade_errors.txt"
SUMMARY_LOG = ERROR_DIR / "soccer_results_grade_summary.txt"

MASTER_FILE = OUTPUT_DIR / "SOCCER_final.csv"
LOCKED_MASTER_FILE = LOCKED_OUTPUT_DIR / "SOCCER_locked_final.csv"

LEAGUE_MAP = {
    "epl": "EPL",
    "bundesliga": "BUNDESLIGA",
    "laliga": "LALIGA",
    "ligue1": "LIGUE1",
    "seriea": "SERIEA",
    "mls": "MLS",
}

SELECT_REQUIRED = [
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
    "edge",
    "kelly",
]

SCORES_REQUIRED = [
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


def clear_output_files() -> None:
    total_deleted = 0
    for output_dir, label in (
        (OUTPUT_DIR, "SELECTED"),
        (LOCKED_OUTPUT_DIR, "LOCKED"),
    ):
        deleted = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(output_dir.glob("*.csv")):
            path.unlink()
            deleted += 1
            total_deleted += 1
            log_summary(f"DELETED OLD {label} OUTPUT | {path}")
        log_summary(f"OLD {label} OUTPUT FILES DELETED | count={deleted}")
    log_summary(f"OLD OUTPUT FILES DELETED TOTAL | count={total_deleted}")


# =========================
# HELPERS
# =========================

def normalize_game_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "<na>"}:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s.strip()


def decimal_to_american(dec) -> float | None:
    if dec is None or pd.isna(dec):
        return None
    try:
        d = float(dec)
    except Exception:
        return None
    if d <= 1.0:
        return None
    if d >= 2.0:
        return round((d - 1.0) * 100.0)
    return round(-100.0 / (d - 1.0))


def safe_read(path: Path, empty_is_error: bool = True) -> pd.DataFrame:
    try:
        if not path.exists():
            log_error(f"MISSING FILE | {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)
        if df.empty:
            if empty_is_error:
                log_error(f"EMPTY FILE | {path}")
            else:
                log_summary(f"EMPTY LOCKED PICK FILE | {path} | zero picks recorded")
            return pd.DataFrame(columns=df.columns)
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


def derive_take_bet(market: str, side: str) -> str:
    market = str(market).lower().strip()
    side = str(side).lower().strip()
    if market == "match_odds":
        return side
    if market == "total25":
        return f"{side}25"
    if market == "total35":
        return f"{side}35"
    if market == "btts":
        return f"btts_{side}"
    return market


def find_score_file(game_date: str, league_raw: str) -> Path | None:
    league_key = str(league_raw).lower().strip()
    league_upper = LEAGUE_MAP.get(league_key, league_key.upper())
    path = FINAL_SCORES_DIR / league_upper / f"{game_date}_{league_upper}.csv"
    return path if path.exists() else None


def load_scores_for_league_date(game_date: str, league_raw: str) -> pd.DataFrame:
    path = find_score_file(game_date, league_raw)
    if path is None:
        log_error(f"NO SCORE FILE | league={league_raw} date={game_date}")
        return pd.DataFrame()

    df = safe_read(path)
    if df.empty:
        return pd.DataFrame()
    if not validate_headers(df, SCORES_REQUIRED, path):
        return pd.DataFrame()

    df = df[SCORES_REQUIRED].copy()
    df["game_id"] = df["game_id"].apply(normalize_game_id)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    before = len(df)
    df = df[df["game_id"] != ""].copy()
    df = df.drop_duplicates(subset=["game_id"])
    after = len(df)
    if before != after:
        log_error(
            f"SCORE GAME_ID ROWS REMOVED | {path} | before={before} after={after}"
        )
    return df


# =========================
# GRADING
# =========================

def grade_row(row) -> str:
    try:
        take_bet = str(row.get("take_bet", "")).lower().strip()
        home = pd.to_numeric(row.get("home_score"), errors="coerce")
        away = pd.to_numeric(row.get("away_score"), errors="coerce")

        if pd.isna(home) or pd.isna(away):
            return "Missing Score"

        goals = home + away
        if take_bet == "home":
            return "Win" if home > away else "Loss"
        if take_bet == "away":
            return "Win" if away > home else "Loss"
        if take_bet == "draw":
            return "Win" if home == away else "Loss"
        if take_bet == "over25":
            return "Win" if goals > 2.5 else "Loss"
        if take_bet == "under25":
            return "Win" if goals < 2.5 else "Loss"
        if take_bet == "over35":
            return "Win" if goals > 3.5 else "Loss"
        if take_bet == "under35":
            return "Win" if goals < 3.5 else "Loss"
        if take_bet == "btts_yes":
            return "Win" if home > 0 and away > 0 else "Loss"
        if take_bet == "btts_no":
            return "Win" if home == 0 or away == 0 else "Loss"

        log_error(
            f"UNKNOWN TAKE_BET | game_id={row.get('game_id', '')} "
            f"market={row.get('market', '')} side={row.get('side', '')} "
            f"take_bet={take_bet}"
        )
        return "Unknown Market"

    except Exception as e:
        log_error(
            f"GRADE ERROR | game_id={row.get('game_id', '')} "
            f"take_bet={row.get('take_bet', '')} | {e}"
        )
        log_error(traceback.format_exc())
        return "Grade Error"


# =========================
# PROCESS
# =========================

def process_source(
    select_dir: Path,
    output_dir: Path,
    master_file: Path,
    label: str,
    empty_is_error: bool,
) -> None:
    select_files = sorted(select_dir.glob("*_soccer_bets.csv"))
    log_summary(f"{label} select files found: {len(select_files)} | {select_dir}")

    if not select_files:
        log_summary(f"NO {label} SELECT FILES FOUND | {select_dir}")
        return

    all_rows = []

    for file in select_files:
        game_date = file.stem.replace("_soccer_bets", "")
        log_summary(f"PROCESSING {label} SELECT | {file.name} | date={game_date}")

        bets_df = safe_read(file, empty_is_error=empty_is_error)
        if bets_df.empty:
            continue
        if not validate_headers(bets_df, SELECT_REQUIRED, file):
            continue

        for col in ["market", "side", "league", "home_team", "away_team", "match_date"]:
            bets_df[col] = bets_df[col].astype(str).str.strip()

        bets_df["game_id"] = bets_df["game_id"].apply(normalize_game_id)
        bets_df["game_date"] = game_date
        bets_df["league_lower"] = bets_df["league"].astype(str).str.lower().str.strip()
        bets_df["market_type"] = bets_df["market"].astype(str).str.lower().str.strip()
        bets_df["take_bet"] = bets_df.apply(
            lambda r: derive_take_bet(r["market"], r["side"]), axis=1
        )

        bets_df["odds_decimal"] = pd.to_numeric(bets_df["odds"], errors="coerce")

        if "american_odds" in bets_df.columns:
            bets_df["odds_american"] = pd.to_numeric(
                bets_df["american_odds"], errors="coerce"
            )
            missing_american = bets_df["odds_american"].isna()
            bets_df.loc[missing_american, "odds_american"] = bets_df.loc[
                missing_american, "odds_decimal"
            ].apply(decimal_to_american)
        else:
            bets_df["odds_american"] = bets_df["odds_decimal"].apply(
                decimal_to_american
            )

        bets_df["edge_pct"] = pd.to_numeric(bets_df["edge"], errors="coerce")
        merged_frames = []

        for league_raw, league_bets in bets_df.groupby("league_lower"):
            league_bets = league_bets.copy()
            league_bets["game_id"] = league_bets["game_id"].apply(normalize_game_id)
            scores = load_scores_for_league_date(game_date, league_raw)

            if scores.empty:
                log_error(
                    f"NO SCORES LOADED | source={label} league={league_raw} date={game_date}"
                )
                league_bets["home_score"] = pd.NA
                league_bets["away_score"] = pd.NA
                league_bets["score_league"] = pd.NA
                league_bets["score_game_date"] = pd.NA
                league_bets["score_match_time"] = pd.NA
                league_bets["score_home_team"] = pd.NA
                league_bets["score_away_team"] = pd.NA
                merged_frames.append(league_bets)
                continue

            scores_for_merge = scores.rename(
                columns={
                    "league": "score_league",
                    "game_date": "score_game_date",
                    "match_time": "score_match_time",
                    "home_team": "score_home_team",
                    "away_team": "score_away_team",
                }
            )

            merged = league_bets.merge(
                scores_for_merge[
                    [
                        "game_id",
                        "score_league",
                        "score_game_date",
                        "score_match_time",
                        "score_home_team",
                        "score_away_team",
                        "home_score",
                        "away_score",
                    ]
                ],
                on="game_id",
                how="left",
            )

            missing = merged["home_score"].isna() | merged["away_score"].isna()
            missing_count = int(missing.sum())
            if missing_count:
                missing_games = merged.loc[
                    missing,
                    [
                        "game_id",
                        "league",
                        "match_date",
                        "home_team",
                        "away_team",
                        "market",
                        "side",
                    ],
                ].drop_duplicates()
                log_error(
                    f"MISSING SCORE MERGE | source={label} select={file.name} "
                    f"league={league_raw} rows={len(merged)} "
                    f"missing_rows={missing_count} unique_missing={len(missing_games)}"
                )
                log_error(missing_games.to_string(index=False))

            merged_frames.append(merged)

        if not merged_frames:
            continue

        day_df = pd.concat(merged_frames, ignore_index=True)
        day_df["bet_result"] = day_df.apply(grade_row, axis=1)

        wins = int((day_df["bet_result"] == "Win").sum())
        losses = int((day_df["bet_result"] == "Loss").sum())
        pushes = int((day_df["bet_result"] == "Push").sum())
        missing_scores = int((day_df["bet_result"] == "Missing Score").sum())
        unknown_markets = int((day_df["bet_result"] == "Unknown Market").sum())
        grade_errors = int((day_df["bet_result"] == "Grade Error").sum())

        log_summary(
            f"{label} DAY RESULT | {file.name} | rows={len(day_df)} "
            f"W={wins} L={losses} P={pushes} MISSING_SCORE={missing_scores} "
            f"UNKNOWN_MARKET={unknown_markets} GRADE_ERROR={grade_errors}"
        )

        out_file = output_dir / f"{game_date}_results_SOCCER.csv"
        day_df.to_csv(out_file, index=False)
        log_summary(f"WROTE {label} | {out_file}")
        all_rows.append(day_df)

    if all_rows:
        final = pd.concat(all_rows, ignore_index=True)
        final.to_csv(master_file, index=False)

        wins = int((final["bet_result"] == "Win").sum())
        losses = int((final["bet_result"] == "Loss").sum())
        pushes = int((final["bet_result"] == "Push").sum())
        missing_scores = int((final["bet_result"] == "Missing Score").sum())
        unknown_markets = int((final["bet_result"] == "Unknown Market").sum())
        grade_errors = int((final["bet_result"] == "Grade Error").sum())

        log_summary(
            f"{label} MASTER WRITTEN | rows={len(final)} W={wins} L={losses} "
            f"P={pushes} MISSING_SCORE={missing_scores} UNKNOWN_MARKET={unknown_markets} "
            f"GRADE_ERROR={grade_errors} | {master_file}"
        )
    else:
        log_summary(f"NO {label} ROWS TO WRITE | master not written")


def process() -> None:
    process_source(
        SELECT_DIR,
        OUTPUT_DIR,
        MASTER_FILE,
        label="SELECTED",
        empty_is_error=True,
    )
    process_source(
        LOCKED_SELECT_DIR,
        LOCKED_OUTPUT_DIR,
        LOCKED_MASTER_FILE,
        label="LOCKED",
        empty_is_error=False,
    )


# =========================
# MAIN
# =========================

def main() -> None:
    reset_logs()
    log_summary(f"=== START 01_soccer_results_grade.py {datetime.now().isoformat()} ===")
    clear_output_files()
    process()
    log_summary(f"=== END 01_soccer_results_grade.py {datetime.now().isoformat()} ===")
    print("Soccer grading complete.")


if __name__ == "__main__":
    main()
