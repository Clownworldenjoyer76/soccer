#!/usr/bin/env python3
# docs/win/soccer/scripts/05_final_scores/03_soccer_results_reports.py
#
# Rebuildable historical reports are restricted to the fixed tuning period
# declared in markets.yaml. These are tuning reports only.
#
# Immutable forward evaluation picks are reported separately under:
#   docs/win/soccer/05_final_scores/locked_evaluation/
#
# Evaluation reports include sample counts and are grouped by the immutable
# markets.yaml SHA-256 stored with each locked pick.

from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import traceback

import pandas as pd
import yaml


# =========================
# PATHS
# =========================

INTERMEDIATE = Path(
    "docs/win/soccer/05_final_scores/intermediate/work_soccer.csv"
)

FINAL_DIR = Path(
    "docs/win/soccer/05_final_scores"
)

CONFIG_PATH = Path(
    "docs/win/soccer/config/markets.yaml"
)

REPORTS_DIR = (
    FINAL_DIR / "reports"
)

LOCKED_MASTER = (
    FINAL_DIR
    / "results"
    / "graded_locked"
    / "SOCCER_locked_final.csv"
)

LOCKED_EVAL_DIR = (
    FINAL_DIR
    / "locked_evaluation"
)

ERROR_DIR = (
    FINAL_DIR / "errors"
)

ALL_TALLY = (
    FINAL_DIR
    / "all_soccer_market_tally.csv"
)

ERROR_LOG = (
    ERROR_DIR
    / "soccer_results_reports_errors.txt"
)

SUMMARY_LOG = (
    ERROR_DIR
    / "soccer_results_reports_summary.txt"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOCKED_EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ERROR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================
# CONFIG
# =========================

MARKET_LAYOUT = {
    "match_odds": (
        "match_odds",
        "home_draw_away",
    ),
    "btts": (
        "btts",
        "yes_no",
    ),
    "total25": (
        "total_25",
        "over_under",
    ),
    "total35": (
        "total_35",
        "over_under",
    ),
}

BUCKETS = [
    (
        "ev_bucket",
        "ev_sort",
        "ev",
        None,
    ),
    (
        "kelly_bucket",
        "kelly_sort",
        "kelly",
        None,
    ),
    (
        "month_bucket",
        "month_sort",
        "month",
        None,
    ),
    (
        "odds_bucket",
        "odds_sort",
        "odds",
        None,
    ),
    (
        "win_prob_bucket",
        "win_prob_sort",
        "win_prob",
        {"match_odds"},
    ),
]

VALID_RESULTS = {
    "Win",
    "Loss",
    "Push",
}


def normalize_policy_date(
    value,
    label: str,
) -> str:
    normalized = (
        str(value or "")
        .strip()
        .replace("-", "_")
    )

    try:
        datetime.strptime(
            normalized,
            "%Y_%m_%d",
        )

    except ValueError as e:
        raise ValueError(
            f"backtest_policy.soccer.{label} "
            "must be YYYY-MM-DD or YYYY_MM_DD"
        ) from e

    return normalized


def load_backtest_policy() -> dict[str, str]:
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f)

    try:
        raw = (
            data["backtest_policy"]["soccer"]
        )

    except (TypeError, KeyError) as e:
        raise ValueError(
            "markets.yaml missing "
            "backtest_policy.soccer"
        ) from e

    if not isinstance(raw, dict):
        raise ValueError(
            "backtest_policy.soccer "
            "must be a mapping"
        )

    policy = {
        "tuning_start": (
            normalize_policy_date(
                raw.get("tuning_start"),
                "tuning_start",
            )
        ),
        "tuning_end": (
            normalize_policy_date(
                raw.get("tuning_end"),
                "tuning_end",
            )
        ),
        "evaluation_start": (
            normalize_policy_date(
                raw.get(
                    "evaluation_start"
                ),
                "evaluation_start",
            )
        ),
    }

    tuning_start = datetime.strptime(
        policy["tuning_start"],
        "%Y_%m_%d",
    )

    tuning_end = datetime.strptime(
        policy["tuning_end"],
        "%Y_%m_%d",
    )

    evaluation_start = datetime.strptime(
        policy["evaluation_start"],
        "%Y_%m_%d",
    )

    if tuning_start > tuning_end:
        raise ValueError(
            "backtest tuning_start "
            "must be <= tuning_end"
        )

    if evaluation_start <= tuning_end:
        raise ValueError(
            "backtest evaluation_start "
            "must be later than tuning_end"
        )

    return policy


BACKTEST_POLICY = (
    load_backtest_policy()
)


def current_config_sha256() -> str:
    return hashlib.sha256(
        CONFIG_PATH.read_bytes()
    ).hexdigest()


LEAGUE_TALLY_FILES = [
    FINAL_DIR
    / "epl_market_tally.csv",
    FINAL_DIR
    / "bundesliga_market_tally.csv",
    FINAL_DIR
    / "laliga_market_tally.csv",
    FINAL_DIR
    / "ligue1_market_tally.csv",
    FINAL_DIR
    / "seriea_market_tally.csv",
    FINAL_DIR
    / "mls_market_tally.csv",
]


# =========================
# LOGGING
# =========================

def reset_logs() -> None:
    SUMMARY_LOG.write_text(
        "",
        encoding="utf-8",
    )

    ERROR_LOG.write_text(
        "",
        encoding="utf-8",
    )


def log_error(msg: str) -> None:
    with open(
        ERROR_LOG,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"[{datetime.now().isoformat()}] "
            f"{msg}\n"
        )


def log_summary(msg: str) -> None:
    with open(
        SUMMARY_LOG,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"[{datetime.now().isoformat()}] "
            f"{msg}\n"
        )


def clear_output_files() -> None:
    deleted_files = 0
    deleted_dirs = 0

    if ALL_TALLY.exists():
        ALL_TALLY.unlink()
        deleted_files += 1

        log_summary(
            f"DELETED OLD OUTPUT | "
            f"{ALL_TALLY}"
        )

    for path in LEAGUE_TALLY_FILES:
        if path.exists():
            path.unlink()
            deleted_files += 1

            log_summary(
                f"DELETED OLD OUTPUT | "
                f"{path}"
            )

    for path in sorted(
        FINAL_DIR.glob(
            "*_market_tally.csv"
        )
    ):
        if path.exists():
            path.unlink()
            deleted_files += 1

            log_summary(
                f"DELETED OLD OUTPUT | "
                f"{path}"
            )

    if REPORTS_DIR.exists():
        shutil.rmtree(
            REPORTS_DIR
        )

        deleted_dirs += 1

        log_summary(
            "DELETED OLD REPORTS DIR | "
            f"{REPORTS_DIR}"
        )

    if LOCKED_EVAL_DIR.exists():
        shutil.rmtree(
            LOCKED_EVAL_DIR
        )

        deleted_dirs += 1

        log_summary(
            "DELETED OLD LOCKED "
            "EVALUATION DIR | "
            f"{LOCKED_EVAL_DIR}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOCKED_EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_summary(
        "OLD REPORT OUTPUTS DELETED | "
        f"files={deleted_files} "
        f"dirs={deleted_dirs}"
    )


# =========================
# IO HELPERS
# =========================

def safe_read_intermediate(
    path: Path,
) -> pd.DataFrame:
    try:
        if not path.exists():
            log_error(
                "INTERMEDIATE FILE MISSING | "
                f"{path} — run 02 first"
            )

            return pd.DataFrame()

        df = pd.read_csv(
            path,
            dtype={
                "month_bucket": str,
                "ev_bucket": str,
                "kelly_bucket": str,
                "odds_bucket": str,
                "win_prob_bucket": str,
            },
        )

        if df.empty:
            log_error(
                "INTERMEDIATE FILE EMPTY | "
                f"{path}"
            )

            return pd.DataFrame()

        return df

    except Exception as e:
        log_error(
            f"READ ERROR | {path} | {e}"
        )

        log_error(
            traceback.format_exc()
        )

        return pd.DataFrame()


def safe_read_locked(
    path: Path,
) -> pd.DataFrame:
    try:
        if not path.exists():
            log_summary(
                "LOCKED MASTER NOT YET "
                f"AVAILABLE | {path}"
            )

            return pd.DataFrame()

        df = pd.read_csv(path)

        if df.empty:
            log_summary(
                f"LOCKED MASTER EMPTY | "
                f"{path}"
            )

            return pd.DataFrame()

        return df

    except Exception as e:
        log_error(
            "LOCKED MASTER READ ERROR | "
            f"{path} | {e}"
        )

        log_error(
            traceback.format_exc()
        )

        return pd.DataFrame()


# =========================
# AGG HELPERS
# =========================

def summarize(
    sub: pd.DataFrame,
) -> dict:
    res = (
        sub["bet_result"]
        .astype(str)
    )

    w = int(
        (res == "Win").sum()
    )

    l = int(
        (res == "Loss").sum()
    )

    p = int(
        (res == "Push").sum()
    )

    total = w + l + p

    pct = (
        round(
            w / (w + l),
            4,
        )
        if (w + l) > 0
        else 0.0
    )

    return {
        "Win": w,
        "Loss": l,
        "Push": p,
        "Total": total,
        "Sample_Count": total,
        "Win_Pct": pct,
    }


def summarize_locked(
    sub: pd.DataFrame,
) -> dict:
    s = summarize(sub)

    return {
        "Win": s["Win"],
        "Loss": s["Loss"],
        "Push": s["Push"],
        "Sample_Count": (
            s["Sample_Count"]
        ),
        "Win_Pct": s["Win_Pct"],
    }


def write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    log_summary(
        f"WROTE {path} "
        f"({len(df)} rows)"
    )


def filter_graded(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "bet_result" not in df.columns:
        log_error(
            "MISSING COLUMN | bet_result"
        )

        return pd.DataFrame()

    return df[
        df["bet_result"]
        .astype(str)
        .isin(VALID_RESULTS)
    ].copy()


def filter_tuning_period(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "match_date" not in df.columns:
        log_error(
            "TUNING FILTER MISSING "
            "COLUMN | match_date"
        )

        return pd.DataFrame()

    dates = (
        df["match_date"]
        .astype(str)
        .str.strip()
        .str.replace(
            "-",
            "_",
            regex=False,
        )
    )

    start = (
        BACKTEST_POLICY["tuning_start"]
    )

    end = (
        BACKTEST_POLICY["tuning_end"]
    )

    mask = (
        (dates >= start)
        & (dates <= end)
    )

    excluded = int(
        (~mask).sum()
    )

    if excluded:
        log_summary(
            "TUNING FILTER EXCLUDED | "
            f"rows={excluded} | "
            f"allowed={start}..{end}"
        )

    out = df.loc[
        mask
    ].copy()

    out["match_date"] = (
        dates.loc[mask]
    )

    return out


# =========================
# EXISTING TALLY FILES
# TUNING PERIOD ONLY
# =========================

def build_all_tally(
    df: pd.DataFrame,
) -> None:
    rows = []

    for (
        market,
        side,
    ), sub in df.groupby(
        [
            "market_type",
            "side",
        ],
        dropna=False,
    ):
        s = summarize(sub)

        rows.append(
            {
                "market": market,
                "market_type": side,
                **s,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "market",
            "market_type",
            "Win",
            "Loss",
            "Push",
            "Total",
            "Sample_Count",
            "Win_Pct",
        ],
    )

    if not out.empty:
        out = (
            out.sort_values(
                [
                    "market",
                    "market_type",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    write_csv(
        out,
        ALL_TALLY,
    )


def build_league_tally(
    df: pd.DataFrame,
    league: str,
) -> None:
    rows = []

    for (
        market,
        side,
    ), sub in df.groupby(
        [
            "market_type",
            "side",
        ],
        dropna=False,
    ):
        s = summarize(sub)

        rows.append(
            {
                "market": market,
                "market_type": side,
                **s,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "market",
            "market_type",
            "Win",
            "Loss",
            "Push",
            "Total",
            "Sample_Count",
            "Win_Pct",
        ],
    )

    if not out.empty:
        out = (
            out.sort_values(
                [
                    "market",
                    "market_type",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    write_csv(
        out,
        FINAL_DIR
        / f"{league}_market_tally.csv",
    )


# =========================
# EXISTING BUCKET REPORTS
# TUNING PERIOD ONLY
# =========================

def by_bucket(
    df: pd.DataFrame,
    bucket_col: str,
    sort_col: str,
) -> pd.DataFrame:
    rows = []

    for bucket, sub in df.groupby(
        bucket_col,
        dropna=False,
    ):
        sort_val = (
            sub[sort_col]
            .dropna()
            .iloc[0]
            if sub[sort_col]
            .notna()
            .any()
            else None
        )

        s = summarize(sub)

        rows.append(
            {
                "bucket": bucket,
                "_sort": sort_val,
                **s,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "bucket",
                "Win",
                "Loss",
                "Push",
                "Total",
                "Sample_Count",
                "Win_Pct",
            ]
        )

    out = pd.DataFrame(rows)

    out["_sort"] = (
        pd.to_numeric(
            out["_sort"],
            errors="coerce",
        )
    )

    out = (
        out.sort_values(
            [
                "_sort",
                "bucket",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    out = out.drop(
        columns=["_sort"]
    )

    return out[
        [
            "bucket",
            "Win",
            "Loss",
            "Push",
            "Total",
            "Sample_Count",
            "Win_Pct",
        ]
    ]


def by_bucket_by_side(
    df: pd.DataFrame,
    bucket_col: str,
    sort_col: str,
) -> pd.DataFrame:
    rows = []

    for (
        bucket,
        side,
    ), sub in df.groupby(
        [
            bucket_col,
            "side",
        ],
        dropna=False,
    ):
        sort_val = (
            sub[sort_col]
            .dropna()
            .iloc[0]
            if sub[sort_col]
            .notna()
            .any()
            else None
        )

        s = summarize(sub)

        rows.append(
            {
                "bucket": bucket,
                "side": side,
                "_sort": sort_val,
                **s,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "bucket",
                "side",
                "Win",
                "Loss",
                "Push",
                "Total",
                "Sample_Count",
                "Win_Pct",
            ]
        )

    out = pd.DataFrame(rows)

    out["_sort"] = (
        pd.to_numeric(
            out["_sort"],
            errors="coerce",
        )
    )

    out = (
        out.sort_values(
            [
                "_sort",
                "bucket",
                "side",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    out = out.drop(
        columns=["_sort"]
    )

    return out[
        [
            "bucket",
            "side",
            "Win",
            "Loss",
            "Push",
            "Total",
            "Sample_Count",
            "Win_Pct",
        ]
    ]


def build_market_reports(
    df: pd.DataFrame,
    league: str,
    market_type: str,
) -> None:
    if market_type not in MARKET_LAYOUT:
        log_error(
            "UNKNOWN market_type | "
            f"{market_type}"
        )

        return

    (
        folder_name,
        sides_label,
    ) = MARKET_LAYOUT[
        market_type
    ]

    out_dir = (
        REPORTS_DIR
        / league
        / folder_name
    )

    sub = df[
        df["market_type"]
        .astype(str)
        == market_type
    ]

    if sub.empty:
        log_summary(
            f"NO ROWS | "
            f"league={league} "
            f"market={market_type} — "
            "skipping reports"
        )

        return

    for (
        bucket_col,
        sort_col,
        by_label,
        allowed,
    ) in BUCKETS:
        if (
            allowed
            and market_type
            not in allowed
        ):
            continue

        if bucket_col not in sub.columns:
            log_error(
                "MISSING COLUMN | "
                f"{bucket_col} "
                f"(league={league} "
                f"market={market_type})"
            )

            continue

        if sort_col not in sub.columns:
            log_error(
                "MISSING COLUMN | "
                f"{sort_col} "
                f"(league={league} "
                f"market={market_type})"
            )

            continue

        combined = by_bucket(
            sub,
            bucket_col,
            sort_col,
        )

        write_csv(
            combined,
            out_dir
            / (
                f"{league}_"
                f"{folder_name}_"
                f"by_{by_label}.csv"
            ),
        )

        bysd = by_bucket_by_side(
            sub,
            bucket_col,
            sort_col,
        )

        write_csv(
            bysd,
            out_dir
            / (
                f"{league}_"
                f"{folder_name}_"
                f"by_{by_label}_"
                f"{sides_label}_summary.csv"
            ),
        )


# =========================
# EXPLICIT TUNING PERFORMANCE
# =========================

def build_tuning_rule_performance(
    df: pd.DataFrame,
) -> None:
    rows = []

    config_sha = (
        current_config_sha256()
    )

    for (
        league,
        market,
        side,
    ), sub in df.groupby(
        [
            "league_lower",
            "market_type",
            "side",
        ],
        dropna=False,
    ):
        rows.append(
            {
                "selection_config_sha256": (
                    config_sha
                ),
                "league": league,
                "market": market,
                "side": side,
                "Tuning_Start": (
                    BACKTEST_POLICY[
                        "tuning_start"
                    ]
                ),
                "Tuning_End": (
                    BACKTEST_POLICY[
                        "tuning_end"
                    ]
                ),
                "First_Match_Date": (
                    sub["match_date"]
                    .astype(str)
                    .min()
                ),
                "Last_Match_Date": (
                    sub["match_date"]
                    .astype(str)
                    .max()
                ),
                "Evidence_Type": (
                    "TUNING_ONLY_"
                    "NOT_PROMOTION_EVIDENCE"
                ),
                **summarize_locked(
                    sub
                ),
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "selection_config_sha256",
            "league",
            "market",
            "side",
            "Tuning_Start",
            "Tuning_End",
            "First_Match_Date",
            "Last_Match_Date",
            "Evidence_Type",
            "Win",
            "Loss",
            "Push",
            "Sample_Count",
            "Win_Pct",
        ],
    )

    if not out.empty:
        out = (
            out.sort_values(
                [
                    "league",
                    "market",
                    "side",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    write_csv(
        out,
        LOCKED_EVAL_DIR
        / "tuning_rule_performance.csv",
    )


# =========================
# LOCKED FORWARD EVALUATION
# =========================

def build_locked_evaluation() -> None:
    raw = safe_read_locked(
        LOCKED_MASTER
    )

    if raw.empty:
        log_summary(
            "NO LOCKED EVALUATION WRITTEN | "
            "no graded locked rows available"
        )

        return

    required = [
        "league_lower",
        "market_type",
        "side",
        "bet_result",
        "match_date",
        "selection_config_sha256",
        "selection_period",
        "tuning_start",
        "tuning_end",
        "evaluation_start",
    ]

    missing = [
        c
        for c in required
        if c not in raw.columns
    ]

    if missing:
        log_error(
            "LOCKED EVALUATION MISSING "
            f"REQUIRED COLUMNS | {missing}"
        )

        return

    df = filter_graded(raw)

    if df.empty:
        log_summary(
            "NO LOCKED GRADED RESULTS YET | "
            "W/L/P sample count is zero"
        )

        return

    for col in [
        "league_lower",
        "market_type",
        "side",
        "selection_config_sha256",
        "selection_period",
        "tuning_start",
        "tuning_end",
        "evaluation_start",
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    df["league_lower"] = (
        df["league_lower"]
        .str.lower()
    )

    df["market_type"] = (
        df["market_type"]
        .str.lower()
    )

    df["side"] = (
        df["side"]
        .str.lower()
    )

    df["selection_period"] = (
        df["selection_period"]
        .str.lower()
    )

    for col in [
        "match_date",
        "tuning_start",
        "tuning_end",
        "evaluation_start",
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(
                "-",
                "_",
                regex=False,
            )
        )

    # A row counts as out-of-sample only if the
    # immutable metadata stored with that selection
    # proves the match happened after tuning ended
    # and on/after that YAML version's evaluation start.
    valid_oos = (
        (
            df["selection_period"]
            == "evaluation"
        )
        & (
            df["match_date"]
            > df["tuning_end"]
        )
        & (
            df["match_date"]
            >= df["evaluation_start"]
        )
    )

    rejected_oos = int(
        (~valid_oos).sum()
    )

    if rejected_oos:
        log_error(
            "LOCKED ROWS REJECTED AS "
            "NON-OOS | "
            f"rows={rejected_oos}"
        )

    df = df.loc[
        valid_oos
    ].copy()

    if df.empty:
        log_summary(
            "NO OUT-OF-SAMPLE LOCKED RESULTS | "
            "zero locked rows passed "
            "period validation"
        )

        return

    # ---------------------------------
    # All immutable forward picks
    # ---------------------------------

    rows = []

    for (
        market,
        side,
    ), sub in df.groupby(
        [
            "market_type",
            "side",
        ],
        dropna=False,
    ):
        rows.append(
            {
                "market": market,
                "side": side,
                **summarize_locked(
                    sub
                ),
            }
        )

    all_tally = pd.DataFrame(
        rows,
        columns=[
            "market",
            "side",
            "Win",
            "Loss",
            "Push",
            "Sample_Count",
            "Win_Pct",
        ],
    )

    if not all_tally.empty:
        all_tally = (
            all_tally.sort_values(
                [
                    "market",
                    "side",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    write_csv(
        all_tally,
        LOCKED_EVAL_DIR
        / "all_soccer_locked_tally.csv",
    )

    # ---------------------------------
    # Per YAML / league / market / side
    # ---------------------------------

    rows = []

    group_cols = [
        "selection_config_sha256",
        "league_lower",
        "market_type",
        "side",
    ]

    for keys, sub in df.groupby(
        group_cols,
        dropna=False,
    ):
        (
            config_sha,
            league,
            market,
            side,
        ) = keys

        rows.append(
            {
                "selection_config_sha256": (
                    config_sha
                ),
                "league": league,
                "market": market,
                "side": side,
                "Tuning_Start": (
                    sub["tuning_start"]
                    .iloc[0]
                ),
                "Tuning_End": (
                    sub["tuning_end"]
                    .iloc[0]
                ),
                "Evaluation_Start": (
                    sub["evaluation_start"]
                    .iloc[0]
                ),
                "First_Match_Date": (
                    sub["match_date"]
                    .min()
                ),
                "Last_Match_Date": (
                    sub["match_date"]
                    .max()
                ),
                "Evidence_Type": (
                    "OUT_OF_SAMPLE_EVALUATION"
                ),
                **summarize_locked(
                    sub
                ),
            }
        )

    rule = pd.DataFrame(
        rows,
        columns=[
            "selection_config_sha256",
            "league",
            "market",
            "side",
            "Tuning_Start",
            "Tuning_End",
            "Evaluation_Start",
            "First_Match_Date",
            "Last_Match_Date",
            "Evidence_Type",
            "Win",
            "Loss",
            "Push",
            "Sample_Count",
            "Win_Pct",
        ],
    )

    if not rule.empty:
        rule = (
            rule.sort_values(
                [
                    "selection_config_sha256",
                    "league",
                    "market",
                    "side",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    write_csv(
        rule,
        LOCKED_EVAL_DIR
        / "locked_rule_performance.csv",
    )

    # ---------------------------------
    # Overall performance per YAML
    # ---------------------------------

    rows = []

    for config_sha, sub in df.groupby(
        "selection_config_sha256",
        dropna=False,
    ):
        rows.append(
            {
                "selection_config_sha256": (
                    config_sha
                ),
                "Tuning_Start": (
                    sub["tuning_start"]
                    .iloc[0]
                ),
                "Tuning_End": (
                    sub["tuning_end"]
                    .iloc[0]
                ),
                "Evaluation_Start": (
                    sub["evaluation_start"]
                    .iloc[0]
                ),
                "First_Match_Date": (
                    sub["match_date"]
                    .min()
                ),
                "Last_Match_Date": (
                    sub["match_date"]
                    .max()
                ),
                "Evidence_Type": (
                    "OUT_OF_SAMPLE_EVALUATION"
                ),
                **summarize_locked(
                    sub
                ),
            }
        )

    config_perf = pd.DataFrame(
        rows,
        columns=[
            "selection_config_sha256",
            "Tuning_Start",
            "Tuning_End",
            "Evaluation_Start",
            "First_Match_Date",
            "Last_Match_Date",
            "Evidence_Type",
            "Win",
            "Loss",
            "Push",
            "Sample_Count",
            "Win_Pct",
        ],
    )

    if not config_perf.empty:
        config_perf = (
            config_perf.sort_values(
                "First_Match_Date"
            )
            .reset_index(
                drop=True
            )
        )

    write_csv(
        config_perf,
        LOCKED_EVAL_DIR
        / "locked_config_performance.csv",
    )

    log_summary(
        "LOCKED FORWARD EVALUATION | "
        f"graded_oos_rows={len(df)} | "
        f"config_versions="
        f"{df['selection_config_sha256'].nunique()}"
    )


# =========================
# MAIN
# =========================

def main() -> None:
    reset_logs()

    log_summary(
        "=== START "
        "03_soccer_results_reports.py "
        f"{datetime.now().isoformat()} ==="
    )

    clear_output_files()

    log_summary(
        "BACKTEST POLICY | "
        f"tuning="
        f"{BACKTEST_POLICY['tuning_start']}.."
        f"{BACKTEST_POLICY['tuning_end']} | "
        f"evaluation_start="
        f"{BACKTEST_POLICY['evaluation_start']}"
    )

    raw = safe_read_intermediate(
        INTERMEDIATE
    )

    if raw.empty:
        log_error(
            "NO STANDARD REPORTS WRITTEN | "
            "intermediate file missing, "
            "empty, unreadable, or invalid"
        )

    else:
        df = filter_graded(raw)

        if df.empty:
            log_error(
                "NO ROWS WITH VALID "
                "bet_result (Win/Loss/Push)"
            )

            log_summary(
                "NO STANDARD REPORTS WRITTEN | "
                "no graded rows"
            )

        else:
            df = filter_tuning_period(
                df
            )

            if df.empty:
                log_error(
                    "NO TUNING-PERIOD "
                    "GRADED ROWS"
                )

                log_summary(
                    "NO STANDARD REPORTS WRITTEN | "
                    "tuning sample is empty"
                )

            else:
                required_cols = [
                    "league_lower",
                    "market_type",
                    "side",
                    "match_date",
                ]

                missing_required = [
                    c
                    for c in required_cols
                    if c not in df.columns
                ]

                if missing_required:
                    log_error(
                        "MISSING REQUIRED REPORT "
                        f"COLUMNS | {missing_required}"
                    )

                    log_summary(
                        "NO STANDARD REPORTS WRITTEN | "
                        "required columns missing"
                    )

                else:
                    df["league_lower"] = (
                        df["league_lower"]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )

                    df["market_type"] = (
                        df["market_type"]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )

                    df["side"] = (
                        df["side"]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )

                    log_summary(
                        "Rows loaded "
                        "(graded tuning only): "
                        f"{len(df)}"
                    )

                    log_summary(
                        "market_type counts: "
                        f"{df['market_type'].value_counts().to_dict()}"
                    )

                    log_summary(
                        "leagues: "
                        f"{df['league_lower'].value_counts().to_dict()}"
                    )

                    log_summary(
                        "TUNING PERIOD | "
                        f"{BACKTEST_POLICY['tuning_start']}.."
                        f"{BACKTEST_POLICY['tuning_end']} | "
                        "standard reports are "
                        "tuning-only evidence"
                    )

                    build_tuning_rule_performance(
                        df
                    )

                    build_all_tally(
                        df
                    )

                    for (
                        league,
                        league_df,
                    ) in df.groupby(
                        "league_lower"
                    ):
                        build_league_tally(
                            league_df,
                            league,
                        )

                    for (
                        league,
                        market_type,
                    ), grp in df.groupby(
                        [
                            "league_lower",
                            "market_type",
                        ]
                    ):
                        build_market_reports(
                            grp,
                            league,
                            market_type,
                        )

    build_locked_evaluation()

    log_summary(
        "=== END "
        "03_soccer_results_reports.py "
        f"{datetime.now().isoformat()} ==="
    )

    print(
        "Soccer reports generated."
    )


if __name__ == "__main__":
    main()